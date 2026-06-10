"""[P3 조건 E] joint end-to-end — enhancer(tv01 init) + detector(YOLOv8n COCO) 동시 학습.

손실: L = λ_det·L_yolo + λ_rec·L_anchor(tv01 distill) + λ_tv·L_grid.
budget = configs/p3_joint_budget.yaml (C/D 공유: epochs/imgsz/lr/batch/aug 중 핵심).
warm-up: 초기 warmup_epochs 동안 detector 만 학습(enhancer lr=0) → enhanced 분포 적응
후 joint co-adapt(enhancer 작은 lr). 입력 파이프라인은 D 와 동일(enhance 후 검출).

P2 train_detection_aware 의 dataset/collate 재사용 + detector unfreeze + optimizer 2-group.
detector 는 v8DetectionLoss(raw preds). 평가는 enhancer(.pth) + detector(.pt) 로 eval.

AMP: D(ultralytics) 와 동일하게 autocast 전체 + GradScaler (검증된 안정 경로).

사용:
  스모크: python experiments/train_p3_joint.py --smoke
  본학습: python experiments/train_p3_joint.py --exp_name p3_E
"""
from __future__ import annotations

import argparse
import csv as csvmod
import math
import sys
import time
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from experiments.train_detection_aware import (
    ExDarkDetectionDataset, exdark_collate, move_targets,
)
from src.losses.detection_aware import grid_tv_loss
from src.models.bilateral_grid import build_from_config
from src.utils.inference import guard_output, is_nonfinite
from src.utils.paths import load_paths

HR = "=" * 80


def _norm_device(d):
    """'0' 같은 ultralytics 식 device 문자열을 torch.load 호환('cuda:0')으로."""
    if d is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(d).isdigit():
        return f"cuda:{d}"
    return str(d)


def build_joint(budget: dict, P, device: str, enh_mode: str = "joint"):
    """enh_mode: 'joint'(E, enhancer trainable) | 'frozen'(D', tv01 동결) | 'none'(C', raw).

    detector(YOLOv8n) 는 셋 다 trainable·동일 설정. enhancer 만 모드별로 다름.
    """
    tv01 = P.runs / "p2_det_l0020_tv01" / "checkpoints" / "last.pth"
    enhancer = None
    base_frozen = None
    if enh_mode in ("joint", "frozen"):
        est = torch.load(tv01, map_location=device, weights_only=False)
        enhancer = build_from_config(est["config"]).to(device)
        enhancer.load_state_dict(est["model"], strict=False)
        if enh_mode == "frozen":
            enhancer.eval()
            for p in enhancer.parameters():
                p.requires_grad_(False)
        else:  # joint → anchor distill 기준(base_frozen)
            base_frozen = build_from_config(est["config"]).to(device).eval()
            base_frozen.load_state_dict(est["model"], strict=False)
            for p in base_frozen.parameters():
                p.requires_grad_(False)

    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss
    yolo = YOLO(str(P.yolov8n))
    m = yolo.model.to(device)
    for p in m.parameters():
        p.requires_grad_(True)               # ★ detector unfreeze (3 모드 공통)
    m.model[-1].training = True
    if not hasattr(m, "args") or m.args is None or isinstance(m.args, dict):
        m.args = DEFAULT_CFG
    det_loss_fn = v8DetectionLoss(m)
    return enhancer, base_frozen, yolo, m, det_loss_fn, tv01


def forward_losses(enhancer, base_frozen, m, det_loss_fn, low, tg, lam, use_amp, enh_mode="joint"):
    """공통 forward. 3 모드 모두 detector grad 스케일(λ_det·L_yolo) 동일 → 공정 비교."""
    lam_det, lam_rec, lam_tv = lam
    with torch.cuda.amp.autocast(enabled=use_amp):
        grid = None
        if enh_mode == "none":                         # C': raw 입력
            enh_pm1 = low
        elif enh_mode == "frozen":                     # D': tv01 동결 enhance
            with torch.no_grad():
                enh_raw, _ = enhancer(low, return_grid=True)
            enh_pm1 = guard_output(enh_raw)
        else:                                          # E: joint (enhancer 학습)
            enh_raw, grid = enhancer(low, return_grid=True)
            enh_pm1 = guard_output(enh_raw)
        enh01 = (enh_pm1 + 1.0) * 0.5                  # detector 입력 [0,1]
        preds = m(enh01)
        batch = {"cls": tg["cls"], "bboxes": tg["bboxes"],
                 "batch_idx": tg["batch_idx"], "img": enh01}
        loss_vec, _ = det_loss_fn(preds, batch)
        L_yolo = loss_vec.sum() if loss_vec.ndim > 0 else loss_vec
        L = lam_det * L_yolo
        L_anchor = torch.zeros((), device=low.device)
        L_tv = torch.zeros((), device=low.device)
        if enh_mode == "joint":                        # enhancer 학습 항 (E 전용)
            with torch.no_grad():
                base_out = base_frozen(low)
            L_anchor = F.l1_loss(enh_pm1, base_out.detach())
            L_tv = grid_tv_loss(grid)
            L = L + lam_rec * L_anchor + lam_tv * L_tv
    return L, {"yolo": float(L_yolo), "anchor": float(L_anchor), "tv": float(L_tv)}, enh_pm1


def make_loader(budget, P, imgsz, batch, workers, max_n=0):
    dc_csv = _LUNA2_ROOT / budget["data"]["split_csv"]
    ds = ExDarkDetectionDataset(P.exdark, dc_csv, "train", image_size=imgsz, filter_empty=True)
    if max_n > 0:
        ds = Subset(ds, list(range(min(max_n, len(ds)))))
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                      drop_last=True, collate_fn=exdark_collate)


def run_smoke(args) -> int:
    device = _norm_device(args.device)
    P = load_paths()
    budget = yaml.safe_load(open(args.config, encoding="utf-8"))
    lam = (0.02, 1.0, 0.1)   # λ_det, λ_rec, λ_tv (P2 tv01 와 동일 비중)
    torch.manual_seed(42)
    print(HR)
    print(" [P3-E 스모크] joint enhancer+detector — 양쪽 학습/finite/비유한 (fp32, 본학습 X)")
    print(HR)
    enhancer, base_frozen, yolo, m, det_loss_fn, tv01 = build_joint(budget, P, device)
    enhancer.train(); m.train(); m.model[-1].training = True
    loader = make_loader(budget, P, imgsz=384, batch=args.smoke_batch, workers=0, max_n=120)
    print(f"  enhancer=tv01({tv01.name})  detector=YOLOv8n(unfrozen)  λ={lam}")
    print(f"  train subset {len(loader.dataset)}  batch {loader.batch_size}  imgsz 384(스모크)")
    print("-" * 80)

    opt = torch.optim.SGD([
        {"params": m.parameters(), "lr": 1e-3},
        {"params": enhancer.parameters(), "lr": 1e-4},
    ], momentum=0.9)
    w_enh0 = enhancer.coefficient_net.to_grid.weight.detach().clone()
    w_det0 = m.model[-1].cv2[0][0].conv.weight.detach().clone()  # Detect head 첫 conv
    nonfinite = 0
    print(f"  {'step':>4} | {'L_total':>8} {'L_yolo':>8} {'L_anchor':>9} {'L_tv':>7} | finite")
    it = iter(loader)
    for step in range(1, args.smoke_steps + 1):
        try:
            low, tg = next(it)
        except StopIteration:
            it = iter(loader); low, tg = next(it)
        low = low.to(device); tg = move_targets(tg, device)
        opt.zero_grad(set_to_none=True)
        L, items, enh_raw = forward_losses(enhancer, base_frozen, m, det_loss_fn, low, tg, lam, use_amp=False, enh_mode="joint")
        if is_nonfinite(enh_raw):
            nonfinite += 1
        fin = bool(torch.isfinite(L))
        L.backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters()) + list(enhancer.parameters()), 10.0)
        opt.step()
        if step <= 5 or step % 5 == 0:
            print(f"  {step:>4} | {float(L):>8.3f} {items['yolo']:>8.3f} "
                  f"{items['anchor']:>9.4f} {items['tv']:>7.4f} | {fin}")
    dw_enh = float((enhancer.coefficient_net.to_grid.weight.detach() - w_enh0).norm())
    dw_det = float((m.model[-1].cv2[0][0].conv.weight.detach() - w_det0).norm())
    print("-" * 80)
    print(f" ★ enhancer Δw = {dw_enh:.4e}  (>0 학습)")
    print(f" ★ detector Δw = {dw_det:.4e}  (>0 학습)")
    print(f"   비유한 원시출력 = {nonfinite}/{args.smoke_steps}")
    ok = dw_enh > 0 and dw_det > 0 and nonfinite == 0
    print(f"   판정: {'양쪽 학습 + 안정 → joint 루프 OK' if ok else '문제 — 확인 필요'}")
    print(HR)
    return 0


def run_train(args) -> int:
    device = _norm_device(args.device)
    P = load_paths()
    budget = yaml.safe_load(open(args.config, encoding="utf-8"))
    tr = budget["train"]
    imgsz = args.imgsz or tr["imgsz"]
    batch = args.batch or tr["batch"]
    epochs = args.epochs or tr["epochs"]
    warmup = int(tr.get("warmup_epochs", 2))
    lr0 = float(tr["lr0"]); lrf = float(tr["lrf"])
    enh_ratio = args.enh_lr_ratio          # enhancer lr = detector lr * ratio
    lam = (args.lambda_det, args.lambda_rec, args.lambda_tv)
    use_amp = True
    enh_mode = args.enh_mode               # 'joint'(E) | 'frozen'(D') | 'none'(C')
    exp = args.exp_name or "p3_E"
    torch.manual_seed(tr.get("seed", 42))

    run_dir = P.runs / exp; ck = run_dir / "checkpoints"; lg = run_dir / "logs"
    ck.mkdir(parents=True, exist_ok=True); lg.mkdir(parents=True, exist_ok=True)

    enhancer, base_frozen, yolo, m, det_loss_fn, tv01 = build_joint(budget, P, device, enh_mode)
    loader = make_loader(budget, P, imgsz, batch, args.num_workers, max_n=args.max_train_samples)

    # optimizer: joint 만 enhancer group 추가 (C'/D' 는 detector 만 — detector grad 스케일은 3 모드 동일)
    groups = [{"params": m.parameters(), "lr": lr0}]
    if enh_mode == "joint":
        groups.append({"params": enhancer.parameters(), "lr": lr0 * enh_ratio})
    opt = torch.optim.SGD(groups, momentum=tr.get("momentum", 0.937),
                          weight_decay=tr.get("weight_decay", 5e-4))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(HR)
    print(f" [P3 본학습] {exp}  enh_mode={enh_mode}  λ(det,rec,tv)={lam}  "
          f"imgsz={imgsz} batch={batch} epochs={epochs}")
    print(f"  detector lr0={lr0} cosine  " +
          (f"enhancer lr0={lr0*enh_ratio} (warmup {warmup}ep 동결)" if enh_mode == "joint"
           else "enhancer: " + ("tv01 동결" if enh_mode == "frozen" else "없음(raw)")))
    print(HR)
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"budget": budget, "lam": list(lam), "imgsz": imgsz,
                        "batch": batch, "enh_lr_ratio": enh_ratio}, f, allow_unicode=True)
    csvf = open(lg / "train_log.csv", "w", newline="", encoding="utf-8")
    cw = csvmod.writer(csvf); cw.writerow(["epoch", "lr_det", "lr_enh", "L_total", "L_yolo", "L_anchor", "L_tv"])

    clip_params = list(m.parameters()) + (list(enhancer.parameters()) if enh_mode == "joint" else [])
    for epoch in range(epochs):
        cos = lrf + 0.5 * (1 - lrf) * (1 + math.cos(math.pi * epoch / epochs))
        lr_det = lr0 * cos
        opt.param_groups[0]["lr"] = lr_det
        lr_enh = 0.0
        if enh_mode == "joint":
            lr_enh = 0.0 if epoch < warmup else lr0 * enh_ratio * cos
            opt.param_groups[1]["lr"] = lr_enh
        if enhancer is not None:
            enhancer.train() if enh_mode == "joint" else enhancer.eval()
        m.train(); m.model[-1].training = True
        s = {"t": 0.0, "y": 0.0, "a": 0.0, "v": 0.0}; nb = 0; t0 = time.time()
        for low, tg in loader:
            low = low.to(device, non_blocking=True); tg = move_targets(tg, device)
            opt.zero_grad(set_to_none=True)
            L, items, _ = forward_losses(enhancer, base_frozen, m, det_loss_fn, low, tg, lam, use_amp, enh_mode)
            scaler.scale(L).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
            scaler.step(opt); scaler.update()
            nb += 1
            s["t"] += float(L); s["y"] += items["yolo"]; s["a"] += items["anchor"]; s["v"] += items["tv"]
        d = max(nb, 1)
        print(f"  [epoch {epoch}] L={s['t']/d:.3f} yolo={s['y']/d:.3f} anchor={s['a']/d:.4f} "
              f"tv={s['v']/d:.4f} lr_det={lr_det:.2e} lr_enh={lr_enh:.2e} ({time.time()-t0:.0f}s)")
        cw.writerow([epoch, f"{lr_det:.3e}", f"{lr_enh:.3e}", f"{s['t']/d:.4f}",
                     f"{s['y']/d:.4f}", f"{s['a']/d:.5f}", f"{s['v']/d:.6f}"])
        csvf.flush()
        # 저장: detector 항상; enhancer 는 joint 만(학습됨). frozen/none 은 tv01/없음.
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            yolo.save(str(ck / "detector.pt"))
            if enh_mode == "joint":
                torch.save({"model": enhancer.state_dict(),
                            "config": torch.load(tv01, map_location="cpu", weights_only=False)["config"],
                            "epoch": epoch}, ck / "enhancer.pth")
    csvf.close()
    print(HR)
    print(f"  완료. detector→{ck/'detector.pt'}" +
          (f"  enhancer→{ck/'enhancer.pth'}" if enh_mode == "joint" else ""))
    if enh_mode == "none":      # C'
        ev = f"python experiments/eval_p3_detail.py --yolo_weights {ck/'detector.pt'} --split test --label \"{exp} [test]\""
    elif enh_mode == "frozen":  # D'
        ev = f"python experiments/eval_p3_detail.py --enhancer_ckpt {tv01} --yolo_weights {ck/'detector.pt'} --split test --label \"{exp} [test]\""
    else:                       # E
        ev = f"python experiments/eval_p3_detail.py --enhancer_ckpt {ck/'enhancer.pth'} --yolo_weights {ck/'detector.pt'} --split test --label \"{exp} [test]\""
    print(f"  평가: {ev}")
    print(HR)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="P3-E joint enhancer+detector")
    p.add_argument("--config", type=str, default="configs/p3_joint_budget.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_steps", type=int, default=20)
    p.add_argument("--smoke_batch", type=int, default=4)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--enh_mode", type=str, default="joint", choices=["joint", "frozen", "none"],
                   help="joint=E(enhancer 학습) | frozen=D'(tv01 동결) | none=C'(raw)")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--enh_lr_ratio", type=float, default=0.1, help="enhancer lr = detector lr * ratio")
    p.add_argument("--lambda_det", type=float, default=0.02)
    p.add_argument("--lambda_rec", type=float, default=1.0)
    p.add_argument("--lambda_tv", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run_smoke(args) if args.smoke else run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())

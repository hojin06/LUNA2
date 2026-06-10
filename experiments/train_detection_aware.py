"""[P2] Detection-aware 학습 루프 + 그래디언트 흐름 스모크.

L_total = λ_rec·L_anchor + λ_det·L_yolo
  · L_anchor = L1(enhanced, base(low).detach())   # base=guidefix 출력 distill (drift 방지)
  · L_yolo   = YoloDetectionLoss(enhanced, GT bbox)  # frozen YOLOv8n, grad→enhancer

데이터: ExDark train split (configs/exdark_split_provisional.csv), parse_bbgt_v3 +
EXDARK_TO_COCO → (cls,cx,cy,w,h)[0,1] → {cls,bboxes,batch_idx}. 학습 배치는 고정
crop(가변 native 는 stack 불가); 평가는 eval_detection_native.py 로 native 수행.

AMP: enhancer forward 만 autocast, 손실(YOLO/L1)은 fp32 (P1 NaN 버그 교훈).

사용:
  스모크:  python experiments/train_detection_aware.py --config configs/joint_train.yaml --smoke
  본학습:  python experiments/train_detection_aware.py --config configs/joint_train.yaml --lambda_det 0.005 --exp_name p2_det_l0005
"""
from __future__ import annotations

import argparse
import csv as csvmod
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from experiments.eval_detection import collect_exdark_samples, parse_bbgt_v3
from src.losses.detection_aware import YoloDetectionLoss, grid_tv_loss
from src.models.bilateral_grid import build_from_config
from src.utils.inference import guard_output, is_nonfinite
from src.utils.paths import load_paths

HR = "=" * 80


# ===========================================================================
# ExDark detection dataset (train split, GT bbox → YOLO 포맷)
# ===========================================================================
class ExDarkDetectionDataset(Dataset):
    """ExDark 이미지 + GT bbox → (img[-1,1] (3,S,S), targets(N,5))=[cls,cx,cy,w,h]."""

    def __init__(self, exdark_root: Path, split_csv: Path, split: str,
                 image_size: int = 384, filter_empty: bool = True) -> None:
        super().__init__()
        self.image_size = image_size
        want = {"train": 1, "val": 2, "test": 3}[split]
        smap = {}
        for r in csvmod.DictReader(open(split_csv, encoding="utf-8")):
            smap[(r["class_dir"], r["image_name"])] = int(r["split"])
        self.items: List[Tuple] = []
        for sm in collect_exdark_samples(exdark_root, splits=None):
            if smap.get((sm.class_dir, sm.image_path.name)) != want:
                continue
            recs = parse_bbgt_v3(sm.ann_path)  # (coco_id,x1,y1,x2,y2) 원본 px
            if filter_empty and not recs:
                continue
            self.items.append((sm.image_path, recs))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, recs = self.items[idx]
        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        img = TF.resize(pil, [self.image_size, self.image_size],
                        interpolation=InterpolationMode.BILINEAR)
        t = TF.to_tensor(img) * 2.0 - 1.0          # [-1,1]
        rows = []
        for (cid, x1, y1, x2, y2) in recs:
            cx = ((x1 + x2) * 0.5) / W
            cy = ((y1 + y2) * 0.5) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
            if bw > 0 and bh > 0:
                rows.append([float(cid), cx, cy, bw, bh])
        tg = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5))
        return t, tg


def exdark_collate(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    cls_p, box_p, bi_p = [], [], []
    for i, tg in enumerate(targets):
        if tg.size(0) == 0:
            continue
        cls_p.append(tg[:, 0:1])
        box_p.append(tg[:, 1:5])
        bi_p.append(torch.full((tg.size(0),), i, dtype=torch.float32))
    if cls_p:
        cls = torch.cat(cls_p); box = torch.cat(box_p); bi = torch.cat(bi_p)
    else:
        cls = torch.zeros((0, 1)); box = torch.zeros((0, 4)); bi = torch.zeros((0,))
    return imgs, {"cls": cls, "bboxes": box, "batch_idx": bi}


def move_targets(tg: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in tg.items()}


# ===========================================================================
# 공통 빌드
# ===========================================================================
def build(cfg: dict, args, device: str):
    P = load_paths()
    base_ckpt = Path(args.base) if args.base else P.p2_base
    state = torch.load(base_ckpt, map_location=device, weights_only=False)
    mcfg = state.get("config") or cfg

    enhancer = build_from_config(mcfg).to(device)
    enhancer.load_state_dict(state["model"], strict=False)

    base_frozen = build_from_config(mcfg).to(device).eval()  # anchor (distill target)
    base_frozen.load_state_dict(state["model"], strict=False)
    for p in base_frozen.parameters():
        p.requires_grad_(False)

    det_loss = YoloDetectionLoss(P.yolov8n, device=device)
    return enhancer, base_frozen, det_loss, base_ckpt


def build_loader(cfg: dict, args, image_size: int):
    P = load_paths()
    dc = cfg["data"]
    split_csv = _LUNA2_ROOT / dc["split_csv"]
    ds = ExDarkDetectionDataset(P.exdark, split_csv, dc.get("train_split", "train"),
                                image_size=image_size,
                                filter_empty=dc.get("filter_empty", True))
    if args.max_train_samples > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.max_train_samples, len(ds)))))
    return DataLoader(ds, batch_size=args.batch_size or cfg["training"]["batch_size"],
                      shuffle=True, num_workers=args.num_workers, drop_last=True,
                      collate_fn=exdark_collate)


# ===========================================================================
# 스모크 — 그래디언트 흐름 확정 ([6])
# ===========================================================================
def run_smoke(cfg: dict, args) -> int:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    image_size = cfg["data"].get("crop_size", 384)
    lam_rec = cfg["loss"].get("lambda_rec", 1.0)
    lam_det = args.lambda_det if args.lambda_det is not None else cfg["loss"].get("lambda_det", 0.005)
    lam_tv = args.lambda_tv if args.lambda_tv is not None else cfg["loss"].get("lambda_tv", 0.0)

    print(HR)
    print(" [P2 스모크] detection-aware 그래디언트 흐름 + 학습성 확인 (fp32, 본학습 X)")
    print(HR)
    enhancer, base_frozen, det_loss, base_ckpt = build(cfg, args, device)
    enhancer.train(); det_loss.train()
    loader = build_loader(cfg, args, image_size)
    print(f"  base(P2)   : {base_ckpt}")
    print(f"  train set  : {len(loader.dataset)} (crop {image_size}, batch {loader.batch_size})")
    print(f"  λ_rec={lam_rec}  λ_det={lam_det}  λ_tv={lam_tv}  device={device}")
    print("-" * 80)

    it = iter(loader)
    low, tg = next(it)
    low = low.to(device); tg = move_targets(tg, device)

    # ---- (a) L_yolo 단독 backward → grad 흐름 확인 ----
    enhancer.zero_grad(set_to_none=True)
    enh = guard_output(enhancer(low))
    L_yolo, items = det_loss(enh, tg)
    L_yolo.backward()
    enh_gnorm = float(sum(p.grad.detach().pow(2).sum()
                          for p in enhancer.parameters() if p.grad is not None) ** 0.5)
    yolo_grad_cnt = sum(1 for p in det_loss.detector.parameters()
                        if p.grad is not None and float(p.grad.abs().sum()) > 0)
    print(" (a) 그래디언트 흐름 (L_yolo 단독 backward):")
    print(f"     L_yolo={float(L_yolo):.4f} (box/cls/dfl={[round(float(x),2) for x in items]})")
    print(f"     ★ enhancer param grad-norm = {enh_gnorm:.4e}  (>0 이어야 → 흐름 OK)")
    print(f"     ★ frozen YOLO param grad>0 개수 = {yolo_grad_cnt}  (0 이어야 → detector 동결)")
    enhancer.zero_grad(set_to_none=True)
    print("-" * 80)

    # ---- (b)(c) 정상 학습 20 step: loss finite / 비유한 출력 0 / Δw>0 ----
    opt = torch.optim.Adam(enhancer.parameters(), lr=1e-4)
    w0 = enhancer.coefficient_net.to_grid.weight.detach().clone()
    nonfinite_out = 0
    n = args.smoke_steps
    print(f" (b)(c) 정상 학습 {n} step (L_total = λ_rec·L1 + λ_det·L_yolo + λ_tv·L_gridtv):")
    print(f"     {'step':>4} | {'L_total':>8} {'L_anchor':>9} {'L_yolo':>8} {'L_tv':>8} | finite")
    it = iter(loader)
    for step in range(1, n + 1):
        try:
            low, tg = next(it)
        except StopIteration:
            it = iter(loader); low, tg = next(it)
        low = low.to(device); tg = move_targets(tg, device)
        opt.zero_grad(set_to_none=True)
        enh_raw, grid = enhancer(low, return_grid=True)
        enh = guard_output(enh_raw)
        if is_nonfinite(enh_raw):  # 가드 전 원시 출력 검사
            nonfinite_out += 1
        with torch.no_grad():
            base_out = base_frozen(low)
        L_anchor = F.l1_loss(enh, base_out.detach())
        L_yolo, items = det_loss(enh, tg)
        L_tv = grid_tv_loss(grid.float())
        L_total = lam_rec * L_anchor + lam_det * L_yolo + lam_tv * L_tv
        fin = bool(torch.isfinite(L_total))
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(enhancer.parameters(), 1.0)
        opt.step()
        if step <= 5 or step % 5 == 0:
            print(f"     {step:>4} | {float(L_total):>8.4f} {float(L_anchor):>9.4f} "
                  f"{float(L_yolo):>8.4f} {float(L_tv):>8.4f} | {fin}")
    dw = float((enhancer.coefficient_net.to_grid.weight.detach() - w0).norm())
    print("-" * 80)
    print(f" (b) 비유한 원시출력 step 수 = {nonfinite_out}/{n}  (0 이어야 정상)")
    print(f" (c) enhancer to_grid ‖Δw‖ = {dw:.4e}  (>0 이어야 param 갱신)")

    # ---- (d) enhanced 5장 저장 (adversarial artifact 육안) ----
    out_dir = load_paths().runs / "p2_smoke" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    enhancer.eval()
    with torch.no_grad():
        low, tg = next(iter(loader))
        low = low.to(device)
        enh = guard_output(enhancer(low))
        k = min(5, low.size(0))
        for i in range(k):
            trio = torch.stack([low[i], enh[i]], 0)
            save_image(((trio + 1) * 0.5).clamp(0, 1), str(out_dir / f"smoke_{i}.png"), nrow=2)
    print(f" (d) enhanced 샘플 {k}장 저장 → {out_dir}  (low|enhanced)")
    print(HR)
    print(" 판정: (a) enhancer grad>0 & YOLO grad=0 → 흐름 확정 / (b)(c) finite·Δw>0 → 학습 가능")
    print(HR)
    return 0


# ===========================================================================
# 본학습 ([7] sweep 용) — 실행은 사용자가
# ===========================================================================
def run_train(cfg: dict, args) -> int:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    P = load_paths()
    tr = cfg["training"]; lg = cfg["logging"]
    image_size = cfg["data"].get("crop_size", 384)
    lam_rec = cfg["loss"].get("lambda_rec", 1.0)
    lam_det = args.lambda_det if args.lambda_det is not None else cfg["loss"].get("lambda_det", 0.005)
    lam_tv = args.lambda_tv if args.lambda_tv is not None else cfg["loss"].get("lambda_tv", 0.0)
    exp = args.exp_name or cfg["experiment_name"]
    use_amp = tr.get("amp", True) and device.startswith("cuda")
    grad_clip = float(tr.get("grad_clip", 1.0))
    epochs = args.epochs or tr.get("num_epochs", 10)

    run_dir = P.runs / exp
    ckpt_dir = run_dir / "checkpoints"; log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)

    enhancer, base_frozen, det_loss, base_ckpt = build(cfg, args, device)
    det_loss.train()
    loader = build_loader(cfg, args, image_size)
    opt = torch.optim.Adam(enhancer.parameters(), lr=float(tr.get("lr", 1e-4)),
                           betas=tuple(tr.get("betas", [0.9, 0.999])))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=float(tr.get("lr", 1e-4)) * float(tr.get("eta_min_ratio", 0.01)))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(HR)
    print(f" [P2 본학습] {exp}  λ_rec={lam_rec} λ_det={lam_det} λ_tv={lam_tv}  base={base_ckpt.name}")
    print(f"  train {len(loader.dataset)}  crop {image_size}  batch {loader.batch_size}  "
          f"epochs {epochs}  amp {use_amp}")
    print(HR)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    csvf = open(log_dir / "train_log.csv", "w", newline="", encoding="utf-8")
    cw = csvmod.writer(csvf); cw.writerow(["epoch", "lr", "L_total", "L_anchor", "L_yolo", "L_tv"])

    gstep = 0
    for epoch in range(epochs):
        enhancer.train(); det_loss.train()
        sums = {"t": 0.0, "a": 0.0, "y": 0.0, "v": 0.0}; nb = 0; t0 = time.time()
        for low, tg in loader:
            low = low.to(device, non_blocking=True); tg = move_targets(tg, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                enh_raw, grid = enhancer(low, return_grid=True)
            enh = guard_output(enh_raw.float())            # 손실은 fp32
            with torch.no_grad():
                base_out = base_frozen(low).float()
            L_anchor = F.l1_loss(enh, base_out.detach())
            L_yolo, _ = det_loss(enh, tg)
            L_tv = grid_tv_loss(grid.float())
            L_total = lam_rec * L_anchor + lam_det * L_yolo + lam_tv * L_tv
            scaler.scale(L_total).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enhancer.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            gstep += 1; nb += 1
            sums["t"] += float(L_total); sums["a"] += float(L_anchor)
            sums["y"] += float(L_yolo); sums["v"] += float(L_tv)
        sched.step()
        d = max(nb, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  [epoch {epoch}] L_total={sums['t']/d:.4f} L_anchor={sums['a']/d:.4f} "
              f"L_yolo={sums['y']/d:.4f} L_tv={sums['v']/d:.5f} lr={lr_now:.2e} ({time.time()-t0:.0f}s)")
        cw.writerow([epoch, f"{lr_now:.3e}", f"{sums['t']/d:.5f}", f"{sums['a']/d:.5f}",
                     f"{sums['y']/d:.5f}", f"{sums['v']/d:.6f}"])
        csvf.flush()
        if (epoch + 1) % lg.get("save_every", 2) == 0 or epoch + 1 == epochs:
            torch.save({"model": enhancer.state_dict(), "config": cfg, "epoch": epoch,
                        "lambda_det": lam_det, "lambda_tv": lam_tv}, ckpt_dir / "last.pth")
    torch.save({"model": enhancer.state_dict(), "config": cfg, "epoch": epochs - 1,
                "lambda_det": lam_det, "lambda_tv": lam_tv}, ckpt_dir / "last.pth")
    csvf.close()
    print(f"  완료. 산출물: {run_dir}")
    print(f"  평가: python experiments/eval_detection_native.py --mode enhance --split test "
          f"--checkpoint {ckpt_dir / 'last.pth'} --label '{exp} [test]'")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2 detection-aware 학습/스모크")
    p.add_argument("--config", type=str, default="configs/joint_train.yaml")
    p.add_argument("--smoke", action="store_true", help="그래디언트 흐름 스모크만")
    p.add_argument("--smoke_steps", type=int, default=20)
    p.add_argument("--lambda_det", type=float, default=None, help="λ_det override (sweep)")
    p.add_argument("--lambda_tv", type=float, default=None, help="λ_tv (grid spatial TV) override")
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--base", type=str, default=None, help="base 체크포인트(기본 paths.p2_base)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        return run_smoke(cfg, args)
    return run_train(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())

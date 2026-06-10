"""LUNA2 Phase 1 학습 — BilateralLowLightNet 지도학습 (적대적 없음).

개요
----
* config(``configs/bilateral_base.yaml``) 기반. 경로는 ``configs/paths.yaml`` 주입.
* 데이터: LOL v1 + LOL-v2 Real + LoLI-Street paired (CombinedDataset) + PairedAugment.
* 손실 : ``CombinedRestorationLoss`` (L1 + VGG perceptual + SSIM, 원본 LUNA 미러링).
* 최적화: Adam + (cosine) schedule + 선택적 AMP. **Stage 1 지도학습만.**
* 평가 : LOL eval15 기준 PSNR/SSIM (``src.utils.metrics.evaluate``). best 체크포인트 저장.
* 산출물: ``runs/{experiment_name}/`` (checkpoints / logs(csv,tensorboard) / samples / config.yaml).
* resume: ``--resume <ckpt>`` 또는 ``runs/{exp}/checkpoints/last.pth`` 자동 감지.

학습 해상도 vs 추론 해상도
--------------------------
학습은 ``crop_size``(기본 256) random crop 으로 진행하지만, 추론 시 모델은
**네이티브 해상도**(예: 640×480)를 그대로 받는다. CoefficientNet 이 어느 해상도든
내부적으로 ``low_res``(256)로 다운샘플하므로 학습/추론 해상도가 달라도 동작이
일관된다 (bilateral grid 의 해상도 불변성).

사용 예
-------
.. code-block:: bash

    python experiments/train.py --config configs/bilateral_base.yaml
    python experiments/train.py --config configs/bilateral_base.yaml --resume runs/bilateral_base/checkpoints/last.pth
    # 스모크 (작은 subset, 50 iter):
    python experiments/train.py --config configs/bilateral_base.yaml \
        --exp_name smoke --datasets lol_v1 --max_train_samples 50 \
        --batch_size 2 --max_iters 50 --num_workers 0
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# --- LUNA2 루트를 sys.path 에 등록 ---
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
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.lowlight_dataset import CombinedDataset, build_dataset_by_name
from src.losses.restoration import build_restoration_loss
from src.models.bilateral_grid import build_from_config
from src.utils.metrics import evaluate
from src.utils.paths import load_paths

HRULE = "=" * 78

# 데이터셋 키 → paths.yaml 의 데이터셋 키
_DATASET_TO_PATHKEY = {
    "lol_v1": "lol_v1",
    "lol_v2_real": "lol_v2",
    "lol_v2_syn": "lol_v2",
    "loli_street": "loli_street",
}


# ===========================================================================
# Config
# ===========================================================================
def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Datasets
# ===========================================================================
def build_train_dataset(cfg: dict, paths, crop_size: int, dataset_keys: List[str]):
    """train_datasets 키 목록 → CombinedDataset (누락은 경고 후 skip)."""
    data_cfg = cfg["data"]
    split = data_cfg.get("train_split", "train")
    full_resize = data_cfg.get("full_resize", False)

    datasets = []
    for key in dataset_keys:
        if key not in _DATASET_TO_PATHKEY:
            print(f"  [warn] 알 수 없는 데이터셋 키 '{key}' — skip")
            continue
        root = paths.dataset(_DATASET_TO_PATHKEY[key])
        try:
            ds = build_dataset_by_name(
                name=key, data_root=root, split=split,
                image_size=crop_size, augment=True, full_resize=full_resize,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [skip] {key}: {e}")
            continue
        datasets.append(ds)
        print(f"  [train] {key}: {len(ds)} pairs  (root={root})")

    if not datasets:
        raise RuntimeError("학습 데이터셋이 0 개입니다. paths.yaml / --datasets 확인.")
    return CombinedDataset(datasets) if len(datasets) > 1 else datasets[0]


def build_eval_dataset(cfg: dict, paths, eval_size: int):
    """eval_dataset → resize-only(augment=False) 데이터셋 (PSNR/SSIM 기준)."""
    data_cfg = cfg["data"]
    key = data_cfg.get("eval_dataset", "lol_v1")
    split = data_cfg.get("eval_split", "eval")
    root = paths.dataset(_DATASET_TO_PATHKEY[key])
    ds = build_dataset_by_name(
        name=key, data_root=root, split=split,
        image_size=eval_size, augment=False, full_resize=False,
    )
    print(f"  [eval]  {key}[{split}]: {len(ds)} pairs  (resize {eval_size}, augment off)")
    return ds


# ===========================================================================
# Checkpoint
# ===========================================================================
def save_ckpt(path: Path, model, optimizer, scheduler, scaler, cfg,
              epoch: int, global_step: int, best_psnr: float, best_ssim: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_cfg": cfg["model"],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "config": cfg,
    }, path)


def load_ckpt(path: Path, model, optimizer, scheduler, scaler, device: str) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


# ===========================================================================
# 샘플 이미지 저장 ([low | enhanced | high])
# ===========================================================================
@torch.no_grad()
def save_samples(model, eval_loader, out_path: Path, n: int, device: str) -> None:
    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        return
    model.eval()
    rows = []
    count = 0
    for low, high in eval_loader:
        low = low.to(device)
        high = high.to(device)
        enh = model(low).clamp(-1.0, 1.0)
        for b in range(low.size(0)):
            trio = torch.stack([low[b], enh[b], high[b]], dim=0)  # (3,3,H,W)
            rows.append(trio)
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    if not rows:
        return
    grid_in = torch.cat(rows, dim=0)            # (n*3, 3, H, W)
    grid_in = (grid_in + 1.0) * 0.5             # [-1,1] → [0,1]
    grid = make_grid(grid_in, nrow=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LUNA2 Phase 1 학습 (BilateralLowLightNet)")
    p.add_argument("--config", type=str, default="configs/bilateral_base.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="체크포인트 경로 (없으면 runs/{exp}/checkpoints/last.pth 자동 탐지)")
    p.add_argument("--device", type=str, default=None)
    # --- override / smoke 옵션 ---
    p.add_argument("--exp_name", type=str, default=None, help="experiment_name 덮어쓰기")
    p.add_argument("--datasets", type=str, default=None,
                   help="train_datasets 덮어쓰기 (쉼표구분, 예: lol_v1,lol_v2_real)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--crop_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_train_samples", type=int, default=0,
                   help=">0 이면 train subset 앞 N개만 (스모크)")
    p.add_argument("--max_eval_samples", type=int, default=0,
                   help=">0 이면 eval subset 앞 N개만 (스모크)")
    p.add_argument("--max_iters", type=int, default=0,
                   help=">0 이면 총 optimizer step 을 N 으로 제한 (스모크)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_tensorboard", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.exp_name:
        cfg["experiment_name"] = args.exp_name
    if args.datasets:
        cfg["data"]["train_datasets"] = [s.strip() for s in args.datasets.split(",") if s.strip()]
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.crop_size is not None:
        cfg["data"]["crop_size"] = args.crop_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.no_amp:
        cfg["training"]["amp"] = False
    if args.no_tensorboard:
        cfg["logging"]["tensorboard"] = False
    return cfg


# ===========================================================================
# Train
# ===========================================================================
def train(cfg: dict, args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))
    paths = load_paths()

    tr_cfg = cfg["training"]
    data_cfg = cfg["data"]
    log_cfg = cfg["logging"]

    exp = cfg["experiment_name"]
    run_dir = paths.runs / exp
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    sample_dir = run_dir / "samples"
    for d in (ckpt_dir, log_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    crop_size = data_cfg.get("crop_size", 256)
    eval_size = data_cfg.get("eval_size", 256)
    batch_size = tr_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    num_epochs = tr_cfg.get("num_epochs", 200)
    use_amp = tr_cfg.get("amp", True) and device.startswith("cuda")
    grad_clip = float(tr_cfg.get("grad_clip", 0.0))

    print(HRULE)
    print(f" LUNA2 Phase 1 학습 — {exp}")
    print(HRULE)
    print(f"  device      : {device}   AMP={use_amp}")
    print(f"  crop_size   : {crop_size}  (학습)   |  추론은 네이티브 해상도")
    print(f"  batch/epoch : {batch_size} / {num_epochs}")
    print(f"  run_dir     : {run_dir}")
    print(HRULE)

    # --- 데이터 ---
    dataset_keys = data_cfg.get("train_datasets", ["lol_v1"])
    train_set = build_train_dataset(cfg, paths, crop_size, dataset_keys)
    eval_set = build_eval_dataset(cfg, paths, eval_size)

    if args.max_train_samples > 0:
        n = min(args.max_train_samples, len(train_set))
        train_set = Subset(train_set, list(range(n)))
        print(f"  [smoke] train subset → {n}")
    if args.max_eval_samples > 0:
        n = min(args.max_eval_samples, len(eval_set))
        eval_set = Subset(eval_set, list(range(n)))
        print(f"  [smoke] eval subset → {n}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.startswith("cuda"),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=1, shuffle=False,
        num_workers=min(num_workers, 2), pin_memory=device.startswith("cuda"),
    )
    print(f"  train iters/epoch : {len(train_loader)}")
    print(HRULE)

    # --- 모델 / 손실 / 옵티마이저 ---
    model = build_from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params : {n_params:,}  ({n_params / 1e3:.1f} K)")

    criterion = build_restoration_loss(cfg).to(device)

    lr = float(tr_cfg.get("lr", 2e-4))
    betas = tuple(tr_cfg.get("betas", [0.9, 0.999]))
    wd = float(tr_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

    scheduler = None
    if tr_cfg.get("scheduler", "cosine") == "cosine":
        eta_min = lr * float(tr_cfg.get("eta_min_ratio", 0.01))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=eta_min)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- resume ---
    start_epoch = 0
    global_step = 0
    best_psnr = -1.0
    best_ssim = -1.0
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
    elif (ckpt_dir / "last.pth").is_file():
        resume_path = ckpt_dir / "last.pth"
    if resume_path and resume_path.is_file():
        state = load_ckpt(resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)
        best_psnr = state.get("best_psnr", -1.0)
        best_ssim = state.get("best_ssim", -1.0)
        print(f"  resumed from {resume_path}  (epoch {start_epoch}, step {global_step})")
        print(HRULE)

    # --- config 덤프 ---
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # --- 로깅 (csv + tensorboard) ---
    csv_path = log_dir / "train_log.csv"
    csv_new = not csv_path.is_file()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_new:
        csv_writer.writerow(["epoch", "global_step", "lr",
                             "train_total", "train_l1", "train_vgg", "train_ssim",
                             "val_psnr", "val_ssim"])

    writer = None
    if log_cfg.get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        except Exception:
            print("  [info] tensorboard 미설치 — CSV 로깅만 사용")

    # --- 학습 루프 ---
    stop = False
    for epoch in range(start_epoch, num_epochs):
        model.train()
        ep_sums = {"total": 0.0, "l1": 0.0, "vgg": 0.0, "ssim": 0.0}
        n_batches = 0
        t0 = time.time()

        for low, high in train_loader:
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # 모델 forward 만 autocast(fp16). 손실(SSIM/VGG 의 division·conv)은
            # autocast 밖 fp32 로 계산 — fp16 SSIM 이 어두운 입력에서 NaN grad 를
            #만들어 GradScaler 가 매 step skip 하던 버그 차단.
            with torch.cuda.amp.autocast(enabled=use_amp):
                enhanced = model(low)
            losses = criterion(enhanced.float(), high.float())
            loss = losses["total"]

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # --- 가벼운 가드: loss finite 확인 + GradScaler step-skip 경고 ---
            if not torch.isfinite(loss):
                print(f"  [WARN] step {global_step + 1}: loss 가 finite 가 아님 "
                      f"({loss.item()}) — backward 비정상.")
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before and global_step < 50:
                print(f"  [WARN] step {global_step + 1}: GradScaler step SKIP "
                      f"(scale {scale_before:.0f}→{scaler.get_scale():.0f}; inf/nan grad). "
                      f"초기 calibration 이면 정상, 지속되면 NaN grad 의심.")

            global_step += 1
            n_batches += 1
            for k in ep_sums:
                ep_sums[k] += float(losses[k])

            if writer is not None:
                writer.add_scalar("train/total", float(loss), global_step)

            if global_step % 10 == 0 or (args.max_iters and global_step <= 5):
                print(f"  e{epoch} step {global_step}  "
                      f"total={float(loss):.4f}  l1={float(losses['l1']):.4f}  "
                      f"vgg={float(losses['vgg']):.4f}  ssim={float(losses['ssim']):.4f}")

            if args.max_iters and global_step >= args.max_iters:
                print(f"  [smoke] max_iters={args.max_iters} 도달 — 학습 중단")
                stop = True
                break

        # epoch 평균
        denom = max(n_batches, 1)
        avg = {k: v / denom for k, v in ep_sums.items()}
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"  [epoch {epoch}] avg total={avg['total']:.4f}  "
              f"l1={avg['l1']:.4f} vgg={avg['vgg']:.4f} ssim={avg['ssim']:.4f}  "
              f"lr={cur_lr:.2e}  ({dt:.1f}s)")

        if scheduler is not None:
            scheduler.step()

        # --- validation ---
        val_psnr = val_ssim = float("nan")
        do_eval = ((epoch + 1) % log_cfg.get("eval_every", 1) == 0) or stop
        if do_eval:
            metrics = evaluate(model, eval_loader, device=device)
            val_psnr, val_ssim = metrics["psnr"], metrics["ssim"]
            print(f"  [val   {epoch}] PSNR={val_psnr:.4f} dB  SSIM={val_ssim:.4f}  (n={metrics['n']})")
            if writer is not None:
                writer.add_scalar("val/psnr", val_psnr, epoch)
                writer.add_scalar("val/ssim", val_ssim, epoch)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_ckpt(ckpt_dir / "best_psnr.pth", model, optimizer, scheduler,
                          scaler, cfg, epoch, global_step, best_psnr, best_ssim)
                print(f"           ↳ best PSNR 갱신 → best_psnr.pth")
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_ckpt(ckpt_dir / "best_ssim.pth", model, optimizer, scheduler,
                          scaler, cfg, epoch, global_step, best_psnr, best_ssim)
                print(f"           ↳ best SSIM 갱신 → best_ssim.pth")

        # --- csv ---
        csv_writer.writerow([epoch, global_step, f"{cur_lr:.6e}",
                             f"{avg['total']:.6f}", f"{avg['l1']:.6f}",
                             f"{avg['vgg']:.6f}", f"{avg['ssim']:.6f}",
                             f"{val_psnr:.6f}", f"{val_ssim:.6f}"])
        csv_file.flush()

        # --- 샘플 이미지 ---
        if ((epoch + 1) % log_cfg.get("sample_every", 5) == 0) or stop:
            save_samples(model, eval_loader, sample_dir / f"epoch_{epoch:03d}.png",
                         n=log_cfg.get("num_samples", 4), device=device)

        # --- last 체크포인트 ---
        if ((epoch + 1) % log_cfg.get("save_every", 10) == 0) or stop or (epoch + 1 == num_epochs):
            save_ckpt(ckpt_dir / "last.pth", model, optimizer, scheduler,
                      scaler, cfg, epoch, global_step, best_psnr, best_ssim)

        if stop:
            break

    # 마지막에 항상 last 저장
    save_ckpt(ckpt_dir / "last.pth", model, optimizer, scheduler,
              scaler, cfg, min(epoch, num_epochs - 1), global_step, best_psnr, best_ssim)
    csv_file.close()
    if writer is not None:
        writer.close()

    print(HRULE)
    print(f"  학습 종료. best PSNR={best_psnr:.4f}  best SSIM={best_ssim:.4f}")
    print(f"  산출물: {run_dir}")
    print(HRULE)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    train(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

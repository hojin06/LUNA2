"""P1 핵심 측정 — BilateralLowLightNet 네이티브 해상도 ExDark 검출 mAP.

목적
----
bilateral_phase1_l1only_fixed 의 enhancer 를 **네이티브 해상도**(256 다운샘플 없이)
로 ExDark 7363장에 적용 → frozen YOLOv8n → mAP. Original(0.447) / down256(0.338) /
LUNA(0.346) 과 **완전 동일 조건**(YOLO 입력·좌표계·DetectionAccumulator)으로 비교.

동일 조건 보장
--------------
* YOLO 호출: ``eval_detection._run_yolo`` 재사용 (imgsz 미전달 → 기본 640 고정).
* 좌표계: enhancer 출력이 원본 H×W 그대로이므로 예측이 GT(원본 픽셀)와 동일 좌표계
  (0단계 진단의 ``native`` / ``down256_up`` 조건과 동일, box 역스케일 불필요).
* 누적: ``eval_detection.DetectionAccumulator`` 재사용.
* 향상 추론: ``src.utils.inference.enhance`` (fp32 + nan_to_num+clamp 가드).
  비유한 출력 이미지 수를 카운트해 가드 동작을 확인.

본학습 없음. paths.yaml 사용.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

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
import torchvision.transforms.functional as TF
from PIL import Image

# 0단계 진단/검출 평가의 native 경로를 그대로 재사용 (동일 조건 보장)
from experiments.eval_detection import (
    DetectionAccumulator, _extract_boxes, _fmt, _run_yolo,
    _try_tqdm, collect_exdark_samples, parse_bbgt_v3,
)
from src.models.bilateral_grid import build_from_config
from src.models.luna_base import norm_tensor_to_uint8_rgb
from src.utils.inference import enhance, is_nonfinite
from src.utils.paths import load_paths

HR = "=" * 104
SUB = "-" * 104

# 이전에 동일 native 파이프라인으로 측정된 참조값 (runs/diagnostic, eval_detection*)
REFERENCE = [
    ("Original (native)", 0.447154, 0.242137, 0.644937, 0.548292),
    ("down256 (256 직접)", 0.337593, 0.181317, 0.647980, 0.422185),
    ("LUNA-LoLI30K (256→up)", 0.346211, 0.187093, 0.661318, 0.422396),
    ("LUNA-LOLv2 (256→up)", 0.281525, 0.152489, 0.654310, 0.349895),
]


def load_bilateral(ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config") or {"model": state.get("model_cfg", {})}
    model = build_from_config(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)
    return model, state


@torch.no_grad()
def enhance_native_uint8(model, pil: Image.Image, device: str):
    """원본 PIL → 네이티브 해상도 향상 uint8 RGB + 비유한 출력 여부.

    256 다운샘플(이미지)을 하지 않는다 — bilateral 모델이 네이티브 입력을 받고
    CoefficientNet 내부에서만 low_res 로 다운샘플한다.
    """
    t = TF.to_tensor(pil) * 2.0 - 1.0          # [-1,1], (3,H,W) 네이티브
    x = t.unsqueeze(0).to(device)
    raw = model(x)                              # 가드 전 (비유한 검사용)
    nonfinite = is_nonfinite(raw)
    out = enhance(model, x)                     # fp32 + nan_to_num + clamp
    rgb = norm_tensor_to_uint8_rgb(out)         # (H,W,3) uint8, 네이티브 (리사이즈 없음)
    return rgb, nonfinite


def make_input_uint8(model, pil: Image.Image, device: str, mode: str):
    """mode='enhance' → 향상 uint8(+비유한), 'original' → 원본 uint8(향상 없음)."""
    if mode == "original":
        return np.array(pil, dtype=np.uint8), False
    return enhance_native_uint8(model, pil, device)


def load_split_map(csv_path: Path):
    """exdark_split CSV → {(class_dir, image_name): split_int}. 없으면 None."""
    import csv as _csv
    if not csv_path.is_file():
        return None
    m = {}
    for r in _csv.DictReader(open(csv_path, encoding="utf-8")):
        m[(r["class_dir"], r["image_name"])] = int(r["split"])
    return m


def parse_args() -> argparse.Namespace:
    P = load_paths()
    default_ckpt = P.runs / "bilateral_phase1_l1only_fixed" / "checkpoints" / "last.pth"
    p = argparse.ArgumentParser(
        description="BilateralLowLightNet 네이티브 해상도 ExDark mAP 측정")
    p.add_argument("--checkpoint", type=str, default=str(default_ckpt))
    p.add_argument("--exdark_root", type=str, default=str(P.exdark))
    p.add_argument("--yolo_weights", type=str, default=str(P.yolov8n))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--results_dir", type=str, default=str(P.runs / "eval_detection_native"))
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--all_splits", action="store_true")
    p.add_argument("--device", type=str, default=None)
    # P2 준비: split 필터 + Original/enhance 모드
    p.add_argument("--mode", type=str, default="enhance", choices=["enhance", "original"],
                   help="enhance=모델 향상, original=원본(향상 없음)")
    p.add_argument("--split_csv", type=str,
                   default=str(_LUNA2_ROOT / "configs" / "exdark_split_provisional.csv"))
    p.add_argument("--split", type=str, default="all",
                   choices=["all", "train", "val", "test"],
                   help="split_csv 기준 평가 대상 split")
    p.add_argument("--label", type=str, default=None, help="결과 행 라벨")
    a = p.parse_args()
    if a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    return a


def main() -> int:
    args = parse_args()
    device = args.device
    ckpt = Path(args.checkpoint)
    exdark_root = Path(args.exdark_root)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(HR)
    print(" P1 핵심 측정 — BilateralLowLightNet 네이티브 해상도 ExDark mAP")
    print(HR)
    print(f"  checkpoint   : {ckpt}")
    print(f"  exdark_root  : {exdark_root}")
    print(f"  yolo_weights : {args.yolo_weights}   conf={args.conf}   YOLO imgsz=기본(640, 미전달)")
    print(f"  device       : {device}")
    print(SUB)
    if not (exdark_root / "images").is_dir():
        print(f"[error] ExDark 폴더 없음: {exdark_root}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치. pip install ultralytics")
        return 1

    print(f"  mode         : {args.mode}   split : {args.split}")
    model = None
    if args.mode == "enhance":
        if not ckpt.is_file():
            print(f"[error] 체크포인트 없음: {ckpt}")
            return 1
        model, state = load_bilateral(ckpt, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  bilateral params : {n_params:,}  (epoch={state.get('epoch')}, "
              f"best_psnr={state.get('best_psnr')})")
    else:
        print("  (original 모드: 향상 없음, 모델 미로드)")
    yolo = YOLO(args.yolo_weights)
    print(f"  YOLO classes : {len(yolo.names)} (COCO)")

    # 전체 샘플 수집 후 split CSV 로 필터 (공식 imageclasslist 부재 → 잠정 split)
    samples = collect_exdark_samples(exdark_root, splits=None)
    if args.split != "all":
        smap = load_split_map(Path(args.split_csv))
        if smap is None:
            print(f"[error] split_csv 없음: {args.split_csv} (make_exdark_split.py 먼저 실행)")
            return 1
        want = {"train": 1, "val": 2, "test": 3}[args.split]
        samples = [s for s in samples if smap.get((s.class_dir, s.image_path.name)) == want]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"  samples      : {len(samples)} (네이티브, split={args.split})")
    print(SUB)

    acc = DetectionAccumulator()
    nonfinite_imgs = 0
    tqdm = _try_tqdm()
    iterator = tqdm(samples, desc="native", unit="img", ncols=100) if tqdm else samples

    with torch.no_grad():
        for img_id, sm in enumerate(iterator):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 로드 실패 {sm.image_path.name}: {e}")
                continue
            recs = parse_bbgt_v3(sm.ann_path)
            if recs:
                gt_boxes = np.array([[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in recs],
                                    dtype=np.float32)
                gt_cls = np.array([c for (c, *_r) in recs], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_cls = np.zeros((0,), dtype=np.int64)

            enh_rgb, nonfinite = make_input_uint8(model, pil, device, args.mode)
            if nonfinite:
                nonfinite_imgs += 1

            result = _run_yolo(yolo, enh_rgb, args.conf, device)
            pb, pc, pf = _extract_boxes(result)
            acc.add(img_id, gt_boxes, gt_cls, pb, pc, pf)

    res = acc.compute()
    print()
    print(HR)
    print(" 결과 — 네이티브 해상도 비교 (모두 동일 YOLO 경로/좌표계/누적기)")
    print(HR)
    print(f"  {'Model':<26} | {'mAP@0.5':>9} | {'mAP@.5:.95':>11} | "
          f"{'Precision':>10} | {'Recall':>8}")
    print(SUB)
    print(f"  {'Original (native)':<26} | {0.447154:>9.4f} | {0.242137:>11.4f} | "
          f"{0.644937:>10.4f} | {0.548292:>8.4f}")
    print(f"  {'down256 (256 직접)':<26} | {0.337593:>9.4f} | {0.181317:>11.4f} | "
          f"{0.647980:>10.4f} | {0.422185:>8.4f}")
    print(f"  {'LUNA-LoLI30K (256→up)':<26} | {0.346211:>9.4f} | {0.187093:>11.4f} | "
          f"{0.661318:>10.4f} | {0.422396:>8.4f}")
    print(f"  {'LUNA-LOLv2 (256→up)':<26} | {0.281525:>9.4f} | {0.152489:>11.4f} | "
          f"{0.654310:>10.4f} | {0.349895:>8.4f}")
    print(SUB)
    meas_label = args.label or f"{args.mode} [{args.split}]"
    print(f"  (위 참조값은 전체 7363장 기준)")
    print(f"  {meas_label:<26} | {res['map50']:>9.4f} | {res['map']:>11.4f} | "
          f"{res['p50']:>10.4f} | {res['r50']:>8.4f}   ← 측정값 (split={args.split}, n={res['n_images']})")
    print(HR)
    d_orig = res["map50"] - 0.447154
    print(f"  vs Original(native) ΔmAP@0.5 = {d_orig:+.4f}   "
          f"({'향상' if d_orig > 0 else '미달'})")
    print(f"  vs LUNA-LoLI30K     ΔmAP@0.5 = {res['map50'] - 0.346211:+.4f}")
    print(f"  vs down256          ΔmAP@0.5 = {res['map50'] - 0.337593:+.4f}")
    print(f"  비유한(NaN/Inf) 출력 이미지 : {nonfinite_imgs}/{len(samples)}  "
          f"(가드로 처리됨; 0 이면 출력 안정)")
    print(f"  n_images={res['n_images']}  n_preds={res['n_preds']}  n_gts={res['n_gts']}")
    print(HR)

    # CSV 저장
    csv_path = results_dir / "native_detection_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "n_images", "map50", "map", "precision", "recall"])
        for name, m50, m, pr, rc in REFERENCE:
            w.writerow([name, "full7363", 7363, f"{m50:.6f}", f"{m:.6f}", f"{pr:.6f}", f"{rc:.6f}"])
        w.writerow([meas_label, args.split, res["n_images"], f"{res['map50']:.6f}",
                    f"{res['map']:.6f}", f"{res['p50']:.6f}", f"{res['r50']:.6f}"])
    print(f"  Saved CSV → {csv_path}")
    print(f"  nonfinite_output_images = {nonfinite_imgs}")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

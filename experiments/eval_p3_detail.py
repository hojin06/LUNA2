"""[P3] class-wise AP + subset(극저조도 / small-object) 상세 평가.

eval_detection_native 와 동일 경로(raw native → YOLO → DetectionAccumulator)지만
per-class AP 와 이미지 subset 별 mAP 를 함께 보고. mode=original(raw)/enhance 지원.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import csv as csvmod
import numpy as np
import torch
from PIL import Image

from experiments.eval_detection import (
    COCO_TO_EXDARK, DetectionAccumulator, _extract_boxes, _run_yolo,
    _try_tqdm, collect_exdark_samples, parse_bbgt_v3,
)
from src.utils.paths import load_paths

HR = "=" * 84
LOWLIGHT_LUMA_THR = 0.10     # 극저조도: 이미지 평균 luma < 0.10
SMALL_AREA_THR = 0.01        # small-object: 최소 GT box normalized 면적 < 0.01


def main() -> int:
    P = load_paths()
    p = argparse.ArgumentParser(description="P3 class-wise + subset 평가")
    p.add_argument("--yolo_weights", type=str, required=True)
    p.add_argument("--model_class", type=str, default="yolo", choices=["yolo", "rtdetr"],
                   help="검출기 클래스 (yolo=YOLO/YOLO11, rtdetr=RT-DETR)")
    p.add_argument("--enhancer_ckpt", type=str, default=None,
                   help="주면 enhance 후 YOLO (D/E); 없으면 raw 입력 (A/C)")
    p.add_argument("--split_csv", type=str, default=str(_LUNA2_ROOT / "configs" / "exdark_split_provisional.csv"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--label", type=str, default="model")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_class == "rtdetr":
        from ultralytics import RTDETR
        yolo = RTDETR(args.yolo_weights)
    else:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_weights)

    # enhancer (D/E) — 있으면 raw 이미지를 enhance 후 YOLO
    enhancer = None
    if args.enhancer_ckpt:
        import torchvision.transforms.functional as TF
        from src.models.bilateral_grid import build_from_config
        from src.models.luna_base import norm_tensor_to_uint8_rgb
        from src.utils.inference import enhance as enhance_fn
        est = torch.load(args.enhancer_ckpt, map_location=device, weights_only=False)
        enhancer = build_from_config(est["config"]).to(device).eval()
        enhancer.load_state_dict(est["model"], strict=False)

    smap = {}
    for r in csvmod.DictReader(open(args.split_csv, encoding="utf-8")):
        smap[(r["class_dir"], r["image_name"])] = int(r["split"])
    want = {"train": 1, "val": 2, "test": 3}[args.split]
    samples = [s for s in collect_exdark_samples(P.exdark, splits=None)
               if smap.get((s.class_dir, s.image_path.name)) == want]

    print(HR)
    print(f" [P3 상세] {args.label}  split={args.split}  n={len(samples)}  weights={Path(args.yolo_weights).name}")
    print(HR)

    acc_all = DetectionAccumulator()
    acc_low = DetectionAccumulator()    # 극저조도 이미지
    acc_small = DetectionAccumulator()  # 작은 객체 포함 이미지
    n_low = n_small = 0
    tqdm = _try_tqdm()
    it = tqdm(samples, desc="eval", unit="img", ncols=90) if tqdm else samples

    with torch.no_grad():
        for img_id, sm in enumerate(it):
            pil = Image.open(sm.image_path).convert("RGB")
            recs = parse_bbgt_v3(sm.ann_path)
            if recs:
                gtb = np.array([[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in recs], dtype=np.float32)
                gtc = np.array([c for (c, *_r) in recs], dtype=np.int64)
            else:
                gtb = np.zeros((0, 4), dtype=np.float32); gtc = np.zeros((0,), dtype=np.int64)
            rgb_raw = np.asarray(pil, dtype=np.uint8)   # subset 판정용 (항상 raw 원본)
            if enhancer is not None:
                t = (TF.to_tensor(pil) * 2.0 - 1.0).unsqueeze(0).to(device)
                rgb = norm_tensor_to_uint8_rgb(enhance_fn(enhancer, t))  # native enhanced (검출 입력)
            else:
                rgb = rgb_raw
            res = _run_yolo(yolo, rgb, args.conf, device)
            pb, pc, pf = _extract_boxes(res)
            acc_all.add(img_id, gtb, gtc, pb, pc, pf)

            arr = rgb_raw.astype(np.float32) / 255.0   # ★ subset 분류는 raw 기준 (C와 동일 집합)
            luma = float((0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean())
            W, H = pil.size
            min_area = min([((x2 - x1) / W) * ((y2 - y1) / H) for (_c, x1, y1, x2, y2) in recs], default=1.0)
            if luma < LOWLIGHT_LUMA_THR:
                acc_low.add(img_id, gtb, gtc, pb, pc, pf); n_low += 1
            if min_area < SMALL_AREA_THR:
                acc_small.add(img_id, gtb, gtc, pb, pc, pf); n_small += 1

    r_all = acc_all.compute()
    print(f"  OVERALL  mAP@0.5={r_all['map50']:.4f}  mAP@.5:.95={r_all['map']:.4f}  "
          f"P={r_all['p50']:.4f}  R={r_all['r50']:.4f}")
    print("-" * 84)
    print("  [class-wise]  (#GT, AP@0.5, AP@.5:.95, P, R)")
    pc_ = r_all["per_class"]
    for cid in sorted(pc_, key=lambda c: -pc_[c]["ap50"]):
        d = pc_[cid]
        print(f"    {d['name']:<11} | #GT {d['n_gt']:>5} | AP50 {d['ap50']:.3f} | "
              f"AP {d['ap']:.3f} | P {d['p']:.3f} | R {d['r']:.3f}")
    print("-" * 84)
    r_low = acc_low.compute(); r_small = acc_small.compute()
    print(f"  [subset] 극저조도(luma<{LOWLIGHT_LUMA_THR}) n={n_low:>4}: "
          f"mAP@0.5={r_low['map50']:.4f}  mAP@.5:.95={r_low['map']:.4f}  R={r_low['r50']:.4f}")
    print(f"  [subset] small-object(minA<{SMALL_AREA_THR}) n={n_small:>4}: "
          f"mAP@0.5={r_small['map50']:.4f}  mAP@.5:.95={r_small['map']:.4f}  R={r_small['r50']:.4f}")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

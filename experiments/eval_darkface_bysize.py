"""[실험 2] DARK FACE 얼굴 크기별(tiny/small/large) mAP@0.5 — 향상 vs Original.

가설: LUNA2 의 검출 열세가 '작은 얼굴 파괴' 때문이라면, **큰 얼굴 구간에선 경쟁력**이
있어야 한다. GT 얼굴 높이로 3구간 층화해 구간별 AP@0.5 / recall 을 비교한다.

구간(얼굴 높이 px): tiny <16, small 16–32, large ≥32  (DARK FACE: 중앙값 11px,
<16px 67%, <32px 92% — 대부분 tiny).

COCO-style 면적범위 AP 근사: 구간 B 평가 시 (1) positive = B 안의 GT, (2) pred 를 conf
순으로 모든 GT 에 그리디 매칭(IoU≥th), (3) B 안 GT 와 매칭→TP, B 밖 GT 와 매칭→무시,
무매칭→FP. recall = TP / |B 안 GT|.

eval_darkface.py 의 향상기/검출기/파서를 그대로 재사용. 1 프로세스 = 1 method.

사용 예::
    python experiments/eval_darkface_bysize.py --method zerodce
    python experiments/eval_darkface_bysize.py --method luna2 --enh_max_side 0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image

from experiments.eval_detection import _box_iou_xyxy, _ap_all_point
from experiments.eval_detection_methods import _safe_enhance
from experiments.eval_darkface import (
    collect_darkface_samples, parse_darkface_label, run_face, make_enhancer, _PRETTY,
)
from src.utils.paths import load_paths

HRULE = "=" * 90
BUCKETS = [("tiny<16", 0.0, 16.0), ("small16-32", 16.0, 32.0), ("large>=32", 32.0, 1e9)]


def _bucket_of(h: float) -> int:
    for i, (_n, lo, hi) in enumerate(BUCKETS):
        if lo <= h < hi:
            return i
    return len(BUCKETS) - 1


class BySizeAccumulator:
    """이미지별 GT(박스+구간) / pred(박스+conf) 누적 → 구간별 AP@0.5, recall."""

    def __init__(self) -> None:
        self._gt: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}   # img_id -> (boxes, bucket_ids)
        self._pred: List[Tuple[int, float, float, float, float, float]] = []
        self._n_gt = [0] * len(BUCKETS)

    def add(self, img_id: int, gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_confs: np.ndarray) -> None:
        if gt_boxes.shape[0]:
            heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            buckets = np.array([_bucket_of(float(h)) for h in heights], dtype=np.int64)
            self._gt[img_id] = (gt_boxes.astype(np.float32), buckets)
            for b in buckets:
                self._n_gt[int(b)] += 1
        for (x1, y1, x2, y2), c in zip(pred_boxes, pred_confs):
            self._pred.append((img_id, float(c), float(x1), float(y1), float(x2), float(y2)))

    def _ap_recall(self, bucket: int, iou_th: float = 0.5) -> Tuple[float, float]:
        n_gt = self._n_gt[bucket]
        if n_gt == 0:
            return 0.0, 0.0
        preds = sorted(self._pred, key=lambda x: -x[1])
        matched: Dict[int, set] = {k: set() for k in self._gt}
        tps, fps = [], []
        for (img_id, _conf, x1, y1, x2, y2) in preds:
            g = self._gt.get(img_id)
            if g is None:
                tps.append(0.0); fps.append(1.0); continue
            boxes, buckets = g
            ious = _box_iou_xyxy(np.array([[x1, y1, x2, y2]], dtype=np.float32), boxes)[0]
            for j in matched[img_id]:
                ious[j] = -1.0
            best_j = int(np.argmax(ious))
            if float(ious[best_j]) >= iou_th:
                if int(buckets[best_j]) == bucket:
                    tps.append(1.0); fps.append(0.0); matched[img_id].add(best_j)
                else:
                    # 다른 구간 GT 와 매칭 → 무시(TP/FP 둘 다 아님), 단 그 GT 점유
                    matched[img_id].add(best_j)
                    continue
            else:
                tps.append(0.0); fps.append(1.0)
        if not tps:
            return 0.0, 0.0
        tp_cum = np.cumsum(tps); fp_cum = np.cumsum(fps)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _ap_all_point(precision, recall)
        return ap, float(recall[-1]) if len(recall) else 0.0

    def compute(self) -> List[dict]:
        out = []
        for i, (name, _lo, _hi) in enumerate(BUCKETS):
            ap, rec = self._ap_recall(i)
            out.append({"bucket": name, "n_gt": self._n_gt[i], "ap50": ap, "recall": rec})
        return out


def parse_args() -> argparse.Namespace:
    P = load_paths()
    p = argparse.ArgumentParser(description="[실험2] DARK FACE 얼굴 크기별 mAP")
    p.add_argument("--method", required=True, choices=["luna", "luna2", "sci", "zerodce"])
    p.add_argument("--sci_weight", default="easy", choices=["easy", "medium", "difficult"])
    p.add_argument("--darkface_root", type=str, default=str(P.exdark.parent / "DARKFACE"))
    p.add_argument("--face_weights", type=str, default=str(Path(str(P.yolov8n)).parent / "yolov8n-face.pt"))
    p.add_argument("--luna_ckpt", type=str, default=str(P.luna_loli30k))
    p.add_argument("--luna2_ckpt", type=str,
                   default=str(_LUNA2_ROOT / "runs" / "bilateral_phase1_l1only_guidefix" / "checkpoints" / "last.pth"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--enh_max_side", type=int, default=640)
    p.add_argument("--results_dir", type=str, default=str(P.runs / "eval_darkface_bysize"))
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> int:
    args = parse_args()
    device = args.device
    label = _PRETTY[args.method] if args.method != "sci" else f"SCI ({args.sci_weight})"
    from ultralytics import YOLO

    results_dir = Path(args.results_dir).resolve(); results_dir.mkdir(parents=True, exist_ok=True)
    print(HRULE)
    print(f" [실험2] DARK FACE 얼굴 크기별 mAP@0.5 — Original vs {label}")
    print(HRULE)
    enh = make_enhancer(args.method, args, device)
    face = YOLO(str(args.face_weights))
    samples = collect_darkface_samples(Path(args.darkface_root))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"  method={label}  cap={args.enh_max_side}  samples={len(samples)}  device={device}")

    acc_o, acc_e = BySizeAccumulator(), BySizeAccumulator()
    try:
        from tqdm import tqdm; it = tqdm(samples, desc=label, unit="img", ncols=100)
    except ImportError:
        it = samples
    with torch.no_grad():
        for img_id, sm in enumerate(it):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception:
                continue
            gt = parse_darkface_label(sm.label_path)
            ob, oc = run_face(face, np.array(pil, dtype=np.uint8), args.conf, device)
            er = _safe_enhance(enh, pil, pil.size, max_side=args.enh_max_side)
            eb, ec = run_face(face, er, args.conf, device)
            acc_o.add(img_id, gt, ob, oc)
            acc_e.add(img_id, gt, eb, ec)

    ro, re = acc_o.compute(), acc_e.compute()
    print("\n" + HRULE)
    print(f"  {'bucket':12} {'#GT':>7} | {'Orig AP50':>10} {'Enh AP50':>10} {'ΔAP50':>8} | {'Orig R':>7} {'Enh R':>7}")
    print("-" * 90)
    for o, e in zip(ro, re):
        print(f"  {o['bucket']:12} {o['n_gt']:>7} | {o['ap50']:>10.4f} {e['ap50']:>10.4f} "
              f"{e['ap50']-o['ap50']:>+8.4f} | {o['recall']:>7.4f} {e['recall']:>7.4f}")
    print(HRULE)

    tag = args.method if args.method != "sci" else f"sci_{args.sci_weight}"
    with (results_dir / f"bysize_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["bucket", "n_gt", "orig_ap50", "enh_ap50", "orig_recall", "enh_recall"])
        for o, e in zip(ro, re):
            w.writerow([o["bucket"], o["n_gt"], f"{o['ap50']:.6f}", f"{e['ap50']:.6f}",
                        f"{o['recall']:.6f}", f"{e['recall']:.6f}"])
    print(f"  Saved → {results_dir / f'bysize_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

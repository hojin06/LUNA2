"""ExDark Downstream 객체 검출 평가 — 원본 vs LUNA 향상 (mAP / P / R).  [이식본]

출처 (Provenance)
-----------------
``SmallSizePM_GAN_model/CODES/experiments/downstream_exdark.py`` 의 평가 로직을
**그대로 이식**. 차이점은 단 두 가지:
  1. generator/전처리 헬퍼를 ``src.models.luna_base`` 에서 import (원본 동일 코드).
  2. 데이터셋/가중치 경로를 하드코딩하지 않고 ``configs/paths.yaml`` 에서 주입.

따라서 동일 체크포인트(``paths.yaml::luna_original``) 로 실행하면 원본
``downstream_exdark.py`` 와 **동일한 mAP/P/R 수치**가 나와야 한다 (동일성 확인용).

평가 내용
---------
ExDark (Loh & Chan, CVIU 2019) GT bbox 기준으로,
  * 원본 저조도 이미지 + YOLOv8n(COCO) 검출 성능
  * LUNA 향상 이미지 + 같은 YOLOv8n 검출 성능
을 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall 로 비교한다.

사용 예
-------
.. code-block:: bash

    pip install ultralytics
    python experiments/eval_detection.py                       # paths.yaml 의 luna_original 사용
    python experiments/eval_detection.py --checkpoint <other.pth>
    python experiments/eval_detection.py --max_samples 50      # 디버그
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- LUNA2 루트를 sys.path 에 등록 (src 패키지 import) ---
_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

# Windows 콘솔(cp949) 한글 출력 안전
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.luna_base import (
    load_luna_generator,
    pil_to_norm_tensor,
    norm_tensor_to_uint8_rgb,
)
from src.utils.paths import load_paths

HRULE = "=" * 110
SUBRULE = "-" * 110


# ===========================================================================
# 1. ExDark 12 클래스 ↔ COCO 클래스 ID 매핑
# ===========================================================================
EXDARK_TO_COCO: Dict[str, int] = {
    "Bicycle":   1, "Boat": 8, "Bottle": 39, "Bus": 5, "Car": 2, "Cat": 15,
    "Chair": 56, "Cup": 41, "Dog": 16, "Motorbike": 3, "People": 0, "Table": 60,
}
EXDARK_CLASSES: Tuple[str, ...] = tuple(EXDARK_TO_COCO.keys())
TARGET_COCO_IDS: frozenset = frozenset(EXDARK_TO_COCO.values())
COCO_TO_EXDARK: Dict[int, str] = {v: k for k, v in EXDARK_TO_COCO.items()}

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ===========================================================================
# 2. ExDark annotation (bbGt v3) + split 파싱
# ===========================================================================
def parse_bbgt_v3(ann_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """annotation 1개 → (coco_id, x1, y1, x2, y2) 리스트."""
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not ann_path.is_file():
        return boxes
    try:
        text = ann_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return boxes
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_name = parts[0]
        canonical = None
        for c in EXDARK_CLASSES:
            if c.lower() == cls_name.lower():
                canonical = c
                break
        if canonical is None:
            continue
        try:
            l, t, w, h = (float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((EXDARK_TO_COCO[canonical], l, t, l + w, t + h))
    return boxes


def parse_imageclasslist(list_path: Path) -> Dict[str, int]:
    """``imageclasslist.txt`` → ``{image_filename: split_int}``."""
    out: Dict[str, int] = {}
    if not list_path.is_file():
        return out
    for raw in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 2:
            continue
        try:
            split = int(toks[-1])
        except ValueError:
            continue
        if split not in (1, 2, 3):
            continue
        out[toks[0]] = split
    return out


class ExDarkSample:
    """ExDark 샘플 1개 (이미지 + annotation 경로)."""

    __slots__ = ("image_path", "ann_path", "class_dir", "split")

    def __init__(self, image_path: Path, ann_path: Path,
                 class_dir: str, split: int) -> None:
        self.image_path = image_path
        self.ann_path = ann_path
        self.class_dir = class_dir
        self.split = split


def collect_exdark_samples(
    target_root: Path,
    splits: Optional[Tuple[int, ...]] = (3,),
) -> List[ExDarkSample]:
    """ExDark 폴더 트리 → ``ExDarkSample`` 리스트 (이미지↔annotation 둘 다 존재)."""
    images_root = target_root / "images"
    ann_root = target_root / "annotations"
    if not images_root.is_dir():
        raise FileNotFoundError(f"images 폴더 없음: {images_root}")
    if not ann_root.is_dir():
        raise FileNotFoundError(f"annotations 폴더 없음: {ann_root}")

    split_map = parse_imageclasslist(ann_root / "imageclasslist.txt")
    if not split_map and splits is not None:
        print("  [warn] imageclasslist.txt 누락/비어있음 — split 필터 무시")
        splits = None

    samples: List[ExDarkSample] = []
    for cls_name in EXDARK_CLASSES:
        img_dir = images_root / cls_name
        ann_dir = ann_root / cls_name
        if not img_dir.is_dir() or not ann_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
                continue
            ann_path = ann_dir / f"{img_path.name}.txt"
            if not ann_path.is_file():
                continue
            sp = split_map.get(img_path.name, 3)
            if splits is not None and sp not in splits:
                continue
            samples.append(ExDarkSample(
                image_path=img_path, ann_path=ann_path,
                class_dir=cls_name, split=sp,
            ))
    return samples


# ===========================================================================
# 3. LUNA 향상 — 원본 크기 복원
# ===========================================================================
def enhance_with_luna(G, pil_image: Image.Image, image_size: int, device: str) -> np.ndarray:
    """저조도 PIL → LUNA 향상 → uint8 RGB (원본 해상도) ndarray."""
    W, H = pil_image.size
    norm = pil_to_norm_tensor(pil_image, image_size).to(device)
    enh_norm = G(norm)
    enh_rgb_256 = norm_tensor_to_uint8_rgb(enh_norm)
    if (H, W) != enh_rgb_256.shape[:2]:
        enh_pil = Image.fromarray(enh_rgb_256).resize((W, H), resample=Image.BILINEAR)
        return np.array(enh_pil, dtype=np.uint8)
    return enh_rgb_256


# ===========================================================================
# 4. YOLOv8 추론 결과 → (boxes_xyxy, classes, confs)
# ===========================================================================
def _run_yolo(model, rgb: np.ndarray, conf: float, device: str):
    bgr = rgb[..., ::-1].copy()
    results = model.predict(bgr, conf=conf, verbose=False, device=device)
    return results[0]


def _extract_boxes(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32))
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    mask = np.array([c in TARGET_COCO_IDS for c in cls], dtype=bool)
    return xyxy[mask], cls[mask], conf[mask]


# ===========================================================================
# 5. mAP / Precision / Recall 계산기
# ===========================================================================
def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _ap_all_point(precisions: np.ndarray, recalls: np.ndarray) -> float:
    if precisions.size == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class DetectionAccumulator:
    """이미지별 검출/GT 누적 → COCO-style mAP / P / R."""

    IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)

    def __init__(self) -> None:
        self._preds: List[Tuple[int, int, float, float, float, float, float]] = []
        self._gts: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        self._img_ids: set = set()

    def add(
        self, img_id: int,
        gt_boxes: np.ndarray, gt_classes: np.ndarray,
        pred_boxes: np.ndarray, pred_classes: np.ndarray, pred_confs: np.ndarray,
    ) -> None:
        self._img_ids.add(img_id)
        for (x1, y1, x2, y2), c in zip(gt_boxes, gt_classes):
            if int(c) in TARGET_COCO_IDS:
                self._gts.setdefault(img_id, []).append(
                    (int(c), float(x1), float(y1), float(x2), float(y2)))
        for (x1, y1, x2, y2), c, conf in zip(pred_boxes, pred_classes, pred_confs):
            if int(c) in TARGET_COCO_IDS:
                self._preds.append(
                    (img_id, int(c), float(conf),
                     float(x1), float(y1), float(x2), float(y2)))

    def _ap_for_class(self, cls_id: int, iou_th: float) -> Tuple[float, int, int, int]:
        preds = [p for p in self._preds if p[1] == cls_id]
        preds.sort(key=lambda x: -x[2])
        gts_per_img: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for img_id, boxes in self._gts.items():
            xs = [(x1, y1, x2, y2) for (c, x1, y1, x2, y2) in boxes if c == cls_id]
            if xs:
                gts_per_img[img_id] = xs
        n_gt = sum(len(v) for v in gts_per_img.values())
        if n_gt == 0 or not preds:
            return 0.0, 0, len(preds), n_gt

        matched: Dict[int, set] = {k: set() for k in gts_per_img}
        tps = np.zeros(len(preds), dtype=np.float32)
        fps = np.zeros(len(preds), dtype=np.float32)
        for i, (img_id, _c, _conf, x1, y1, x2, y2) in enumerate(preds):
            gts = gts_per_img.get(img_id)
            if not gts:
                fps[i] = 1.0
                continue
            box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            gt_arr = np.array(gts, dtype=np.float32)
            ious = _box_iou_xyxy(box, gt_arr)[0]
            for j in matched[img_id]:
                ious[j] = -1.0
            best_j = int(np.argmax(ious))
            best_iou = float(ious[best_j])
            if best_iou >= iou_th:
                tps[i] = 1.0
                matched[img_id].add(best_j)
            else:
                fps[i] = 1.0

        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _ap_all_point(precision, recall)
        return ap, int(tps.sum()), int(fps.sum()), n_gt

    def compute(self) -> Dict[str, Any]:
        per_class: Dict[int, Dict[str, Any]] = {}
        ap50_list: List[float] = []
        ap_list: List[float] = []
        total_tp = total_fp = total_gt = 0

        for cls_id in sorted(TARGET_COCO_IDS):
            ap_per_iou: List[float] = []
            tp_at_50 = fp_at_50 = n_gt = 0
            for k, iou_th in enumerate(self.IOU_THRESHOLDS):
                ap, tp_c, fp_c, gt_c = self._ap_for_class(cls_id, float(iou_th))
                ap_per_iou.append(ap)
                if k == 0:
                    tp_at_50, fp_at_50, n_gt = tp_c, fp_c, gt_c
            if n_gt == 0:
                continue
            ap50 = ap_per_iou[0]
            ap = float(np.mean(ap_per_iou))
            p_c = tp_at_50 / max(tp_at_50 + fp_at_50, 1)
            r_c = tp_at_50 / max(n_gt, 1)
            per_class[cls_id] = {
                "name": COCO_TO_EXDARK.get(cls_id, str(cls_id)),
                "n_gt": int(n_gt), "n_pred": int(tp_at_50 + fp_at_50),
                "ap50": float(ap50), "ap": float(ap),
                "p": float(p_c), "r": float(r_c),
            }
            ap50_list.append(ap50)
            ap_list.append(ap)
            total_tp += tp_at_50
            total_fp += fp_at_50
            total_gt += n_gt

        return {
            "map50": float(np.mean(ap50_list)) if ap50_list else 0.0,
            "map": float(np.mean(ap_list)) if ap_list else 0.0,
            "p50": total_tp / max(total_tp + total_fp, 1),
            "r50": total_tp / max(total_gt, 1),
            "per_class": per_class,
            "n_images": len(self._img_ids),
            "n_preds": len(self._preds),
            "n_gts": sum(len(v) for v in self._gts.values()),
        }


# ===========================================================================
# 6. 콘솔 비교표
# ===========================================================================
def _fmt(x: float, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{d}f}"


def print_comparison_table(res_orig: Dict[str, Any], res_enh: Dict[str, Any]) -> None:
    print(HRULE)
    print(" ExDark Downstream Detection — Original (low) vs Enhanced (LUNA)  /  GT = ExDark annotation")
    print(HRULE)
    print(f"  {'Class':<14} | {'#GT':>5} | "
          f"{'Orig AP50':>10} {'Enh AP50':>9} {'ΔAP50':>8} | "
          f"{'Orig AP':>9} {'Enh AP':>8} {'ΔAP':>8} | "
          f"{'Orig P':>7} {'Enh P':>7} | {'Orig R':>7} {'Enh R':>7}")
    print(SUBRULE)
    classes = sorted(set(res_orig["per_class"]) | set(res_enh["per_class"]))
    for cid in classes:
        po = res_orig["per_class"].get(cid)
        pe = res_enh["per_class"].get(cid)
        name = (po or pe)["name"]
        n_gt = (po or pe)["n_gt"]
        ap50_o = po["ap50"] if po else 0.0
        ap50_e = pe["ap50"] if pe else 0.0
        ap_o = po["ap"] if po else 0.0
        ap_e = pe["ap"] if pe else 0.0
        p_o = po["p"] if po else 0.0
        p_e = pe["p"] if pe else 0.0
        r_o = po["r"] if po else 0.0
        r_e = pe["r"] if pe else 0.0
        print(f"  {name:<14} | {n_gt:>5} | "
              f"{_fmt(ap50_o, 3):>10} {_fmt(ap50_e, 3):>9} {ap50_e - ap50_o:+.3f} | "
              f"{_fmt(ap_o, 3):>9} {_fmt(ap_e, 3):>8} {ap_e - ap_o:+.3f} | "
              f"{_fmt(p_o, 3):>7} {_fmt(p_e, 3):>7} | "
              f"{_fmt(r_o, 3):>7} {_fmt(r_e, 3):>7}")
    print(SUBRULE)
    print(f"  {'OVERALL':<14} | {res_orig['n_gts']:>5} | "
          f"{_fmt(res_orig['map50'], 3):>10} {_fmt(res_enh['map50'], 3):>9} "
          f"{(res_enh['map50'] - res_orig['map50']):+.3f} | "
          f"{_fmt(res_orig['map'], 3):>9} {_fmt(res_enh['map'], 3):>8} "
          f"{(res_enh['map'] - res_orig['map']):+.3f} | "
          f"{_fmt(res_orig['p50'], 3):>7} {_fmt(res_enh['p50'], 3):>7} | "
          f"{_fmt(res_orig['r50'], 3):>7} {_fmt(res_enh['r50'], 3):>7}")
    print(HRULE)
    print(f"  ΔmAP@0.5      : {(res_enh['map50'] - res_orig['map50']):+.4f} "
          f"({_fmt(res_orig['map50'], 4)} → {_fmt(res_enh['map50'], 4)})")
    print(f"  ΔmAP@0.5:0.95 : {(res_enh['map'] - res_orig['map']):+.4f} "
          f"({_fmt(res_orig['map'], 4)} → {_fmt(res_enh['map'], 4)})")
    print(f"  ΔPrecision    : {(res_enh['p50'] - res_orig['p50']):+.4f}")
    print(f"  ΔRecall       : {(res_enh['r50'] - res_orig['r50']):+.4f}")
    print(HRULE)


# ===========================================================================
# 7. CSV 저장
# ===========================================================================
def save_results_csv(res_orig: Dict[str, Any], res_enh: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class", "n_gt", "orig_n_pred", "enh_n_pred",
        "orig_ap50", "enh_ap50", "delta_ap50",
        "orig_ap", "enh_ap", "delta_ap",
        "orig_p", "enh_p", "delta_p",
        "orig_r", "enh_r", "delta_r",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        classes = sorted(set(res_orig["per_class"]) | set(res_enh["per_class"]))
        for cid in classes:
            po = res_orig["per_class"].get(cid)
            pe = res_enh["per_class"].get(cid)
            name = (po or pe)["name"]
            n_gt = (po or pe)["n_gt"]

            def g(d: Optional[Dict[str, Any]], key: str) -> float:
                return float(d[key]) if d else 0.0

            w.writerow({
                "class": name, "n_gt": n_gt,
                "orig_n_pred": po["n_pred"] if po else 0,
                "enh_n_pred": pe["n_pred"] if pe else 0,
                "orig_ap50": f"{g(po, 'ap50'):.6f}", "enh_ap50": f"{g(pe, 'ap50'):.6f}",
                "delta_ap50": f"{g(pe, 'ap50') - g(po, 'ap50'):+.6f}",
                "orig_ap": f"{g(po, 'ap'):.6f}", "enh_ap": f"{g(pe, 'ap'):.6f}",
                "delta_ap": f"{g(pe, 'ap') - g(po, 'ap'):+.6f}",
                "orig_p": f"{g(po, 'p'):.6f}", "enh_p": f"{g(pe, 'p'):.6f}",
                "delta_p": f"{g(pe, 'p') - g(po, 'p'):+.6f}",
                "orig_r": f"{g(po, 'r'):.6f}", "enh_r": f"{g(pe, 'r'):.6f}",
                "delta_r": f"{g(pe, 'r') - g(po, 'r'):+.6f}",
            })
        w.writerow({
            "class": "OVERALL", "n_gt": res_orig["n_gts"],
            "orig_n_pred": res_orig["n_preds"], "enh_n_pred": res_enh["n_preds"],
            "orig_ap50": f"{res_orig['map50']:.6f}", "enh_ap50": f"{res_enh['map50']:.6f}",
            "delta_ap50": f"{res_enh['map50'] - res_orig['map50']:+.6f}",
            "orig_ap": f"{res_orig['map']:.6f}", "enh_ap": f"{res_enh['map']:.6f}",
            "delta_ap": f"{res_enh['map'] - res_orig['map']:+.6f}",
            "orig_p": f"{res_orig['p50']:.6f}", "enh_p": f"{res_enh['p50']:.6f}",
            "delta_p": f"{res_enh['p50'] - res_orig['p50']:+.6f}",
            "orig_r": f"{res_orig['r50']:.6f}", "enh_r": f"{res_enh['r50']:.6f}",
            "delta_r": f"{res_enh['r50'] - res_orig['r50']:+.6f}",
        })


# ===========================================================================
# 8. Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    P = load_paths()
    p = argparse.ArgumentParser(
        description="ExDark Downstream 검출 평가 — 원본 vs LUNA 향상 (mAP/P/R)")
    p.add_argument("--exdark_root", type=str, default=str(P.exdark),
                   help="ExDark 데이터셋 루트 (기본: paths.yaml::datasets.exdark)")
    p.add_argument("--checkpoint", type=str, default=str(P.luna_original),
                   help="LUNA generator 가중치 (기본: paths.yaml::weights.luna_original)")
    p.add_argument("--yolo_weights", type=str, default=str(P.yolov8n),
                   help="YOLOv8 가중치 (기본: paths.yaml::weights.yolov8n)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--image_size", type=int, default=256, help="LUNA 입력 해상도")
    p.add_argument("--results_dir", type=str, default=str(P.runs / "eval_detection"),
                   help="CSV 저장 디렉토리")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0=split 필터 전체, 양수=앞 N개만 (디버그)")
    p.add_argument("--all_splits", action="store_true",
                   help="split 필터 끄고 전체(1+2+3) 사용")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _try_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except ImportError:
        return None


def main() -> int:
    args = parse_args()
    device = args.device
    random.seed(args.seed)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치.  pip install ultralytics")
        return 1

    exdark_root = Path(args.exdark_root)
    ckpt_path = Path(args.checkpoint)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(HRULE)
    print(" ExDark Downstream Detection (YOLOv8n) — Original vs LUNA-Enhanced  [LUNA2]")
    print(HRULE)
    print(f"  exdark_root  : {exdark_root}")
    print(f"  checkpoint   : {ckpt_path}")
    print(f"  yolo_weights : {args.yolo_weights}")
    print(f"  conf_thresh  : {args.conf}")
    print(f"  image_size   : {args.image_size}  (LUNA 입력)")
    print(f"  device       : {device}")
    print(f"  results_dir  : {results_dir}")
    print(SUBRULE)

    if not ckpt_path.is_file():
        print(f"[error] LUNA 체크포인트가 없습니다: {ckpt_path}")
        return 1
    if not (exdark_root / "images").is_dir() or not (exdark_root / "annotations").is_dir():
        print(f"[error] ExDark 폴더 구조 누락: {exdark_root}")
        return 1

    G = load_luna_generator(ckpt_path, device=device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  LUNA params  : {n_params:,}  ({n_params / 1e3:.1f} K)")

    print(f"  YOLO loading : {args.yolo_weights} ...")
    yolo = YOLO(args.yolo_weights)
    print(f"  YOLO classes : {len(yolo.names)} (COCO pre-trained)")
    print(SUBRULE)

    splits = None if args.all_splits else (3,)
    samples = collect_exdark_samples(exdark_root, splits=splits)
    if not samples:
        print("[error] 샘플이 0 개입니다. imageclasslist.txt / annotation 확인 필요.")
        return 1
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    split_desc = "Test only (split=3)" if not args.all_splits else "ALL splits (1+2+3)"
    print(f"  samples      : {len(samples)}  [{split_desc}]")
    print(SUBRULE)

    acc_orig = DetectionAccumulator()
    acc_enh = DetectionAccumulator()

    tqdm = _try_tqdm()
    iterator = tqdm(samples, desc="ExDark", unit="img", ncols=100) if tqdm else samples

    with torch.no_grad():
        for img_id, sm in enumerate(iterator):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 이미지 로드 실패 {sm.image_path.name}: {e}")
                continue

            gt_records = parse_bbgt_v3(sm.ann_path)
            if gt_records:
                gt_boxes = np.array([[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in gt_records],
                                    dtype=np.float32)
                gt_classes = np.array([c for (c, *_r) in gt_records], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.zeros((0,), dtype=np.int64)

            orig_rgb = np.array(pil, dtype=np.uint8)
            enh_rgb = enhance_with_luna(G, pil, args.image_size, device)

            r_orig = _run_yolo(yolo, orig_rgb, args.conf, device)
            r_enh = _run_yolo(yolo, enh_rgb, args.conf, device)
            pb_o, pc_o, pf_o = _extract_boxes(r_orig)
            pb_e, pc_e, pf_e = _extract_boxes(r_enh)

            acc_orig.add(img_id, gt_boxes, gt_classes, pb_o, pc_o, pf_o)
            acc_enh.add(img_id, gt_boxes, gt_classes, pb_e, pc_e, pf_e)

    print("\n 평가 중 (mAP / Precision / Recall) ...")
    res_orig = acc_orig.compute()
    res_enh = acc_enh.compute()

    print()
    print_comparison_table(res_orig, res_enh)

    csv_path = results_dir / "detection_comparison.csv"
    save_results_csv(res_orig, res_enh, csv_path)
    print(f"  Saved CSV   → {csv_path}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

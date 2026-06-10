"""0단계 진단 — ExDark 검출 격차에서 256×256 해상도 페널티의 비중 분리.

연구 질문
---------
LUNA 는 저조도 이미지를 256×256 으로 다운샘플 → 향상 → 원본 해상도로 업샘플한
뒤 frozen YOLOv8n 에 넣는다. ExDark 검출 격차(기록상 Original mAP@0.5 ≈ 0.447 →
LUNA ≈ 0.346)에서, **향상(enhancement) 자체의 손실이 아니라 256 다운샘플로 인한
해상도 열화가 얼마를 차지하는지** 를 분리 측정한다.

방법 (enhancement 미사용)
-------------------------
원본 ``downstream_exdark.py`` 의 YOLO 입력 경로를 **그대로 재사용**하되,
"enhancement" 자리에 다음 3가지 *해상도 변환만* 끼워넣는다 (모델·가중치 없음):

  * ``native``      : 변환 없음. 원본 이미지를 그대로 YOLO 에 → Original 재현(≈0.447).
  * ``down256_up``  : 원본 → 256×256 다운샘플 → 원본 H×W 업샘플 → YOLO.
                      (LUNA 의 256 병목 + 업샘플 왕복과 동일한 해상도 손실)
  * ``down256``     : 원본 → 256×256 다운샘플 → YOLO (256 배열을 직접 입력).
                      예측 좌표를 원본 픽셀로 역스케일하여 GT 와 비교.

YOLO ``imgsz`` 는 원본 eval 과 동일하게 **기본값(640)으로 고정**(predict 에
imgsz 미전달) — 검출기 입력 크기가 아니라 *콘텐츠 열화*만 조건 간 차이나게 한다.

mAP 계산기(``DetectionAccumulator``), 클래스 매핑, GT 파싱, YOLO 호출은 모두
``eval_detection.py`` 이식본을 재사용하므로 ``native`` 는 원본과 동일 수치가 나온다.

출력
----
* 콘솔: 조건별 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall 표.
* CSV : ``runs/diagnostic/resolution_diagnostic.csv``.
* 분해(decomposition): 해상도 페널티 = native − down256(_up), 그리고 기록상
  LUNA mAP 와의 격차에서 해상도가 차지하는 비율(%).

사용 예
-------
.. code-block:: bash

    python experiments/00_resolution_diagnostic.py                 # 전체 7363장
    python experiments/00_resolution_diagnostic.py --interp bicubic
    python experiments/00_resolution_diagnostic.py --max_samples 100   # 스모크
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- LUNA2 루트를 sys.path 에 등록 (src / experiments import) ---
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
from PIL import Image

# 검출 평가 경로를 그대로 재사용 — native 가 원본과 동일 수치를 내도록 보장.
from experiments.eval_detection import (
    DetectionAccumulator,
    _extract_boxes,
    _fmt,
    _run_yolo,
    _try_tqdm,
    collect_exdark_samples,
    parse_bbgt_v3,
)
from src.utils.paths import load_paths

HRULE = "=" * 96
SUBRULE = "-" * 96

CONDITIONS = ("native", "down256_up", "down256")
_LOW_RES = 256  # LUNA 의 내부 처리 해상도 (병목)

# 보간 방식 → PIL resample 상수. 'area' 는 PIL BOX (다운스케일 시 영역 평균).
_INTERP_TO_PIL = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "area": Image.BOX,
    "nearest": Image.NEAREST,
}


# ===========================================================================
# 해상도 변환 (enhancement 자리에 끼워넣는 유일한 연산)
# ===========================================================================
def degrade(pil: Image.Image, condition: str, resample: int) -> Tuple[np.ndarray, float]:
    """원본 PIL → (조건별 변환 RGB uint8 ndarray, 예측 역스케일 계수).

    Returns
    -------
    rgb : np.ndarray
        YOLO 에 넣을 RGB uint8 (H, W, 3).
    box_scale : float
        예측 box 를 원본 픽셀로 되돌릴 때 곱할 비율. native / down256_up 은
        이미 원본 해상도이므로 1.0; down256 은 256 좌표 → 원본 이므로 ≠ 1.0
        (가로/세로 비율이 달라 (sx, sy) 가 다를 수 있어 호출부에서 처리).
    """
    W, H = pil.size  # (width, height)

    if condition == "native":
        return np.array(pil, dtype=np.uint8), 1.0

    if condition == "down256_up":
        # 원본 → 256 → 원본 (W,H) 왕복. YOLO 는 원본 해상도 배열을 받는다.
        small = pil.resize((_LOW_RES, _LOW_RES), resample=resample)
        back = small.resize((W, H), resample=resample)
        return np.array(back, dtype=np.uint8), 1.0

    if condition == "down256":
        # 원본 → 256×256 배열을 그대로 YOLO 에 입력. 예측은 256 좌표로 나온다.
        small = pil.resize((_LOW_RES, _LOW_RES), resample=resample)
        return np.array(small, dtype=np.uint8), 1.0  # 역스케일은 호출부에서 (sx, sy)

    raise ValueError(f"알 수 없는 condition: {condition}")


def _rescale_boxes_to_original(
    boxes_xyxy: np.ndarray, fed_wh: Tuple[int, int], orig_wh: Tuple[int, int],
) -> np.ndarray:
    """fed-image 좌표의 예측 box → 원본 픽셀 좌표로 역스케일.

    GT 가 원본 픽셀 좌표이므로 256 배열을 직접 넣은 down256 의 예측을 원본
    스케일로 되돌려 동일 좌표계에서 IoU 가 계산되도록 한다.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    fw, fh = fed_wh
    ow, oh = orig_wh
    sx = ow / float(fw)
    sy = oh / float(fh)
    out = boxes_xyxy.copy()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


# ===========================================================================
# 단일 조건 평가
# ===========================================================================
def evaluate_condition(
    samples,
    yolo,
    condition: str,
    resample: int,
    conf: float,
    device: str,
    progress_desc: str,
) -> Dict[str, Any]:
    """한 해상도 조건에 대해 ExDark 전체를 돌려 mAP/P/R 누적·계산."""
    acc = DetectionAccumulator()
    tqdm = _try_tqdm()
    iterator = (tqdm(samples, desc=progress_desc, unit="img", ncols=100)
                if tqdm else samples)

    with torch.no_grad():
        for img_id, sm in enumerate(iterator):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 이미지 로드 실패 {sm.image_path.name}: {e}")
                continue
            W, H = pil.size

            # GT (원본 픽셀 좌표) — 모든 조건에서 동일 (열화 미적용)
            gt_records = parse_bbgt_v3(sm.ann_path)
            if gt_records:
                gt_boxes = np.array(
                    [[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in gt_records],
                    dtype=np.float32)
                gt_classes = np.array([c for (c, *_r) in gt_records], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.zeros((0,), dtype=np.int64)

            # enhancement 자리 = 해상도 변환만
            rgb, _ = degrade(pil, condition, resample)
            fed_h, fed_w = rgb.shape[:2]

            result = _run_yolo(yolo, rgb, conf, device)  # imgsz 미전달 → 기본 640
            pb, pc, pf = _extract_boxes(result)

            # down256 은 256 좌표 예측 → 원본 픽셀로 역스케일
            if condition == "down256":
                pb = _rescale_boxes_to_original(pb, (fed_w, fed_h), (W, H))

            acc.add(img_id, gt_boxes, gt_classes, pb, pc, pf)

    return acc.compute()


# ===========================================================================
# 출력 — 표 / CSV / 분해
# ===========================================================================
def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    print(HRULE)
    print(" Resolution Diagnostic — ExDark (enhancement 미사용, 해상도 열화만)")
    print(HRULE)
    print(f"  {'Condition':<12} | {'mAP@0.5':>9} | {'mAP@.5:.95':>11} | "
          f"{'Precision':>10} | {'Recall':>8} | {'#img':>6} | {'#GT':>6}")
    print(SUBRULE)
    for cond in CONDITIONS:
        r = results[cond]
        print(f"  {cond:<12} | {_fmt(r['map50'], 4):>9} | {_fmt(r['map'], 4):>11} | "
              f"{_fmt(r['p50'], 4):>10} | {_fmt(r['r50'], 4):>8} | "
              f"{r['n_images']:>6} | {r['n_gts']:>6}")
    print(HRULE)


def save_csv(results: Dict[str, Dict[str, Any]], out_path: Path, interp: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "interp", "map50", "map",
                    "precision", "recall", "n_images", "n_preds", "n_gts"])
        for cond in CONDITIONS:
            r = results[cond]
            w.writerow([
                cond, interp,
                f"{r['map50']:.6f}", f"{r['map']:.6f}",
                f"{r['p50']:.6f}", f"{r['r50']:.6f}",
                r["n_images"], r["n_preds"], r["n_gts"],
            ])


def print_decomposition(
    results: Dict[str, Dict[str, Any]],
    luna_map50: float,
    luna_map: float,
) -> None:
    """해상도 페널티 분해 + 기록상 LUNA 격차 대비 비율."""
    nat50 = results["native"]["map50"]
    up50 = results["down256_up"]["map50"]
    d50 = results["down256"]["map50"]
    nat = results["native"]["map"]
    up = results["down256_up"]["map"]
    d = results["down256"]["map"]

    def _share(penalty: float, total_gap: float) -> str:
        if abs(total_gap) < 1e-9:
            return "—  (격차≈0)"
        return f"{penalty / total_gap * 100:+.1f}%"

    gap50 = nat50 - luna_map50  # 측정 native vs 기록 LUNA
    gap = nat - luna_map

    print(" 분해 (Decomposition) — mAP@0.5 기준")
    print(SUBRULE)
    print(f"  native (Original 재현)        : {nat50:.4f}")
    print(f"  down256_up (LUNA 해상도 아날로그): {up50:.4f}")
    print(f"  down256 (256 직접 입력)        : {d50:.4f}")
    print(f"  기록상 LUNA mAP@0.5            : {luna_map50:.4f}")
    print(SUBRULE)
    res_pen_up = nat50 - up50
    res_pen_256 = nat50 - d50
    print(f"  해상도 페널티 (native − down256_up) : {res_pen_up:+.4f}")
    print(f"  해상도 페널티 (native − down256)    : {res_pen_256:+.4f}")
    print(f"  전체 격차     (native − LUNA)       : {gap50:+.4f}")
    print(SUBRULE)
    print(f"  → 격차 중 해상도(왕복) 기여 비율 : {_share(res_pen_up, gap50)}")
    print(f"  → 격차 중 해상도(256직접) 기여   : {_share(res_pen_256, gap50)}")
    print(f"  → 나머지(향상/기타) 기여         : "
          f"{_share(gap50 - res_pen_up, gap50)}  (왕복 기준)")
    print(HRULE)
    # mAP@0.5:0.95 도 참고로
    print(" 참고 — mAP@0.5:0.95 기준")
    print(SUBRULE)
    print(f"  native={nat:.4f}  down256_up={up:.4f}  down256={d:.4f}  "
          f"LUNA(기록)={luna_map:.4f}")
    print(f"  해상도 페널티(왕복)={nat - up:+.4f}  전체 격차={gap:+.4f}  "
          f"비율={_share(nat - up, gap)}")
    print(HRULE)


# ===========================================================================
# Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    P = load_paths()
    p = argparse.ArgumentParser(
        description="0단계 해상도 진단 — 256 다운샘플 페널티 분리 (enhancement 미사용)")
    p.add_argument("--exdark_root", type=str, default=str(P.exdark),
                   help="ExDark 루트 (기본: paths.yaml::datasets.exdark)")
    p.add_argument("--yolo_weights", type=str, default=str(P.yolov8n),
                   help="YOLOv8 가중치 (기본: paths.yaml::weights.yolov8n)")
    p.add_argument("--interp", type=str, default="bilinear",
                   choices=list(_INTERP_TO_PIL),
                   help="해상도 변환 보간 방식 (기본 bilinear, LUNA 와 동일)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold (원본 eval 과 동일)")
    p.add_argument("--results_dir", type=str, default=str(P.runs / "diagnostic"),
                   help="CSV 저장 디렉토리")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0=split 결과 전체, 양수=앞 N개만 (스모크 테스트)")
    p.add_argument("--all_splits", action="store_true",
                   help="split 필터 끄고 전체(1+2+3) 사용. (imageclasslist.txt 없으면 자동 전체)")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu")
    # 기록상 LUNA 검출 성능 (격차 분해용). 기본값은 과제에 제시된 수치.
    p.add_argument("--luna_map50", type=float, default=0.346,
                   help="기록상 LUNA mAP@0.5 (격차 분해 기준)")
    p.add_argument("--luna_map", type=float, default=0.0,
                   help="기록상 LUNA mAP@0.5:0.95 (0이면 참고 비율 생략 가능)")
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def run_diagnostic(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """3개 해상도 조건에 대해 ExDark 평가를 수행하고 결과 dict 반환."""
    device = args.device
    exdark_root = Path(args.exdark_root)
    resample = _INTERP_TO_PIL[args.interp]

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[error] ultralytics 미설치.  pip install ultralytics")

    if not (exdark_root / "images").is_dir() or not (exdark_root / "annotations").is_dir():
        raise SystemExit(f"[error] ExDark 폴더 구조 누락: {exdark_root}")

    print(HRULE)
    print(" 0단계 해상도 진단 (Resolution Diagnostic) — enhancement 미사용")
    print(HRULE)
    print(f"  exdark_root  : {exdark_root}")
    print(f"  yolo_weights : {args.yolo_weights}")
    print(f"  interp       : {args.interp}  (down/up 양방향)")
    print(f"  conf_thresh  : {args.conf}   |   YOLO imgsz : 기본값(640, 미전달·고정)")
    print(f"  device       : {device}")
    print(f"  conditions   : {', '.join(CONDITIONS)}")
    print(SUBRULE)

    yolo = YOLO(args.yolo_weights)
    print(f"  YOLO classes : {len(yolo.names)} (COCO pre-trained)")

    splits = None if args.all_splits else (3,)
    samples = collect_exdark_samples(exdark_root, splits=splits)
    if not samples:
        raise SystemExit("[error] 샘플이 0 개입니다.")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"  samples      : {len(samples)}")
    print(SUBRULE)

    results: Dict[str, Dict[str, Any]] = {}
    for cond in CONDITIONS:
        results[cond] = evaluate_condition(
            samples, yolo, cond, resample, args.conf, device,
            progress_desc=cond,
        )
    return results


def main() -> int:
    args = parse_args()
    results = run_diagnostic(args)

    print()
    print_summary_table(results)

    out_path = Path(args.results_dir).resolve() / "resolution_diagnostic.csv"
    save_csv(results, out_path, args.interp)
    print(f"  Saved CSV → {out_path}")
    print(HRULE)

    print()
    print_decomposition(results, luna_map50=args.luna_map50, luna_map=args.luna_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""DARK FACE 극저조도 얼굴검출 평가 — Original vs 향상(LUNA / LUNA2 / SCI / Zero-DCE).

[목적]
ExDark 실험과 동일한 "향상 → 고정 검출기 → mAP" 프로토콜을 **극저조도 얼굴검출**로 옮겨,
극단적 저조도에서 LUNA/LUNA2가 SCI/Zero-DCE 대비 경쟁력이 있는지 검증한다.

[ExDark 와의 차이]
- 데이터셋: DARK FACE train/val 6,000장 (UG2+ 극저조도). label/<stem>.txt = 첫 줄 얼굴 수 N,
  이후 N줄 "x1 y1 x2 y2"(절대 픽셀).
- 검출기: YOLOv8n-COCO 대신 **YOLOv8n-face**(WIDER FACE 학습). 단일 클래스(face).
- 평가: 단일 클래스 mAP@0.5 / mAP@0.5:0.95 / P / R (AP 계산은 eval_detection 의 검증된
  _box_iou_xyxy / _ap_all_point 재사용).

[프로토콜 통일] 향상 입력 긴 변 640 캡(--enh_max_side) → native 복원 → 검출. ExDark 의
신규 4종 측정과 동일. 1 프로세스 = 1 method (모듈 충돌 회피).

사용 예::
    python experiments/eval_darkface.py --method zerodce --max_samples 30   # 디버그(소량)
    python experiments/eval_darkface.py --method luna
    python experiments/eval_darkface.py --method luna2
    python experiments/eval_darkface.py --method sci --sci_weight easy
    python experiments/eval_darkface.py --method zerodce
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")      # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")      # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image

from experiments.eval_detection import _box_iou_xyxy, _ap_all_point
from experiments.eval_detection_methods import (
    build_zerodce, build_sci, _safe_enhance,
)
from src.utils.paths import load_paths

_PROJECT_ROOT = _LUNA2_ROOT.parent
HRULE = "=" * 100
SUBRULE = "-" * 100
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ===========================================================================
# 1. DARK FACE 샘플 수집 + 라벨 파싱
# ===========================================================================
class DarkFaceSample:
    __slots__ = ("image_path", "label_path")

    def __init__(self, image_path: Path, label_path: Path) -> None:
        self.image_path = image_path
        self.label_path = label_path


def collect_darkface_samples(root: Path) -> List[DarkFaceSample]:
    img_dir = root / "image"
    lbl_dir = root / "label"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"DARK FACE 구조 누락(image/ label/): {root}")
    out: List[DarkFaceSample] = []
    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in _IMG_EXTS:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.is_file():
            out.append(DarkFaceSample(img, lbl))
    return out


def parse_darkface_label(path: Path) -> np.ndarray:
    """label .txt → (N,4) xyxy float. 첫 줄 = 얼굴 수(헤더), 이후 'x1 y1 x2 y2'."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return np.zeros((0, 4), dtype=np.float32)
    boxes: List[Tuple[float, float, float, float]] = []
    start = 0
    # 첫 줄이 정수 1개면 헤더(얼굴 수)로 보고 스킵
    if lines:
        toks0 = lines[0].split()
        if len(toks0) == 1 and toks0[0].lstrip("-").isdigit():
            start = 1
    for raw in lines[start:]:
        p = raw.split()
        if len(p) < 4:
            continue
        try:
            x1, y1, x2, y2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        except ValueError:
            continue
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


# ===========================================================================
# 2. 단일 클래스(face) mAP / P / R 누적기
# ===========================================================================
class FaceAccumulator:
    IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)

    def __init__(self) -> None:
        # preds: (img_id, conf, x1,y1,x2,y2)
        self._preds: List[Tuple[int, float, float, float, float, float]] = []
        self._gts: Dict[int, np.ndarray] = {}
        self._img_ids: set = set()
        self._n_gt = 0

    def add(self, img_id: int, gt_boxes: np.ndarray,
            pred_boxes: np.ndarray, pred_confs: np.ndarray) -> None:
        self._img_ids.add(img_id)
        if gt_boxes.shape[0]:
            self._gts[img_id] = gt_boxes.astype(np.float32)
            self._n_gt += gt_boxes.shape[0]
        for (x1, y1, x2, y2), c in zip(pred_boxes, pred_confs):
            self._preds.append((img_id, float(c), float(x1), float(y1), float(x2), float(y2)))

    def _pr_at_iou(self, iou_th: float) -> Tuple[float, np.ndarray, np.ndarray]:
        preds = sorted(self._preds, key=lambda x: -x[1])
        matched: Dict[int, set] = {k: set() for k in self._gts}
        tps = np.zeros(len(preds), dtype=np.float32)
        fps = np.zeros(len(preds), dtype=np.float32)
        for i, (img_id, _conf, x1, y1, x2, y2) in enumerate(preds):
            gts = self._gts.get(img_id)
            if gts is None or gts.shape[0] == 0:
                fps[i] = 1.0
                continue
            ious = _box_iou_xyxy(np.array([[x1, y1, x2, y2]], dtype=np.float32), gts)[0]
            for j in matched[img_id]:
                ious[j] = -1.0
            best_j = int(np.argmax(ious))
            if float(ious[best_j]) >= iou_th:
                tps[i] = 1.0
                matched[img_id].add(best_j)
            else:
                fps[i] = 1.0
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recall = tp_cum / max(self._n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _ap_all_point(precision, recall)
        return ap, tps, fps

    def compute(self) -> Dict[str, float]:
        if self._n_gt == 0 or not self._preds:
            return {"map50": 0.0, "map": 0.0, "p50": 0.0, "r50": 0.0,
                    "n_images": len(self._img_ids), "n_preds": len(self._preds), "n_gts": self._n_gt}
        aps = []
        for k, th in enumerate(self.IOU_THRESHOLDS):
            ap, tps, fps = self._pr_at_iou(float(th))
            aps.append(ap)
            if k == 0:
                tp50, fp50 = float(tps.sum()), float(fps.sum())
        return {
            "map50": float(aps[0]),
            "map": float(np.mean(aps)),
            "p50": tp50 / max(tp50 + fp50, 1.0),
            "r50": tp50 / max(self._n_gt, 1),
            "n_images": len(self._img_ids),
            "n_preds": len(self._preds),
            "n_gts": self._n_gt,
        }


# ===========================================================================
# 3. 향상 방법 빌더 — LUNA / LUNA2 (+ SCI / Zero-DCE 는 eval_detection_methods 재사용)
# ===========================================================================
def build_luna(device: str, ckpt: str, image_size: int = 256) -> Callable[[Image.Image], np.ndarray]:
    from src.models.luna_base import (
        load_luna_generator, pil_to_norm_tensor, norm_tensor_to_uint8_rgb)
    G = load_luna_generator(Path(ckpt), device=device)

    @torch.no_grad()
    def enh(pil: Image.Image) -> np.ndarray:
        W, H = pil.size
        norm = pil_to_norm_tensor(pil, image_size).to(device)
        rgb256 = norm_tensor_to_uint8_rgb(G(norm))
        if rgb256.shape[:2] != (H, W):
            return np.asarray(Image.fromarray(rgb256).resize((W, H), Image.BILINEAR), dtype=np.uint8)
        return rgb256

    return enh


def build_luna2(device: str, ckpt: str) -> Callable[[Image.Image], np.ndarray]:
    import torchvision.transforms.functional as TF
    from src.models.bilateral_grid import build_from_config
    from src.models.luna_base import norm_tensor_to_uint8_rgb
    from src.utils.inference import enhance as luna2_enhance
    est = torch.load(ckpt, map_location=device, weights_only=False)
    enhancer = build_from_config(est["config"]).to(device).eval()
    enhancer.load_state_dict(est["model"], strict=False)

    @torch.no_grad()
    def enh(pil: Image.Image) -> np.ndarray:
        t = (TF.to_tensor(pil.convert("RGB")) * 2.0 - 1.0).unsqueeze(0).to(device)
        return norm_tensor_to_uint8_rgb(luna2_enhance(enhancer, t))

    return enh


def make_enhancer(method: str, args, device: str) -> Callable[[Image.Image], np.ndarray]:
    if method == "zerodce":
        return build_zerodce(device)
    if method == "sci":
        return build_sci(device, args.sci_weight)
    if method == "luna":
        return build_luna(device, args.luna_ckpt)
    if method == "luna2":
        return build_luna2(device, args.luna2_ckpt)
    raise ValueError(method)


_PRETTY = {"zerodce": "Zero-DCE", "sci": "SCI", "luna": "LUNA", "luna2": "LUNA2"}


# ===========================================================================
# 4. 얼굴 검출기 추론 → (boxes_xyxy, confs)  [단일 클래스]
# ===========================================================================
def run_face(model, rgb: np.ndarray, conf: float, device: str):
    bgr = rgb[..., ::-1].copy()
    res = model.predict(bgr, conf=conf, verbose=False, device=device)[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    xyxy = b.xyxy.detach().cpu().numpy().astype(np.float32)
    cf = b.conf.detach().cpu().numpy().astype(np.float32)
    return xyxy, cf


# ===========================================================================
# 5. Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    P = load_paths()
    default_df = P.exdark.parent / "DARKFACE"
    default_face = Path(str(P.yolov8n)).parent / "yolov8n-face.pt"
    default_luna2 = _LUNA2_ROOT / "runs" / "bilateral_phase1_l1only_guidefix" / "checkpoints" / "last.pth"
    p = argparse.ArgumentParser(description="DARK FACE 극저조도 얼굴검출 평가 (mAP/P/R)")
    p.add_argument("--method", required=True, choices=["luna", "luna2", "sci", "zerodce"])
    p.add_argument("--sci_weight", default="easy", choices=["easy", "medium", "difficult"])
    p.add_argument("--darkface_root", type=str, default=str(default_df))
    p.add_argument("--face_weights", type=str, default=str(default_face))
    p.add_argument("--luna_ckpt", type=str, default=str(P.luna_loli30k))
    p.add_argument("--luna2_ckpt", type=str, default=str(default_luna2))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--enh_max_side", type=int, default=640)
    p.add_argument("--results_dir", type=str, default=str(P.runs / "eval_darkface"))
    p.add_argument("--max_samples", type=int, default=0, help="0=전체, 양수=앞 N개(디버그)")
    p.add_argument("--image_list", type=str, default="",
                   help="평가할 이미지 stem 목록 txt(한 줄 1개). 지정 시 그 부분집합만 평가(누수 차단용).")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> int:
    args = parse_args()
    device = args.device
    label = _PRETTY[args.method]
    if args.method == "sci":
        label = f"SCI ({args.sci_weight})"
    if args.method == "luna":
        label = f"LUNA ({Path(args.luna_ckpt).stem})"

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치.")
        return 1

    df_root = Path(args.darkface_root)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(HRULE)
    print(f" DARK FACE Low-light Face Detection (YOLOv8n-face) — Original vs {label}")
    print(HRULE)
    print(f"  darkface_root: {df_root}")
    print(f"  face_weights : {args.face_weights}")
    print(f"  method       : {args.method}  ({label})")
    print(f"  conf / cap   : {args.conf} / {args.enh_max_side}")
    print(f"  device       : {device}")
    print(SUBRULE)

    face_w = Path(args.face_weights)
    if not face_w.is_file():
        print(f"[error] 얼굴 검출기 가중치 없음: {face_w}")
        return 1

    print(f"  향상기 로딩  : {label} ...")
    enh = make_enhancer(args.method, args, device)
    print(f"  검출기 로딩  : {face_w.name} ...")
    face = YOLO(str(face_w))
    print(f"  face classes : {face.names}")
    print(SUBRULE)

    samples = collect_darkface_samples(df_root)
    if not samples:
        print("[error] 샘플 0개.")
        return 1
    if args.image_list:
        wl = {s.strip() for s in Path(args.image_list).read_text(encoding="utf-8").splitlines() if s.strip()}
        samples = [s for s in samples if s.image_path.stem in wl]
        print(f"  image_list   : {args.image_list} → {len(samples)}장으로 제한")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"  samples      : {len(samples)}")
    print(SUBRULE)

    acc_orig, acc_enh = FaceAccumulator(), FaceAccumulator()
    try:
        from tqdm import tqdm
        it = tqdm(samples, desc=label, unit="img", ncols=100)
    except ImportError:
        it = samples

    with torch.no_grad():
        for img_id, sm in enumerate(it):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 로드 실패 {sm.image_path.name}: {e}")
                continue
            gt = parse_darkface_label(sm.label_path)
            orig_rgb = np.array(pil, dtype=np.uint8)
            enh_rgb = _safe_enhance(enh, pil, pil.size, max_side=args.enh_max_side)

            ob, oc = run_face(face, orig_rgb, args.conf, device)
            eb, ec = run_face(face, enh_rgb, args.conf, device)
            acc_orig.add(img_id, gt, ob, oc)
            acc_enh.add(img_id, gt, eb, ec)

    print("\n 평가 중 (mAP / Precision / Recall) ...")
    ro, re = acc_orig.compute(), acc_enh.compute()

    print()
    print(HRULE)
    print(f"  {'':14} {'mAP@0.5':>9} {'mAP@.5:.95':>11} {'Precision':>10} {'Recall':>9}")
    print(SUBRULE)
    print(f"  {'Original':14} {ro['map50']:9.4f} {ro['map']:11.4f} {ro['p50']:10.4f} {ro['r50']:9.4f}")
    print(f"  {label:14} {re['map50']:9.4f} {re['map']:11.4f} {re['p50']:10.4f} {re['r50']:9.4f}")
    print(SUBRULE)
    print(f"  Δ mAP@0.5 = {re['map50'] - ro['map50']:+.4f}   "
          f"(GT faces={ro['n_gts']}, images={ro['n_images']})")
    print(HRULE)

    tag = args.method if args.method != "sci" else f"sci_{args.sci_weight}"
    import csv
    csv_path = results_dir / f"darkface_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_images", "n_gts", "map50", "map", "precision", "recall"])
        w.writerow(["Original", ro["n_images"], ro["n_gts"],
                    f"{ro['map50']:.6f}", f"{ro['map']:.6f}", f"{ro['p50']:.6f}", f"{ro['r50']:.6f}"])
        w.writerow([label, re["n_images"], re["n_gts"],
                    f"{re['map50']:.6f}", f"{re['map']:.6f}", f"{re['p50']:.6f}", f"{re['r50']:.6f}"])
    print(f"  Saved CSV   → {csv_path}")
    print(f"  [SUMMARY] {label}: mAP@0.5={re['map50']:.4f} P={re['p50']:.4f} R={re['r50']:.4f}  "
          f"(Original mAP50={ro['map50']:.4f})")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

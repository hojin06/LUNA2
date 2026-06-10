"""ExDark Downstream 검출 평가 — 외부 향상 방법 (EnlightenGAN / FUnIE-GAN / Zero-DCE / SCI).

[목적]
``eval_detection.py`` 의 mAP/P/R 계산기(``DetectionAccumulator``)·ExDark GT 파서·YOLO
래퍼를 **그대로 재사용**하고, "향상(enhance)" 단계만 외부 4개 방법으로 교체한다.
따라서 LUNA 차트와 **완전히 동일한 평가 프로토콜**(동일 GT, 동일 YOLOv8n COCO,
동일 IoU/AP 계산, 동일 클래스 매핑)로 4개 방법의 mAP@0.5 / mAP@0.5:0.95 / P / R 을 얻는다.

[공정성]
- 모든 방법: 향상 결과를 **native 해상도로 복원**한 뒤 동일 YOLOv8n 입력.
- Zero-DCE / SCI / EnlightenGAN : native 해상도에서 향상.
- FUnIE-GAN : 구조상 256×256 고정 입력 → 향상 후 native 복원 (LUNA 의 256→native 와 동일 정책).
- 검증: ``--method`` 무관하게 Original(저조도) baseline 은 항상 동일하게 재현되어야 함
  (LUNA 차트의 Original mAP@0.5 ≈ 0.447 / P ≈ 0.645 / R ≈ 0.548 와 일치하면 배선 정상).

[모듈 충돌 회피] Zero-DCE 와 SCI 는 둘 다 ``model.py`` 를 가지므로 한 프로세스에 같이
import 하면 충돌한다. 따라서 **1 프로세스 = 1 method** 로 ``--method`` 인자를 받아 따로 실행.

사용 예::
    python experiments/eval_detection_methods.py --method zerodce
    python experiments/eval_detection_methods.py --method sci --sci_weight medium
    python experiments/eval_detection_methods.py --method enlighten
    python experiments/eval_detection_methods.py --method funie
    python experiments/eval_detection_methods.py --method zerodce --max_samples 30   # 디버그
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

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

from experiments.eval_detection import (
    collect_exdark_samples,
    parse_bbgt_v3,
    DetectionAccumulator,
    _run_yolo,
    _extract_boxes,
    print_comparison_table,
    save_results_csv,
)
from src.utils.paths import load_paths

# 외부 repo 들의 루트 (LUNA_paperWORKS 아래).
_PROJECT_ROOT = _LUNA2_ROOT.parent
HRULE = "=" * 110
SUBRULE = "-" * 110


# ===========================================================================
# 향상 방법 빌더 — 각각 (pil_rgb) -> uint8 HxWx3 RGB (native 해상도) 콜백 반환
# ===========================================================================
def build_zerodce(device: str) -> Callable[[Image.Image], np.ndarray]:
    p = _PROJECT_ROOT / "Zero-DCE" / "Zero-DCE_code"
    sys.path.insert(0, str(p))
    import model as zdce  # noqa: E402  (Zero-DCE/Zero-DCE_code/model.py)
    net = zdce.enhance_net_nopool().to(device)
    sd = torch.load(str(p / "snapshots" / "Epoch99.pth"), map_location=device)
    net.load_state_dict(sd)
    net.eval()

    @torch.no_grad()
    def enh(pil: Image.Image) -> np.ndarray:
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        _, out, _ = net(x)
        return (out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    return enh


def build_sci(device: str, which: str) -> Callable[[Image.Image], np.ndarray]:
    p = _PROJECT_ROOT / "SCI" / "CVPR"
    sys.path.insert(0, str(p))
    from model import Finetunemodel  # noqa: E402  (SCI/CVPR/model.py)
    import torchvision.transforms.functional as TF
    w = p / "weights" / f"{which}.pt"
    if not w.is_file():
        raise FileNotFoundError(f"SCI weight 없음: {w}")
    net = Finetunemodel(str(w)).to(device).eval()

    @torch.no_grad()
    def enh(pil: Image.Image) -> np.ndarray:
        x = TF.to_tensor(pil.convert("RGB")).unsqueeze(0).to(device)
        _i, r = net(x)  # r = 향상 결과 (test.py 가 저장하는 텐서), [0,1]
        return (r.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    return enh


def build_enlighten(device: str) -> Callable[[Image.Image], np.ndarray]:
    p = _PROJECT_ROOT / "EnlightenGAN"
    sys.path.insert(0, str(p))
    import enlighten_single as eg  # noqa: E402  (검증된 standalone 추론)
    net = eg.load_generator(
        str(p / "checkpoints" / "enlightening" / "200_net_G_A.pth"), device)

    @torch.no_grad()
    def enh(pil: Image.Image) -> np.ndarray:
        out_pil = eg.enhance(net, pil, device)  # PIL, native 해상도로 복원됨
        return np.asarray(out_pil.convert("RGB"), dtype=np.uint8)

    return enh


def build_funie(device: str) -> Callable[[Image.Image], np.ndarray]:
    p = _PROJECT_ROOT / "FUnIE-GAN" / "PyTorch"
    sys.path.insert(0, str(p))
    from nets.funiegan import GeneratorFunieGAN  # noqa: E402
    import torchvision.transforms as T
    model = GeneratorFunieGAN()
    model.load_state_dict(torch.load(str(p / "models" / "funie_generator.pth"),
                                     map_location=device))
    model.to(device).eval()
    tf = T.Compose([
        T.Resize((256, 256), Image.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    to_pil = T.ToPILImage()

    @torch.no_grad()
    def enh(pil: Image.Image) -> np.ndarray:
        pil = pil.convert("RGB")
        W, H = pil.size
        x = tf(pil).unsqueeze(0).to(device)
        out = model(x)  # [-1,1], 256x256
        out = ((out.squeeze(0).cpu() + 1.0) / 2.0).clamp(0, 1)
        op = to_pil(out).resize((W, H), Image.BICUBIC)
        return np.asarray(op, dtype=np.uint8)

    return enh


_BUILDERS = {
    "zerodce": lambda dev, args: build_zerodce(dev),
    "sci": lambda dev, args: build_sci(dev, args.sci_weight),
    "enlighten": lambda dev, args: build_enlighten(dev),
    "funie": lambda dev, args: build_funie(dev),
}
_PRETTY = {
    "zerodce": "Zero-DCE", "sci": "SCI",
    "enlighten": "EnlightenGAN", "funie": "FUnIE-GAN",
}


def _safe_enhance(enh: Callable, pil: Image.Image, native_wh, max_side: int = 0) -> np.ndarray:
    """향상 입력을 max_side 로 캡(>0 일 때) → 향상 → native 해상도로 복원.

    YOLOv8 은 어차피 입력을 ~640px 로 줄여 검출하므로, 향상을 대형 native 해상도에서
    수행하는 것은 낭비이며 8GB GPU 에서 OOM 을 유발한다. 긴 변을 max_side 로 캡하면
    LUNA 차트의 256→native 복원 프로토콜과 동일한 방식이 되어 타당성도 유지된다.
    GT bbox 가 native 좌표이므로 향상 결과는 항상 native(W,H)로 복원해 YOLO 에 넣는다.
    """
    W, H = native_wh
    inp = pil
    if max_side and max(W, H) > max_side:
        s = max_side / float(max(W, H))
        inp = pil.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BICUBIC)
    try:
        rgb = enh(inp)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        iw, ih = inp.size
        s = 512.0 / max(iw, ih)
        small = inp.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.BICUBIC)
        rgb = enh(small)
    if rgb.shape[:2] != (H, W):
        rgb = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR), dtype=np.uint8)
    return rgb


# ===========================================================================
# Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    P = load_paths()
    p = argparse.ArgumentParser(description="ExDark 검출 평가 — 외부 향상 방법 (mAP/P/R)")
    p.add_argument("--method", required=True, choices=list(_BUILDERS.keys()))
    p.add_argument("--sci_weight", default="medium", choices=["easy", "medium", "difficult"],
                   help="SCI 가중치 변종 (method=sci 일 때만)")
    p.add_argument("--exdark_root", type=str, default=str(P.exdark))
    p.add_argument("--yolo_weights", type=str, default=str(P.yolov8n))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--enh_max_side", type=int, default=640,
                   help="향상 입력 긴 변 캡 (0=native). YOLO 가 ~640 로 줄이므로 기본 640.")
    p.add_argument("--results_dir", type=str, default=str(P.runs / "eval_detection_methods"))
    p.add_argument("--max_samples", type=int, default=0, help="0=전체, 양수=앞 N개 (디버그)")
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

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치.  pip install ultralytics")
        return 1

    exdark_root = Path(args.exdark_root)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(HRULE)
    print(f" ExDark Downstream Detection (YOLOv8n) — Original vs {label}-Enhanced  [LUNA2]")
    print(HRULE)
    print(f"  method       : {args.method}  ({label})")
    print(f"  exdark_root  : {exdark_root}")
    print(f"  yolo_weights : {args.yolo_weights}")
    print(f"  conf_thresh  : {args.conf}")
    print(f"  enh_max_side : {args.enh_max_side}  (0=native; 향상 입력 긴 변 캡)")
    print(f"  device       : {device}")
    print(f"  results_dir  : {results_dir}")
    print(SUBRULE)

    if not (exdark_root / "images").is_dir() or not (exdark_root / "annotations").is_dir():
        print(f"[error] ExDark 폴더 구조 누락: {exdark_root}")
        return 1

    print(f"  향상기 로딩  : {label} ...")
    enh = _BUILDERS[args.method](device, args)

    print(f"  YOLO loading : {args.yolo_weights} ...")
    yolo = YOLO(args.yolo_weights)
    print(f"  YOLO classes : {len(yolo.names)} (COCO pre-trained)")
    print(SUBRULE)

    samples = collect_exdark_samples(exdark_root, splits=(3,))
    if not samples:
        print("[error] 샘플 0개.")
        return 1
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"  samples      : {len(samples)}")
    print(SUBRULE)

    acc_orig = DetectionAccumulator()
    acc_enh = DetectionAccumulator()

    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc=label, unit="img", ncols=100)
    except ImportError:
        iterator = samples

    with torch.no_grad():
        for img_id, sm in enumerate(iterator):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 로드 실패 {sm.image_path.name}: {e}")
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
            enh_rgb = _safe_enhance(enh, pil, pil.size, max_side=args.enh_max_side)

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

    tag = args.method if args.method != "sci" else f"sci_{args.sci_weight}"
    csv_path = results_dir / f"detection_{tag}.csv"
    save_results_csv(res_orig, res_enh, csv_path)
    print(f"  Saved CSV   → {csv_path}")

    # 한 줄 요약 (표 채우기용)
    print(SUBRULE)
    print(f"  [SUMMARY] {label}: "
          f"mAP@0.5={res_enh['map50']:.4f}  mAP@0.5:0.95={res_enh['map']:.4f}  "
          f"P={res_enh['p50']:.4f}  R={res_enh['r50']:.4f}   "
          f"(Original: mAP50={res_orig['map50']:.4f} P={res_orig['p50']:.4f} R={res_orig['r50']:.4f})")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

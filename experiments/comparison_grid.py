"""저조도 복원 전/후 시각 비교 그리드 생성 (DARK FACE / ExDark).

모듈 충돌(Zero-DCE·SCI 의 model.py) 회피를 위해 **2단계**:
  1) --mode enhance --method X : 선택 샘플을 방법 X 로 향상해 PNG 저장 (1프로세스=1방법)
                                 + 원본도 original/ 에 저장.
  2) --mode grid               : original/ 과 각 method/ 폴더의 PNG 를 읽어 라벨 격자 합성
                                 (순수 PIL, 모델 로드 없음) → comparison_grid.png

샘플 선택: --images "1,57,123" 로 stem 지정, 없으면 DARK FACE 에서 **가장 어두운 --n 장** 자동 선택.

사용 예::
    # 1단계 (각 방법, GPU)
    python experiments/comparison_grid.py --mode enhance --method luna    --n 5
    python experiments/comparison_grid.py --mode enhance --method luna2   --n 5
    python experiments/comparison_grid.py --mode enhance --method zerodce --n 5
    python experiments/comparison_grid.py --mode enhance --method sci --sci_weight easy --n 5
    # 2단계 (합성, CPU)
    python experiments/comparison_grid.py --mode grid --methods luna,luna2,sci_easy,zerodce
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.paths import load_paths

_PRETTY = {"original": "Original (low-light)", "luna": "LUNA", "luna2": "LUNA2",
           "zerodce": "Zero-DCE", "sci_easy": "SCI (easy)",
           "sci_medium": "SCI (medium)", "sci_difficult": "SCI (difficult)"}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------------
# 샘플 선택
# ---------------------------------------------------------------------------
def _darkface_image_dir(df_root: Path) -> Path:
    return df_root / "image"


def select_samples(df_root: Path, images_arg: str, n: int) -> List[Path]:
    img_dir = _darkface_image_dir(df_root)
    all_imgs = sorted(p for p in img_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if images_arg:
        want = {s.strip() for s in images_arg.split(",") if s.strip()}
        picked = [p for p in all_imgs if p.stem in want]
        if picked:
            return picked
        print(f"  [warn] --images 매칭 0개 → 가장 어두운 {n}장 자동 선택")
    # 가장 어두운 n장
    lums = []
    for p in all_imgs:
        try:
            im = Image.open(p).convert("L"); im.thumbnail((48, 48))
            lums.append((float(np.asarray(im, dtype=np.float32).mean()), p))
        except Exception:
            pass
    lums.sort(key=lambda x: x[0])
    return [p for _l, p in lums[:n]]


# ---------------------------------------------------------------------------
# 1단계: enhance
# ---------------------------------------------------------------------------
def run_enhance(args) -> int:
    import torch
    from experiments.eval_detection_methods import build_zerodce, build_sci, _safe_enhance
    from experiments.eval_darkface import build_luna, build_luna2

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "original").mkdir(parents=True, exist_ok=True)
    method_dir = out / (args.method if args.method != "sci" else f"sci_{args.sci_weight}")
    method_dir.mkdir(parents=True, exist_ok=True)

    samples = select_samples(Path(args.darkface_root), args.images, args.n)
    print(f"  samples ({len(samples)}): {[p.stem for p in samples]}")

    if args.method == "zerodce":
        enh = build_zerodce(device)
    elif args.method == "sci":
        enh = build_sci(device, args.sci_weight)
    elif args.method == "luna":
        enh = build_luna(device, args.luna_ckpt)
    elif args.method == "luna2":
        enh = build_luna2(device, args.luna2_ckpt)
    else:
        raise ValueError(args.method)

    for p in samples:
        pil = Image.open(p).convert("RGB")
        # 원본 저장(한 번이면 충분하나 매 방법마다 동일하게 갱신 — 무해)
        pil.save(out / "original" / f"{p.stem}.png")
        rgb = _safe_enhance(enh, pil, pil.size, max_side=args.enh_max_side)
        Image.fromarray(rgb).save(method_dir / f"{p.stem}.png")
    print(f"  saved → {method_dir}")
    return 0


# ---------------------------------------------------------------------------
# 2단계: grid (순수 PIL)
# ---------------------------------------------------------------------------
def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def run_grid(args) -> int:
    out = Path(args.out)
    cols = ["original"] + [m.strip() for m in args.methods.split(",") if m.strip()]
    return _compose_grid(out, cols, out / "comparison_grid.png", args.cell_h)


def _compose_grid(base: Path, cols, out_path: Path, cell_h: int) -> int:
    """base/<col>/<stem>.png 들을 라벨 격자(montage)로 합성."""
    orig_dir = base / "original"
    if not orig_dir.is_dir():
        print(f"[error] {orig_dir} 없음 — 먼저 enhance/detect 실행.")
        return 1
    stems = sorted(p.stem for p in orig_dir.glob("*.png"))
    if not stems:
        print("[error] original/ 비어있음.")
        return 1
    pad, hdr = 6, 34
    font = _font(20)

    def load_cell(col: str, stem: str):
        fp = base / col / f"{stem}.png"
        if not fp.is_file():
            img = Image.new("RGB", (int(cell_h * 1.4), cell_h), (40, 40, 40))
            ImageDraw.Draw(img).text((8, cell_h // 2), "(missing)", fill=(200, 80, 80), font=font)
            return img
        im = Image.open(fp).convert("RGB")
        w = max(1, int(im.width * cell_h / im.height))
        return im.resize((w, cell_h), Image.BILINEAR)

    col_w, cell_cache = {}, {}
    for c in cols:
        mx = 1
        for s in stems:
            im = load_cell(c, s); cell_cache[(c, s)] = im; mx = max(mx, im.width)
        col_w[c] = mx
    total_w = sum(col_w[c] + pad for c in cols) + pad
    total_h = hdr + len(stems) * (cell_h + pad) + pad
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = pad
    for c in cols:
        draw.text((x + 4, 8), _PRETTY.get(c, c), fill=(0, 0, 0), font=font)
        x += col_w[c] + pad
    y = hdr
    for s in stems:
        x = pad
        for c in cols:
            canvas.paste(cell_cache[(c, s)], (x, y)); x += col_w[c] + pad
        y += cell_h + pad
    canvas.save(out_path)
    print(f"  grid: {len(stems)} rows × {len(cols)} cols → {out_path}")
    return 0


def _dense_crop(gt: np.ndarray, W: int, H: int, frac: float):
    """GT 얼굴이 가장 밀집한 frac×frac 창을 찾아 (x1,y1,x2,y2) 크롭 박스 반환."""
    if frac <= 0 or frac >= 1 or gt.shape[0] == 0:
        return (0, 0, W, H)
    cw, ch = max(1, int(W * frac)), max(1, int(H * frac))
    cx = (gt[:, 0] + gt[:, 2]) * 0.5
    cy = (gt[:, 1] + gt[:, 3]) * 0.5
    xs = sorted({0, max(0, W - cw)} | {int(min(max(0, c - cw / 2), W - cw)) for c in cx})
    ys = sorted({0, max(0, H - ch)} | {int(min(max(0, c - ch / 2), H - ch)) for c in cy})
    best = (0, 0, -1)
    for x0 in xs:
        for y0 in ys:
            cnt = int(np.sum((cx >= x0) & (cx <= x0 + cw) & (cy >= y0) & (cy <= y0 + ch)))
            if cnt > best[2]:
                best = (x0, y0, cnt)
    x0, y0, _ = best
    return (x0, y0, x0 + cw, y0 + ch)


def run_detect_grid(args) -> int:
    """기존 enhance 산출물(original/+각 method/)에 얼굴검출기를 돌려
    예측 박스(초록)+GT(빨강)를 그린 뒤 격자 합성. → comparison_grid_det.png
    --crop_frac>0 이면 GT 밀집 영역만 크롭해 확대(모든 열 동일 영역)."""
    import torch
    from ultralytics import YOLO
    from experiments.eval_darkface import parse_darkface_label

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); det = out / "det"
    cols = ["original"] + [m.strip() for m in args.methods.split(",") if m.strip()]
    face = YOLO(str(args.face_weights))
    df_lbl = Path(args.darkface_root) / "label"
    cf = args.crop_frac
    bw = 2 if cf <= 0 else 3  # 크롭 확대 시 박스 두껍게
    print(f"  detector: {Path(args.face_weights).name}  conf={args.conf}  crop_frac={cf}  device={device}")

    crop_cache: dict = {}
    n = 0
    for col in cols:
        srcdir = out / col
        if not srcdir.is_dir():
            print(f"  [skip] {srcdir} 없음"); continue
        (det / col).mkdir(parents=True, exist_ok=True)
        for fp in sorted(srcdir.glob("*.png")):
            im = Image.open(fp).convert("RGB")
            W, H = im.size
            gt = parse_darkface_label(df_lbl / f"{fp.stem}.txt")
            if fp.stem not in crop_cache:
                crop_cache[fp.stem] = _dense_crop(gt, W, H, cf)
            res = face.predict(np.asarray(im)[..., ::-1].copy(), conf=args.conf,
                               verbose=False, device=device)[0]
            d = ImageDraw.Draw(im)
            for (x1, y1, x2, y2) in gt:
                d.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=bw - 1 or 1)  # GT 빨강
            b = res.boxes; npred = 0
            if b is not None and len(b):
                for xy in b.xyxy.detach().cpu().numpy():
                    d.rectangle([float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])],
                                outline=(40, 230, 40), width=bw)                          # pred 초록
                    npred += 1
            cb = crop_cache[fp.stem]
            if cb != (0, 0, W, H):
                im = im.crop(cb)
            ImageDraw.Draw(im).text((4, 4), f"det:{npred}", fill=(40, 230, 40), font=_font(18))
            im.save(det / col / fp.name)
            n += 1
    print(f"  annotated {n} images (초록=검출, 빨강=GT) → {det}")
    return _compose_grid(det, cols, out / "comparison_grid_det.png", args.cell_h)


def main() -> int:
    P = load_paths()
    default_df = P.exdark.parent / "DARKFACE"
    default_luna2 = _LUNA2_ROOT / "runs" / "bilateral_phase1_l1only_guidefix" / "checkpoints" / "last.pth"
    p = argparse.ArgumentParser(description="저조도 복원 전/후 시각 비교 그리드")
    p.add_argument("--mode", required=True, choices=["enhance", "grid", "detect_grid"])
    p.add_argument("--face_weights", type=str,
                   default=str(Path(str(P.yolov8n)).parent / "yolov8n-face.pt"),
                   help="detect_grid: 얼굴 검출기 가중치")
    p.add_argument("--conf", type=float, default=0.25, help="detect_grid: 검출 conf 임계값")
    p.add_argument("--crop_frac", type=float, default=0.0,
                   help="detect_grid: GT 밀집 영역을 이미지의 frac×frac 만큼만 크롭해 확대(0=전체, 예 0.35)")
    p.add_argument("--method", choices=["luna", "luna2", "sci", "zerodce"])
    p.add_argument("--methods", default="luna,luna2,sci_easy,zerodce",
                   help="grid 모드: 열 순서 (original 은 자동 맨 앞)")
    p.add_argument("--sci_weight", default="easy", choices=["easy", "medium", "difficult"])
    p.add_argument("--images", default="", help='샘플 stem 콤마구분 (예: "1,57,123"). 없으면 가장 어두운 N장')
    p.add_argument("--n", type=int, default=5, help="자동 선택 시 샘플 수")
    p.add_argument("--darkface_root", type=str, default=str(default_df))
    p.add_argument("--luna_ckpt", type=str, default=str(P.luna_loli30k))
    p.add_argument("--luna2_ckpt", type=str, default=str(default_luna2))
    p.add_argument("--enh_max_side", type=int, default=640)
    p.add_argument("--cell_h", type=int, default=200, help="grid 셀 높이(px)")
    p.add_argument("--out", type=str, default=str(P.runs / "comparison_darkface"))
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.mode == "enhance":
        if not args.method:
            print("[error] --mode enhance 에는 --method 필요.")
            return 1
        return run_enhance(args)
    if args.mode == "detect_grid":
        return run_detect_grid(args)
    return run_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())

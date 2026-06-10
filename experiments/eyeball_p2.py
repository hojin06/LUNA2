"""[육안] P2 체크포인트로 ExDark test 의 '어둡고 작은 객체' 이미지 native enhance.

선별 기준 (어둡고 + 작은 객체):
  · darkness = 1 - mean(luma01)               (밝기 낮을수록 ↑)
  · small    = 가장 작은 GT box 의 normalized 면적 (작을수록 작은 객체)
  · 점수 = darkness  (단, GT>0 & 최소 box 면적 < small_thr 인 후보 중에서)
배치 없이 1장씩 native 해상도로 enhance(출력 가드) → 원본|enhanced concat PNG.

읽기/추론만 (본학습 없음). paths.yaml.
"""
from __future__ import annotations

import argparse
import csv as csvmod
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

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from experiments.eval_detection import collect_exdark_samples, parse_bbgt_v3
from src.models.bilateral_grid import build_from_config
from src.models.luna_base import norm_tensor_to_uint8_rgb
from src.utils.inference import enhance, is_nonfinite
from src.utils.paths import load_paths

HR = "=" * 78


def load_split_test(split_csv: Path):
    want = 3  # test
    m = {}
    for r in csvmod.DictReader(open(split_csv, encoding="utf-8")):
        m[(r["class_dir"], r["image_name"])] = int(r["split"])
    return m, want


def main() -> int:
    p = argparse.ArgumentParser(description="P2 육안 enhance (어둡고 작은 객체 test 8장)")
    P = load_paths()
    p.add_argument("--checkpoint", type=str,
                   default=str(P.runs / "p2_det_l0020" / "checkpoints" / "last.pth"))
    p.add_argument("--split_csv", type=str,
                   default=str(_LUNA2_ROOT / "configs" / "exdark_split_provisional.csv"))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--small_thr", type=float, default=0.03,
                   help="작은 객체: 최소 GT box normalized 면적 < thr")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent.parent / "eyeball"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(HR)
    print(" [육안] P2 enhance — ExDark test, 어둡고 작은 객체 우선")
    print(HR)
    print(f"  checkpoint : {ckpt}")
    if not ckpt.is_file():
        print(f"[error] 체크포인트 없음: {ckpt}")
        return 1

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_from_config(state["config"]).to(device).eval()
    model.load_state_dict(state["model"], strict=False)
    print(f"  epoch={state.get('epoch')} lambda_det={state.get('lambda_det')}")

    smap, want = load_split_test(Path(args.split_csv))
    samples = [s for s in collect_exdark_samples(P.exdark, splits=None)
               if smap.get((s.class_dir, s.image_path.name)) == want]
    print(f"  test 샘플  : {len(samples)}")

    # 후보 점수화: GT>0 & 최소 box 면적 < small_thr → darkness 내림차순
    scored = []
    for sm in samples:
        recs = parse_bbgt_v3(sm.ann_path)
        if not recs:
            continue
        try:
            pil = Image.open(sm.image_path).convert("RGB")
        except Exception:
            continue
        W, H = pil.size
        areas = [((x2 - x1) / W) * ((y2 - y1) / H) for (_c, x1, y1, x2, y2) in recs]
        min_area = min(areas)
        # darkness (luma [0,1] 평균)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean()
        darkness = 1.0 - float(luma)
        scored.append((sm, darkness, min_area, float(luma), len(recs)))

    # 작은 객체 포함 후보 우선, 부족하면 완화
    cand = [s for s in scored if s[2] < args.small_thr]
    if len(cand) < args.n:
        cand = sorted(scored, key=lambda t: t[2])[: max(args.n * 4, 32)]
    cand.sort(key=lambda t: -t[1])              # darkness 내림차순
    chosen = cand[: args.n]

    print(f"  후보(작은객체) {len([s for s in scored if s[2] < args.small_thr])}, "
          f"선택 {len(chosen)}장 (어두움 우선)")
    print("-" * 78)

    nonfinite_cnt = 0
    for i, (sm, dark, min_area, luma, nrec) in enumerate(chosen):
        pil = Image.open(sm.image_path).convert("RGB")
        W, H = pil.size
        t = (TF.to_tensor(pil) * 2.0 - 1.0).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(t)
            if is_nonfinite(raw):
                nonfinite_cnt += 1
            out = enhance(model, t)             # fp32 + nan_to_num + clamp
        enh_rgb = norm_tensor_to_uint8_rgb(out)  # (H,W,3) native
        orig_rgb = np.asarray(pil, dtype=np.uint8)
        sep = np.full((H, 4, 3), 255, dtype=np.uint8)
        concat = np.concatenate([orig_rgb, sep, enh_rgb], axis=1)
        name = f"{i:02d}_{sm.class_dir}_{sm.image_path.stem}_luma{luma:.2f}_minA{min_area:.4f}.png"
        Image.fromarray(concat).save(out_dir / name)
        print(f"  [{i}] {sm.class_dir}/{sm.image_path.name} ({W}x{H}) "
              f"luma={luma:.3f} min_box_area={min_area:.4f} n_gt={nrec}")

    print("-" * 78)
    print(f"  저장 → {out_dir}  ({len(chosen)} PNG, 좌=원본 | 우=enhanced)")
    print(f"  비유한(NaN/Inf) 원시출력 이미지 : {nonfinite_cnt}/{len(chosen)}")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

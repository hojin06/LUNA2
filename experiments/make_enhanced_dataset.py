"""[P3-D 준비] enhancer 로 ExDark train+val 을 enhance 해 YOLO 학습용 데이터셋 구성.

enhancer(예: tv01) 로 train+val 이미지를 **native 해상도**로 enhance(출력 가드) → uint8
PNG 저장. labels 는 C 와 동일 GT(COCO id cx cy w h, enhance 무관). test 는 학습에
넣지 않는다(누수 차단; 평가는 eval 시 on-the-fly enhance).

출력: <out_root>/{images,labels}/<class>/*, train.txt, val.txt, exdark.yaml.
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

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from experiments.eval_detection import collect_exdark_samples, parse_bbgt_v3
from src.models.bilateral_grid import build_from_config
from src.models.luna_base import norm_tensor_to_uint8_rgb
from src.utils.inference import enhance, is_nonfinite
from src.utils.paths import load_paths

HR = "=" * 78


def main() -> int:
    P = load_paths()
    p = argparse.ArgumentParser(description="enhancer 로 ExDark train+val enhance → YOLO 데이터셋")
    p.add_argument("--enhancer_ckpt", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--split_csv", type=str, default=str(_LUNA2_ROOT / "configs" / "exdark_split_provisional.csv"))
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    print(HR)
    print(" [P3-D 준비] enhance train+val → YOLO 데이터셋 (test 제외, 누수 차단)")
    print(HR)
    est = torch.load(args.enhancer_ckpt, map_location=device, weights_only=False)
    enhancer = build_from_config(est["config"]).to(device).eval()
    enhancer.load_state_dict(est["model"], strict=False)
    print(f"  enhancer : {args.enhancer_ckpt}")

    smap = {}
    for r in csvmod.DictReader(open(args.split_csv, encoding="utf-8")):
        smap[(r["class_dir"], r["image_name"])] = int(r["split"])

    samples = collect_exdark_samples(P.exdark, splits=None)
    train_list, val_list = [], []
    nonfinite = 0
    tqdm = None
    try:
        from tqdm import tqdm as _t; tqdm = _t
    except ImportError:
        pass
    it = tqdm(samples, ncols=90) if tqdm else samples

    with torch.no_grad():
        for sm in it:
            sp = smap.get((sm.class_dir, sm.image_path.name))
            split_name = {1: "train", 2: "val"}.get(sp)
            if split_name is None:           # test → 제외
                continue
            pil = Image.open(sm.image_path).convert("RGB")
            W, H = pil.size
            t = (TF.to_tensor(pil) * 2.0 - 1.0).unsqueeze(0).to(device)
            if is_nonfinite(enhancer(t)):
                nonfinite += 1
            rgb = norm_tensor_to_uint8_rgb(enhance(enhancer, t))   # native enhanced uint8
            out_img = out_root / "images" / sm.class_dir / f"{sm.image_path.stem}.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out_img)
            # labels (C 와 동일 GT)
            recs = parse_bbgt_v3(sm.ann_path)
            out_lbl = out_root / "labels" / sm.class_dir / f"{sm.image_path.stem}.txt"
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for (cid, x1, y1, x2, y2) in recs:
                cx = ((x1 + x2) * 0.5) / W; cy = ((y1 + y2) * 0.5) / H
                bw = (x2 - x1) / W; bh = (y2 - y1) / H
                if bw > 0 and bh > 0:
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_lbl.write_text("\n".join(lines), encoding="utf-8")
            (train_list if split_name == "train" else val_list).append(str(out_img))

    (out_root / "train.txt").write_text("\n".join(train_list), encoding="utf-8")
    (out_root / "val.txt").write_text("\n".join(val_list), encoding="utf-8")
    from ultralytics import YOLO
    names = YOLO(str(P.yolov8n)).names
    yaml_path = out_root / "exdark.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out_root), "train": "train.txt", "val": "val.txt",
                        "names": {int(k): v for k, v in names.items()}},
                       f, allow_unicode=True, sort_keys=False)
    print(f"  train {len(train_list)}  val {len(val_list)}  비유한 enhance {nonfinite}")
    print(f"  data yaml → {yaml_path}")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

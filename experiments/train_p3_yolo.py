"""[P3 조건 C] YOLOv8n(COCO init) ExDark train fine-tune (enhancer 없음, raw 저조도).

P3 바(baseline) 확정용. budget = configs/p3_joint_budget.yaml (C/D/E 공유).

데이터 준비 (원본 ExDark 불변 유지)
-----------------------------------
* images: LUNA2/runs/p3_yolo_data/images → ExDark/images **junction** (복사 0).
* labels: LUNA2/runs/p3_yolo_data/labels/<class>/<name>.txt 생성 (COCO id cx cy w h).
* train.txt/val.txt: train/val split 이미지 경로 (test 제외 → 누수 차단).
* exdark.yaml: ultralytics data 설정 (names = COCO 80 유지).

GT = 기존 parse_bbgt_v3 + EXDARK_TO_COCO (COCO id). 평가는 eval_detection_native.py
``--mode original --yolo_weights <best.pt>`` (enhancer 없음) 로 test 에서 수행.

사용:
  python experiments/train_p3_yolo.py --config configs/p3_joint_budget.yaml
  (학습 후) python experiments/eval_detection_native.py --mode original --split test \
            --yolo_weights runs/p3_yolo_C/train/weights/best.pt --label "C [test]"
"""
from __future__ import annotations

import argparse
import csv as csvmod
import subprocess
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

import yaml
from PIL import Image

from experiments.eval_detection import collect_exdark_samples, parse_bbgt_v3
from src.utils.paths import load_paths

HR = "=" * 78


def make_junction(link: Path, target: Path):
    """Windows junction (권한 불필요, 복사 0). 이미 있으면 그대로."""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                   check=True, capture_output=True)


def prepare_data(P, budget: dict, data_root: Path) -> Path:
    """labels 생성 + images junction + train/val txt + exdark.yaml → yaml 경로."""
    dc = budget["data"]
    split_csv = _LUNA2_ROOT / dc["split_csv"]
    smap = {}
    for r in csvmod.DictReader(open(split_csv, encoding="utf-8")):
        smap[(r["class_dir"], r["image_name"])] = int(r["split"])
    want = {"train": 1, "val": 2, "test": 3}

    img_dir = data_root / "images"
    lbl_dir = data_root / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    make_junction(img_dir, P.exdark / "images")   # images → ExDark/images

    samples = collect_exdark_samples(P.exdark, splits=None)
    train_list, val_list = [], []
    n_box = {"train": 0, "val": 0}
    for sm in samples:
        sp = smap.get((sm.class_dir, sm.image_path.name))
        split_name = {1: "train", 2: "val"}.get(sp)
        if split_name is None:        # test → 학습/검증 제외 (누수 차단)
            continue
        recs = parse_bbgt_v3(sm.ann_path)         # (coco_id,x1,y1,x2,y2) 원본 px
        W, H = Image.open(sm.image_path).size
        # 라벨 txt (COCO id cx cy w h normalized)
        out_lbl = lbl_dir / sm.class_dir / f"{sm.image_path.stem}.txt"
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for (cid, x1, y1, x2, y2) in recs:
            cx = ((x1 + x2) * 0.5) / W; cy = ((y1 + y2) * 0.5) / H
            bw = (x2 - x1) / W; bh = (y2 - y1) / H
            if bw > 0 and bh > 0:
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out_lbl.write_text("\n".join(lines), encoding="utf-8")
        n_box[split_name] += len(lines)
        # 이미지 경로 (junction 경유 → labels 자동 치환)
        img_path = img_dir / sm.class_dir / sm.image_path.name
        (train_list if split_name == "train" else val_list).append(str(img_path))

    (data_root / "train.txt").write_text("\n".join(train_list), encoding="utf-8")
    (data_root / "val.txt").write_text("\n".join(val_list), encoding="utf-8")

    # names: COCO 80 (pretrained head 유지)
    from ultralytics import YOLO
    names = YOLO(str(P.yolov8n)).names
    yaml_path = data_root / "exdark.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": str(data_root),
            "train": "train.txt",
            "val": "val.txt",
            "names": {int(k): v for k, v in names.items()},
        }, f, allow_unicode=True, sort_keys=False)

    print(f"  train 이미지 {len(train_list)} (box {n_box['train']}), "
          f"val {len(val_list)} (box {n_box['val']})")
    print(f"  data yaml → {yaml_path}")
    return yaml_path


def main() -> int:
    p = argparse.ArgumentParser(description="P3 조건 C — YOLOv8n ExDark fine-tune")
    p.add_argument("--config", type=str, default="configs/p3_joint_budget.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--prepare_only", action="store_true", help="데이터 준비만")
    p.add_argument("--data_yaml", type=str, default=None,
                   help="외부 data.yaml (주면 raw prepare 스킵; D/E enhanced 데이터용)")
    p.add_argument("--project", type=str, default=None, help="결과 디렉토리 (기본 runs/p3_yolo_C)")
    args = p.parse_args()

    P = load_paths()
    budget = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = args.device or "0"
    data_root = P.runs / "p3_yolo_data"
    project = Path(args.project) if args.project else P.runs / "p3_yolo_C"

    print(HR)
    print(" [P3] YOLOv8n fine-tune  (budget 공유: %s)" % args.config)
    print(HR)
    print(f"  budget : {args.config}")
    if args.data_yaml:
        yaml_path = Path(args.data_yaml)
        print(f"  data   : (외부) {yaml_path}")
    else:
        yaml_path = prepare_data(P, budget, data_root)
    if args.prepare_only:
        print("  (prepare_only) 데이터 준비 완료.")
        return 0

    tr = budget["train"]
    from ultralytics import YOLO
    model = YOLO(str(P.yolov8n))   # COCO pretrained init
    print(f"  init   : {P.yolov8n.name}  epochs={tr['epochs']} imgsz={tr['imgsz']} "
          f"batch={tr['batch']} lr0={tr['lr0']}")
    print(HR)
    model.train(
        data=str(yaml_path),
        epochs=tr["epochs"], imgsz=tr["imgsz"], batch=tr["batch"],
        optimizer=tr["optimizer"], lr0=tr["lr0"], lrf=tr["lrf"], cos_lr=tr["cos_lr"],
        momentum=tr["momentum"], weight_decay=tr["weight_decay"],
        warmup_epochs=tr["warmup_epochs"], seed=tr["seed"],
        hsv_h=tr["hsv_h"], hsv_s=tr["hsv_s"], hsv_v=tr["hsv_v"],
        degrees=tr["degrees"], translate=tr["translate"], scale=tr["scale"],
        fliplr=tr["fliplr"], mosaic=tr["mosaic"], close_mosaic=tr["close_mosaic"],
        project=str(project), name="train", exist_ok=True,
        device=device, verbose=True, plots=False,
    )
    best = project / "train" / "weights" / "best.pt"
    print(HR)
    print(f"  완료. best → {best}")
    print(f"  평가: python experiments/eval_detection_native.py --mode original --split test "
          f"--yolo_weights {best} --label 'C [test]' --results_dir runs/eval_test_C")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

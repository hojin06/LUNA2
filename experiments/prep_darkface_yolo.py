"""[실험 4 준비] DARK FACE → YOLO 학습 포맷 변환 + train/val 분할 (CPU).

검출기 적응(YOLOv8-face 를 DARK FACE 도메인에 파인튜닝) 실험용 데이터셋을 만든다.
- 원본 이미지는 복사하지 않고 **하드링크**(동일 NTFS 볼륨, 권한 불필요, 디스크 0)로 연결.
- 라벨: DARK FACE 'x1 y1 x2 y2'(절대) → YOLO '0 cx cy w h'(정규화), 단일 클래스 face.
- 분할: stem 정렬 후 5장마다 1장을 val(20%), 나머지 train. 결정적(난수 없음).

출력 <out>/:
    images/<stem>.png      (원본 하드링크)
    labels/<stem>.txt      (YOLO)
    train.txt / val.txt    (이미지 경로 목록; ultralytics 용)
    val_stems.txt          (val 이미지 stem 목록; eval_darkface --image_list 용)
    darkface.yaml          (ultralytics data 설정)

ultralytics 는 이미지경로의 '/images/'→'/labels/' 치환으로 라벨을 찾으므로 이 레이아웃이면
자동 인식된다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    except Exception:
        pass

from PIL import Image

from experiments.eval_darkface import collect_darkface_samples, parse_darkface_label
from src.utils.paths import load_paths


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)            # 하드링크 (동일 볼륨, 권한 불필요, 추가 용량 0)
    except OSError:
        import shutil
        shutil.copy2(src, dst)       # 다른 볼륨 등 → 복사 폴백


def main() -> int:
    P = load_paths()
    ap = argparse.ArgumentParser(description="[실험4 준비] DARK FACE → YOLO 포맷 + 분할")
    ap.add_argument("--darkface_root", type=str, default=str(P.exdark.parent / "DARKFACE"))
    ap.add_argument("--out", type=str, default=str(P.exdark.parent / "DARKFACE_yolo"))
    ap.add_argument("--val_every", type=int, default=5, help="N장마다 1장을 val (기본 5=20%%)")
    args = ap.parse_args()

    root = Path(args.darkface_root)
    out = Path(args.out)
    img_out = out / "images"; lbl_out = out / "labels"
    img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)

    samples = sorted(collect_darkface_samples(root), key=lambda s: s.image_path.stem)
    print(f"  samples: {len(samples)}  → out: {out}")

    train_paths, val_paths, val_stems = [], [], []
    n_box = 0
    for i, sm in enumerate(samples):
        stem = sm.image_path.stem
        try:
            with Image.open(sm.image_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"  [warn] {stem} 크기 실패: {e}"); continue

        dst_img = img_out / f"{stem}.png"
        _link_or_copy(sm.image_path, dst_img)

        boxes = parse_darkface_label(sm.label_path)
        lines = []
        for (x1, y1, x2, y2) in boxes:
            cx = ((x1 + x2) * 0.5) / W; cy = ((y1 + y2) * 0.5) / H
            bw = (x2 - x1) / W; bh = (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            bw = min(bw, 1.0); bh = min(bh, 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_box += len(lines)

        if i % args.val_every == 0:
            val_paths.append(str(dst_img)); val_stems.append(stem)
        else:
            train_paths.append(str(dst_img))

    (out / "train.txt").write_text("\n".join(train_paths), encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_paths), encoding="utf-8")
    (out / "val_stems.txt").write_text("\n".join(val_stems), encoding="utf-8")

    import yaml
    yaml_path = out / "darkface.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out), "train": "train.txt", "val": "val.txt",
                        "names": {0: "face"}}, f, allow_unicode=True, sort_keys=False)

    print(f"  train {len(train_paths)} | val {len(val_paths)} | boxes {n_box}")
    print(f"  yaml → {yaml_path}")
    print(f"  val_stems → {out / 'val_stems.txt'}  (eval --image_list 용)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

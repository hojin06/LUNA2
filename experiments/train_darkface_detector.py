"""[실험 4] DARK FACE 도메인에 YOLOv8-face 파인튜닝(검출기 적응).

WIDER FACE(정상조명) 로 학습된 검출기를 DARK FACE 의 **원본 저조도** train split 으로
파인튜닝한다. 적응된 검출기로 향상 기법들을 다시 평가하면, '고정 WIDER-FACE 검출기의
도메인 편향'이 LUNA2 열세의 원인인지 분리할 수 있다.

선행: experiments/prep_darkface_yolo.py 로 DARKFACE_yolo/ 생성 완료 상태여야 함.

사용 예::
    python experiments/train_darkface_detector.py --epochs 30
출력: runs/detect/darkface_adapt/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

from src.utils.paths import load_paths


def main() -> int:
    P = load_paths()
    default_data = P.exdark.parent / "DARKFACE_yolo" / "darkface.yaml"
    # NOTE: yolov8n-face.pt 는 pose(랜드마크) 모델이라 박스-only DARK FACE 로 학습 불가.
    # 일반 detection 모델(yolov8n.pt, COCO)에서 단일 클래스(face) 로 파인튜닝한다.
    default_model = Path(str(P.yolov8n))  # CODES/yolov8n.pt (detection)
    ap = argparse.ArgumentParser(description="[실험4] YOLOv8-face DARK FACE 적응 파인튜닝")
    ap.add_argument("--model", type=str, default=str(default_model))
    ap.add_argument("--data", type=str, default=str(default_data))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default=str(_LUNA2_ROOT / "runs" / "detect"))
    ap.add_argument("--name", type=str, default="darkface_adapt")
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"  model : {args.model}")
    print(f"  data  : {args.data}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name=args.name, exist_ok=True)
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n  적응 가중치 → {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# DARK FACE 극저조도 얼굴검출 평가 — 즉시 실행 가이드

충전기 연결 후, 아래 명령 **하나만** 붙여넣으면 4개 모델(LUNA / LUNA2 / SCI / Zero-DCE)이
전체 6,000장에 대해 순차 실행되고 결과 CSV + 요약이 저장됩니다.

준비 완료 상태 (이미 끝남):
- DARK FACE 6,000장 + 라벨 : `SmallSizePM_GAN_model/DataSet/DARKFACE/{image,label}`
- 얼굴 검출기            : `SmallSizePM_GAN_model/CODES/yolov8n-face.pt`
- 평가 스크립트          : `experiments/eval_darkface.py` (CPU 3장 검증 통과)
- 프로토콜              : 향상 입력 640px 캡 → native 복원 → YOLOv8n-face 검출, conf 0.25

---

## A. 일괄 실행 (권장) — 백그라운드 한 방

```bash
cd "c:/대학교/LUNA_paperWORKS/LUNA2"
mkdir -p runs/eval_darkface/logs
{
  python experiments/eval_darkface.py --method luna   > runs/eval_darkface/logs/luna.log   2>&1
  python experiments/eval_darkface.py --method luna2  > runs/eval_darkface/logs/luna2.log  2>&1
  python experiments/eval_darkface.py --method zerodce> runs/eval_darkface/logs/zerodce.log 2>&1
  python experiments/eval_darkface.py --method sci --sci_weight easy      > runs/eval_darkface/logs/sci_easy.log      2>&1
  python experiments/eval_darkface.py --method sci --sci_weight medium    > runs/eval_darkface/logs/sci_medium.log    2>&1
  python experiments/eval_darkface.py --method sci --sci_weight difficult > runs/eval_darkface/logs/sci_difficult.log 2>&1
  echo ALL_COMPLETE
} > runs/eval_darkface/logs/_driver.log 2>&1
```

예상 시간: 모델당 약 4~6분 (640 캡, GPU) → 전체 약 25~30분.

## B. 개별 실행 / 디버그

```bash
# 빠른 sanity (30장)
python experiments/eval_darkface.py --method luna --max_samples 30

# 개별 전체
python experiments/eval_darkface.py --method luna
python experiments/eval_darkface.py --method luna2
python experiments/eval_darkface.py --method zerodce
python experiments/eval_darkface.py --method sci --sci_weight easy
```

## 옵션
- `--luna_ckpt <path>`  : LUNA 체크포인트 (기본 = LoLI-30K best). LOL-v2 비교하려면 교체.
- `--enh_max_side 0`    : 캡 끄고 native 향상 (느림, OOM 위험).
- `--conf 0.001`        : AP용 full PR 곡선 원하면 낮춤 (현재 0.25, ExDark 실험과 일관).
- `--max_samples N`     : 앞 N장만.

## 결과 위치
- CSV  : `runs/eval_darkface/darkface_<method>.csv` (Original vs 향상, map50/map/P/R)
- 요약 : 각 로그 끝 `[SUMMARY]` 줄

## 주의
- SCI 는 원본 로더가 CUDA 하드코딩이라 **GPU 필요** (CPU 단독 실행 불가). 나머지 3종은 CPU 도 가능.
- 모든 모델의 Original baseline 은 동일하게 재현되어야 함(파이프라인 정상 확인용).

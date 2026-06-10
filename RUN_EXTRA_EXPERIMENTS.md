# 추가 실험 1·2·4 — 즉시 실행 가이드 (DARK FACE)

"시각적으론 LUNA2가 더 나은데 검출은 Zero-DCE가 이김"의 원인(검출기 도메인 편향 +
작은 얼굴 파괴)을 분리 검증. 모두 준비·CPU 검증 완료. 충전 상태에서 명령만 붙여넣으면 됨.

전제: DARK FACE 6000장 + `yolov8n-face.pt` + `DARKFACE_yolo/`(실험4용, 이미 생성됨) 준비됨.

---

## 실험 1 — native 해상도 재실행 (640 캡 제거)

작은 얼굴(중앙값 11px)이 캡 때문에 손해봤는지 확인. **LUNA2/Zero-DCE/SCI 에 영향**
(LUNA 는 내부적으로 항상 256 → 변화 없음, 생략).

```powershell
cd "c:\대학교\LUNA_paperWORKS\LUNA2"; foreach ($m in 'luna2','zerodce') { python experiments/eval_darkface.py --method $m --enh_max_side 0 --results_dir runs/eval_darkface_native }; foreach ($w in 'easy','medium','difficult') { python experiments/eval_darkface.py --method sci --sci_weight $w --enh_max_side 0 --results_dir runs/eval_darkface_native }
```
결과: `runs/eval_darkface_native/darkface_*.csv`. → 캡본 대비 LUNA2 가 따라잡으면 "해상도/캡" 요인 확정.

## 실험 2 — 얼굴 크기별(tiny/small/large) mAP

LUNA2 가 *큰 얼굴*에선 경쟁력 있고 *작은 얼굴*에서만 지는지 → "공간복원+작은얼굴" 메커니즘 확정.

```powershell
cd "c:\대학교\LUNA_paperWORKS\LUNA2"; foreach ($m in 'luna','luna2','zerodce') { python experiments/eval_darkface_bysize.py --method $m }; python experiments/eval_darkface_bysize.py --method sci --sci_weight easy
```
결과: `runs/eval_darkface_bysize/bysize_*.csv` (버킷별 orig/enh AP50 + recall).
(native 로 보려면 각 명령에 `--enh_max_side 0` 추가)

## 실험 4 — 검출기 적응 (2단계: 학습 → 적응 검출기로 재평가)

**4-A. DARK FACE 도메인에 YOLOv8-face 파인튜닝** (train 4800장, GPU, ~30~60분)
```powershell
cd "c:\대학교\LUNA_paperWORKS\LUNA2"; python experiments/train_darkface_detector.py --epochs 30 --batch 16
```
→ 적응 가중치: `runs/detect/darkface_adapt/weights/best.pt`

**4-B. 적응 검출기로 val(1200장)만 재평가** (누수 차단: train 에 없던 val 만)
```powershell
cd "c:\대학교\LUNA_paperWORKS\LUNA2"; $vs="../SmallSizePM_GAN_model/DataSet/DARKFACE_yolo/val_stems.txt"; $bw="runs/detect/darkface_adapt/weights/best.pt"; foreach ($m in 'luna','luna2','zerodce') { python experiments/eval_darkface.py --method $m --face_weights $bw --image_list $vs --results_dir runs/eval_darkface_adapted }; python experiments/eval_darkface.py --method sci --sci_weight easy --face_weights $bw --image_list $vs --results_dir runs/eval_darkface_adapted
```

**4-C. (비교 기준) 고정 WIDER-FACE 검출기로도 같은 val 만 평가** — 적응 전/후 비교용
```powershell
cd "c:\대학교\LUNA_paperWORKS\LUNA2"; $vs="../SmallSizePM_GAN_model/DataSet/DARKFACE_yolo/val_stems.txt"; foreach ($m in 'luna','luna2','zerodce') { python experiments/eval_darkface.py --method $m --image_list $vs --results_dir runs/eval_darkface_frozen_val }; python experiments/eval_darkface.py --method sci --sci_weight easy --image_list $vs --results_dir runs/eval_darkface_frozen_val
```

해석: **4-C(고정) vs 4-B(적응)** 에서 LUNA2 의 상대 순위가 적응 후 올라가면 → "고정 검출기
도메인 편향"이 원인임을 입증. 그대로면 → LUNA2 의 향상 자체가 검출에 덜 유용.

> 설계 주의: 4-A 는 *원본 저조도*로 적응한 뒤 *향상 이미지*로 평가(약간의 도메인 시프트
> 잔존). 더 엄밀히는 방법별로 enhance한 train 으로 각각 적응해야 하나(학습 ×4), 비용이 커서
> 1차로 원본 적응 버전을 제공. 결과가 유의미하면 방법별 적응으로 확장.

# LUNA2 — 저조도 향상 기반 객체 검출 향상 (경량 모델 연구)

LUNA(LightEnhanceGenerator)의 후속 모델 연구 디렉터리. **연구 목표는 PSNR/SSIM
같은 픽셀 지표가 아니라 저조도 환경에서의 객체 검출 성능(ExDark mAP) 향상**이다.
이를 위해 (1) bilateral grid 기반 full-resolution 경량 향상기와 (2) frozen
YOLOv8n 신호를 활용한 detection-aware joint training 을 단계적으로 도입한다.

핵심 가설: 픽셀 충실도만 최적화하면 검출에 중요한 경계/텍스처가 평활화된다.
검출기 친화적 향상을 직접 최적화하면 같은(혹은 더 적은) 연산으로 mAP 를 높일 수 있다.

---

## 디렉터리 구조

```
LUNA2/
├── configs/
│   ├── paths.yaml              # ★ 데이터/가중치/체크포인트 경로의 단일 정의처
│   ├── diagnostic.yaml         # 0단계 진단 설정 (템플릿)
│   ├── bilateral_base.yaml     # bilateral 단계 학습 설정 (템플릿)
│   └── joint_train.yaml        # joint 단계 학습 설정 (템플릿)
├── src/
│   ├── models/
│   │   ├── luna_base.py        # [이식] 원본 LightEnhanceGenerator + 로드/전처리 헬퍼
│   │   └── bilateral_grid.py   # [스텁] bilateral grid 경량 향상기
│   ├── losses/
│   │   ├── restoration.py      # [스텁] L1 / SSIM / perceptual
│   │   └── detection_aware.py  # [스텁] frozen YOLO feature / detection loss
│   ├── data/
│   │   └── lowlight_dataset.py # [이식] LOL v1/v2 · LoLI-Street 페어 로더 + PairedAugment
│   └── utils/
│       ├── paths.py            # paths.yaml → 절대경로 해석기
│       ├── metrics.py          # [이식] PSNR / SSIM / LPIPS / evaluate
│       └── profile.py          # [스텁] params / MACs / FPS 프로파일링
├── experiments/
│   ├── 00_resolution_diagnostic.py  # [스텁] 0단계: 해상도 ↔ 검출 영향 진단
│   ├── train.py                     # [스텁] bilateral / joint 학습 엔트리
│   ├── eval_restoration.py          # [이식·동작] PSNR/SSIM/LPIPS 평가
│   └── eval_detection.py            # [이식·동작] ExDark mAP / P / R 평가
├── deploy/
│   ├── export_onnx.py          # [스텁] ONNX export
│   └── bench_fps.py            # [스텁] FPS 벤치마크
├── README.md
├── requirements.txt
└── .gitignore
```

**[이식]** = 원본 `SmallSizePM_GAN_model/CODES` 에서 그대로 옮겨와 동작하는 코드.
**[스텁]** = docstring + 시그니처만 있고 `NotImplementedError`. 단계별로 채운다.

---

## 경로 설정 (paths.yaml — 단일 수정 지점)

데이터/가중치 경로는 **코드에 절대 하드코딩하지 않는다.** 전부
[`configs/paths.yaml`](configs/paths.yaml) 한 곳에 정의하고,
[`src/utils/paths.py`](src/utils/paths.py)`::load_paths()` 가 `root` + 상대경로를
합쳐 절대 `Path` 로 해석한다. **다른 머신으로 옮길 때 `root` 한 줄만 고치면 된다.**

정의된 항목:
- 데이터셋: ExDark / LOLdataset(LOL v1) / LOL-v2 / LoLI-Street
- 가중치: frozen YOLOv8n, 원본 LUNA 체크포인트(`ext_lol_v2_real_stage2_best.pth`)

> 데이터·가중치·모델은 **복사하지 않는다.** 원본 위치를 참조만 한다.

---

## 이식 코드의 원본 동일성 확인

이식한 모델/평가가 원본과 같은 결과를 내는지 다음으로 확인할 수 있다 (원본
체크포인트 기준):

```bash
# 1) 복원 품질 (PSNR/SSIM) — 원본 evaluate() 와 동일 수치여야 함
python experiments/eval_restoration.py --dataset lol_v1 --split eval

# 2) ExDark 검출 (mAP/P/R) — 원본 downstream_exdark.py 와 동일 수치여야 함
python experiments/eval_detection.py            # paths.yaml 의 luna_original 사용
python experiments/eval_detection.py --max_samples 50   # 빠른 스모크 테스트
```

- 모델 정의(`luna_base.LightEnhanceGenerator`), hybrid_v1 conv_config 상수,
  전처리(256 BILINEAR → [-1,1]), mAP 계산기는 모두 원본을 그대로 옮긴 것이다.
- `eval_detection.py` 는 원본과 동일한 generator/전처리/누적기를 쓰되 경로만
  `paths.yaml` 에서 주입한다 → 동일 입력·동일 코드 → 동일 출력.

---

## 실행 순서 (연구 로드맵)

```
0단계 진단  →  bilateral 단계  →  joint 단계
```

### 0단계 — 진단 (스텁)
해상도 down→up 왕복이 검출 병목인지 확인하여 full-res 처리(bilateral)의 동기를 세운다.
```bash
python experiments/00_resolution_diagnostic.py   # configs/diagnostic.yaml
```

### 1단계 — bilateral 향상기 학습 (스텁)
restoration loss 로 bilateral grid 향상기를 학습. baseline 대비 효율/품질 비교.
```bash
python experiments/train.py --config configs/bilateral_base.yaml
python experiments/eval_restoration.py --checkpoint runs/bilateral_base/best.pth
python experiments/eval_detection.py   --checkpoint runs/bilateral_base/best.pth
```

### 2단계 — detection-aware joint training (스텁)
frozen YOLOv8n 신호로 검출 친화적 향상을 직접 최적화. **ExDark mAP 가 최종 지표.**
```bash
python experiments/train.py --config configs/joint_train.yaml
python experiments/eval_detection.py --checkpoint runs/joint_train/best.pth
```

### 배포/효율 (스텁)
```bash
python deploy/export_onnx.py --checkpoint runs/joint_train/best.pth --out luna2.onnx
python deploy/bench_fps.py   --checkpoint runs/joint_train/best.pth
```

---

## 설치

```bash
pip install -r requirements.txt
# PyTorch 는 CUDA 버전에 맞춰 https://pytorch.org 에서 설치 권장
```

`ultralytics` 는 ExDark 검출 평가에, `lpips` 는 `--lpips` 옵션에만 필요하다.
`tqdm` 미설치 시 진행률 표시만 생략된다.

# LUNA2 — Detection-Aware Low-Light Enhancement

Successor to **LUNA** (`LightEnhanceGenerator`). LUNA was optimized for *pixel
fidelity* (PSNR/SSIM); LUNA2 is optimized for what actually matters downstream:
**object-detection performance in low light** (ExDark / DARK FACE mAP).

The core hypothesis: **optimizing pixel fidelity alone smooths away the edges and
texture a detector relies on.** If we instead optimize a *detector-friendly*
enhancement — at full resolution, with the detector in the loop — we can raise mAP
at the same (or lower) compute.

> 한국어 버전은 문서 맨 아래에 있습니다 → [한국어 README](#luna2--저조도-향상-기반-객체-검출-한국어).

---

## TL;DR — what changed and why it matters

| | **LUNA** (original) | **LUNA2** |
|---|---|---|
| **Objective** | PSNR / SSIM (pixel fidelity) | **Detection mAP** (downstream task) |
| **Enhancer** | `LightEnhanceGenerator`: 4-stage DSConv U-Net + light attention | `BilateralLowLightNet`: HDRNet-style bilateral grid (low-res coeff net + full-res guidance + slice/affine + DSConv refine) |
| **Resolution** | fixed **256×256** (down→up round-trip) | **native full resolution** |
| **Training** | 2-stage: supervised (L1+VGG+SSIM) → **PatchGAN** adversarial | P1 supervised → **P2 detection-aware (frozen YOLOv8n)** → P3 joint |
| **Detector in loop** | ✗ none | ✓ frozen YOLOv8n detection loss |
| **Data** | LOL (485+15 pairs) | LOL v1/v2 + LoLI-Street (P1), ExDark (P2/P3) |
| **Params** | 205,093 (~205K) | bilateral-grid net (profiled via `deploy/`) |

**Headline result (DARK FACE, 6,000 images, frozen YOLOv8n-face, mAP@50):**

| Method | mAP@50 | Δ vs raw | Recall |
|---|---:|---:|---:|
| Original (raw dark) | 0.1132 | — | 0.1212 |
| **LUNA** (original) | 0.0771 | **−0.0361** ❌ | 0.0870 |
| **LUNA2** | **0.1270** | **+0.0138** ✅ | 0.1365 |
| SCI (easy) | 0.1461 | +0.0329 | 0.1576 |
| Zero-DCE | 0.1968 | +0.0837 | 0.2150 |

**LUNA2 reverses LUNA's regression.** The original LUNA *lowered* detection mAP by
−3.6 points below doing nothing; LUNA2 turns that into a **+1.4 point gain** (and a
**+65 % relative jump over LUNA**). LUNA2 does not yet beat the best dedicated
low-light methods (Zero-DCE, SCI) — the analysis below explains exactly why, with
data.

---

## 1. Motivation — the problem LUNA2 was built to fix

LUNA produced visually pleasant, high-PSNR images, but when we put those enhanced
images through a *frozen* object detector, **detection got worse, not better.**

**ExDark, frozen YOLOv8n (COCO), native resolution — mAP@50 vs the raw baseline 0.4472:**

| Method (enhancer) | mAP@50 | Δ |
|---|---:|---:|
| **Original (no enhancement)** | **0.4472** | — |
| SCI (easy) | 0.4523 | **+0.0051** ✅ |
| EnlightenGAN | 0.4203 | −0.0269 |
| Zero-DCE | 0.4046 | −0.0426 |
| **LUNA** (LoLI-30K ckpt) | 0.3462 | **−0.1009** |
| **LUNA** (LOL-v2 ckpt) | 0.2815 | **−0.1656** |
| FUnIE-GAN | 0.1970 | −0.2501 |

Two uncomfortable facts fall out of this table:

1. **Most "enhancement" hurts a frozen detector.** Only SCI nudges mAP up; every
   other method — including LUNA — drags it *down*. Pretty pixels ≠ detectable pixels.
2. **LUNA is among the worst offenders** (−0.10 to −0.17 mAP), precisely because it
   was tuned hard for PSNR.

This is the gap LUNA2 targets.

---

## 2. Diagnosis — resolution is a first-order bottleneck

Before redesigning the model we ran a controlled diagnostic: feed the detector the
*same* ExDark images but force them through different resolution pipelines (no
enhancement, only resampling).

**ExDark, mAP@50:**

| Pipeline | mAP@50 | Δ vs native |
|---|---:|---:|
| **native (full resolution)** | **0.4472** | — |
| down → 256 → up (round-trip) | 0.3992 | −0.0480 |
| down → 256 (stay small) | 0.3376 | −0.1096 |

The 256×256 round-trip *alone* — exactly what LUNA does internally — costs **~5–11
mAP points**, before the enhancer changes a single color. **A 256px enhancer can
never win at detection**, because the downsampling destroys the small-object detail
the detector needs. This directly motivates LUNA2's full-resolution bilateral design.

---

## 3. What LUNA2 changes (the three improvements)

### (a) New objective — optimize the detector, not the pixels
LUNA2 keeps a reconstruction term for stability but adds a **detection-aware loss**:
a *frozen* YOLOv8n scores the enhanced image against ground-truth boxes, and that
gradient flows back into the enhancer. The enhancer learns to make images the
detector likes — not images that merely look bright.

### (b) New architecture — full-resolution bilateral grid
`LightEnhanceGenerator` (256px U-Net) is replaced by **`BilateralLowLightNet`**, an
HDRNet-style design:

- a **low-res coefficient network** predicts a bilateral grid of per-pixel affine
  color transforms (cheap, runs at 256);
- a **full-res guidance map** + **trilinear slice** applies those transforms at the
  image's *native* resolution (no down/up round-trip → keeps small-object detail);
- a lightweight **DSConv refinement** residual cleans up the affine output.

This is *not* the old U-Net retrained — it is a different, full-resolution model.
(`src/models/bilateral_grid.py` is fully implemented, not a stub.)

### (c) New training program — staged, detector-in-the-loop
| Phase | What | Loss | Detector |
|---|---|---|---|
| **P1** | Pretrain the bilateral enhancer for restoration | L1 + SSIM + perceptual (VGG) | — |
| **P2** | Detection-aware fine-tune on ExDark | λ_rec·L1(anchor) + λ_det·YOLO-loss | **frozen** YOLOv8n |
| **P3** | End-to-end joint co-adaptation | detection + reconstruction + grid-TV | **unfrozen** YOLOv8n |

(P2 sweeps the detection weight λ_det ∈ {0.001 … 0.05}; P3 compares detection-only
fine-tune vs frozen-enhancer vs jointly-trained enhancer under a shared budget.)

Versus LUNA's recipe (supervised L1/VGG/SSIM → PatchGAN adversarial), LUNA2 **drops
the GAN entirely** and replaces "look real" with "be detectable."

---

## 4. Results & honest analysis

### 4.1 DARK FACE — LUNA2 fixes LUNA, but Zero-DCE/SCI still lead
See the headline table above. LUNA2 is the *only* LUNA-lineage model that helps
detection (+0.014), reversing LUNA's −0.036. But Zero-DCE (+0.084) and SCI (+0.033)
still win. **Why?** The next two experiments answer that — and the answers are the
real research contribution.

### 4.2 Why LUNA2 trails — it destroys *tiny* faces
DARK FACE is dominated by extremely small faces (median ≈ 11 px; **33,892** of 50k
GT faces are <16 px). Breaking AP@50 down by face size:

| Face size | GT | Original | LUNA | **LUNA2** | Zero-DCE | SCI |
|---|---:|---:|---:|---:|---:|---:|
| tiny (<16 px) | 33,892 | 0.0035 | 0.0001 | **0.0013** | **0.0098** | 0.0038 |
| small (16–32) | 12,202 | 0.2479 | 0.1227 | **0.2694** | 0.4415 | 0.3204 |
| large (≥32) | 4,302 | 0.4601 | 0.4410 | **0.5798** | 0.6937 | 0.5922 |

Read it this way:
- **LUNA2 is genuinely strong on large faces** (0.46 → 0.58, +0.12) and helps small
  faces — its full-res spatial processing *recovers* mid/large objects.
- But on **tiny faces it barely moves the needle** (0.0035 → 0.0013), and DARK FACE
  is *mostly* tiny faces — so the overall average is dragged down.
- **Zero-DCE wins because it is a per-pixel curve** that brightens uniformly and
  leaves tiny faces structurally intact (tiny AP 0.0035 → 0.0098, ~3× recall).
- **LUNA (original) collapses everywhere** — it actively erases tiny (→0.0001) and
  small faces.

So LUNA2's bilateral/spatial machinery is a double-edged sword: great for objects
with structure, still too aggressive for sub-16px faces.

### 4.3 The bigger caveat — enhancement only helps a *mismatched* detector
We then fine-tuned the face detector on the dark domain itself and re-evaluated the
held-out val split (no leakage):

**DARK FACE val (1,200 images), mAP@50:**

| Detector | Original (raw) | LUNA2 | Zero-DCE |
|---|---:|---:|---:|
| **Frozen** (WIDER-FACE) | 0.1057 | **0.1221** (+0.016) | — |
| **Adapted** (fine-tuned on dark) | **0.3747** | 0.1772 (−0.197) | 0.3196 (−0.055) |

This is the most important — and most sobering — finding:

- With a **frozen, domain-mismatched** detector, enhancement helps (LUNA2 +0.016).
- But once the detector is **adapted to dark images**, the **raw images win
  outright** (0.375), and *every* enhancement *hurts* — LUNA2 most of all (−0.197),
  because it alters the image most.

**Conclusion:** the value of low-light enhancement for detection is **conditional on
detector domain mismatch.** If you can fine-tune the detector, do that first. LUNA2
helps in the realistic case where the detector is fixed/off-the-shelf, but it is not
a substitute for detector adaptation. This is an honest negative result that frames
the next research step (per-method detector co-adaptation, P3).

---

## 5. Comparison with other models (one-look summary)

| Model | Family | ExDark Δ mAP@50¹ | DARK FACE mAP@50² | Note |
|---|---|---:|---:|---|
| Original (none) | — | 0.0000 | 0.1132 | raw baseline |
| **LUNA** | GAN, 256px, PSNR | −0.1009 / −0.1656 | 0.0771 | pixel-optimized → hurts detection |
| **LUNA2** | bilateral grid, full-res, detection-aware | (see §4.3) | **0.1270** | reverses LUNA; best in LUNA lineage |
| Zero-DCE | per-pixel curve | −0.0426 | **0.1968** | best on tiny faces, detection leader |
| SCI | self-calibrated | **+0.0051** | 0.1461 | only method that helps frozen ExDark |
| EnlightenGAN | unpaired GAN | −0.0269 | — | mild detection hit |
| FUnIE-GAN | underwater GAN | −0.2501 | — | severe detection hit (domain mismatch) |

¹ ExDark, frozen YOLOv8n (COCO), native res, vs 0.4472 baseline.
² DARK FACE, 6,000 imgs, frozen YOLOv8n-face, conf 0.25.

**Takeaways:** simple per-pixel methods (Zero-DCE, SCI) are remarkably hard to beat
for *frozen-detector* low-light detection because they preserve structure; heavier
generative models (LUNA, FUnIE, EnlightenGAN) tend to hurt because they hallucinate
or smooth. LUNA2's contribution is showing that a full-res, detection-aware redesign
moves the LUNA lineage from clearly-harmful to net-positive — and quantifying the
remaining gap (tiny objects + detector adaptation).

---

## 6. Repository structure

```
LUNA2/
├── configs/
│   ├── paths.yaml              # ★ single source of truth for data/weight/ckpt paths
│   ├── diagnostic.yaml         # stage-0 resolution diagnostic
│   ├── bilateral_base.yaml     # P1 bilateral pretraining config
│   ├── joint_train.yaml        # P2 detection-aware config
│   └── p3_joint_budget.yaml    # P3 joint co-training (C/D/E conditions)
├── src/
│   ├── models/
│   │   ├── luna_base.py         # [ported] original LightEnhanceGenerator + helpers
│   │   └── bilateral_grid.py    # BilateralLowLightNet (HDRNet-style, full-res)
│   ├── losses/
│   │   ├── restoration.py       # L1 / SSIM / perceptual
│   │   └── detection_aware.py   # frozen-YOLO detection loss
│   ├── data/lowlight_dataset.py # LOL v1/v2 · LoLI-Street paired loader + PairedAugment
│   └── utils/
│       ├── paths.py             # paths.yaml → absolute-path resolver
│       ├── metrics.py           # PSNR / SSIM / LPIPS / mAP
│       ├── inference.py         # LUNA2 full-res enhance()
│       └── profile.py           # params / MACs / FPS profiling
├── experiments/
│   ├── 00_resolution_diagnostic.py   # §2 resolution diagnostic
│   ├── train.py / train_detection_aware.py / train_p3_joint.py  # P1 / P2 / P3
│   ├── eval_restoration.py           # PSNR/SSIM/LPIPS
│   ├── eval_detection*.py            # ExDark mAP (frozen / native / per-method)
│   ├── eval_darkface*.py             # DARK FACE mAP (overall / by-size / adapted)
│   └── train_darkface_detector.py    # §4.3 detector adaptation
├── deploy/                      # export_onnx.py, bench_fps.py
├── RUN_DARKFACE.md              # one-shot DARK FACE eval guide
├── RUN_EXTRA_EXPERIMENTS.md     # §4.2/§4.3 extra-experiment guide
└── README.md
```

> **Paths.** Data/weights are never hardcoded — they live in
> [`configs/paths.yaml`](configs/paths.yaml), resolved by
> [`src/utils/paths.py`](src/utils/paths.py)`::load_paths()`. To move machines, edit
> the single `root` line. Datasets, weights, and checkpoints are **referenced, not
> copied** (and are git-ignored: `runs/`, `*.pth`, `*.pt`, `*.onnx`).

---

## 7. Reproducing the numbers

```bash
pip install -r requirements.txt    # PyTorch per your CUDA from https://pytorch.org
```

```bash
# Verify the ported model matches the original LUNA (restoration + ExDark)
python experiments/eval_restoration.py --dataset lol_v1 --split eval
python experiments/eval_detection.py                         # ExDark mAP

# §2 resolution diagnostic
python experiments/00_resolution_diagnostic.py

# §4.1 DARK FACE — all four models (see RUN_DARKFACE.md)
python experiments/eval_darkface.py --method luna2
python experiments/eval_darkface.py --method zerodce
# ... luna / sci

# §4.2 by-size  /  §4.3 detector adaptation  (see RUN_EXTRA_EXPERIMENTS.md)
python experiments/eval_darkface_bysize.py --method luna2
python experiments/train_darkface_detector.py --epochs 30 --batch 16
```

`ultralytics` is needed for detection eval; `lpips` only for `--lpips`; `tqdm` is
optional (progress bars only).

> **Numbers in this README** come from the CSVs under `runs/eval_*` produced by the
> scripts above (frozen YOLOv8n / YOLOv8n-face, conf 0.25). They are git-ignored
> artifacts — regenerate them with the commands above.

---
---

# LUNA2 — 저조도 향상 기반 객체 검출 (한국어)

**LUNA**(`LightEnhanceGenerator`)의 후속 모델. LUNA는 *픽셀 충실도*(PSNR/SSIM)를
최적화했지만, LUNA2는 실제로 중요한 것 — **저조도에서의 객체 검출 성능**(ExDark /
DARK FACE mAP) — 을 최적화한다.

핵심 가설: **픽셀 충실도만 최적화하면 검출에 중요한 경계·텍스처가 평활화된다.**
대신 *검출기 친화적* 향상을 — full resolution 에서, 검출기를 학습 루프에 넣어 —
최적화하면 같은(혹은 더 적은) 연산으로 mAP 를 높일 수 있다.

---

## 한눈에 — 무엇이, 왜 바뀌었나

| | **LUNA** (원본) | **LUNA2** |
|---|---|---|
| **목표** | PSNR / SSIM (픽셀 충실도) | **검출 mAP** (다운스트림 과제) |
| **향상기** | `LightEnhanceGenerator`: 4단계 DSConv U-Net + 경량 어텐션 | `BilateralLowLightNet`: HDRNet식 bilateral grid (저해상 계수망 + full-res guidance + slice/affine + DSConv refine) |
| **해상도** | **256×256 고정** (다운→업 왕복) | **네이티브 full resolution** |
| **학습** | 2단계: 지도학습(L1+VGG+SSIM) → **PatchGAN** 적대학습 | P1 지도학습 → **P2 검출인지(frozen YOLOv8n)** → P3 joint |
| **검출기 루프 내** | ✗ 없음 | ✓ frozen YOLOv8n 검출 손실 |
| **데이터** | LOL (485+15 쌍) | LOL v1/v2 + LoLI-Street (P1), ExDark (P2/P3) |
| **파라미터** | 205,093 (~205K) | bilateral-grid 망 (`deploy/`로 프로파일) |

**대표 결과 (DARK FACE, 6,000장, frozen YOLOv8n-face, mAP@50):**

| 방법 | mAP@50 | 원본 대비 Δ | Recall |
|---|---:|---:|---:|
| Original (원본 저조도) | 0.1132 | — | 0.1212 |
| **LUNA** (원본) | 0.0771 | **−0.0361** ❌ | 0.0870 |
| **LUNA2** | **0.1270** | **+0.0138** ✅ | 0.1365 |
| SCI (easy) | 0.1461 | +0.0329 | 0.1576 |
| Zero-DCE | 0.1968 | +0.0837 | 0.2150 |

**LUNA2 는 LUNA 의 검출 성능 하락을 반전시킨다.** 원본 LUNA 는 아무것도 안 한
것보다 mAP 를 −3.6pt *떨어뜨렸지만*, LUNA2 는 이를 **+1.4pt 상승**으로 바꿨다
(LUNA 대비 **+65% 상대 향상**). 다만 최고의 전용 저조도 모델(Zero-DCE, SCI)은
아직 못 이긴다 — 그 이유를 아래에서 데이터로 분석한다.

---

## 1. 동기 — LUNA2 가 고치려는 문제

LUNA 는 시각적으로 보기 좋고 PSNR 높은 이미지를 만들었지만, 그 향상 이미지를
*고정된* 객체 검출기에 통과시키면 **검출이 좋아지긴커녕 나빠졌다.**

**ExDark, frozen YOLOv8n(COCO), native — 원본 baseline 0.4472 대비 mAP@50:**

| 방법 (향상기) | mAP@50 | Δ |
|---|---:|---:|
| **Original (향상 없음)** | **0.4472** | — |
| SCI (easy) | 0.4523 | **+0.0051** ✅ |
| EnlightenGAN | 0.4203 | −0.0269 |
| Zero-DCE | 0.4046 | −0.0426 |
| **LUNA** (LoLI-30K) | 0.3462 | **−0.1009** |
| **LUNA** (LOL-v2) | 0.2815 | **−0.1656** |
| FUnIE-GAN | 0.1970 | −0.2501 |

이 표에서 두 가지 불편한 사실이 나온다:
1. **대부분의 "향상"은 고정 검출기를 해친다.** SCI 만 mAP 를 올리고, LUNA 를 포함한
   나머지는 전부 *떨어뜨린다.* 예쁜 픽셀 ≠ 검출 가능한 픽셀.
2. **LUNA 는 그중에서도 최악권**(−0.10 ~ −0.17). PSNR 에 강하게 맞췄기 때문이다.

이 격차가 LUNA2 의 타깃이다.

---

## 2. 진단 — 해상도가 1차 병목

모델 재설계 전, 통제된 진단을 했다: *같은* ExDark 이미지를 향상 없이 서로 다른
해상도 파이프라인(리샘플링만)으로 통과시킨다.

**ExDark, mAP@50:**

| 파이프라인 | mAP@50 | native 대비 Δ |
|---|---:|---:|
| **native (full resolution)** | **0.4472** | — |
| down → 256 → up (왕복) | 0.3992 | −0.0480 |
| down → 256 (그대로) | 0.3376 | −0.1096 |

256×256 왕복 *그 자체* — LUNA 가 내부적으로 하는 바로 그것 — 만으로 색 하나
바꾸기 전에 이미 **~5–11 mAP 손실**이다. **256px 향상기는 검출에서 절대 못 이긴다.**
다운샘플링이 검출에 필요한 작은 객체 디테일을 파괴하기 때문. 이것이 LUNA2 의
full-resolution bilateral 설계의 직접적 동기다.

---

## 3. LUNA2 가 바꾼 세 가지

### (a) 새 목표 — 픽셀이 아니라 검출기를 최적화
안정성을 위한 복원 항은 남기되 **검출인지 손실**을 추가한다: *고정된* YOLOv8n 이
향상 이미지를 GT 박스 기준으로 채점하고, 그 gradient 가 향상기로 역전파된다.
향상기는 단지 밝은 이미지가 아니라 *검출기가 좋아하는* 이미지를 학습한다.

### (b) 새 아키텍처 — full-resolution bilateral grid
`LightEnhanceGenerator`(256px U-Net)를 **`BilateralLowLightNet`**(HDRNet식)으로 교체:
- **저해상 계수망**이 픽셀별 affine 색변환의 bilateral grid 를 예측(256 에서 저비용),
- **full-res guidance 맵** + **trilinear slice** 로 *네이티브* 해상도에 적용(왕복 없음 → 작은 객체 디테일 보존),
- 경량 **DSConv refinement** 잔차로 마무리.

이는 옛 U-Net 의 재학습이 *아니라* 다른 full-res 모델이다.
(`src/models/bilateral_grid.py` 는 스텁이 아니라 완전 구현됨.)

### (c) 새 학습 프로그램 — 단계적, 검출기 루프 내
| 단계 | 내용 | 손실 | 검출기 |
|---|---|---|---|
| **P1** | bilateral 향상기 복원 사전학습 | L1 + SSIM + perceptual(VGG) | — |
| **P2** | ExDark 에서 검출인지 파인튜닝 | λ_rec·L1(anchor) + λ_det·YOLO손실 | **frozen** YOLOv8n |
| **P3** | end-to-end joint 공동적응 | 검출 + 복원 + grid-TV | **unfrozen** YOLOv8n |

(P2 는 검출 가중치 λ_det ∈ {0.001 … 0.05} 스윕; P3 는 검출 전용 / 향상기 고정 /
향상기 공동학습을 공통 예산 하에서 비교.) LUNA 의 레시피(지도학습 → PatchGAN)
대비, LUNA2 는 **GAN 을 완전히 빼고** "진짜처럼 보이기"를 "검출되기"로 바꿨다.

---

## 4. 결과 & 정직한 분석

### 4.1 DARK FACE — LUNA2 는 LUNA 를 고치지만 Zero-DCE/SCI 가 여전히 앞선다
위 대표 표 참조. LUNA2 는 검출을 돕는 *유일한* LUNA 계열 모델(+0.014)로, LUNA 의
−0.036 을 반전시킨다. 그래도 Zero-DCE(+0.084)·SCI(+0.033)가 앞선다. **왜?** 다음 두
실험이 답하며, 그 답이 진짜 연구 기여다.

### 4.2 왜 뒤처지나 — *아주 작은* 얼굴을 파괴
DARK FACE 는 극히 작은 얼굴이 지배적(중앙값 ≈ 11px; GT 5만 중 **33,892개**가 16px
미만). AP@50 을 얼굴 크기별로 분해하면:

| 얼굴 크기 | GT | Original | LUNA | **LUNA2** | Zero-DCE | SCI |
|---|---:|---:|---:|---:|---:|---:|
| tiny (<16px) | 33,892 | 0.0035 | 0.0001 | **0.0013** | **0.0098** | 0.0038 |
| small (16–32) | 12,202 | 0.2479 | 0.1227 | **0.2694** | 0.4415 | 0.3204 |
| large (≥32) | 4,302 | 0.4601 | 0.4410 | **0.5798** | 0.6937 | 0.5922 |

해석:
- **LUNA2 는 큰 얼굴에서 확실히 강하다**(0.46 → 0.58, +0.12). full-res 공간 처리가
  중·대형 객체를 *복원*한다.
- 그러나 **tiny 얼굴에선 거의 못 살린다**(0.0035 → 0.0013). DARK FACE 는 *대부분*
  tiny 라서 전체 평균이 끌려 내려간다.
- **Zero-DCE 가 이기는 이유는 픽셀별 곡선**이기 때문 — 균일하게 밝히고 tiny 얼굴의
  구조를 온전히 둔다(tiny AP 0.0035 → 0.0098, recall ~3배).
- **LUNA(원본)는 전 구간 붕괴** — tiny(→0.0001)·small 얼굴을 적극적으로 지운다.

즉 LUNA2 의 bilateral/공간 처리는 양날의 검 — 구조가 있는 객체엔 훌륭하지만 16px
미만 얼굴엔 여전히 너무 공격적이다.

### 4.3 더 큰 단서 — 향상은 *도메인 불일치* 검출기에만 도움
이어서 얼굴 검출기를 저조도 도메인에 파인튜닝하고 held-out val 만 재평가(누수 없음):

**DARK FACE val (1,200장), mAP@50:**

| 검출기 | Original (원본) | LUNA2 | Zero-DCE |
|---|---:|---:|---:|
| **고정** (WIDER-FACE) | 0.1057 | **0.1221** (+0.016) | — |
| **적응** (저조도 파인튜닝) | **0.3747** | 0.1772 (−0.197) | 0.3196 (−0.055) |

가장 중요하고 가장 냉정한 발견:
- **고정·도메인 불일치** 검출기에서는 향상이 돕는다(LUNA2 +0.016).
- 그러나 검출기가 **저조도에 적응**하면 **원본 이미지가 압승**(0.375)하고, *모든*
  향상이 *해를 끼친다* — 이미지를 가장 많이 바꾸는 LUNA2 가 가장 크게(−0.197).

**결론:** 검출을 위한 저조도 향상의 가치는 **검출기 도메인 불일치에 조건부**다.
검출기를 파인튜닝할 수 있으면 그것부터 하라. LUNA2 는 검출기가 고정/기성품인 현실적
상황에서 돕지만, 검출기 적응의 대체재는 아니다. 이 정직한 부정적 결과가 다음 연구
단계(방법별 검출기 공동적응, P3)를 규정한다.

---

## 5. 타 모델과의 비교 (요약)

| 모델 | 계열 | ExDark Δ mAP@50¹ | DARK FACE mAP@50² | 비고 |
|---|---|---:|---:|---|
| Original (없음) | — | 0.0000 | 0.1132 | 원본 baseline |
| **LUNA** | GAN, 256px, PSNR | −0.1009 / −0.1656 | 0.0771 | 픽셀 최적화 → 검출 악화 |
| **LUNA2** | bilateral grid, full-res, 검출인지 | (§4.3 참조) | **0.1270** | LUNA 반전; 계열 내 최고 |
| Zero-DCE | 픽셀별 곡선 | −0.0426 | **0.1968** | tiny 얼굴 최강, 검출 1위 |
| SCI | self-calibrated | **+0.0051** | 0.1461 | 고정 ExDark 를 돕는 유일 방법 |
| EnlightenGAN | unpaired GAN | −0.0269 | — | 검출 소폭 하락 |
| FUnIE-GAN | 수중 GAN | −0.2501 | — | 검출 큰 하락(도메인 불일치) |

¹ ExDark, frozen YOLOv8n(COCO), native, 0.4472 대비. ² DARK FACE, 6,000장,
frozen YOLOv8n-face, conf 0.25.

**정리:** 단순 픽셀별 방법(Zero-DCE, SCI)은 구조를 보존하므로 *고정 검출기* 저조도
검출에서 이기기 매우 어렵다. 무거운 생성 모델(LUNA, FUnIE, EnlightenGAN)은
환각·평활화로 오히려 해친다. LUNA2 의 기여는 full-res·검출인지 재설계가 LUNA 계열을
"명백히 해로움"에서 "순이득"으로 옮겼음을 보이고, 남은 격차(tiny 객체 + 검출기 적응)를
정량화한 것이다.

---

## 6. 디렉터리 구조 / 7. 재현

영어 본문의 [§6 Repository structure](#6-repository-structure) 와
[§7 Reproducing the numbers](#7-reproducing-the-numbers) 와 동일하다. 핵심만:

- 경로는 코드에 하드코딩하지 않고 [`configs/paths.yaml`](configs/paths.yaml) 한 곳에
  정의 → `root` 한 줄만 고치면 머신 이동 가능. 데이터·가중치·체크포인트는 **복사하지
  않고 참조**하며 git 에서 제외(`runs/`, `*.pth`, `*.pt`, `*.onnx`).
- 설치: `pip install -r requirements.txt` (PyTorch 는 CUDA 버전에 맞춰 별도 설치).
- 일괄 실행 가이드: [`RUN_DARKFACE.md`](RUN_DARKFACE.md),
  [`RUN_EXTRA_EXPERIMENTS.md`](RUN_EXTRA_EXPERIMENTS.md).
- 본 README 의 수치는 위 스크립트가 `runs/eval_*` 에 생성한 CSV (frozen
  YOLOv8n / YOLOv8n-face, conf 0.25) 에서 나온 것이며, git 제외 산출물이라 위
  명령으로 재생성한다.

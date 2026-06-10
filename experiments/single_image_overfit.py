"""[A] single-image overfit — train.py 완전 배제, 최소 자체 구현.

목적
----
BilateralLowLightNet(identity prior 적용본) 이 LOL 1쌍을 과적합할 수 있는지로
"아키텍처/학습 가능성" 과 "train.py 파이프라인 버그" 를 분리 판정한다.

규칙 (사용자 지시)
------------------
* train.py 를 import/재사용하지 않는다. 모델 클래스만 import.
* loss 는 인라인 L1 = (out-target).abs().mean(). (CombinedRestorationLoss 금지)
* optimizer Adam lr=1e-3, AMP off, scheduler off, batch=1, fp32, 1000 step.
* 데이터는 현 파이프라인 컨벤션과 동일하게 [-1,1] 정규화 (PIL→256→[0,1]→*2-1).

이 스크립트는 메모리상 임시 모델만 학습하며 체크포인트/기존 산출물을 건드리지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# 모델 클래스만 import (train.py / losses / dataset 모듈 일절 미사용)
from src.models.bilateral_grid import BilateralLowLightNet
from src.utils.paths import load_paths

IMAGE_SIZE = 256
N_STEPS = 1000
LOG_STEPS = {0, 1, 10, 50, 100, 200, 500, 1000}
LOG_EVERY = 50  # 그 외에도 50 step 마다
HR = "=" * 96
SUB = "-" * 96


# --- 데이터 로드 (인라인, [-1,1] 정규화 = 현 파이프라인 컨벤션) --------------
def load_pair_norm(low_path: Path, high_path: Path, size: int, device: str):
    def _load(p: Path) -> torch.Tensor:
        img = Image.open(p).convert("RGB")
        img = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR)
        t = TF.to_tensor(img) * 2.0 - 1.0          # [0,1] → [-1,1]
        return t.unsqueeze(0).to(device)            # (1,3,H,W)
    return _load(low_path), _load(high_path)


def find_first_lol_train_pair(lol_root: Path):
    """LOL v1 our485/{low,high} 에서 파일명 일치 1쌍 (사전순 첫 번째)."""
    low_dir = lol_root / "our485" / "low"
    high_dir = lol_root / "our485" / "high"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    highs = {p.stem: p for p in high_dir.iterdir()
             if p.is_file() and p.suffix.lower() in exts}
    for lp in sorted(low_dir.iterdir()):
        if lp.is_file() and lp.suffix.lower() in exts and lp.stem in highs:
            return lp, highs[lp.stem]
    raise FileNotFoundError(f"LOL train 쌍을 찾지 못함: {low_dir}")


# --- metric: PSNR ([-1,1]→[0,1], data_range 1.0) ----------------------------
def psnr_pm1(out: torch.Tensor, target: torch.Tensor) -> float:
    o = ((out + 1.0) * 0.5).clamp(0.0, 1.0)
    t = ((target + 1.0) * 0.5).clamp(0.0, 1.0)
    mse = (o - t).pow(2).mean().clamp(min=1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


# --- sliced affine scale 대각 / bias 통계 (모델 내부 경로 재사용) -----------
@torch.no_grad()
def affine_stats(model: BilateralLowLightNet, x: torch.Tensor):
    inter = model.intermediate(x)            # grid/guidance/coeffs/affine_out/out
    coeffs = inter["coeffs"]                  # (1,12,H,W)
    B, _, H, W = coeffs.shape
    A = coeffs.view(B, 3, 4, H, W)
    # 주의: apply_affine 의 identity prior(+1)는 여기 raw coeffs 에는 미반영 →
    # 유효 scale 대각 = raw 대각 + 1.
    diag_eff = torch.stack([A[:, 0, 0] + 1.0, A[:, 1, 1] + 1.0, A[:, 2, 2] + 1.0])
    bias = A[:, :, 3]
    return float(diag_eff.mean()), float(bias.mean())


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    P = load_paths()

    low_p, high_p = find_first_lol_train_pair(P.lol_v1)
    low, high = load_pair_norm(low_p, high_p, IMAGE_SIZE, device)

    print(HR)
    print(" [A] single-image overfit (train.py 미사용, 인라인 L1, Adam lr=1e-3, fp32, 1000 step)")
    print(HR)
    print(f"  device   : {device}")
    print(f"  pair     : {low_p.name} ↔ {high_p.name}  (LOL v1 our485)")
    print(f"  input    : {tuple(low.shape)}  range[{low.min():.3f},{low.max():.3f}]  mean {low.mean():.3f}")
    print(f"  baseline PSNR(low vs high) = {psnr_pm1(low, high):.3f} dB")
    print(SUB)

    # 모델: random init + identity prior (apply_affine 내장)
    model = BilateralLowLightNet().to(device).train()
    to_grid = model.coefficient_net.to_grid   # 마지막 grid head layer

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # AMP off, scheduler off

    # step0 기준 스냅샷
    with torch.no_grad():
        out0 = model(low).detach().clone()
        w0 = to_grid.weight.detach().clone()
        b0 = to_grid.bias.detach().clone() if to_grid.bias is not None else None
        sc0, bi0 = affine_stats(model, low)

    print(f"  {'step':>5} | {'L1':>8} | {'PSNR':>7} | {'out_min':>7} {'out_max':>7} "
          f"{'out_mean':>8} {'out_std':>7} | {'|out-in|':>8} {'Δout_L2':>8} | "
          f"{'Δw_L2':>9} {'gradN':>9} | {'scDiag':>7} {'Δsc':>7} {'bias':>7} {'Δbias':>7} | "
          f"{'Δparam_step':>11}")
    print(SUB)

    for step in range(0, N_STEPS + 1):
        # ---- forward (batch=1, fp32) ----
        out = model(low)
        loss = (out - high).abs().mean()      # 인라인 L1

        do_log = (step in LOG_STEPS) or (step % LOG_EVERY == 0)

        if step == 0:
            # step0: 학습 전 상태만 로깅 (업데이트 없음)
            gradN = 0.0
            dparam_step = 0.0
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradN = float(sum(p.grad.detach().pow(2).sum()
                              for p in to_grid.parameters() if p.grad is not None) ** 0.5)
            # optimizer.step() 직전/직후 to_grid param L2 차이 (업데이트 실제 반영 여부)
            pre = torch.cat([p.detach().flatten() for p in to_grid.parameters()]).clone()
            optimizer.step()
            post = torch.cat([p.detach().flatten() for p in to_grid.parameters()])
            dparam_step = float((post - pre).norm())

        if do_log:
            with torch.no_grad():
                o = model(low)
                ps = psnr_pm1(o, high)
                dout = float((o - out0).norm())
                dwin = float((o - low).abs().mean())
                dw = float((to_grid.weight.detach() - w0).norm())
                if b0 is not None:
                    dw = float(((to_grid.weight.detach() - w0).pow(2).sum()
                                + (to_grid.bias.detach() - b0).pow(2).sum()) ** 0.5)
                sc, bi = affine_stats(model, low)
                print(f"  {step:>5} | {loss.item():>8.5f} | {ps:>7.3f} | "
                      f"{o.min():>7.3f} {o.max():>7.3f} {o.mean():>8.3f} {o.std():>7.3f} | "
                      f"{dwin:>8.4f} {dout:>8.4f} | {dw:>9.4f} {gradN:>9.3e} | "
                      f"{sc:>7.3f} {sc-sc0:>+7.3f} {bi:>7.3f} {bi-bi0:>+7.3f} | "
                      f"{dparam_step:>11.3e}")

    print(SUB)
    with torch.no_grad():
        final_ps = psnr_pm1(model(low), high)
        final_dw = float((to_grid.weight.detach() - w0).norm())
    print(f"  최종 PSNR = {final_ps:.3f} dB   |   to_grid weight 총 이동 ‖Δw‖ = {final_dw:.4f}")
    print(SUB)
    print(" 판정 가이드:")
    print("  · PSNR 25~30dB+ 상승        → 아키텍처 정상, 버그는 train.py 파이프라인")
    print("  · 7~12dB 고정 + Δparam≈0     → optimizer/update 미반영 or 구조 단절")
    print("  · param 변함 + affine/out 불변 → slice/apply/guidance 경로 문제")
    print("  · loss↓ 인데 PSNR 이상       → metric/range 문제")
    # 자동 1차 판정
    base = psnr_pm1(low, high)
    if final_ps >= 25:
        verd = "아키텍처 정상 → 버그는 train.py 파이프라인 (모델/identity prior 아님)"
    elif final_ps - base < 5 and final_dw < 1e-4:
        verd = "optimizer/update 미반영 (param 안 움직임)"
    elif final_ps - base < 5:
        verd = "param 은 변하나 출력 개선 없음 → slice/apply/guidance 경로"
    else:
        verd = "부분 개선 — 추가 분석 필요"
    print(SUB)
    print(f"  → 자동 판정: {verd}")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

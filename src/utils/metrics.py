"""평가 지표 — PSNR, SSIM, LPIPS 및 데이터셋 단위 evaluation 헬퍼 (LUNA2 이식본).

출처 (Provenance)
-----------------
``SmallSizePM_GAN_model/CODES/utils/metrics.py`` 의 PSNR/SSIM/LPIPS/evaluate 를
**그대로 이식**. 원본과 동일한 PSNR/SSIM 수치를 재현하기 위함이다.

원본의 ``benchmark_model_full`` (params/MACs/FPS 종합) 은 ``model_analysis`` 모듈
의존성이 있어 이식하지 않았다. 해당 프로파일링은 LUNA2 의 ``src/utils/profile.py``
(예정) 에서 새로 다룬다.

설계 메모
---------
* 입력 텐서 범위는 ``[-1, 1]`` 가정. 내부에서 ``[0, 1]`` 로 변환 후
  ``data_range=1.0`` 으로 계산 → 학술 문헌과 동일한 PSNR/SSIM.
* SSIM 은 Wang et al. 2004 single-scale, multi-channel 평균.
"""
from __future__ import annotations

import contextlib
import io as _io
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1] (clamp)."""
    return ((x + 1.0) * 0.5).clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
@torch.no_grad()
def psnr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> float:
    """PSNR (dB). 입력 ``(B, C, H, W)`` ∈ ``[-1, 1]``."""
    p = _to_01(pred)
    t = _to_01(target)
    mse = F.mse_loss(p, t, reduction="none").mean(dim=(1, 2, 3))  # (B,)
    mse = mse.clamp(min=1e-10)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    if reduction == "mean":
        return float(psnr.mean().item())
    return psnr  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _make_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    k1 = _gaussian_kernel(window_size, sigma)
    k2 = k1.unsqueeze(0) * k1.unsqueeze(1)
    return k2.expand(channels, 1, window_size, window_size).contiguous()


_SSIM_WINDOW_CACHE: Dict[tuple, torch.Tensor] = {}


def _get_window(window_size: int, sigma: float, channels: int,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (window_size, sigma, channels, device, dtype)
    w = _SSIM_WINDOW_CACHE.get(key)
    if w is None:
        w = _make_window(window_size, sigma, channels).to(device=device, dtype=dtype)
        _SSIM_WINDOW_CACHE[key] = w
    return w


@torch.no_grad()
def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Mean SSIM over batch & channels (single-scale, Gaussian window)."""
    p = _to_01(pred)
    t = _to_01(target)
    B, C, H, W = p.shape
    window = _get_window(window_size, sigma, C, p.device, p.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(p, window, padding=pad, groups=C)
    mu_t = F.conv2d(t, window, padding=pad, groups=C)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sig_p2 = F.conv2d(p * p, window, padding=pad, groups=C) - mu_p2
    sig_t2 = F.conv2d(t * t, window, padding=pad, groups=C) - mu_t2
    sig_pt = F.conv2d(p * t, window, padding=pad, groups=C) - mu_pt

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
    return float((num / den).mean().item())


# ---------------------------------------------------------------------------
# LPIPS — Learned Perceptual Image Patch Similarity (Zhang et al., 2018)
# ---------------------------------------------------------------------------
class LPIPSEvaluator:
    """LPIPS metric (lazy init, AlexNet backbone by default).

    입력 가정: 두 텐서 모두 ``[-1, 1]`` 범위. lpips 미설치 시 ``available=False``,
    ``__call__`` 은 ``float('nan')`` 반환.
    """

    def __init__(self, net: str = "alex"):
        self.net = net
        self._model: Optional[nn.Module] = None
        self.available = True
        try:
            import lpips  # noqa: F401
        except ImportError:
            self.available = False

    def _build(self, device: torch.device) -> nn.Module:
        import lpips
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = lpips.LPIPS(net=self.net, verbose=False)
        for p in m.parameters():
            p.requires_grad = False
        return m.to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not self.available:
            return float("nan")
        device = pred.device
        if self._model is None:
            self._model = self._build(device)
        elif next(self._model.parameters()).device != device:
            self._model = self._model.to(device)
        d = self._model(pred, target)
        return float(d.mean().item())


_LPIPS_CACHE: Dict[str, LPIPSEvaluator] = {}


def get_lpips_evaluator(net: str = "alex") -> LPIPSEvaluator:
    if net not in _LPIPS_CACHE:
        _LPIPS_CACHE[net] = LPIPSEvaluator(net=net)
    return _LPIPS_CACHE[net]


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    generator: torch.nn.Module,
    loader,  # torch.utils.data.DataLoader
    device: str = "cuda",
    compute_lpips: bool = False,
    lpips_net: str = "alex",
) -> Dict[str, float]:
    """eval loader 전체에 대해 PSNR / SSIM (옵션 LPIPS) 평균.

    Returns
    -------
    dict
        ``{"psnr": float, "ssim": float, "n": int}``, ``compute_lpips=True``
        이면 ``"lpips"`` 추가.
    """
    was_training = generator.training
    generator.eval()

    lpips_eval: Optional[LPIPSEvaluator] = None
    if compute_lpips:
        lpips_eval = get_lpips_evaluator(net=lpips_net)
        if not lpips_eval.available:
            print("[evaluate] WARN: lpips 라이브러리 미설치 — LPIPS = NaN")

    from .inference import guard_output  # 추론 공용 가드 (nan_to_num + clamp)

    psnr_sum, ssim_sum, lpips_sum, n = 0.0, 0.0, 0.0, 0
    for low, high in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        # validation 은 autocast 없이 fp32 forward (evaluate 는 autocast 미사용).
        # 출력 가드로 NaN/Inf 를 제거해 PSNR/SSIM 오염 방지.
        fake = guard_output(generator(low))
        bs = low.size(0)
        psnr_sum += psnr_metric(fake, high) * bs
        ssim_sum += ssim_metric(fake, high) * bs
        if lpips_eval is not None and lpips_eval.available:
            lpips_sum += lpips_eval(fake, high) * bs
        n += bs

    if was_training:
        generator.train()

    if n == 0:
        out = {"psnr": 0.0, "ssim": 0.0, "n": 0}
        if compute_lpips:
            out["lpips"] = float("nan")
        return out

    out = {"psnr": psnr_sum / n, "ssim": ssim_sum / n, "n": n}
    if compute_lpips:
        out["lpips"] = lpips_sum / n if lpips_eval and lpips_eval.available else float("nan")
    return out

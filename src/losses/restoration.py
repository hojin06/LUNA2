"""복원(restoration) 손실 — L1 / VGG perceptual / SSIM 조합 (Phase 1 지도학습).

출처 / 동일성 (Provenance)
--------------------------
``SmallSizePM_GAN_model/CODES/models/losses.py`` 의 ``SupervisedLoss`` 구성을
**그대로 미러링**하여 동일 baseline 을 재현한다:

* **L1**         : ``nn.L1Loss`` (pixel).
* **Perceptual** : VGG16 ``features[:16]`` = **relu3_3** feature-space L1.
                   입력 [-1,1] → [0,1] → ImageNet normalize. VGG freeze+eval.
* **SSIM**       : window 11, σ 1.5, 3채널, data_range 1.0. ``1 - mean_ssim``.
* **가중치**     : λ_L1 = 1.0, λ_VGG = 0.5, λ_SSIM = 1.0 (원본 SupervisedLoss 동일).

원본은 perceptual 을 relu3_3 단일 레이어로 쓴다 (``features[:16]``). 본 구현은
기본값으로 그것을 정확히 재현하되, ``layers`` 인자로 multi-layer (relu1_2/2_2/3_3)
확장도 지원한다.

모듈러 구조 (Phase 2 대비)
--------------------------
``CombinedRestorationLoss`` 는 forward 에서 항목별 dict 를 반환하고, ``add_term``
으로 추가 손실(예: detection-aware, see [[detection_aware]])을 등록할 수 있어
Phase 2 joint training 에서 그대로 확장된다.

입력 규약: 모든 텐서 ``[-1, 1]``, ``(B, 3, H, W)``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VGG16 perceptual loss
# ---------------------------------------------------------------------------
# VGG16 features 의 ReLU 출력 인덱스 (relu_x_y → features 내 index).
_VGG_RELU_INDEX = {
    "relu1_1": 1, "relu1_2": 3,
    "relu2_1": 6, "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15,
    "relu4_1": 18, "relu4_2": 20, "relu4_3": 22,
}


class PerceptualLoss(nn.Module):
    """VGG16 feature-space L1 perceptual loss.

    Parameters
    ----------
    layers : Sequence[str]
        추출할 ReLU 레이어 이름. 기본 ``("relu3_3",)`` → 원본 LUNA 와 동일
        (``vgg.features[:16]`` 출력 L1).
    layer_weights : Sequence[float] | None
        레이어별 가중치. None 이면 모두 1.0.

    입력 가정: ``[-1, 1]``. 내부에서 ``[0, 1]`` → ImageNet normalize 후 VGG 통과.
    VGG 파라미터는 freeze (``requires_grad=False``) + eval.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("relu3_3",),
        layer_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        from torchvision.models import vgg16

        try:
            from torchvision.models import VGG16_Weights  # type: ignore
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                vgg = vgg16(pretrained=True)
            except Exception:
                vgg = vgg16(weights=None)
                print("[PerceptualLoss] WARNING: VGG16 pretrained weights 미가용 "
                      "— perceptual loss 가 무의미할 수 있음 (random init).")

        for name in layers:
            if name not in _VGG_RELU_INDEX:
                raise ValueError(f"알 수 없는 VGG 레이어 '{name}'. "
                                 f"가능: {list(_VGG_RELU_INDEX)}")
        self.layers = tuple(layers)
        self.layer_indices = [_VGG_RELU_INDEX[n] for n in layers]
        if layer_weights is None:
            layer_weights = [1.0] * len(layers)
        self.layer_weights = list(layer_weights)

        # 필요한 최대 index 까지만 보관 (원본 relu3_3 → features[:16])
        max_idx = max(self.layer_indices)
        features = vgg.features[: max_idx + 1].eval()
        for p in features.parameters():
            p.requires_grad = False
        self.features = features

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [0, 1] → ImageNet normalize."""
        x01 = (x + 1.0) * 0.5
        return (x01 - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """요청 레이어들의 feature 리스트를 순차 forward 로 추출."""
        feats: List[torch.Tensor] = []
        want = set(self.layer_indices)
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                feats.append(h)
            if i >= max(self.layer_indices):
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self._extract(self._prepare(pred))
        ft = self._extract(self._prepare(target))
        loss = pred.new_zeros(())
        for w, a, b in zip(self.layer_weights, fp, ft):
            loss = loss + w * F.l1_loss(a, b)
        return loss


# ---------------------------------------------------------------------------
# SSIM loss (원본 SSIMLoss 미러링)
# ---------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """``1 - mean_SSIM`` (Wang et al., 2004). single-scale, multi-channel.

    입력 ``[-1, 1]`` → 내부 ``[0, 1]`` 변환. window 11, σ 1.5, data_range 1.0.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 3,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range

        kernel_1d = self._gaussian_window(window_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel = kernel_2d.expand(channels, 1, window_size, window_size)
        self.register_buffer("window", kernel.contiguous())

        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        groups = self.channels
        w = self.window.to(dtype=x.dtype)

        mu_x = F.conv2d(x, w, padding=pad, groups=groups)
        mu_y = F.conv2d(y, w, padding=pad, groups=groups)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, w, padding=pad, groups=groups) - mu_x2
        sigma_y2 = F.conv2d(y * y, w, padding=pad, groups=groups) - mu_y2
        sigma_xy = F.conv2d(x * y, w, padding=pad, groups=groups) - mu_xy

        num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim((pred + 1.0) * 0.5, (target + 1.0) * 0.5)


# ---------------------------------------------------------------------------
# Combined restoration loss (Phase 1) — Phase 2 확장 가능 모듈러 구조
# ---------------------------------------------------------------------------
# 추가 손실 항 시그니처: fn(pred, target, **ctx) -> scalar Tensor
ExtraLossFn = Callable[..., torch.Tensor]


class CombinedRestorationLoss(nn.Module):
    """L_total = λ_L1·L1 + λ_VGG·VGG + λ_SSIM·SSIM  (+ 등록된 추가 항).

    기본 가중치는 원본 LUNA ``SupervisedLoss`` 와 동일(1.0 / 0.5 / 1.0).

    Phase 2 확장
    ------------
    ``add_term(name, fn, weight)`` 로 detection-aware 등 추가 손실을 등록하면
    forward 가 ``λ·fn(pred, target, **ctx)`` 를 합산하고 dict 에 항목을 추가한다.
    forward(..., **ctx) 의 ctx 는 추가 항으로 그대로 전달된다 (예: YOLO target).
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.5,
        lambda_ssim: float = 1.0,
        use_perceptual: bool = True,
        perceptual_layers: Sequence[str] = ("relu3_3",),
    ) -> None:
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_vgg = float(lambda_vgg)
        self.lambda_ssim = float(lambda_ssim)
        self.use_perceptual = use_perceptual

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual: Optional[PerceptualLoss] = (
            PerceptualLoss(layers=perceptual_layers) if use_perceptual else None
        )

        # 추가 손실 항: name -> (weight, fn)
        self._extra: Dict[str, Tuple[float, ExtraLossFn]] = {}

    # ------------------------------------------------------------------
    def add_term(self, name: str, fn: ExtraLossFn, weight: float = 1.0) -> None:
        """Phase 2 용 추가 손실 항 등록 (예: detection-aware)."""
        if name in ("total", "l1", "vgg", "ssim"):
            raise ValueError(f"예약된 이름은 사용할 수 없습니다: '{name}'")
        self._extra[name] = (float(weight), fn)

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **ctx,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict: ``total`` + 항목별 detach 모니터링 값.

        가중치가 0 인 항은 **계산을 생략**한다 (가드). lambda=0 이어도 항을
        계산하면 그 항의 NaN/Inf gradient 가 ``0×NaN=NaN`` 으로 total 을
        오염시키므로(특히 fp16 SSIM/VGG), 살아있는 항만 그래프에 포함한다.
        """
        zero = pred.new_zeros(())

        l_l1 = self.l1(pred, target) if self.lambda_l1 > 0 else zero
        l_ssim = self.ssim(pred, target) if self.lambda_ssim > 0 else zero
        if self.perceptual is not None and self.lambda_vgg > 0:
            l_vgg = self.perceptual(pred, target)
        else:
            l_vgg = zero

        total = (self.lambda_l1 * l_l1
                 + self.lambda_vgg * l_vgg
                 + self.lambda_ssim * l_ssim)

        out: Dict[str, torch.Tensor] = {
            "l1": l_l1.detach(),
            "vgg": l_vgg.detach(),
            "ssim": l_ssim.detach(),
        }
        for name, (w, fn) in self._extra.items():
            term = fn(pred, target, **ctx)
            total = total + w * term
            out[name] = term.detach()

        out["total"] = total
        return out


def build_restoration_loss(cfg: dict) -> CombinedRestorationLoss:
    """``configs/bilateral_base.yaml`` 의 ``loss`` 블록 → 손실 인스턴스."""
    lc = cfg.get("loss", cfg)
    return CombinedRestorationLoss(
        lambda_l1=lc.get("lambda_l1", 1.0),
        lambda_vgg=lc.get("lambda_vgg", 0.5),
        lambda_ssim=lc.get("lambda_ssim", 1.0),
        use_perceptual=lc.get("use_perceptual", True),
        perceptual_layers=tuple(lc.get("perceptual_layers", ("relu3_3",))),
    )

"""LUNA base generator — 원본 LightEnhanceGenerator 이식본 (LUNA2).

출처 (Provenance)
-----------------
이 파일은 ``SmallSizePM_GAN_model/CODES/models/generator.py`` 의
``LightEnhanceGenerator`` 정의를 **그대로 이식**한 것이다. LUNA2 후속 연구
(bilateral grid + detection-aware joint training) 의 baseline / 비교군으로
원본과 동일한 결과를 재현하기 위해 모델 정의는 한 글자도 바꾸지 않는다.

추가된 것은 원본 ``train_hybrid_v1_final.py`` / ``downstream_detection.py`` 에
흩어져 있던 **체크포인트 로드 + 전처리 헬퍼**를 한곳에 모은 것뿐이며, 이들도
원본 로직과 동일하다 (hybrid_v1 사양 상수 포함).

원본과 동일성 확인 방법
-----------------------
``experiments/eval_restoration.py`` (PSNR/SSIM) 와 ``experiments/eval_detection.py``
(ExDark mAP) 를 ``configs/paths.yaml`` 의 ``luna_original`` 체크포인트로 실행하면
원본 ``downstream_exdark.py`` / hybrid_v1 평가와 동일 수치가 나와야 한다.

채널 구성 (base_filters=32): 3 → 32 → 64 → 128 → 256 → … → 3.
입출력 범위: [-1, 1] (Tanh).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


# ===========================================================================
# Hybrid v1 사양 (원본 train_hybrid_v1_final.py 에서 이식 — 절대 변경 금지)
# ===========================================================================
HYBRID_V1_CONV_CONFIG: Dict[str, str] = {
    "input_conv": "standard",
    "enc1":       "dsconv",
    "enc2":       "dsconv",
    "enc3":       "dsconv",
    "bottleneck": "dsconv",
    "dec3":       "dsconv",
    "dec2":       "dsconv",
    "dec1":       "dsconv",
}
HYBRID_V1_BASE_FILTERS = 32
HYBRID_V1_USE_ATTENTION = True


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DSConv(nn.Module):
    """Depthwise Separable Convolution block.

    구조: ``DW-Conv3x3 → BN → ReLU → PW-Conv1x1 → BN → ReLU``.
    표준 3×3 conv 대비 MAC 절감 비율 ≈ ``1/c_out + 1/9`` (MobileNetV1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=not use_bn,
        )
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class ConvBlock(nn.Module):
    """표준 Conv 블록 — DSConv 의 ablation 대조군 (use_dsconv=False).

    구조: ``Conv3x3 → BN → ReLU``. DSConv 와 동일한 인터페이스를 유지하므로
    architecture 그대로 두고 블록만 swap 가능.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel Attention Module (CBAM, Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM, Woo et al., 2018)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel_size must be 3, 5, or 7"
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class LightAttention(nn.Module):
    """Lightweight Attention = ChannelAttention → SpatialAttention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class LightEnhanceGenerator(nn.Module):
    """저조도 이미지 향상용 4-stage DSConv U-Net Generator (원본 이식).

    ``conv_config=None`` 이면 legacy 8-블록 layout, dict 면 hybrid 9-블록
    layout. 원본과 동일하게 두 layout 모두 지원하여 기존 체크포인트와 호환된다.
    """

    HYBRID_BLOCK_NAMES = (
        "input_conv", "enc1", "enc2", "enc3", "bottleneck",
        "dec3", "dec2", "dec1",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 32,
        use_attention: bool = True,
        use_dsconv: bool = True,
        conv_config: "Optional[Dict[str, str]]" = None,
    ) -> None:
        super().__init__()
        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4
        c4 = base_filters * 8

        self._base_filters = base_filters
        self._use_attention = use_attention
        self._use_dsconv = use_dsconv
        self._conv_config = conv_config
        self._layout = "hybrid" if conv_config is not None else "legacy"

        if conv_config is None:
            block = DSConv if use_dsconv else ConvBlock

            self.enc1 = block(in_channels, c1, stride=1)  # 256, skip1
            self.enc2 = block(c1, c2, stride=2)           # 128, skip2
            self.enc3 = block(c2, c3, stride=2)           # 64 , skip3
            self.enc4 = block(c3, c4, stride=2)           # 32 , bottleneck

            self.attn = LightAttention(c4) if use_attention else nn.Identity()

            self.dec3 = block(c4 + c3, c3, stride=1)
            self.dec2 = block(c3 + c2, c2, stride=1)
            self.dec1 = block(c2 + c1, c1, stride=1)

            self.out_proj = nn.Conv2d(c1, out_channels, kernel_size=1)
        else:
            self._validate_conv_config(conv_config)

            def _block(name: str, c_in: int, c_out: int, stride: int = 1) -> nn.Module:
                kind = conv_config.get(name, "dsconv")
                if kind == "dsconv":
                    return DSConv(c_in, c_out, stride=stride)
                return ConvBlock(c_in, c_out, stride=stride)

            self.input_conv = _block("input_conv", in_channels, c1, stride=1)  # 256
            self.enc1       = _block("enc1", c1, c2, stride=2)                 # 128
            self.enc2       = _block("enc2", c2, c3, stride=2)                 # 64
            self.enc3       = _block("enc3", c3, c4, stride=2)                 # 32
            self.bottleneck = _block("bottleneck", c4, c4, stride=1)           # 32

            self.attn = LightAttention(c4) if use_attention else nn.Identity()

            self.dec3 = _block("dec3", c4 + c3, c3, stride=1)
            self.dec2 = _block("dec2", c3 + c2, c2, stride=1)
            self.dec1 = _block("dec1", c2 + c1, c1, stride=1)

            self.output_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    @classmethod
    def _validate_conv_config(cls, cfg: Dict[str, str]) -> None:
        missing = [k for k in cls.HYBRID_BLOCK_NAMES if k not in cfg]
        if missing:
            raise ValueError(
                f"conv_config 누락 키: {missing}.  필요한 키: {list(cls.HYBRID_BLOCK_NAMES)}"
            )
        for name, kind in cfg.items():
            if name not in cls.HYBRID_BLOCK_NAMES:
                raise ValueError(f"conv_config 의 알 수 없는 블록 이름: '{name}'")
            if kind not in ("dsconv", "standard"):
                raise ValueError(
                    f"conv_config['{name}'] 는 'dsconv' 또는 'standard' 여야 합니다 (got '{kind}')"
                )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming initialization (저조도 입력의 작은 dynamic range 보완)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        """Bilinear ×2 up-sampling (transpose-conv 대비 체커보드 artefact 없음)."""
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._layout == "legacy":
            return self._forward_legacy(x)
        return self._forward_hybrid(x)

    def _forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)   # c1 × 256
        s2 = self.enc2(s1)  # c2 × 128
        s3 = self.enc3(s2)  # c3 × 64
        s4 = self.enc4(s3)  # c4 × 32
        b = self.attn(s4)
        u3 = self.dec3(torch.cat([self._up(b), s3], dim=1))
        u2 = self.dec2(torch.cat([self._up(u3), s2], dim=1))
        u1 = self.dec1(torch.cat([self._up(u2), s1], dim=1))
        return torch.tanh(self.out_proj(u1))

    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.input_conv(x)   # c1 × 256, skip → dec1
        s1 = self.enc1(s0)        # c2 × 128, skip → dec2
        s2 = self.enc2(s1)        # c3 × 64 , skip → dec3
        s3 = self.enc3(s2)        # c4 × 32
        b = self.bottleneck(s3)   # c4 × 32
        b = self.attn(b)          # c4 × 32
        u3 = self.dec3(torch.cat([self._up(b),  s2], dim=1))  # c3 × 64
        u2 = self.dec2(torch.cat([self._up(u3), s1], dim=1))  # c2 × 128
        u1 = self.dec1(torch.cat([self._up(u2), s0], dim=1))  # c1 × 256
        return torch.tanh(self.output_conv(u1))


# ===========================================================================
# 체크포인트 로드 (원본 downstream_detection.load_luna_generator 이식)
# ===========================================================================
def load_luna_generator(
    ckpt_path: Path | str,
    device: str = "cuda",
) -> LightEnhanceGenerator:
    """체크포인트로부터 hybrid_v1 LUNA generator 재구성 + weight 로드.

    체크포인트 내부에 ``conv_config`` / ``base_filters`` / ``use_attention`` 이
    있으면 그 값을 우선 사용하고, 없으면 hybrid_v1 상수를 쓴다. 원본
    ``downstream_detection.load_luna_generator`` 와 동일 동작.
    """
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    bf = HYBRID_V1_BASE_FILTERS
    use_attn = HYBRID_V1_USE_ATTENTION
    conv_cfg: Dict[str, str] = HYBRID_V1_CONV_CONFIG.copy()
    if isinstance(state, dict):
        bf = int(state.get("base_filters", bf))
        use_attn = bool(state.get("use_attention", use_attn))
        if "conv_config" in state and isinstance(state["conv_config"], dict):
            conv_cfg = dict(state["conv_config"])

    G = LightEnhanceGenerator(
        base_filters=bf,
        use_attention=use_attn,
        conv_config=conv_cfg,
    ).to(device)

    sd = state["generator"] if isinstance(state, dict) and "generator" in state else state
    G.load_state_dict(sd)
    G.eval()
    return G


# ===========================================================================
# 전처리 / 후처리 헬퍼 (원본 downstream_detection 이식 — 평가 동일성 보장)
# ===========================================================================
def pil_to_norm_tensor(img: Image.Image, image_size: int = 256) -> torch.Tensor:
    """PIL RGB → image_size BILINEAR resize → tensor [-1, 1] (1, 3, H, W)."""
    img = TF.resize(img, [image_size, image_size],
                    interpolation=InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)            # [0, 1]
    t = t * 2.0 - 1.0                # [-1, 1]
    return t.unsqueeze(0)            # (1, 3, H, W)


def norm_tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """generator 출력 ([-1, 1], (1, 3, H, W)) → uint8 RGB (H, W, 3)."""
    t = t.detach().clamp(-1.0, 1.0)
    t = (t + 1.0) * 0.5
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    return arr  # RGB

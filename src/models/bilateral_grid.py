"""BilateralLowLightNet — HDRNet 기반 네이티브 해상도 저조도 향상 모델.

설계 출처
---------
Gharbi et al., "Deep Bilateral Learning for Real-Time Image Enhancement"
(SIGGRAPH 2017). 저해상도에서 *bilateral grid* 의 per-cell affine 계수를 학습하고,
풀해상도 가이드맵으로 slice(trilinear) 하여 풀해상도에 affine 을 적용한다.
이미지를 직접 재합성(decoder upsampling)하지 않으므로 풀해상도 처리 비용이
거의 추가되지 않는다 — LUNA2 의 핵심 가설(256 다운샘플 해상도 페널티 제거)을
직접 겨냥한다 (see ``experiments/00_resolution_diagnostic.py``).

연구 목표
---------
PSNR/SSIM 이 아니라 **frozen YOLOv8n 의 ExDark 검출 성능 향상**. 따라서 네이티브
해상도(예: 640×480, 1280×720)를 보존한 채 향상하여 검출 입력의 해상도 손실을
없앤다.

파이프라인
----------
입력(B,3,H,W) ∈ [-1,1]
  ├─[256 다운샘플]→ CoefficientNet ─→ bilateral grid (B, 12, depth, gh, gw)
  ├─ GuidanceNet(풀해상도) ─────────→ guidance map  (B, 1, H, W) ∈ [0,1]
  ├─ Slice (F.grid_sample, trilinear) ─→ per-pixel affine 계수 (B, 12, H, W)
  ├─ Apply affine (풀해상도) ─────────→ affine 출력 (B, 3, H, W)
  └─ Refine: concat[입력, affine출력] → 2-block DSConv → residual → 출력(B,3,H,W)

전 과정 미분 가능 (grid_sample 포함). 출력 H×W = 입력 H×W (네이티브 보존).
256 다운샘플은 CoefficientNet 입력에만 적용된다.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# affine = 3×4 (RGB 3채널 × [3 가중치 + 1 bias])
AFFINE_PARAMS = 12


# ===========================================================================
# 기본 블록 — norm(BN/IN/none) 선택 가능
# ===========================================================================
def _make_norm(norm: str, ch: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "in":
        return nn.InstanceNorm2d(ch, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"norm 은 'bn'|'in'|'none' (got '{norm}')")


class ConvBlock(nn.Module):
    """Conv(k×k, stride) → Norm → ReLU."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, stride: int = 1,
                 norm: str = "in", act: bool = True) -> None:
        super().__init__()
        use_bias = (norm == "none")
        self.conv = nn.Conv2d(c_in, c_out, k, stride=stride,
                              padding=k // 2, bias=use_bias)
        self.norm = _make_norm(norm, c_out)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise Separable Conv: DW3×3 → Norm → ReLU → PW1×1 → Norm → ReLU."""

    def __init__(self, c_in: int, c_out: int, norm: str = "in") -> None:
        super().__init__()
        use_bias = (norm == "none")
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=use_bias)
        self.bn1 = _make_norm(norm, c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=use_bias)
        self.bn2 = _make_norm(norm, c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


# ===========================================================================
# 1. Coefficient Net — 저해상도(256)에서 bilateral grid 계수 예측
# ===========================================================================
class CoefficientNet(nn.Module):
    """저해상도 입력 → bilateral grid affine 계수.

    Parameters
    ----------
    cw : int
        base channel width. splat 채널이 cw→2cw→4cw→4cw 로 증가.
    grid_size : int
        bilateral grid 의 공간 해상도 gh=gw.
    depth : int
        grid 의 luma(밝기) 축 bin 수.
    low_res : int
        coefficient net 입력 해상도 (splat 4× stride2 → low_res/16).
    norm : str
        'bn' | 'in' | 'none'.

    출력: ``(B, AFFINE_PARAMS, depth, grid_size, grid_size)`` 의 bilateral grid.
    """

    def __init__(
        self,
        cw: int = 32,
        grid_size: int = 16,
        depth: int = 8,
        low_res: int = 256,
        norm: str = "in",
    ) -> None:
        super().__init__()
        self.cw = cw
        self.grid_size = grid_size
        self.depth = depth
        self.low_res = low_res

        c1, c2, c4 = cw, cw * 2, cw * 4

        # --- Splat: 3 → cw → 2cw → 4cw → 4cw, 각 stride 2 (256 → 16) ---
        self.splat = nn.Sequential(
            ConvBlock(3, c1, k=3, stride=2, norm=norm),    # 256 → 128
            ConvBlock(c1, c2, k=3, stride=2, norm=norm),   # 128 → 64
            ConvBlock(c2, c4, k=3, stride=2, norm=norm),   # 64  → 32
            ConvBlock(c4, c4, k=3, stride=2, norm=norm),   # 32  → 16
        )

        # --- Local features: 16×16 에서 conv 2개 (공간 유지) ---
        self.local = nn.Sequential(
            ConvBlock(c4, c4, k=3, stride=1, norm=norm),
            ConvBlock(c4, c4, k=3, stride=1, norm="none", act=False),  # 마지막은 선형
        )

        # --- Global features: 추가 stride conv → GAP → FC 2개 ---
        self.global_conv = nn.Sequential(
            ConvBlock(c4, c4, k=3, stride=2, norm=norm),   # 16 → 8
            ConvBlock(c4, c4, k=3, stride=2, norm=norm),   # 8  → 4
        )
        self.global_fc = nn.Sequential(
            nn.Linear(c4, c4), nn.ReLU(inplace=True),
            nn.Linear(c4, c4),
        )

        # 융합 후 비선형
        self.fuse_act = nn.ReLU(inplace=True)

        # --- 1×1 conv → grid (B, depth*12, gh, gw) ---
        self.to_grid = nn.Conv2d(c4, depth * AFFINE_PARAMS, kernel_size=1)

    def forward(self, x_low: torch.Tensor) -> torch.Tensor:
        # x_low: (B, 3, low_res, low_res)
        s = self.splat(x_low)                  # (B, 4cw, s, s)  (s = low_res/16)

        local = self.local(s)                  # (B, 4cw, s, s)

        g = self.global_conv(s)                # (B, 4cw, s/4, s/4)
        g = F.adaptive_avg_pool2d(g, 1).flatten(1)   # (B, 4cw)
        g = self.global_fc(g)                  # (B, 4cw)
        g = g.view(g.size(0), g.size(1), 1, 1) # (B, 4cw, 1, 1)

        fused = self.fuse_act(local + g)       # broadcast add → (B, 4cw, s, s)

        # grid_size 와 공간 해상도가 다르면 맞춰준다 (low_res != 256 대비)
        if fused.shape[-1] != self.grid_size or fused.shape[-2] != self.grid_size:
            fused = F.interpolate(fused, size=(self.grid_size, self.grid_size),
                                  mode="bilinear", align_corners=False)

        grid = self.to_grid(fused)             # (B, depth*12, gh, gw)
        B = grid.size(0)
        grid = grid.view(B, AFFINE_PARAMS, self.depth, self.grid_size, self.grid_size)
        return grid                            # (B, 12, depth, gh, gw)


# ===========================================================================
# 2. Guidance Net — 풀해상도 per-pixel 가이드맵
# ===========================================================================
class GuidanceNet(nn.Module):
    """풀해상도 입력 → 1채널 guidance map ∈ [0,1] (luma-anchored).

    구조: ``luma(입력 휘도) + 0.5·tanh(conv3∘conv2∘conv1(x))`` → clamp[0,1].

    설계 근거 (붕괴 수정)
    ---------------------
    기존(``sigmoid(conv-only)``) 은 저조도 입력에서 학습이 guidance 를 ~0.53
    상수로 수렴시켜 bilateral grid 의 z(luma)축이 1슬라이스만 쓰이는 degenerate
    붕괴가 발생했다. 이를 막기 위해 **입력 luma(Rec.601)** 를 guidance 의 base
    로 anchor 하고, 학습 가능한 residual 을 ``±0.5·tanh`` 로 더한다.

    * conv3 는 **zero-init** → random-init 시 ``guidance ≈ luma`` 이므로 입력
      휘도에 따라 z 가 depth 전 구간에 자연 분포 (붕괴 원천 차단).
    * residual 은 ``tanh`` 로 ±0.5 bound → guidance 가 [0,1] 근방을 안정 유지.
    * slice_grid 의 z 매핑(z=guidance·(D-1), align_corners=True)은 올바르므로
      그대로 둔다 (좌표 매핑 문제 아님 — 출력 분포 문제였음).
    """

    def __init__(self, c_hidden: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, c_hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(c_hidden, c_hidden, 1)
        self.conv3 = nn.Conv2d(c_hidden, 1, 1)
        self.act = nn.ReLU(inplace=True)
        # Rec.601 luma 가중치 (입력 [-1,1]→[0,1] 변환 후 적용)
        self.register_buffer("luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        # zero-init → 학습 초기 guidance ≈ luma (z 전구간 prior)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x_full: torch.Tensor) -> torch.Tensor:
        x01 = (x_full + 1.0) * 0.5                          # [-1,1] → [0,1]
        luma = (x01 * self.luma_w).sum(dim=1, keepdim=True)  # (B,1,H,W) ∈ [0,1]
        r = self.act(self.conv1(x_full))
        r = self.act(self.conv2(r))
        r = self.conv3(r)                                   # residual logit
        g = luma + 0.5 * torch.tanh(r)                      # 휘도 anchor + 학습 보정
        return g.clamp(0.0, 1.0)                            # (B, 1, H, W) ∈ [0,1]


# ===========================================================================
# 3. Slice — grid_sample 기반 trilinear 보간
# ===========================================================================
def slice_grid(grid: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
    """bilateral grid 를 가이드맵 기준으로 trilinear slice → 풀해상도 affine 계수.

    Parameters
    ----------
    grid : Tensor ``(B, 12, depth, gh, gw)``
        CoefficientNet 출력. 5D grid_sample 의 ``(N, C, D_in, H_in, W_in)``.
    guidance : Tensor ``(B, 1, H, W)`` ∈ [0,1]

    Returns
    -------
    Tensor ``(B, 12, H, W)`` — per-pixel affine 계수.

    구현: 5D ``F.grid_sample`` 의 sampling grid 를
    ``(x=W축, y=H축, z=guidance)`` 로 구성. bilinear(=trilinear) + align_corners.
    전 과정 미분 가능.
    """
    B, C, D, gh, gw = grid.shape
    _, _, H, W = guidance.shape
    device, dtype = grid.device, grid.dtype

    # 공간 좌표 (정규화 [-1, 1]) — x: W축, y: H축
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")        # (H, W)
    gx = gx.unsqueeze(0).expand(B, H, W)
    gy = gy.unsqueeze(0).expand(B, H, W)

    # z 좌표 = guidance [0,1] → [-1,1] (depth 축)
    gz = guidance.squeeze(1) * 2.0 - 1.0                  # (B, H, W)

    # sampling grid: (B, D_out=1, H, W, 3), 마지막 차원 순서 (x, y, z)
    sample_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)  # (B,1,H,W,3)

    sliced = F.grid_sample(
        grid, sample_grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )                                                     # (B, 12, 1, H, W)
    return sliced.squeeze(2)                              # (B, 12, H, W)


# ===========================================================================
# 4. Apply affine — 풀해상도 per-pixel 3×4 affine
# ===========================================================================
def apply_affine(x_full: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """풀해상도 입력에 per-pixel affine 적용.

    Parameters
    ----------
    x_full : Tensor ``(B, 3, H, W)``
    coeffs : Tensor ``(B, 12, H, W)`` → ``(B, 3, 4, H, W)`` 로 해석.

    Returns
    -------
    Tensor ``(B, 3, H, W)`` — ``out_c = Σ_j A[c,j]·in_j + A[c,3]``.

    Identity prior
    --------------
    affine 을 **identity 로부터의 residual** 로 구성한다: 예측된 3×3 scale 의
    대각 성분에 ``+1`` (eye(3)) 을 더해 적용한다. 따라서 ``to_grid`` 예측이 0 이면
    ``scale = I``, ``bias = 0`` → ``out = x`` (정확한 passthrough). off-diagonal
    scale 과 bias 는 예측값을 그대로 사용한다. (slice/guidance/refine 불변.)
    """
    B, _, H, W = x_full.shape
    A = coeffs.view(B, 3, 4, H, W)                # (B, 3, 4, H, W)
    # identity prior: 예측 scale + I (대각에만 +1). 예측=0 → affine=identity.
    eye = torch.eye(3, device=A.device, dtype=A.dtype).view(1, 3, 3, 1, 1)
    weight = A[:, :, :3, :, :] + eye              # (B, 3, 3, H, W)
    bias = A[:, :, 3, :, :]                        # (B, 3, H, W)
    out = (weight * x_full.unsqueeze(1)).sum(dim=2) + bias  # (B, 3, H, W)
    return out


# ===========================================================================
# 5. BilateralLowLightNet — 전체 모델
# ===========================================================================
class BilateralLowLightNet(nn.Module):
    """HDRNet 기반 저조도 향상기 (네이티브 해상도 보존).

    Parameters
    ----------
    cw : int
        CoefficientNet base channel width (기본 32).
    grid_size : int
        bilateral grid 공간 해상도 gh=gw (기본 16).
    depth : int
        grid luma 축 bin 수 (기본 8).
    low_res : int
        CoefficientNet 입력 다운샘플 해상도 (기본 256).
    refine_channels : int
        refinement DSConv 채널 (기본 24).
    refine_blocks : int
        refinement DSConv 블록 수 (기본 2).
    guidance_channels : int
        GuidanceNet hidden 채널 (기본 16).
    norm : str
        'bn' | 'in' | 'none' (기본 'in').

    forward(x): 임의 ``(B,3,H,W)`` [-1,1] → 동일 ``(B,3,H,W)``.
    """

    def __init__(
        self,
        cw: int = 32,
        grid_size: int = 16,
        depth: int = 8,
        low_res: int = 256,
        refine_channels: int = 24,
        refine_blocks: int = 2,
        guidance_channels: int = 16,
        norm: str = "in",
    ) -> None:
        super().__init__()
        self.low_res = low_res

        self.coefficient_net = CoefficientNet(
            cw=cw, grid_size=grid_size, depth=depth, low_res=low_res, norm=norm,
        )
        self.guidance_net = GuidanceNet(c_hidden=guidance_channels)

        # --- Refinement: concat[입력(3), affine출력(3)] = 6채널 → DSConv → residual ---
        rc = refine_channels
        layers = [DSConv(6, rc, norm=norm)]
        for _ in range(max(refine_blocks - 1, 0)):
            layers.append(DSConv(rc, rc, norm=norm))
        self.refine = nn.Sequential(*layers)
        self.refine_out = nn.Conv2d(rc, 3, kernel_size=1)
        # residual 출력이 처음엔 0 이 되도록 초기화 (affine 결과를 그대로 통과)
        nn.init.zeros_(self.refine_out.weight)
        nn.init.zeros_(self.refine_out.bias)

    # ------------------------------------------------------------------
    def _downsample_for_coeff(self, x: torch.Tensor) -> torch.Tensor:
        """CoefficientNet 입력용 256×256 다운샘플 (coefficient net 에만 적용)."""
        return F.interpolate(
            x, size=(self.low_res, self.low_res),
            mode="bilinear", align_corners=False,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_grid: bool = False):
        """임의 H×W 입력 → 동일 H×W 출력 (네이티브 해상도 보존).

        return_grid=True 면 ``(out, grid)`` 반환 — grid spatial TV 정규화 등
        bilateral grid 계수에 직접 손실을 걸 때 사용 (grad 연결 유지).
        """
        # 1) 저해상도 스트림: 256 다운샘플 → bilateral grid
        x_low = self._downsample_for_coeff(x)
        grid = self.coefficient_net(x_low)            # (B, 12, depth, gh, gw)

        # 2) 풀해상도 가이드맵
        guidance = self.guidance_net(x)               # (B, 1, H, W)

        # 3) slice → per-pixel affine 계수
        coeffs = slice_grid(grid, guidance)           # (B, 12, H, W)

        # 4) apply affine
        affine_out = apply_affine(x, coeffs)          # (B, 3, H, W)

        # 5) refine (residual)
        r = self.refine(torch.cat([x, affine_out], dim=1))
        residual = self.refine_out(r)                 # (B, 3, H, W)
        out = affine_out + residual

        if return_grid:
            return out, grid
        return out                                    # (B, 3, H, W), 입력과 동일 크기

    # ------------------------------------------------------------------
    @torch.no_grad()
    def intermediate(self, x: torch.Tensor) -> dict:
        """디버그/시각화용 — 중간 텐서들을 dict 로 반환."""
        x_low = self._downsample_for_coeff(x)
        grid = self.coefficient_net(x_low)
        guidance = self.guidance_net(x)
        coeffs = slice_grid(grid, guidance)
        affine_out = apply_affine(x, coeffs)
        out = self.forward(x)
        return {
            "grid": grid, "guidance": guidance, "coeffs": coeffs,
            "affine_out": affine_out, "out": out,
        }


def build_from_config(cfg: dict) -> BilateralLowLightNet:
    """``configs/bilateral_base.yaml`` 의 ``model`` 블록 → 모델 인스턴스."""
    m = cfg.get("model", cfg)
    return BilateralLowLightNet(
        cw=m.get("cw", 32),
        grid_size=m.get("grid_size", 16),
        depth=m.get("depth", 8),
        low_res=m.get("low_res", 256),
        refine_channels=m.get("refine_channels", 24),
        refine_blocks=m.get("refine_blocks", 2),
        guidance_channels=m.get("guidance_channels", 16),
        norm=m.get("norm", "in"),
    )


# ===========================================================================
# 데모 — forward sanity check (학습 없음) + profiling
# ===========================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.profile import profile_bilateral  # noqa: E402

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BilateralLowLightNet().to(device).eval()

    print("=" * 78)
    print(" BilateralLowLightNet — forward sanity check (학습 없음)")
    print("=" * 78)
    for (H, W) in [(480, 640), (384, 512), (720, 1280)]:
        x = torch.randn(1, 3, H, W, device=device)
        with torch.no_grad():
            y = model(x)
        ok = (y.shape == x.shape)
        print(f"  in (1,3,{H},{W})  →  out {tuple(y.shape)}   "
              f"shape-match: {ok}   out-range[{y.min():.2f},{y.max():.2f}]")
        assert ok, "출력 shape 가 입력과 다릅니다!"
    print("=" * 78)

    # 프로파일링 (저해상 256 스트림 / 풀해상 640×480 분리 + micro-bench)
    profile_bilateral(model, device=device)

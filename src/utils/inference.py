"""추론 공용 헬퍼 — fp32 forward + 출력 가드(nan_to_num + clamp).

배경
----
AMP/수치 불안정으로 enhancer 출력에 NaN/Inf 가 생기면 (a) validation PSNR/SSIM 이
오염되고 (b) 검출 평가 시 uint8 변환이 깨진다. 학습/추론/평가가 모두 동일한
"가드된 fp32 추론" 을 쓰도록 한 곳에 함수화한다.

규약: 입출력 모두 ``[-1, 1]``.
"""
from __future__ import annotations

import torch


def guard_output(t: torch.Tensor) -> torch.Tensor:
    """NaN/Inf 제거 후 ``[-1, 1]`` clamp.

    NaN→0, +Inf→+1, -Inf→-1 로 치환한 뒤 clamp. 정상(유한) 출력에 대해서는
    단순 clamp 와 동일하므로 baseline 재현성에 영향 없음.
    """
    return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


@torch.no_grad()
def enhance(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """가드된 fp32 추론: eval 모드 + autocast 없이 forward → ``guard_output``.

    Parameters
    ----------
    model : nn.Module   (입력 [-1,1] → 출력 [-1,1])
    x : Tensor (B,3,H,W) in [-1,1]

    Returns
    -------
    Tensor (B,3,H,W) in [-1,1] (NaN/Inf 가드 적용).
    """
    was_training = model.training
    model.eval()
    out = guard_output(model(x))
    if was_training:
        model.train()
    return out


@torch.no_grad()
def is_nonfinite(t: torch.Tensor) -> bool:
    """텐서에 NaN/Inf 가 하나라도 있으면 True (가드 동작 카운트용)."""
    return bool((~torch.isfinite(t)).any())

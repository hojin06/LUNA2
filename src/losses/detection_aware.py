"""Detection-aware 손실 — frozen YOLOv8n + v8DetectionLoss (Phase 2).

핵심 (검증됨)
-------------
``Detect`` head 를 ``training=True`` 로 두면 raw 3-scale preds 를 반환하고,
``v8DetectionLoss(preds, targets)`` 의 grad 가 **frozen detector 의 activation 을
통해 입력(enhanced) 까지 흐른다**. detector 파라미터는 ``requires_grad=False`` 라
학습되지 않는다 (검출기는 고정, enhancer 만 검출 친화적으로 유도).

입력 규약: enhanced ``[-1,1]`` → 내부 ``[0,1]`` 변환 후 detector 통과.
targets: ``{cls:(N,1), bboxes:(N,4) normalized cxcywh, batch_idx:(N,)}``.

원본 ``SmallSizePM_GAN_model/CODES/train_detection_loss.py`` 의 통합 패턴을 이식.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


def grid_tv_loss(grid: torch.Tensor) -> torch.Tensor:
    """Bilateral grid 계수의 spatial(gh, gw) Total Variation.

    Parameters
    ----------
    grid : Tensor ``(B, C, D, gh, gw)`` — CoefficientNet 출력 (affine 계수).

    인접 grid **셀** 간 계수 차이(L1)를 페널티 → 셀 경계 체커보드 artifact 완화.
    ★ 출력 이미지가 아니라 **grid 계수에만** 적용하므로 실제 이미지 엣지는 보존된다
    (guidance slicing 은 그대로 → 휘도 기반 톤 곡선 유지). depth(luma) 축은 TV 하지
    않는다 (톤 변화는 허용; 공간 부드러움만 유도).
    """
    dh = (grid[..., 1:, :] - grid[..., :-1, :]).abs().mean()   # gh 축 인접 차이
    dw = (grid[..., :, 1:] - grid[..., :, :-1]).abs().mean()   # gw 축 인접 차이
    return dh + dw


def build_frozen_yolo(weights: Path | str, device: str = "cuda"):
    """frozen YOLOv8n DetectionModel + v8DetectionLoss 반환.

    * 모든 파라미터 ``requires_grad=False`` + ``eval()`` (BN running stats 동결).
    * Detect head ``training=True`` → raw 3-scale preds (v8DetectionLoss 입력 포맷).
    * ``model.args`` 가 dict/None 이면 DEFAULT_CFG 로 교체 (box/cls/dfl attr 필요).
    """
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss

    yolo = YOLO(str(weights))
    m: nn.Module = yolo.model.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    detect = m.model[-1]  # type: ignore[index]
    if type(detect).__name__ != "Detect":
        raise RuntimeError(f"YOLO 마지막 module 이 Detect 아님: {type(detect).__name__}")
    detect.training = True

    if not hasattr(m, "args") or m.args is None or isinstance(m.args, dict):
        m.args = DEFAULT_CFG

    det_loss_fn = v8DetectionLoss(m)
    return m, det_loss_fn


class YoloDetectionLoss(nn.Module):
    """frozen YOLOv8n detection loss (enhanced[-1,1] → scalar, grad→enhancer).

    Parameters
    ----------
    weights : Path | str   YOLOv8n 가중치 (paths.yaml::yolov8n).
    device : str
    """

    def __init__(self, weights: Path | str, device: str = "cuda") -> None:
        super().__init__()
        self.detector, self.det_loss_fn = build_frozen_yolo(weights, device)
        self._device = device

    def train(self, mode: bool = True):
        """enhancer 쪽에서 train() 전파돼도 detector 는 항상 frozen-eval 유지."""
        nn.Module.train(self, mode)
        self.detector.eval()
        self.detector.model[-1].training = True  # type: ignore[index]
        return self

    @staticmethod
    def _to01(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(
        self,
        enhanced_pm1: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (L_yolo scalar, items(3,) detached [box,cls,dfl])."""
        enh01 = self._to01(enhanced_pm1)
        batch = {
            "cls": targets["cls"],
            "bboxes": targets["bboxes"],
            "batch_idx": targets["batch_idx"],
            "img": enh01,  # v8DetectionLoss 가 shape 참조
        }
        preds = self.detector(enh01)
        loss_vec, items = self.det_loss_fn(preds, batch)
        L = loss_vec.sum() if loss_vec.ndim > 0 else loss_vec
        return L, items.detach()

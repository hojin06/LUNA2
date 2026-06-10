"""ONNX export — LUNA2 향상기를 임베디드 추론용으로 내보내기. **스텁 (구현 예정)**.

목적
----
경량 모델 연구의 산출물을 Jetson Orin Nano 등에서 ONNX Runtime / TensorRT 로
구동하기 위한 export. dynamic axes (가변 해상도) 지원이 핵심 — bilateral grid 는
full-res 입력을 받기 때문.

구현 시 채울 것
---------------
* src.models.bilateral_grid.BilateralEnhanceNet (또는 baseline luna_base) 로드.
* torch.onnx.export (opset ≥ 17, dynamic H/W).
* onnxsim 단순화 + onnxruntime 로 출력 일치(allclose) 검증.

실행:  ``python deploy/export_onnx.py --checkpoint <ckpt> --out model.onnx``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LUNA2 ONNX export (스텁)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="luna2.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--dynamic", action="store_true", help="가변 해상도 axes")
    return p.parse_args()


def export_onnx(checkpoint: Path | str, out_path: Path | str, **kwargs) -> None:
    """체크포인트 → ONNX. **스텁.**"""
    raise NotImplementedError("export_onnx: 구현 예정")


def main() -> int:
    _ = parse_args()
    raise NotImplementedError("export_onnx: 구현 예정")


if __name__ == "__main__":
    raise SystemExit(main())

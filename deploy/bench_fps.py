"""FPS 벤치마크 — PyTorch / ONNX Runtime latency·throughput. **스텁 (구현 예정)**.

목적
----
경량성 주장(임베디드 30 FPS 등)을 뒷받침하는 추론 속도 측정. ``src.utils.profile``
(예정) 의 ``benchmark_fps`` 를 호출해 PyTorch eager 와 ONNX Runtime 양쪽을 잰다.

구현 시 채울 것
---------------
* PyTorch: warmup 후 GPU/CPU latency, p50/p95, FPS.
* ONNX:  onnxruntime InferenceSession (CUDA/CPU EP) 동일 입력 벤치.
* 해상도 sweep (256/512/원본) 표 출력 → runs/bench/.

실행:  ``python deploy/bench_fps.py --checkpoint <ckpt>``
       ``python deploy/bench_fps.py --onnx model.onnx``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LUNA2 FPS 벤치마크 (스텁)")
    p.add_argument("--checkpoint", type=str, default=None, help="PyTorch 체크포인트")
    p.add_argument("--onnx", type=str, default=None, help="ONNX 모델 경로")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--n_runs", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def bench(checkpoint=None, onnx=None, **kwargs) -> dict:
    """latency / FPS 측정 → 결과 dict. **스텁.**"""
    raise NotImplementedError("bench_fps: 구현 예정")


def main() -> int:
    _ = parse_args()
    raise NotImplementedError("bench_fps: 구현 예정")


if __name__ == "__main__":
    raise SystemExit(main())

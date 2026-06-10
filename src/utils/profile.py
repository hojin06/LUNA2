"""모델 프로파일링 — params / FLOPs (스트림 분리) / latency / FPS.

연구 맥락
---------
LUNA2 는 "경량" 연구이므로 ExDark mAP 향상과 함께 **연산 비용**을 보고해야 한다.
BilateralLowLightNet 은 두 스트림으로 나뉜다:

  * **저해상도 스트림** : CoefficientNet (256×256 에서만 동작) — 무거운 conv 다수.
  * **풀해상도 스트림** : GuidanceNet + slice + affine + refine — 풀해상도지만 매우 가벼움.

본 모듈은 이 둘의 params·FLOPs 를 **분리**해 보고하고(=풀해상도 처리가 싸다는
주장의 근거), dev GPU 에서 640×480 입력 100회 forward 평균 latency·FPS micro-bench
를 출력한다.

FLOP 백엔드: fvcore → thop 순으로 가용한 것을 사용. 둘 다 conv/linear/matmul 의
MAC 을 센다. ``F.grid_sample`` (slice) 과 per-pixel affine(elementwise mul/add)
는 두 백엔드 모두 카운트하지 않는다 — memory-bound 이며 FLOP 기여가 작다(보고에 명시).

주의: 여기 latency 는 **dev GPU** 기준이다. 임베디드(Jetson Orin Nano 등) 실측은
별도 측정이 필요하다 (TensorRT/INT8, 전력 제약 등으로 수치가 달라짐).
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. 파라미터 / 크기
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(total, trainable) 파라미터 수."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """가중치+버퍼 디스크 크기(MB). FP32=4B, FP16=2B, INT8=1B."""
    bpp = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}.get(dtype, 4)
    n = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    return n * bpp / (1024 ** 2)


# ---------------------------------------------------------------------------
# 2. FLOP/MAC 카운터 (fvcore → thop fallback)
# ---------------------------------------------------------------------------
def _backend_name() -> str:
    try:
        import fvcore  # noqa: F401
        return "fvcore"
    except ImportError:
        pass
    try:
        import thop  # noqa: F401
        return "thop"
    except ImportError:
        return "none"


def compute_macs(module: nn.Module, example_input: torch.Tensor) -> float:
    """``module(example_input)`` 의 MAC 수. 백엔드 없으면 NaN.

    반환값은 MAC(multiply-accumulate). FLOPs ≈ 2 × MACs.
    측정 중 모델/입력이 변형되지 않도록 deepcopy + no_grad 로 격리한다.
    """
    backend = _backend_name()
    if backend == "none":
        return float("nan")

    mod = copy.deepcopy(module).eval()
    x = example_input.detach().clone()
    # 같은 device 로 정렬
    dev = next(mod.parameters()).device
    x = x.to(dev)

    with torch.no_grad():
        if backend == "fvcore":
            from fvcore.nn import FlopCountAnalysis
            import logging
            logging.getLogger("fvcore").setLevel(logging.ERROR)
            fca = FlopCountAnalysis(mod, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            return float(fca.total())  # fvcore total() = MACs
        else:  # thop
            from thop import profile as thop_profile
            macs, _ = thop_profile(mod, inputs=(x,), verbose=False)
            return float(macs)


# ---------------------------------------------------------------------------
# 3. Micro-benchmark (latency / FPS)
# ---------------------------------------------------------------------------
@torch.no_grad()
def benchmark_fps(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (3, 480, 640),
    device: str = "cuda",
    n_warmup: int = 10,
    n_runs: int = 100,
) -> Dict[str, float]:
    """``input_shape`` (C,H,W) 단일 배치 forward 의 평균 latency / FPS.

    CUDA 면 ``torch.cuda.Event`` 로 GPU 시간을, CPU 면 ``time.perf_counter`` 로 측정.
    """
    import time

    model = model.to(device).eval()
    x = torch.randn(1, *input_shape, device=device)

    # warmup
    for _ in range(n_warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    if device.startswith("cuda"):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(n_runs):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))  # ms
        t = torch.tensor(times_ms)
    else:
        times_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        t = torch.tensor(times_ms)

    mean_ms = float(t.mean())
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": float(t.std()),
        "latency_ms_p50": float(t.median()),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# 4. 종합 프로파일
# ---------------------------------------------------------------------------
def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (3, 480, 640),
    device: str = "cuda",
) -> Dict[str, float]:
    """params + MACs + FLOPs + size + FPS 종합 (단일 모델, 스트림 분리 없음)."""
    total_p, train_p = count_parameters(model)
    macs = compute_macs(model, torch.randn(1, *input_shape))
    bench = benchmark_fps(model, input_shape, device=device)
    return {
        "params": float(total_p),
        "trainable": float(train_p),
        "macs": macs,
        "flops": 2 * macs if macs == macs else float("nan"),
        "size_mb": model_size_mb(model),
        **bench,
    }


def _fmt_macs(macs: float) -> str:
    if macs != macs:  # NaN
        return "n/a (백엔드 없음)"
    if macs >= 1e9:
        return f"{macs / 1e9:.3f} GMAC ({2 * macs / 1e9:.3f} GFLOP)"
    return f"{macs / 1e6:.3f} MMAC ({2 * macs / 1e6:.3f} MFLOP)"


def profile_bilateral(
    model,  # BilateralLowLightNet
    full_hw: Tuple[int, int] = (480, 640),
    device: str = "cuda",
    n_runs: int = 100,
) -> Dict[str, object]:
    """BilateralLowLightNet 전용 — 저해상(256) / 풀해상 스트림 분리 프로파일.

    분리 방식
    ---------
    * low-res 스트림 MACs  = MACs(model.coefficient_net, 256×256 입력).
    * total MACs          = MACs(model, full_hw 입력) — 내부에서 256 다운샘플
      coefficient + 풀해상도 스트림을 모두 포함.
    * full-res 스트림 MACs = total − low-res.  (guidance + refine 의 conv 비용;
      slice/affine 의 elementwise 는 백엔드가 세지 않으므로 별도 표기.)
    """
    HRULE = "=" * 78
    SUB = "-" * 78
    H, W = full_hw
    low = model.low_res

    backend = _backend_name()

    # --- params (컴포넌트별) ---
    tot_p, _ = count_parameters(model)
    coeff_p, _ = count_parameters(model.coefficient_net)
    guide_p, _ = count_parameters(model.guidance_net)
    refine_p, _ = count_parameters(model.refine) if hasattr(model, "refine") else (0, 0)
    refine_p += count_parameters(model.refine_out)[0] if hasattr(model, "refine_out") else 0

    # --- MACs (스트림 분리) ---
    low_macs = compute_macs(model.coefficient_net, torch.randn(1, 3, low, low))
    total_macs = compute_macs(model, torch.randn(1, 3, H, W))
    full_macs = (total_macs - low_macs) if (total_macs == total_macs and low_macs == low_macs) else float("nan")

    # --- micro-bench (풀해상도) ---
    bench = benchmark_fps(model, (3, H, W), device=device, n_runs=n_runs)

    print(HRULE)
    print(" BilateralLowLightNet — Profile")
    print(HRULE)
    print(f"  FLOP backend : {backend}"
          + ("  (설치 없음 → MACs=n/a)" if backend == "none" else ""))
    print(f"  device       : {device}")
    print(f"  total params : {tot_p:,}  ({tot_p / 1e3:.1f} K)   size(FP32) {model_size_mb(model):.3f} MB")
    print(SUB)
    print("  [Params 분해]")
    print(f"    CoefficientNet (저해상) : {coeff_p:,}  ({coeff_p / 1e3:.1f} K)")
    print(f"    GuidanceNet   (풀해상)  : {guide_p:,}")
    print(f"    Refine        (풀해상)  : {refine_p:,}")
    print(SUB)
    print("  [FLOPs 분해]  (MAC = multiply-accumulate, FLOP ≈ 2·MAC)")
    print(f"    저해상 스트림 @ {low}×{low}  : {_fmt_macs(low_macs)}")
    print(f"    풀해상 스트림 @ {W}×{H}  : {_fmt_macs(full_macs)}")
    print(f"    합계        @ {W}×{H}  : {_fmt_macs(total_macs)}")
    print(f"    * slice(grid_sample)/affine 의 elementwise 연산은 백엔드 미집계 "
          f"(memory-bound, FLOP 기여 작음)")
    print(SUB)
    print(f"  [Micro-bench]  {W}×{H} 입력, {bench['n_runs']}회 forward 평균  (dev GPU)")
    print(f"    latency : {bench['latency_ms_mean']:.3f} ± {bench['latency_ms_std']:.3f} ms  "
          f"(p50 {bench['latency_ms_p50']:.3f} ms)")
    print(f"    FPS     : {bench['fps']:.1f}")
    print(f"    * Jetson 등 임베디드 실측은 별도 (TensorRT/INT8·전력 제약으로 상이).")
    print(HRULE)

    return {
        "params_total": tot_p, "params_coeff": coeff_p,
        "params_guidance": guide_p, "params_refine": refine_p,
        "macs_lowres": low_macs, "macs_fullres": full_macs, "macs_total": total_macs,
        "backend": backend, **bench,
    }

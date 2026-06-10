"""복원 품질 평가 — LOL 계열에서 PSNR / SSIM / (옵션) LPIPS.  [이식본]

출처 (Provenance)
-----------------
모델 정의(``src.models.luna_base``), 지표(``src.utils.metrics``), 데이터 로더
(``src.data.lowlight_dataset``) 모두 원본 ``SmallSizePM_GAN_model/CODES`` 이식본을
사용한다. 따라서 원본 ``evaluate()`` 와 **동일한 PSNR/SSIM** 을 재현한다.

평가 프로토콜은 원본 학습/평가와 동일:
  * eval split 을 ``PairedAugment(training=False)`` 로 256×256 BILINEAR resize.
  * generator 출력 clamp([-1,1]) 후 PSNR/SSIM 계산 (data_range=1.0, [0,1] 변환).

사용 예
-------
.. code-block:: bash

    python experiments/eval_restoration.py                       # LOL v1 eval15, luna_original
    python experiments/eval_restoration.py --dataset lol_v2_real --split test
    python experiments/eval_restoration.py --lpips                # LPIPS 추가 (lpips 설치 필요)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- LUNA2 루트를 sys.path 에 등록 ---
_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader

from src.data.lowlight_dataset import build_dataset_by_name, DATASET_REGISTRY
from src.models.luna_base import load_luna_generator
from src.utils.metrics import evaluate
from src.utils.paths import load_paths

HRULE = "=" * 72

# dataset 키 → paths.yaml 의 데이터셋 키
_DATASET_TO_PATHKEY = {
    "lol_v1": "lol_v1",
    "lol_v2_real": "lol_v2",
    "lol_v2_syn": "lol_v2",
    "loli_street": "loli_street",
}


def parse_args() -> argparse.Namespace:
    P = load_paths()
    p = argparse.ArgumentParser(description="LUNA2 복원 품질 평가 (PSNR/SSIM/LPIPS)")
    p.add_argument("--dataset", type=str, default="lol_v1",
                   choices=list(DATASET_REGISTRY),
                   help="평가 데이터셋 키")
    p.add_argument("--split", type=str, default="eval",
                   help="LOL v1: eval / LOL-v2·LoLI: test")
    p.add_argument("--data_root", type=str, default=None,
                   help="데이터셋 루트 (미지정 시 paths.yaml 에서 자동 해석)")
    p.add_argument("--checkpoint", type=str, default=str(P.luna_original),
                   help="LUNA generator 가중치 (기본: paths.yaml::weights.luna_original)")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lpips", action="store_true", help="LPIPS 도 계산")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data_root is None:
        args.data_root = str(P.dataset(_DATASET_TO_PATHKEY[args.dataset]))
    return args


def main() -> int:
    args = parse_args()
    device = args.device

    print(HRULE)
    print(" LUNA2 Restoration Eval — PSNR / SSIM" + (" / LPIPS" if args.lpips else ""))
    print(HRULE)
    print(f"  dataset    : {args.dataset} [{args.split}]")
    print(f"  data_root  : {args.data_root}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")
    print(HRULE)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"[error] 체크포인트가 없습니다: {ckpt}")
        return 1

    # eval split: augment=False → resize-only (결정론적)
    dataset = build_dataset_by_name(
        name=args.dataset, data_root=args.data_root, split=args.split,
        image_size=args.image_size, augment=False, full_resize=False,
    )
    print(f"  {dataset!r}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device != "cpu"),
    )

    G = load_luna_generator(ckpt, device=device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  LUNA params: {n_params:,}  ({n_params / 1e3:.1f} K)")
    print(HRULE)

    metrics = evaluate(G, loader, device=device, compute_lpips=args.lpips)

    print(f"  PSNR  : {metrics['psnr']:.4f} dB")
    print(f"  SSIM  : {metrics['ssim']:.4f}")
    if args.lpips:
        print(f"  LPIPS : {metrics['lpips']:.4f}")
    print(f"  n     : {metrics['n']}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""저조도 페어 데이터셋 (LOL v1/v2 + LoLI-Street) + PairedAugment (LUNA2 이식본).

출처 (Provenance)
-----------------
``SmallSizePM_GAN_model/CODES/data/dataset.py`` + ``data/augmentation.py`` 를
**그대로 이식**. PSNR/SSIM 복원 평가(``experiments/eval_restoration.py``) 가
원본과 동일한 전처리/페어 매칭을 쓰도록 보장한다.

공통 인터페이스
---------------
모든 클래스가 ``(low_tensor, high_tensor)`` ∈ ``[-1, 1]`` 페어를 반환.
``PairedAugment(training=False)`` 는 resize-only 라 평가 결과가 결정론적이다.

데이터 경로는 코드에 하드코딩하지 않고 ``configs/paths.yaml`` →
``src.utils.paths.load_paths()`` 로 주입한다 (데이터 복사 금지).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import InterpolationMode

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

PairTensor = Tuple[torch.Tensor, torch.Tensor]
PairPIL = Tuple[Image.Image, Image.Image]


# ===========================================================================
# Paired augmentation (원본 augmentation.py 이식)
# ===========================================================================
class PairedAugment:
    """LOL (low, high) 페어 동기화 + low 전용 저조도 강화.

    ``training=False`` 면 BILINEAR resize 만 적용 (평가용, 결정론적).
    기하 변환은 동일 난수로 두 이미지에 적용하여 페어 정합을 유지한다.
    """

    def __init__(
        self,
        image_size: int = 256,
        training: bool = True,
        full_resize: bool = False,
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        rotate_deg: float = 10.0,
        crop_scale: Tuple[float, float] = (0.7, 1.0),
        p_lowerhalf_crop: float = 0.5,
        p_perspective: float = 0.4,
        perspective_keep: Tuple[float, float] = (0.6, 0.7),
        p_gamma: float = 0.4,
        gamma_range: Tuple[float, float] = (1.5, 3.0),
        p_brightness: float = 0.4,
        brightness_range: Tuple[float, float] = (0.3, 0.7),
        p_noise: float = 0.5,
        noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
    ) -> None:
        self.image_size = image_size
        self.training = training
        self.full_resize = full_resize
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.rotate_deg = rotate_deg
        self.crop_scale = crop_scale
        self.p_lowerhalf_crop = p_lowerhalf_crop
        self.p_perspective = p_perspective
        self.perspective_keep = perspective_keep
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_brightness = p_brightness
        self.brightness_range = brightness_range
        self.p_noise = p_noise
        self.noise_sigma_range = noise_sigma_range

    def __call__(self, low_img: Image.Image, high_img: Image.Image) -> PairTensor:
        if not self.training:
            return self._eval_path(low_img, high_img)
        return self._train_path(low_img, high_img)

    def _eval_path(self, low_img: Image.Image, high_img: Image.Image) -> PairTensor:
        size = [self.image_size, self.image_size]
        low_img = TF.resize(low_img, size, interpolation=InterpolationMode.BILINEAR)
        high_img = TF.resize(high_img, size, interpolation=InterpolationMode.BILINEAR)
        return self._to_norm_tensor(low_img), self._to_norm_tensor(high_img)

    def _train_path(self, low_img: Image.Image, high_img: Image.Image) -> PairTensor:
        if self.full_resize:
            size = [self.image_size, self.image_size]
            low_img = TF.resize(low_img, size, interpolation=InterpolationMode.BILINEAR)
            high_img = TF.resize(high_img, size, interpolation=InterpolationMode.BILINEAR)
        else:
            low_img, high_img = self._paired_crop_resize(low_img, high_img)
        low_img, high_img = self._paired_hflip(low_img, high_img)
        low_img, high_img = self._paired_rotate(low_img, high_img)
        low_img, high_img = self._paired_perspective(low_img, high_img)

        low_t = TF.to_tensor(low_img)
        high_t = TF.to_tensor(high_img)
        low_t = self._photometric_low_only(low_t)
        return low_t * 2.0 - 1.0, high_t * 2.0 - 1.0

    def _paired_crop_resize(self, low: Image.Image, high: Image.Image) -> PairPIL:
        W, H = low.size
        scale = random.uniform(*self.crop_scale)
        crop_h = max(int(H * scale), 16)
        crop_w = max(int(W * scale), 16)
        if random.random() < self.p_lowerhalf_crop:
            lo = int((H - crop_h) * 0.3)
            hi = max(H - crop_h, lo)
            top = random.randint(lo, hi) if hi > lo else 0
        else:
            top = random.randint(0, max(H - crop_h, 0))
        left = random.randint(0, max(W - crop_w, 0))
        size = [self.image_size, self.image_size]
        low = TF.resized_crop(low, top, left, crop_h, crop_w, size,
                              interpolation=InterpolationMode.BILINEAR)
        high = TF.resized_crop(high, top, left, crop_h, crop_w, size,
                               interpolation=InterpolationMode.BILINEAR)
        return low, high

    def _paired_hflip(self, low: Image.Image, high: Image.Image) -> PairPIL:
        if random.random() < self.p_flip:
            return TF.hflip(low), TF.hflip(high)
        return low, high

    def _paired_rotate(self, low: Image.Image, high: Image.Image) -> PairPIL:
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            low = TF.rotate(low, angle,
                            interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
            high = TF.rotate(high, angle,
                             interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
        return low, high

    def _paired_perspective(self, low: Image.Image, high: Image.Image) -> PairPIL:
        if random.random() >= self.p_perspective:
            return low, high
        s = self.image_size
        keep = random.uniform(*self.perspective_keep)
        top_y = int(s * (1.0 - keep))
        narrow = int(random.uniform(0.0, 0.15) * s)
        startpoints = [[narrow, top_y], [s - narrow, top_y], [s, s], [0, s]]
        endpoints = [[0, 0], [s, 0], [s, s], [0, s]]
        low = TF.perspective(low, startpoints, endpoints,
                             interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
        high = TF.perspective(high, startpoints, endpoints,
                              interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
        return low, high

    def _photometric_low_only(self, low_t: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_gamma:
            gamma = random.uniform(*self.gamma_range)
            low_t = low_t.clamp(min=1e-6).pow(gamma)
        if random.random() < self.p_brightness:
            factor = random.uniform(*self.brightness_range)
            low_t = low_t * factor
        if random.random() < self.p_noise:
            sigma = random.uniform(*self.noise_sigma_range)
            low_t = low_t + torch.randn_like(low_t) * sigma
        return low_t.clamp(0.0, 1.0)

    @staticmethod
    def _to_norm_tensor(img: Image.Image) -> torch.Tensor:
        return TF.to_tensor(img) * 2.0 - 1.0


# ===========================================================================
# 공통 유틸리티 (원본 dataset.py 이식)
# ===========================================================================
def _find_subdir(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """``root`` 아래에서 후보 중 처음 존재하는 디렉토리 반환 (대소문자 무관)."""
    if not root.is_dir():
        return None
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    lower_targets = {c.lower() for c in candidates}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in lower_targets:
            return p
    return None


def _build_pairs(low_dir: Path, high_dir: Path) -> List[Tuple[Path, Path]]:
    """파일명(stem) 일치 (low, high) 페어 매칭. 확장자 자유."""
    if not low_dir.is_dir() or not high_dir.is_dir():
        return []
    high_index = {p.stem: p for p in high_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS}
    pairs: List[Tuple[Path, Path]] = []
    for low_p in sorted(low_dir.iterdir()):
        if not low_p.is_file() or low_p.suffix.lower() not in _IMG_EXTS:
            continue
        high_p = high_index.get(low_p.stem)
        if high_p is not None:
            pairs.append((low_p, high_p))
    return pairs


# ===========================================================================
# 0. 일반 페어 데이터셋 (base)
# ===========================================================================
class PairedImageDataset(Dataset):
    """``(low_dir, high_dir)`` 직접 지정 방식의 일반 페어 데이터셋."""

    def __init__(
        self,
        low_dir: Path,
        high_dir: Path,
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image], PairTensor]] = None,
        name: str = "PairedImageDataset",
    ) -> None:
        super().__init__()
        self.name = name
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.image_size = image_size

        if not self.low_dir.is_dir() or not self.high_dir.is_dir():
            raise FileNotFoundError(
                f"{name}: low/high 디렉토리를 찾을 수 없습니다.\n"
                f"  low  = {self.low_dir}\n  high = {self.high_dir}"
            )

        self.pairs = _build_pairs(self.low_dir, self.high_dir)
        if not self.pairs:
            raise RuntimeError(
                f"{name}: (low, high) 페어가 0 개입니다.\n"
                f"  low  = {self.low_dir}\n  high = {self.high_dir}"
            )

        self.transform = transform or PairedAugment(
            image_size=image_size, training=augment, full_resize=full_resize,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairTensor:
        low_p, high_p = self.pairs[idx]
        low_img = Image.open(low_p).convert("RGB")
        high_img = Image.open(high_p).convert("RGB")
        return self.transform(low_img, high_img)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"n_pairs={len(self.pairs)}, image_size={self.image_size})")


# ===========================================================================
# 1. LOL v1
# ===========================================================================
class LOLDataset(PairedImageDataset):
    """LOL v1 paired (low, high) dataset. split: train→our485, eval→eval15."""

    SPLIT_TO_FOLDER = {"train": "our485", "eval": "eval15"}

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image], PairTensor]] = None,
    ) -> None:
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'")
        self.data_root = Path(data_root)
        self.split = split
        folder = self.SPLIT_TO_FOLDER[split]
        low_dir = self.data_root / folder / "low"
        high_dir = self.data_root / folder / "high"
        if not low_dir.is_dir() or not high_dir.is_dir():
            raise FileNotFoundError(
                f"LOL v1 dataset not found. Expected:\n  {low_dir}\n  {high_dir}"
            )
        super().__init__(
            low_dir=low_dir, high_dir=high_dir,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            name=f"LOLv1[{split}]",
        )


# ===========================================================================
# 2. LOL-v2 Real
# ===========================================================================
class LOLv2RealDataset(PairedImageDataset):
    """LOL-v2 Real captured paired dataset (Yang et al., TIP 2021)."""

    SPLIT_TO_FOLDER = {"train": "Train", "eval": "Test", "test": "Test"}
    LOW_CANDIDATES = ("Low", "low", "Low_images", "LOW")
    HIGH_CANDIDATES = ("Normal", "normal", "Normal_images", "high", "High", "GT", "gt")

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image], PairTensor]] = None,
        subset_dir: str = "Real_captured",
    ) -> None:
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'")
        self.data_root = Path(data_root)
        self.split = split

        split_dir = self.data_root / subset_dir / self.SPLIT_TO_FOLDER[split]
        if not split_dir.is_dir():
            split_dir_found = _find_subdir(
                self.data_root / subset_dir,
                [self.SPLIT_TO_FOLDER[split], self.SPLIT_TO_FOLDER[split].lower()],
            )
            if split_dir_found is None:
                raise FileNotFoundError(f"LOL-v2 Real split 디렉토리 없음: {split_dir}")
            split_dir = split_dir_found

        low_dir = _find_subdir(split_dir, self.LOW_CANDIDATES)
        high_dir = _find_subdir(split_dir, self.HIGH_CANDIDATES)
        if low_dir is None or high_dir is None:
            raise FileNotFoundError(
                f"LOL-v2 Real 의 low/high 폴더를 찾을 수 없습니다.\n  탐색 위치 : {split_dir}"
            )
        super().__init__(
            low_dir=low_dir, high_dir=high_dir,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            name=f"LOLv2-Real[{split}]",
        )


# ===========================================================================
# 3. LOL-v2 Synthetic
# ===========================================================================
class LOLv2SyntheticDataset(LOLv2RealDataset):
    """LOL-v2 Synthetic paired dataset (subset_dir='Synthetic')."""

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image], PairTensor]] = None,
    ) -> None:
        super().__init__(
            data_root=data_root, split=split,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            subset_dir="Synthetic",
        )
        self.name = f"LOLv2-Syn[{split}]"


# ===========================================================================
# 4. LoLI-Street
# ===========================================================================
class LoLIStreetDataset(PairedImageDataset):
    """LoLI-Street paired dataset (arXiv 2410.09831, 2024)."""

    SPLIT_TO_FOLDER = {"train": "train", "eval": "test", "test": "test", "val": "val"}
    LOW_CANDIDATES = ("low", "Low", "input", "dark", "lowlight")
    HIGH_CANDIDATES = ("high", "High", "normal", "Normal", "gt", "GT", "target", "reference")

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image], PairTensor]] = None,
        low_subdir: Optional[str] = None,
        high_subdir: Optional[str] = None,
    ) -> None:
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'")
        self.data_root = Path(data_root)
        self.split = split

        if low_subdir is not None and high_subdir is not None:
            low_dir = self.data_root / low_subdir
            high_dir = self.data_root / high_subdir
        else:
            split_dir = _find_subdir(
                self.data_root,
                [self.SPLIT_TO_FOLDER[split], self.SPLIT_TO_FOLDER[split].lower()],
            )
            if split_dir is None:
                split_dir = self.data_root
            low_dir = _find_subdir(split_dir, self.LOW_CANDIDATES)
            high_dir = _find_subdir(split_dir, self.HIGH_CANDIDATES)

        if low_dir is None or high_dir is None:
            raise FileNotFoundError(
                f"LoLI-Street 의 low/high 폴더를 찾을 수 없습니다.\n"
                f"  data_root : {self.data_root}\n  split : {split}\n"
                f"  hint : low_subdir / high_subdir 인자로 직접 지정 가능."
            )
        super().__init__(
            low_dir=low_dir, high_dir=high_dir,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            name=f"LoLI-Street[{split}]",
        )


# ===========================================================================
# 5. CombinedDataset
# ===========================================================================
class CombinedDataset(ConcatDataset):
    """여러 페어 데이터셋을 ``ConcatDataset`` 으로 합침."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        if not datasets:
            raise ValueError("CombinedDataset 은 빈 리스트를 받을 수 없습니다.")
        super().__init__(list(datasets))

    def per_dataset_lengths(self) -> List[int]:
        prev = 0
        out: List[int] = []
        for c in self.cumulative_sizes:
            out.append(c - prev)
            prev = c
        return out


# ===========================================================================
# 6. 팩토리
# ===========================================================================
DATASET_REGISTRY = {
    "lol_v1":      LOLDataset,
    "lol_v2_real": LOLv2RealDataset,
    "lol_v2_syn":  LOLv2SyntheticDataset,
    "loli_street": LoLIStreetDataset,
}


def build_dataset_by_name(
    name: str,
    data_root: str | os.PathLike,
    split: str = "train",
    image_size: int = 256,
    augment: bool = True,
    full_resize: bool = False,
    **kwargs,
) -> Dataset:
    """문자열 키 → 데이터셋 인스턴스."""
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. available: {list(DATASET_REGISTRY)}")
    cls = DATASET_REGISTRY[name]
    return cls(
        data_root=data_root, split=split, image_size=image_size,
        augment=augment, full_resize=full_resize, **kwargs,
    )

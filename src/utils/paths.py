"""경로 해석기 — ``configs/paths.yaml`` 한 곳에서 모든 절대경로를 만든다.

설계
----
* ``paths.yaml`` 의 ``root`` (절대경로) + 각 항목의 상대경로 → 절대 ``Path``.
* 코드 어디에서도 데이터/가중치 경로를 하드코딩하지 않고 ``load_paths()`` 로만
  접근한다. 머신 이동 시 ``paths.yaml`` 의 ``root`` 한 줄만 고치면 된다.
* 외부 의존성은 ``pyyaml`` 뿐.

사용 예
-------
>>> from src.utils.paths import load_paths
>>> P = load_paths()
>>> P.exdark           # WindowsPath('.../DataSet/ExDark')
>>> P.yolov8n          # WindowsPath('.../CODES/yolov8n.pt')
>>> P.luna_original    # WindowsPath('.../checkpoints/ext_lol_v2_real_stage2_best.pth')
>>> P.dataset("lol_v1")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

# 이 파일: LUNA2/src/utils/paths.py  →  LUNA2/ 는 parents[2]
_LUNA2_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATHS_YAML = _LUNA2_ROOT / "configs" / "paths.yaml"


@dataclass
class ResolvedPaths:
    """``paths.yaml`` 을 절대경로로 해석한 결과 묶음.

    Attributes
    ----------
    root : Path
        원본 SmallSizePM_GAN_model 루트.
    datasets / weights / outputs : dict[str, Path]
        키별 절대경로 매핑.
    """

    root: Path
    luna2_root: Path
    datasets: Dict[str, Path] = field(default_factory=dict)
    weights: Dict[str, Path] = field(default_factory=dict)
    outputs: Dict[str, Path] = field(default_factory=dict)
    phase2: Dict[str, Path] = field(default_factory=dict)

    # ----- 데이터셋 편의 접근자 -----
    def dataset(self, key: str) -> Path:
        if key not in self.datasets:
            raise KeyError(f"datasets['{key}'] 미정의. 가능: {list(self.datasets)}")
        return self.datasets[key]

    @property
    def exdark(self) -> Path:
        return self.dataset("exdark")

    @property
    def lol_v1(self) -> Path:
        return self.dataset("lol_v1")

    @property
    def lol_v2(self) -> Path:
        return self.dataset("lol_v2")

    @property
    def loli_street(self) -> Path:
        return self.dataset("loli_street")

    # ----- 가중치 편의 접근자 -----
    @property
    def yolov8n(self) -> Path:
        return self.weights["yolov8n"]

    @property
    def luna_original(self) -> Path:
        return self.weights["luna_original"]

    @property
    def luna_lolv2(self) -> Path:
        """LOL-v2 Real 학습 체크포인트 (ExDark mAP@0.5 = 0.282)."""
        return self.weights["luna_lolv2"]

    @property
    def luna_loli30k(self) -> Path:
        """LoLI-Street 30K 학습 체크포인트 (ExDark mAP@0.5 ≈ 0.346)."""
        return self.weights["luna_loli30k"]

    # ----- 산출물 -----
    @property
    def runs(self) -> Path:
        return self.outputs.get("runs", self.luna2_root / "runs")

    @property
    def p2_base(self) -> Path:
        """Phase 2 detection-aware 학습 시작 체크포인트 (= guidefix)."""
        return self.phase2["base_checkpoint"]


def load_paths(config_path: Optional[Path | str] = None) -> ResolvedPaths:
    """``paths.yaml`` → 절대경로로 해석한 :class:`ResolvedPaths`.

    Parameters
    ----------
    config_path : Path | str | None
        paths.yaml 경로. None 이면 ``LUNA2/configs/paths.yaml``.

    Notes
    -----
    존재 여부 검증은 하지 않는다 (경로 정의 ≠ 파일 존재). 실제 사용 지점에서
    각 스크립트가 검증하도록 한다.
    """
    cfg_path = Path(config_path) if config_path else _DEFAULT_PATHS_YAML
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["root"]).resolve()

    def _resolve_map(section: str) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for key, rel in (cfg.get(section) or {}).items():
            p = Path(rel)
            out[key] = p if p.is_absolute() else (root / p)
        return out

    # outputs / phase2 는 LUNA2 루트 기준 (산출물은 LUNA2 안에 둔다)
    def _resolve_luna2(section: str) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for key, rel in (cfg.get(section) or {}).items():
            p = Path(rel)
            out[key] = p if p.is_absolute() else (_LUNA2_ROOT / p)
        return out

    return ResolvedPaths(
        root=root,
        luna2_root=_LUNA2_ROOT,
        datasets=_resolve_map("datasets"),
        weights=_resolve_map("weights"),
        outputs=_resolve_luna2("outputs"),
        phase2=_resolve_luna2("phase2"),
    )

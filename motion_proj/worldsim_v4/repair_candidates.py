"""M2 修复候选与外部 Gaussian 资产绑定。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


REPAIR_METHODS = ("OBSERVED", "DONOR", "GENERATED")


def _unit_interval(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return number


@dataclass(frozen=True)
class GaussianAssetBinding:
    """内容寻址的外部 Gaussian delta；候选对象不复制 base checkpoint。"""

    path: str
    sha256: str
    bytes: int
    gaussian_count: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("Gaussian asset path must be non-empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("Gaussian asset sha256 must be lowercase hexadecimal")
        if int(self.bytes) <= 0:
            raise ValueError("Gaussian asset bytes must be positive")
        if int(self.gaussian_count) <= 0:
            raise ValueError("Gaussian asset gaussian_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": int(self.bytes),
            "gaussian_count": int(self.gaussian_count),
        }


@dataclass(frozen=True)
class RepairCandidate:
    """与 V4 计划一致的统一 repair candidate。"""

    candidate_id: str
    method: str
    gaussians: GaussianAssetBinding
    photo_risk: float
    geometry_risk: float
    temporal_risk: float
    uncertainty: float
    compute_cost: float
    provenance: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.method not in REPAIR_METHODS:
            raise ValueError(f"unsupported repair method: {self.method}")
        if not self.provenance:
            raise ValueError("provenance must be non-empty")
        for name in (
            "photo_risk",
            "geometry_risk",
            "temporal_risk",
            "uncertainty",
            "compute_cost",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name=name))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "gaussians": self.gaussians.to_dict(),
            "photo_risk": self.photo_risk,
            "geometry_risk": self.geometry_risk,
            "temporal_risk": self.temporal_risk,
            "uncertainty": self.uncertainty,
            "compute_cost": self.compute_cost,
            "provenance": self.provenance,
            "evidence": dict(self.evidence),
        }

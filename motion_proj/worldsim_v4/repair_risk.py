"""M2 Bayes-style repair risk 的可审计分解。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .repair_candidates import RepairCandidate


@dataclass(frozen=True)
class RepairRiskWeights:
    photo: float
    geometry: float
    temporal: float
    uncertainty: float
    compute_cost: float

    def __post_init__(self) -> None:
        values = (
            self.photo,
            self.geometry,
            self.temporal,
            self.uncertainty,
            self.compute_cost,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("repair risk weights must be finite and non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one repair risk weight must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "photo": float(self.photo),
            "geometry": float(self.geometry),
            "temporal": float(self.temporal),
            "uncertainty": float(self.uncertainty),
            "compute_cost": float(self.compute_cost),
        }


@dataclass(frozen=True)
class RepairRiskScore:
    candidate_id: str
    method: str
    photo: float
    geometry: float
    temporal: float
    uncertainty: float
    compute_cost: float
    total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "components": {
                "photo": self.photo,
                "geometry": self.geometry,
                "temporal": self.temporal,
                "uncertainty": self.uncertainty,
                "compute_cost": self.compute_cost,
            },
            "total": self.total,
        }


def score_repair_candidate(
    candidate: RepairCandidate, weights: RepairRiskWeights
) -> RepairRiskScore:
    components = {
        "photo": weights.photo * candidate.photo_risk,
        "geometry": weights.geometry * candidate.geometry_risk,
        "temporal": weights.temporal * candidate.temporal_risk,
        "uncertainty": weights.uncertainty * candidate.uncertainty,
        "compute_cost": weights.compute_cost * candidate.compute_cost,
    }
    return RepairRiskScore(
        candidate_id=candidate.candidate_id,
        method=candidate.method,
        total=float(math.fsum(components.values())),
        **components,
    )

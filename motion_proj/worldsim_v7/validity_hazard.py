"""有效性与危险性的显式因子化接口。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -40.0, 40.0))))


@dataclass(frozen=True)
class ValidityFeatures:
    """只描述证据、几何与时序一致性，不含危险变量。"""

    free_space_violation_rate: float
    surface_residual_m: float
    temporal_jitter_m: float
    unknown_fraction: float
    provenance_support: float
    ray_termination_consistency: float

    def __post_init__(self) -> None:
        unit_values = (
            self.free_space_violation_rate,
            self.unknown_fraction,
            self.provenance_support,
            self.ray_termination_consistency,
        )
        if any(value < 0.0 or value > 1.0 for value in unit_values):
            raise ValueError("rate-like validity features must be in [0, 1]")
        if self.surface_residual_m < 0.0 or self.temporal_jitter_m < 0.0:
            raise ValueError("validity distances must be non-negative")


@dataclass(frozen=True)
class HazardFeatures:
    """只描述 Actor 动力学与 Ego 交互，不含几何有效性变量。"""

    minimum_ttc_s: float
    minimum_clearance_m: float
    closing_speed_mps: float
    hard_brake_score: float
    crossing_probability: float

    def __post_init__(self) -> None:
        if self.minimum_ttc_s < 0.0 or self.minimum_clearance_m < 0.0:
            raise ValueError("TTC and clearance must be non-negative")
        if not 0.0 <= self.hard_brake_score <= 1.0:
            raise ValueError("hard_brake_score must be in [0, 1]")
        if not 0.0 <= self.crossing_probability <= 1.0:
            raise ValueError("crossing_probability must be in [0, 1]")


@dataclass(frozen=True)
class FactorizedScores:
    artifact_probability: float
    hazard_probability: float


class ValidityHazardFactorizer:
    """低容量确定性基线；学习模型也必须保持相同输入边界。"""

    def score_validity(self, features: ValidityFeatures) -> float:
        signal = (
            -3.0
            + 5.0 * features.free_space_violation_rate
            + 2.0 * min(features.surface_residual_m / 0.5, 2.0)
            + 1.5 * min(features.temporal_jitter_m / 0.25, 2.0)
            + 2.0 * features.unknown_fraction
            + 2.0 * (1.0 - features.provenance_support)
            + 2.0 * (1.0 - features.ray_termination_consistency)
        )
        return _sigmoid(signal)

    def score_hazard(self, features: HazardFeatures) -> float:
        inverse_ttc = 1.0 / max(features.minimum_ttc_s, 0.25)
        inverse_clearance = 1.0 / max(features.minimum_clearance_m, 0.25)
        signal = (
            -4.0
            + 2.5 * inverse_ttc
            + 1.2 * inverse_clearance
            + 0.12 * max(features.closing_speed_mps, 0.0)
            + 2.0 * features.hard_brake_score
            + 2.0 * features.crossing_probability
        )
        return _sigmoid(signal)

    def score(
        self, validity: ValidityFeatures, hazard: HazardFeatures
    ) -> FactorizedScores:
        return FactorizedScores(
            artifact_probability=self.score_validity(validity),
            hazard_probability=self.score_hazard(hazard),
        )


def paired_conditional_invariance(
    safe_scores: Sequence[FactorizedScores],
    hazard_scores: Sequence[FactorizedScores],
    clean_scores: Sequence[FactorizedScores],
    artifact_scores: Sequence[FactorizedScores],
) -> dict[str, float]:
    """计算 safe→hazard 与 clean→artifact 的交叉泄漏。"""

    lengths = {len(safe_scores), len(hazard_scores), len(clean_scores), len(artifact_scores)}
    if len(lengths) != 1 or not safe_scores:
        raise ValueError("paired score collections must be non-empty and aligned")
    safe_artifact = np.asarray([item.artifact_probability for item in safe_scores])
    hazard_artifact = np.asarray([item.artifact_probability for item in hazard_scores])
    clean_hazard = np.asarray([item.hazard_probability for item in clean_scores])
    artifact_hazard = np.asarray([item.hazard_probability for item in artifact_scores])
    return {
        "safe_to_hazard_artifact_score_shift": float(
            np.mean(np.abs(safe_artifact - hazard_artifact))
        ),
        "clean_to_artifact_hazard_score_shift": float(
            np.mean(np.abs(clean_hazard - artifact_hazard))
        ),
    }

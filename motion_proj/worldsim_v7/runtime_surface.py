"""预算与时域单调的运行时可靠性曲面。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from motion_proj.worldsim_v7.boundary_cost_density import LogCostMixtureDensity


@dataclass(frozen=True)
class ReliabilitySurface:
    horizons_s: np.ndarray
    budgets: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        horizons = np.asarray(self.horizons_s, dtype=np.float64)
        budgets = np.asarray(self.budgets, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if horizons.ndim != 1 or budgets.ndim != 1:
            raise ValueError("horizons and budgets must be one-dimensional")
        if len(horizons) == 0 or len(budgets) == 0:
            raise ValueError("horizons and budgets must be non-empty")
        if np.any(np.diff(horizons) <= 0.0) or np.any(np.diff(budgets) <= 0.0):
            raise ValueError("horizons and budgets must be strictly increasing")
        if probabilities.shape != (len(horizons), len(budgets)):
            raise ValueError("probabilities must have shape (H, B)")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("probabilities must be in [0, 1]")
        if np.any(np.diff(probabilities, axis=1) < -1e-9):
            raise ValueError("reliability must not decrease with budget")
        if np.any(np.diff(probabilities, axis=0) > 1e-9):
            raise ValueError("reliability must not increase with horizon")
        object.__setattr__(self, "horizons_s", horizons)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def project_monotone(
        cls,
        horizons_s: np.ndarray,
        budgets: np.ndarray,
        raw_probabilities: np.ndarray,
    ) -> "ReliabilitySurface":
        values = np.clip(np.asarray(raw_probabilities, dtype=np.float64), 0.0, 1.0)
        values = np.maximum.accumulate(values, axis=1)
        values = np.minimum.accumulate(values, axis=0)
        return cls(horizons_s, budgets, values)

    @classmethod
    def from_densities(
        cls,
        horizons_s: np.ndarray,
        budgets: np.ndarray,
        densities: Sequence[LogCostMixtureDensity],
    ) -> "ReliabilitySurface":
        if len(densities) != len(horizons_s):
            raise ValueError("one density is required for each horizon")
        raw = np.stack([density.cdf(budgets) for density in densities], axis=0)
        return cls.project_monotone(horizons_s, budgets, raw)

    def query(self, horizon_s: float, budget: float) -> float:
        """对冻结网格做保持单调性的双线性插值。"""

        if budget < 0.0:
            raise ValueError("budget must be non-negative")
        horizon = float(np.clip(horizon_s, self.horizons_s[0], self.horizons_s[-1]))
        local_budget = float(np.clip(budget, self.budgets[0], self.budgets[-1]))
        by_horizon = np.asarray(
            [np.interp(local_budget, self.budgets, row) for row in self.probabilities]
        )
        return float(np.interp(horizon, self.horizons_s, by_horizon))


def fit_binary_isotonic_map(
    scores: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """用 PAV 拟合无超参数的一维概率校准映射。"""

    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) == 0:
        raise ValueError("scores and targets must be aligned non-empty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("scores and targets must be finite")
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("binary targets must be in [0, 1]")
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    sorted_y = y[order]
    thresholds, inverse, counts = np.unique(
        sorted_x, return_inverse=True, return_counts=True
    )
    tied_means = np.bincount(inverse, weights=sorted_y) / counts
    blocks: list[list[float]] = []
    for index, (value, count) in enumerate(zip(tied_means, counts)):
        blocks.append([float(index), float(index), float(count), float(value)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            mass = left[2] + right[2]
            mean = (left[2] * left[3] + right[2] * right[3]) / mass
            blocks.append([left[0], right[1], mass, mean])
    calibrated = np.empty(len(thresholds), dtype=np.float64)
    for start, end, _, value in blocks:
        calibrated[int(start) : int(end) + 1] = value
    return thresholds, calibrated


def apply_binary_isotonic_map(
    scores: np.ndarray, thresholds: np.ndarray, values: np.ndarray
) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float64)
    thresholds_array = np.asarray(thresholds, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    if (
        thresholds_array.ndim != 1
        or len(thresholds_array) == 0
        or values_array.shape != thresholds_array.shape
    ):
        raise ValueError("thresholds and values must be aligned non-empty vectors")
    if np.any(np.diff(thresholds_array) <= 0.0):
        raise ValueError("thresholds must be strictly increasing")
    indices = np.searchsorted(thresholds_array, scores_array, side="right") - 1
    return values_array[np.clip(indices, 0, len(values_array) - 1)]

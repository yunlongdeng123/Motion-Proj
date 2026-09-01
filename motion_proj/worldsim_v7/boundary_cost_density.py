"""连续轨迹边界代价与条件 log-cost 密度。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def boundary_state_cost(
    residual_xy_m: np.ndarray,
    boundary_normals_xy: np.ndarray,
    signed_clearance_m: np.ndarray,
    clearance_floor_m: float = 0.05,
) -> np.ndarray:
    """计算 max_t |n_t^T epsilon_t| / max(|d_t|, floor)。"""

    residual = np.asarray(residual_xy_m, dtype=np.float64)
    normals = np.asarray(boundary_normals_xy, dtype=np.float64)
    clearance = np.asarray(signed_clearance_m, dtype=np.float64)
    if residual.shape[-1] != 2 or normals.shape[-1] != 2:
        raise ValueError("residuals and normals must end in XY coordinates")
    if residual.shape[-2:] != normals.shape[-2:]:
        raise ValueError("residuals and normals must share horizon and XY dimensions")
    if clearance.shape[-1] != residual.shape[-2]:
        raise ValueError("clearance must share the horizon dimension")
    if clearance_floor_m <= 0.0:
        raise ValueError("clearance_floor_m must be positive")
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("boundary normals must be non-zero")
    unit = normals / norm
    projected_error = np.abs(np.sum(unit * residual, axis=-1))
    denominator = np.maximum(np.abs(clearance), float(clearance_floor_m))
    return np.max(projected_error / denominator, axis=-1)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    output = np.fromiter(
        (0.5 * math.erfc(-float(value) / math.sqrt(2.0)) for value in flat),
        dtype=np.float64,
        count=len(flat),
    )
    return output.reshape(np.asarray(values).shape)


@dataclass(frozen=True)
class LogCostMixtureDensity:
    """单个条件查询的 Gaussian mixture over log(1 + C)。"""

    weights: np.ndarray
    log_cost_means: np.ndarray
    log_cost_scales: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.log_cost_means, dtype=np.float64)
        scales = np.asarray(self.log_cost_scales, dtype=np.float64)
        if weights.ndim != 1 or weights.shape != means.shape or weights.shape != scales.shape:
            raise ValueError("mixture parameters must be aligned one-dimensional vectors")
        if len(weights) == 0 or np.any(weights < 0.0) or not np.isfinite(weights).all():
            raise ValueError("mixture weights must be finite and non-negative")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("mixture weights must have positive mass")
        if np.any(scales <= 0.0):
            raise ValueError("log_cost_scales must be positive")
        object.__setattr__(self, "weights", weights / total)
        object.__setattr__(self, "log_cost_means", means)
        object.__setattr__(self, "log_cost_scales", scales)

    @classmethod
    def from_logits(
        cls, logits: np.ndarray, means: np.ndarray, scales: np.ndarray
    ) -> "LogCostMixtureDensity":
        logits_array = np.asarray(logits, dtype=np.float64)
        shifted = logits_array - np.max(logits_array)
        weights = np.exp(shifted)
        return cls(weights, means, scales)

    def cdf(self, budget: float | np.ndarray) -> np.ndarray:
        budgets = np.asarray(budget, dtype=np.float64)
        if np.any(budgets < 0.0):
            raise ValueError("reliability budgets must be non-negative")
        target = np.log1p(budgets)[..., None]
        standardized = (target - self.log_cost_means) / self.log_cost_scales
        return np.sum(self.weights * _normal_cdf(standardized), axis=-1)

    def expected_cost(self) -> float:
        expected_one_plus_cost = np.sum(
            self.weights
            * np.exp(self.log_cost_means + 0.5 * np.square(self.log_cost_scales))
        )
        return float(max(expected_one_plus_cost - 1.0, 0.0))

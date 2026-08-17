"""Audit the proposed V5.1 Kish effective-count concentration cap."""

from __future__ import annotations

import numpy as np


def kish_effective_count(
    weight_sum: np.ndarray,
    squared_weight_sum: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    weight_sum = np.asarray(weight_sum, dtype=np.float64)
    squared_weight_sum = np.asarray(squared_weight_sum, dtype=np.float64)
    if weight_sum.shape != squared_weight_sum.shape:
        raise ValueError("weight sums must have matching shapes")
    if not np.isfinite(weight_sum).all() or np.any(weight_sum < 0.0):
        raise ValueError("weight_sum must be finite and nonnegative")
    if not np.isfinite(squared_weight_sum).all() or np.any(
        squared_weight_sum < 0.0
    ):
        raise ValueError("squared_weight_sum must be finite and nonnegative")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    result = np.zeros_like(weight_sum)
    np.divide(
        np.square(weight_sum),
        squared_weight_sum + epsilon,
        out=result,
        where=weight_sum > 0.0,
    )
    return result


def audit_fractional_concentration_cap(
    weight_sum: np.ndarray,
    squared_weight_sum: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Compare A3-0 fractional mass with the proposed A3-1 Kish cap."""

    raw = np.asarray(weight_sum, dtype=np.float64)
    squared = np.asarray(squared_weight_sum, dtype=np.float64)
    kish_without_epsilon = kish_effective_count(raw, squared, epsilon=0.0)
    kish = kish_effective_count(raw, squared, epsilon=epsilon)
    capped = np.minimum(raw, kish)
    return {
        "fractional_concentration": raw,
        "kish_effective_count_without_epsilon": kish_without_epsilon,
        "kish_effective_count": kish,
        "capped_concentration": capped,
        "cap_reduction": raw - capped,
        "replacement_amplification": kish - raw,
    }

"""V5.1 semantic UNKNOWN state and selective-risk accounting."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def posterior_entropy(probability: np.ndarray) -> np.ndarray:
    """Return normalized Bernoulli entropy in [0, 1]."""

    value = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("probability must be finite and within [0,1]")
    clipped = np.clip(value, 1e-12, 1.0 - 1e-12)
    entropy = -(
        clipped * np.log2(clipped)
        + (1.0 - clipped) * np.log2(1.0 - clipped)
    )
    return entropy.astype(np.float64)


def build_semantic_unknown_state(
    *,
    conditional_actor_probability: np.ndarray,
    effective_observation_count: np.ndarray,
    cross_view_disagreement: np.ndarray,
    effective_count_maximum: float,
    entropy_minimum: float,
    disagreement_minimum: float,
) -> dict[str, np.ndarray]:
    """Split A1 conditional posterior into ACTOR/BACKGROUND/UNKNOWN mass.

    UNKNOWN is deliberately conjunctive: high entropy is required, together
    with either sparse effective evidence or high cross-view disagreement.
    This prevents zero-count but decisive frozen priors from becoming UNKNOWN.
    """

    actor_conditional = np.asarray(
        conditional_actor_probability, dtype=np.float64
    )
    count = np.asarray(effective_observation_count, dtype=np.float64)
    disagreement = np.asarray(cross_view_disagreement, dtype=np.float64)
    if not (actor_conditional.shape == count.shape == disagreement.shape):
        raise ValueError("A2 posterior/count/disagreement shapes must match")
    if actor_conditional.ndim != 1:
        raise ValueError("A2 Gaussian arrays must be one-dimensional")
    if not np.isfinite(count).all() or np.any(count < 0.0):
        raise ValueError("effective_observation_count must be finite and nonnegative")
    if not np.isfinite(disagreement).all() or np.any(
        (disagreement < 0.0) | (disagreement > 1.0)
    ):
        raise ValueError("cross_view_disagreement must be finite and within [0,1]")
    for name, value, lower, upper in (
        ("effective_count_maximum", effective_count_maximum, 0.0, np.inf),
        ("entropy_minimum", entropy_minimum, 0.0, 1.0),
        ("disagreement_minimum", disagreement_minimum, 0.0, 1.0),
    ):
        if not np.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"{name} is outside its valid range")

    entropy = posterior_entropy(actor_conditional)
    low_count = count <= effective_count_maximum
    high_entropy = entropy >= entropy_minimum
    high_disagreement = disagreement >= disagreement_minimum
    unknown = high_entropy & (low_count | high_disagreement)
    unknown_probability = unknown.astype(np.float32)
    known_probability = 1.0 - unknown_probability
    posterior_actor = (
        known_probability * actor_conditional.astype(np.float32)
    ).astype(np.float32)
    posterior_background = (
        known_probability * (1.0 - actor_conditional).astype(np.float32)
    ).astype(np.float32)
    return {
        "posterior_actor": posterior_actor,
        "posterior_background": posterior_background,
        "unknown_probability": unknown_probability,
        "effective_observation_count": count.astype(np.float32),
        "conditional_actor_probability": actor_conditional.astype(np.float32),
        "posterior_entropy": entropy.astype(np.float32),
        "low_effective_count": low_count.astype(np.int8),
        "high_posterior_entropy": high_entropy.astype(np.int8),
        "high_cross_view_disagreement": high_disagreement.astype(np.int8),
    }


def selective_semantic_statistics(
    probability: np.ndarray,
    target: np.ndarray,
    unknown_probability: np.ndarray,
    *,
    probability_threshold: float,
    abstain_threshold: float,
) -> dict[str, float | int]:
    """Collect denominator-preserving sufficient statistics for selective risk."""

    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    unknown_probability = np.asarray(unknown_probability, dtype=np.float64)
    if not (probability.shape == target.shape == unknown_probability.shape):
        raise ValueError("probability/target/unknown_probability shapes must match")
    if probability.size == 0:
        raise ValueError("selective metric denominator must be non-empty")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("probability must be finite and within [0,1]")
    if not np.isfinite(unknown_probability).all() or np.any(
        (unknown_probability < 0.0) | (unknown_probability > 1.0)
    ):
        raise ValueError("unknown_probability must be finite and within [0,1]")
    if not np.isfinite(target).all() or np.any((target != 0.0) & (target != 1.0)):
        raise ValueError("target must be binary")
    for name, value in (
        ("probability_threshold", probability_threshold),
        ("abstain_threshold", abstain_threshold),
    ):
        if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
            raise ValueError(f"{name} must be finite and in (0,1)")

    abstained = unknown_probability >= abstain_threshold
    accepted = ~abstained
    binary_error = (probability >= probability_threshold) != target.astype(bool)
    absolute_error = np.abs(probability - target)
    squared_error = np.square(probability - target)
    return {
        "total_pixel_count": int(probability.size),
        "accepted_pixel_count": int(accepted.sum()),
        "abstained_pixel_count": int(abstained.sum()),
        "binary_error_pixel_count": int(binary_error.sum()),
        "accepted_binary_error_pixel_count": int((accepted & binary_error).sum()),
        "abstained_binary_error_pixel_count": int((abstained & binary_error).sum()),
        "accepted_absolute_error_sum": float(absolute_error[accepted].sum()),
        "abstained_absolute_error_sum": float(absolute_error[abstained].sum()),
        "accepted_squared_error_sum": float(squared_error[accepted].sum()),
    }


def merge_selective_semantic_statistics(
    rows: list[Mapping[str, float | int]],
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("selective statistics rows must be non-empty")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("selective statistics schemas must match")
    integer_keys = {name for name in keys if name.endswith("_count")}
    return {
        name: (
            int(sum(int(row[name]) for row in rows))
            if name in integer_keys
            else float(sum(float(row[name]) for row in rows))
        )
        for name in keys
    }


def _safe_ratio(numerator: float, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def finalize_selective_semantic_metrics(
    statistics: Mapping[str, float | int],
) -> dict[str, Any]:
    """Finalize coverage and error concentration without dropping abstentions."""

    total = int(statistics["total_pixel_count"])
    accepted = int(statistics["accepted_pixel_count"])
    abstained = int(statistics["abstained_pixel_count"])
    errors = int(statistics["binary_error_pixel_count"])
    if total <= 0 or accepted + abstained != total:
        raise ValueError("selective denominator accounting is inconsistent")
    return {
        "coverage": float(accepted / total),
        "unknown_ratio": float(abstained / total),
        "error_at_coverage": _safe_ratio(
            float(statistics["accepted_binary_error_pixel_count"]), accepted
        ),
        "selective_semantic_risk": _safe_ratio(
            float(statistics["accepted_squared_error_sum"]), accepted
        ),
        "unknown_precision": _safe_ratio(
            float(statistics["abstained_binary_error_pixel_count"]), abstained
        ),
        "unknown_recall_on_errors": _safe_ratio(
            float(statistics["abstained_binary_error_pixel_count"]), errors
        ),
        "accepted_subset_error": _safe_ratio(
            float(statistics["accepted_absolute_error_sum"]), accepted
        ),
        "abstained_subset_error": _safe_ratio(
            float(statistics["abstained_absolute_error_sum"]), abstained
        ),
        "denominators": {
            "total_pixel_count": total,
            "accepted_pixel_count": accepted,
            "abstained_pixel_count": abstained,
            "binary_error_pixel_count": errors,
        },
    }

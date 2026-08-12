"""Probability and mask metrics used by WorldSim V4 M1 gates."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _paired(
    probability: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probability/target must have the same non-empty shape")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("probability/target must be finite")
    if np.any((p < 0.0) | (p > 1.0)) or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("probability/target must lie in [0, 1]")
    return p.reshape(-1), y.reshape(-1)


def brier_score(probability: np.ndarray, target: np.ndarray) -> float:
    p, y = _paired(probability, target)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probability: np.ndarray, target: np.ndarray, *, bins: int = 15
) -> float:
    p, y = _paired(probability, target)
    if bins <= 0:
        raise ValueError("bins must be positive")
    indices = np.minimum((p * bins).astype(np.int64), bins - 1)
    total = p.size
    result = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            result += (selected.sum() / total) * abs(
                float(p[selected].mean() - y[selected].mean())
            )
    return float(result)


def probability_mass_metrics(
    probability: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    p, y = _paired(probability, target)
    positive = y
    negative = 1.0 - y
    return {
        "false_positive_semantic_mass": float((p * negative).sum() / max(negative.sum(), 1.0)),
        "false_negative_semantic_mass": float(((1.0 - p) * positive).sum() / max(positive.sum(), 1.0)),
    }


def probability_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float]:
    output = probability_mass_metrics(probability, target)
    output.update(
        brier=brier_score(probability, target),
        ece=expected_calibration_error(probability, target, bins=bins),
    )
    return output


def aggregate_scene_metrics(
    rows: Sequence[Mapping[str, float]], *, metric_names: Sequence[str]
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty metric rows")
    output = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"metric {name} contains non-finite values")
        output[name] = float(values.mean())
    return output

"""Minimal algebraic contract for CIF-style occupancy/identity separation."""

from __future__ import annotations

import numpy as np


def compose_decoupled_actor_opacity(
    *,
    appearance_opacity: np.ndarray,
    occupancy_probability: np.ndarray,
    conditional_actor_probability: np.ndarray,
) -> np.ndarray:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            appearance_opacity,
            occupancy_probability,
            conditional_actor_probability,
        )
    ]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("appearance/occupancy/identity shapes must match")
    if arrays[0].ndim != 1:
        raise ValueError("decoupled Gaussian variables must be one-dimensional")
    for name, value in zip(
        ("appearance_opacity", "occupancy_probability", "conditional_identity"),
        arrays,
    ):
        if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} must be finite and within [0,1]")
    return (arrays[0] * arrays[1] * arrays[2]).astype(np.float32)

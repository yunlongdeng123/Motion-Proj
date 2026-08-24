"""Utilities for native IR-WM logits and source-grid mapping."""

from __future__ import annotations

import numpy as np


def native_uncertainty(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return argmax, entropy and top-one/top-two probability margin."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("native logits must end in at least two classes")
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    argmax = probs.argmax(axis=-1).astype(np.uint8)
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=-1)
    top2 = np.partition(probs, -2, axis=-1)[..., -2:]
    margin = top2.max(axis=-1) - top2.min(axis=-1)
    return argmax, entropy.astype(np.float16), margin.astype(np.float16)


def target_points_to_native_indices(
    points_m: np.ndarray,
    *,
    source_origin_m: np.ndarray,
    source_voxel_size_m: float,
    source_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map target-frame metric points to the frozen native grid."""
    points = np.asarray(points_m, dtype=np.float64)
    indices = np.floor(
        (points - np.asarray(source_origin_m, dtype=np.float64))
        / float(source_voxel_size_m)
    ).astype(np.int32)
    shape = np.asarray(source_shape, dtype=np.int32)
    valid = ((indices >= 0) & (indices < shape)).all(axis=-1)
    return indices, valid

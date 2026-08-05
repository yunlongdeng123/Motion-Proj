"""Deterministic region helpers for WorldSim V3 actor diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _require_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    return value


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask with a square kernel and zero padding."""

    value = _require_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return value.copy()
    padded = np.pad(value, radius, mode="constant", constant_values=False)
    height, width = value.shape
    result = np.zeros_like(value)
    for row in range(2 * radius + 1):
        for column in range(2 * radius + 1):
            result |= padded[row : row + height, column : column + width]
    return result


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a binary mask with a square kernel and zero padding."""

    value = _require_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return value.copy()
    padded = np.pad(value, radius, mode="constant", constant_values=False)
    height, width = value.shape
    result = np.ones_like(value)
    for row in range(2 * radius + 1):
        for column in range(2 * radius + 1):
            result &= padded[row : row + height, column : column + width]
    return result


def boundary_band(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return the symmetric inner/outer boundary band of ``mask``."""

    value = _require_mask(mask)
    return binary_dilate(value, radius) & ~binary_erode(value, radius)


def counterfactual_effect_mask(
    with_actor: np.ndarray,
    without_actor: np.ndarray,
    *,
    threshold_uint8: int = 2,
    dilation_radius: int = 2,
) -> np.ndarray:
    """Derive a diagnostic footprint from paired uint8 RGB renders."""

    first = np.asarray(with_actor)
    second = np.asarray(without_actor)
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError("paired RGB images must have the same HxWx3 shape")
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError("paired RGB images must use uint8 quantization")
    if not 0 <= threshold_uint8 <= 255:
        raise ValueError("threshold_uint8 must be in [0, 255]")
    difference = np.max(
        np.abs(first.astype(np.int16) - second.astype(np.int16)), axis=-1
    )
    return binary_dilate(difference >= threshold_uint8, dilation_radius)


def region_error_sums(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    """Return additive RGB error statistics for a region in [0, 1]."""

    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    selected = _require_mask(mask)
    if predicted.shape != expected.shape or predicted.ndim != 3:
        raise ValueError("prediction and target must share an HxWxC shape")
    if predicted.shape[:2] != selected.shape:
        raise ValueError("mask shape must match the image spatial shape")
    difference = predicted[selected] - expected[selected]
    return {
        "pixel_count": int(selected.sum()),
        "value_count": int(difference.size),
        "squared_error_sum": float(np.square(difference).sum()),
        "absolute_error_sum": float(np.abs(difference).sum()),
    }


def psnr_from_sums(squared_error_sum: float, value_count: int) -> float:
    if value_count <= 0:
        return float("nan")
    mse = float(squared_error_sum) / int(value_count)
    return float("inf") if mse == 0 else float(-10.0 * math.log10(mse))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")

"""M4 对象编辑闭环使用的确定性图像与几何诊断。"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def binary_dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    if radius < 0:
        raise ValueError("radius 不能为负数")
    if radius == 0:
        return value.copy()
    padded = np.pad(value, radius, mode="constant")
    result = np.zeros_like(value)
    height, width = value.shape
    for row in range(2 * radius + 1):
        for column in range(2 * radius + 1):
            result |= padded[row : row + height, column : column + width]
    return result


def counterfactual_effect_mask(
    with_actor: np.ndarray,
    without_actor: np.ndarray,
    *,
    threshold_uint8: int = 2,
    dilation_radius: int = 2,
) -> np.ndarray:
    first = np.asarray(with_actor)
    second = np.asarray(without_actor)
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError("两张 RGB 图必须具有相同的 HxWx3 形状")
    difference = np.max(
        np.abs(first.astype(np.int16) - second.astype(np.int16)), axis=-1
    )
    return binary_dilate(difference >= int(threshold_uint8), dilation_radius)


def masked_mean_abs(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("nan")
    difference = np.abs(
        np.asarray(reference, dtype=np.float32)
        - np.asarray(candidate, dtype=np.float32)
    ) / 255.0
    return float(difference[selected].mean())


def masked_psnr(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("nan")
    difference = (
        np.asarray(reference, dtype=np.float64)
        - np.asarray(candidate, dtype=np.float64)
    ) / 255.0
    mse = float(np.square(difference[selected]).mean())
    return float("inf") if mse == 0 else float(-10.0 * np.log10(mse))


def depth_order_diagnostic(
    edited_depth: np.ndarray,
    background_depth: np.ndarray,
    edited_footprint: np.ndarray,
    *,
    tolerance_m: float = 0.05,
) -> dict[str, float | int]:
    edited = np.asarray(edited_depth, dtype=np.float32)
    background = np.asarray(background_depth, dtype=np.float32)
    footprint = np.asarray(edited_footprint, dtype=bool)
    valid = footprint & np.isfinite(edited) & np.isfinite(background)
    valid &= (edited > 0) & (background > 0)
    count = int(valid.sum())
    if count == 0:
        return {
            "valid_pixel_count": 0,
            "actor_in_front_rate": float("nan"),
            "depth_ordering_violation_rate": float("nan"),
            "mean_depth_delta_m": float("nan"),
        }
    delta = background[valid] - edited[valid]
    in_front = delta > tolerance_m
    violation = delta < -tolerance_m
    return {
        "valid_pixel_count": count,
        "actor_in_front_rate": float(in_front.mean()),
        "depth_ordering_violation_rate": float(violation.mean()),
        "mean_depth_delta_m": float(delta.mean()),
    }


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def finite_max(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.max()) if finite.size else float("nan")

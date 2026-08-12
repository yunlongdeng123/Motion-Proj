"""连续 clip 的基础时序一致性指标。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _sequence(value: np.ndarray, *, name: str, minimum_ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < minimum_ndim or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite sequence with at least two frames")
    return array


def frame_flicker_l1(frames: np.ndarray) -> float:
    images = _sequence(frames, name="frames", minimum_ndim=2).astype(np.float64)
    return float(np.mean(np.abs(np.diff(images, axis=0))))


def warp_l1(
    current_frames: np.ndarray,
    warped_previous_frames: np.ndarray,
    valid_masks: np.ndarray | None = None,
) -> float:
    current = np.asarray(current_frames, dtype=np.float64)
    warped = np.asarray(warped_previous_frames, dtype=np.float64)
    if current.shape != warped.shape or current.size == 0:
        raise ValueError("current/warped sequences must have the same non-empty shape")
    residual = np.abs(current - warped)
    if valid_masks is None:
        return float(np.mean(residual))
    mask = np.asarray(valid_masks, dtype=bool)
    if mask.shape != residual.shape and mask.shape != residual.shape[:-1]:
        raise ValueError("valid mask shape does not match warp residual")
    if mask.shape == residual.shape[:-1]:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.broadcast_to(mask, residual.shape)
    if not mask.any():
        raise ValueError("valid warp mask is empty")
    return float(np.mean(residual[mask]))


def mask_iou_jitter(masks: np.ndarray) -> float:
    binary = _sequence(masks, name="masks", minimum_ndim=3).astype(bool)
    ious = []
    for before, after in zip(binary[:-1], binary[1:]):
        union = np.logical_or(before, after).sum()
        iou = 1.0 if union == 0 else np.logical_and(before, after).sum() / union
        ious.append(float(iou))
    return float(np.std(ious))


def centroid_acceleration_jitter(centroids: np.ndarray) -> float:
    points = _sequence(centroids, name="centroids", minimum_ndim=2).astype(np.float64)
    if points.ndim != 2 or points.shape[1] not in {2, 3} or points.shape[0] < 3:
        raise ValueError("centroids must have shape (T>=3, 2|3)")
    acceleration = np.diff(points, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(acceleration**2, axis=1))))


def identity_switch_count(track_ids: Sequence[int | str | None]) -> int:
    if len(track_ids) < 2:
        raise ValueError("track IDs require at least two frames")
    switches = 0
    previous = track_ids[0]
    for current in track_ids[1:]:
        if current is not None and previous is not None and current != previous:
            switches += 1
        if current is not None:
            previous = current
    return switches


def relative_error_improvement(baseline: float, candidate: float) -> float:
    before = float(baseline)
    after = float(candidate)
    if not math.isfinite(before) or not math.isfinite(after) or before <= 0.0:
        raise ValueError("relative error improvement requires finite baseline > 0")
    return (before - after) / before

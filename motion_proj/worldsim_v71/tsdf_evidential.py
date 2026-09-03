"""Actor-local TSDF/evidential fusion（V7.1 B4 非学习基线）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EvidentialTSDF:
    """稠密小型Actor网格；未观测cell保持UNKNOWN。"""

    origin_m: np.ndarray
    voxel_size_m: float
    tsdf: np.ndarray
    weight: np.ndarray


def _deterministic_limit(
    endpoints: np.ndarray, origins: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(endpoints) <= int(maximum):
        return endpoints, origins
    selected = np.linspace(0, len(endpoints) - 1, num=int(maximum), dtype=np.int64)
    return endpoints[selected], origins[selected]


def integrate_actor_tsdf(
    endpoint_parts: Iterable[np.ndarray],
    sensor_origins: Iterable[np.ndarray],
    size_lwh_m: np.ndarray,
    *,
    voxel_size_m: float = 0.12,
    truncation_m: float = 0.36,
    padding_m: float = 0.24,
    maximum_rays: int = 16384,
) -> EvidentialTSDF:
    """沿真实观测射线的窄截断带融合SDF，不填充未观测空间。"""
    points = [np.asarray(value, dtype=np.float32).reshape(-1, 3) for value in endpoint_parts]
    origins = [
        np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(part), axis=0)
        for part, origin in zip(points, sensor_origins)
    ]
    endpoints = np.concatenate(points, axis=0)
    ray_origins = np.concatenate(origins, axis=0)
    endpoints, ray_origins = _deterministic_limit(endpoints, ray_origins, maximum_rays)

    half_size = np.asarray(size_lwh_m, dtype=np.float32) * 0.5
    origin_m = -half_size - float(padding_m)
    extent = 2.0 * half_size + 2.0 * float(padding_m)
    shape = np.maximum(np.ceil(extent / float(voxel_size_m)).astype(np.int64), 2)
    value_sum = np.zeros(tuple(shape), dtype=np.float32)
    weight = np.zeros(tuple(shape), dtype=np.float32)

    vectors = endpoints - ray_origins
    ranges = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(ranges) & (ranges > 1.0e-4)
    endpoints = endpoints[valid]
    directions = vectors[valid] / ranges[valid, None]
    offsets = np.arange(
        -float(truncation_m),
        float(truncation_m) + 0.5 * float(voxel_size_m),
        float(voxel_size_m),
        dtype=np.float32,
    )
    samples = endpoints[:, None, :] + directions[:, None, :] * offsets[None, :, None]
    signed = np.broadcast_to((-offsets / float(truncation_m))[None, :], samples.shape[:2])
    indices = np.floor((samples - origin_m[None, None, :]) / float(voxel_size_m)).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < shape[None, None, :]), axis=2)
    selected_indices = indices[inside]
    selected_values = signed[inside].astype(np.float32)
    np.add.at(
        value_sum,
        (selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]),
        selected_values,
    )
    np.add.at(
        weight,
        (selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]),
        1.0,
    )
    tsdf = np.ones(tuple(shape), dtype=np.float32)
    observed = weight > 0.0
    tsdf[observed] = value_sum[observed] / weight[observed]
    return EvidentialTSDF(origin_m, float(voxel_size_m), tsdf, weight)


def extract_tsdf_zero_crossings(
    field: EvidentialTSDF,
    *,
    minimum_weight: float = 2.0,
    maximum_points: int = 8192,
) -> np.ndarray:
    """仅在两侧均有证据的相邻cell间提取zero crossing。"""
    points: list[np.ndarray] = []
    shape = np.asarray(field.tsdf.shape, dtype=np.int64)
    for axis in range(3):
        left_slices = [slice(None)] * 3
        right_slices = [slice(None)] * 3
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)
        left_slices_tuple = tuple(left_slices)
        right_slices_tuple = tuple(right_slices)
        left = field.tsdf[left_slices_tuple]
        right = field.tsdf[right_slices_tuple]
        known = (
            (field.weight[left_slices_tuple] >= float(minimum_weight))
            & (field.weight[right_slices_tuple] >= float(minimum_weight))
        )
        crossings = np.argwhere(known & (left * right <= 0.0))
        if not len(crossings):
            continue
        left_values = left[tuple(crossings.T)]
        right_values = right[tuple(crossings.T)]
        fraction = np.abs(left_values) / np.maximum(
            np.abs(left_values) + np.abs(right_values), 1.0e-6
        )
        centers = field.origin_m[None, :] + (
            crossings.astype(np.float32) + 0.5
        ) * float(field.voxel_size_m)
        centers[:, axis] += fraction * float(field.voxel_size_m)
        points.append(centers)
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    surface = np.concatenate(points, axis=0).astype(np.float32)
    keys = np.floor(surface / (0.5 * float(field.voxel_size_m))).astype(np.int64)
    _, selected = np.unique(keys, axis=0, return_index=True)
    surface = surface[np.sort(selected)]
    if len(surface) > int(maximum_points):
        selected = np.linspace(0, len(surface) - 1, num=int(maximum_points), dtype=np.int64)
        surface = surface[selected]
    return surface


def build_b4_surface(
    endpoint_parts: Iterable[np.ndarray],
    sensor_origins: Iterable[np.ndarray],
    observed_anchors: np.ndarray,
    size_lwh_m: np.ndarray,
    **kwargs: float | int,
) -> np.ndarray:
    """输出证据zero crossing与冻结observed anchors的并集。"""
    field = integrate_actor_tsdf(
        endpoint_parts,
        sensor_origins,
        size_lwh_m,
        voxel_size_m=float(kwargs.get("voxel_size_m", 0.12)),
        truncation_m=float(kwargs.get("truncation_m", 0.36)),
        padding_m=float(kwargs.get("padding_m", 0.24)),
        maximum_rays=int(kwargs.get("maximum_rays", 16384)),
    )
    crossing = extract_tsdf_zero_crossings(
        field,
        minimum_weight=float(kwargs.get("minimum_weight", 2.0)),
        maximum_points=int(kwargs.get("maximum_points", 8192)),
    )
    anchors = np.asarray(observed_anchors, dtype=np.float32).reshape(-1, 3)
    if not len(crossing):
        return anchors.copy()
    output = np.concatenate([anchors, crossing], axis=0)
    voxel = float(kwargs.get("output_voxel_size_m", 0.06))
    keys = np.floor(output / voxel).astype(np.int64)
    _, selected = np.unique(keys, axis=0, return_index=True)
    return output[np.sort(selected)].astype(np.float32)

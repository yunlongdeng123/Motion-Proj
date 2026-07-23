"""V7.1 连续 OBB 安全几何。

本模块只处理标注/状态给出的几何，不读取 Gaussian 或观测 occupancy。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, pi, sin
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    voxel_size: float

    @property
    def shape(self) -> tuple[int, int, int]:
        span = np.asarray(self.maximum) - np.asarray(self.minimum)
        return tuple(np.rint(span / self.voxel_size).astype(int))

    def centers_for_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        lo = np.floor((lower - np.asarray(self.minimum)) / self.voxel_size).astype(int)
        hi = np.ceil((upper - np.asarray(self.minimum)) / self.voxel_size).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.shape))
        axes = tuple(
            np.asarray(self.minimum)[axis]
            + (np.arange(lo[axis], hi[axis]) + 0.5) * self.voxel_size
            for axis in range(3)
        )
        if any(len(axis) == 0 for axis in axes):
            return np.empty((0, 3)), tuple(np.array([], dtype=int) for _ in range(3))
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack(mesh, axis=-1).reshape(-1, 3)
        indices = tuple(np.arange(lo[i], hi[i], dtype=int) for i in range(3))
        return points, indices


@dataclass(frozen=True)
class OrientedBox:
    center: tuple[float, float, float]
    dimensions_lwh: tuple[float, float, float]
    yaw: float
    actor_id: int | str | None = None

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.dimensions_lwh):
            raise ValueError("OBB dimensions 必须为正")
        if not np.isfinite([*self.center, *self.dimensions_lwh, self.yaw]).all():
            raise ValueError("OBB 不允许 NaN/Inf")

    @property
    def rotation(self) -> np.ndarray:
        c, s = cos(self.yaw), sin(self.yaw)
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def corners(self) -> np.ndarray:
        half = np.asarray(self.dimensions_lwh) / 2.0
        signs = np.asarray(
            [
                [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            ],
            dtype=float,
        )
        return signs * half @ self.rotation.T + np.asarray(self.center)

    def contains(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        local = (np.asarray(points) - np.asarray(self.center)) @ self.rotation
        return np.all(
            np.abs(local) <= np.asarray(self.dimensions_lwh) / 2.0 + tolerance,
            axis=-1,
        )


def voxelize_oriented_box(box: OrientedBox, grid: GridSpec) -> np.ndarray:
    """用 voxel center-in-OBB 规则生成真正的 oriented box mask。"""
    corners = box.corners()
    points, indices = grid.centers_for_bounds(corners.min(0), corners.max(0))
    mask = np.zeros(grid.shape, dtype=bool)
    if len(points) == 0:
        return mask
    local_shape = tuple(len(index) for index in indices)
    inside = box.contains(points).reshape(local_shape)
    mask[np.ix_(*indices)] = inside
    return mask


def voxelize_aabb_baseline(box: OrientedBox, grid: GridSpec) -> np.ndarray:
    """旧 O0 的角点 AABB 粗填充，仅供预注册消融。"""
    corners = box.corners()
    points, indices = grid.centers_for_bounds(corners.min(0), corners.max(0))
    mask = np.zeros(grid.shape, dtype=bool)
    if len(points):
        mask[np.ix_(*indices)] = True
    return mask


def _horizontal_axes(box: OrientedBox) -> list[np.ndarray]:
    rotation = box.rotation[:2, :2]
    return [rotation[:, 0], rotation[:, 1]]


def _projection_radius(box: OrientedBox, axis: np.ndarray) -> float:
    rotation = box.rotation[:2, :2]
    half = np.asarray(box.dimensions_lwh[:2]) / 2.0
    return float(
        half[0] * abs(np.dot(axis, rotation[:, 0]))
        + half[1] * abs(np.dot(axis, rotation[:, 1]))
    )


def obb_signed_separation(a: OrientedBox, b: OrientedBox) -> float:
    """返回 SAT 分离量；<=0 表示 3D OBB 相交。"""
    az0, az1 = a.center[2] - a.dimensions_lwh[2] / 2, a.center[2] + a.dimensions_lwh[2] / 2
    bz0, bz1 = b.center[2] - b.dimensions_lwh[2] / 2, b.center[2] + b.dimensions_lwh[2] / 2
    z_gap = max(az0 - bz1, bz0 - az1)
    delta = np.asarray(b.center[:2]) - np.asarray(a.center[:2])
    gaps = []
    for axis in _horizontal_axes(a) + _horizontal_axes(b):
        axis = axis / np.linalg.norm(axis)
        gaps.append(
            abs(float(np.dot(delta, axis)))
            - _projection_radius(a, axis)
            - _projection_radius(b, axis)
        )
    return max([z_gap, *gaps])


def obb_intersects(a: OrientedBox, b: OrientedBox, clearance_m: float = 0.0) -> bool:
    return obb_signed_separation(a, b) <= clearance_m


def _angle_delta(start: float, end: float) -> float:
    return (end - start + pi) % (2 * pi) - pi


def interpolate_box(start: OrientedBox, end: OrientedBox, fraction: float) -> OrientedBox:
    center = (1.0 - fraction) * np.asarray(start.center) + fraction * np.asarray(end.center)
    dimensions = (
        (1.0 - fraction) * np.asarray(start.dimensions_lwh)
        + fraction * np.asarray(end.dimensions_lwh)
    )
    return OrientedBox(
        tuple(center),
        tuple(dimensions),
        start.yaw + fraction * _angle_delta(start.yaw, end.yaw),
        start.actor_id,
    )


def swept_obb_collision(
    a0: OrientedBox,
    a1: OrientedBox,
    b0: OrientedBox,
    b1: OrientedBox,
    *,
    clearance_m: float = 0.0,
    max_translation_step_m: float = 0.1,
    max_yaw_step_rad: float = 0.02,
) -> dict:
    """保守离散连续 evaluator；步长由位移和 yaw 上界决定。"""
    translation = max(
        np.linalg.norm(np.asarray(a1.center) - np.asarray(a0.center)),
        np.linalg.norm(np.asarray(b1.center) - np.asarray(b0.center)),
    )
    yaw = max(abs(_angle_delta(a0.yaw, a1.yaw)), abs(_angle_delta(b0.yaw, b1.yaw)))
    steps = max(
        1,
        int(ceil(translation / max_translation_step_m)),
        int(ceil(yaw / max_yaw_step_rad)),
    )
    minimum = float("inf")
    collision_fraction = None
    for index in range(steps + 1):
        fraction = index / steps
        a = interpolate_box(a0, a1, fraction)
        b = interpolate_box(b0, b1, fraction)
        separation = obb_signed_separation(a, b)
        minimum = min(minimum, separation)
        if separation <= clearance_m and collision_fraction is None:
            collision_fraction = fraction
    return {
        "collision": collision_fraction is not None,
        "first_collision_fraction": collision_fraction,
        "minimum_signed_separation_m": minimum,
        "substeps": steps,
        "evaluator": "continuous-obb-adaptive-v1",
    }


def any_swept_collision(
    trajectory: Iterable[OrientedBox],
    others: Iterable[Iterable[OrientedBox]],
) -> bool:
    own = list(trajectory)
    for other_values in others:
        other = list(other_values)
        for index in range(min(len(own), len(other)) - 1):
            if swept_obb_collision(own[index], own[index + 1], other[index], other[index + 1])["collision"]:
                return True
    return False

"""CPU geometry helpers for counterfactual case qualification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def interpolate_transform(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate planar vehicle poses while preserving a valid rigid transform."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    yaw_a = math.atan2(float(first[1, 0]), float(first[0, 0]))
    yaw_b = math.atan2(float(second[1, 0]), float(second[0, 0]))
    delta = math.atan2(math.sin(yaw_b - yaw_a), math.cos(yaw_b - yaw_a))
    yaw = yaw_a + alpha * delta
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
    result[:3, :3] = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return result


def pose_at(poses: dict[int, np.ndarray], frame: float) -> np.ndarray | None:
    """Return a linearly interpolated pose for a fractional frame index."""

    lower = math.floor(frame)
    upper = math.ceil(frame)
    if lower not in poses or upper not in poses:
        return None
    if lower == upper:
        return poses[lower].copy()
    return interpolate_transform(poses[lower], poses[upper], frame - lower)


def shifted_pose(pose: np.ndarray, local_offset: Sequence[float]) -> np.ndarray:
    result = pose.copy()
    offset = np.asarray([*local_offset[:3], 1.0], dtype=np.float64)
    result[:3, 3] = (pose @ offset)[:3]
    return result


def box_corners_world(pose: np.ndarray, size: Sequence[float]) -> np.ndarray:
    length, width, height = map(float, size[:3])
    corners = np.asarray(
        [
            [x * length / 2.0, y * width / 2.0, z * height / 2.0, 1.0]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    ).T
    return pose @ corners


def project_box(
    scene_root: Path,
    frame: int,
    camera: int,
    pose: np.ndarray,
    size: Sequence[float],
    image_size: tuple[int, int] = (1600, 900),
) -> dict[str, Any] | None:
    """Project a 3D box using DriveStudio camera-to-world extrinsics."""

    extrinsic_path = scene_root / "extrinsics" / f"{frame:03d}_{camera}.txt"
    intrinsic_path = scene_root / "intrinsics" / f"{camera}.txt"
    if not extrinsic_path.is_file() or not intrinsic_path.is_file():
        return None
    camera_points = np.linalg.inv(np.loadtxt(extrinsic_path)) @ box_corners_world(pose, size)
    if np.any(camera_points[2] <= 0.1):
        return None
    fx, fy, cx, cy = np.loadtxt(intrinsic_path)[:4]
    u = fx * camera_points[0] / camera_points[2] + cx
    v = fy * camera_points[1] / camera_points[2] + cy
    image_width, image_height = image_size
    left = max(0.0, float(u.min()))
    top = max(0.0, float(v.min()))
    right = min(float(image_width), float(u.max()))
    bottom = min(float(image_height), float(v.max()))
    area = max(0.0, right - left) * max(0.0, bottom - top)
    if area <= 0.0:
        return None
    return {
        "frame": int(frame),
        "camera": int(camera),
        "bbox_xyxy": [left, top, right, bottom],
        "area_px2": area,
        "min_depth_m": float(camera_points[2].min()),
        "max_depth_m": float(camera_points[2].max()),
    }


def footprint(pose: np.ndarray, size: Sequence[float]) -> np.ndarray:
    length, width = map(float, size[:2])
    corners = np.asarray(
        [
            [length / 2.0, width / 2.0, 0.0, 1.0],
            [length / 2.0, -width / 2.0, 0.0, 1.0],
            [-length / 2.0, -width / 2.0, 0.0, 1.0],
            [-length / 2.0, width / 2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ).T
    return (pose @ corners)[:2].T


def footprints_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating-axis test for two convex rectangular footprints."""

    for polygon in (first, second):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm == 0.0:
                continue
            axis /= norm
            projection_a = first @ axis
            projection_b = second @ axis
            if projection_a.max() <= projection_b.min() or projection_b.max() <= projection_a.min():
                return False
    return True

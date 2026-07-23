"""显式 T_dst_src 刚体变换与投影工具。"""
from __future__ import annotations

import numpy as np


class CoordinateError(ValueError):
    """坐标变换不满足刚体约束。"""


def validate_transform(T_dst_src, *, atol: float = 1e-6) -> np.ndarray:
    matrix = np.asarray(T_dst_src, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise CoordinateError(f"变换必须是 4x4，实际为 {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise CoordinateError("变换禁止 NaN/Inf")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise CoordinateError("齐次变换末行非法")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise CoordinateError("旋转矩阵非正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise CoordinateError("旋转矩阵 determinant 必须为 +1")
    return matrix


def invert_transform(T_dst_src) -> np.ndarray:
    matrix = validate_transform(T_dst_src)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transform(T_dst_mid, T_mid_src) -> np.ndarray:
    return validate_transform(
        validate_transform(T_dst_mid) @ validate_transform(T_mid_src)
    )


def transform_points(T_dst_src, points_src) -> np.ndarray:
    matrix = validate_transform(T_dst_src)
    points = np.asarray(points_src, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CoordinateError("points 必须是 Nx3")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def box_corners_actor(dimensions_lwh) -> np.ndarray:
    length, width, height = np.asarray(dimensions_lwh, dtype=np.float64)
    signs = np.asarray(
        [
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
        ],
        dtype=np.float64,
    )
    return signs * np.asarray([length, width, height]) / 2.0


def project_world_points(
    points_world,
    *,
    T_camera_world,
    intrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = transform_points(T_camera_world, points_world)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    if calibration.shape != (3, 3):
        raise CoordinateError("intrinsics 必须是 3x3")
    depth = points_camera[:, 2]
    pixels_h = (calibration @ points_camera.T).T
    pixels = pixels_h[:, :2] / depth[:, None]
    return pixels, depth

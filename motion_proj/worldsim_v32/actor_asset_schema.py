"""V3.2 生成 actor Gaussian 的坐标、协方差与 schema 合同。"""

from __future__ import annotations

from typing import Any

import numpy as np


SH_C0 = 0.28209479177387814


def normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    values = np.asarray(quats, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quats 必须是 [N,4] wxyz")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("quats 包含零范数或非有限值")
    normalized = values / norms
    normalized[normalized[:, 0] < 0] *= -1
    return normalized


def quaternion_to_matrix_wxyz(quats: np.ndarray) -> np.ndarray:
    values = normalize_quaternions(quats)
    w, x, y, z = values.T
    matrices = np.empty((len(values), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def matrix_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("rotation matrices 必须是 [N,3,3]")
    result = np.empty((len(values), 4), dtype=np.float64)
    for index, matrix in enumerate(values):
        trace = float(np.trace(matrix))
        if trace > 0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            result[index] = [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = 2.0 * np.sqrt(
                    max(1e-15, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
                )
                result[index] = [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            elif axis == 1:
                scale = 2.0 * np.sqrt(
                    max(1e-15, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
                )
                result[index] = [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            else:
                scale = 2.0 * np.sqrt(
                    max(1e-15, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
                )
                result[index] = [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
    return normalize_quaternions(result)


def gaussian_covariances(scales: np.ndarray, quats: np.ndarray) -> np.ndarray:
    scale_values = np.asarray(scales, dtype=np.float64)
    if scale_values.ndim != 2 or scale_values.shape[1] != 3:
        raise ValueError("scales 必须是 [N,3]")
    if np.any(~np.isfinite(scale_values)) or np.any(scale_values <= 0):
        raise ValueError("scales 必须是有限正数")
    rotations = quaternion_to_matrix_wxyz(quats)
    return (rotations * np.square(scale_values)[:, None, :]) @ np.swapaxes(
        rotations, 1, 2
    )


def covariance_to_scale_quaternion(
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(covariances, dtype=np.float64)
    eigenvalues, rotations = np.linalg.eigh(values)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise ValueError("Gaussian covariance 不是正定矩阵")
    negative_det = np.linalg.det(rotations) < 0
    rotations[negative_det, :, 0] *= -1
    return np.sqrt(eigenvalues), matrix_to_quaternion_wxyz(rotations)


def gaussian_support_bounds(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    support_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_values = np.asarray(means, dtype=np.float64)
    if mean_values.ndim != 2 or mean_values.shape[1] != 3 or len(mean_values) == 0:
        raise ValueError("means 必须是非空 [N,3]")
    if support_sigma <= 0:
        raise ValueError("support_sigma 必须为正")
    covariances = gaussian_covariances(scales, quats)
    axis_std = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    lower = np.min(mean_values - support_sigma * axis_std, axis=0)
    upper = np.max(mean_values + support_sigma * axis_std, axis=0)
    return lower, upper


def transform_gaussians(
    *,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    linear: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_values = np.asarray(means, dtype=np.float64)
    linear_values = np.asarray(linear, dtype=np.float64)
    translation_values = np.asarray(translation, dtype=np.float64)
    if linear_values.shape != (3, 3) or translation_values.shape != (3,):
        raise ValueError("Gaussian affine transform shape 不合法")
    transformed_means = mean_values @ linear_values.T + translation_values
    covariance = gaussian_covariances(scales, quats)
    transformed_covariance = linear_values[None] @ covariance @ linear_values.T[None]
    transformed_scales, transformed_quats = covariance_to_scale_quaternion(
        transformed_covariance
    )
    return transformed_means, transformed_scales, transformed_quats


def fit_gaussians_to_actor_box(
    *,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    target_lwh: np.ndarray,
    orientation_y_degrees: float = 90.0,
    support_sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """按官方 NuRec Y 旋转后，把 3σ Gaussian bounds 配准到 actor L/W/H。"""
    target = np.asarray(target_lwh, dtype=np.float64)
    if target.shape != (3,) or np.any(~np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("target_lwh 必须是有限正数 (3,)")
    radians = np.deg2rad(float(orientation_y_degrees))
    orientation = np.asarray(
        [
            [np.cos(radians), 0.0, -np.sin(radians)],
            [0.0, 1.0, 0.0],
            [np.sin(radians), 0.0, np.cos(radians)],
        ],
        dtype=np.float64,
    )
    oriented_means, oriented_scales, oriented_quats = transform_gaussians(
        means=means,
        scales=scales,
        quats=quats,
        linear=orientation,
        translation=np.zeros(3),
    )
    lower, upper = gaussian_support_bounds(
        oriented_means, oriented_scales, oriented_quats, support_sigma
    )
    extent = upper - lower
    if np.any(extent <= 1e-8):
        raise ValueError("Asset Harvester Gaussian bounds 退化")
    scale_xyz = target / extent
    linear = np.diag(scale_xyz) @ orientation
    translation = -np.diag(scale_xyz) @ ((lower + upper) / 2.0)
    actor_means, actor_scales, actor_quats = transform_gaussians(
        means=means,
        scales=scales,
        quats=quats,
        linear=linear,
        translation=translation,
    )
    actor_lower, actor_upper = gaussian_support_bounds(
        actor_means, actor_scales, actor_quats, support_sigma
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = translation
    return {
        "means": actor_means.astype(np.float32),
        "scales": actor_scales.astype(np.float32),
        "quats": actor_quats.astype(np.float32),
        "T_actor_asset": transform,
        "bounds_lower": actor_lower,
        "bounds_upper": actor_upper,
        "target_lwh": target,
        "scale_xyz": scale_xyz,
    }


def validate_actor_asset(arrays: dict[str, Any]) -> None:
    required = {"means", "scales", "quats", "rgb", "opacity"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"actor asset 缺字段: {sorted(missing)}")
    means = np.asarray(arrays["means"])
    count = len(means)
    expected = {
        "means": (count, 3),
        "scales": (count, 3),
        "quats": (count, 4),
        "rgb": (count, 3),
        "opacity": (count,),
    }
    if count == 0:
        raise ValueError("actor asset 为空")
    for name, shape in expected.items():
        values = np.asarray(arrays[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"actor asset {name} shape/有限性错误: {values.shape}")
    if np.any(np.asarray(arrays["scales"]) <= 0):
        raise ValueError("actor asset scales 非正")
    if np.any((np.asarray(arrays["rgb"]) < 0) | (np.asarray(arrays["rgb"]) > 1)):
        raise ValueError("actor asset rgb 越界")
    if np.any(
        (np.asarray(arrays["opacity"]) <= 0)
        | (np.asarray(arrays["opacity"]) >= 1)
    ):
        raise ValueError("actor asset opacity 越界")
    normalize_quaternions(np.asarray(arrays["quats"]))


def project_actor_points(
    *,
    means_actor: np.ndarray,
    T_world_actor: np.ndarray,
    T_camera_world: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按 actor-local → world → camera 合同投影，返回 pixel xy 与相机深度。"""
    means = np.asarray(means_actor, dtype=np.float64)
    world_actor = np.asarray(T_world_actor, dtype=np.float64)
    camera_world = np.asarray(T_camera_world, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means_actor 必须是 [N,3]")
    if world_actor.shape != (4, 4) or camera_world.shape != (4, 4):
        raise ValueError("actor/world/camera transform 必须是 4x4")
    if k.shape != (3, 3):
        raise ValueError("intrinsics 必须是 3x3")
    homogeneous = np.concatenate([means, np.ones((len(means), 1))], axis=1)
    world = (world_actor @ homogeneous.T).T
    camera = (np.linalg.inv(camera_world) @ world.T).T[:, :3]
    depth = camera[:, 2]
    projected = (k @ camera.T).T
    pixels = np.full((len(means), 2), np.nan, dtype=np.float64)
    valid = depth > 0
    pixels[valid] = projected[valid, :2] / depth[valid, None]
    return pixels, depth

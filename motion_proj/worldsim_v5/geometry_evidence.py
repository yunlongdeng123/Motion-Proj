"""V5 ownership evidence 使用的冻结 Gaussian geometry 特征。"""

from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v32.actor_asset_schema import gaussian_covariances


def normal_proxies(
    covariances: np.ndarray,
    centers: np.ndarray,
    reference_camera_center: np.ndarray,
) -> np.ndarray:
    covariance = np.asarray(covariances, dtype=np.float64)
    points = np.asarray(centers, dtype=np.float64)
    camera = np.asarray(reference_camera_center, dtype=np.float64)
    count = points.shape[0]
    if covariance.shape != (count, 3, 3) or points.shape != (count, 3):
        raise ValueError("covariance/center shape 不一致")
    if camera.shape != (3,):
        raise ValueError("reference_camera_center 必须为 (3,)")
    if not np.isfinite(covariance).all() or not np.isfinite(points).all() or not np.isfinite(camera).all():
        raise ValueError("geometry 必须有限")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Gaussian covariance 必须正定")
    normals = eigenvectors[:, :, 0]
    camera_vectors = camera[None, :] - points
    flip = np.sum(normals * camera_vectors, axis=1) < 0.0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norms, 1e-12)).astype(np.float32)


def gaussian_geometry(
    *,
    means: np.ndarray,
    scales: np.ndarray,
    quaternions_wxyz: np.ndarray,
    reference_camera_center: np.ndarray,
) -> dict[str, np.ndarray]:
    centers = np.asarray(means, dtype=np.float32)
    covariance = gaussian_covariances(scales, quaternions_wxyz).astype(np.float32)
    normals = normal_proxies(covariance, centers, reference_camera_center)
    return {
        "center": centers,
        "covariance": covariance,
        "normal_proxy": normals,
        "normal_available": np.ones(centers.shape[0], dtype=np.int8),
    }


def view_angle_cosine(
    *, centers: np.ndarray, normals: np.ndarray, camera_center: np.ndarray
) -> np.ndarray:
    points = np.asarray(centers, dtype=np.float64)
    directions = np.asarray(camera_center, dtype=np.float64)[None, :] - points
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    normal_values = np.asarray(normals, dtype=np.float64)
    if normal_values.shape != points.shape:
        raise ValueError("normal/center shape 不一致")
    cosine = np.abs(np.sum(normal_values * directions, axis=1))
    return np.clip(cosine, 0.0, 1.0).astype(np.float32)

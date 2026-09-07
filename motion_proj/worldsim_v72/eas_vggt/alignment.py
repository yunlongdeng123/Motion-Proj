"""原生参考系到 metric world 的显式 Sim(3) 对齐与公共表面读出。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        return (self.scale * np.einsum("ij,...j->...i", self.rotation, array) + self.translation).astype(np.float32)


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Umeyama 最小二乘 Sim(3)；至少三个非退化对应。"""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Sim(3) 对齐至少需要三个三维对应")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1.0e-12:
        raise ValueError("source 对应退化，无法估计尺度")
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("估计得到的 Sim(3) 尺度无效")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def aligned_camera_center_rmse_m(
    reference_from_camera: np.ndarray,
    world_from_camera: np.ndarray,
) -> tuple[SimilarityTransform, float]:
    source = np.asarray(reference_from_camera, dtype=np.float64)[:, :3, 3]
    target = np.asarray(world_from_camera, dtype=np.float64)[:, :3, 3]
    transform = estimate_similarity(source, target)
    residual = transform.apply(source).astype(np.float64) - target
    return transform, float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def common_surface_mask(
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    *,
    confidence_quantile: float,
) -> np.ndarray:
    if not 0.0 <= confidence_quantile < 1.0:
        raise ValueError("confidence_quantile 必须位于 [0,1)")
    confidence = np.asarray(confidence, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return valid
    threshold = float(np.quantile(confidence[valid], confidence_quantile))
    return valid & (confidence >= threshold)


def nearest_pixel_indices(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    *,
    image_width: int,
) -> np.ndarray:
    """返回每个离散像素最近投影在原输入数组中的索引。"""
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    if not (len(x) == len(y) == len(depth)):
        raise ValueError("x/y/depth 数量必须一致")
    if image_width <= 0 or np.any(x < 0) or np.any(x >= image_width) or np.any(y < 0):
        raise ValueError("像素坐标或 image_width 无效")
    if np.any(~np.isfinite(depth)):
        raise ValueError("depth 必须有限")
    pixel_index = y * image_width + x
    order = np.lexsort((depth, pixel_index))
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = pixel_index[order][1:] != pixel_index[order][:-1]
    return order[first]

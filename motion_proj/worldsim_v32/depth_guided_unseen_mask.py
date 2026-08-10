"""3DGIC 风格的深度引导跨视图可见背景判定。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossViewSplat:
    """一个支持视图在目标图像上的 z-buffer 投影。"""

    rgb: np.ndarray
    depth: np.ndarray
    score: np.ndarray
    observed: np.ndarray


def _camera_arrays(
    intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsics 必须为 (3, 3)，得到 {k.shape}")
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world 必须为 (4, 4)，得到 {c2w.shape}")
    if not np.isfinite(k).all() or not np.isfinite(c2w).all():
        raise ValueError("相机矩阵含非有限数")
    if abs(np.linalg.det(k)) < 1e-12 or abs(np.linalg.det(c2w)) < 1e-12:
        raise ValueError("相机矩阵不可逆")
    return k, c2w


def unproject_rgbd(
    *,
    depth: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把有效 RGB-D 像素反投影到世界坐标。"""

    if stride <= 0:
        raise ValueError("stride 必须为正整数")
    z = np.asarray(depth, dtype=np.float64).squeeze()
    colors = np.asarray(rgb)
    keep = np.asarray(valid, dtype=bool).squeeze()
    if z.ndim != 2 or colors.shape != z.shape + (3,) or keep.shape != z.shape:
        raise ValueError("RGB/depth/valid 尺寸不对齐")
    k, c2w = _camera_arrays(intrinsics, camera_to_world)
    sampled = np.zeros_like(keep)
    sampled[::stride, ::stride] = True
    keep &= sampled & np.isfinite(z) & (z > 1e-4)
    y, x = np.nonzero(keep)
    if x.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int64),
        )
    pixels = np.stack([x, y, np.ones_like(x)], axis=0).astype(np.float64)
    camera = (np.linalg.inv(k) @ pixels) * z[y, x][None]
    homogeneous = np.concatenate(
        [camera, np.ones((1, camera.shape[1]), dtype=np.float64)], axis=0
    )
    world = (c2w @ homogeneous)[:3].T
    return world, colors[y, x].astype(np.uint8, copy=False), np.stack([x, y], axis=1)


def project_world(
    *,
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把世界点投影到目标图像，返回浮点 uv 和相机 z。"""

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world 必须为 (N, 3)")
    k, c2w = _camera_arrays(intrinsics, camera_to_world)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera = (np.linalg.inv(c2w) @ homogeneous.T)[:3]
    projected = k @ camera
    z = camera[2]
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    positive = z > 1e-4
    uv[positive] = (projected[:2, positive] / projected[2, positive]).T
    return uv, z


def splat_rgbd_to_target(
    *,
    source_depth: np.ndarray,
    source_rgb: np.ndarray,
    source_valid: np.ndarray,
    source_intrinsics: np.ndarray,
    source_camera_to_world: np.ndarray,
    target_depth: np.ndarray,
    target_mask: np.ndarray,
    target_intrinsics: np.ndarray,
    target_camera_to_world: np.ndarray,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
    stride: int = 1,
) -> CrossViewSplat:
    """仅保留落入目标 mask 且与目标删除深度一致的跨视图背景。"""

    if absolute_depth_tolerance_m < 0 or relative_depth_tolerance < 0:
        raise ValueError("深度容差不能为负")
    target_z = np.asarray(target_depth, dtype=np.float64).squeeze()
    mask = np.asarray(target_mask, dtype=bool).squeeze()
    if target_z.ndim != 2 or mask.shape != target_z.shape:
        raise ValueError("target depth/mask 尺寸不对齐")
    height, width = target_z.shape
    points, colors, _ = unproject_rgbd(
        depth=source_depth,
        rgb=source_rgb,
        valid=source_valid,
        intrinsics=source_intrinsics,
        camera_to_world=source_camera_to_world,
        stride=stride,
    )
    output_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    output_depth = np.full((height, width), np.nan, dtype=np.float32)
    output_score = np.full((height, width), np.inf, dtype=np.float32)
    observed = np.zeros((height, width), dtype=bool)
    if points.shape[0] == 0:
        return CrossViewSplat(output_rgb, output_depth, output_score, observed)

    uv, projected_z = project_world(
        points_world=points,
        intrinsics=target_intrinsics,
        camera_to_world=target_camera_to_world,
    )
    # 先把不可投影点放到图像外，再转整数；避免 NaN 转整数产生告警。
    rounded_uv = np.where(np.isfinite(uv), np.rint(uv), -1.0)
    x = rounded_uv[:, 0].astype(np.int64)
    y = rounded_uv[:, 1].astype(np.int64)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (projected_z > 1e-4)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    indices = np.where(inside)[0]
    if indices.size == 0:
        return CrossViewSplat(output_rgb, output_depth, output_score, observed)
    x, y = x[indices], y[indices]
    projected_z = projected_z[indices]
    colors = colors[indices]
    inside_mask = mask[y, x]
    target_at_pixel = target_z[y, x]
    target_valid = np.isfinite(target_at_pixel) & (target_at_pixel > 1e-4)
    tolerance = np.maximum(
        float(absolute_depth_tolerance_m),
        float(relative_depth_tolerance) * np.maximum(target_at_pixel, projected_z),
    )
    consistent = ~target_valid | (np.abs(projected_z - target_at_pixel) <= tolerance)
    keep = inside_mask & consistent
    if not keep.any():
        return CrossViewSplat(output_rgb, output_depth, output_score, observed)
    x, y = x[keep], y[keep]
    projected_z = projected_z[keep]
    colors = colors[keep]
    target_at_pixel = target_at_pixel[keep]
    target_valid = target_valid[keep]
    score = np.where(
        target_valid,
        np.abs(projected_z - target_at_pixel),
        projected_z,
    )
    flat = y * width + x
    order = np.lexsort((score, flat))
    flat_sorted = flat[order]
    first = np.r_[True, flat_sorted[1:] != flat_sorted[:-1]]
    chosen = order[first]
    x, y = x[chosen], y[chosen]
    output_rgb[y, x] = colors[chosen]
    output_depth[y, x] = projected_z[chosen].astype(np.float32)
    output_score[y, x] = score[chosen].astype(np.float32)
    observed[y, x] = True
    return CrossViewSplat(output_rgb, output_depth, output_score, observed)

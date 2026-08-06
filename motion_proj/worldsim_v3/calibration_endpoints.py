"""WorldSim V3 A1 几何与光度诊断端点的纯函数实现。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


ABSTAIN = "ABSTAIN"


def validate_endpoint_contract(payload: Mapping[str, Any]) -> None:
    """拒绝不完整或被静默改写的 A1-E0 合同。"""

    if payload.get("schema_version") != 1:
        raise ValueError("endpoint contract schema_version must be 1")
    if payload.get("endpoint_version") != "A1-E0-v1":
        raise ValueError("endpoint_version must be A1-E0-v1")
    e1 = payload.get("e1")
    e2 = payload.get("e2")
    if not isinstance(e1, Mapping) or not isinstance(e2, Mapping):
        raise ValueError("endpoint contract requires e1 and e2 mappings")

    expected_pairs = [
        ["CAM_FRONT_LEFT", "CAM_FRONT"],
        ["CAM_FRONT", "CAM_FRONT_RIGHT"],
    ]
    if e1.get("camera_pairs") != expected_pairs:
        raise ValueError(f"E1 camera_pairs must be {expected_pairs}")
    if e1.get("directions") != "bidirectional":
        raise ValueError("E1 directions must be bidirectional")
    positive_e1 = (
        "grid_stride_pixels",
        "minimum_rendered_opacity",
        "maximum_dynamic_opacity",
        "maximum_relative_depth_edge",
        "maximum_relative_occlusion_error",
        "near_far_split_meters",
        "minimum_valid_supports",
        "minimum_coverage",
    )
    for key in positive_e1:
        if float(e1.get(key, 0)) <= 0:
            raise ValueError(f"E1 {key} must be positive")
    if int(e1["grid_stride_pixels"]) != e1["grid_stride_pixels"]:
        raise ValueError("E1 grid_stride_pixels must be an integer")

    if e2.get("effect_mask_source") != "a0_paired_original_delete_uint8":
        raise ValueError("E2 must reuse the frozen A0 paired-render mask")
    if e2.get("boundary_extractor") != "inner_8_connected_one_pixel":
        raise ValueError("E2 boundary_extractor must remain frozen")
    if e2.get("distance_normalization") != "image_diagonal":
        raise ValueError("E2 distance_normalization must be image_diagonal")
    positive_e2 = (
        "effect_threshold_uint8",
        "effect_dilation_radius_pixels",
        "minimum_actor_support_opacity",
        "minimum_valid_images",
        "minimum_coverage",
    )
    for key in positive_e2:
        if float(e2.get(key, 0)) <= 0:
            raise ValueError(f"E2 {key} must be positive")


def canonicalize_observed_rgb(
    rgb: np.ndarray,
    affine_matrix: np.ndarray | None,
) -> np.ndarray:
    """把观测 RGB 逆映射到模型的共享辐射空间。"""

    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must have shape [H, W, 3]")
    if affine_matrix is None:
        return image.copy()
    transform = np.asarray(affine_matrix, dtype=np.float64)
    if transform.shape == (3, 4):
        linear = transform[:, :3]
        offset = transform[:, 3]
    elif transform.shape == image.shape[:2] + (3, 4):
        linear = transform[..., :3]
        offset = transform[..., 3]
    else:
        raise ValueError(
            "affine_matrix must have shape [3, 4] or [H, W, 3, 4]"
        )
    determinants = np.linalg.det(linear)
    if np.any(np.abs(determinants) <= 1e-8):
        raise ValueError("affine linear component is singular")
    if linear.ndim == 2:
        flat = image.reshape(-1, 3)
        canonical = np.linalg.solve(linear, (flat - offset).T).T.reshape(image.shape)
    else:
        canonical = np.linalg.solve(linear, image - offset)
    if not np.isfinite(canonical).all():
        raise ValueError("canonical RGB contains non-finite values")
    return canonical


def depth_continuity_mask(
    depth: np.ndarray,
    maximum_relative_edge: float,
    dilation_radius: int,
) -> np.ndarray:
    """标记无明显深度断裂的像素，断裂邻域按方形核膨胀。"""

    values = np.asarray(depth, dtype=np.float64).squeeze()
    if values.ndim != 2:
        raise ValueError("depth must be two-dimensional")
    valid = np.isfinite(values) & (values > 0)
    discontinuity = ~valid
    horizontal = np.zeros_like(valid)
    vertical = np.zeros_like(valid)
    denom_h = np.maximum(np.maximum(values[:, 1:], values[:, :-1]), 1e-8)
    denom_v = np.maximum(np.maximum(values[1:, :], values[:-1, :]), 1e-8)
    horizontal[:, 1:] = (
        np.abs(values[:, 1:] - values[:, :-1]) / denom_h > maximum_relative_edge
    )
    horizontal[:, :-1] |= horizontal[:, 1:]
    vertical[1:, :] = (
        np.abs(values[1:, :] - values[:-1, :]) / denom_v > maximum_relative_edge
    )
    vertical[:-1, :] |= vertical[1:, :]
    discontinuity |= horizontal | vertical
    if dilation_radius > 0:
        from scipy.ndimage import binary_dilation

        size = 2 * int(dilation_radius) + 1
        discontinuity = binary_dilation(
            discontinuity, structure=np.ones((size, size), dtype=bool)
        )
    return valid & ~discontinuity


def static_support_mask(
    *,
    sky_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    egocar_mask: np.ndarray,
    rendered_opacity: np.ndarray,
    dynamic_opacity: np.ndarray,
    depth: np.ndarray,
    minimum_rendered_opacity: float,
    maximum_dynamic_opacity: float,
    maximum_relative_depth_edge: float,
    depth_edge_dilation_pixels: int,
) -> np.ndarray:
    """构造 E1 的静态、可见且深度连续支持域。"""

    squeeze_bool = lambda value: np.asarray(value).squeeze() >= 0.5
    opacity = np.asarray(rendered_opacity, dtype=np.float64).squeeze()
    dynamic = np.asarray(dynamic_opacity, dtype=np.float64).squeeze()
    continuous = depth_continuity_mask(
        depth, maximum_relative_depth_edge, depth_edge_dilation_pixels
    )
    result = (
        ~squeeze_bool(sky_mask)
        & ~squeeze_bool(dynamic_mask)
        & ~squeeze_bool(egocar_mask)
        & (opacity >= minimum_rendered_opacity)
        & (dynamic <= maximum_dynamic_opacity)
        & continuous
    )
    if result.ndim != 2:
        raise ValueError("static support masks must resolve to [H, W]")
    return result


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, values.shape[1] - 1)
    y1 = np.minimum(y0 + 1, values.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    if values.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]
    return (
        values[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + values[y0, x1] * wx * (1.0 - wy)
        + values[y1, x0] * (1.0 - wx) * wy
        + values[y1, x1] * wx * wy
    )


def _nearest_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    return values[yi, xi]


def cross_camera_residuals(
    *,
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    source_support: np.ndarray,
    source_camera_to_world: np.ndarray,
    source_intrinsics: np.ndarray,
    target_rgb: np.ndarray,
    target_depth: np.ndarray,
    target_support: np.ndarray,
    target_camera_to_world: np.ndarray,
    target_intrinsics: np.ndarray,
    grid_stride_pixels: int,
    maximum_relative_occlusion_error: float,
) -> dict[str, np.ndarray | int]:
    """将源相机静态支持投到目标相机并返回有效颜色残差。"""

    depth = np.asarray(source_depth, dtype=np.float64).squeeze()
    support = np.asarray(source_support, dtype=bool)
    height, width = support.shape
    yy, xx = np.mgrid[0:height:grid_stride_pixels, 0:width:grid_stride_pixels]
    sampled_support = support[yy, xx]
    x = xx[sampled_support].astype(np.float64)
    y = yy[sampled_support].astype(np.float64)
    z = depth[yy[sampled_support], xx[sampled_support]]
    candidate_count = int(z.size)
    empty = {
        "residuals": np.empty(0, dtype=np.float64),
        "depths": np.empty(0, dtype=np.float64),
        "candidate_depths": z.copy(),
        "target_x": np.empty(0, dtype=np.float64),
        "target_y": np.empty(0, dtype=np.float64),
        "candidate_count": candidate_count,
    }
    if candidate_count == 0:
        return empty

    pixels = np.stack((x, y, np.ones_like(x)), axis=0)
    camera_points = np.linalg.solve(np.asarray(source_intrinsics), pixels) * z
    homogeneous = np.concatenate((camera_points, np.ones((1, candidate_count))), axis=0)
    world = np.asarray(source_camera_to_world) @ homogeneous
    target_camera = np.linalg.inv(np.asarray(target_camera_to_world)) @ world
    target_z = target_camera[2]
    projected = np.asarray(target_intrinsics) @ target_camera[:3]
    target_x = projected[0] / target_z
    target_y = projected[1] / target_z
    target_height, target_width = np.asarray(target_support).shape
    valid = (
        np.isfinite(target_x)
        & np.isfinite(target_y)
        & np.isfinite(target_z)
        & (target_z > 0)
        & (target_x >= 0)
        & (target_y >= 0)
        & (target_x <= target_width - 1)
        & (target_y <= target_height - 1)
    )
    if not valid.any():
        return empty
    x, y, z = x[valid], y[valid], z[valid]
    target_x, target_y, target_z = target_x[valid], target_y[valid], target_z[valid]
    target_valid = _nearest_sample(target_support, target_x, target_y).astype(bool)
    rendered_target_depth = _bilinear_sample(target_depth, target_x, target_y).squeeze()
    relative_occlusion_error = np.abs(rendered_target_depth - target_z) / np.maximum(
        np.maximum(rendered_target_depth, target_z), 1e-8
    )
    valid = (
        target_valid
        & np.isfinite(rendered_target_depth)
        & (rendered_target_depth > 0)
        & (relative_occlusion_error <= maximum_relative_occlusion_error)
    )
    if not valid.any():
        return empty
    source_colors = _bilinear_sample(source_rgb, x[valid], y[valid])
    target_colors = _bilinear_sample(target_rgb, target_x[valid], target_y[valid])
    residuals = np.mean(np.abs(source_colors - target_colors), axis=-1)
    if not np.isfinite(residuals).all():
        raise ValueError("E1 residuals contain non-finite values")
    return {
        "residuals": residuals,
        "depths": z[valid],
        "candidate_depths": empty["candidate_depths"],
        "target_x": target_x[valid],
        "target_y": target_y[valid],
        "candidate_count": candidate_count,
    }


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """提取一像素、8 邻接的内部边界。"""

    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError("boundary mask must be two-dimensional")
    if not selected.any():
        return np.zeros_like(selected)
    eroded = binary_erosion(
        selected, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return selected & ~eroded


def bidirectional_boundary_distances(
    support_mask: np.ndarray,
    effect_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """返回按图像对角线归一化的双向边界距离。"""

    support_boundary = inner_boundary(support_mask)
    effect_boundary = inner_boundary(effect_mask)
    if not support_boundary.any() or not effect_boundary.any():
        return {
            "support_to_effect": np.empty(0, dtype=np.float64),
            "effect_to_support": np.empty(0, dtype=np.float64),
            "combined": np.empty(0, dtype=np.float64),
        }
    diagonal = float(np.hypot(*support_boundary.shape))
    support_to_effect = distance_transform_edt(~effect_boundary)[support_boundary] / diagonal
    effect_to_support = distance_transform_edt(~support_boundary)[effect_boundary] / diagonal
    return {
        "support_to_effect": support_to_effect,
        "effect_to_support": effect_to_support,
        "combined": np.concatenate((support_to_effect, effect_to_support)),
    }


def summarize_distribution(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    if not np.isfinite(array).all():
        raise ValueError("metric distribution contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def coverage_status(
    *,
    valid_count: int,
    candidate_count: int,
    minimum_valid_count: int,
    minimum_coverage: float,
    zero_reason: str,
) -> dict[str, Any]:
    coverage = valid_count / candidate_count if candidate_count else 0.0
    if candidate_count == 0:
        status, reason = ABSTAIN, "NO_CANDIDATE_SUPPORT"
    elif valid_count == 0:
        status, reason = ABSTAIN, zero_reason
    elif valid_count < minimum_valid_count:
        status, reason = ABSTAIN, "INSUFFICIENT_VALID_SUPPORT"
    elif coverage < minimum_coverage:
        status, reason = ABSTAIN, "INSUFFICIENT_COVERAGE"
    else:
        status, reason = "done", None
    return {
        "status": status,
        "reason": reason,
        "candidate_count": int(candidate_count),
        "valid_count": int(valid_count),
        "coverage": float(coverage),
    }

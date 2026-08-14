"""从 renderer intersection 与 SAM2 logit 构建 V5 稀疏 per-view evidence。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from .bayesian_unary import observation_reliability
from .evidence_schema import validate_observation_chunk


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def signed_boundary_distance(binary_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("binary_mask 必须为 HxW")
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return (inside - outside).astype(np.float32)


def sparse_contribution_selection(
    contribution_weight: np.ndarray, *, minimum_weight: float
) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.asarray(contribution_weight, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("contribution_weight 必须是一维有限非负")
    if not np.isfinite(minimum_weight) or minimum_weight <= 0.0:
        raise ValueError("minimum_weight 必须有限且大于 0")
    selected = weights >= minimum_weight
    kept_mass = float(weights[selected].sum())
    total_mass = float(weights.sum())
    return selected, {
        "minimum_weight": float(minimum_weight),
        "input_count": int(weights.size),
        "kept_count": int(selected.sum()),
        "dropped_count": int((~selected).sum()),
        "total_contribution_mass": total_mass,
        "kept_contribution_mass": kept_mass,
        "dropped_contribution_mass": total_mass - kept_mass,
    }


def build_observation_chunk(
    *,
    scene: str,
    role: str,
    view_id: int,
    frame_id: int,
    camera_id: int,
    gaussian_count: int,
    gaussian_id: np.ndarray,
    pixel_id: np.ndarray,
    projected_pixel: np.ndarray,
    contribution_weight: np.ndarray,
    projected_depth: np.ndarray,
    first_hit_depth: np.ndarray,
    first_hit_valid: np.ndarray,
    mask_logits: np.ndarray,
    mask_binary: np.ndarray,
    mask_quality_accepted: bool,
    view_angle_cosine: np.ndarray,
    lidar_support: np.ndarray | None,
    depth_absolute_tolerance_m: float,
    depth_relative_tolerance: float,
    sam_confidence_floor: float,
    boundary_distance_scale_px: float,
    depth_residual_scale_m: float,
    sam_probability_available: bool = True,
) -> dict[str, np.ndarray]:
    gids = np.asarray(gaussian_id, dtype=np.int64)
    pids = np.asarray(pixel_id, dtype=np.int64)
    pixels = np.asarray(projected_pixel, dtype=np.float32)
    weights = np.asarray(contribution_weight, dtype=np.float32)
    depths = np.asarray(projected_depth, dtype=np.float32)
    count = gids.size
    for name, value, shape in (
        ("pixel_id", pids, (count,)),
        ("projected_pixel", pixels, (count, 2)),
        ("contribution_weight", weights, (count,)),
        ("projected_depth", depths, (count,)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} shape 必须为 {shape}")
    if np.any(gids < 0) or np.any(gids >= gaussian_count):
        raise ValueError("gaussian_id 越界")
    logits_image = np.asarray(mask_logits, dtype=np.float32)
    binary_image = np.asarray(mask_binary, dtype=bool)
    first = np.asarray(first_hit_depth, dtype=np.float32).reshape(-1)
    valid = np.asarray(first_hit_valid, dtype=bool).reshape(-1)
    if logits_image.shape != binary_image.shape:
        raise ValueError("mask_logits/mask_binary shape 不一致")
    if first.size != binary_image.size or valid.size != binary_image.size:
        raise ValueError("first-hit 与 mask 像素分母不一致")
    if np.any(pids < 0) or np.any(pids >= binary_image.size):
        raise ValueError("pixel_id 越界")
    if not np.isfinite(depth_absolute_tolerance_m) or depth_absolute_tolerance_m <= 0:
        raise ValueError("depth_absolute_tolerance_m 必须有限且大于 0")
    if not np.isfinite(depth_relative_tolerance) or depth_relative_tolerance < 0:
        raise ValueError("depth_relative_tolerance 必须有限且非负")

    sampled_logits = logits_image.reshape(-1)[pids]
    probability = stable_sigmoid(sampled_logits).astype(np.float32)
    residual = depths - first[pids]
    tolerance = np.maximum(
        depth_absolute_tolerance_m,
        depth_relative_tolerance * np.abs(first[pids]),
    )
    depth_consistent = (
        valid[pids] & np.isfinite(depths) & np.isfinite(residual) & (np.abs(residual) <= tolerance)
    )
    residual = np.where(np.isfinite(residual), residual, 0.0).astype(np.float32)
    boundary = signed_boundary_distance(binary_image).reshape(-1)[pids]
    available = np.full(count, bool(sam_probability_available), dtype=np.int8)
    accepted = np.full(count, bool(mask_quality_accepted), dtype=np.int8)
    usable = bool(mask_quality_accepted) and bool(sam_probability_available)
    sampled_binary = binary_image.reshape(-1)[pids]
    positive = (sampled_binary & usable).astype(np.int8)
    negative = ((~sampled_binary) & usable).astype(np.int8)
    if lidar_support is None:
        lidar = np.zeros(count, dtype=np.float32)
        lidar_available = np.zeros(count, dtype=np.int8)
    else:
        values = np.asarray(lidar_support, dtype=np.float32)
        if values.shape != (gaussian_count,):
            raise ValueError("lidar_support 必须按全局 Gaussian 对齐")
        lidar = values[gids]
        lidar_available = np.ones(count, dtype=np.int8)
    chunk = {
        "scene": np.asarray(scene),
        "role": np.asarray(role),
        "sam_probability_source": np.asarray("sigmoid_sam2_logit"),
        "gaussian_id": gids,
        "view_id": np.full(count, view_id, dtype=np.int32),
        "frame_id": np.full(count, frame_id, dtype=np.int32),
        "camera_id": np.full(count, camera_id, dtype=np.int8),
        "projected_pixel": pixels,
        "visibility": np.clip(weights, 0.0, 1.0),
        "sam_probability": probability,
        "sam_logit": sampled_logits.astype(np.float32),
        "sam_probability_available": available,
        "mask_quality_accepted": accepted,
        "mask_boundary_distance": boundary.astype(np.float32),
        "depth_residual": residual,
        "depth_consistent": depth_consistent.astype(np.int8),
        "lidar_support": lidar,
        "lidar_support_available": lidar_available,
        "view_angle_cosine": np.clip(
            np.asarray(view_angle_cosine, dtype=np.float32)[gids], 0.0, 1.0
        ),
        "positive_observation": positive,
        "negative_observation": negative,
        "reliability": np.zeros(count, dtype=np.float32),
        "contribution_weight": weights,
    }
    chunk["reliability"] = observation_reliability(
        chunk,
        sam_confidence_floor=sam_confidence_floor,
        boundary_distance_scale_px=boundary_distance_scale_px,
        depth_residual_scale_m=depth_residual_scale_m,
    ) * accepted.astype(np.float32) * available.astype(np.float32)
    validate_observation_chunk(chunk, gaussian_count=gaussian_count)
    return chunk

"""把逐 intersection renderer evidence 收缩为每 Gaussian、每 view 一条观测。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .bayesian_unary import observation_reliability
from .evidence_schema import validate_observation_chunk


def _single_integer(chunk: Mapping[str, np.ndarray], name: str) -> int:
    values = np.asarray(chunk[name], dtype=np.int64)
    unique = np.unique(values)
    if unique.size != 1:
        raise ValueError(f"intersection chunk 的 {name} 必须属于单一 view")
    return int(unique[0])


def _weighted_bincount(
    inverse: np.ndarray,
    weight: np.ndarray,
    value: np.ndarray,
    count: int,
) -> np.ndarray:
    return np.bincount(inverse, weights=weight * value, minlength=count)


def aggregate_intersection_observations(
    intersections: Mapping[str, np.ndarray],
    *,
    gaussian_count: int,
    minimum_contribution_mass: float,
    sam_confidence_floor: float,
    boundary_distance_scale_px: float,
    depth_residual_scale_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """确定性聚合同一 view 内的重复 Gaussian-pixel intersections。

    几何与 SAM 数值按真实 alpha contribution mass 加权；每个 Gaussian 的 view
    visibility 使用 ``1-exp(-mass)`` 饱和，避免像素占用面积被误当独立 view 数。
    """

    if not np.isfinite(minimum_contribution_mass) or minimum_contribution_mass <= 0:
        raise ValueError("minimum_contribution_mass 必须有限且大于 0")
    validate_observation_chunk(intersections, gaussian_count=gaussian_count)
    for name in ("view_id", "frame_id", "camera_id"):
        _single_integer(intersections, name)
    gids = np.asarray(intersections["gaussian_id"], dtype=np.int64)
    contribution = np.asarray(
        intersections["contribution_weight"], dtype=np.float64
    )
    positive_rows = contribution > 0.0
    if not positive_rows.any():
        raise ValueError("intersection chunk 没有正 contribution")
    gids = gids[positive_rows]
    contribution = contribution[positive_rows]
    unique_gids, inverse = np.unique(gids, return_inverse=True)
    unique_count = unique_gids.size
    mass = np.bincount(inverse, weights=contribution, minlength=unique_count)
    keep = mass >= float(minimum_contribution_mass)
    if not keep.any():
        raise ValueError("聚合后没有 Gaussian 达到 contribution floor")

    def aggregate(name: str) -> np.ndarray:
        values = np.asarray(intersections[name], dtype=np.float64)[positive_rows]
        numerator = _weighted_bincount(
            inverse, contribution, values, unique_count
        )
        return np.divide(
            numerator,
            mass,
            out=np.zeros_like(numerator),
            where=mass > 0.0,
        )[keep]

    pixels = np.asarray(intersections["projected_pixel"], dtype=np.float64)[
        positive_rows
    ]
    pixel_x = _weighted_bincount(
        inverse, contribution, pixels[:, 0], unique_count
    )
    pixel_y = _weighted_bincount(
        inverse, contribution, pixels[:, 1], unique_count
    )
    projected_pixel = np.stack(
        (
            np.divide(pixel_x, mass, out=np.zeros_like(pixel_x), where=mass > 0),
            np.divide(pixel_y, mass, out=np.zeros_like(pixel_y), where=mass > 0),
        ),
        axis=1,
    )[keep]
    probability = np.clip(aggregate("sam_probability"), 1e-6, 1.0 - 1e-6)
    logit = np.log(probability) - np.log1p(-probability)
    selected_mass = mass[keep]
    count = int(keep.sum())
    chunk = {
        "scene": np.asarray(str(np.asarray(intersections["scene"]).item())),
        "role": np.asarray(str(np.asarray(intersections["role"]).item())),
        "sam_probability_source": np.asarray("aggregated_sigmoid_sam2_logit"),
        "gaussian_id": unique_gids[keep].astype(np.int64),
        "view_id": np.full(
            count, _single_integer(intersections, "view_id"), dtype=np.int32
        ),
        "frame_id": np.full(
            count, _single_integer(intersections, "frame_id"), dtype=np.int32
        ),
        "camera_id": np.full(
            count, _single_integer(intersections, "camera_id"), dtype=np.int8
        ),
        "projected_pixel": projected_pixel.astype(np.float32),
        "visibility": (1.0 - np.exp(-selected_mass)).astype(np.float32),
        "sam_probability": probability.astype(np.float32),
        "sam_logit": logit.astype(np.float32),
        "sam_probability_available": (
            aggregate("sam_probability_available") >= 1.0 - 1e-6
        ).astype(np.int8),
        "mask_quality_accepted": (
            aggregate("mask_quality_accepted") >= 1.0 - 1e-6
        ).astype(np.int8),
        "mask_boundary_distance": aggregate("mask_boundary_distance").astype(
            np.float32
        ),
        "depth_residual": aggregate("depth_residual").astype(np.float32),
        "depth_consistent": np.clip(
            aggregate("depth_consistent"), 0.0, 1.0
        ).astype(np.float32),
        "lidar_support": np.clip(
            aggregate("lidar_support"), 0.0, 1.0
        ).astype(np.float32),
        "lidar_support_available": (
            aggregate("lidar_support_available") > 0.0
        ).astype(np.int8),
        "view_angle_cosine": np.clip(
            aggregate("view_angle_cosine"), 0.0, 1.0
        ).astype(np.float32),
        "positive_observation": np.clip(
            aggregate("positive_observation"), 0.0, 1.0
        ).astype(np.float32),
        "negative_observation": np.clip(
            aggregate("negative_observation"), 0.0, 1.0
        ).astype(np.float32),
        "reliability": np.zeros(count, dtype=np.float32),
        "contribution_weight": selected_mass.astype(np.float32),
    }
    accepted = np.asarray(chunk["mask_quality_accepted"], dtype=np.float32)
    available = np.asarray(chunk["sam_probability_available"], dtype=np.float32)
    chunk["reliability"] = observation_reliability(
        chunk,
        sam_confidence_floor=sam_confidence_floor,
        boundary_distance_scale_px=boundary_distance_scale_px,
        depth_residual_scale_m=depth_residual_scale_m,
    ) * accepted * available
    validate_observation_chunk(chunk, gaussian_count=gaussian_count)
    report = {
        "input_intersection_count": int(positive_rows.sum()),
        "input_unique_gaussian_count": int(unique_count),
        "kept_gaussian_count": count,
        "dropped_gaussian_count": int((~keep).sum()),
        "minimum_contribution_mass": float(minimum_contribution_mass),
        "total_contribution_mass": float(mass.sum()),
        "kept_contribution_mass": float(selected_mass.sum()),
        "dropped_contribution_mass": float(mass[~keep].sum()),
    }
    return chunk, report

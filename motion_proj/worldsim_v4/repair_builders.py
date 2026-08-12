"""M2 OBSERVED/TELEA/DONOR Gaussian 候选的确定性构造函数。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from motion_proj.worldsim_v32.inpainting_adapter import CompletionPoints
from motion_proj.worldsim_v33.roadpatch import sh_dc_from_rgb

from .repair_assets import build_repair_asset


def completion_points_to_repair_asset(
    points: CompletionPoints,
    *,
    candidate_id: str,
    method: str,
    provenance: str,
    features_rest_shape: tuple[int, ...],
    opacity: float,
    target_frame: int,
    target_camera_id: int,
) -> dict[str, np.ndarray]:
    """把冻结深度反投影点转为 StreetGS-compatible V4 外部资产。"""

    count = int(points.means.shape[0])
    if count <= 0:
        raise ValueError("completion candidate 没有 Gaussian")
    if points.rgb.shape != (count, 3) or points.scales.shape != (count, 3):
        raise ValueError("CompletionPoints schema 漂移")
    if not 0.0 < float(opacity) < 1.0:
        raise ValueError("candidate opacity 必须在 (0,1)")
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    raw_opacity = float(np.log(float(opacity) / (1.0 - float(opacity))))
    return build_repair_asset(
        candidate_id=candidate_id,
        method=method,
        provenance=provenance,
        means=np.asarray(points.means, dtype=np.float32),
        raw_scales=np.log(np.maximum(points.scales, 1e-8)).astype(np.float32),
        quats=quats,
        features_dc=sh_dc_from_rgb(
            np.clip(np.asarray(points.rgb), 1e-5, 1.0 - 1e-5)
        ).astype(np.float32),
        features_rest=np.zeros((count,) + tuple(features_rest_shape), dtype=np.float32),
        raw_opacities=np.full((count, 1), raw_opacity, dtype=np.float32),
        confidence=np.asarray(points.confidence, dtype=np.float32),
        source_frames=np.full(count, int(target_frame), dtype=np.int32),
        source_camera_ids=np.full(count, int(target_camera_id), dtype=np.int16),
        source_pixels_xy=np.asarray(points.source_pixels_xy, dtype=np.int32),
    )


def roadpatch_delta_to_repair_asset(
    delta: Mapping[str, np.ndarray],
    *,
    candidate_id: str,
    confidence: float,
    target_frame: int,
    target_camera_id: int,
) -> dict[str, np.ndarray]:
    """保留 native donor lineage，将 RoadPatch delta 转成统一 DONOR 资产。"""

    count = int(np.asarray(delta["means"]).shape[0])
    if count <= 0:
        raise ValueError("DONOR delta 为空")
    return build_repair_asset(
        candidate_id=candidate_id,
        method="DONOR",
        provenance="native_scene_donor",
        means=np.asarray(delta["means"]),
        raw_scales=np.asarray(delta["raw_scales"]),
        quats=np.asarray(delta["quats"]),
        features_dc=np.asarray(delta["features_dc"]),
        features_rest=np.asarray(delta["features_rest"]),
        raw_opacities=np.asarray(delta["raw_opacities"]),
        confidence=np.full(count, float(confidence), dtype=np.float32),
        source_gaussian_ids=np.asarray(delta["source_gaussian_ids"], dtype=np.int64),
        source_frames=np.full(count, int(target_frame), dtype=np.int32),
        source_camera_ids=np.full(count, int(target_camera_id), dtype=np.int16),
    )


def normalized_repair_risks(
    *,
    photo_l1_uint8: float,
    geometry_mae_m: float,
    temporal_std_uint8: float | None,
    uncertainty: float,
    gaussian_count: int,
    normalization: Mapping[str, float],
) -> dict[str, float]:
    """把物理量映射到 router 要求的 [0,1]；映射常数必须由配置冻结。"""

    def clipped(value: float, scale: float) -> float:
        if not np.isfinite(value) or float(scale) <= 0:
            return 1.0
        return float(np.clip(float(value) / float(scale), 0.0, 1.0))

    return {
        "photo_risk": clipped(photo_l1_uint8, normalization["photo_l1_uint8"]),
        "geometry_risk": clipped(geometry_mae_m, normalization["geometry_mae_m"]),
        "temporal_risk": clipped(
            float("inf") if temporal_std_uint8 is None else temporal_std_uint8,
            normalization["temporal_std_uint8"],
        ),
        "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
        "compute_cost": clipped(float(gaussian_count), normalization["gaussian_count"]),
    }

"""不读取 target depth 的点表面查询与评估端射线状态。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from motion_proj.worldsim_v72.data.schema import QueryRayBatch, RayTargets


class RayOutcome(str, Enum):
    INVALID_EMISSION = "invalid_emission"
    CENSORED_OR_MISSING = "censored_or_missing"
    NO_RETURN = "no_return"
    BACKGROUND_OCCLUSION = "background_occlusion"
    ACTOR_RETURN = "actor_return"
    OTHER_RETURN = "other_return"


@dataclass(frozen=True)
class PointSurfaceReturnPrediction:
    return_valid: np.ndarray
    range_m: np.ndarray


def predict_point_surface_returns(
    surface_points_m: np.ndarray,
    queries: QueryRayBatch,
    *,
    lateral_tolerance_m: float,
) -> PointSurfaceReturnPrediction:
    """在固定 QueryRayBatch 量程内读取最早的 beam-tube 点。"""
    if lateral_tolerance_m <= 0.0:
        raise ValueError("lateral_tolerance_m 必须为正数")
    surface = np.asarray(surface_points_m, dtype=np.float32).reshape(-1, 3)
    ranges = np.full(len(queries), np.nan, dtype=np.float32)
    valid = np.zeros(len(queries), dtype=bool)
    for index in range(len(queries)):
        if not queries.valid_emission_mask[index] or len(surface) == 0:
            continue
        relative = surface - queries.origin_m[index]
        depth = relative @ queries.unit_direction[index]
        lateral = np.linalg.norm(
            relative - depth[:, None] * queries.unit_direction[index][None, :], axis=1
        )
        eligible = (
            (depth >= queries.range_min_m[index])
            & (depth <= queries.range_max_m[index])
            & (lateral <= float(lateral_tolerance_m))
        )
        if np.any(eligible):
            ranges[index] = float(np.min(depth[eligible]))
            valid[index] = True
    return PointSurfaceReturnPrediction(return_valid=valid, range_m=ranges)


def classify_ray_outcomes(
    queries: QueryRayBatch,
    targets: RayTargets,
    *,
    expected_actor_id: str,
) -> np.ndarray:
    """仅在评估端区分无效、缺测、无回波和背景遮挡。"""
    if len(queries) != len(targets):
        raise ValueError("query 与 target 射线数量不一致")
    outcomes = np.full(len(queries), RayOutcome.OTHER_RETURN.value, dtype="U32")
    for index in range(len(queries)):
        if not queries.valid_emission_mask[index]:
            outcomes[index] = RayOutcome.INVALID_EMISSION.value
        elif targets.censoring_or_missing_mask[index]:
            outcomes[index] = RayOutcome.CENSORED_OR_MISSING.value
        elif not targets.return_valid[index]:
            outcomes[index] = RayOutcome.NO_RETURN.value
        elif targets.object_id_for_evaluation_only[index] == expected_actor_id:
            outcomes[index] = RayOutcome.ACTOR_RETURN.value
        elif targets.object_id_for_evaluation_only[index]:
            outcomes[index] = RayOutcome.BACKGROUND_OCCLUSION.value
    return outcomes


def known_free_intervals(
    queries: QueryRayBatch,
    targets: RayTargets,
    *,
    endpoint_tolerance_m: float,
) -> np.ndarray:
    """只从可靠返回前方生成 FREE 区间；无回波不自动当作全量程 FREE。"""
    if len(queries) != len(targets):
        raise ValueError("query 与 target 射线数量不一致")
    intervals = np.full((len(queries), 2), np.nan, dtype=np.float32)
    usable = queries.valid_emission_mask & targets.return_valid & ~targets.censoring_or_missing_mask
    intervals[usable, 0] = queries.range_min_m[usable]
    intervals[usable, 1] = np.maximum(
        queries.range_min_m[usable],
        targets.range_m[usable] - float(endpoint_tolerance_m),
    )
    return intervals

"""把 renderer visibility 与 semantic evidence eligibility 显式解耦。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from motion_proj.worldsim_v5.bayesian_unary import (
    accumulate_effective_count_statistics,
)
from motion_proj.worldsim_v5.evidence_schema import validate_observation_chunk


def semantic_visibility_mask(
    observations: Mapping[str, np.ndarray],
    *,
    minimum_visibility: float,
) -> np.ndarray:
    """返回有资格更新 semantic posterior 的 observation mask。"""

    if (
        not np.isfinite(minimum_visibility)
        or minimum_visibility <= 0.0
        or minimum_visibility > 1.0
    ):
        raise ValueError("minimum_visibility 必须为有限 (0,1]")
    validate_observation_chunk(observations)
    accepted = np.asarray(observations["mask_quality_accepted"], dtype=bool)
    available = np.asarray(observations["sam_probability_available"], dtype=bool)
    visibility = np.asarray(observations["visibility"])
    applied_threshold = np.asarray(minimum_visibility, dtype=visibility.dtype).item()
    visible = visibility >= applied_threshold
    return accepted & available & visible


def accumulate_visibility_masked_b3_statistics(
    statistics: dict[str, np.ndarray],
    *,
    observations: Mapping[str, np.ndarray],
    gaussian_count: int,
    minimum_visibility: float,
    sam_confidence_floor: float,
    boundary_distance_scale_px: float,
    depth_residual_scale_m: float,
) -> dict[str, Any]:
    """只让可见且语义可用的 observation 更新冻结 B3 统计量。

    唯一机制变化是 hard visibility eligibility；通过门的 observation 继续使用
    V5 B3 的原始 reliability 和 soft SAM fractional count。
    """

    validate_observation_chunk(observations, gaussian_count=gaussian_count)
    eligibility = semantic_visibility_mask(
        observations, minimum_visibility=minimum_visibility
    )
    applied_threshold = np.asarray(
        minimum_visibility, dtype=np.asarray(observations["visibility"]).dtype
    ).item()
    semantic_available = (
        np.asarray(observations["mask_quality_accepted"], dtype=bool)
        & np.asarray(observations["sam_probability_available"], dtype=bool)
    )
    gated = dict(observations)
    gated["mask_quality_accepted"] = eligibility.astype(np.int8)
    gated["positive_observation"] = (
        np.asarray(observations["positive_observation"], dtype=np.float32)
        * eligibility
    ).astype(np.float32)
    gated["negative_observation"] = (
        np.asarray(observations["negative_observation"], dtype=np.float32)
        * eligibility
    ).astype(np.float32)
    reliability = accumulate_effective_count_statistics(
        statistics,
        observations=gated,
        gaussian_count=gaussian_count,
        sam_confidence_floor=sam_confidence_floor,
        boundary_distance_scale_px=boundary_distance_scale_px,
        depth_residual_scale_m=depth_residual_scale_m,
    )
    return {
        "observation_count": int(eligibility.size),
        "semantic_available_count": int(semantic_available.sum()),
        "visibility_qualified_count": int(eligibility.sum()),
        "visibility_rejected_count": int((semantic_available & ~eligibility).sum()),
        "semantic_unavailable_count": int((~semantic_available).sum()),
        "qualified_reliability_sum": float(reliability.sum(dtype=np.float64)),
        "configured_minimum_visibility": float(minimum_visibility),
        "applied_minimum_visibility": float(applied_threshold),
    }

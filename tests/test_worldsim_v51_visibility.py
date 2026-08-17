from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v5.bayesian_unary import (
    empty_effective_count_statistics,
    finalize_effective_count_unary,
)
from motion_proj.worldsim_v51.evidence.visibility import (
    accumulate_visibility_masked_b3_statistics,
    semantic_visibility_mask,
)


def _observations() -> dict[str, np.ndarray]:
    count = 3
    return {
        "scene": np.asarray("scene-0001"),
        "role": np.asarray("moving_rigid_union"),
        "sam_probability_source": np.asarray("sigmoid_sam2_logit"),
        "gaussian_id": np.asarray([0, 1, 2], dtype=np.int64),
        "view_id": np.arange(count, dtype=np.int32),
        "frame_id": np.zeros(count, dtype=np.int32),
        "camera_id": np.zeros(count, dtype=np.int8),
        "projected_pixel": np.zeros((count, 2), dtype=np.float32),
        "visibility": np.asarray([0.5, 0.005, 0.5], dtype=np.float32),
        "sam_probability": np.asarray([0.9, 0.9, 0.1], dtype=np.float32),
        "sam_logit": np.asarray([2.2, 2.2, -2.2], dtype=np.float32),
        "sam_probability_available": np.asarray([1, 1, 0], dtype=np.int8),
        "mask_quality_accepted": np.ones(count, dtype=np.int8),
        "mask_boundary_distance": np.ones(count, dtype=np.float32),
        "depth_residual": np.zeros(count, dtype=np.float32),
        "depth_consistent": np.ones(count, dtype=np.int8),
        "lidar_support": np.zeros(count, dtype=np.float32),
        "lidar_support_available": np.zeros(count, dtype=np.int8),
        "view_angle_cosine": np.ones(count, dtype=np.float32),
        "positive_observation": np.asarray([1, 1, 0], dtype=np.float32),
        "negative_observation": np.zeros(count, dtype=np.float32),
        "reliability": np.ones(count, dtype=np.float32),
        "contribution_weight": np.ones(count, dtype=np.float32),
    }


def test_missing_or_low_visibility_is_not_negative_evidence() -> None:
    observations = _observations()
    statistics = empty_effective_count_statistics(3)

    diagnostics = accumulate_visibility_masked_b3_statistics(
        statistics,
        observations=observations,
        gaussian_count=3,
        minimum_visibility=0.01,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    result = finalize_effective_count_unary(
        prior_probability=np.asarray([0.2, 0.2, 0.2]),
        prior_strength=2.0,
        statistics=statistics,
    )

    assert diagnostics["visibility_qualified_count"] == 1
    assert diagnostics["visibility_rejected_count"] == 1
    assert diagnostics["semantic_unavailable_count"] == 1
    assert result["unary_posterior"][0] > 0.2
    assert result["unary_posterior"][1] == pytest.approx(0.2)
    assert result["unary_posterior"][2] == pytest.approx(0.2)
    assert statistics["negative_mass"][1] == 0.0
    assert statistics["negative_mass"][2] == 0.0


def test_visibility_threshold_is_inclusive_and_fail_closed() -> None:
    observations = _observations()
    observations["visibility"] = np.asarray([0.01, 0.0099, 1.0], dtype=np.float32)

    assert semantic_visibility_mask(observations, minimum_visibility=0.01).tolist() == [
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="minimum_visibility"):
        semantic_visibility_mask(observations, minimum_visibility=0.0)


def test_visibility_config_changes_only_a1_mechanism() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/m1_unary_visibility_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["comparator"] == "B3"
    assert config["candidate"] == "A1"
    assert config["visibility"]["minimum_visibility"] == 0.01
    assert config["visibility"]["threshold_selection"]["quality_read"] is False
    assert config["matched_ablation"]["only_variable"] == (
        "hard_semantic_visibility_eligibility"
    )
    assert config["restrictions"]["graph"] is False

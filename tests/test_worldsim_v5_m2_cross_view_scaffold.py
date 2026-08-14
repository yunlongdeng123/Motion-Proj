from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from motion_proj.worldsim_v5.cross_view_scaffold import (
    CrossViewScaffoldError,
    ProjectedDepthStack,
    frozen_multicamera_source_views,
    frozen_source_views,
    fuse_cross_view_scaffold,
    lidar_agreement_audit,
)
from scripts.run_worldsim_v5_m2_cross_view_scaffold import (
    candidate_decision,
    load_config,
)


def _stack(depths: list[np.ndarray]) -> ProjectedDepthStack:
    values = np.stack(depths).astype(np.float32)
    return ProjectedDepthStack(
        depth=values,
        observed=np.isfinite(values),
        source_views=tuple((index, 0) for index in range(len(depths))),
        valid_source_pixels=tuple(int(np.isfinite(value).sum()) for value in values),
    )


def test_frozen_source_views_clip_scene_boundary_without_target_leakage() -> None:
    assert frozen_source_views(
        target_frame=2,
        camera_id=1,
        temporal_offsets=(-15, -10, -5, 5, 10, 15),
        minimum_frame=0,
        maximum_frame=195,
    ) == ((7, 1), (12, 1), (17, 1))


def test_frozen_multicamera_grid_adds_other_cameras_and_excludes_target() -> None:
    views = frozen_multicamera_source_views(
        target_frame=2,
        target_camera_id=1,
        camera_ids=(0, 1, 2),
        temporal_offsets=(-15, -10, -5, 5, 10, 15),
        minimum_frame=0,
        maximum_frame=195,
        include_same_frame_other_cameras=True,
    )
    assert len(views) == 11
    assert (2, 1) not in views
    assert (2, 0) in views and (2, 2) in views
    assert (7, 0) in views and (17, 2) in views


def test_fusion_requires_multiview_agreement_and_uses_bounded_extrapolation() -> None:
    fallback = np.full((5, 5), 30.0, dtype=np.float32)
    target = np.ones((5, 5), dtype=bool)
    first = np.full((5, 5), np.nan, dtype=np.float32)
    second = first.copy()
    first[2, 2] = 10.0
    second[2, 2] = 10.2
    first[0, 0] = 5.0
    second[0, 0] = 9.0
    result = fuse_cross_view_scaffold(
        fallback_depth=fallback,
        target_mask=target,
        projected=_stack([first, second]),
        minimum_support_views=2,
        maximum_absolute_disagreement_m=1.0,
        maximum_relative_disagreement=0.05,
        maximum_extrapolation_pixels=1.0,
    )
    assert result.direct_support[2, 2]
    assert not result.direct_support[0, 0]
    assert np.isclose(result.depth[2, 2], 10.1)
    assert np.isclose(result.depth[2, 3], 10.1)
    assert result.extrapolated_support[2, 3]
    assert result.depth[4, 4] == 30.0


def test_fusion_rejects_single_view_as_trusted_geometry() -> None:
    fallback = np.full((2, 2), 7.0, dtype=np.float32)
    source = np.full((2, 2), 4.0, dtype=np.float32)
    result = fuse_cross_view_scaffold(
        fallback_depth=fallback,
        target_mask=np.ones((2, 2), dtype=bool),
        projected=_stack([source]),
        minimum_support_views=2,
        maximum_absolute_disagreement_m=1.0,
        maximum_relative_disagreement=0.05,
        maximum_extrapolation_pixels=2.0,
    )
    assert not result.direct_support.any()
    assert np.array_equal(result.depth, fallback)


def test_lidar_is_audit_only() -> None:
    fallback = np.full((2, 2), 10.0, dtype=np.float32)
    result = fuse_cross_view_scaffold(
        fallback_depth=fallback,
        target_mask=np.ones((2, 2), dtype=bool),
        projected=_stack([fallback, fallback]),
        minimum_support_views=2,
        maximum_absolute_disagreement_m=1.0,
        maximum_relative_disagreement=0.05,
        maximum_extrapolation_pixels=0.0,
    )
    lidar = _stack([np.full((2, 2), 11.0, dtype=np.float32)])
    audit = lidar_agreement_audit(
        scaffold=result,
        lidar_projected=lidar,
        target_mask=np.ones((2, 2), dtype=bool),
    )
    assert audit["scaffold_lidar_mae_m"] == 1.0
    assert audit["lidar_used_to_modify_candidate"] is False


def test_source_view_contract_rejects_target_offset() -> None:
    with pytest.raises(CrossViewScaffoldError):
        frozen_source_views(
            target_frame=2,
            camera_id=0,
            temporal_offsets=(-1, 0, 1),
            minimum_frame=0,
            maximum_frame=10,
        )


def _decision_row(raw: float, post: float, g0: float = 2.0, dense: float = 2.1) -> dict:
    return {
        "status": "done",
        "baseline_replay_exact": True,
        "baseline": {
            "g0_raw_geometry_error": {"mae_m": g0},
            "dense_post_geometry_error": {"mae_m": dense},
        },
        "candidate": {
            "raw_geometry_error": {"mae_m": raw},
            "post_geometry_error": {"mae_m": post},
        },
    }


def test_candidate_decision_separates_relative_and_absolute_geometry_gates() -> None:
    gate = {
        "minimum_evaluable_request_count": 18,
        "minimum_raw_improvement_m": 0.5,
        "minimum_raw_improvement_request_count": 14,
        "minimum_post_improvement_m": 0.1,
        "minimum_post_improvement_request_count": 14,
        "require_mean_raw_delta_below_m": 0.0,
        "require_median_raw_delta_below_m": 0.0,
        "require_mean_post_delta_below_m": 0.0,
        "require_median_post_delta_below_m": 0.0,
        "geometry_safe_mae_m": 0.5,
        "minimum_geometry_safe_request_count": 14,
        "maximum_mean_post_minus_raw_mae_m": 0.1,
        "maximum_median_post_minus_raw_mae_m": 0.1,
    }
    result = candidate_decision([_decision_row(1.0, 1.0) for _ in range(18)], gate)
    assert result["relative_gate_passed"] is True
    assert result["absolute_geometry_safe_gate_passed"] is False
    assert (
        result["conclusion"]
        == "g4_cross_view_scaffold_relative_supported_absolute_safe_gate_failed"
    )
    assert result["method_arm_selected"] is False


def test_formal_config_withholds_target_reference_and_forbids_search() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_config(
        project / "configs/worldsim_v5/m2_cross_view_scaffold_scene0471_v1.yaml"
    )
    assert config["request_protocol"][
        "target_reference_interior_available_to_candidate"
    ] is False
    assert config["projection"]["target_depth_passed_to_projection"] == "all_nan"
    assert config["gaussianization"]["asset_provenance"] == "native_scene_donor"
    assert config["scope"]["parameter_search_performed"] is False
    assert config["scope"]["method_arm_selection_performed"] is False


def test_g5_config_changes_only_frozen_source_grid_and_keeps_gates() -> None:
    project = Path(__file__).resolve().parents[1]
    g4 = load_config(
        project / "configs/worldsim_v5/m2_cross_view_scaffold_scene0471_v1.yaml"
    )
    g5 = load_config(
        project / "configs/worldsim_v5/m2_multicamera_scaffold_scene0471_v1.yaml"
    )
    assert g5["source_views"]["camera_ids"] == [0, 1, 2]
    assert g5["source_views"]["expected_source_count_frame002"] == 11
    assert g5["source_views"]["expected_source_count_frame042"] == 20
    assert g5["candidate_gate"] == g4["candidate_gate"]
    assert g5["scaffold"]["minimum_support_views"] == g4["scaffold"][
        "minimum_support_views"
    ]
    assert g5["scaffold"]["maximum_extrapolation_pixels"] == g4["scaffold"][
        "maximum_extrapolation_pixels"
    ]
    assert g5["stop_condition"]["threshold_or_source_grid_search_after_result"] == "forbidden"

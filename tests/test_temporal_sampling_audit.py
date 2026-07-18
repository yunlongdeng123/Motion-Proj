from __future__ import annotations

import math

import pytest

from motion_proj.diagnostics.temporal_sampling_audit import (
    decide_temporal_conditioning,
    finite_summary,
    paired_relative_effect,
    quaternion_angle_rad,
    select_scene_distinct_clip_records,
)


def _thresholds() -> dict:
    return {
        "minimum_paired_groups": 4,
        "bootstrap_samples": 400,
        "bootstrap_seed": 11,
        "minimum_motion_relative_change": 0.10,
        "maximum_first_frame_psnr_drop_db": 1.0,
        "minimum_sharpness_ratio": 0.85,
        "maximum_flicker_ratio": 1.25,
        "minimum_survival_ratio": 0.90,
        "maximum_acceleration_p95_ratio": 1.25,
        "minimum_motion_floor_ratio": 0.80,
    }


def _rows(*, fps2_dynamic: float = 1.2, fps2_velocity: float = 1.2, fps2_sharpness: float = 1.0):
    rows = []
    for index in range(4):
        condition = f"c{index}"
        common = {
            "first_frame_psnr_db": 30.0,
            "spatial_laplacian_energy": 1.0,
            "global_luma_flicker_p95": 1.0,
            "survival_rate": 1.0,
            "acceleration_p95_px_per_frame2": 1.0,
        }
        for fps, dynamic, velocity, sharpness in (
            (2, fps2_dynamic, fps2_velocity, fps2_sharpness),
            (4, 1.0, 1.0, 1.0),
            (7, 1.0, 1.0, 1.0),
        ):
            rows.append(
                {
                    "condition_id": condition,
                    "generation_seed": 7,
                    "fps_input": fps,
                    "valid": True,
                    "metrics": {
                        **common,
                        "spatial_laplacian_energy": sharpness,
                        "dynamic_degree_mean_abs_rgb": dynamic,
                        "image_plane_velocity_rms_px": velocity,
                    },
                }
            )
    return rows


def test_scene_distinct_selection_uses_first_clip_per_scene() -> None:
    rows = [
        {"scene_name": "scene-b", "scene_token": "b", "start_index": 8, "sample_id": "b8"},
        {"scene_name": "scene-a", "scene_token": "a", "start_index": 8, "sample_id": "a8"},
        {"scene_name": "scene-a", "scene_token": "a", "start_index": 0, "sample_id": "a0"},
        {"scene_name": "scene-b", "scene_token": "b", "start_index": 0, "sample_id": "b0"},
    ]
    selected = select_scene_distinct_clip_records(rows, count=2)
    assert [row["sample_id"] for row in selected] == ["a0", "b0"]


def test_quaternion_angle_uses_shortest_sign_invariant_arc() -> None:
    identity = [1.0, 0.0, 0.0, 0.0]
    assert quaternion_angle_rad(identity, [-1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    yaw_90 = [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]
    assert quaternion_angle_rad(identity, yaw_90) == pytest.approx(math.pi / 2.0)


def test_finite_summary_does_not_zero_fill_invalid_values() -> None:
    summary = finite_summary([1.0, 2.0, float("nan")])
    assert summary["count"] == 2
    assert summary["median"] == pytest.approx(1.5)
    assert finite_summary([])["status"] == "invalid"


def test_paired_effect_bootstrap_detects_consistent_change() -> None:
    effect = paired_relative_effect(
        [(1.2, 1.0), (2.4, 2.0), (3.6, 3.0), (4.8, 4.0)],
        bootstrap_samples=200,
        seed=1,
    )
    assert effect["median_relative_change"] == pytest.approx(0.2)
    assert effect["ci95_low"] > 0.0


def test_decision_selects_real_fps_when_motion_changes_without_quality_harm() -> None:
    decision = decide_temporal_conditioning(
        _rows(),
        fps_values=[2, 4, 7],
        reference_fps=7,
        real_effective_fps=2.0,
        thresholds=_thresholds(),
    )
    assert decision["status"] == "done"
    assert decision["selected_fps"] == 2
    assert decision["comparisons"]["2"]["eligible"] is True


def test_decision_keeps_reference_for_low_motion_or_quality_collapse() -> None:
    low_motion = decide_temporal_conditioning(
        _rows(fps2_dynamic=0.7, fps2_velocity=0.7),
        fps_values=[2, 4, 7],
        reference_fps=7,
        real_effective_fps=2.0,
        thresholds=_thresholds(),
    )
    assert low_motion["selected_fps"] == 7
    assert low_motion["comparisons"]["2"]["quality_checks"]["dynamic_motion_floor"] is False

    blurred = decide_temporal_conditioning(
        _rows(fps2_sharpness=0.5),
        fps_values=[2, 4, 7],
        reference_fps=7,
        real_effective_fps=2.0,
        thresholds=_thresholds(),
    )
    assert blurred["selected_fps"] == 7
    assert blurred["comparisons"]["2"]["quality_checks"]["sharpness"] is False


def test_decision_blocks_when_paired_support_is_incomplete() -> None:
    rows = _rows()[:-3]
    decision = decide_temporal_conditioning(
        rows,
        fps_values=[2, 4, 7],
        reference_fps=7,
        real_effective_fps=2.0,
        thresholds=_thresholds(),
    )
    assert decision["status"] == "blocked"
    assert decision["selected_fps"] is None

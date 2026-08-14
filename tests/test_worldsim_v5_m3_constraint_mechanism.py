from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_worldsim_v5_m3_constraint_mechanism import (
    build_desired_trajectory,
    decide,
    load_config,
    quaternion_yaw,
)


PROJECT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(
        PROJECT / "configs/worldsim_v5/m3_constraint_projection_mechanism_v1.yaml"
    )


def _selected() -> dict:
    return {
        "scene": "scene-0001",
        "instance_token": "actor",
        "frames": [
            {
                "timestamp": index * 500_000,
                "translation": [float(index), 0.0, 1.0],
                "bottom_z": 0.0,
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
            for index in range(7)
        ],
    }


def _row(baseline: int, candidate: int) -> dict:
    return {
        "arms": {
            "T2_V4_FROZEN_SE3_BSPLINE": {"total_violation_count": baseline},
            "T5_T4_VEHICLE_KINEMATICS": {
                "total_violation_count": candidate,
                "endpoint_translation_error_m": 0.0,
                "endpoint_yaw_error_rad": 0.0,
                "violation_counts": {"contact": 0},
            },
        }
    }


def test_quaternion_yaw_and_local_lateral_template() -> None:
    assert quaternion_yaw([1.0, 0.0, 0.0, 0.0]) == 0.0
    config = _config()
    desired, road = build_desired_trajectory(
        _selected(), config["desired_edit_templates"]["LATERAL"]
    )
    assert np.array_equal(desired.times, np.linspace(0.0, 3.0, 7))
    assert np.array_equal(desired.positions[:3, 1], np.zeros(3))
    assert np.array_equal(desired.positions[3:, 1], np.ones(4))
    assert np.array_equal(road, np.zeros(7))


def test_decision_passes_only_as_mechanism_without_method_selection() -> None:
    config = _config()
    rows = [_row(4, 1) for _ in range(12)] + [_row(0, 0) for _ in range(4)]
    decision = decide(rows, config)
    assert decision["mechanism_gate_passed"] is True
    assert decision["matched_render_implementation_unlocked"] is True
    assert decision["method_arm_selected"] is False
    assert decision["validation_unlocked"] is False


def test_decision_is_inconclusive_without_t2_violation_signal() -> None:
    config = _config()
    rows = [_row(2, 0) for _ in range(7)] + [_row(0, 0) for _ in range(9)]
    decision = decide(rows, config)
    assert decision["mechanism_gate_passed"] is False
    assert decision["conclusion"].endswith("insufficient_t2_violation_signal")


def test_config_keeps_collision_render_and_method_selection_locked() -> None:
    config = _config()
    assert config["gate"]["request_count"] == 16
    assert config["gate"]["collision_gate_assessed"] is False
    assert config["gate"]["render_gate_assessed"] is False
    assert config["gate"]["method_selection_allowed"] is False


def test_v2_measurement_replay_is_explicitly_nonconfirmatory() -> None:
    config = load_config(
        PROJECT / "configs/worldsim_v5/m3_constraint_projection_mechanism_v2.yaml"
    )
    assert config["scope"]["result_aware_measurement_correction"] is True
    assert config["scope"]["confirmatory_claim_allowed"] is False
    assert config["physical_constraints"]["minimum_heading_speed_mps"] == 1.0
    assert config["physical_constraints"]["allow_reverse_heading"] is True

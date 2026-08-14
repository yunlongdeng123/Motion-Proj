from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_worldsim_v5_m2_closeout import (
    M2CloseoutError,
    build_decision,
    load_config,
)


def _expected() -> dict:
    return {
        "request_unit": "one_actor_one_view_one_hole",
        "request_count": 23,
        "evaluable_request_count": 22,
        "abstain_request_count": 1,
        "g0_raw_absolute_fail_count": 22,
        "rejected_surface_arms": [
            "G1_PIECEWISE_PLANE",
            "G2_MOVING_LEAST_SQUARES",
            "G3_ROBUST_QUADRATIC",
        ],
        "density_mechanism_supported": True,
        "opacity_mechanism_supported": False,
        "g4_conclusion": "g4_cross_view_scaffold_relative_gate_rejected",
        "g5_conclusion": "g5_multicamera_cross_view_scaffold_relative_supported_absolute_safe_gate_failed",
        "g5_raw_improvement_request_count": 15,
        "g5_post_improvement_request_count": 19,
        "g5_raw_geometry_safe_request_count": 1,
        "g5_post_geometry_safe_request_count": 0,
        "next_independent_task": "WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01",
    }


def _summaries() -> dict:
    summaries = {run_id: {"status": "done"} for run_id in (
        "r004", "r005", "r006", "r008", "r009", "r011", "r013", "r014"
    )}
    summaries["r005"].update(
        {
            "request_count": 23,
            "evaluable_request_count": 22,
            "abstain_request_count": 1,
            "mechanism": {"raw_failure_count": 22},
        }
    )
    summaries["r006"]["conclusion"] = "g1_piecewise_surface_rejected_on_model_proxy"
    summaries["r008"]["conclusion"] = "g2_moving_least_squares_surface_rejected_on_model_proxy"
    summaries["r009"]["conclusion"] = "g3_quadratic_surface_rejected_on_model_proxy"
    summaries["r011"]["mechanism_decision"] = {
        "supported_diagnostic_arms": ["DENSE", "DENSE_OPAQUE"]
    }
    summaries["r013"]["conclusion"] = "g4_cross_view_scaffold_relative_gate_rejected"
    summaries["r014"].update(
        {
            "conclusion": "g5_multicamera_cross_view_scaffold_relative_supported_absolute_safe_gate_failed",
            "validation_unlocked": False,
            "method_arm_selected": False,
            "candidate_decision": {
                "raw_improvement_request_count": 15,
                "post_improvement_request_count": 19,
                "raw_geometry_safe_request_count": 1,
                "post_geometry_safe_request_count": 0,
                "absolute_geometry_safe_gate_passed": False,
                "relative_gate_passed": True,
            },
        }
    )
    return summaries


def test_closeout_rejects_m2_without_unlocking_router_or_validation() -> None:
    decision = build_decision(_summaries(), _expected())
    assert decision["task_status"] == "rejected"
    assert decision["conclusion"] == "m2_rejected_no_absolute_geometry_safe_candidate"
    assert decision["g5_relative_gate_passed"] is True
    assert decision["absolute_geometry_safe_gate_passed"] is False
    assert decision["router_unlocked"] is False
    assert decision["validation_unlocked"] is False


def test_closeout_fails_if_absolute_safe_gate_is_rewritten() -> None:
    summaries = _summaries()
    summaries["r014"]["candidate_decision"]["absolute_geometry_safe_gate_passed"] = True
    with pytest.raises(M2CloseoutError):
        build_decision(summaries, _expected())


def test_formal_closeout_config_binds_all_terminals_and_no_launch_scope() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_config(project / "configs/worldsim_v5/m2_geometry_first_closeout_v1.yaml")
    assert set(config["blocked_terminals"]) == {"r001", "r007", "r010", "r012"}
    assert len(config["completed_runs"]) == 8
    assert config["scope"]["gpu_required"] is False
    assert config["expected_decision"]["final_task_status"] == "rejected"
    assert config["expected_decision"]["next_independent_task"] == (
        "WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01"
    )

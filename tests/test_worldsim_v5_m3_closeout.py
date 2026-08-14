from __future__ import annotations

from pathlib import Path

from scripts.run_worldsim_v5_m3_closeout import build_decision, load_config


PROJECT = Path(__file__).resolve().parents[1]


def test_closeout_rejects_projection_without_fishing_for_stress_templates() -> None:
    config = load_config(PROJECT / "configs/worldsim_v5/m3_closeout_v1.yaml")
    summaries = {
        "r001": {"conclusion": "m3_result_blind_protocol_frozen_development_implementation_unlocked"},
        "r003": {"ready_scene_count": 8},
        "r004": {"conclusion": "m3_constraint_projection_insufficient_t2_violation_signal"},
        "r005": {
            "conclusion": "m3_constraint_projection_insufficient_t2_violation_signal",
            "decision": {
                "request_count": 16,
                "t2_safe_request_count": 15,
                "t2_violation_evaluable_request_count": 1,
                "t2_total_violation_count": 2,
                "t5_total_violation_count": 1,
                "matched_render_implementation_unlocked": False,
                "mechanism_gate_passed": False,
            },
        },
    }
    decision = build_decision(summaries, config["expected_decision"])
    assert decision["task_status"] == "rejected"
    assert decision["post_hoc_stress_template_search_forbidden"] is True
    assert decision["renderer_unlocked"] is False
    assert decision["preserve_v4_temporal_result_as_historical_baseline"] is True


def test_closeout_scope_reads_no_new_quality() -> None:
    config = load_config(PROJECT / "configs/worldsim_v5/m3_closeout_v1.yaml")
    assert config["scope"]["gpu_required"] is False
    assert config["scope"]["renderer_started"] is False
    assert config["scope"]["validation_quality_read"] is False
    assert config["scope"]["test_quality_read"] is False

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_worldsim_v5_m3_protocol_audit import (
    M3ProtocolAuditError,
    build_protocol_lock,
    load_config,
)


PROJECT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT / "configs/worldsim_v5/m3_protocol_audit_v1.yaml")


def _m2() -> dict:
    return {
        "task_status": "rejected",
        "decision": {
            "router_unlocked": False,
            "next_independent_task": "WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01",
        },
    }


def _cohort(config: dict) -> dict:
    return {
        "status": "done",
        "freeze": {"scene_roles": {"development": config["protocol"]["fresh_development_scenes"]}},
    }


def _base(config: dict) -> dict:
    scenes = config["protocol"]["fresh_development_scenes"]
    return {
        "status": "done",
        "completed_scene_count": 8,
        "runs": [{"scene": scene} for scene in scenes],
        "validation_quality_read": False,
        "test_quality_read": False,
    }


def _historical() -> tuple[dict, dict]:
    return (
        {"scene_denominator": 6, "evaluable_scene_count": 3, "baseline_arm": "FRAME_INDEPENDENT"},
        {"scene_denominator": 18, "evaluable_scene_count": 12, "baseline_arm": "FRAME_INDEPENDENT"},
    )


def test_protocol_lock_separates_m2_remove_and_v4_statistics() -> None:
    config = _config()
    validation, test = _historical()
    lock = build_protocol_lock(
        config=config,
        m2=_m2(),
        cohort=_cohort(config),
        base_audit=_base(config),
        v4_validation=validation,
        v4_test=test,
    )
    assert lock["trajectory_primary_operations"] == ["LATERAL", "INSERT"]
    assert lock["comparator"] == "T2_V4_FROZEN_SE3_BSPLINE"
    assert lock["v4_statistics_reusable_for_v5_comparator"] is False
    assert lock["validation_unlocked"] is False


def test_protocol_rejects_m2_router_rewrite() -> None:
    config = _config()
    validation, test = _historical()
    m2 = _m2()
    m2["decision"]["router_unlocked"] = True
    with pytest.raises(M3ProtocolAuditError):
        build_protocol_lock(
            config=config,
            m2=m2,
            cohort=_cohort(config),
            base_audit=_base(config),
            v4_validation=validation,
            v4_test=test,
        )


def test_result_blind_config_keeps_optional_residual_and_quality_locked() -> None:
    config = _config()
    assert config["arms"]["optional_local_residual"]["enabled"] is False
    assert config["protocol"]["remove_physics_denominator"] is False
    assert config["scope"]["development_quality_read"] is False
    assert config["evidence_boundaries"]["fresh_test_unlock"] is False

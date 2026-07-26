import json
from pathlib import Path

from motion_proj.resim.cutin_receiver import (
    STRICT_V2_SCHEMA_VERSION,
    adapt_v1_evidence_to_v2,
    strict_v2_result,
)


def test_branch_merge_is_abstain_and_never_machine_positive():
    result = strict_v2_result(
        status="PASS",
        maneuver_mode="receiver_branch_merge",
    )
    assert result["schema_version"] == STRICT_V2_SCHEMA_VERSION
    assert result["status"] == "ABSTAIN"
    assert result["primary_reason"] == "UNSUPPORTED_BRANCH_MERGE_MODE"
    assert result["machine_positive"] is False


def test_reason_priority_and_machine_positive_are_fail_closed():
    result = strict_v2_result(
        status="FAIL",
        maneuver_mode="parallel_lane_change",
        reasons=["PATH_NOT_CLEAR", "NO_RAW_LATERAL_ENTRY"],
    )
    assert result["all_reasons"] == ["NO_RAW_LATERAL_ENTRY", "PATH_NOT_CLEAR"]
    assert result["primary_reason"] == "NO_RAW_LATERAL_ENTRY"
    assert result["machine_positive"] is False

    interpolation = strict_v2_result(
        status="PASS",
        maneuver_mode="parallel_lane_change",
        reasons=["INTERPOLATION_ONLY"],
    )
    assert interpolation["status"] == "ABSTAIN"
    assert interpolation["machine_positive"] is False


def test_old_v1_evidence_adapter_is_read_only_diagnostic():
    record = {
        "event_record_sha256": "source-hash",
        "maneuver_mode": "parallel_lane_change",
        "actor_id": 7,
        "source_run": {"token": "source"},
        "target_run": {"token": "target"},
        "cutin": {
            "schema_version": "receiver-centric-cutin-v1",
            "status": "UNKNOWN",
            "pre_keyframes": [],
            "post_keyframes": [],
            "per_frame": [],
            "subject_checks": {},
            "receiver_checks": {},
        },
    }
    result = adapt_v1_evidence_to_v2(record)
    assert result["status"] == "ABSTAIN"
    assert result["primary_reason"] == "INSUFFICIENT_RAW_SUPPORT"
    assert result["provenance"]["source_event_record_sha256"] == "source-hash"
    assert result["legacy_v1_diagnostic"]["schema_version"] == "receiver-centric-cutin-v1"
    assert result["uses_interpolated_physics"] is False


def test_frozen_k4_v1_evidence_remains_readable():
    fixture = Path(__file__).parent / "fixtures" / "n1_cutin_k4" / "evidence"
    branch = adapt_v1_evidence_to_v2(
        json.loads((fixture / "K4-001.json").read_text(encoding="utf-8"))
    )
    parallel = adapt_v1_evidence_to_v2(
        json.loads((fixture / "K4-010.json").read_text(encoding="utf-8"))
    )
    assert branch["status"] == "ABSTAIN"
    assert branch["primary_reason"] == "UNSUPPORTED_BRANCH_MERGE_MODE"
    assert parallel["schema_version"] == STRICT_V2_SCHEMA_VERSION

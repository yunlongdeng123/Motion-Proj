#!/usr/bin/env python3
"""冻结 V5 M3 result-blind 协议；只读身份和历史证据，不启动质量评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    finalize_formal_run,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01"
SCHEMA_VERSION = "worldsim_v5_m3_protocol_audit_v1"


class M3ProtocolAuditError(RuntimeError):
    """M3 协议身份、范围或证据边界漂移。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M3ProtocolAuditError("M3 protocol config schema 漂移")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "running":
        raise M3ProtocolAuditError("M3 protocol task/status 漂移")
    scope = payload["scope"]
    for name in (
        "gpu_required",
        "training",
        "model_inference",
        "development_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_quality_read",
        "parameter_search_performed",
        "method_arm_selection_performed",
    ):
        if scope.get(name) is not False:
            raise M3ProtocolAuditError(f"M3 result-blind restriction 漂移: {name}")
    protocol = payload["protocol"]
    if (
        protocol.get("trajectory_primary_operations") != ["LATERAL", "INSERT"]
        or protocol.get("remove_physics_denominator") is not False
        or payload["arms"].get("comparator") != "T2_V4_FROZEN_SE3_BSPLINE"
        or payload["arms"]["optional_local_residual"].get("enabled") is not False
    ):
        raise M3ProtocolAuditError("M3 arm/operation scope 漂移")
    boundaries = payload["evidence_boundaries"]
    for name in (
        "reuse_v4_aggregate_as_v5_comparator_evidence",
        "m2_geometry_claim_reuse",
        "m3_success_may_rewrite_m2",
        "fresh_validation_unlock",
        "fresh_test_unlock",
        "kitti_parameter_tuning",
    ):
        if boundaries.get(name) is not False:
            raise M3ProtocolAuditError(f"M3 evidence boundary 漂移: {name}")
    return payload


def build_protocol_lock(
    *,
    config: Mapping[str, Any],
    m2: Mapping[str, Any],
    cohort: Mapping[str, Any],
    base_audit: Mapping[str, Any],
    v4_validation: Mapping[str, Any],
    v4_test: Mapping[str, Any],
) -> dict[str, Any]:
    expected = config["expected_audit"]
    development = cohort.get("freeze", {}).get("scene_roles", {}).get("development", [])
    base_scenes = [row.get("scene") for row in base_audit.get("runs", [])]
    if (
        m2.get("task_status") != expected["m2_task_status"]
        or m2.get("decision", {}).get("router_unlocked") is not expected["m2_router_unlocked"]
        or m2.get("decision", {}).get("next_independent_task") != TASK_ID
    ):
        raise M3ProtocolAuditError("M2 closeout 未合法交接 M3")
    if (
        cohort.get("status") != "done"
        or development != config["protocol"]["fresh_development_scenes"]
        or len(development) != int(expected["fresh_development_scene_count"])
    ):
        raise M3ProtocolAuditError("fresh development cohort 漂移")
    if (
        base_audit.get("status") != "done"
        or base_audit.get("completed_scene_count")
        != int(expected["completed_base_scene_count"])
        or base_scenes != development
        or base_audit.get("validation_quality_read") is not False
        or base_audit.get("test_quality_read") is not False
    ):
        raise M3ProtocolAuditError("fresh development base identity 漂移")
    if (
        v4_validation.get("scene_denominator")
        != int(expected["v4_validation_scene_denominator"])
        or v4_validation.get("evaluable_scene_count")
        != int(expected["v4_validation_evaluable_scene_count"])
        or v4_test.get("scene_denominator")
        != int(expected["v4_test_scene_denominator"])
        or v4_test.get("evaluable_scene_count")
        != int(expected["v4_test_evaluable_scene_count"])
        or v4_validation.get("baseline_arm") != "FRAME_INDEPENDENT"
        or v4_test.get("baseline_arm") != "FRAME_INDEPENDENT"
    ):
        raise M3ProtocolAuditError("V4 historical M3 evidence 漂移")
    return {
        "conclusion": expected["conclusion"],
        "m2_task_status": m2["task_status"],
        "m2_router_unlocked": False,
        "fresh_development_scenes": list(development),
        "fresh_development_scene_count": len(development),
        "completed_base_scene_count": int(base_audit["completed_scene_count"]),
        "operations": list(config["protocol"]["operations"]),
        "trajectory_primary_operations": list(
            config["protocol"]["trajectory_primary_operations"]
        ),
        "remove_policy": config["protocol"]["remove_policy"],
        "comparator": config["arms"]["comparator"],
        "candidate_arms": list(config["arms"]["candidates"]),
        "optional_local_residual_enabled": False,
        "physical_constraints": dict(config["physical_constraints"]),
        "development_gates": dict(config["development_gates"]),
        "v4_historical_baseline": "FRAME_INDEPENDENT",
        "v4_statistics_reusable_for_v5_claim": False,
        "v4_statistics_reusable_for_v5_comparator": False,
        "development_implementation_unlocked": True,
        "development_quality_unlocked": False,
        "validation_unlocked": False,
        "test_unlocked": False,
        "kitti_parameter_tuning_unlocked": False,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events = [{"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}]
    write_events(run_dir, events)
    started = time.perf_counter()
    bindings = {
        name: verify_file(item["path"], item["sha256"])
        for name, item in config["frozen_inputs"].items()
    }
    payloads = {
        name: (
            yaml.safe_load(Path(binding["path"]).read_text(encoding="utf-8"))
            if Path(binding["path"]).suffix in {".yaml", ".yml"}
            else json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
        )
        for name, binding in bindings.items()
    }
    lock = build_protocol_lock(
        config=config,
        m2=payloads["m2_closeout_summary"],
        cohort=payloads["fresh_cohort"],
        base_audit=payloads["development_base_audit_summary"],
        v4_validation=payloads["v4_validation_summary"],
        v4_test=payloads["v4_test_summary"],
    )
    artifact = run_dir / "artifacts/protocol_lock.json"
    atomic_json(artifact, lock)
    snapshot = copy_source_snapshot(
        run_dir,
        [
            config_path,
            PROJECT / "scripts/run_worldsim_v5_m3_protocol_audit.py",
            PROJECT / "tests/test_worldsim_v5_m3_protocol_audit.py",
        ],
        PROJECT,
    )
    summary = {
        "schema_version": "worldsim_v5_m3_protocol_audit_summary_v1",
        "task_id": TASK_ID,
        "task_status": "running",
        "status": "done",
        "phase": config["phase"],
        "source_commit": source_head,
        "conclusion": lock["conclusion"],
        "protocol_lock": lock,
        "protocol_lock_sha256": sha256_file(artifact),
        "input_count": len(bindings),
        "source_snapshot_count": len(snapshot),
        "duration_seconds": time.perf_counter() - started,
        "gpu_started": False,
        "development_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_quality_read": False,
        "parameter_search_performed": False,
        "method_arm_selected": False,
    }
    events.append({"event": "run_done", "at_utc": utc_now(), **lock})
    events_record = write_events(run_dir, events)
    status = finalize_formal_run(
        run_dir=run_dir,
        task_id=TASK_ID,
        task_status="running",
        conclusion=lock["conclusion"],
        project_head=source_head,
        input_bindings=bindings,
        summary=summary,
        resolved_config_record=resolved,
        events_record=events_record,
    )
    return {**summary, "formal_status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""只读收口 V5 M3 development rejection，不启动 renderer 或质量读取。"""

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
SCHEMA_VERSION = "worldsim_v5_m3_closeout_v1"


class M3CloseoutError(RuntimeError):
    """M3 closeout evidence 或停止规则漂移。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M3CloseoutError("M3 closeout config schema 漂移")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "running":
        raise M3CloseoutError("M3 closeout task/status 漂移")
    if set(payload["completed_runs"]) != {"r001", "r003", "r004", "r005"}:
        raise M3CloseoutError("M3 completed denominator 漂移")
    if set(payload["blocked_terminal"]) != {"r002"}:
        raise M3CloseoutError("M3 blocked denominator 漂移")
    expected = payload["expected_decision"]
    if (
        expected.get("final_task_status") != "rejected"
        or expected.get("render_unlocked") is not False
        or expected.get("validation_unlocked") is not False
        or expected.get("post_hoc_stress_template_search_forbidden") is not True
    ):
        raise M3CloseoutError("M3 rejection/stop rule 漂移")
    for name, value in payload["scope"].items():
        if name.endswith(("required", "read", "started", "performed")) and value is not False:
            raise M3CloseoutError(f"M3 closeout scope 漂移: {name}")
    return payload


def build_decision(
    summaries: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, Any]:
    r001, r003, r004, r005 = (summaries[name] for name in ("r001", "r003", "r004", "r005"))
    decision = r005.get("decision", {})
    if (
        r001.get("conclusion")
        != "m3_result_blind_protocol_frozen_development_implementation_unlocked"
        or r003.get("ready_scene_count") != 8
        or r004.get("conclusion") != "m3_constraint_projection_insufficient_t2_violation_signal"
        or r005.get("conclusion") != "m3_constraint_projection_insufficient_t2_violation_signal"
        or decision.get("request_count") != int(expected["request_count"])
        or decision.get("t2_safe_request_count") != int(expected["t2_safe_request_count"])
        or decision.get("t2_violation_evaluable_request_count")
        != int(expected["t2_violation_evaluable_request_count"])
        or decision.get("t2_total_violation_count") != int(expected["t2_total_violation_count"])
        or decision.get("t5_total_violation_count") != int(expected["t5_total_violation_count"])
        or decision.get("matched_render_implementation_unlocked") is not False
        or decision.get("mechanism_gate_passed") is not False
    ):
        raise M3CloseoutError("M3 r001/r003/r004/r005 evidence 漂移")
    return {
        "task_status": "rejected",
        "conclusion": expected["conclusion"],
        "request_count": decision["request_count"],
        "t2_safe_request_count": decision["t2_safe_request_count"],
        "t2_violation_evaluable_request_count": decision[
            "t2_violation_evaluable_request_count"
        ],
        "minimum_required_evaluable_request_count": int(
            expected["minimum_required_evaluable_request_count"]
        ),
        "t2_total_violation_count": decision["t2_total_violation_count"],
        "t5_total_violation_count": decision["t5_total_violation_count"],
        "mechanism_gate_passed": False,
        "renderer_unlocked": False,
        "collision_gate_assessed": False,
        "method_arm_selected": False,
        "validation_unlocked": False,
        "test_unlocked": False,
        "kitti_parameter_tuning_unlocked": False,
        "preserve_v4_temporal_result_as_historical_baseline": True,
        "post_hoc_stress_template_search_forbidden": True,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events = [{"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}]
    write_events(run_dir, events)
    started = time.perf_counter()
    blocked_binding = config["blocked_terminal"]["r002"]
    blocked = verify_file(blocked_binding["path"], blocked_binding["sha256"])
    blocked_payload = json.loads(Path(blocked["path"]).read_text())
    if blocked_payload.get("status") != "blocked":
        raise M3CloseoutError("r002 不再是 blocked terminal")
    summaries: dict[str, Any] = {}
    inputs: dict[str, Any] = {"r002": blocked}
    for run_id, bindings in config["completed_runs"].items():
        summary_binding = verify_file(bindings["summary"]["path"], bindings["summary"]["sha256"])
        status_binding = verify_file(bindings["status"]["path"], bindings["status"]["sha256"])
        summary = json.loads(Path(summary_binding["path"]).read_text())
        status = json.loads(Path(status_binding["path"]).read_text())
        if status.get("status") != "done" or status.get("summary_sha256") != bindings["summary"]["sha256"]:
            raise M3CloseoutError(f"{run_id} summary/status binding 漂移")
        summaries[run_id] = summary
        inputs[run_id] = {"summary": summary_binding, "status": status_binding}
    decision = build_decision(summaries, config["expected_decision"])
    ledger_path = run_dir / "artifacts/decision_ledger.json"
    atomic_json(
        ledger_path,
        {
            "schema_version": "worldsim_v5_m3_closeout_decision_ledger_v1",
            "task_id": TASK_ID,
            "status": "done",
            "task_status": "rejected",
            "decision": decision,
            "blocked_terminal": {"r002": blocked_payload},
            "completed_conclusions": {name: row.get("conclusion") for name, row in summaries.items()},
        },
    )
    snapshot = copy_source_snapshot(
        run_dir,
        [config_path, PROJECT / "scripts/run_worldsim_v5_m3_closeout.py", PROJECT / "tests/test_worldsim_v5_m3_closeout.py"],
        PROJECT,
    )
    summary = {
        "schema_version": "worldsim_v5_m3_closeout_summary_v1",
        "task_id": TASK_ID,
        "task_status": "rejected",
        "status": "done",
        "phase": config["phase"],
        "source_commit": source_head,
        "conclusion": decision["conclusion"],
        "decision": decision,
        "completed_run_count": len(summaries),
        "blocked_run_count": 1,
        "decision_ledger_sha256": sha256_file(ledger_path),
        "source_snapshot_count": len(snapshot),
        "duration_seconds": time.perf_counter() - started,
        "gpu_started": False,
        "development_render_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_quality_read": False,
        "parameter_search_performed": False,
        "method_arm_selected": False,
    }
    events.append({"event": "run_done", "at_utc": utc_now(), **decision})
    events_record = write_events(run_dir, events)
    status = finalize_formal_run(
        run_dir=run_dir,
        task_id=TASK_ID,
        task_status="rejected",
        conclusion=decision["conclusion"],
        project_head=source_head,
        input_bindings=inputs,
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
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""只读收口 V5 M2 geometry-first development，不启动 GPU 或新质量读取。"""

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
    inventory_files,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M2-GEOMETRY-FIRST-REPAIR-01"
SCHEMA_VERSION = "worldsim_v5_m2_geometry_first_closeout_v1"


class M2CloseoutError(RuntimeError):
    """M2 closeout 输入、结论或 no-launch 合约失败。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2CloseoutError("M2 closeout config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "geometry_first_development_closeout"
        or payload["expected_decision"]["final_task_status"] != "rejected"
        or payload["expected_decision"]["router_unlocked"] is not False
        or payload["expected_decision"]["validation_unlocked"] is not False
    ):
        raise M2CloseoutError("M2 closeout task/decision 漂移")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "kitti_quality_read",
        "parameter_search_performed",
        "method_arm_selection_performed",
        "router_refit_performed",
        "gpu_required",
    ):
        if payload["scope"].get(name) is not False:
            raise M2CloseoutError(f"M2 closeout restriction 漂移: {name}")
    if set(payload["blocked_terminals"]) != {"r001", "r007", "r010", "r012"}:
        raise M2CloseoutError("blocked terminal denominator 漂移")
    if set(payload["completed_runs"]) != {
        "r004",
        "r005",
        "r006",
        "r008",
        "r009",
        "r011",
        "r013",
        "r014",
    }:
        raise M2CloseoutError("completed run denominator 漂移")
    return payload


def build_decision(
    summaries: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, Any]:
    required_done = set(summaries)
    if required_done != {
        "r004",
        "r005",
        "r006",
        "r008",
        "r009",
        "r011",
        "r013",
        "r014",
    } or any(summary.get("status") != "done" for summary in summaries.values()):
        raise M2CloseoutError("closeout completed summary denominator/status 漂移")
    g0 = summaries["r005"]
    surface_conclusions = {
        "G1_PIECEWISE_PLANE": summaries["r006"].get("conclusion"),
        "G2_MOVING_LEAST_SQUARES": summaries["r008"].get("conclusion"),
        "G3_ROBUST_QUADRATIC": summaries["r009"].get("conclusion"),
    }
    if (
        g0.get("request_count") != int(expected["request_count"])
        or g0.get("evaluable_request_count")
        != int(expected["evaluable_request_count"])
        or g0.get("abstain_request_count") != int(expected["abstain_request_count"])
        or g0.get("mechanism", {}).get("raw_failure_count")
        != int(expected["g0_raw_absolute_fail_count"])
        or set(surface_conclusions)
        != set(expected["rejected_surface_arms"])
        or any(not value.endswith("rejected_on_model_proxy") for value in surface_conclusions.values())
    ):
        raise M2CloseoutError("G0-G3 closeout evidence 漂移")
    gaussian = summaries["r011"]
    gaussian_decision = gaussian.get("mechanism_decision", {})
    density_supported = "DENSE" in gaussian_decision.get(
        "supported_diagnostic_arms", []
    )
    opacity_supported = "OPAQUE" in gaussian_decision.get(
        "supported_diagnostic_arms", []
    )
    g4 = summaries["r013"]
    g5 = summaries["r014"]
    g5_decision = g5.get("candidate_decision", {})
    if (
        density_supported is not bool(expected["density_mechanism_supported"])
        or opacity_supported is not bool(expected["opacity_mechanism_supported"])
        or g4.get("conclusion") != expected["g4_conclusion"]
        or g5.get("conclusion") != expected["g5_conclusion"]
        or g5_decision.get("raw_improvement_request_count")
        != int(expected["g5_raw_improvement_request_count"])
        or g5_decision.get("post_improvement_request_count")
        != int(expected["g5_post_improvement_request_count"])
        or g5_decision.get("raw_geometry_safe_request_count")
        != int(expected["g5_raw_geometry_safe_request_count"])
        or g5_decision.get("post_geometry_safe_request_count")
        != int(expected["g5_post_geometry_safe_request_count"])
        or g5_decision.get("absolute_geometry_safe_gate_passed") is not False
        or g5.get("validation_unlocked") is not False
        or g5.get("method_arm_selected") is not False
    ):
        raise M2CloseoutError("Gaussianization/G4/G5 closeout evidence 漂移")
    return {
        "task_status": "rejected",
        "conclusion": "m2_rejected_no_absolute_geometry_safe_candidate",
        "request_unit": expected["request_unit"],
        "request_count": int(expected["request_count"]),
        "evaluable_request_count": int(expected["evaluable_request_count"]),
        "abstain_request_count": int(expected["abstain_request_count"]),
        "g0_raw_absolute_fail_count": int(expected["g0_raw_absolute_fail_count"]),
        "surface_conclusions": surface_conclusions,
        "density_mechanism_supported": density_supported,
        "opacity_mechanism_supported": opacity_supported,
        "g4_conclusion": g4["conclusion"],
        "g5_conclusion": g5["conclusion"],
        "g5_relative_gate_passed": bool(
            g5_decision["relative_gate_passed"]
        ),
        "g5_raw_geometry_safe_request_count": int(
            g5_decision["raw_geometry_safe_request_count"]
        ),
        "g5_post_geometry_safe_request_count": int(
            g5_decision["post_geometry_safe_request_count"]
        ),
        "absolute_geometry_safe_gate_passed": False,
        "method_arm_selected": False,
        "router_unlocked": False,
        "validation_unlocked": False,
        "neural_surface_automatic_unlock": False,
        "parameter_search_performed": False,
        "router_refit_performed": False,
        "reference_scope": "model_derived_proxy_not_independent_ground_truth",
        "next_independent_task": expected["next_independent_task"],
    }


def _terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m2_closeout_status_v1",
            "task_id": TASK_ID,
            "task_status": "rejected" if status == "done" else "running",
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    started = time.perf_counter()
    try:
        blocked = {}
        for run_id, binding in config["blocked_terminals"].items():
            verified = verify_file(binding["path"], binding["sha256"])
            payload = json.loads(Path(verified["path"]).read_text())
            if payload.get("status") != "blocked":
                raise M2CloseoutError(f"{run_id} 不再是 blocked terminal")
            blocked[run_id] = {
                "failure_class": binding["failure_class"],
                "verified": verified,
                "terminal_reason": payload.get("reason"),
            }
        summaries = {}
        inputs = {"blocked_terminals": blocked, "completed_runs": {}}
        for run_id, bindings in config["completed_runs"].items():
            summary_binding = verify_file(
                bindings["summary"]["path"], bindings["summary"]["sha256"]
            )
            status_binding = verify_file(
                bindings["status"]["path"], bindings["status"]["sha256"]
            )
            summary = json.loads(Path(summary_binding["path"]).read_text())
            status = json.loads(Path(status_binding["path"]).read_text())
            if status.get("status") != "done" or status.get("summary_sha256") != bindings[
                "summary"
            ]["sha256"]:
                raise M2CloseoutError(f"{run_id} summary/status binding 漂移")
            summaries[run_id] = summary
            inputs["completed_runs"][run_id] = {
                "summary": summary_binding,
                "status": status_binding,
            }
        decision = build_decision(summaries, config["expected_decision"])
        ledger = {
            "schema_version": "worldsim_v5_m2_closeout_decision_ledger_v1",
            "task_id": TASK_ID,
            "status": "done",
            "task_status": "rejected",
            "decision": decision,
            "blocked_runs": blocked,
            "completed_run_conclusions": {
                run_id: summary.get("conclusion")
                for run_id, summary in summaries.items()
            },
            "quality_reads": {
                "validation": False,
                "heldout": False,
                "test": False,
                "kitti": False,
            },
        }
        ledger_path = run_dir / "artifacts/decision_ledger.json"
        atomic_json(ledger_path, ledger)
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_closeout.py",
                PROJECT / "tests/test_worldsim_v5_m2_closeout.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m2_closeout_summary_v1",
            "task_id": TASK_ID,
            "task_status": "rejected",
            "status": "done",
            "phase": config["phase"],
            "source_commit": source_head,
            "conclusion": decision["conclusion"],
            "decision": decision,
            "blocked_run_count": len(blocked),
            "completed_run_count": len(summaries),
            "decision_ledger_sha256": sha256_file(ledger_path),
            "duration_seconds": time.perf_counter() - started,
            "gpu_started": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "kitti_quality_read": False,
            "parameter_search_performed": False,
            "method_arm_selected": False,
            "router_refit_performed": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v5_m2_closeout_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_head,
                "source_clean": True,
                "resolved_config": resolved,
                "inputs": inputs,
                "source_snapshot": snapshot,
                "gpu_started": False,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now(), **decision})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_closeout_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "task_status": "rejected",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
            reason=None,
        )
        return summary
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        write_events(run_dir, events)
        _terminal(
            run_dir,
            status="blocked",
            source_head=source_head,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

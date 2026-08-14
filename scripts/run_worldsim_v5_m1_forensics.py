#!/usr/bin/env python3
"""Reproduce and freeze the V5 M1 diagnosis from immutable V4 artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from worldsim_v5_forensics_common import (
    ForensicAuditError,
    atomic_json,
    copy_source_snapshot,
    finalize_formal_run,
    load_json_mapping,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    verify_named_bindings,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M1-D0-BAYES-FORENSICS-01"


def summarize_state(path: Path, required_fields: list[str]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as state:
        present = set(state.files)
        missing = sorted(set(required_fields) - present)
        if missing:
            raise ForensicAuditError(f"V4 M1 state schema drift: {path} missing={missing}")
        count = int(state["gaussian_id"].shape[0])
        for name in required_fields:
            if name in {"actor_instance_id", "actor_token"}:
                continue
            if state[name].shape != (count,):
                raise ForensicAuditError(
                    f"V4 M1 state row shape drift: {path} {name}={state[name].shape}"
                )
        actor_id = int(np.asarray(state["actor_instance_id"]).item())
        hard = state["hard_instance_id"]
        posterior = state["posterior"]
        uncertainty = state["uncertainty"]
        positive = state["positive_count"]
        negative = state["negative_count"]
        target = hard == actor_id
        above_half = posterior >= 0.5
        target_count = int(np.count_nonzero(target))
        target_recalled = int(np.count_nonzero(target & above_half))
        return {
            "gaussian_count": count,
            "actor_instance_id": actor_id,
            "actor_token": str(np.asarray(state["actor_token"]).item()),
            "o1_proxy_target_count": target_count,
            "posterior_above_half_count": int(np.count_nonzero(above_half)),
            "o1_proxy_target_recalled_count": target_recalled,
            "o1_proxy_target_recall": target_recalled / target_count if target_count else None,
            "posterior_extreme_count": int(
                np.count_nonzero((posterior <= 0.01) | (posterior >= 0.99))
            ),
            "posterior_extreme_fraction": float(
                np.mean((posterior <= 0.01) | (posterior >= 0.99))
            ),
            "uncertainty_le_1e3_count": int(np.count_nonzero(uncertainty <= 1e-3)),
            "uncertainty_le_1e3_fraction": float(np.mean(uncertainty <= 1e-3)),
            "unobserved_count": int(np.count_nonzero((positive + negative) == 0)),
            "unobserved_fraction": float(np.mean((positive + negative) == 0)),
            "mixed_positive_negative_count": int(
                np.count_nonzero((positive > 0) & (negative > 0))
            ),
            "mixed_positive_negative_fraction": float(
                np.mean((positive > 0) & (negative > 0))
            ),
            "present_fields": sorted(present),
        }


def missing_collection_fields(
    present_fields: set[str], binding: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, list[str]]:
    virtual = {"scene", "role"}
    available = present_fields | virtual | set(binding)
    return {
        group: sorted(field for field in fields if field not in available)
        for group, fields in contract.items()
    }


def audit(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema_version") != "worldsim_v5_m1_forensics_v1":
        raise ForensicAuditError("M1 forensic config schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ForensicAuditError("M1 forensic task registration drift")
    restrictions = config["restrictions"]
    for key in (
        "fresh_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "training_performed",
    ):
        if restrictions.get(key) is not False:
            raise ForensicAuditError(f"M1 forensic restriction violated: {key}")

    historical = config["historical_binding"]
    verified_terminal = verify_named_bindings(historical["run"], historical["files"])
    summary = load_json_mapping(Path(verified_terminal["summary.json"]["path"]))
    metrics = load_json_mapping(Path(verified_terminal["metrics.json"]["path"]))
    manifest = load_json_mapping(Path(verified_terminal["manifest.json"]["path"]))
    if summary.get("status") != "done" or manifest.get("status") != "done":
        raise ForensicAuditError("M1 r200 is not terminal done")
    gate = summary.get("confirmation_gate")
    if gate != metrics.get("confirmation_gate") or gate.get("status") != "reject":
        raise ForensicAuditError("M1 r200 rejection gate drift")
    if gate.get("directional_support_scene_count") != 0:
        raise ForensicAuditError("M1 r200 directional support drift")
    for key in (
        "arm_search_performed",
        "calibration_fit_performed",
        "threshold_search_performed",
    ):
        if gate.get(key) is not False:
            raise ForensicAuditError(f"M1 r200 unexpectedly optimized: {key}")

    required_fields = list(config["historical_state_schema"]["required_fields"])
    collection_contract = config["future_collection_contract"]
    state_rows: list[dict[str, Any]] = []
    missing_by_state: dict[str, Any] = {}
    verified_states: dict[str, Any] = {}
    for binding in historical["state_files"]:
        record = verify_file(binding["path"], binding["sha256"])
        key = f'{binding["scene"]}/{binding["role"]}'
        stats = summarize_state(Path(record["path"]), required_fields)
        stats.update({"scene": binding["scene"], "role": binding["role"]})
        state_rows.append(stats)
        verified_states[key] = record
        missing_by_state[key] = missing_collection_fields(
            set(stats["present_fields"]), binding, collection_contract
        )

    expected = config["expected_historical_result"]
    denominator = summary["cohort_accounting"]
    observed = {
        "required_scene_count": denominator["required_scene_count"],
        "evaluable_scene_count": denominator["evaluable_scene_count"],
        "abstain_scene_count": denominator["abstain_scene_count"],
        "directional_support_scene_count": gate["directional_support_scene_count"],
        "state_count": len(state_rows),
    }
    if observed != expected["counts"]:
        raise ForensicAuditError(f"M1 historical count drift: {observed}")
    missing_counter = Counter(
        field
        for groups in missing_by_state.values()
        for fields in groups.values()
        for field in fields
    )
    required_blockers = set(config["required_blocked_evidence"])
    if not required_blockers <= set(missing_counter):
        raise ForensicAuditError(
            f"M1 required missing-evidence blocker disappeared: {required_blockers - set(missing_counter)}"
        )
    state_audit = {
        "schema_version": "worldsim_v5_m1_state_audit_v1",
        "states": state_rows,
        "missing_collection_fields_by_state": missing_by_state,
        "missing_field_state_counts": dict(sorted(missing_counter.items())),
    }
    audit_summary = {
        "schema_version": "worldsim_v5_m1_forensics_summary_v1",
        "task_id": TASK_ID,
        "task_status": "done",
        "status": "done",
        "conclusion": "blocked_evidence_missing_contract_frozen",
        "scope": "v4_historical_diagnostic_only",
        "historical_run": historical["run"],
        "historical_counts": observed,
        "historical_mean_deltas": gate["mean_gate"]["deltas"],
        "historical_per_scene_directions": gate["per_scene_directions"],
        "state_audit_path": "artifacts/state_audit.json",
        "required_blocked_evidence": sorted(required_blockers),
        "missing_evidence_collection_contract_frozen": True,
        "full_m1_implementation_authorized_by_this_run": False,
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "training_performed": False,
        "finished_at_utc": utc_now(),
    }
    bindings = {"terminal_files": verified_terminal, "state_files": verified_states}
    return audit_summary, {"state_audit": state_audit, "input_bindings": bindings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    project_head = prepare_formal_run(args.run_dir, TASK_ID, project)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved_config = write_resolved_config(args.run_dir, config)
    source_inventory = copy_source_snapshot(
        args.run_dir,
        [Path(__file__), Path(__file__).with_name("worldsim_v5_forensics_common.py"), args.config],
        project,
    )
    started = utc_now()
    summary, payload = audit(config)
    summary = dict(summary)
    summary["project_git_head"] = project_head
    summary["source_snapshot"] = source_inventory
    atomic_json(args.run_dir / "artifacts/state_audit.json", payload["state_audit"])
    events = write_events(
        args.run_dir,
        [
            {"event": "audit_started", "task_id": TASK_ID, "timestamp_utc": started},
            {
                "event": "audit_completed",
                "task_id": TASK_ID,
                "conclusion": summary["conclusion"],
                "timestamp_utc": utc_now(),
            },
        ],
    )
    status = finalize_formal_run(
        run_dir=args.run_dir,
        task_id=TASK_ID,
        task_status="done",
        conclusion=summary["conclusion"],
        project_head=project_head,
        input_bindings=payload["input_bindings"],
        summary=summary,
        resolved_config_record=resolved_config,
        events_record=events,
    )
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

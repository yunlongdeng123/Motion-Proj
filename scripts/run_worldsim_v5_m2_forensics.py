#!/usr/bin/env python3
"""Reproduce and freeze the V5 M2 geometry diagnosis from immutable V4 artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import yaml

from worldsim_v5_forensics_common import (
    ForensicAuditError,
    atomic_json,
    copy_source_snapshot,
    finalize_formal_run,
    load_json_list,
    load_json_mapping,
    prepare_formal_run,
    utc_now,
    verify_file,
    verify_named_bindings,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M2-D0-GEOMETRY-FORENSICS-01"


def recursive_field_names(value: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            fields.add(str(key))
            fields.update(recursive_field_names(child))
    elif isinstance(value, list):
        for child in value:
            fields.update(recursive_field_names(child))
    return fields


def extract_requests(
    scene_payloads: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    requests: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for scene, payload in scene_payloads.items():
        for source_key in ("requests", "blocked_requests"):
            for request in payload.get(source_key, []):
                request_id = str(request["request_id"])
                if request_id in requests:
                    raise ForensicAuditError(f"duplicate M2 request_id: {request_id}")
                normalized = dict(request)
                normalized["scene"] = scene
                normalized["source_key"] = source_key
                requests[request_id] = normalized
                for entry in request.get("candidates", []):
                    candidate = entry["candidate"]
                    metrics = entry["metrics"]
                    candidates.append(
                        {
                            "request_id": request_id,
                            "scene": scene,
                            "arm": entry["arm"],
                            "candidate_id": candidate["candidate_id"],
                            "geometry_risk": float(candidate["geometry_risk"]),
                            "hole_geometry_mae_m": float(metrics["hole_geometry_mae_m"]),
                            "raw": entry,
                        }
                    )
    return requests, candidates


def _baseline_telea_mae(request: Mapping[str, Any]) -> float:
    for entry in request.get("candidates", []):
        if entry.get("arm") == "TELEA":
            return float(entry["metrics"]["hole_geometry_mae_m"])
    for entry in request.get("matched_arms", []):
        if entry.get("arm") == "ABSTAIN" and "metrics" in entry:
            return float(entry["metrics"]["hole_geometry_mae_m"])
    raise ForensicAuditError(f'no TELEA/ABSTAIN baseline for {request.get("request_id")}')


def _group_geometry(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    router = [float(row["router_mae_m"]) for row in materialized]
    telea = [float(row["telea_mae_m"]) for row in materialized]
    per_scene: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in materialized:
        per_scene[str(row["scene"])].append(
            (float(row["router_mae_m"]), float(row["telea_mae_m"]))
        )
    scene_deltas = [
        mean(pair[0] for pair in values) - mean(pair[1] for pair in values)
        for values in per_scene.values()
    ]
    return {
        "count": len(materialized),
        "router_request_mean_mae_m": mean(router),
        "telea_request_mean_mae_m": mean(telea),
        "request_mean_delta_m": mean(router) - mean(telea),
        "scene_balanced_delta_m": mean(scene_deltas),
        "full_denominator_additive_contribution_m": sum(
            r - t for r, t in zip(router, telea)
        ),
    }


def geometry_diagnostics(
    requests: Mapping[str, Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    saturation_scale_m: float,
) -> dict[str, Any]:
    candidates_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidates:
        candidates_by_request[row["request_id"]].append(row)
        candidate_by_id[row["candidate_id"]] = row
        expected_risk = min(max(row["hole_geometry_mae_m"] / saturation_scale_m, 0.0), 1.0)
        if not math.isclose(row["geometry_risk"], expected_risk, abs_tol=1e-12):
            raise ForensicAuditError(
                f'geometry risk mapping drift: {row["candidate_id"]}'
            )

    arm_counts = Counter(row["arm"] for row in candidates)
    saturation_counts = Counter(
        row["arm"] for row in candidates if math.isclose(row["geometry_risk"], 1.0)
    )
    unique_by_arm = {
        arm: len({row["geometry_risk"] for row in candidates if row["arm"] == arm})
        for arm in sorted(arm_counts)
    }
    threshold_checks = {}
    for threshold in (0.5, 1.0, 2.0, 5.0):
        eligible = [row for row in candidates if row["hole_geometry_mae_m"] >= threshold]
        threshold_checks[str(threshold)] = {
            "candidate_count": len(eligible),
            "saturated_count": sum(
                math.isclose(row["geometry_risk"], 1.0) for row in eligible
            ),
        }
    collision_count = 0
    for rows in candidates_by_request.values():
        if (
            len(rows) > 1
            and len({row["hole_geometry_mae_m"] for row in rows}) > 1
            and len({row["geometry_risk"] for row in rows}) == 1
        ):
            collision_count += 1

    decision_by_request = {str(row["request_id"]): row for row in decisions}
    if set(decision_by_request) != set(requests):
        raise ForensicAuditError("M2 request/decision denominator drift")
    accepted_regrets: list[float] = []
    exact_oracle = 0
    oracle_arm_counts: Counter[str] = Counter()
    positive_regret_selected_arms: Counter[str] = Counter()
    denominator_rows: list[dict[str, Any]] = []
    for request_id, request in requests.items():
        decision = decision_by_request[request_id]
        rows = candidates_by_request.get(request_id, [])
        if rows:
            oracle = min(rows, key=lambda row: row["hole_geometry_mae_m"])
            oracle_arm_counts[oracle["arm"]] += 1
        if decision["accepted"]:
            selected_id = str(decision["selected_candidate_id"])
            if selected_id not in candidate_by_id:
                raise ForensicAuditError(f"accepted candidate not found: {selected_id}")
            selected = candidate_by_id[selected_id]
            oracle_mae = min(row["hole_geometry_mae_m"] for row in rows)
            regret = selected["hole_geometry_mae_m"] - oracle_mae
            if regret < -1e-12:
                raise ForensicAuditError(f"negative oracle regret: {request_id}")
            regret = max(regret, 0.0)
            accepted_regrets.append(regret)
            if math.isclose(regret, 0.0, abs_tol=1e-12):
                exact_oracle += 1
            else:
                positive_regret_selected_arms[selected["arm"]] += 1

        if decision["accepted"]:
            group = "accepted"
        elif rows:
            group = "risk_abstain"
        else:
            group = "role_asset_blocked"
        denominator_rows.append(
            {
                "request_id": request_id,
                "scene": request["scene"],
                "group": group,
                "router_mae_m": float(decision["metrics"]["hole_geometry_mae_m"]),
                "telea_mae_m": _baseline_telea_mae(request),
            }
        )

    denominator = {
        group: _group_geometry(row for row in denominator_rows if row["group"] == group)
        for group in ("accepted", "risk_abstain", "role_asset_blocked")
    }
    denominator["full_denominator"] = _group_geometry(denominator_rows)
    full_count = denominator["full_denominator"]["count"]
    for group in ("accepted", "risk_abstain", "role_asset_blocked"):
        denominator[group]["full_denominator_additive_contribution_m"] /= full_count
    denominator["full_denominator"]["full_denominator_additive_contribution_m"] /= full_count

    all_fields: set[str] = set()
    for row in candidates:
        all_fields.update(recursive_field_names(row["raw"]))
    return {
        "request_count": len(requests),
        "measured_request_count": len(candidates_by_request),
        "candidate_count": len(candidates),
        "arm_counts": dict(sorted(arm_counts.items())),
        "saturation_counts": dict(sorted(saturation_counts.items())),
        "saturation_fraction": sum(saturation_counts.values()) / len(candidates),
        "unique_risk_count": len({row["geometry_risk"] for row in candidates}),
        "unique_risk_count_by_arm": unique_by_arm,
        "threshold_saturation_checks": threshold_checks,
        "same_risk_different_mae_request_count": collision_count,
        "router": {
            "accepted_count": sum(bool(row["accepted"]) for row in decisions),
            "abstain_count": sum(not bool(row["accepted"]) for row in decisions),
        },
        "geometry_oracle": {
            "measured_request_count": len(candidates_by_request),
            "oracle_arm_counts": dict(sorted(oracle_arm_counts.items())),
            "accepted_count": len(accepted_regrets),
            "accepted_exact_oracle_count": exact_oracle,
            "accepted_positive_regret_count": sum(value > 0 for value in accepted_regrets),
            "accepted_regret_mean_m": mean(accepted_regrets),
            "accepted_regret_median_m": median(accepted_regrets),
            "accepted_regret_max_m": max(accepted_regrets),
            "accepted_regret_sum_m": sum(accepted_regrets),
            "positive_regret_selected_arm_counts": dict(
                sorted(positive_regret_selected_arms.items())
            ),
        },
        "denominator_decomposition": denominator,
        "present_candidate_field_names": sorted(all_fields),
    }


def _assert_expected(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        observed = actual.get(key)
        if isinstance(value, Mapping):
            if not isinstance(observed, Mapping):
                raise ForensicAuditError(f"M2 expected mapping missing: {key}")
            _assert_expected(observed, value)
        elif isinstance(value, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), value, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ForensicAuditError(f"M2 expected float drift: {key}={observed} != {value}")
        elif observed != value:
            raise ForensicAuditError(f"M2 expected value drift: {key}={observed} != {value}")


def audit(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema_version") != "worldsim_v5_m2_forensics_v1":
        raise ForensicAuditError("M2 forensic config schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") not in {"running", "done"}:
        raise ForensicAuditError("M2 forensic task registration drift")
    restrictions = config["restrictions"]
    for key in (
        "fresh_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "router_refit_performed",
        "training_performed",
    ):
        if restrictions.get(key) is not False:
            raise ForensicAuditError(f"M2 forensic restriction violated: {key}")

    historical = config["historical_binding"]
    verified_terminal = verify_named_bindings(historical["run"], historical["files"])
    summary = load_json_mapping(Path(verified_terminal["summary.json"]["path"]))
    decisions = load_json_list(Path(verified_terminal["router_decisions.json"]["path"]))
    if summary.get("status") != "done" or summary.get("validation_optimization_read") is not False:
        raise ForensicAuditError("M2 r222 terminal/provenance drift")
    if summary.get("test_quality_read") is not False or summary.get("heldout_content_read") is not False:
        raise ForensicAuditError("M2 r222 heldout/test provenance drift")

    scene_payloads: dict[str, dict[str, Any]] = {}
    verified_scenes: dict[str, Any] = {}
    for scene, binding in historical["scene_summary_files"].items():
        record = verify_file(binding["path"], binding["sha256"])
        verified_scenes[scene] = record
        payload = load_json_mapping(Path(record["path"]))
        if payload.get("scene") != scene:
            raise ForensicAuditError(f"M2 scene binding drift: {scene}")
        if "requests" in payload or "blocked_requests" in payload:
            scene_payloads[scene] = payload
    requests, candidates = extract_requests(scene_payloads)
    diagnostics = geometry_diagnostics(
        requests,
        candidates,
        decisions,
        float(config["frozen_v4_mapping"]["mae_saturation_scale_m"]),
    )
    _assert_expected(diagnostics, config["expected_historical_result"])

    present = set(diagnostics["present_candidate_field_names"])
    missing_contract = {
        group: sorted(field for field in fields if field not in present)
        for group, fields in config["future_collection_contract"].items()
    }
    missing = {field for fields in missing_contract.values() for field in fields}
    blockers = set(config["required_blocked_evidence"])
    if not blockers <= missing:
        raise ForensicAuditError(
            f"M2 required missing-evidence blocker disappeared: {blockers - missing}"
        )
    diagnostics["missing_collection_contract_fields"] = missing_contract
    audit_summary = {
        "schema_version": "worldsim_v5_m2_forensics_summary_v1",
        "task_id": TASK_ID,
        "task_status": "done",
        "status": "done",
        "conclusion": "risk_saturation_and_blocked_evidence_missing_contract_frozen",
        "scope": "v4_historical_diagnostic_only",
        "historical_run": historical["run"],
        "geometry_audit_path": "artifacts/geometry_audit.json",
        "historical_counts": {
            "request_count": diagnostics["request_count"],
            "measured_request_count": diagnostics["measured_request_count"],
            "candidate_count": diagnostics["candidate_count"],
            **diagnostics["router"],
        },
        "saturation_fraction": diagnostics["saturation_fraction"],
        "geometry_oracle": diagnostics["geometry_oracle"],
        "denominator_decomposition": diagnostics["denominator_decomposition"],
        "required_blocked_evidence": sorted(blockers),
        "reference_and_pre_post_geometry_contract_frozen": True,
        "full_m2_implementation_authorized_by_this_run": False,
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "router_refit_performed": False,
        "training_performed": False,
        "finished_at_utc": utc_now(),
    }
    bindings = {"terminal_files": verified_terminal, "scene_summary_files": verified_scenes}
    return audit_summary, {"geometry_audit": diagnostics, "input_bindings": bindings}


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
    atomic_json(args.run_dir / "artifacts/geometry_audit.json", payload["geometry_audit"])
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

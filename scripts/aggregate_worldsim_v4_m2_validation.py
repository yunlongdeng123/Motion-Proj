#!/usr/bin/env python3
"""按冻结路由聚合 nuScenes M2 validation，并生成只读确认裁决。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

from scripts.aggregate_worldsim_v4_m2_development import (
    M2DevelopmentAggregationError,
    _matched_table,
    _route_request,
    _selection_statistics,
    _selective_curve,
    evaluate_acceptance_gates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "WS-V4-M2-REPAIR-ROUTER-01"


class M2ValidationAggregationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M2ValidationAggregationError(f"JSON root is not a mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M2ValidationAggregationError(f"YAML root is not a mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )


def _verify_binding(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise M2ValidationAggregationError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise M2ValidationAggregationError(
            f"{label} SHA drift: expected={expected} actual={actual}"
        )


def _validate_matched_rows(
    request: Mapping[str, Any], expected_arms: Sequence[str], *, blocked: bool
) -> None:
    if [row.get("arm") for row in request.get("matched_arms", [])] != list(expected_arms):
        raise M2ValidationAggregationError(
            f"{request.get('request_id')} matched arm order drift"
        )
    abstain = [
        row
        for row in request["matched_arms"]
        if row.get("arm") == "ABSTAIN" and row.get("status") == "atomic_noop"
    ]
    if len(abstain) != 1 or abstain[0].get("metrics", {}).get("atomic_noop") is not True:
        raise M2ValidationAggregationError(
            f"{request.get('request_id')} lacks measured atomic no-op"
        )
    if blocked:
        if (
            request.get("status") != "abstain"
            or not str(request.get("reason", "")).startswith("ABSTAIN_")
            or request.get("candidates") != []
        ):
            raise M2ValidationAggregationError(
                f"{request.get('request_id')} blocked-request contract drift"
            )


def collect_validation_requests(
    summaries: Sequence[Mapping[str, Any]], expected_arms: Sequence[str]
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for summary in summaries:
        if summary.get("status") != "done":
            continue
        scene = str(summary["scene"])
        measured = list(summary.get("requests", []))
        blocked = list(summary.get("blocked_requests", []))
        if len(measured) != int(summary.get("request_count", -1)):
            raise M2ValidationAggregationError(f"{scene} request accounting drift")
        if len(blocked) != int(summary.get("blocked_request_count", -1)):
            raise M2ValidationAggregationError(f"{scene} blocked accounting drift")
        if len(measured) + len(blocked) != int(summary.get("total_request_count", -1)):
            raise M2ValidationAggregationError(f"{scene} denominator accounting drift")
        for request in measured:
            _validate_matched_rows(request, expected_arms, blocked=False)
            requests.append({"scene": scene, **request})
        for request in blocked:
            _validate_matched_rows(request, expected_arms, blocked=True)
            requests.append({"scene": scene, **request})
    if not requests:
        raise M2ValidationAggregationError("validation has no measured repair requests")
    if len({row["request_id"] for row in requests}) != len(requests):
        raise M2ValidationAggregationError("duplicate validation request ID")
    return requests


def select_frozen_baseline(
    table: Sequence[Mapping[str, Any]], development_summary: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected = str(development_summary["best_matched_non_router"])
    matches = [row for row in table if row.get("arm") == expected]
    if len(matches) != 1 or expected == "RISK_ROUTER":
        raise M2ValidationAggregationError("frozen development baseline is unavailable")
    return matches[0]


def _load_verified_inputs(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol_binding = config["validation_protocol"]
    protocol_path = Path(protocol_binding["path"])
    _verify_binding(protocol_path, protocol_binding["sha256"], "validation protocol")
    protocol = _yaml(protocol_path)
    if (
        protocol.get("schema_version") != "worldsim_v4_m2_validation_v1"
        or protocol.get("task_id") != TASK_ID
        or protocol.get("partition") != "validation"
        or protocol.get("status") != "pending"
    ):
        raise M2ValidationAggregationError("validation protocol identity drift")

    freeze_binding = config["development_freeze"]
    freeze_path = Path(freeze_binding["run"]) / "summary.json"
    _verify_binding(freeze_path, freeze_binding["summary_sha256"], "development freeze")
    freeze = _json(freeze_path)
    frozen_router = freeze.get("frozen_router", {})
    if (
        freeze.get("status") != "done"
        or freeze.get("development_gate_passed") is not True
        or freeze.get("validation_authorized") is not True
        or freeze.get("best_matched_non_router")
        != freeze_binding["expected_best_matched_non_router"]
        or frozen_router.get("weight_name") != freeze_binding["expected_weight_name"]
        or frozen_router.get("weights") != freeze_binding["expected_weights"]
        or float(frozen_router.get("threshold"))
        != float(freeze_binding["expected_threshold"])
        or frozen_router.get("tie_priority") != freeze_binding["expected_tie_priority"]
    ):
        raise M2ValidationAggregationError("development freeze drift")

    expected_scenes = list(protocol["protocol"]["scene_order"])
    if list(config["scene_runs"]) != expected_scenes:
        raise M2ValidationAggregationError("validation scene order is not frozen cohort")
    summaries: list[dict[str, Any]] = []
    candidate_protocol_path = Path(protocol["candidate_protocol"]["path"])
    _verify_binding(
        candidate_protocol_path,
        protocol["candidate_protocol"]["sha256"],
        "candidate protocol",
    )
    candidate_protocol = _yaml(candidate_protocol_path)
    expected_arms = list(candidate_protocol["ablations"]["matched_repair_arms"])
    for scene in expected_scenes:
        binding = config["scene_runs"][scene]
        summary_path = Path(binding["path"]) / "summary.json"
        _verify_binding(summary_path, binding["summary_sha256"], f"{scene} summary")
        summary = _json(summary_path)
        if (
            summary.get("task_id") != TASK_ID
            or summary.get("scene") != scene
            or summary.get("partition") != "validation"
            or summary.get("development_content_read") is not False
            or summary.get("development_optimization_read") is not False
            or summary.get("validation_content_read") is not True
            or summary.get("validation_optimization_read") is not False
            or summary.get("heldout_content_read") is not False
            or summary.get("test_quality_read") is not False
            or summary.get("project_git_head") != protocol_binding["project_git_head"]
        ):
            raise M2ValidationAggregationError(f"{scene} terminal contract drift")
        scene_config_binding = summary.get("input_scene_config", {})
        snapshot = Path(scene_config_binding.get("snapshot_path", ""))
        _verify_binding(snapshot, scene_config_binding.get("sha256", ""), f"{scene} input config")
        scene_config = _yaml(snapshot)
        if scene_config.get("source_config", {}).get("sha256") != protocol_binding["sha256"]:
            raise M2ValidationAggregationError(f"{scene} validation protocol binding drift")
        if summary["status"] == "done":
            if (
                summary.get("phase") != "formal_validation"
                or summary.get("project_git_dirty") is not False
                or summary.get("checkpoint_immutable") is not True
                or summary.get("checkpoint_sha256_before")
                != summary.get("checkpoint_sha256_after")
                or summary.get("blocked_requests_measured") is not True
            ):
                raise M2ValidationAggregationError(f"{scene} formal renderer contract drift")
            if summary.get("frozen_router") != frozen_router:
                raise M2ValidationAggregationError(f"{scene} frozen router drift")
            for request in list(summary.get("requests", [])) + list(
                summary.get("blocked_requests", [])
            ):
                _validate_matched_rows(
                    request,
                    expected_arms,
                    blocked=request in summary.get("blocked_requests", []),
                )
                for candidate in request.get("candidates", []):
                    asset = candidate["candidate"]["gaussians"]
                    asset_path = Path(asset["path"])
                    _verify_binding(asset_path, asset["sha256"], f"{candidate['arm']} asset")
                    if asset_path.stat().st_size != int(asset["bytes"]):
                        raise M2ValidationAggregationError("candidate asset byte drift")
        elif (
            summary["status"] != "abstain"
            or summary.get("retained_in_denominator") is not True
            or not str(summary.get("reason", "")).startswith("ABSTAIN_")
            or summary.get("request_count") != 0
            or summary.get("total_request_count") != 0
        ):
            raise M2ValidationAggregationError(f"{scene} is neither done nor retained abstain")
        summaries.append(summary)
    requests = collect_validation_requests(summaries, expected_arms)
    return protocol, freeze, summaries, requests


def aggregate(config_path: Path, run_dir: Path) -> dict[str, Any]:
    if _git_dirty():
        raise M2ValidationAggregationError("formal validation aggregation requires clean git")
    config = _yaml(config_path)
    reporting = config.get("reporting", {})
    if (
        config.get("schema_version") != "worldsim_v4_m2_validation_selection_v1"
        or config.get("task_id") != TASK_ID
        or config.get("partition") != "validation"
        or config.get("dataset") != "nuScenes"
        or reporting.get("retain_all_six_scenes_in_denominator") is not True
        or reporting.get("retain_role_abstain_requests_in_denominator") is not True
        or reporting.get("development_content_read") is not False
        or reporting.get("development_optimization_read") is not False
        or reporting.get("validation_content_read") is not True
        or reporting.get("validation_optimization_read") is not False
        or reporting.get("heldout_content_read") is not False
        or reporting.get("test_quality_read") is not False
    ):
        raise M2ValidationAggregationError("validation selection root contract drift")

    protocol, freeze, summaries, requests = _load_verified_inputs(config)
    frozen_router = freeze["frozen_router"]
    decisions = []
    for request in requests:
        decision = _route_request(
            request,
            weights=frozen_router["weights"],
            threshold=float(frozen_router["threshold"]),
            tie_priority=frozen_router["tie_priority"],
        )
        decision["scene"] = request["scene"]
        decisions.append(decision)
    selective = _selection_statistics(decisions)
    arms = list(
        _yaml(Path(protocol["candidate_protocol"]["path"]))["ablations"][
            "matched_repair_arms"
        ]
    )
    table = _matched_table(
        requests,
        arms=arms,
        router_decisions=decisions,
        cohort_scene_count=len(summaries),
    )
    router = next(row for row in table if row["arm"] == "RISK_ROUTER")
    baseline = select_frozen_baseline(table, freeze)
    gate_report = evaluate_acceptance_gates(
        router=router,
        baseline=baseline,
        selective=selective,
        gates=protocol["acceptance_gates"],
    )
    curve = _selective_curve(decisions, reporting["requested_selective_coverages"])

    run_dir.mkdir(parents=True)
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    shutil.copy2(config_path, source_snapshot / config_path.name)
    shutil.copy2(Path(config["validation_protocol"]["path"]), source_snapshot / "m2_validation_v1.yaml")
    shutil.copy2(Path(__file__), source_snapshot / Path(__file__).name)
    _write_json(run_dir / "artifacts/router_decisions.json", decisions)
    _write_json(run_dir / "artifacts/matched_repair_table.json", table)
    _write_json(run_dir / "artifacts/selective_risk_curve.json", curve)
    _write_json(run_dir / "artifacts/gate_report.json", gate_report)

    passed = bool(gate_report["all_gates_passed"])
    summary = {
        "schema_version": "worldsim_v4_m2_validation_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "task_status": "done" if passed else "rejected",
        "phase": "six_scene_validation_confirmation",
        "partition": "validation",
        "dataset": "nuScenes",
        "scene_count": len(summaries),
        "evaluable_scene_count": sum(row["status"] == "done" for row in summaries),
        "retained_abstain_scene_count": sum(row["status"] == "abstain" for row in summaries),
        "request_count": len(requests),
        "asset_blocked_request_count": sum(
            len(row.get("blocked_requests", [])) for row in summaries
        ),
        "candidate_count": sum(len(row.get("candidates", [])) for row in requests),
        "frozen_router": frozen_router,
        "selection_statistics": selective,
        "frozen_matched_non_router": baseline["arm"],
        "validation_gate": gate_report,
        "validation_gate_passed": passed,
        "m3_authorized": passed,
        "validation_protocol": config["validation_protocol"],
        "development_freeze": config["development_freeze"],
        "scene_summary_bindings": config["scene_runs"],
        "project_git_head": _git_head(),
        "project_git_dirty": _git_dirty(),
        "development_content_read": False,
        "development_optimization_read": False,
        "validation_content_read": True,
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    inventory = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "worldsim_v4_m2_validation_manifest_v1",
        "task_id": TASK_ID,
        "inventory": inventory,
        "development_optimization_read": False,
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "task_status": summary["task_status"],
            "validation_gate_passed": passed,
            "m3_authorized": passed,
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "test_quality_read": False,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    try:
        summary = aggregate(args.config.resolve(), args.run_dir.resolve())
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "task_status": summary["task_status"],
                    "scene_count": summary["scene_count"],
                    "request_count": summary["request_count"],
                    "frozen_non_router": summary["frozen_matched_non_router"],
                    "validation_gate_passed": summary["validation_gate_passed"],
                    "m3_authorized": summary["m3_authorized"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.run_dir / "status.json",
            {
                "task_id": TASK_ID,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "heldout_content_read": False,
                "test_quality_read": False,
            },
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the frozen M1/M2/M3 chain and exact-once 18-scene test closeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from motion_proj.worldsim_v4.test_freeze import (
    TASK_ID,
    load_mapping,
    sha256_file,
    validate_execution_plan,
)
from scripts.run_worldsim_v4_m3_scene import atomic_json


class V4CloseoutError(RuntimeError):
    pass


def git(project_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(project_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise V4CloseoutError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def verify_run(run: Path, label: str) -> dict[str, Any]:
    status_path = run / "status.json"
    summary_path = run / "summary.json"
    manifest_path = run / "manifest.json"
    for path in (status_path, summary_path, manifest_path):
        if not path.is_file():
            raise V4CloseoutError(f"{label} evidence missing: {path}")
    status = load_mapping(status_path)
    summary = load_mapping(summary_path)
    if (
        status.get("status") != "done"
        or status.get("summary_sha256") != sha256_file(summary_path)
        or status.get("manifest_sha256") != sha256_file(manifest_path)
        or summary.get("status") != "done"
    ):
        raise V4CloseoutError(f"{label} status/summary/manifest drift")
    return summary


def verify_freeze_history(
    project_root: Path, freeze_path: Path, freeze: dict[str, Any]
) -> dict[str, str]:
    if git(project_root, "status", "--porcelain"):
        raise V4CloseoutError("closeout requires a clean worktree")
    relative = freeze_path.resolve().relative_to(project_root.resolve()).as_posix()
    additions = git(
        project_root,
        "log",
        "--format=%H",
        "--diff-filter=A",
        "--",
        relative,
    ).splitlines()
    if len(additions) != 1:
        raise V4CloseoutError("V4 test freeze must be added by exactly one commit")
    freeze_commit = additions[0]
    parent = git(project_root, "rev-parse", f"{freeze_commit}^")
    if parent != freeze.get("source_commit"):
        raise V4CloseoutError("V4 test freeze parent/source commit drift")
    changed = git(
        project_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        freeze_commit,
    ).splitlines()
    if changed != [relative]:
        raise V4CloseoutError("V4 test freeze commit is not freeze-only")
    committed = subprocess.run(
        ["git", "-C", str(project_root), "show", f"{freeze_commit}:{relative}"],
        capture_output=True,
        check=False,
    )
    if committed.returncode or committed.stdout != freeze_path.read_bytes():
        raise V4CloseoutError("V4 test freeze bytes differ from committed blob")
    head = git(project_root, "rev-parse", "HEAD")
    if subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", freeze_commit, head],
        check=False,
    ).returncode:
        raise V4CloseoutError("current HEAD is not a descendant of test freeze")
    return {"source_commit": parent, "freeze_commit": freeze_commit, "head": head}


def manifest(run_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "status.json"}:
            files[path.relative_to(run_dir).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": "worldsim_v4_m1_m2_m3_test_closeout_manifest_v1",
        "task_id": TASK_ID,
        "status": "done",
        "files": files,
        "test_quality_read": True,
        "test_source_content_reread": False,
    }


def audit(
    *,
    project_root: Path,
    m1_validation_run: Path,
    m1_rejection_run: Path,
    m2_validation_run: Path,
    m3_validation_run: Path,
    freeze_path: Path,
    test_aggregate_run: Path,
    research_status_path: Path,
    failures_path: Path,
    experiments_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    m1 = verify_run(m1_validation_run, "M1 validation")
    m1_rejection = verify_run(m1_rejection_run, "M1 rejection")
    m2 = verify_run(m2_validation_run, "M2 validation")
    m3 = verify_run(m3_validation_run, "M3 validation")
    test = verify_run(test_aggregate_run, "M3 test aggregate")
    if (
        m1.get("task_id") != "WS-V4-M1-EVIDENCE-FIELD-01"
        or m1.get("confirmation_gate", {}).get("status") != "reject"
        or m1.get("confirmation_gate", {}).get("directional_support_scene_count") != 0
        or m1.get("confirmation_gate", {}).get("required_scene_count") != 6
        or m1.get("cohort_accounting", {}).get("evaluable_scene_count") != 3
        or m1.get("cohort_accounting", {}).get("abstain_scene_count") != 3
        or m1.get("validation_optimization_read") is not False
        or m1.get("test_quality_read") is not False
        or m1_rejection.get("task_status") != "rejected"
        or m1_rejection.get("m1_feature_expansion_authorized") is not False
        or m1_rejection.get("m2_fallback_authorized") is not True
        or m1_rejection.get("test_quality_read") is not False
    ):
        raise V4CloseoutError("M1 rejection freeze contract drift")
    hole_endpoints = m2.get("validation_gate", {}).get("checks", {}).get(
        "hole_any_endpoint", {}
    ).get("endpoints", [])
    geometry = [row for row in hole_endpoints if row.get("metric") == "hole_geometry_mae_m"]
    if (
        m2.get("task_id") != "WS-V4-M2-REPAIR-ROUTER-01"
        or m2.get("task_status") != "done"
        or m2.get("validation_gate_passed") is not True
        or m2.get("m3_authorized") is not True
        or m2.get("scene_count") != 6
        or m2.get("evaluable_scene_count") != 3
        or m2.get("retained_abstain_scene_count") != 3
        or m2.get("request_count") != 154
        or m2.get("selection_statistics", {}).get("accepted_count") != 83
        or m2.get("selection_statistics", {}).get("abstain_count") != 71
        or m2.get("frozen_router", {}).get("weight_name") != "uncertainty_forward"
        or m2.get("frozen_router", {}).get("threshold") != 1.0
        or m2.get("frozen_matched_non_router") != "TELEA"
        or m2.get("validation_optimization_read") is not False
        or m2.get("test_quality_read") is not False
        or len(geometry) != 1
        or geometry[0].get("signed_improvement", 0.0) >= 0.0
        or geometry[0].get("strictly_improved") is not False
    ):
        raise V4CloseoutError("M2 validation/risk-router tradeoff contract drift")
    if (
        m3.get("task_id") != TASK_ID
        or m3.get("partition") != "validation"
        or m3.get("scene_denominator") != 6
        or m3.get("evaluable_scene_count") != 3
        or m3.get("abstain_scene_count") != 3
        or m3.get("validation_gate_passed") is not True
        or m3.get("test_freeze_authorized") is not True
        or m3.get("validation_optimization_read") is not False
        or m3.get("test_quality_read") is not False
        or not all(row.get("passed") for row in m3.get("gates", {}).values())
    ):
        raise V4CloseoutError("M3 validation freeze contract drift")
    freeze = load_mapping(freeze_path)
    history = verify_freeze_history(project_root, freeze_path, freeze)
    plan = validate_execution_plan(freeze)
    if freeze.get("test_authorized") is not True or freeze.get("test_read_count") != 1:
        raise V4CloseoutError("test freeze authorization/read-count drift")
    if (
        test.get("task_id") != TASK_ID
        or test.get("partition") != "test"
        or test.get("scene_denominator") != 18
        or test.get("evaluable_scene_count", 0) + test.get("abstain_scene_count", 0) != 18
        or test.get("exact_once_attempt_count") != 18
        or test.get("exact_once_completion_count") != 18
        or test.get("freeze_commit") != history["freeze_commit"]
        or test.get("freeze_sha256") != sha256_file(freeze_path)
        or test.get("test_quality_read") is not True
        or test.get("test_source_content_reread_by_aggregator") is not False
        or test.get("parameter_search") is not False
        or test.get("threshold_search") is not False
    ):
        raise V4CloseoutError("18-scene exact-once test aggregate contract drift")
    ledger = Path(freeze["ledger_dir"])
    terminal = load_mapping(ledger / "terminal.json")
    attempts = sorted((ledger / "attempts").glob("*.json"))
    completions = sorted((ledger / "completions").glob("*.json"))
    if (
        terminal.get("state") != "done"
        or terminal.get("attempt_count") != 18
        or terminal.get("completion_count") != 18
        or terminal.get("freeze_commit") != history["freeze_commit"]
        or len(attempts) != 18
        or len(completions) != 18
    ):
        raise V4CloseoutError("18-scene exact-once ledger drift")
    for row in plan:
        completion_path = ledger / "completions" / f"{row['attempt_id']}.json"
        completion = load_mapping(completion_path)
        if (
            completion.get("state") != "completed"
            or completion.get("scene") != row["scene"]
            or completion.get("attempt_id") != row["attempt_id"]
            or terminal.get("completion_sha256", {}).get(row["scene"])
            != sha256_file(completion_path)
        ):
            raise V4CloseoutError(f"test completion ledger drift: {row['scene']}")
    docs = {
        "research_status": research_status_path.read_text(encoding="utf-8"),
        "failures": failures_path.read_text(encoding="utf-8"),
        "experiments": experiments_path.read_text(encoding="utf-8"),
    }
    aggregate_name = test_aggregate_run.name
    if (
        aggregate_name not in docs["research_status"]
        or aggregate_name not in docs["experiments"]
        or str(test["conclusion"]) not in docs["research_status"]
        or "3.3908096237" not in docs["failures"]
        or "exact-once" not in docs["failures"]
    ):
        raise V4CloseoutError("M1/M2/M3/test documentation closeout drift")

    run_dir.mkdir(parents=True)
    snapshots = run_dir / "source_snapshot"
    snapshots.mkdir()
    for source in (
        Path(__file__),
        freeze_path,
        research_status_path,
        failures_path,
        experiments_path,
        m1_validation_run / "summary.json",
        m1_rejection_run / "summary.json",
        m2_validation_run / "summary.json",
        m3_validation_run / "summary.json",
        test_aggregate_run / "summary.json",
        ledger / "terminal.json",
    ):
        target = snapshots / source.name
        if target.exists():
            target = snapshots / f"{source.parent.name}__{source.name}"
        shutil.copy2(source, target)
    summary = {
        "schema_version": "worldsim_v4_m1_m2_m3_test_closeout_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "objective_complete": True,
        "m1_task_status": "rejected",
        "m1_directional_support": "0/6",
        "m2_task_status": "done",
        "m2_validation_gate_passed": True,
        "m2_hole_geometry_mae_signed_improvement_m": geometry[0]["signed_improvement"],
        "m3_validation_gate_passed": True,
        "test_scene_denominator": 18,
        "test_evaluable_scene_count": test["evaluable_scene_count"],
        "test_abstain_scene_count": test["abstain_scene_count"],
        "test_conclusion": test["conclusion"],
        "test_gate_passed": test["test_gate_passed"],
        "exact_once_attempt_count": 18,
        "exact_once_completion_count": 18,
        "freeze_source_commit": history["source_commit"],
        "freeze_commit": history["freeze_commit"],
        "closeout_head": history["head"],
        "test_quality_read": True,
        "test_source_content_reread_by_closeout": False,
        "duration_seconds": time.monotonic() - started,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "fingerprint.json",
        {
            "project_head": history["head"],
            "freeze_sha256": sha256_file(freeze_path),
            "test_aggregate_summary_sha256": sha256_file(test_aggregate_run / "summary.json"),
            "ledger_terminal_sha256": sha256_file(ledger / "terminal.json"),
            "test_source_content_reread": False,
        },
    )
    atomic_json(run_dir / "manifest.json", manifest(run_dir))
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
            "objective_complete": True,
            "test_quality_read": True,
            "test_source_content_reread": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--m1-validation-run", type=Path, required=True)
    parser.add_argument("--m1-rejection-run", type=Path, required=True)
    parser.add_argument("--m2-validation-run", type=Path, required=True)
    parser.add_argument("--m3-validation-run", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--test-aggregate-run", type=Path, required=True)
    parser.add_argument("--research-status", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        project_root=args.project_root.resolve(),
        m1_validation_run=args.m1_validation_run.resolve(),
        m1_rejection_run=args.m1_rejection_run.resolve(),
        m2_validation_run=args.m2_validation_run.resolve(),
        m3_validation_run=args.m3_validation_run.resolve(),
        freeze_path=args.freeze.resolve(),
        test_aggregate_run=args.test_aggregate_run.resolve(),
        research_status_path=args.research_status.resolve(),
        failures_path=args.failures.resolve(),
        experiments_path=args.experiments.resolve(),
        run_dir=args.run_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

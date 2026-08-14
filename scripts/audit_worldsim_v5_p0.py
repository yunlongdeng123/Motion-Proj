#!/usr/bin/env python3
"""Fail-closed closeout audit for WorldSim V5 P0 scope and forensic freezes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

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
    verify_named_bindings,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-P0-SCOPE-FREEZE-01"
EXPECTED_REQUIREMENTS = {
    "all_output_paths_exist",
    "p0_contract_tests_pass",
    "m1_missing_evidence_collection_contract_frozen",
    "m2_reference_and_pre_post_geometry_contract_frozen",
    "m1_formal_forensic_run_done",
    "m2_formal_forensic_run_done",
    "plan_and_resolved_config_sha256_recorded",
    "freeze_only_commit",
}


def validate_freeze_file_set(actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ForensicAuditError(
            f"P0 freeze commit file set drift: missing={missing} extra={extra}"
        )


def _manifest_has(manifest: Mapping[str, Any], path: str, sha256: str) -> bool:
    return any(
        row.get("path") == path and row.get("sha256") == sha256
        for row in manifest.get("inventory", [])
    )


def audit_registered_forensic(
    name: str, registration: Mapping[str, Any]
) -> dict[str, Any]:
    verified = verify_named_bindings(registration["run"], registration["files"])
    status = load_json_mapping(Path(verified["status.json"]["path"]))
    summary = load_json_mapping(Path(verified["summary.json"]["path"]))
    manifest = load_json_mapping(Path(verified["manifest.json"]["path"]))
    fingerprint = load_json_mapping(Path(verified["fingerprint.json"]["path"]))
    for payload_name, payload in (("status", status), ("summary", summary)):
        if payload.get("task_id") != registration["task_id"]:
            raise ForensicAuditError(f"{name} {payload_name} task_id drift")
        if payload.get("status") != "done" or payload.get("task_status") != "done":
            raise ForensicAuditError(f"{name} {payload_name} not terminal done")
        if payload.get("conclusion") != registration["conclusion"]:
            raise ForensicAuditError(f"{name} {payload_name} conclusion drift")
        for key in ("fresh_quality_read", "test_quality_read", "parameter_search_performed"):
            if payload.get(key) is not False:
                raise ForensicAuditError(f"{name} provenance violated: {key}")
    if status.get("project_git_head") != registration["project_git_head"]:
        raise ForensicAuditError(f"{name} formal source commit drift")
    if status.get("summary_sha256") != verified["summary.json"]["sha256"]:
        raise ForensicAuditError(f"{name} terminal summary SHA drift")
    if status.get("manifest_sha256") != verified["manifest.json"]["sha256"]:
        raise ForensicAuditError(f"{name} terminal manifest SHA drift")
    if fingerprint.get("project_git_head") != registration["project_git_head"]:
        raise ForensicAuditError(f"{name} fingerprint source commit drift")
    if fingerprint.get("resolved_config", {}).get("sha256") != status.get(
        "resolved_config_sha256"
    ):
        raise ForensicAuditError(f"{name} resolved config SHA drift")
    if not _manifest_has(manifest, "summary.json", verified["summary.json"]["sha256"]):
        raise ForensicAuditError(f"{name} manifest omits frozen summary")
    contract_key = {
        "m1": "missing_evidence_collection_contract_frozen",
        "m2": "reference_and_pre_post_geometry_contract_frozen",
    }[name]
    if summary.get(contract_key) is not True:
        raise ForensicAuditError(f"{name} collection contract is not frozen")
    return {"registration": dict(registration), "verified_files": verified}


def audit(scope: Mapping[str, Any], project: Path, current_head: str) -> dict[str, Any]:
    if scope.get("schema_version") != "worldsim_v5_p0_scope_v1":
        raise ForensicAuditError("P0 scope schema drift")
    if scope.get("task_id") != TASK_ID or scope.get("status") != "running":
        raise ForensicAuditError("P0 scope is not registered running")
    project_cfg = scope["project"]
    freeze_commit = str(project_cfg["p0_freeze_commit"])
    if not freeze_commit or project_cfg.get("resolved_plan_commit") != freeze_commit:
        raise ForensicAuditError("P0 freeze/resolved-plan commit missing")
    ancestor = subprocess.run(
        ["git", "-C", str(project), "merge-base", "--is-ancestor", freeze_commit, current_head],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ForensicAuditError("P0 freeze commit is not an ancestor of audit commit")
    actual_files = subprocess.check_output(
        [
            "git",
            "-C",
            str(project),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            freeze_commit,
        ],
        text=True,
    ).splitlines()
    validate_freeze_file_set(actual_files, list(project_cfg["freeze_commit_files"]))

    plan = project / str(project_cfg["plan_path"])
    plan_sha = sha256_file(plan)
    if plan_sha != project_cfg.get("resolved_plan_sha256"):
        raise ForensicAuditError(
            f"P0 resolved plan SHA drift: expected={project_cfg.get('resolved_plan_sha256')} actual={plan_sha}"
        )
    output_records = {}
    for name, relative in scope["outputs"].items():
        path = project / str(relative)
        if not path.is_file():
            raise ForensicAuditError(f"P0 registered output missing: {name}={path}")
        output_records[name] = {
            "path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    gate = scope["closeout_gate"]
    if gate.get("status") != "running" or set(gate.get("requires", [])) != EXPECTED_REQUIREMENTS:
        raise ForensicAuditError("P0 closeout requirements drift")
    for key in (
        "training_started",
        "fresh_quality_read",
        "kitti_quality_read",
        "parameter_search_performed",
    ):
        if gate.get(key) is not False:
            raise ForensicAuditError(f"P0 forbidden action observed: {key}")
    if scope["task_registry"].get("WS-V5-M1-D0-BAYES-FORENSICS-01") != "done":
        raise ForensicAuditError("M1 D0 not registered done")
    if scope["task_registry"].get("WS-V5-M2-D0-GEOMETRY-FORENSICS-01") != "done":
        raise ForensicAuditError("M2 D0 not registered done")
    forensic = {
        name: audit_registered_forensic(name, registration)
        for name, registration in scope["forensic_results"].items()
    }
    return {
        "schema_version": "worldsim_v5_p0_checklist_v1",
        "freeze_commit": freeze_commit,
        "audit_commit": current_head,
        "freeze_commit_files": actual_files,
        "resolved_plan": {"path": str(project_cfg["plan_path"]), "sha256": plan_sha},
        "registered_outputs": output_records,
        "forensic_results": forensic,
        "requirements": {name: True for name in sorted(EXPECTED_REQUIREMENTS)},
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
    }


def run_contract_tests(project: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_worldsim_v5_forensics.py",
        "tests/test_worldsim_v5_p0_contract.py",
        "tests/test_worldsim_v5_p0_audit.py",
        "tests/test_audit_worldsim_v5_kitti_archives.py",
    ]
    result = subprocess.run(command, cwd=project, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ForensicAuditError(
            f"P0 contract tests failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    project_head = prepare_formal_run(args.run_dir, TASK_ID, project)
    scope = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved_config = write_resolved_config(args.run_dir, scope)
    source_inventory = copy_source_snapshot(
        args.run_dir,
        [
            Path(__file__),
            Path(__file__).with_name("worldsim_v5_forensics_common.py"),
            args.config,
            project / scope["project"]["plan_path"],
            project / "docs/WS_V5_M1_FAILURE_FORENSICS.md",
            project / "docs/WS_V5_M2_GEOMETRY_FORENSICS.md",
        ],
        project,
    )
    started = utc_now()
    checklist = audit(scope, project, project_head)
    tests = run_contract_tests(project)
    checklist["contract_tests"] = tests
    checklist["requirements"]["p0_contract_tests_pass"] = True
    atomic_json(args.run_dir / "artifacts/p0_checklist.json", checklist)
    summary = {
        "schema_version": "worldsim_v5_p0_audit_summary_v1",
        "task_id": TASK_ID,
        "task_status": "done",
        "status": "done",
        "conclusion": "p0_scope_and_forensic_contracts_closed",
        "project_git_head": project_head,
        "freeze_commit": checklist["freeze_commit"],
        "resolved_plan": checklist["resolved_plan"],
        "p0_checklist_path": "artifacts/p0_checklist.json",
        "contract_test_exit_code": tests["exit_code"],
        "all_closeout_requirements_passed": all(checklist["requirements"].values()),
        "next_authorization": [
            "result_blind_nuscenes_fresh_cohort_freeze",
            "m1_evidence_schema_instrumentation",
            "m2_reference_and_staged_geometry_instrumentation",
        ],
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "training_performed": False,
        "source_snapshot": source_inventory,
        "finished_at_utc": utc_now(),
    }
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
        input_bindings=checklist["forensic_results"],
        summary=summary,
        resolved_config_record=resolved_config,
        events_record=events,
    )
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed audit for the frozen WorldSim V4 M1 validation decision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V4-M1-EVIDENCE-FIELD-01"
RUN_ROOT = Path(f"/root/autodl-tmp/runs/worldsim_v4/{TASK_ID}")
TERMINAL_FILES = ("status.json", "summary.json", "metrics.json", "manifest.json")


class M1ValidationAuditError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M1ValidationAuditError(f"JSON root is not a mapping: {path}")
    return payload


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_files(registration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    run_dir = Path(str(registration["run"])).resolve()
    bindings = registration.get("files", {})
    if set(bindings) != set(TERMINAL_FILES):
        raise M1ValidationAuditError("validation terminal file registry drift")
    verified = {}
    for name in TERMINAL_FILES:
        path = run_dir / name
        expected = str(bindings[name]["sha256"])
        if not path.is_file():
            raise M1ValidationAuditError(f"registered validation file missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise M1ValidationAuditError(
                f"registered validation SHA drift: {name} expected={expected} actual={actual}"
            )
        verified[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    return verified


def audit(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v4_m1_evidence_v1":
        raise M1ValidationAuditError("M1 config schema drift")
    if config.get("status") != "rejected":
        raise M1ValidationAuditError("M1 config is not frozen rejected")
    registration = config.get("validation_result")
    if not isinstance(registration, Mapping):
        raise M1ValidationAuditError("validation result registration missing")
    if registration.get("status") != "done" or registration.get("gate_status") != "reject":
        raise M1ValidationAuditError("validation result is not terminal/reject")
    verified = _verify_files(registration)
    status = load_json(Path(verified["status.json"]["path"]))
    summary = load_json(Path(verified["summary.json"]["path"]))
    metrics = load_json(Path(verified["metrics.json"]["path"]))
    manifest = load_json(Path(verified["manifest.json"]["path"]))
    if (
        status.get("status") != "done"
        or status.get("confirmation_status") != "reject"
        or status.get("phase") != "six_scene_validation_confirmation"
        or status.get("validation_content_read") is not True
        or status.get("validation_optimization_read") is not False
        or status.get("heldout_content_read") is not False
        or status.get("test_quality_read") is not False
    ):
        raise M1ValidationAuditError("validation status provenance drift")
    if status.get("summary_sha256") != verified["summary.json"]["sha256"]:
        raise M1ValidationAuditError("validation summary terminal SHA drift")
    if status.get("manifest_sha256") != verified["manifest.json"]["sha256"]:
        raise M1ValidationAuditError("validation manifest terminal SHA drift")
    scenes = list(config["protocol"]["validation_scenes"])
    if status.get("scenes") != scenes or summary.get("scenes") != scenes:
        raise M1ValidationAuditError("validation scene order/set drift")
    if summary.get("evaluable_scenes") != registration.get("evaluable_scenes"):
        raise M1ValidationAuditError("validation evaluable scenes drift")
    if summary.get("abstain_scenes") != registration.get("abstain_scenes"):
        raise M1ValidationAuditError("validation abstain scenes drift")
    accounting = summary.get("cohort_accounting", {})
    expected_accounting = {
        "required_scene_count": len(scenes),
        "evaluable_scene_count": len(registration["evaluable_scenes"]),
        "abstain_scene_count": len(registration["abstain_scenes"]),
        "coverage_denominator": "all_required_scenes",
        "quality_metric_denominator": "evaluable_scenes_only",
    }
    if any(accounting.get(key) != value for key, value in expected_accounting.items()):
        raise M1ValidationAuditError("validation cohort accounting drift")
    gate = summary.get("confirmation_gate", {})
    if gate != metrics.get("confirmation_gate") or gate.get("status") != "reject":
        raise M1ValidationAuditError("validation rejection gate drift")
    if gate.get("required_directional_support_scene_count") != len(scenes) // 2 + 1:
        raise M1ValidationAuditError("validation strict-majority threshold drift")
    for key in ("arm_search_performed", "calibration_fit_performed", "threshold_search_performed"):
        if gate.get(key) is not False or metrics.get(key) is not False:
            raise M1ValidationAuditError(f"validation optimization was performed: {key}")
    if not all(row.get("exact") is True for row in summary["base_rgb_render_checks"]):
        raise M1ValidationAuditError("validation base RGB immutability failed")
    if not all(row.get("exact") is True for row in summary["checkpoint_checks"]):
        raise M1ValidationAuditError("validation checkpoint immutability failed")
    selection = config["frozen_selection"]
    if any(summary["frozen_selection"].get(key) != selection.get(key) for key in (
        "evidence_arm", "calibration", "mask_threshold", "temporal_retention"
    )):
        raise M1ValidationAuditError("validation frozen selection drift")
    if manifest.get("status") != "done" or manifest.get("task_id") != TASK_ID:
        raise M1ValidationAuditError("validation manifest terminal drift")
    return {
        "schema_version": "worldsim_v4_m1_validation_audit_summary_v1",
        "task_id": TASK_ID,
        "task_status": "rejected",
        "status": "done",
        "phase": "validation_rejection_audit",
        "validation_run": registration["run"],
        "validation_project_git_head": registration["project_git_head"],
        "scenes": scenes,
        "evaluable_scenes": registration["evaluable_scenes"],
        "abstain_scenes": registration["abstain_scenes"],
        "cohort_accounting": accounting,
        "confirmation_gate": gate,
        "verified_files": verified,
        "m2_fallback_authorized": True,
        "m2_fallback_scope": "evidence_routed_delta_compiler",
        "m1_feature_expansion_authorized": False,
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"run directory exists: {args.run_dir}")
    if RUN_ROOT.resolve() not in args.run_dir.resolve().parents:
        raise M1ValidationAuditError(f"audit run must be under {RUN_ROOT}")
    project = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"], text=True
    ).strip():
        raise M1ValidationAuditError("formal validation audit requires a clean project")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = audit(config)
    args.run_dir.mkdir(parents=True)
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v4_m1_validation_audit_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "files": [{
                "path": "summary.json",
                "bytes": (args.run_dir / "summary.json").stat().st_size,
                "sha256": sha256_file(args.run_dir / "summary.json"),
            }],
        },
    )
    status = {
        "schema_version": "worldsim_v4_m1_validation_audit_status_v1",
        "task_id": TASK_ID,
        "task_status": "rejected",
        "status": "done",
        "phase": "validation_rejection_audit",
        "summary_sha256": sha256_file(args.run_dir / "summary.json"),
        "manifest_sha256": sha256_file(args.run_dir / "manifest.json"),
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.run_dir / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

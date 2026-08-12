#!/usr/bin/env python3
"""Fail-closed audit for the WorldSim V4 M1 development freeze candidate."""

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


class M1FreezeAuditError(RuntimeError):
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
        raise M1FreezeAuditError(f"JSON root is not a mapping: {path}")
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
    run_dir = Path(registration["run"])
    verified = {}
    for name, binding in registration["files"].items():
        path = run_dir / name
        if not path.is_file():
            raise M1FreezeAuditError(f"registered file missing: {path}")
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise M1FreezeAuditError(
                f"registered file SHA drift: {name} expected={binding['sha256']} actual={actual}"
            )
        verified[name] = {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return verified


def audit(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v4_m1_evidence_v1":
        raise M1FreezeAuditError("M1 config schema drift")
    if config.get("status") != "development_candidate":
        raise M1FreezeAuditError("M1 config is not a development freeze candidate")
    protocol = config["protocol"]
    registration = config["development_result"]
    selection = config["frozen_selection"]
    if registration.get("status") != "done" or registration.get("gate_status") != "pass":
        raise M1FreezeAuditError("development result is not terminal/pass")
    verified = _verify_files(registration)
    status = load_json(Path(verified["status.json"]["path"]))
    summary = load_json(Path(verified["summary.json"]["path"]))
    metrics = load_json(Path(verified["metrics.json"]["path"]))
    calibration = load_json(Path(verified["calibration.json"]["path"]))
    if (
        status.get("status") != "done"
        or status.get("phase") != "six_scene_development"
        or status.get("development_content_read") is not True
        or status.get("heldout_content_read") is not False
        or status.get("test_quality_read") is not False
    ):
        raise M1FreezeAuditError("development status provenance drift")
    if summary.get("scenes") != protocol["development_scenes"]:
        raise M1FreezeAuditError("development scene order/set drift")
    for key in ("evaluable_scenes", "abstain_scenes"):
        if summary.get(key) != registration.get(key):
            raise M1FreezeAuditError(f"development {key} drift")
    accounting = summary.get("cohort_accounting", {})
    for key in ("required_scene_count", "evaluable_scene_count", "abstain_scene_count"):
        if accounting.get(key) != registration.get(key):
            raise M1FreezeAuditError(f"development cohort accounting drift: {key}")
    if (
        accounting.get("coverage_denominator") != "all_required_scenes"
        or accounting.get("quality_metric_denominator") != "evaluable_scenes_only"
    ):
        raise M1FreezeAuditError("development denominator contract drift")
    reasons = {
        row["scene"]: row.get("reason")
        for row in summary.get("scene_records", [])
        if row.get("status") == "abstain"
    }
    expected_reasons = {
        **{
            scene: specification["reason"]
            for scene, specification in protocol.get(
                "development_abstentions", {}
            ).items()
        },
        **{
            scene: "ABSTAIN_NO_ACTOR"
            for scene in protocol["abstain_no_actor_scenes"]
        },
    }
    if reasons != expected_reasons:
        raise M1FreezeAuditError("development abstention reason drift")
    if metrics.get("gate_preview", {}).get("status") != "pass":
        raise M1FreezeAuditError("development gate did not pass")
    if not all(
        metrics["gate_preview"]["checks"].get(name) is True
        for name in (
            "boundary_f1",
            "false_negative_semantic_mass",
            "calibration",
            "base_rgb_exact",
        )
    ):
        raise M1FreezeAuditError("development gate checks are incomplete")
    if not all(row.get("exact") is True for row in summary["base_rgb_render_checks"]):
        raise M1FreezeAuditError("base RGB immutability failed")
    if not all(row.get("exact") is True for row in summary["checkpoint_checks"]):
        raise M1FreezeAuditError("checkpoint immutability failed")
    selected = {
        "evidence_arm": summary["selected_evidence_arm"],
        "calibration": summary["selected_calibration"],
        "mask_threshold": summary["selected_mask_threshold"],
        "temporal_retention": float(config["evidence"]["temporal"]["retention"]),
    }
    for key, value in selected.items():
        if selection.get(key) != value:
            raise M1FreezeAuditError(f"frozen selection drift: {key}")
    if calibration.get("selected") != selection["calibration"]:
        raise M1FreezeAuditError("calibration artifact selection drift")
    return {
        "schema_version": "worldsim_v4_m1_freeze_audit_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": "development_freeze_audit",
        "development_run": registration["run"],
        "development_project_git_head": registration["project_git_head"],
        "scenes": registration["scenes"],
        "evaluable_scenes": registration["evaluable_scenes"],
        "abstain_scenes": registration["abstain_scenes"],
        "cohort_accounting": accounting,
        "selected": selected,
        "gate": metrics["gate_preview"],
        "verified_files": verified,
        "development_content_read": False,
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
        raise M1FreezeAuditError(f"audit run must be under {RUN_ROOT}")
    project = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"], text=True
    ).strip():
        raise M1FreezeAuditError("formal freeze audit requires a clean project")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = audit(config)
    args.run_dir.mkdir(parents=True)
    atomic_json(args.run_dir / "summary.json", summary)
    manifest = {
        "schema_version": "worldsim_v4_m1_freeze_audit_manifest_v1",
        "task_id": TASK_ID,
        "status": "done",
        "files": [
            {
                "path": "summary.json",
                "bytes": (args.run_dir / "summary.json").stat().st_size,
                "sha256": sha256_file(args.run_dir / "summary.json"),
            }
        ],
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    status = {
        "schema_version": "worldsim_v4_m1_freeze_audit_status_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": "development_freeze_audit",
        "summary_sha256": sha256_file(args.run_dir / "summary.json"),
        "manifest_sha256": sha256_file(args.run_dir / "manifest.json"),
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.run_dir / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

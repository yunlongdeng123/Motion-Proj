#!/usr/bin/env python3
"""机器收口 V5 M1 八场景 StreetGS formal30k base batch。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M1-STRUCTURED-OWNERSHIP-01"
SCHEMA_VERSION = "worldsim_v5_m1_formal_batch_audit_v1"


class FormalBatchAuditError(RuntimeError):
    """formal batch denominator、identity 或 payload 漂移。"""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FormalBatchAuditError(f"formal artifact 缺失: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalBatchAuditError(f"formal artifact 不是 object: {path}")
    return payload


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise FormalBatchAuditError("formal batch audit config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "development_base_reconstruction_audit"
    ):
        raise FormalBatchAuditError("formal batch audit task/phase/status 漂移")
    rows = payload.get("runs", [])
    if len(rows) != 8 or len({row["scene"] for row in rows}) != 8:
        raise FormalBatchAuditError("formal batch 必须冻结 8 个不同 scene")
    return payload


def audit_run(
    row: dict[str, Any], expected_source_commit: str, expected_config_sha256: str
) -> dict[str, Any]:
    run_dir = Path(row["path"]).resolve()
    if run_dir.name != row["run_id"] or not run_dir.is_dir():
        raise FormalBatchAuditError(f"formal run identity 漂移: {run_dir}")
    summary_path = run_dir / "summary.json"
    status_path = run_dir / "status.json"
    fingerprint_path = run_dir / "fingerprint.json"
    manifest_path = run_dir / "manifest.json"
    summary = _load_json(summary_path)
    status = _load_json(status_path)
    fingerprint = _load_json(fingerprint_path)
    manifest = _load_json(manifest_path)
    for name, payload in (
        ("summary", summary),
        ("status", status),
        ("manifest", manifest),
    ):
        if payload.get("task_id") != TASK_ID or payload.get("status") != "done":
            raise FormalBatchAuditError(f"{row['scene']} {name} terminal 漂移")
    if (
        summary.get("scene") != row["scene"]
        or int(summary.get("scene_index", -1)) != int(row["scene_index"])
        or summary.get("mode") != "formal"
        or summary.get("phase") != "development_base_reconstruction"
        or int(summary.get("iterations", -1)) != 30000
    ):
        raise FormalBatchAuditError(f"{row['scene']} summary identity/iteration 漂移")
    if (
        summary.get("validation_quality_read") is not False
        or summary.get("test_quality_read") is not False
        or summary.get("model_inference_started") is not False
        or summary.get("project_git", {}).get("dirty") is not False
        or summary.get("project_git", {}).get("head") != expected_source_commit
    ):
        raise FormalBatchAuditError(f"{row['scene']} leakage/source contract 漂移")
    summary_sha256 = sha256_file(summary_path)
    if status.get("summary_sha256") != summary_sha256:
        raise FormalBatchAuditError(f"{row['scene']} status→summary binding 漂移")
    checkpoint = summary.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if (
        not checkpoint_path.is_file()
        or int(checkpoint.get("step", -1)) != 30000
        or checkpoint.get("means_finite") is not True
        or set(checkpoint.get("gaussian_counts", {})) != {"Background", "RigidNodes"}
        or any(int(value) <= 0 for value in checkpoint["gaussian_counts"].values())
        or checkpoint_path.stat().st_size != int(checkpoint.get("bytes", -1))
        or sha256_file(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise FormalBatchAuditError(f"{row['scene']} checkpoint payload 漂移")
    manifest_checkpoint = manifest.get("artifacts", {}).get("work_dirs_checkpoint", {})
    if manifest_checkpoint != checkpoint:
        raise FormalBatchAuditError(f"{row['scene']} manifest checkpoint binding 漂移")
    for relative, record in manifest.get("artifacts", {}).items():
        if relative == "work_dirs_checkpoint":
            continue
        artifact = run_dir / relative
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["bytes"])
            or sha256_file(artifact) != record["sha256"]
        ):
            raise FormalBatchAuditError(
                f"{row['scene']} manifest payload 漂移: {relative}"
            )
    if fingerprint.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise FormalBatchAuditError(f"{row['scene']} fingerprint checkpoint 漂移")
    if (
        fingerprint.get("config_binding", {}).get("overlay_sha256")
        != expected_config_sha256
    ):
        raise FormalBatchAuditError(f"{row['scene']} formal config binding 漂移")
    return {
        "scene": row["scene"],
        "scene_index": int(row["scene_index"]),
        "run": str(run_dir),
        "run_id": run_dir.name,
        "summary_sha256": summary_sha256,
        "status_sha256": sha256_file(status_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint": checkpoint,
        "duration_seconds": float(summary["duration_seconds"]),
        "resources": summary["resources"],
        "validation_quality_read": False,
        "test_quality_read": False,
        "model_inference_started": False,
        "payload_rehashed": True,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events = [{"event": "run_started", "at_utc": utc_now()}]
    write_events(run_dir, events)
    try:
        records = [
            audit_run(
                row,
                str(config["formal_source_commit"]),
                str(config["formal_config_sha256"]),
            )
            for row in config["runs"]
        ]
        summary = {
            "schema_version": "worldsim_v5_m1_formal_batch_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "phase": "development_base_reconstruction_audit",
            "source_commit": source_head,
            "formal_source_commit": config["formal_source_commit"],
            "formal_config_sha256": config["formal_config_sha256"],
            "scene_count": len(records),
            "completed_scene_count": len(records),
            "iteration_count_each": 30000,
            "total_duration_seconds": float(
                sum(record["duration_seconds"] for record in records)
            ),
            "total_checkpoint_bytes": int(
                sum(int(record["checkpoint"]["bytes"]) for record in records)
            ),
            "maximum_peak_gpu_memory_mib": max(
                int(record["resources"]["peak_gpu_memory_mib"]) for record in records
            ),
            "maximum_peak_cgroup_memory_bytes": max(
                int(record["resources"]["peak_cgroup_memory_bytes"])
                for record in records
            ),
            "all_checkpoint_payload_rehashed_exact": True,
            "validation_quality_read": False,
            "test_quality_read": False,
            "model_inference_started": False,
            "parameter_search_performed": False,
            "runs": records,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/audit_worldsim_v5_m1_formal_batch.py",
                PROJECT / "scripts/worldsim_v5_forensics_common.py",
            ],
            PROJECT,
        )
        fingerprint_path = run_dir / "fingerprint.json"
        atomic_json(
            fingerprint_path,
            {
                "schema_version": "worldsim_v5_m1_formal_batch_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_head,
                "source_clean": True,
                "resolved_config": resolved,
                "source_snapshot": snapshot,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        status_path = run_dir / "status.json"
        atomic_json(
            status_path,
            {
                "schema_version": "worldsim_v5_m1_formal_batch_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "source_commit": source_head,
                "summary_sha256": sha256_file(summary_path),
                "finished_at_utc": utc_now(),
            },
        )
        manifest_path = run_dir / "manifest.json"
        atomic_json(
            manifest_path,
            {
                "schema_version": "worldsim_v5_m1_formal_batch_run_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": inventory_files(run_dir, {"manifest.json"}),
            },
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
        atomic_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v5_m1_formal_batch_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "source_commit": source_head,
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": utc_now(),
            },
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

#!/usr/bin/env python3
"""缺失时从官方 Hugging Face 精确恢复 V4 sky-mask 模型快照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/sky_masks_v1.yaml",
    "scripts/restore_worldsim_v4_sky_model.py",
    "tests/test_restore_worldsim_v4_sky_model.py",
)


class SkyModelRestoreError(RuntimeError):
    """sky model 精确恢复失败。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SkyModelRestoreError("config 必须为 mapping")
    restore = value.get("model", {}).get("restore", {})
    if restore.get("endpoint") != "https://huggingface.co" or restore.get("policy") != "official_exact_revision_if_missing":
        raise SkyModelRestoreError("restore endpoint/policy 漂移")
    return value


def audit_required(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["model"]["snapshot"])
    files = {}
    for name, expected in config["model"]["required_files"].items():
        path = root / name
        if not path.is_file():
            raise SkyModelRestoreError(f"required model file 缺失：{path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise SkyModelRestoreError(f"model file 漂移：{name}: {actual} != {expected}")
        files[name] = {"path": str(path), **actual}
    return {"snapshot": str(root), "files": files}


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise SkyModelRestoreError(process.stderr.strip())
    return process.stdout.strip()


def run(config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise SkyModelRestoreError(f"run 目录已存在，禁止复用：{run_dir}")
    config = load_config(config_path)
    model = config["model"]
    target = Path(model["snapshot"])
    present_before = all((target / name).is_file() for name in model["required_files"])
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    os.environ["HF_ENDPOINT"] = model["restore"]["endpoint"]
    resolved = Path(snapshot_download(
        repo_id=model["id"],
        revision=model["revision"],
        cache_dir=model["restore"]["cache_dir"],
        allow_patterns=sorted(model["required_files"]),
        local_files_only=present_before,
    ))
    if resolved.resolve() != target.resolve():
        raise SkyModelRestoreError(f"resolved snapshot 漂移：{resolved} != {target}")
    audit = audit_required(config)
    audit.update({"model_id": model["id"], "revision": model["revision"], "endpoint": model["restore"]["endpoint"], "present_before": present_before, "network_attempted": not present_before})
    _write_json(run_dir / "artifacts/model_snapshot_audit.json", audit)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        destination = run_dir / "source_snapshot" / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshots[relpath] = {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
    project_git = {"head": _git(project_root, "rev-parse", "HEAD"), "branch": _git(project_root, "branch", "--show-current"), "dirty": bool(_git(project_root, "status", "--porcelain"))}
    fingerprint = {"config_sha256": sha256_file(config_path), "audit_sha256": sha256_file(run_dir / "artifacts/model_snapshot_audit.json"), "source_snapshots": snapshots, "project_git": project_git}
    _write_json(run_dir / "fingerprint.json", fingerprint)
    now = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "events.jsonl", {"at_utc": now, "event": "sky_model_restore_complete", "status": "done", "network_attempted": not present_before})
    summary = {"schema_version": "worldsim_v4_sky_model_restore_summary_v1", "task_id": TASK_ID, "status": "done", "finished_at_utc": now, "model_id": model["id"], "revision": model["revision"], "present_before": present_before, "network_attempted": not present_before, "audit_sha256": fingerprint["audit_sha256"], "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "project_git": project_git, "training_started": False, "model_inference_started": False, "test_quality_read": False}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_sky_model_restore_manifest_v1", "task_id": TASK_ID, "status": "done", "artifacts": artifacts, "network_attempted": not present_before, "test_quality_read": False})
    return summary


def record_blocked(config_path: Path, run_dir: Path, error: BaseException) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    now = datetime.now(timezone.utc).isoformat()
    event = {"at_utc": now, "event": "sky_model_restore_blocked", "error_type": type(error).__name__, "message": str(error)}
    _write_json(run_dir / "events.jsonl", event)
    _write_json(run_dir / "fingerprint.json", {"config_sha256": sha256_file(config_path) if config_path.is_file() else None, "error": event})
    summary = {"schema_version": "worldsim_v4_sky_model_restore_summary_v1", "task_id": TASK_ID, "status": "blocked", "finished_at_utc": now, "reason": "sky_model_restore_failed", "error": event, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "training_started": False, "model_inference_started": False, "test_quality_read": False}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "blocked", "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_sky_model_restore_manifest_v1", "task_id": TASK_ID, "status": "blocked", "artifacts": artifacts, "test_quality_read": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复 WorldSim V4 sky-mask 模型")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        summary = run(args.config, args.run_dir, args.project_root)
    except BaseException as error:
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

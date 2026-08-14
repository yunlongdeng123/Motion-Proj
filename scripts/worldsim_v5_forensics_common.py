#!/usr/bin/env python3
"""Shared fail-closed helpers for WorldSim V5 historical forensic audits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import yaml


RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v5")


class ForensicAuditError(RuntimeError):
    """Raised when a frozen forensic binding or protocol invariant drifts."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
    )


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ForensicAuditError(f"JSON 根节点不是 mapping: {path}")
    return payload


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ForensicAuditError(f"JSON 根节点不是 object list: {path}")
    return payload


def verify_file(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ForensicAuditError(f"冻结输入不存在: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ForensicAuditError(
            f"冻结输入 SHA 漂移: {resolved} expected={expected_sha256} actual={actual}"
        )
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def verify_named_bindings(
    run_dir: str | Path, bindings: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    root = Path(run_dir).resolve()
    return {
        name: verify_file(root / str(binding.get("path", name)), str(binding["sha256"]))
        for name, binding in bindings.items()
    }


def prepare_formal_run(run_dir: Path, task_id: str, project: Path) -> str:
    resolved = run_dir.resolve()
    task_root = (RUN_ROOT / task_id).resolve()
    if resolved.exists():
        raise ForensicAuditError(f"run 目录已存在，禁止覆盖: {resolved}")
    if task_root not in resolved.parents:
        raise ForensicAuditError(f"run 必须位于 {task_root} 下: {resolved}")
    dirty = subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ForensicAuditError("正式 forensic run 要求 clean git worktree")
    head = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    resolved.mkdir(parents=True)
    return head


def copy_source_snapshot(
    run_dir: Path, files: Iterable[Path], project: Path
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    snapshot = run_dir / "source_snapshot"
    for source in files:
        source = source.resolve()
        try:
            relative = source.relative_to(project.resolve())
        except ValueError:
            relative = Path(source.name)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append(
            {
                "path": str(destination.relative_to(run_dir)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    return inventory


def write_resolved_config(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    atomic_text(path, yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False))
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_events(run_dir: Path, events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path = run_dir / "events.jsonl"
    lines = [
        json.dumps(dict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for event in events
    ]
    atomic_text(path, "\n".join(lines) + "\n")
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_files(run_dir: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(run_dir))
        if relative in excluded:
            continue
        rows.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return rows


def finalize_formal_run(
    *,
    run_dir: Path,
    task_id: str,
    task_status: str,
    conclusion: str,
    project_head: str,
    input_bindings: Mapping[str, Any],
    summary: Mapping[str, Any],
    resolved_config_record: Mapping[str, Any],
    events_record: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    atomic_json(summary_path, summary)
    fingerprint = {
        "schema_version": "worldsim_v5_forensic_fingerprint_v1",
        "task_id": task_id,
        "project_git_head": project_head,
        "resolved_config": dict(resolved_config_record),
        "input_bindings": dict(input_bindings),
        "checkpoint": {"applicable": False, "reason": "historical_read_only_audit"},
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    manifest = {
        "schema_version": "worldsim_v5_forensic_manifest_v1",
        "task_id": task_id,
        "status": "done",
        "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    status = {
        "schema_version": "worldsim_v5_forensic_status_v1",
        "task_id": task_id,
        "task_status": task_status,
        "status": "done",
        "conclusion": conclusion,
        "project_git_head": project_head,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "resolved_config_sha256": str(resolved_config_record["sha256"]),
        "events_sha256": str(events_record["sha256"]),
        "checkpoint_applicable": False,
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "finished_at_utc": utc_now(),
    }
    atomic_json(run_dir / "status.json", status)
    return status

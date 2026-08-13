#!/usr/bin/env python3
"""Register 18 train-only StreetGS checkpoints without reading test quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"
SCHEMA = "worldsim_v4_m3_test_checkpoint_registry_v1"


class TestCheckpointRegistryError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TestCheckpointRegistryError(f"JSON root is not a mapping: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TestCheckpointRegistryError(f"YAML root is not a mapping: {path}")
    return payload


def parse_run_bindings(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        scene, separator, path = value.partition("=")
        if not separator or not scene or not path or scene in output:
            raise TestCheckpointRegistryError(f"invalid/duplicate run binding: {value}")
        output[scene] = Path(path).resolve()
    return output


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        process = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True, check=False)
        if process.returncode:
            raise TestCheckpointRegistryError(process.stderr.strip())
        return process.stdout.strip()

    state = {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status": run("status", "--porcelain")}
    state["dirty"] = bool(state["status"])
    return state


def build_registry(config_path: Path, run_bindings: Mapping[str, Path]) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v4_streetgs_training_v1" or config.get("task_id") != TASK_ID:
        raise TestCheckpointRegistryError("M3 test reconstruction config drift")
    expected = list(config.get("scenes", {}))
    if len(expected) != 18 or set(run_bindings) != set(expected):
        raise TestCheckpointRegistryError("test checkpoint scene set drift")
    checkpoints: dict[str, Any] = {}
    for scene in expected:
        run = run_bindings[scene]
        status = load_json(run / "status.json")
        summary = load_json(run / "summary.json")
        manifest = load_json(run / "manifest.json")
        scene_index = int(config["scenes"][scene])
        if (
            status.get("status") != "done"
            or status.get("task_id") != TASK_ID
            or status.get("scene") != scene
            or status.get("mode") != "formal"
            or status.get("summary_sha256") != sha256_file(run / "summary.json")
            or summary.get("status") != "done"
            or summary.get("task_id") != TASK_ID
            or summary.get("scene") != scene
            or summary.get("scene_index") != scene_index
            or summary.get("mode") != "formal"
            or summary.get("iterations") != 30000
            or summary.get("project_git", {}).get("dirty") is not False
            or summary.get("model_inference_started") is not False
            or summary.get("test_quality_read") is not False
            or manifest.get("status") != "done"
            or manifest.get("scene") != scene
            or manifest.get("mode") != "formal"
            or manifest.get("test_quality_read") is not False
        ):
            raise TestCheckpointRegistryError(f"train-only StreetGS contract drift: {scene}")
        checkpoint = dict(summary["checkpoint"])
        path = Path(checkpoint["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(checkpoint["bytes"])
            or sha256_file(path) != checkpoint["sha256"]
            or checkpoint.get("step") != 30000
            or checkpoint.get("means_finite") is not True
            or manifest.get("artifacts", {}).get("work_dirs_checkpoint") != checkpoint
        ):
            raise TestCheckpointRegistryError(f"checkpoint content drift: {scene}")
        source_config = path.parent / "config.yaml"
        if not source_config.is_file():
            raise TestCheckpointRegistryError(f"DriveStudio config missing: {scene}")
        checkpoints[scene] = {
            **checkpoint,
            "scene_index": scene_index,
            "run": str(run),
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "summary_sha256": sha256_file(run / "summary.json"),
            "manifest_sha256": sha256_file(run / "manifest.json"),
            "status_sha256": sha256_file(run / "status.json"),
            "fingerprint_sha256": sha256_file(run / "fingerprint.json"),
        }
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "status": "done",
        "split": "test_asset_preparation",
        "partition_contract": "sample_index_mod_5",
        "scene_order": expected,
        "checkpoints": checkpoints,
        "source_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "training_partition_only": True,
        "render_started": False,
        "test_quality_read": False,
    }


def manifest(run_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(run_dir).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema_version": "worldsim_v4_m3_test_checkpoint_manifest_v1", "task_id": TASK_ID, "status": "done", "files": files, "test_quality_read": False}


def run(config_path: Path, run_bindings: Mapping[str, Path], run_dir: Path, project_root: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    state = git_state(project_root)
    if state["dirty"]:
        raise TestCheckpointRegistryError("formal registry requires a clean project tree")
    payload = build_registry(config_path, run_bindings)
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "source_snapshot").mkdir()
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    for relpath in ("configs/worldsim_v4/m3_test_reconstruction_v1.yaml", "scripts/register_worldsim_v4_m3_test_checkpoints.py"):
        source = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    output = run_dir / "artifacts/m3_test_checkpoint_registry.yaml"
    atomic_yaml(output, payload)
    now = datetime.now(timezone.utc).isoformat()
    fingerprint = {"source_commit": state["head"], "config_sha256": sha256_file(config_path), "checkpoint_registry_sha256": sha256_file(output), "test_quality_read": False}
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    summary = {"schema_version": "worldsim_v4_m3_test_checkpoint_summary_v1", "task_id": TASK_ID, "status": "done", "scene_count": 18, "checkpoint_registry": {"path": str(output), "sha256": sha256_file(output)}, "project_git": state, "render_started": False, "test_quality_read": False, "finished_at_utc": now}
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "events.jsonl", {"at_utc": now, "event": "test_train_only_checkpoints_registered", "scene_count": 18, "test_quality_read": False})
    atomic_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "summary_sha256": sha256_file(run_dir / "summary.json"), "test_quality_read": False})
    atomic_json(run_dir / "manifest.json", manifest(run_dir))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = run(args.config.resolve(), parse_run_bindings(args.run), args.run_dir.resolve(), args.project_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

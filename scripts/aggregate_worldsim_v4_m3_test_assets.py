#!/usr/bin/env python3
"""Aggregate 18 checkpoint/actor bindings before the test-quality freeze."""

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


class M3TestAssetError(RuntimeError):
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3TestAssetError(f"JSON root must be mapping: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3TestAssetError(f"YAML root must be mapping: {path}")
    return value


def parse_bindings(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        scene, separator, path = value.partition("=")
        if not separator or not scene or not path or scene in output:
            raise M3TestAssetError(f"invalid/duplicate actor run binding: {value}")
        output[scene] = Path(path).resolve()
    return output


def git_state(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        process = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True, check=False)
        if process.returncode:
            raise M3TestAssetError(process.stderr.strip())
        return process.stdout.strip()

    state = {"head": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"), "status": command("status", "--porcelain")}
    state["dirty"] = bool(state["status"])
    return state


def verified_binding(run: Path, scene: str) -> dict[str, Any]:
    status = load_json(run / "status.json")
    summary = load_json(run / "summary.json")
    manifest = load_json(run / "manifest.json")
    binding_path = run / "artifacts/scene_binding.json"
    binding = load_json(binding_path)
    if (
        status.get("status") != "done"
        or status.get("scene") != scene
        or status.get("summary_sha256") != sha256_file(run / "summary.json")
        or status.get("test_quality_read") is not False
        or summary.get("status") != "done"
        or summary.get("scene") != scene
        or summary.get("scene_binding", {}).get("sha256") != sha256_file(binding_path)
        or summary.get("render_started") is not False
        or summary.get("test_quality_read") is not False
        or manifest.get("status") != "done"
        or manifest.get("test_quality_read") is not False
        or binding.get("schema_version") != "worldsim_v4_m3_test_scene_binding_v1"
        or binding.get("scene") != scene
        or binding.get("partition") != "test"
        or binding.get("render_started") is not False
        or binding.get("test_quality_read") is not False
    ):
        raise M3TestAssetError(f"actor registry run contract drift: {scene}")
    for label in ("checkpoint", "drivestudio_config", "registry"):
        row = binding[label]
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise M3TestAssetError(f"{label} content drift: {scene}")
    return binding


def build_inventory(cohort_path: Path, checkpoint_registry_path: Path, actor_runs: Mapping[str, Path]) -> dict[str, Any]:
    cohort = load_yaml(cohort_path)
    checkpoint_registry = load_yaml(checkpoint_registry_path)
    expected = cohort.get("freeze", {}).get("scene_roles", {}).get("test", [])
    if len(expected) != 18 or set(actor_runs) != set(expected):
        raise M3TestAssetError("actor registry runs must match the frozen 18-scene test cohort")
    if checkpoint_registry.get("scene_order") != expected or checkpoint_registry.get("test_quality_read") is not False:
        raise M3TestAssetError("checkpoint registry order/quality contract drift")
    scenes: dict[str, Any] = {}
    for scene in expected:
        binding = verified_binding(actor_runs[scene], scene)
        scenes[scene] = {
            key: binding[key]
            for key in ("partition", "status", "reason", "instance_token", "clip", "actor", "checkpoint", "drivestudio_config", "registry")
        }
    return {
        "schema_version": "worldsim_v4_m3_test_asset_inventory_v1",
        "task_id": TASK_ID,
        "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
        "checkpoint_registry": {"path": str(checkpoint_registry_path), "sha256": sha256_file(checkpoint_registry_path)},
        "processed_keyframe_stride": 5,
        "drivestudio_checkout": "/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0",
        "drivestudio_python": "/root/autodl-tmp/envs/drivestudio/bin/python",
        "camera_ids": [0, 1, 2],
        "scene_order": expected,
        "scenes": scenes,
        "asset_summary": {"scene_count": 18, "ready_count": sum(row["status"] == "ready" for row in scenes.values()), "abstain_count": sum(row["status"] == "abstain" for row in scenes.values())},
        "asset_preparation_only": True,
        "render_started": False,
        "test_quality_read": False,
    }


def manifest(run_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(run_dir).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema_version": "worldsim_v4_m3_test_asset_manifest_v1", "task_id": TASK_ID, "status": "done", "files": files, "test_quality_read": False}


def run(cohort_path: Path, checkpoint_registry_path: Path, actor_runs: Mapping[str, Path], run_dir: Path, project_root: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    state = git_state(project_root)
    if state["dirty"]:
        raise M3TestAssetError("formal asset aggregation requires a clean project tree")
    inventory = build_inventory(cohort_path, checkpoint_registry_path, actor_runs)
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    for name, source in (("cohort.yaml", cohort_path), ("checkpoint_registry.yaml", checkpoint_registry_path)):
        shutil.copy2(source, run_dir / name)
    for relpath in ("scripts/aggregate_worldsim_v4_m3_test_assets.py", "scripts/build_worldsim_v4_m3_test_actor_registry.py"):
        source = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    output = run_dir / "artifacts/m3_test_asset_inventory.yaml"
    atomic_yaml(output, inventory)
    now = datetime.now(timezone.utc).isoformat()
    summary = {"schema_version": "worldsim_v4_m3_test_asset_summary_v1", "task_id": TASK_ID, "status": "done", **inventory["asset_summary"], "inventory": {"path": str(output), "sha256": sha256_file(output)}, "project_git": state, "render_started": False, "test_quality_read": False, "finished_at_utc": now}
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "events.jsonl", {"at_utc": now, "event": "test_assets_aggregated_without_quality_read", **inventory["asset_summary"], "test_quality_read": False})
    atomic_json(run_dir / "fingerprint.json", {"source_commit": state["head"], "cohort_sha256": sha256_file(cohort_path), "checkpoint_registry_sha256": sha256_file(checkpoint_registry_path), "inventory_sha256": summary["inventory"]["sha256"], "test_quality_read": False})
    atomic_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "summary_sha256": sha256_file(run_dir / "summary.json"), "test_quality_read": False})
    atomic_json(run_dir / "manifest.json", manifest(run_dir))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--actor-run", action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = run(args.cohort.resolve(), args.checkpoint_registry.resolve(), parse_bindings(args.actor_run), args.run_dir.resolve(), args.project_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

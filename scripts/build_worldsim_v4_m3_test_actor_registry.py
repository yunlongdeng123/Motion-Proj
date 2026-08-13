#!/usr/bin/env python3
"""Bind one frozen test actor to a train-only StreetGS checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"


class M3TestActorRegistryError(RuntimeError):
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


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3TestActorRegistryError(f"YAML root must be mapping: {path}")
    return value


def git_state(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        process = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True, check=False)
        if process.returncode:
            raise M3TestActorRegistryError(process.stderr.strip())
        return process.stdout.strip()

    state = {"head": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"), "status": command("status", "--porcelain")}
    state["dirty"] = bool(state["status"])
    return state


def resolve_scene_metadata(cohort: Mapping[str, Any], scene: str) -> dict[str, Any]:
    test_scenes = cohort.get("freeze", {}).get("scene_roles", {}).get("test", [])
    if len(test_scenes) != 18 or scene not in test_scenes:
        raise M3TestActorRegistryError("scene is not in the frozen 18-scene test cohort")
    matches = [row for row in cohort["freeze"]["scene_records"] if row.get("scene") == scene and row.get("role") == "test"]
    if len(matches) != 1:
        raise M3TestActorRegistryError("frozen test scene record must match exactly once")
    row = matches[0]
    clip = row.get("continuous_clip", {})
    if clip.get("status") != "ready" or len(clip.get("sample_tokens", [])) != 7 or int(clip.get("end_index", -1)) - int(clip.get("start_index", -1)) != 6:
        raise M3TestActorRegistryError("test scene continuous clip contract drift")
    token = str(clip.get("actor_instance_token", ""))
    if token != row.get("actors", {}).get("high_support", {}).get("instance_token"):
        raise M3TestActorRegistryError("continuous clip actor token drift")
    return {"scene": scene, "instance_token": token, "clip": {"start_index": int(clip["start_index"]), "end_index": int(clip["end_index"]), "duration_s": float(clip["duration_s"])}}


def bind_actor(registry: Mapping[str, Any], token: str) -> dict[str, Any]:
    actors = [row for row in registry.get("actors", []) if row.get("instance_token") == token]
    if len(actors) != 1:
        raise M3TestActorRegistryError("target actor must match registry exactly once")
    actor = actors[0]
    availability = str(actor.get("availability"))
    gaussian_count = int(actor.get("checkpoint_tensor_slice", {}).get("gaussian_count", 0))
    ready = availability == "available" and gaussian_count > 0 and actor.get("rigid_model_index") is not None
    if availability == "available" and not ready:
        raise M3TestActorRegistryError("available actor has an empty checkpoint binding")
    return {
        "status": "ready" if ready else "abstain",
        "reason": None if ready else availability,
        "actor": {"model_index": int(actor["rigid_model_index"]) if ready else None, "gaussian_count": gaussian_count},
    }


def manifest(run_dir: Path, status: str) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(run_dir).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema_version": "worldsim_v4_m3_test_actor_registry_manifest_v1", "task_id": TASK_ID, "status": status, "files": files, "test_quality_read": False}


def run(checkpoint_registry_path: Path, cohort_path: Path, project_root: Path, scene: str, run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    state = git_state(project_root)
    if state["dirty"]:
        raise M3TestActorRegistryError("formal actor registry requires a clean project tree")
    checkpoint_registry = load_yaml(checkpoint_registry_path)
    cohort = load_yaml(cohort_path)
    if checkpoint_registry.get("schema_version") != "worldsim_v4_m3_test_checkpoint_registry_v1" or checkpoint_registry.get("task_id") != TASK_ID or checkpoint_registry.get("test_quality_read") is not False:
        raise M3TestActorRegistryError("test checkpoint registry drift")
    if checkpoint_registry.get("scene_order") != cohort.get("freeze", {}).get("scene_roles", {}).get("test"):
        raise M3TestActorRegistryError("checkpoint registry order differs from D0 test freeze")
    metadata = resolve_scene_metadata(cohort, scene)
    checkpoint = checkpoint_registry.get("checkpoints", {}).get(scene)
    if not isinstance(checkpoint, Mapping):
        raise M3TestActorRegistryError("scene checkpoint missing")
    checkpoint_path = Path(str(checkpoint["path"]))
    source_config = Path(str(checkpoint["source_config"]))
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != int(checkpoint["bytes"])
        or sha256_file(checkpoint_path) != checkpoint["sha256"]
        or not source_config.is_file()
        or sha256_file(source_config) != checkpoint["source_config_sha256"]
    ):
        raise M3TestActorRegistryError("scene checkpoint/config content drift")

    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()
    for name, source in (("checkpoint_registry.yaml", checkpoint_registry_path), ("cohort.yaml", cohort_path)):
        shutil.copy2(source, run_dir / name)
    for relpath in ("scripts/build_dr_v2_drivestudio_registry.py", "scripts/build_worldsim_v4_m3_test_actor_registry.py", "motion_proj/dynamic_editing_v2/drivestudio_registry.py"):
        source = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    registry_path = run_dir / "artifacts/actor_registry.json"
    raw_metadata = Path(str(cohort["dataset"]["metadata_root"])) / "v1.0-trainval"
    command = [
        "/root/autodl-tmp/envs/drivestudio/bin/python",
        str(project_root / "scripts/build_dr_v2_drivestudio_registry.py"),
        "--checkpoint", str(checkpoint_path),
        "--drivestudio-root", "/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0",
        "--raw-metadata", str(raw_metadata),
        "--scene-name", scene,
        "--selected-token", metadata["instance_token"],
        "--requested-token", metadata["instance_token"],
        "--allow-missing-selected",
        "--output", str(registry_path),
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": f"{project_root}:/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0"})
    with (run_dir / "logs/registry.log").open("xb") as log:
        process = subprocess.run(command, cwd=project_root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
    if process.returncode:
        raise M3TestActorRegistryError(f"actor registry child failed: rc={process.returncode}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    bound = bind_actor(registry, metadata["instance_token"])
    binding = {
        "schema_version": "worldsim_v4_m3_test_scene_binding_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "partition": "test",
        "status": bound["status"],
        "reason": bound["reason"],
        "instance_token": metadata["instance_token"],
        "clip": metadata["clip"],
        "actor": bound["actor"],
        "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
        "drivestudio_config": {"path": str(source_config), "bytes": source_config.stat().st_size, "sha256": sha256_file(source_config)},
        "registry": {"path": str(registry_path), "bytes": registry_path.stat().st_size, "sha256": sha256_file(registry_path)},
        "checkpoint_registry": {"path": str(checkpoint_registry_path), "sha256": sha256_file(checkpoint_registry_path)},
        "render_started": False,
        "test_quality_read": False,
    }
    atomic_json(run_dir / "artifacts/scene_binding.json", binding)
    now = datetime.now(timezone.utc).isoformat()
    summary = {"schema_version": "worldsim_v4_m3_test_actor_registry_summary_v1", "task_id": TASK_ID, "scene": scene, "status": "done", "asset_status": bound["status"], "reason": bound["reason"], "actor": bound["actor"], "scene_binding": {"path": str(run_dir / "artifacts/scene_binding.json"), "sha256": sha256_file(run_dir / "artifacts/scene_binding.json")}, "project_git": state, "model_inference_started": False, "render_started": False, "test_quality_read": False, "finished_at_utc": now}
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "events.jsonl", {"at_utc": now, "event": "test_actor_registry_bound", "scene": scene, "asset_status": bound["status"], "test_quality_read": False})
    atomic_json(run_dir / "fingerprint.json", {"source_commit": state["head"], "cohort_sha256": sha256_file(cohort_path), "checkpoint_registry_sha256": sha256_file(checkpoint_registry_path), "scene_binding_sha256": summary["scene_binding"]["sha256"], "test_quality_read": False})
    atomic_json(run_dir / "status.json", {"task_id": TASK_ID, "scene": scene, "status": "done", "summary_sha256": sha256_file(run_dir / "summary.json"), "test_quality_read": False})
    atomic_json(run_dir / "manifest.json", manifest(run_dir, "done"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.checkpoint_registry.resolve(), args.cohort.resolve(), args.project_root.resolve(), args.scene, args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

"""为 V7.2 干净角色生成与 V7/V7.1 EAS 接口兼容的 build-only Actor corpus。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.actor_corpus import materialize_actor_cache
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _scene_rows(metadata_root: Path, log_ids: list[str]) -> tuple[list[str], dict[str, str]]:
    rows = json.loads((metadata_root / "scene.json").read_text(encoding="utf-8"))
    wanted = set(log_ids)
    scene_to_log = {
        str(row["name"]): str(row["log_token"])
        for row in rows
        if str(row["log_token"]) in wanted
    }
    return list(scene_to_log), scene_to_log


def run(config_path: Path, role: str, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if role not in {"dev", "route_select", "source_test"}:
        raise PermissionError(f"unsupported clean role: {role}")
    if role == "source_test" and not bool(config["source_test_read"]):
        raise PermissionError("source_test requires a separately frozen read configuration")
    if role != "source_test" and bool(config["source_test_read"]):
        raise PermissionError("source_test_read must remain false for development roles")
    if bool(config["external_test_read"]):
        raise PermissionError("EAS corpus builder never reads external test")
    roles = load_data_roles(REPO_ROOT / config["roles"])
    log_ids = require_role_access(roles, "nuscenes", role)
    metadata_root = Path(config["dataset_root"]) / "v1.0-trainval"
    scene_names, scene_to_log = _scene_rows(metadata_root, log_ids)
    role_root = Path(config["output_root"]) / role
    if role_root.exists():
        raise FileExistsError(f"EAS role corpus already exists: {role_root}")
    role_root.mkdir(parents=True)
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    manifest = {
        "schema_version": "worldsim_v72.eas_actor_corpus_run.v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "status": "running",
        "role": role,
        "log_ids": log_ids,
        "scene_count": len(scene_names),
        "git_commit": commit,
        "target_mount_separate": True,
        "source_test_read": role == "source_test",
        "external_test_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "index"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump({**config, "role": role, "run_id": run_id}, sort_keys=False),
        encoding="utf-8",
    )
    started = time.monotonic()
    try:
        compiler = yaml.safe_load((REPO_ROOT / config["compiler_config"]).read_text(encoding="utf-8"))
        _deep_update(compiler, config["compiler_overrides"])
        index = build_v71_index(Path(config["dataset_root"]), {"roles": {role: scene_names}})
        device = torch.device(str(config["device"]))
        rows: list[dict[str, Any]] = []
        for scene_index, scene_name in enumerate(scene_names):
            bundles = compile_source_scene(scene_name, index, config["actors"], compiler, device)
            for bundle in bundles:
                track_id = str(bundle["row"]["track_id"])
                output = role_root / scene_name / f"{track_id}.npz"
                row = materialize_actor_cache(
                    bundle,
                    output,
                    config["cache"],
                    oracle_by_track={},
                    device=device,
                )
                rows.append(
                    {
                        "log_id": scene_to_log[scene_name],
                        "scene_name": scene_name,
                        "track_id": track_id,
                        "relative_path": str(output.relative_to(role_root)),
                        **row,
                    }
                )
            _write_json(
                run_dir / "status.json",
                {
                    "status": "running",
                    "phase": "materialize",
                    "scenes": scene_index + 1,
                    "actors": len(rows),
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "eas_actor_corpus",
                        "role": role,
                        "progress": f"{scene_index + 1}/{len(scene_names)}",
                        "actors": len(rows),
                    }
                ),
                flush=True,
            )
        if not rows:
            raise RuntimeError(f"{role} produced no EAS actors")
        _write_jsonl(role_root / "ACTORS.jsonl", rows)
        fingerprint = hashlib.sha256(
            json.dumps(
                [[row["log_id"], row["scene_name"], row["track_id"]] for row in rows]
            ).encode()
        ).hexdigest()
        summary = {
            **manifest,
            "status": "done",
            "actor_count": len(rows),
            "materialized_scene_count": len({row["scene_name"] for row in rows}),
            "hazard_actor_count": sum(int(row["hazardous"]) for row in rows),
            "candidate_count": sum(int(row["candidate_count"]) for row in rows),
            "oracle_target_actor_count": sum(int(row["oracle_target"]) for row in rows),
            "fingerprint": fingerprint,
            "resources": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(role_root / "manifest.json", summary)
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.role, arguments.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

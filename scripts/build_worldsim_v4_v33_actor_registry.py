#!/usr/bin/env python3
"""为单个 V4 development scene 构建内容寻址的 V3.3 actor registry。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import (
    V33ReplayError,
    bind_actor_registry,
    load_yaml,
    sha256_file,
)


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def resolve_scene_row(inputs: Mapping[str, Any], scene: str) -> dict[str, Any]:
    if inputs.get("schema_version") != "worldsim_v4_v33_replay_inputs_v1":
        raise V33ReplayError("resolved inputs schema 漂移")
    if inputs.get("partition_contract") != "sample_index_mod_5":
        raise V33ReplayError("resolved inputs partition 漂移")
    if inputs.get("test_quality_read") is not False:
        raise V33ReplayError("registry 阶段禁止读取 test quality")
    rows = [row for row in inputs.get("scenes", []) if row.get("scene") == scene]
    if len(rows) != 1:
        raise V33ReplayError(f"scene 必须精确命中一次：{scene}")
    return dict(rows[0])


def build_command(
    *,
    project_root: Path,
    replay_config: Mapping[str, Any],
    scene_row: Mapping[str, Any],
    output: Path,
) -> list[str]:
    return [
        str(replay_config["runtimes"]["drivestudio_python"]),
        str(project_root / "scripts/build_dr_v2_drivestudio_registry.py"),
        "--checkpoint",
        str(scene_row["base_checkpoint"]["path"]),
        "--drivestudio-root",
        str(replay_config["runtimes"]["drivestudio_checkout"]),
        "--raw-metadata",
        str(replay_config["inputs"]["raw_metadata"]),
        "--scene-name",
        str(scene_row["scene"]),
        "--selected-token",
        str(scene_row["actors"]["high_support"]["instance_token"]),
        "--requested-token",
        str(scene_row["actors"]["high_support"]["instance_token"]),
        "--requested-token",
        str(scene_row["actors"]["boundary_support"]["instance_token"]),
        "--allow-missing-selected",
        "--output",
        str(output),
    ]


def build_environment(
    *, project_root: Path, replay_config: Mapping[str, Any]
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(
        replay_config["runtimes"]["cuda_visible_devices"]
    )
    environment["PYTHONPATH"] = (
        f"{project_root}:{replay_config['runtimes']['drivestudio_checkout']}"
    )
    return environment


def _manifest(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "worldsim_v4_v33_registry_manifest_v1",
        "task_id": TASK_ID,
        "files": rows,
    }


def run(
    *,
    replay_config_path: Path,
    resolved_inputs_path: Path,
    project_root: Path,
    scene: str,
    run_dir: Path,
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run 目录已存在，禁止覆盖：{run_dir}")
    replay_config = load_yaml(replay_config_path)
    inputs = json.loads(resolved_inputs_path.read_text(encoding="utf-8"))
    scene_row = resolve_scene_row(inputs, scene)
    checkpoint = Path(scene_row["base_checkpoint"]["path"])
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != scene_row["base_checkpoint"]["bytes"]
        or sha256_file(checkpoint) != scene_row["base_checkpoint"]["sha256"]
    ):
        raise V33ReplayError("registry 输入 checkpoint bytes/SHA 漂移")

    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()
    for name, source in (
        ("replay_config.yaml", replay_config_path),
        ("resolved_inputs.json", resolved_inputs_path),
    ):
        shutil.copy2(source, run_dir / name)
    for relative in (
        "motion_proj/worldsim_v4/v33_replay.py",
        "scripts/build_dr_v2_drivestudio_registry.py",
        "scripts/build_worldsim_v4_v33_actor_registry.py",
    ):
        source = project_root / relative
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_json(run_dir / "resolved_scene.json", scene_row)
    status = {
        "schema_version": "worldsim_v4_v33_registry_status_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "status": "running",
        "test_quality_read": False,
    }
    atomic_json(run_dir / "status.json", status)

    registry_path = run_dir / "artifacts" / "actor_registry.json"
    command = build_command(
        project_root=project_root,
        replay_config=replay_config,
        scene_row=scene_row,
        output=registry_path,
    )
    environment = build_environment(
        project_root=project_root,
        replay_config=replay_config,
    )
    with (run_dir / "logs" / "registry.log").open("xb") as log:
        process = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    if process.returncode != 0:
        raise V33ReplayError(f"actor registry child 失败：exit={process.returncode}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    bound = bind_actor_registry(
        scene_row,
        registry,
        require_high_available=bool(
            replay_config["gates"]["require_available_high_actor"]
        ),
    )
    bound.update(
        schema_version="worldsim_v4_v33_bound_scene_v1",
        actor_registry={
            "path": str(registry_path),
            "bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
        },
        partition_contract=inputs["partition_contract"],
        algorithm_commit=inputs["algorithm_commit"],
    )
    atomic_json(run_dir / "artifacts" / "bound_scene.json", bound)
    summary = {
        "schema_version": "worldsim_v4_v33_registry_summary_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "status": "done",
        "algorithm_commit": inputs["algorithm_commit"],
        "partition_contract": inputs["partition_contract"],
        "base_checkpoint_sha256": scene_row["base_checkpoint"]["sha256"],
        "registry": {
            "path": str(registry_path),
            "bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
        },
        "actors": {
            role: {
                key: actor[key]
                for key in (
                    "instance_token",
                    "dataset_instance_id",
                    "availability",
                    "rigid_model_index",
                )
            }
            for role, actor in bound["actors"].items()
        },
        "runtime": {
            "dataset_device": "cuda",
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        },
        "model_inference_started": False,
        "training_started": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "manifest.json", _manifest(run_dir))
    status.update(
        status="done",
        summary_sha256=sha256_file(run_dir / "summary.json"),
        manifest_sha256=sha256_file(run_dir / "manifest.json"),
    )
    atomic_json(run_dir / "status.json", status)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolved-inputs", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_preexisted = run_dir.exists()
    try:
        summary = run(
            replay_config_path=args.config.resolve(),
            resolved_inputs_path=args.resolved_inputs.resolve(),
            project_root=args.project_root.resolve(),
            scene=args.scene,
            run_dir=run_dir,
        )
    except Exception as error:
        if not run_preexisted and run_dir.is_dir():
            status_path = run_dir / "status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {
                    "schema_version": "worldsim_v4_v33_registry_status_v1",
                    "task_id": TASK_ID,
                    "scene": args.scene,
                }
            status.update(
                status="failed",
                reason=type(error).__name__,
                error=str(error),
                test_quality_read=False,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(status_path, status)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""将 V5 development raw manifests 原子转换为 DriveStudio 10 Hz 场景。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVESTUDIO_PYTHON = Path("/root/autodl-tmp/envs/drivestudio/bin/python")
UPSTREAM_ROOT = Path("/root/autodl-tmp/third_party/drivestudio")


class PreprocessError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def processor_output_root(target_dir: Path) -> Path:
    transformed = str(target_dir).replace("processed", "processed_10Hz")
    if transformed == str(target_dir):
        raise PreprocessError("processor target-dir 必须包含 processed，以匹配上游输出合同")
    return Path(transformed) / "trainval"


def expected_frames(sample_count: int, interpolate_n: int = 4) -> int:
    if sample_count < 2:
        raise PreprocessError(f"sample_count 必须 >=2，实际 {sample_count}")
    return (sample_count - 1) * (interpolate_n + 1) + 1


def nonempty_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.glob(pattern))
    empty = [str(path) for path in files if path.stat().st_size <= 0]
    if empty:
        raise PreprocessError(f"存在空产物，示例: {empty[:3]}")
    return files


def validate_processed_scene(
    scene_root: Path, frame_count: int, camera_count: int = 6
) -> dict[str, Any]:
    if not scene_root.is_dir() or scene_root.is_symlink():
        raise PreprocessError(f"processed scene 缺失或为 symlink: {scene_root}")
    groups = {
        "images": nonempty_files(scene_root / "images", "*.jpg"),
        "extrinsics": nonempty_files(scene_root / "extrinsics", "*.txt"),
        "intrinsics": nonempty_files(scene_root / "intrinsics", "*.txt"),
        "lidar": nonempty_files(scene_root / "lidar", "*.bin"),
        "lidar_pose": nonempty_files(scene_root / "lidar_pose", "*.txt"),
        "dynamic_masks_all": nonempty_files(
            scene_root / "dynamic_masks/all", "*.png"
        ),
        "dynamic_masks_human": nonempty_files(
            scene_root / "dynamic_masks/human", "*.png"
        ),
        "dynamic_masks_vehicle": nonempty_files(
            scene_root / "dynamic_masks/vehicle", "*.png"
        ),
    }
    expected = {
        "images": frame_count * camera_count,
        "extrinsics": frame_count * camera_count,
        "intrinsics": camera_count,
        "lidar": frame_count,
        "lidar_pose": frame_count,
        "dynamic_masks_all": frame_count * camera_count,
        "dynamic_masks_human": frame_count * camera_count,
        "dynamic_masks_vehicle": frame_count * camera_count,
    }
    for name, wanted in expected.items():
        if len(groups[name]) != wanted:
            raise PreprocessError(
                f"{scene_root.name}/{name}: {len(groups[name])} != {wanted}"
            )
    instances = [
        scene_root / "instances/instances_info.json",
        scene_root / "instances/frame_instances.json",
    ]
    for path in instances:
        if not (path.is_file() and path.stat().st_size > 0):
            raise PreprocessError(f"缺少实例产物: {path}")
        json.loads(path.read_text(encoding="utf-8"))
    samples = [
        groups["images"][0],
        groups["images"][-1],
        groups["lidar"][0],
        groups["lidar"][-1],
        *instances,
    ]
    return {
        "frame_count": frame_count,
        "camera_count": camera_count,
        "counts": {name: len(files) for name, files in groups.items()},
        "instance_json_count": len(instances),
        "sample_artifacts": {
            str(path.relative_to(scene_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in samples
        },
    }


def load_inputs(config_path: Path, raw_batch_path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_batch = json.loads(raw_batch_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "WS-V5-M1-STRUCTURED-OWNERSHIP-01":
        raise PreprocessError("M1 task_id 漂移")
    if config.get("status") != "running":
        raise PreprocessError("M1 必须保持 running")
    if raw_batch.get("complete") is not True:
        raise PreprocessError("raw batch manifest 未完成")
    if raw_batch.get("quality_read") is not False:
        raise PreprocessError("raw batch quality_read 合同漂移")
    config_scenes = {
        row["scene"]: int(row["scene_index"])
        for row in config["fresh_cohort_binding"]["development_scenes"]
    }
    raw_scenes = {
        row["scene_name"]: int(row["scene_index"])
        for row in raw_batch["scenes"]
    }
    if config_scenes != raw_scenes:
        raise PreprocessError("raw batch 与 frozen development identity 不一致")
    return config, raw_batch


def verify_raw_scene(raw_root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise PreprocessError(f"raw scene manifest 未完成: {manifest_path}")
    for row in manifest["files"]:
        path = raw_root / row["filename"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise PreprocessError(f"raw bytes 漂移: {path}")
    return manifest


def build_inventory(scene_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in scene_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(scene_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    raw_batch_path = args.raw_batch_manifest.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise PreprocessError(f"run 目录已存在，禁止覆盖: {run_dir}")
    if git_output("status", "--porcelain"):
        raise PreprocessError("正式 preprocess 要求 clean worktree")
    config, raw_batch = load_inputs(config_path, raw_batch_path)
    raw_root = Path(config["data_readiness"]["raw_root"])
    final_root = Path(config["data_readiness"]["processed_root"])
    configured_target = Path(config["data_readiness"]["processed_target_dir"])
    if processor_output_root(configured_target) != final_root:
        raise PreprocessError("processed target/output root 合同漂移")

    requested = set(args.scene_name or [])
    scene_rows = [
        row for row in raw_batch["scenes"] if not requested or row["scene_name"] in requested
    ]
    unknown = requested - {row["scene_name"] for row in scene_rows}
    if unknown:
        raise PreprocessError(f"请求了非 development scene: {sorted(unknown)}")
    if not scene_rows:
        raise PreprocessError("没有待处理 scene")

    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "artifacts").mkdir()
    (run_dir / "source_snapshot").mkdir()
    shutil.copy2(config_path, run_dir / "resolved_config.yaml")
    shutil.copy2(raw_batch_path, run_dir / "raw_batch_manifest.json")
    shutil.copy2(Path(__file__), run_dir / "source_snapshot" / Path(__file__).name)
    preprocess_wrapper = PROJECT_ROOT / "scripts/preprocess_dr_v2_nuscenes_single.py"
    shutil.copy2(
        preprocess_wrapper,
        run_dir / "source_snapshot" / preprocess_wrapper.name,
    )
    events = run_dir / "events.jsonl"
    started_at_utc = now_utc()
    append_event(
        events,
        {
            "at_utc": started_at_utc,
            "event": "preprocess_batch_started",
            "scene_count": len(scene_rows),
            "quality_read": False,
        },
    )
    started = time.monotonic()
    results = []
    staging_root = (
        final_root.parents[1]
        / "preprocess_staging"
        / f".partial.{run_dir.name}"
    )
    try:
        for row in scene_rows:
            scene_name = row["scene_name"]
            scene_index = int(row["scene_index"])
            raw_manifest_path = Path(row["manifest"])
            raw_manifest = verify_raw_scene(raw_root, raw_manifest_path)
            frames = expected_frames(int(raw_manifest["sample_count"]))
            final_scene = final_root / f"{scene_index:03d}"
            if final_scene.exists():
                raise PreprocessError(f"final scene 已存在，禁止覆盖: {final_scene}")
            scene_stage = staging_root / scene_name
            target_dir = scene_stage / "drivestudio_processed"
            staged_scene = processor_output_root(target_dir) / f"{scene_index:03d}"
            command = [
                str(DRIVESTUDIO_PYTHON),
                str(PROJECT_ROOT / "scripts/preprocess_dr_v2_nuscenes_single.py"),
                "--data-root",
                str(raw_root),
                "--target-dir",
                str(target_dir),
                "--scene-index",
                str(scene_index),
                "--upstream-root",
                str(UPSTREAM_ROOT),
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = f"{PROJECT_ROOT}:{UPSTREAM_ROOT}"
            append_event(
                events,
                {
                    "at_utc": now_utc(),
                    "event": "scene_preprocess_started",
                    "scene_name": scene_name,
                    "scene_index": scene_index,
                },
            )
            scene_started = time.monotonic()
            log_path = run_dir / "logs" / f"{scene_name}.log"
            with log_path.open("xb") as log:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.scene_timeout_seconds,
                    check=False,
                )
            if completed.returncode != 0:
                raise PreprocessError(
                    f"{scene_name} preprocess returncode={completed.returncode}; 见 {log_path}"
                )
            validation = validate_processed_scene(staged_scene, frames)
            inventory = build_inventory(staged_scene)
            final_scene.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_scene, final_scene)
            scene_artifact = {
                "schema_version": "worldsim_v5_m1_processed_scene_v1",
                "task_id": config["task_id"],
                "scene_name": scene_name,
                "scene_index": scene_index,
                "status": "done",
                "command": command,
                "command_executed": True,
                "raw_manifest": str(raw_manifest_path),
                "raw_manifest_sha256": sha256_file(raw_manifest_path),
                "output": str(final_scene),
                "duration_seconds": time.monotonic() - scene_started,
                "validation": validation,
                "inventory": inventory,
                "inventory_sha256": hashlib.sha256(
                    json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "checkpoint": "N/A_data_preparation",
                "quality_read": False,
                "training_started": False,
                "model_inference_started": False,
            }
            artifact_path = run_dir / "artifacts" / f"{scene_name}.json"
            atomic_json(artifact_path, scene_artifact)
            results.append(
                {
                    "scene_name": scene_name,
                    "scene_index": scene_index,
                    "artifact": str(artifact_path),
                    "artifact_sha256": sha256_file(artifact_path),
                    "inventory_sha256": scene_artifact["inventory_sha256"],
                }
            )
            append_event(
                events,
                {
                    "at_utc": now_utc(),
                    "event": "scene_preprocess_complete",
                    "scene_name": scene_name,
                    "scene_index": scene_index,
                    "status": "done",
                },
            )
    except Exception as error:
        summary = {
            "schema_version": "worldsim_v5_m1_preprocess_summary_v1",
            "task_id": config["task_id"],
            "stage": "development_preprocess",
            "status": "blocked",
            "run_id": run_dir.name,
            "started_at_utc": started_at_utc,
            "finished_at_utc": now_utc(),
            "completed_scenes": results,
            "error": f"{type(error).__name__}: {error}",
            "staging_root": str(staging_root),
            "quality_read": False,
            "training_started": False,
            "model_inference_started": False,
        }
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(
            run_dir / "status.json",
            {
                "task_id": config["task_id"],
                "stage": "development_preprocess",
                "status": "blocked",
                "summary_sha256": sha256_file(run_dir / "summary.json"),
            },
        )
        raise

    summary = {
        "schema_version": "worldsim_v5_m1_preprocess_summary_v1",
        "task_id": config["task_id"],
        "stage": "development_preprocess",
        "status": "done",
        "run_id": run_dir.name,
        "started_at_utc": started_at_utc,
        "finished_at_utc": now_utc(),
        "duration_seconds": time.monotonic() - started,
        "scene_count": len(results),
        "scenes": results,
        "processed_root": str(final_root),
        "checkpoint": "N/A_data_preparation",
        "quality_read": False,
        "training_started": False,
        "model_inference_started": False,
        "project_git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "dirty": False,
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    fingerprint = {
        "config_sha256": sha256_file(config_path),
        "raw_batch_manifest_sha256": sha256_file(raw_batch_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "preprocess_wrapper_sha256": sha256_file(preprocess_wrapper),
        "upstream_preprocessor_sha256": sha256_file(
            UPSTREAM_ROOT / "datasets/nuscenes/nuscenes_preprocess.py"
        ),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": config["task_id"],
            "stage": "development_preprocess",
            "status": "done",
            "summary_sha256": fingerprint["summary_sha256"],
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_m1_preprocess_run_manifest_v1",
            "task_id": config["task_id"],
            "status": "done",
            "artifacts": artifacts,
            "quality_read": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/worldsim_v5/m1_structured_ownership_v1.yaml",
    )
    parser.add_argument(
        "--raw-batch-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/data/worldsim_v5/manifests/"
            "m1_development_raw_batch_v1.json"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--scene-timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

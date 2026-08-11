#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from motion_proj.worldsim_v4.datasets.nuscenes import (
    TASK_ID,
    CohortError,
    build_cohort,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)


SNAPSHOT_RELPATHS = (
    "motion_proj/worldsim_v4/datasets/nuscenes.py",
    "scripts/build_worldsim_v4_nuscenes_cohort.py",
    "tests/test_worldsim_v4_nuscenes_split.py",
    "configs/worldsim_v4/nuscenes_cohort_v1.yaml",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task_id") != TASK_ID:
        raise CohortError("D0 配置 task_id 非法")
    if payload.get("status") not in {"running", "done"}:
        raise CohortError("D0 配置状态只允许 running/done")
    return payload


def _git(project_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise CohortError(f"git {' '.join(args)} 失败：{proc.stderr.strip()}")
    return proc.stdout.strip()


def _validate_smoke_scene(scene: str, record: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(record["processed_root"]))
    if not root.is_dir():
        raise CohortError(f"smoke processed scene 不存在：{scene}: {root}")
    expected_frames = int(record["expected_frames"])
    expected_cameras = int(record["expected_cameras"])
    image_paths = sorted((root / "images").glob("*.jpg")) + sorted((root / "images").glob("*.png"))
    lidar_paths = sorted((root / "lidar").glob("*.bin"))
    if not lidar_paths:
        lidar_paths = sorted((root / "lidar").glob("*.ply"))
    expected_images = expected_frames * expected_cameras
    if len(image_paths) != expected_images:
        raise CohortError(f"{scene} image count {len(image_paths)} != {expected_images}")
    if len(lidar_paths) != expected_frames:
        raise CohortError(f"{scene} LiDAR count {len(lidar_paths)} != {expected_frames}")
    required = (
        root / "instances" / "instances_info.json",
        root / "instances" / "frame_instances.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise CohortError(f"{scene} preprocess artifact 缺失：{path}")
        json.loads(path.read_text(encoding="utf-8"))
    sample_paths = [image_paths[0], image_paths[-1], lidar_paths[0], lidar_paths[-1], *required]
    if any(path.stat().st_size == 0 for path in sample_paths):
        raise CohortError(f"{scene} smoke 发现空文件")
    return {
        "scene": scene,
        "status": "passed_existing_preprocess_reuse",
        "processed_root": str(root),
        "frames": expected_frames,
        "cameras": expected_cameras,
        "image_count": len(image_paths),
        "lidar_count": len(lidar_paths),
        "sample_artifacts": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sample_paths
        },
        "model_training": False,
        "quality_read": False,
    }


def _record_blocked_terminal(
    config_path: Path,
    run_dir: Path,
    project_root: Path,
    error: CohortError,
) -> None:
    run_dir = run_dir.resolve()
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    snapshots: dict[str, Any] = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        if not source.is_file():
            continue
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        snapshots[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "at_utc": now,
        "event": "preflight_contract_blocked",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    (run_dir / "events.jsonl").write_bytes(canonical_json_bytes(event))
    fingerprint = {
        "schema_version": "worldsim_v4_d0_failure_fingerprint_v1",
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "source_snapshots": snapshots,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    try:
        git_state = {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": _git(project_root, "status", "--porcelain") != "",
        }
    except CohortError:
        git_state = {"head": "unknown", "branch": "unknown", "dirty": True}
    summary = {
        "schema_version": "worldsim_v4_d0_summary_v1",
        "task_id": TASK_ID,
        "status": "blocked",
        "finished_at_utc": now,
        "reason": "preflight_contract_failed",
        "error": event,
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "project_git": git_state,
        "no_training": True,
        "no_model_inference": True,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    status = {
        "task_id": TASK_ID,
        "status": "blocked",
        "finished_at_utc": now,
        "summary_sha256": sha256_file(run_dir / "summary.json"),
    }
    _write_json(run_dir / "status.json", status)
    artifact_records = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v4_d0_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "blocked",
            "artifacts": artifact_records,
            "no_training": True,
            "no_model_inference": True,
            "test_quality_read": False,
        },
    )


def run(config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise CohortError(f"run 目录已存在，禁止复用：{run_dir}")
    config = _load_config(config_path)
    dataset = config["dataset"]
    manifest, candidates = build_cohort(dataset["metadata_root"], config["protocol"])
    cohort_sha = canonical_json_sha256(manifest)
    expected_sha = config.get("freeze", {}).get("expected_cohort_sha256")
    expected_roles = config.get("freeze", {}).get("scene_roles")
    expected_records = config.get("freeze", {}).get("scene_records")
    if expected_sha is not None and cohort_sha != expected_sha:
        raise CohortError(f"cohort SHA 漂移：{cohort_sha} != {expected_sha}")
    actual_roles = {
        role: [row["scene"] for row in manifest["scenes"] if row["role"] == role]
        for role in ("development", "validation", "test")
    }
    if expected_roles is not None and actual_roles != expected_roles:
        raise CohortError("frozen scene_roles 与重建结果不一致")
    if expected_records is not None and expected_records != manifest["scenes"]:
        raise CohortError("frozen scene_records 与重建结果不一致")
    formal = (
        expected_sha is not None
        and expected_roles is not None
        and expected_records is not None
        and config["status"] == "done"
    )
    smoke_rows = [
        _validate_smoke_scene(scene, record)
        for scene, record in config.get("smoke", {}).get("processed_scenes", {}).items()
    ]
    smoke_scene_names = {row["scene"] for row in smoke_rows}
    development_names = set(actual_roles["development"])
    if smoke_scene_names and (
        len(smoke_scene_names) != 2 or not smoke_scene_names.issubset(development_names)
    ):
        raise CohortError("D0 smoke 必须恰好使用两个 development scenes")

    run_dir.mkdir(parents=True)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    cohort_path = artifacts / "nuscenes_cohort.json"
    cohort_path.write_bytes(canonical_json_bytes(manifest))
    candidates_path = artifacts / "nuscenes_candidates.jsonl"
    with candidates_path.open("wb") as handle:
        for row in sorted(candidates, key=lambda item: item["scene"]):
            summary = {
                key: row[key]
                for key in (
                    "scene",
                    "scene_token",
                    "official_split",
                    "location",
                    "time_of_day",
                    "weather",
                    "road_geometry",
                    "actor_class",
                    "speed_regime",
                    "distance_regime",
                    "occlusion",
                    "donor_support",
                    "eligible_actor_count",
                    "sensor_contract_complete",
                    "sample_count",
                )
            }
            handle.write(canonical_json_bytes(summary))
    smoke_path = artifacts / "smoke.json"
    _write_json(
        smoke_path,
        {
            "status": "passed" if len(smoke_rows) == 2 else "not_run",
            "scenes": smoke_rows,
            "model_training": False,
            "quality_read": False,
        },
    )
    (artifacts / "split.sha256").write_text(
        f"{cohort_sha}  nuscenes_cohort.json\n", encoding="utf-8"
    )
    shutil.copy2(config_path, run_dir / "resolved.yaml")

    snapshots: dict[str, Any] = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        if not source.is_file():
            raise CohortError(f"source snapshot 文件不存在：{source}")
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }

    now = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "worldsim_v4_d0_summary_v1",
        "task_id": TASK_ID,
        "status": "done" if formal else "diagnostic",
        "finished_at_utc": now,
        "cohort_sha256": cohort_sha,
        "scene_counts": manifest["scene_counts"],
        "scene_roles": actual_roles,
        "candidate_scene_count": manifest["candidate_scene_count"],
        "metadata_fingerprints": manifest["metadata_fingerprints"],
        "smoke": json.loads(smoke_path.read_text(encoding="utf-8")),
        "source_snapshots": snapshots,
        "project_git": {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": _git(project_root, "status", "--porcelain") != "",
        },
        "no_training": True,
        "no_model_inference": True,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    status = {
        "task_id": TASK_ID,
        "status": summary["status"],
        "finished_at_utc": now,
        "summary_sha256": sha256_file(run_dir / "summary.json"),
    }
    _write_json(run_dir / "status.json", status)
    artifact_records = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    run_manifest = {
        "schema_version": "worldsim_v4_d0_run_manifest_v1",
        "task_id": TASK_ID,
        "status": summary["status"],
        "artifacts": artifact_records,
        "no_training": True,
        "no_model_inference": True,
        "test_quality_read": False,
    }
    _write_json(run_dir / "manifest.json", run_manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 WorldSim V4 30-scene nuScenes cohort")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    try:
        summary = run(args.config, args.run_dir, args.project_root)
    except CohortError as error:
        _record_blocked_terminal(args.config, args.run_dir, args.project_root, error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

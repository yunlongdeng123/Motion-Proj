#!/usr/bin/env python3
"""一次扫描官方 nuScenes shards，提取 B0 缺失的三个场景。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/baseline_data_v1.yaml",
    "configs/worldsim_v4/nuscenes_cohort_v1.yaml",
    "scripts/prepare_worldsim_v4_baseline_data.py",
    "scripts/prepare_dr_v2_drivestudio_scene.py",
    "scripts/build_adgs_nuscenes_assets.py",
    "tests/test_prepare_worldsim_v4_baseline_data.py",
)


class BaselineDataError(RuntimeError):
    """B0 数据准备合同失败。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaselineDataError(f"配置根节点必须为 mapping：{path}")
    return value


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BaselineDataError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_config(config: Mapping[str, Any], cohort: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v4_baseline_data_v1" or config.get("task_id") != TASK_ID:
        raise BaselineDataError("B0 data config schema/task 漂移")
    if config.get("status") != "running":
        raise BaselineDataError("B0 data config 必须保持 running")
    protocol = config.get("protocol", {})
    expected_sensors = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK", "LIDAR_TOP"]
    if protocol.get("sensors") != expected_sensors or protocol.get("no_download") is not True or protocol.get("test_quality_read") is not False:
        raise BaselineDataError("sensor/no-download/test-unread 合同漂移")
    scenes = config.get("scenes", {})
    frozen = cohort.get("freeze", {}).get("scene_roles", {}).get("development", [])
    if len(scenes) != 6 or set(scenes) != set(frozen):
        raise BaselineDataError("data scenes 必须精确匹配 D0 development")
    extracting = sorted(scene for scene, row in scenes.items() if row.get("state") == "extract_and_preprocess")
    if len(extracting) != int(config["gates"]["expected_extract_scene_count"]):
        raise BaselineDataError("extract scene count 漂移")
    if set(extracting) != {"scene-0048", "scene-0994", "scene-0139"}:
        raise BaselineDataError("只允许提取三个冻结缺失场景")
    return {"scenes": list(scenes), "extract_scenes": extracting}


def _resource_state(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    memory = {}
    for name in ("memory.max", "memory.current", "memory.events"):
        source = Path("/sys/fs/cgroup") / name
        if source.is_file():
            memory[name] = source.read_text(encoding="utf-8").strip()
    return {"disk_free_bytes": usage.free, "cgroup": memory}


def _git(project_root: Path, *args: str) -> str:
    process = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def run_extract(config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise BaselineDataError(f"run 目录已存在，禁止复用：{run_dir}")
    config = _load_yaml(config_path)
    cohort = _load_yaml(project_root / "configs/worldsim_v4/nuscenes_cohort_v1.yaml")
    validated = validate_config(config, cohort)
    dataset = config["dataset"]
    meta_root = Path(dataset["metadata_root"])
    metadata = meta_root / "v1.0-trainval"
    tar_root = Path(dataset["sensor_archive_root"])
    raw_root = Path(dataset["raw_union_root"])
    manifest_root = Path(dataset["raw_manifest_root"])
    if not metadata.is_dir() or not tar_root.is_dir():
        raise BaselineDataError("官方 metadata 或 sensor archive 不存在")
    if not all((tar_root / f"v1.0-trainval{index:02d}_blobs.tgz").is_file() for index in range(1, 11)):
        raise BaselineDataError("nuScenes 10 个官方 blobs shard 不完整")
    pre = _resource_state(Path("/root/autodl-tmp"))
    minimum = int(config["gates"]["minimum_disk_free_gib"]) * 2**30
    if pre["disk_free_bytes"] < minimum:
        raise BaselineDataError(f"可用磁盘低于 {minimum} bytes")

    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    scene_helper = _load_script(project_root / "scripts/prepare_dr_v2_drivestudio_scene.py", "worldsim_v4_scene_prepare_helper")
    archive_helper = scene_helper.load_asset_module(project_root)
    payloads = {scene: scene_helper.collect_required(metadata, scene) for scene in validated["extract_scenes"]}
    required = {row["filename"] for payload in payloads.values() for row in payload["sample_data"]}
    if len(required) != sum(len(payload["sample_data"]) for payload in payloads.values()):
        raise BaselineDataError("三个场景出现重复 sensor filename")
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_helper.link_metadata(metadata, raw_root)
    auxiliary = archive_helper.link_auxiliary_files(meta_root, raw_root)
    started = time.monotonic()
    index_path = manifest_root / "worldsim_v4_b0_missing3_member_shards.json"
    mapping, extracted = archive_helper.scan_shards(
        tar_dir=tar_root,
        members=required,
        index_path=index_path,
        dst=raw_root,
        workers=int(config["protocol"]["extraction_workers"]),
    )
    scene_manifests: dict[str, Any] = {}
    for scene, payload in payloads.items():
        files = []
        for row in payload["sample_data"]:
            name = row["filename"]
            path = raw_root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise BaselineDataError(f"提取后文件缺失：{name}")
            files.append({"filename": name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "shard": mapping[name], "extracted_this_run": name in extracted})
        scene_manifest = {
            "schema_version": "worldsim_v4_raw_scene_manifest_v1",
            "task_id": TASK_ID,
            "scene_name": scene,
            "scene_index": int(config["scenes"][scene]["scene_index"]),
            "scene_token": payload["scene_token"],
            "sample_count": payload["sample_count"],
            "sensor_counts": payload["sensor_counts"],
            "raw_root": str(raw_root),
            "required_count": len(files),
            "files": files,
        }
        manifest_path = manifest_root / f"{scene}_raw_manifest_v4.json"
        _write_json(manifest_path, scene_manifest)
        scene_manifests[scene] = {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "required_count": len(files)}
    elapsed = time.monotonic() - started
    inventory = {
        "schema_version": "worldsim_v4_raw_union_inventory_v1",
        "task_id": TASK_ID,
        "status": "done",
        "raw_root": str(raw_root),
        "scene_manifests": scene_manifests,
        "union_required_count": len(required),
        "extracted_this_run_count": len(extracted),
        "member_index": {"path": str(index_path), "bytes": index_path.stat().st_size, "sha256": sha256_file(index_path)},
        "auxiliary": auxiliary,
        "duration_seconds": elapsed,
        "resource_pre": pre,
        "resource_post": _resource_state(Path("/root/autodl-tmp")),
        "download_attempted": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "artifacts/raw_union_inventory.json", inventory)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        if not source.is_file():
            raise BaselineDataError(f"source snapshot 缺失：{relpath}")
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    fingerprint = {
        "config_sha256": sha256_file(config_path),
        "cohort_sha256": sha256_file(project_root / "configs/worldsim_v4/nuscenes_cohort_v1.yaml"),
        "raw_inventory_sha256": sha256_file(run_dir / "artifacts/raw_union_inventory.json"),
        "source_snapshots": snapshots,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    now = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "events.jsonl", {"at_utc": now, "event": "missing_scene_raw_extract_complete", "status": "done"})
    summary = {
        "schema_version": "worldsim_v4_b0_data_summary_v1",
        "task_id": TASK_ID,
        "stage": "extract_missing3",
        "status": "done",
        "finished_at_utc": now,
        "scenes": validated["extract_scenes"],
        "union_required_count": len(required),
        "duration_seconds": elapsed,
        "raw_inventory_sha256": fingerprint["raw_inventory_sha256"],
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "project_git": {"head": _git(project_root, "rev-parse", "HEAD"), "branch": _git(project_root, "branch", "--show-current"), "dirty": bool(_git(project_root, "status", "--porcelain"))},
        "download_attempted": False,
        "training_started": False,
        "model_inference_started": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "stage": "extract_missing3", "status": "done", "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_b0_data_run_manifest_v1", "task_id": TASK_ID, "status": "done", "artifacts": artifacts, "download_attempted": False, "test_quality_read": False})
    return summary


def validate_processed_scene(scene_root: Path, expected_frames: int, expected_cameras: int) -> dict[str, Any]:
    if not scene_root.is_dir():
        raise BaselineDataError(f"processed scene 不存在：{scene_root}")
    images = sorted((scene_root / "images").glob("*.jpg")) + sorted((scene_root / "images").glob("*.png"))
    lidar = sorted((scene_root / "lidar").glob("*.bin")) + sorted((scene_root / "lidar").glob("*.ply"))
    required_json = [scene_root / "instances/instances_info.json", scene_root / "instances/frame_instances.json"]
    if len(images) != expected_frames * expected_cameras:
        raise BaselineDataError(f"image count {len(images)} != {expected_frames * expected_cameras}")
    if len(lidar) != expected_frames:
        raise BaselineDataError(f"lidar count {len(lidar)} != {expected_frames}")
    for path in required_json:
        if not path.is_file() or path.stat().st_size == 0:
            raise BaselineDataError(f"processed JSON 缺失：{path}")
        json.loads(path.read_text(encoding="utf-8"))
    samples = [images[0], images[-1], lidar[0], lidar[-1], *required_json]
    return {
        "scene_root": str(scene_root),
        "frames": expected_frames,
        "cameras": expected_cameras,
        "image_count": len(images),
        "lidar_count": len(lidar),
        "sample_artifacts": {str(path.relative_to(scene_root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in samples},
    }


def scene_directory_name(scene_index: int) -> str:
    if scene_index < 0:
        raise BaselineDataError("scene index 必须非负")
    return f"{scene_index:03d}"


def _ensure_reused_processed(config: Mapping[str, Any], processed_root: Path) -> list[dict[str, Any]]:
    destination = processed_root / "trainval"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene, record in config["scenes"].items():
        if record["state"] != "reuse_processed":
            continue
        source = Path(record["processed_source"]).resolve()
        target = destination / scene_directory_name(int(record["scene_index"]))
        if not source.is_dir():
            raise BaselineDataError(f"reuse processed source 缺失：{source}")
        if target.is_symlink():
            if target.resolve() != source:
                raise BaselineDataError(f"processed symlink 漂移：{target}")
        elif target.exists():
            raise BaselineDataError(f"processed target 已存在且不是 symlink：{target}")
        else:
            target.symlink_to(source, target_is_directory=True)
        rows.append({"scene": scene, "scene_index": int(record["scene_index"]), "source": str(source), "target": str(target), "target_resolved": str(target.resolve())})
    return rows


def run_preprocess(config_path: Path, run_dir: Path, project_root: Path, scene: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise BaselineDataError(f"run 目录已存在，禁止复用：{run_dir}")
    config = _load_yaml(config_path)
    cohort = _load_yaml(project_root / "configs/worldsim_v4/nuscenes_cohort_v1.yaml")
    validate_config(config, cohort)
    record = config["scenes"].get(scene)
    if not isinstance(record, Mapping) or record.get("state") != "extract_and_preprocess":
        raise BaselineDataError(f"scene 不属于待 preprocess 集合：{scene}")
    raw_root = Path(config["dataset"]["raw_union_root"])
    processor_save_dir = Path(config["dataset"]["processor_save_dir"])
    processed_root = Path(config["dataset"]["processed_root"])
    expected_processed_root = processor_save_dir.with_name(processor_save_dir.name + "_10Hz")
    if processed_root != expected_processed_root:
        raise BaselineDataError(f"processor save/output root 合同漂移：{processor_save_dir} -> {expected_processed_root} != {processed_root}")
    raw_manifest = Path(config["dataset"]["raw_manifest_root"]) / f"{scene}_raw_manifest_v4.json"
    if not raw_manifest.is_file():
        raise BaselineDataError(f"raw manifest 缺失：{raw_manifest}")
    raw_payload = json.loads(raw_manifest.read_text(encoding="utf-8"))
    for row in raw_payload.get("files", []):
        path = raw_root / row["filename"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise BaselineDataError(f"raw payload bytes 漂移：{path}")
    scene_index = int(record["scene_index"])
    scene_root = processed_root / "trainval" / scene_directory_name(scene_index)
    if scene_root.exists() or scene_root.is_symlink():
        raise BaselineDataError(f"processed scene target 已存在，禁止覆盖：{scene_root}")
    pre = _resource_state(Path("/root/autodl-tmp"))
    minimum = int(config["gates"]["minimum_disk_free_gib"]) * 2**30
    if pre["disk_free_bytes"] < minimum:
        raise BaselineDataError(f"可用磁盘低于 {minimum} bytes")
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    reused = _ensure_reused_processed(config, processed_root)
    command = [
        "/root/autodl-tmp/envs/drivestudio/bin/python",
        str(project_root / "scripts/preprocess_dr_v2_nuscenes_single.py"),
        "--data-root", str(raw_root),
        "--target-dir", str(processor_save_dir),
        "--scene-index", str(scene_index),
        "--upstream-root", "/root/autodl-tmp/third_party/drivestudio",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{project_root}:/root/autodl-tmp/third_party/drivestudio"
    started = time.monotonic()
    log_path = run_dir / "logs/preprocess.log"
    with log_path.open("xb") as log:
        process = subprocess.run(command, cwd=project_root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False)
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise BaselineDataError(f"preprocess return code={process.returncode}；见 {log_path}")
    validation = validate_processed_scene(scene_root, int(config["gates"]["expected_frames_10hz"]), int(config["gates"]["expected_cameras"]))
    artifact = {
        "schema_version": "worldsim_v4_processed_scene_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "scene_index": scene_index,
        "status": "done",
        "command": command,
        "duration_seconds": elapsed,
        "raw_manifest": {"path": str(raw_manifest), "bytes": raw_manifest.stat().st_size, "sha256": sha256_file(raw_manifest)},
        "reused_processed": reused,
        "validation": validation,
        "resource_pre": pre,
        "resource_post": _resource_state(Path("/root/autodl-tmp")),
        "training_started": False,
        "model_inference_started": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "artifacts/processed_scene.json", artifact)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    fingerprint = {"config_sha256": sha256_file(config_path), "raw_manifest_sha256": sha256_file(raw_manifest), "processed_artifact_sha256": sha256_file(run_dir / "artifacts/processed_scene.json"), "source_snapshots": snapshots}
    _write_json(run_dir / "fingerprint.json", fingerprint)
    now = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "events.jsonl", {"at_utc": now, "event": "scene_preprocess_complete", "scene": scene, "status": "done"})
    summary = {
        "schema_version": "worldsim_v4_b0_data_summary_v1",
        "task_id": TASK_ID,
        "stage": "preprocess_scene",
        "status": "done",
        "finished_at_utc": now,
        "scene": scene,
        "scene_index": scene_index,
        "duration_seconds": elapsed,
        "processed_artifact_sha256": fingerprint["processed_artifact_sha256"],
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "project_git": {"head": _git(project_root, "rev-parse", "HEAD"), "branch": _git(project_root, "branch", "--show-current"), "dirty": bool(_git(project_root, "status", "--porcelain"))},
        "training_started": False,
        "model_inference_started": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "stage": "preprocess_scene", "status": "done", "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_b0_data_run_manifest_v1", "task_id": TASK_ID, "status": "done", "artifacts": artifacts, "test_quality_read": False})
    return summary


def record_blocked(config_path: Path, run_dir: Path, project_root: Path, error: Exception, stage: str) -> None:
    """保留失败现场；永不把旧 run 倒写为成功。"""
    run_dir = run_dir.resolve()
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    now = datetime.now(timezone.utc).isoformat()
    event = {"at_utc": now, "event": f"{stage}_blocked", "error_type": type(error).__name__, "message": str(error)}
    with (run_dir / "events.jsonl").open("ab") as handle:
        handle.write(canonical_json_bytes(event))
    fingerprint = {
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "project_head": _git(project_root, "rev-parse", "HEAD"),
        "error": event,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    summary = {
        "schema_version": "worldsim_v4_b0_data_summary_v1",
        "task_id": TASK_ID,
        "stage": stage,
        "status": "blocked",
        "finished_at_utc": now,
        "reason": f"{stage}_failed",
        "error": event,
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "download_attempted": False,
        "training_started": False,
        "model_inference_started": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "stage": stage, "status": "blocked", "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_b0_data_run_manifest_v1", "task_id": TASK_ID, "status": "blocked", "artifacts": artifacts, "download_attempted": False, "test_quality_read": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 WorldSim V4 B0 nuScenes 输入")
    parser.add_argument("--stage", choices=["extract", "preprocess"], default="extract")
    parser.add_argument("--scene")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    failure_stage = "extract_missing3" if args.stage == "extract" else "preprocess_scene"
    try:
        if args.stage == "extract":
            summary = run_extract(args.config, args.run_dir, args.project_root)
        else:
            if not args.scene:
                raise BaselineDataError("preprocess stage 必须指定 --scene")
            summary = run_preprocess(args.config, args.run_dir, args.project_root, args.scene)
    except Exception as error:
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), args.project_root.resolve(), error, failure_stage)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

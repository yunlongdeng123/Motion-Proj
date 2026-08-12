#!/usr/bin/env python3
"""使用已审计本地 SegFormer 快照原子生成 V4 StreetGS sky masks。"""

from __future__ import annotations

import argparse
import hashlib
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


PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))
from scripts.prepare_worldsim_v4_baseline_data import scene_directory_name


DEFAULT_TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
TASK_SCENES = {
    DEFAULT_TASK_ID: {
        "scene-0048": 45,
        "scene-0139": 110,
        "scene-0994": 752,
    },
    "WS-V4-M1-EVIDENCE-FIELD-01": {
        "scene-0071": 68,
        "scene-0317": 251,
        "scene-0450": 364,
        "scene-0862": 652,
        "scene-1012": 770,
        "scene-1089": 829,
    },
}
TASK_FRAME_COUNTS = {
    DEFAULT_TASK_ID: {scene: 196 for scene in TASK_SCENES[DEFAULT_TASK_ID]},
    "WS-V4-M1-EVIDENCE-FIELD-01": {
        "scene-0071": 196,
        "scene-0317": 191,
        "scene-0450": 196,
        "scene-0862": 196,
        "scene-1012": 196,
        "scene-1089": 196,
    },
}
SOURCE_SNAPSHOT_RELPATHS = (
    "scripts/build_worldsim_v4_sky_masks.py",
    "tests/test_build_worldsim_v4_sky_masks.py",
    "tests/test_build_worldsim_v4_m1_validation_sky_masks.py",
)


class SkyMaskError(RuntimeError):
    """V4 sky-mask 合同失败。"""


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


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SkyMaskError("sky-mask config 根节点必须为 mapping")
    return value


def _git(project_root: Path, *args: str) -> str:
    process = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise SkyMaskError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def configured_task_id(config_path: Path) -> str:
    try:
        value = _load_yaml(config_path)
    except (OSError, UnicodeError, yaml.YAMLError, SkyMaskError):
        return DEFAULT_TASK_ID
    task_id = value.get("task_id")
    return str(task_id) if isinstance(task_id, str) and task_id else DEFAULT_TASK_ID


def source_snapshot_relpaths(config_path: Path, project_root: Path) -> tuple[str, ...]:
    try:
        config_relpath = config_path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise SkyMaskError("sky-mask config must be inside the project root") from error
    return (config_relpath, *SOURCE_SNAPSHOT_RELPATHS)


def expected_sky_counts(config: Mapping[str, Any], scene: str) -> tuple[int, int]:
    data = config.get("data", {})
    timestep_rows = data.get("expected_timesteps_by_scene")
    mask_rows = data.get("expected_masks_by_scene")
    try:
        if isinstance(timestep_rows, Mapping) and isinstance(mask_rows, Mapping):
            timesteps = int(timestep_rows[scene])
            masks = int(mask_rows[scene])
        else:
            timesteps = int(data["expected_timesteps"])
            masks = int(data["expected_masks"])
    except (KeyError, TypeError, ValueError) as error:
        raise SkyMaskError(f"scene sky count contract 缺失：{scene}") from error
    if timesteps <= 0 or masks != timesteps * len(data.get("cameras", [])):
        raise SkyMaskError(f"scene sky count contract 非法：{scene}: {timesteps}/{masks}")
    return timesteps, masks


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v4_sky_masks_v1" or config.get("status") != "running":
        raise SkyMaskError("sky-mask config schema/task/status 漂移")
    task_id = config.get("task_id")
    expected_scenes = TASK_SCENES.get(str(task_id))
    if expected_scenes is None:
        raise SkyMaskError("sky-mask config schema/task/status 漂移")
    model = config.get("model", {})
    if model.get("revision") != "2c6f153e4c23c229e2fa2b188eb250607e030cd8" or model.get("local_files_only") is not True:
        raise SkyMaskError("SegFormer revision/local-only 合同漂移")
    data = config.get("data", {})
    if data.get("cameras") != [0, 1, 2]:
        raise SkyMaskError("camera/timestep/mask 合同漂移")
    if data.get("scenes") != expected_scenes:
        raise SkyMaskError("sky-mask scene 集合漂移")
    expected_frames = TASK_FRAME_COUNTS[str(task_id)]
    if task_id == DEFAULT_TASK_ID:
        if data.get("expected_timesteps") != 196 or data.get("expected_masks") != 588:
            raise SkyMaskError("camera/timestep/mask 合同漂移")
    else:
        if data.get("expected_timesteps_by_scene") != expected_frames:
            raise SkyMaskError("sky-mask per-scene frame contract 漂移")
        expected_masks = {scene: frames * 3 for scene, frames in expected_frames.items()}
        if data.get("expected_masks_by_scene") != expected_masks:
            raise SkyMaskError("sky-mask per-scene mask contract 漂移")
    observed_frames = {
        scene: expected_sky_counts(config, scene)[0]
        for scene in expected_scenes
    }
    if observed_frames != expected_frames:
        raise SkyMaskError("sky-mask per-scene frame contract 漂移")
    runtime = config.get("runtime", {})
    if runtime.get("generation_network_access") is not False or runtime.get("no_test_quality_read") is not True:
        raise SkyMaskError("generation-network/test-unread 合同漂移")
    return {"scene_count": len(expected_scenes), "expected_masks": 588}


def audit_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["model"]["snapshot"])
    if not root.is_dir():
        raise SkyMaskError(f"本地模型快照不存在：{root}")
    rows = {}
    for name, expected in config["model"]["required_files"].items():
        path = root / name
        if not path.is_file():
            raise SkyMaskError(f"模型文件不存在：{path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise SkyMaskError(f"模型文件漂移：{name}: {actual} != {expected}")
        rows[name] = {"path": str(path), **actual}
    return {"snapshot": str(root), "revision": config["model"]["revision"], "files": rows}


def _memory_events() -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8").splitlines())}


def _gpu() -> dict[str, Any]:
    process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    fields = [value.strip() for value in process.stdout.strip().split(",")]
    if process.returncode != 0 or len(fields) != 4:
        raise SkyMaskError("nvidia-smi 采样失败")
    return {"name": fields[0], "memory_total_mib": int(fields[1]), "memory_used_mib": int(fields[2]), "utilization_percent": int(fields[3])}


def resource_sample(event: str, completed_masks: int) -> dict[str, Any]:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {"at_utc": datetime.now(timezone.utc).isoformat(), "event": event, "completed_masks": completed_masks, "memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()), "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(), "memory_events": _memory_events(), "disk_free_bytes": disk.free, "gpu": _gpu()}


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(payload))


def validate_output_paths(target: Path, partial: Path) -> bool:
    """允许 preprocess 预建的空目录；任何已有产物仍 fail-closed。"""
    if partial.exists():
        raise SkyMaskError(f"sky-mask partial 已存在：{partial}")
    if not target.exists():
        return False
    if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
        raise SkyMaskError(f"sky-mask target 已存在且非空：{target}")
    return True


def run(config_path: Path, run_dir: Path, project_root: Path, scene: str) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise SkyMaskError(f"run 目录已存在，禁止复用：{run_dir}")
    config = _load_yaml(config_path)
    validate_config(config)
    task_id = str(config["task_id"])
    snapshot_relpaths = source_snapshot_relpaths(config_path, project_root)
    if scene not in config["data"]["scenes"]:
        raise SkyMaskError(f"scene 未冻结：{scene}")
    snapshot = audit_snapshot(config)
    scene_index = int(config["data"]["scenes"][scene])
    expected_timesteps, expected_masks = expected_sky_counts(config, scene)
    scene_root = Path(config["data"]["processed_root"]) / scene_directory_name(scene_index)
    images = sorted(path for path in (scene_root / "images").glob("*.jpg") if int(path.stem.rsplit("_", 1)[1]) in set(config["data"]["cameras"]))
    if len(images) != expected_masks:
        raise SkyMaskError(f"训练相机 image count {len(images)} != {expected_masks}")
    target = scene_root / "sky_masks"
    partial = scene_root / f"sky_masks.partial.{run_dir.name}"
    precreated_empty_target = validate_output_paths(target, partial)
    pre = resource_sample("preflight", 0)
    if int(pre["gpu"]["memory_used_mib"]) > int(config["runtime"]["maximum_gpu_used_at_start_mib"]):
        raise SkyMaskError(f"GPU 非空闲：{pre['gpu']}")
    if int(pre["disk_free_bytes"]) < int(config["runtime"]["minimum_disk_free_gib"]) * 2**30:
        raise SkyMaskError("磁盘余量不足")
    for name in ("artifacts", "source_snapshot"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    for relpath in snapshot_relpaths:
        source = project_root / relpath
        destination = run_dir / "source_snapshot" / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    append_jsonl(run_dir / "resource.jsonl", pre)
    partial.mkdir()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(config["model"]["snapshot"])
    processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(root, local_files_only=True).to(config["runtime"]["device"])
    model.eval()
    labels = {int(key): str(value) for key, value in model.config.id2label.items()}
    sky_ids = [key for key, value in labels.items() if value.strip().lower() == "sky"]
    if len(sky_ids) != 1:
        raise SkyMaskError(f"sky class 不唯一：{sky_ids}")
    sky_id = sky_ids[0]
    rows = []
    peak_gpu = int(pre["gpu"]["memory_used_mib"])
    peak_memory = int(pre["memory_current_bytes"])
    started = time.monotonic()
    interval = int(config["runtime"]["resource_sample_every_masks"])
    for index, image_path in enumerate(images, 1):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(config["runtime"]["device"]) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            mask = logits.argmax(dim=1)[0].eq(sky_id).cpu().numpy().astype(np.uint8) * 255
        output = partial / f"{image_path.stem}.png"
        Image.fromarray(mask).save(output)
        if output.stat().st_size == 0:
            raise SkyMaskError(f"空 sky mask：{output}")
        rows.append({"image": image_path.name, "mask": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output), "sky_fraction": float((mask > 0).mean())})
        if index % interval == 0 or index == len(images):
            torch.cuda.synchronize()
            sample = resource_sample("running", index)
            append_jsonl(run_dir / "resource.jsonl", sample)
            peak_gpu = max(peak_gpu, int(sample["gpu"]["memory_used_mib"]))
            peak_memory = max(peak_memory, int(sample["memory_current_bytes"]))
    if len(rows) != expected_masks or {row["mask"] for row in rows} != {f"{path.stem}.png" for path in images}:
        raise SkyMaskError("sky-mask 输出集合漂移")
    manifest = {"schema_version": "worldsim_v4_sky_mask_manifest_v1", "task_id": task_id, "status": "done", "scene": scene, "scene_index": scene_index, "expected_timesteps": expected_timesteps, "expected_masks": expected_masks, "model": snapshot, "sky_class_id": sky_id, "sky_class_label": labels[sky_id], "image_count": len(images), "mask_count": len(rows), "mean_sky_fraction": float(np.mean([row["sky_fraction"] for row in rows])), "files": rows, "network_accessed": False, "test_quality_read": False}
    _write_json(run_dir / "artifacts/sky_mask_manifest.json", manifest)
    del model, processor, inputs, logits
    gc_collect = getattr(torch.cuda, "empty_cache", None)
    if gc_collect:
        gc_collect()
    if precreated_empty_target:
        target.rmdir()
    os.replace(partial, target)
    elapsed = time.monotonic() - started
    post = resource_sample("completed", len(rows))
    append_jsonl(run_dir / "resource.jsonl", post)
    project_git = {"head": _git(project_root, "rev-parse", "HEAD"), "branch": _git(project_root, "branch", "--show-current"), "dirty": bool(_git(project_root, "status", "--porcelain"))}
    fingerprint = {"config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(run_dir / "artifacts/sky_mask_manifest.json"), "model": snapshot, "project_git": project_git, "source_snapshots": {relpath: sha256_file(run_dir / "source_snapshot" / relpath) for relpath in snapshot_relpaths}}
    _write_json(run_dir / "fingerprint.json", fingerprint)
    now = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "events.jsonl", {"at_utc": now, "event": "sky_masks_complete", "scene": scene, "status": "done"})
    summary = {"schema_version": "worldsim_v4_sky_mask_summary_v1", "task_id": task_id, "status": "done", "scene": scene, "scene_index": scene_index, "finished_at_utc": now, "duration_seconds": elapsed, "mask_count": len(rows), "mean_sky_fraction": manifest["mean_sky_fraction"], "manifest_sha256": fingerprint["manifest_sha256"], "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "resources": {"peak_gpu_memory_mib": peak_gpu, "peak_cgroup_memory_bytes": peak_memory}, "project_git": project_git, "network_accessed": False, "test_quality_read": False}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": task_id, "status": "done", "scene": scene, "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_sky_mask_run_manifest_v1", "task_id": task_id, "status": "done", "scene": scene, "artifacts": artifacts, "network_accessed": False, "test_quality_read": False})
    return summary


def record_blocked(config_path: Path, run_dir: Path, scene: str, error: BaseException) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    task_id = configured_task_id(config_path)
    now = datetime.now(timezone.utc).isoformat()
    event = {"at_utc": now, "event": "sky_masks_blocked", "scene": scene, "error_type": type(error).__name__, "message": str(error)}
    append_jsonl(run_dir / "events.jsonl", event)
    _write_json(run_dir / "fingerprint.json", {"config_sha256": sha256_file(config_path) if config_path.is_file() else None, "error": event})
    summary = {"schema_version": "worldsim_v4_sky_mask_summary_v1", "task_id": task_id, "status": "blocked", "scene": scene, "finished_at_utc": now, "reason": "sky_mask_stage_failed", "error": event, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "network_accessed": False, "test_quality_read": False}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": task_id, "status": "blocked", "scene": scene, "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_sky_mask_run_manifest_v1", "task_id": task_id, "status": "blocked", "scene": scene, "artifacts": artifacts, "network_accessed": False, "test_quality_read": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 WorldSim V4 本地 sky masks")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        summary = run(args.config, args.run_dir, args.project_root, args.scene)
    except BaseException as error:
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), args.scene, error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

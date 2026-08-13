#!/usr/bin/env python3
"""资源守卫下训练单个 WorldSim V4 StreetGS scene。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from scripts.prepare_worldsim_v4_baseline_data import scene_directory_name, validate_processed_scene


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
M1_TASK_ID = "WS-V4-M1-EVIDENCE-FIELD-01"
M3_TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"
SCENE_CONTRACTS = {
    TASK_ID: {
        "scene-0230": 179,
        "scene-0242": 191,
        "scene-0255": 204,
        "scene-0048": 45,
        "scene-0994": 752,
        "scene-0139": 110,
    },
    M1_TASK_ID: {
        "scene-0071": 68,
        "scene-1089": 829,
        "scene-0317": 251,
        "scene-0862": 652,
        "scene-1012": 770,
        "scene-0450": 364,
    },
    M3_TASK_ID: {
        "scene-0919": 704,
        "scene-0100": 82,
        "scene-0520": 410,
        "scene-0634": 488,
        "scene-1062": 802,
        "scene-0626": 482,
        "scene-0015": 14,
        "scene-0552": 436,
        "scene-0924": 709,
        "scene-0906": 692,
        "scene-0519": 409,
        "scene-0781": 604,
        "scene-1072": 812,
        "scene-0554": 438,
        "scene-0911": 697,
        "scene-0966": 731,
        "scene-0800": 620,
        "scene-0632": 486,
    },
}
FRAME_CONTRACTS = {
    TASK_ID: {scene: 196 for scene in SCENE_CONTRACTS[TASK_ID]},
    M1_TASK_ID: {
        "scene-0071": 196,
        "scene-0317": 191,
        "scene-0450": 196,
        "scene-0862": 196,
        "scene-1012": 196,
        "scene-1089": 196,
    },
    M3_TASK_ID: {
        "scene-0919": 201,
        "scene-0100": 196,
        "scene-0520": 201,
        "scene-0634": 196,
        "scene-1062": 196,
        "scene-0626": 196,
        "scene-0015": 196,
        "scene-0552": 201,
        "scene-0924": 196,
        "scene-0906": 201,
        "scene-0519": 201,
        "scene-0781": 196,
        "scene-1072": 196,
        "scene-0554": 201,
        "scene-0911": 196,
        "scene-0966": 201,
        "scene-0800": 196,
        "scene-0632": 196,
    },
}
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/streetgs_training_v1.yaml",
    "configs/worldsim_v4/m1_validation_reconstruction_v1.yaml",
    "configs/worldsim_v4/m3_test_reconstruction_v1.yaml",
    "configs/worldsim_v4/baseline_data_v1.yaml",
    "scripts/run_worldsim_v4_streetgs_scene.py",
    "scripts/prepare_worldsim_v3_drivestudio.py",
    "compatibility/DriveStudio-2026-08-05.patch",
    "compatibility/DriveStudio-2026-08-13-m3-test-timeline.patch",
    "motion_proj/worldsim_v3/drivestudio_compat.py",
    "tests/test_run_worldsim_v4_streetgs_scene.py",
)
ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None


class StreetGSTrainingError(RuntimeError):
    """StreetGS 训练合同或资源门失败。"""


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
        raise StreetGSTrainingError(f"配置根节点必须为 mapping：{path}")
    return value


def _git(path: Path, *args: str) -> str:
    process = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise StreetGSTrainingError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def expected_scene_frames(config: Mapping[str, Any], scene: str) -> int:
    data = config.get("data", {})
    rows = data.get("expected_frames_by_scene")
    try:
        frames = int(rows[scene]) if isinstance(rows, Mapping) else int(data["expected_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise StreetGSTrainingError(f"scene frame contract 缺失：{scene}") from error
    if frames <= 0:
        raise StreetGSTrainingError(f"scene frame contract 非法：{scene}: {frames}")
    return frames


def validate_config(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    task_id = config.get("task_id")
    if (
        config.get("schema_version") != "worldsim_v4_streetgs_training_v1"
        or task_id not in SCENE_CONTRACTS
        or config.get("status") != "running"
    ):
        raise StreetGSTrainingError("StreetGS config schema/task/status 漂移")
    implementation = config.get("implementation", {})
    patch = project_root / str(implementation.get("compatibility_patch"))
    if not patch.is_file() or sha256_file(patch) != implementation.get("compatibility_patch_sha256"):
        raise StreetGSTrainingError("DriveStudio compatibility patch 漂移")
    if config.get("training", {}).get("seed") != 0 or config.get("training", {}).get("modes") != {"profile100": 100, "formal": 30000}:
        raise StreetGSTrainingError("seed/iteration 合同漂移")
    partition = config.get("data", {}).get("frame_partition", {})
    if partition != {
        "modulus": 5,
        "development_remainder": 2,
        "heldout_remainder": 4,
        "train_remainders": [0, 1, 3],
    }:
        raise StreetGSTrainingError("frame partition 合同漂移")
    if int(config.get("data", {}).get("test_image_stride", -1)) != 0:
        raise StreetGSTrainingError("matched partition 禁止 stride split")
    scenes = config.get("scenes", {})
    if scenes != SCENE_CONTRACTS[task_id]:
        raise StreetGSTrainingError("冻结场景 index 合同漂移")
    observed_frames = {
        scene: expected_scene_frames(config, scene)
        for scene in scenes
    }
    if observed_frames != FRAME_CONTRACTS[task_id]:
        raise StreetGSTrainingError("冻结场景 frame 合同漂移")
    return {
        "task_id": task_id,
        "scene_count": len(scenes),
        "patch_sha256": sha256_file(patch),
    }


def _memory_value(path: str) -> int | None:
    raw = Path(path).read_text(encoding="utf-8").strip()
    return None if raw == "max" else int(raw)


def _memory_events() -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8").splitlines())}


def _gpu() -> dict[str, Any]:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = [value.strip() for value in process.stdout.strip().split(",")]
    if process.returncode != 0 or len(fields) != 5:
        raise StreetGSTrainingError("nvidia-smi GPU sample 失败")
    return {"name": fields[0], "driver": fields[1], "memory_total_mib": int(fields[2]), "memory_used_mib": int(fields[3]), "utilization_percent": int(fields[4])}


def resource_sample(stage: str, event: str) -> dict[str, Any]:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "event": event,
        "memory_current_bytes": _memory_value("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": _memory_value("/sys/fs/cgroup/memory.max"),
        "memory_events": _memory_events(),
        "disk_free_bytes": disk.free,
        "gpu": _gpu(),
    }


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(payload))


def preflight_resource(sample: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    limits = config["resources"]
    gpu = sample["gpu"]
    if int(gpu["memory_total_mib"]) < int(limits["minimum_gpu_memory_mib"]):
        raise StreetGSTrainingError(f"GPU total memory 不足：{gpu}")
    if int(gpu["memory_used_mib"]) > int(limits["maximum_gpu_used_at_start_mib"]):
        raise StreetGSTrainingError(f"GPU 非空闲：{gpu}")
    maximum = sample["memory_max_bytes"]
    if maximum is None or int(maximum) < int(limits["minimum_cgroup_memory_gib"]) * 2**30:
        raise StreetGSTrainingError("cgroup memory 不足")
    if int(sample["disk_free_bytes"]) < int(limits["minimum_disk_free_at_start_gib"]) * 2**30:
        raise StreetGSTrainingError("训练前磁盘不足")


def build_train_command(config: Mapping[str, Any], scene: str, mode: str, run_dir: Path) -> tuple[list[str], Path, int]:
    iterations = int(config["training"]["modes"][mode])
    scene_index = int(config["scenes"][scene])
    project = "worldsim_v4_streetgs"
    run_name = f"{scene.replace('-', '')}_{mode}_s0"
    output_root = run_dir / "work_dirs"
    checkpoint = output_root / project / run_name / "checkpoint_final.pth"
    partition = config["data"]["frame_partition"]
    excluded_remainders = [
        int(partition["development_remainder"]),
        int(partition["heldout_remainder"]),
    ]
    command = [
        str(Path(config["implementation"]["environment"]) / "bin/python"),
        "tools/train.py",
        "--config_file", str(config["implementation"]["config_file"]),
        "--output_root", str(output_root),
        "--project", project,
        "--run_name", run_name,
        f"dataset={config['data']['dataset_config']}",
        f"data.data_root={config['data']['processed_root']}",
        f"data.scene_idx={scene_index}",
        f"data.start_timestep={config['data']['start_timestep']}",
        f"data.end_timestep={config['data']['end_timestep']}",
        f"data.pixel_source.load_smpl={str(bool(config['data']['load_smpl'])).lower()}",
        f"data.pixel_source.test_image_stride={int(config['data']['test_image_stride'])}",
        f"+data.pixel_source.partition_modulus={int(partition['modulus'])}",
        "+data.pixel_source.excluded_remainders=" + json.dumps(excluded_remainders, separators=(",", ":")),
        f"trainer.optim.num_iters={iterations}",
        f"logging.saveckpt_freq={iterations}",
        "logging.vis_freq=-1",
        "render.render_full=false",
        "render.render_test=false",
        "render.render_novel=null",
    ]
    return command, checkpoint, iterations


def checkpoint_contract(path: Path, expected_step: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise StreetGSTrainingError(f"checkpoint 缺失：{path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping) or int(state.get("step", -1)) != expected_step:
        raise StreetGSTrainingError("checkpoint step/schema 漂移")
    models = state.get("models", {})
    if not isinstance(models, Mapping):
        raise StreetGSTrainingError("checkpoint models 缺失")
    counts = {}
    finite = True
    for name in ("Background", "RigidNodes"):
        model = models.get(name, {})
        means = model.get("_means") if isinstance(model, Mapping) else None
        counts[name] = int(means.shape[0]) if isinstance(means, torch.Tensor) else None
        if isinstance(means, torch.Tensor):
            finite = finite and bool(torch.isfinite(means).all().item())
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "step": expected_step, "model_keys": sorted(models), "gaussian_counts": counts, "means_finite": finite}
    del state, models
    gc.collect()
    if not finite:
        raise StreetGSTrainingError("checkpoint Gaussian means 非有限")
    return result


def run(config_path: Path, run_dir: Path, project_root: Path, scene: str, mode: str) -> dict[str, Any]:
    global ACTIVE_PROCESS
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise StreetGSTrainingError(f"run 目录已存在，禁止复用：{run_dir}")
    config = _load_yaml(config_path)
    validated = validate_config(config, project_root)
    task_id = str(validated["task_id"])
    if scene not in config["scenes"] or mode not in config["training"]["modes"]:
        raise StreetGSTrainingError("scene/mode 未冻结")
    upstream = Path(config["implementation"]["upstream_root"])
    if _git(upstream, "rev-parse", "HEAD") != config["implementation"]["upstream_commit"] or _git(upstream, "status", "--short") != config["implementation"]["expected_git_status"]:
        raise StreetGSTrainingError("patched DriveStudio HEAD/status 漂移")
    patch = project_root / config["implementation"]["compatibility_patch"]
    reverse = subprocess.run(["git", "-C", str(upstream), "apply", "--reverse", "--check", str(patch)], capture_output=True, text=True, check=False)
    if reverse.returncode != 0:
        raise StreetGSTrainingError("DriveStudio patch reverse-check 失败")
    scene_index = int(config["scenes"][scene])
    processed = Path(config["data"]["processed_root"]) / scene_directory_name(scene_index)
    data_validation = validate_processed_scene(
        processed,
        expected_scene_frames(config, scene),
        int(config["data"]["expected_cameras"]),
    )

    for name in ("artifacts", "logs", "source_snapshot", "stages", "work_dirs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    stage_name = f"train_{mode}"
    pre = resource_sample(stage_name, "preflight")
    append_jsonl(run_dir / "resource.jsonl", pre)
    preflight_resource(pre, config)
    command, checkpoint, iterations = build_train_command(config, scene, mode, run_dir)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": f"{project_root}:{upstream}",
        "WANDB_MODE": "disabled",
        "HF_HOME": "/root/autodl-tmp/hf_cache",
        "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
        "TORCH_HOME": "/root/autodl-tmp/cache/torch",
        "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
        "TMPDIR": "/root/autodl-tmp/tmp",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    })
    log_path = run_dir / "logs" / f"{stage_name}.log"
    baseline_events = pre["memory_events"]
    peak_gpu = int(pre["gpu"]["memory_used_mib"])
    peak_memory = int(pre["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    started = time.monotonic()
    with log_path.open("xb") as log:
        ACTIVE_PROCESS = subprocess.Popen(command, cwd=upstream, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        interval = float(config["resources"]["sample_interval_seconds"])
        timeout = float(config["resources"]["timeout_seconds"][mode])
        while ACTIVE_PROCESS.poll() is None:
            time.sleep(interval)
            sample = resource_sample(stage_name, "running")
            append_jsonl(run_dir / "resource.jsonl", sample)
            peak_gpu = max(peak_gpu, int(sample["gpu"]["memory_used_mib"]))
            peak_memory = max(peak_memory, int(sample["memory_current_bytes"] or 0))
            maximum, current = sample["memory_max_bytes"], sample["memory_current_bytes"]
            over_memory = over_memory + 1 if maximum and current and current / maximum >= float(config["resources"]["stop_cgroup_ratio"]) else 0
            events = sample["memory_events"]
            if over_memory >= int(config["resources"]["stop_cgroup_consecutive_samples"]):
                stop_reason = "cgroup_memory_ratio"
            elif events.get("oom", 0) > baseline_events.get("oom", 0) or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0):
                stop_reason = "cgroup_oom_event"
            elif int(sample["disk_free_bytes"]) < int(config["resources"]["stop_disk_free_gib"]) * 2**30:
                stop_reason = "disk_free"
            elif time.monotonic() - started > timeout:
                stop_reason = "timeout"
            if stop_reason:
                os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
                try:
                    ACTIVE_PROCESS.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(ACTIVE_PROCESS.pid, signal.SIGKILL)
                break
        return_code = ACTIVE_PROCESS.wait()
        ACTIVE_PROCESS = None
    elapsed = time.monotonic() - started
    append_jsonl(run_dir / "resource.jsonl", resource_sample(stage_name, "completed"))
    invalid_configuration = "invalid configuration argument" in log_path.read_text(encoding="utf-8", errors="replace")
    if return_code != 0 or stop_reason or invalid_configuration:
        raise StreetGSTrainingError(f"training blocked: rc={return_code} stop={stop_reason} invalid_config={invalid_configuration}")
    checkpoint_row = checkpoint_contract(checkpoint, iterations)
    stage = {"stage": stage_name, "status": "done", "scene": scene, "scene_index": scene_index, "mode": mode, "iterations": iterations, "duration_seconds": elapsed, "command": command, "checkpoint": checkpoint_row, "peak_gpu_memory_mib": peak_gpu, "peak_cgroup_memory_bytes": peak_memory, "stop_reason": None, "invalid_configuration_observed": False}
    _write_json(run_dir / "stages" / f"{stage_name}.json", stage)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    fingerprint = {"config_sha256": sha256_file(config_path), "data_validation": data_validation, "upstream": {"head": _git(upstream, "rev-parse", "HEAD"), "status": _git(upstream, "status", "--short"), "diff_sha256": hashlib.sha256(_git(upstream, "diff", "--binary").encode()).hexdigest()}, "checkpoint_sha256": checkpoint_row["sha256"], "source_snapshots": snapshots}
    _write_json(run_dir / "fingerprint.json", fingerprint)
    now = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "events.jsonl", {"at_utc": now, "event": "streetgs_training_complete", "scene": scene, "mode": mode, "status": "done"})
    summary = {"schema_version": "worldsim_v4_streetgs_summary_v1", "task_id": task_id, "status": "done", "scene": scene, "scene_index": scene_index, "mode": mode, "iterations": iterations, "finished_at_utc": now, "duration_seconds": elapsed, "checkpoint": checkpoint_row, "resources": {"peak_gpu_memory_mib": peak_gpu, "peak_cgroup_memory_bytes": peak_memory}, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "project_git": {"head": _git(project_root, "rev-parse", "HEAD"), "branch": _git(project_root, "branch", "--show-current"), "dirty": bool(_git(project_root, "status", "--porcelain"))}, "training_started": True, "model_inference_started": False, "test_quality_read": False}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": task_id, "status": "done", "scene": scene, "mode": mode, "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json" and not path.is_relative_to(run_dir / "work_dirs")}
    artifacts["work_dirs_checkpoint"] = checkpoint_row
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_streetgs_run_manifest_v1", "task_id": task_id, "status": "done", "scene": scene, "mode": mode, "artifacts": artifacts, "test_quality_read": False})
    return summary


def record_blocked(config_path: Path, run_dir: Path, project_root: Path, scene: str, mode: str, error: Exception) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    now = datetime.now(timezone.utc).isoformat()
    event = {"at_utc": now, "event": "streetgs_training_blocked", "scene": scene, "mode": mode, "error_type": type(error).__name__, "message": str(error)}
    append_jsonl(run_dir / "events.jsonl", event)
    fingerprint = {"config_sha256": sha256_file(config_path) if config_path.is_file() else None, "project_head": _git(project_root, "rev-parse", "HEAD"), "error": event}
    _write_json(run_dir / "fingerprint.json", fingerprint)
    try:
        task_id = str(_load_yaml(config_path).get("task_id", TASK_ID))
    except Exception:
        task_id = TASK_ID
    summary = {"schema_version": "worldsim_v4_streetgs_summary_v1", "task_id": task_id, "status": "blocked", "scene": scene, "mode": mode, "finished_at_utc": now, "reason": "streetgs_training_failed", "error": event, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "training_started": (run_dir / "resource.jsonl").exists(), "model_inference_started": False, "test_quality_read": False}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": task_id, "status": "blocked", "scene": scene, "mode": mode, "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json" and not path.is_relative_to(run_dir / "work_dirs")}
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_streetgs_run_manifest_v1", "task_id": task_id, "status": "blocked", "scene": scene, "mode": mode, "artifacts": artifacts, "test_quality_read": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 WorldSim V4 单场景 StreetGS")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", choices=["profile100", "formal"], required=True)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        summary = run(args.config, args.run_dir, args.project_root, args.scene, args.mode)
    except BaseException as error:
        global ACTIVE_PROCESS
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
            ACTIVE_PROCESS.wait(timeout=30)
            ACTIVE_PROCESS = None
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), args.project_root.resolve(), args.scene, args.mode, error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Resource-guarded DriveStudio preprocessing and sky-mask stages for M3."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def int_file(path: str) -> int | None:
    value = Path(path).read_text().strip()
    return None if value == "max" else int(value)


def events() -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines())}


def gpu() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    return {
        "name": fields[0] if len(fields) > 0 else None,
        "driver": fields[1] if len(fields) > 1 else None,
        "memory_total_mib": int(fields[2]) if len(fields) > 2 else None,
        "memory_used_mib": int(fields[3]) if len(fields) > 3 else None,
    }


def sample(stage: str, event: str) -> dict:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": now(),
        "stage": stage,
        "event": event,
        "memory_current_bytes": int_file("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": int_file("/sys/fs/cgroup/memory.max"),
        "memory_events": events(),
        "disk_free_bytes": disk.free,
        "gpu": gpu(),
    }


def append(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["preprocess", "sky_masks"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/motion_proj"))
    parser.add_argument("--upstream-root", type=Path, default=Path("/root/autodl-tmp/third_party/drivestudio"))
    parser.add_argument("--environment", type=Path, default=Path("/root/autodl-tmp/envs/drivestudio"))
    parser.add_argument("--raw-root", default="/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_raw_scene0230")
    parser.add_argument("--processed-base", default="/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed")
    args = parser.parse_args()

    stage_path = args.run_dir / "stages" / f"{args.stage}.json"
    log_path = args.run_dir / "logs" / f"{args.stage}.log"
    if stage_path.exists() or log_path.exists():
        raise FileExistsError(f"refuse to overwrite {args.stage} artifacts")
    processed_root = Path(args.processed_base.replace("processed", "processed_10Hz")) / "trainval"
    scene_root = processed_root / "179"
    if args.stage == "preprocess":
        raw_manifest = Path("/root/autodl-tmp/data/dynamic_editing_v2/manifests/scene-0230_raw_manifest.json")
        if not raw_manifest.is_file() or not json.loads(raw_manifest.read_text()).get("complete"):
            raise RuntimeError("complete scene-0230 raw manifest is required")
        command = [
            str(args.environment / "bin" / "python"),
            "datasets/preprocess.py",
            "--data_root", args.raw_root,
            "--target_dir", args.processed_base,
            "--dataset", "nuscenes",
            "--split", "v1.0-trainval",
            "--interpolate_N", "4",
            # Upstream's workers==1 + explicit scene_ids branch calls convert_one()
            # directly and silently bypasses convert_one_interpolated().  With one
            # scene and workers=2, only one conversion job runs while convert()
            # correctly dispatches the requested 10 Hz interpolation path.
            "--workers", "2",
            "--scene_ids", "179",
            "--process_keys", "images", "lidar", "calib", "dynamic_masks", "objects",
        ]
        cwd = args.upstream_root
    else:
        if not (scene_root / "images").is_dir():
            raise RuntimeError(f"processed images missing: {scene_root}")
        command = [
            str(args.environment / "bin" / "python"),
            str(args.project_root / "scripts" / "build_dr_v2_sky_masks.py"),
            "--scene-root", str(scene_root),
            "--revision", "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
            "--manifest", str(args.run_dir / "environment" / "sky_mask_model_and_files.json"),
        ]
        cwd = args.project_root

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(args.upstream_root),
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
            "TMPDIR": "/root/autodl-tmp/tmp",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )
    resource_path = args.run_dir / "resource.jsonl"
    pre = sample(args.stage, "preflight")
    append(resource_path, pre)
    if pre["disk_free_bytes"] < 60 * 2**30:
        raise RuntimeError("data disk below 60 GiB before data stage")
    if args.stage == "sky_masks" and int(pre["gpu"]["memory_used_mib"] or 0) > 2048:
        raise RuntimeError("GPU is not idle before sky-mask stage")
    baseline_events = pre["memory_events"]
    started = time.monotonic()
    peak_memory = int(pre["memory_current_bytes"] or 0)
    peak_gpu = int(pre["gpu"]["memory_used_mib"] or 0)
    over_memory = 0
    stop_reason = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            time.sleep(10)
            current = sample(args.stage, "running")
            append(resource_path, current)
            peak_memory = max(peak_memory, int(current["memory_current_bytes"] or 0))
            peak_gpu = max(peak_gpu, int(current["gpu"]["memory_used_mib"] or 0))
            maximum = current["memory_max_bytes"]
            used = current["memory_current_bytes"]
            over_memory = over_memory + 1 if maximum and used and used / maximum >= 0.90 else 0
            if over_memory >= 2:
                stop_reason = "memory ratio >= 0.90 twice"
            elif current["memory_events"].get("oom", 0) > baseline_events.get("oom", 0) or current["memory_events"].get("oom_kill", 0) > baseline_events.get("oom_kill", 0):
                stop_reason = "cgroup oom event increased"
            elif current["disk_free_bytes"] < 20 * 2**30:
                stop_reason = "disk free below 20 GiB"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                break
        return_code = process.wait()

    counts = {
        "images": len(list((scene_root / "images").glob("*.jpg"))),
        "lidar": len(list((scene_root / "lidar").glob("*.bin"))),
        "lidar_pose": len(list((scene_root / "lidar_pose").glob("*.txt"))),
        "extrinsics": len(list((scene_root / "extrinsics").glob("*.txt"))),
        "sky_masks": len(list((scene_root / "sky_masks").glob("*.png"))),
    }
    if args.stage == "preprocess":
        output_ok = counts["images"] == 196 * 6 and counts["lidar"] == 196 and counts["lidar_pose"] == 196 and counts["extrinsics"] == 196 * 6 and (scene_root / "instances" / "instances_info.json").is_file()
    else:
        output_ok = counts["sky_masks"] == 196 * 3
    stage = {
        "stage": args.stage,
        "status": "done" if return_code == 0 and output_ok and stop_reason is None else "blocked",
        "return_code": return_code,
        "stop_reason": stop_reason,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "log": str(log_path),
        "scene_root": str(scene_root),
        "counts": counts,
        "expected": {"timesteps": 196, "all_camera_images": 1176, "training_camera_masks": 588},
        "peak_cgroup_memory_bytes": peak_memory,
        "peak_gpu_memory_mib": peak_gpu,
    }
    atomic_json(stage_path, stage)
    append(resource_path, sample(args.stage, "completed"))
    print(json.dumps(stage, sort_keys=True))
    if stage["status"] != "done":
        raise SystemExit(return_code or 2)


if __name__ == "__main__":
    main()

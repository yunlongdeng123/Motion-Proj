#!/usr/bin/env python
"""Resource-guarded DriveStudio StreetGS training stage for M3."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


MODES = {
    "profile100": 100,
    "profile1000": 1000,
    "formal": 30000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def memory_value(path: str) -> int | None:
    raw = Path(path).read_text().strip()
    return None if raw == "max" else int(raw)


def memory_events() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )
    }


def gpu_sample() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
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


def sample_resource() -> dict:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": timestamp(),
        "memory_current_bytes": memory_value("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": memory_value("/sys/fs/cgroup/memory.max"),
        "memory_events": memory_events(),
        "disk_free_bytes": disk.free,
        "gpu": gpu_sample(),
    }


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def preflight(sample: dict) -> None:
    gpu = sample["gpu"]
    if gpu["memory_total_mib"] is None or gpu["memory_total_mib"] < 24000:
        raise RuntimeError(f"GPU memory contract failed: {gpu}")
    if gpu["memory_used_mib"] is None or gpu["memory_used_mib"] > 2048:
        raise RuntimeError(f"GPU must be idle at stage start: {gpu}")
    if sample["memory_max_bytes"] is None or sample["memory_max_bytes"] < 32 * 2**30:
        raise RuntimeError("cgroup memory must be at least 32 GiB")
    if sample["disk_free_bytes"] < 60 * 2**30:
        raise RuntimeError("data disk must have at least 60 GiB free before training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--processed-root", default="/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval")
    parser.add_argument("--scene-index", type=int, default=179)
    parser.add_argument("--upstream-root", type=Path, default=Path("/root/autodl-tmp/third_party/drivestudio"))
    parser.add_argument("--environment", type=Path, default=Path("/root/autodl-tmp/envs/drivestudio"))
    args = parser.parse_args()

    iterations = MODES[args.mode]
    stage_name = f"train_{args.mode}"
    stage_path = args.run_dir / "stages" / f"{stage_name}.json"
    log_path = args.run_dir / "logs" / f"{stage_name}.log"
    resource_path = args.run_dir / "resource.jsonl"
    for path in (stage_path, log_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite stage artifact: {path}")
    pre = sample_resource()
    pre["stage"] = stage_name
    pre["event"] = "preflight"
    append_jsonl(resource_path, pre)
    preflight(pre)

    output_root = args.run_dir / "work_dirs"
    project = "m3_formal" if args.mode == "formal" else "m3_profile"
    run_name = f"scene0230_{args.mode}_s0"
    log_dir = output_root / project / run_name
    command = [
        str(args.environment / "bin" / "python"),
        "tools/train.py",
        "--config_file",
        "configs/streetgs.yaml",
        "--output_root",
        str(output_root),
        "--project",
        project,
        "--run_name",
        run_name,
        "dataset=nuscenes/3cams",
        f"data.data_root={args.processed_root}",
        f"data.scene_idx={args.scene_index}",
        "data.start_timestep=0",
        "data.end_timestep=-1",
        "data.pixel_source.load_smpl=false",
    ]
    if args.mode != "formal":
        command.extend(
            [
                f"trainer.optim.num_iters={iterations}",
                f"logging.saveckpt_freq={iterations}",
                "render.render_full=false",
                "render.render_test=false",
                "render.render_novel=null",
            ]
        )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(args.upstream_root),
            "WANDB_MODE": "disabled",
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
            "TMPDIR": "/root/autodl-tmp/tmp",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )
    started = time.monotonic()
    baseline_events = pre["memory_events"]
    over_memory_samples = 0
    stop_reason = None
    peak_gpu_mib = int(pre["gpu"]["memory_used_mib"] or 0)
    peak_memory_bytes = int(pre["memory_current_bytes"] or 0)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=args.upstream_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(10)
            sample = sample_resource()
            sample["stage"] = stage_name
            sample["event"] = "running"
            append_jsonl(resource_path, sample)
            peak_gpu_mib = max(peak_gpu_mib, int(sample["gpu"]["memory_used_mib"] or 0))
            peak_memory_bytes = max(peak_memory_bytes, int(sample["memory_current_bytes"] or 0))
            maximum = sample["memory_max_bytes"]
            current = sample["memory_current_bytes"]
            over_memory_samples = over_memory_samples + 1 if maximum and current and current / maximum >= 0.90 else 0
            events = sample["memory_events"]
            if over_memory_samples >= 2:
                stop_reason = "memory.current/memory.max >= 0.90 twice"
            elif events.get("oom", 0) > baseline_events.get("oom", 0) or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0):
                stop_reason = "cgroup oom event increased"
            elif sample["disk_free_bytes"] < 20 * 2**30:
                stop_reason = "disk free below 20 GiB"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()

    final = sample_resource()
    final["stage"] = stage_name
    final["event"] = "completed"
    append_jsonl(resource_path, final)
    checkpoint = log_dir / "checkpoint_final.pth"
    gsplat_binary = args.upstream_root.parent / "gsplat" / "gsplat" / "csrc.so"
    nvdiffrast_binary = (
        args.environment
        / "lib"
        / "python3.9"
        / "site-packages"
        / "_nvdiffrast_c.cpython-39-x86_64-linux-gnu.so"
    )
    stage = {
        "stage": stage_name,
        "mode": args.mode,
        "status": "done" if return_code == 0 and checkpoint.is_file() and stop_reason is None else "blocked",
        "return_code": return_code,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "upstream_root": str(args.upstream_root),
        "log": str(log_path),
        "log_dir": str(log_dir),
        "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "peak_gpu_memory_mib": peak_gpu_mib,
        "peak_cgroup_memory_bytes": peak_memory_bytes,
        "input_compatibility_override": {
            "data.pixel_source.load_smpl": False,
            "reason": "StreetGS config has no SMPLNodes; avoid loading an unused optional human-pose asset.",
        },
        "cuda_extensions": {
            "required_compute_capability": "8.6",
            "gsplat": {
                "path": str(gsplat_binary),
                "bytes": gsplat_binary.stat().st_size if gsplat_binary.is_file() else 0,
                "sha256": sha256_file(gsplat_binary) if gsplat_binary.is_file() else None,
            },
            "nvdiffrast": {
                "path": str(nvdiffrast_binary),
                "bytes": nvdiffrast_binary.stat().st_size if nvdiffrast_binary.is_file() else 0,
                "sha256": sha256_file(nvdiffrast_binary) if nvdiffrast_binary.is_file() else None,
            },
        },
        "profile_only_overrides": None if args.mode == "formal" else {
            "trainer.optim.num_iters": iterations,
            "logging.saveckpt_freq": iterations,
            "render.render_full": False,
            "render.render_test": False,
            "render.render_novel": None,
        },
    }
    atomic_json(stage_path, stage)
    print(json.dumps(stage, sort_keys=True))
    if stage["status"] != "done":
        raise SystemExit(return_code or 2)


if __name__ == "__main__":
    main()

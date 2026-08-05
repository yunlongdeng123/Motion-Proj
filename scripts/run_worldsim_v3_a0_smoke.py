#!/usr/bin/env python
"""Run the scene-0255 one-step DriveStudio compatibility smoke for V3 A0."""

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

import torch

from motion_proj.worldsim_v3.drivestudio_compat import (
    concatenate_paired_lidar_chunks,
)


TASK_ID = "WS-V3-A0-NATIVE-BASELINE-01"
PROJECT = Path("/root/autodl-tmp/motion_proj")
PATCHED_DRIVESTUDIO = Path(
    "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-r2"
)
DRIVESTUDIO_ENV = Path("/root/autodl-tmp/envs/drivestudio")
PROCESSED_ROOT = Path(
    "/root/autodl-tmp/data/dynamic_editing_v2/"
    "drivestudio_processed_10Hz/trainval"
)
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command_output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def int_file(path: str) -> int | None:
    value = Path(path).read_text(encoding="utf-8").strip()
    return None if value == "max" else int(value)


def memory_events() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split()
            for line in Path("/sys/fs/cgroup/memory.events")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }


def gpu_sample() -> dict[str, object]:
    output = command_output(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    fields = [field.strip() for field in output.split(",")]
    return {
        "name": fields[0],
        "driver": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
    }


def resource_sample(stage: str, event: str) -> dict[str, object]:
    usage = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": now(),
        "stage": stage,
        "event": event,
        "memory_current_bytes": int_file("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": int_file("/sys/fs/cgroup/memory.max"),
        "memory_events": memory_events(),
        "disk_free_bytes": usage.free,
        "gpu": gpu_sample(),
    }


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def scene0255_contract() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shapes = (
        [(0, 3)] * 152
        + [(1, 3)] * 3
        + [(2, 3)] * 3
        + [(4, 3)] * 3
        + [(5, 3), (7, 3), (8, 3)]
        + [(9, 3)] * 2
    )
    points = [
        torch.arange(rows * 3, device="cuda", dtype=torch.float32).reshape(
            rows, 3
        )
        for rows, _ in shapes
    ]
    return points, [chunk + 1000 for chunk in points]


def run_contract_probe() -> dict[str, object]:
    point_chunks, color_chunks = scene0255_contract()
    native_error = None
    try:
        torch.cat(point_chunks, dim=0)
        torch.cuda.synchronize()
    except RuntimeError as error:
        native_error = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        torch.cuda.synchronize()
    points, colors = concatenate_paired_lidar_chunks(
        point_chunks, color_chunks
    )
    torch.cuda.synchronize()
    if points.shape != (59, 3) or points.numel() != 177:
        raise RuntimeError(f"patched point contract changed: {points.shape}")
    if not torch.equal(colors, points + 1000):
        raise RuntimeError("point/color pairing changed")
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "native_error": native_error,
        "patched_shape": list(points.shape),
        "patched_numel": points.numel(),
        "point_color_pairing": "exact",
    }


def build_train_command(run_dir: Path, num_iters: int) -> list[str]:
    return [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PATCHED_DRIVESTUDIO / "tools/train.py"),
        "--config_file",
        "configs/streetgs.yaml",
        "--output_root",
        str(run_dir / "work_dirs"),
        "--project",
        "worldsim_v3",
        "--run_name",
        "scene0255_a0_catfix_s0",
        "dataset=nuscenes/3cams",
        f"data.data_root={PROCESSED_ROOT}",
        "data.scene_idx=204",
        "data.start_timestep=0",
        "data.end_timestep=-1",
        "data.pixel_source.load_smpl=false",
        "data.pixel_source.test_image_stride=10",
        f"trainer.optim.num_iters={num_iters}",
        f"logging.saveckpt_freq={num_iters}",
        "render.render_full=false",
        "render.render_test=false",
        "render.render_novel=null",
    ]


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    args = parser.parse_args()
    if args.num_iters < 1:
        raise ValueError("num-iters must be positive")
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    patch = PROJECT / "compatibility/DriveStudio-2026-08-05.patch"
    sources = (
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "scripts/prepare_worldsim_v3_drivestudio.py",
        PROJECT / "motion_proj/worldsim_v3/drivestudio_compat.py",
        patch,
    )
    for source in sources:
        destination = args.run_dir / "source_snapshot" / source.relative_to(
            PROJECT
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    verify = subprocess.run(
        [
            "/root/miniconda3/bin/python",
            str(PROJECT / "scripts/prepare_worldsim_v3_drivestudio.py"),
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compatibility = json.loads(verify.stdout)
    probe = run_contract_probe()
    atomic_json(args.run_dir / "stages/contract_probe.json", probe)
    command = build_train_command(args.run_dir, args.num_iters)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": "scene-0255 patched DriveStudio one-step smoke",
        "scene_name": "scene-0255",
        "scene_index": 204,
        "seed": 0,
        "command": command,
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "patched_drivestudio": compatibility,
        "patch_sha256": sha256_file(patch),
        "contract_probe": probe,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{PROJECT}:{PATCHED_DRIVESTUDIO}",
            "WANDB_MODE": "disabled",
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
        }
    )
    pre = resource_sample("train_smoke", "preflight")
    append_jsonl(args.run_dir / "resource.jsonl", pre)
    if pre["disk_free_bytes"] < 20 * 2**30:
        raise RuntimeError("smoke requires at least 20 GiB free disk")
    if pre["gpu"]["memory_used_mib"] > 2048:
        raise RuntimeError(f"GPU is not idle: {pre['gpu']}")

    log_path = args.run_dir / "logs/train_smoke.log"
    started = time.monotonic()
    peak_gpu = int(pre["gpu"]["memory_used_mib"])
    peak_memory = int(pre["memory_current_bytes"] or 0)
    baseline_events = pre["memory_events"]
    stop_reason = None
    over_memory = 0
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=PATCHED_DRIVESTUDIO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(5)
            current = resource_sample("train_smoke", "running")
            append_jsonl(args.run_dir / "resource.jsonl", current)
            peak_gpu = max(peak_gpu, int(current["gpu"]["memory_used_mib"]))
            peak_memory = max(
                peak_memory, int(current["memory_current_bytes"] or 0)
            )
            maximum = current["memory_max_bytes"]
            used = current["memory_current_bytes"]
            over_memory = (
                over_memory + 1
                if maximum and used and used / maximum >= 0.90
                else 0
            )
            events = current["memory_events"]
            if over_memory >= 2:
                stop_reason = "memory.current/memory.max >= 0.90 twice"
            elif (
                events.get("oom", 0) > baseline_events.get("oom", 0)
                or events.get("oom_kill", 0)
                > baseline_events.get("oom_kill", 0)
            ):
                stop_reason = "cgroup oom event increased"
            elif time.monotonic() - started > args.timeout_seconds:
                stop_reason = f"timeout after {args.timeout_seconds:.0f}s"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()

    output_files = [
        str(path.relative_to(args.run_dir))
        for path in (args.run_dir / "work_dirs").rglob("*")
        if path.is_file()
    ]
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    invalid_configuration = "invalid configuration argument" in log_text
    stage = {
        "status": "done"
        if return_code == 0
        and stop_reason is None
        and not invalid_configuration
        and output_files
        else "blocked",
        "return_code": return_code,
        "stop_reason": stop_reason,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
        "invalid_configuration_observed": invalid_configuration,
        "output_file_count": len(output_files),
        "output_files": output_files,
    }
    atomic_json(args.run_dir / "stages/train_smoke.json", stage)
    append_jsonl(
        args.run_dir / "resource.jsonl",
        resource_sample("train_smoke", "completed"),
    )
    terminal = {
        "status": stage["status"],
        "updated_at": now(),
        "failure": None
        if stage["status"] == "done"
        else {"code": "A0_SCENE0255_CAT_FIX_SMOKE_FAILED", "detail": stage},
    }
    atomic_json(args.run_dir / "terminal.json", terminal)
    _TERMINAL_FINAL = True
    if stage["status"] != "done":
        raise RuntimeError(terminal["failure"])
    print(json.dumps(stage, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if (
            _ACTIVE_RUN_DIR is not None
            and _ACTIVE_RUN_DIR.is_dir()
            and not _TERMINAL_FINAL
        ):
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "code": "A0_SMOKE_UNCAUGHT",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

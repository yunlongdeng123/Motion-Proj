#!/usr/bin/env python
"""Train and evaluate the scene-0255 native StreetGS baseline for V3 A0."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch

from scripts.run_worldsim_v3_a0_smoke import (
    DRIVESTUDIO_ENV,
    PATCHED_DRIVESTUDIO,
    PROCESSED_ROOT,
    PROJECT,
    TASK_ID,
    append_jsonl,
    atomic_json,
    command_output,
    now,
    resource_sample,
    sha256_file,
)


SCENE_NAME = "scene-0255"
SCENE_INDEX = 204
HIGH_TOKEN = "f4aa30b8d0b44e2381a4abeafbe17642"
BOUNDARY_TOKEN = "80c08b992f1d47359de644be24f491df"
NUM_ITERS = 30_000
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def build_train_command(run_dir: Path) -> tuple[list[str], Path]:
    run_name = "scene0255_a0_native_heldout_s0"
    checkpoint = (
        run_dir
        / "work_dirs"
        / "worldsim_v3"
        / run_name
        / "checkpoint_final.pth"
    )
    command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PATCHED_DRIVESTUDIO / "tools/train.py"),
        "--config_file",
        "configs/streetgs.yaml",
        "--output_root",
        str(run_dir / "work_dirs"),
        "--project",
        "worldsim_v3",
        "--run_name",
        run_name,
        "dataset=nuscenes/3cams",
        f"data.data_root={PROCESSED_ROOT}",
        f"data.scene_idx={SCENE_INDEX}",
        "data.start_timestep=0",
        "data.end_timestep=-1",
        "data.pixel_source.load_smpl=false",
        "data.pixel_source.test_image_stride=10",
        f"trainer.optim.num_iters={NUM_ITERS}",
        f"logging.saveckpt_freq={NUM_ITERS}",
        "render.render_full=false",
        "render.render_test=false",
        "render.render_novel=null",
    ]
    return command, checkpoint


def build_eval_command(checkpoint: Path) -> list[str]:
    return [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PATCHED_DRIVESTUDIO / "tools/eval.py"),
        "--resume_from",
        str(checkpoint),
        "--render_video_postfix",
        "a0_native",
        "render.render_test=true",
        "render.render_full=false",
        "render.render_novel=null",
    ]


def common_environment() -> dict[str, str]:
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
    return environment


def log_torch_peak_mib(log_path: Path) -> int | None:
    values = [
        int(value)
        for value in re.findall(
            r"max mem:\s*(\d+)",
            log_path.read_text(encoding="utf-8", errors="replace"),
        )
    ]
    return max(values) if values else None


def run_stage(
    *,
    run_dir: Path,
    stage: str,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    validate: Callable[[], tuple[bool, dict[str, object]]],
    timeout_seconds: float,
    gpu_idle_required: bool = True,
) -> dict[str, object]:
    stage_path = run_dir / "stages" / f"{stage}.json"
    log_path = run_dir / "logs" / f"{stage}.log"
    if stage_path.exists() or log_path.exists():
        raise FileExistsError(f"refusing to overwrite stage {stage}")
    pre = resource_sample(stage, "preflight")
    append_jsonl(run_dir / "resource.jsonl", pre)
    if pre["disk_free_bytes"] < 20 * 2**30:
        raise RuntimeError(f"{stage}: requires at least 20 GiB free disk")
    if gpu_idle_required and pre["gpu"]["memory_used_mib"] > 2048:
        raise RuntimeError(f"{stage}: GPU is not idle: {pre['gpu']}")

    baseline_events = pre["memory_events"]
    peak_gpu = int(pre["gpu"]["memory_used_mib"])
    peak_memory = int(pre["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(10)
            current = resource_sample(stage, "running")
            append_jsonl(run_dir / "resource.jsonl", current)
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
            elif time.monotonic() - started > timeout_seconds:
                stop_reason = f"timeout after {timeout_seconds:.0f}s"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()

    output_ok, evidence = validate()
    invalid_configuration = "invalid configuration argument" in log_path.read_text(
        encoding="utf-8", errors="replace"
    )
    payload: dict[str, object] = {
        "stage": stage,
        "status": "done"
        if return_code == 0
        and stop_reason is None
        and output_ok
        and not invalid_configuration
        else "blocked",
        "return_code": return_code,
        "stop_reason": stop_reason,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "log": str(log_path),
        "output_validation": evidence,
        "peak_gpu_memory_mib_sampled": peak_gpu,
        "peak_gpu_memory_mib_torch_log": log_torch_peak_mib(log_path),
        "peak_cgroup_memory_bytes": peak_memory,
        "invalid_configuration_observed": invalid_configuration,
    }
    atomic_json(stage_path, payload)
    append_jsonl(
        run_dir / "resource.jsonl", resource_sample(stage, "completed")
    )
    if payload["status"] != "done":
        raise RuntimeError(f"stage {stage} blocked: {payload}")
    return payload


def checkpoint_contract(checkpoint: Path) -> dict[str, object]:
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return {"checkpoint": str(checkpoint), "exists": False}
    state = torch.load(checkpoint, map_location="cpu")
    models = state.get("models", {})
    background = models.get("Background", {}).get("_means")
    rigid = models.get("RigidNodes", {}).get("_means")
    result = {
        "checkpoint": str(checkpoint),
        "exists": True,
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "step": int(state.get("step", -1)),
        "model_keys": sorted(models),
        "background_gaussians": int(background.shape[0])
        if background is not None
        else None,
        "rigid_gaussians": int(rigid.shape[0]) if rigid is not None else None,
    }
    del state, models, background, rigid
    gc.collect()
    return result


def compact_actor(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    tensor_slice = row.get("checkpoint_tensor_slice") or {}
    return {
        "instance_token": row.get("instance_token"),
        "class_name": row.get("class_name"),
        "availability": row.get("availability"),
        "rigid_model_index": row.get("rigid_model_index"),
        "gaussian_count": tensor_slice.get("gaussian_count"),
    }


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-timeout-seconds", type=float, default=9000)
    parser.add_argument("--eval-timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    patch = PROJECT / "compatibility/DriveStudio-2026-08-05.patch"
    sources = (
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "scripts/prepare_worldsim_v3_drivestudio.py",
        PROJECT / "scripts/build_dr_v2_drivestudio_registry.py",
        PROJECT / "motion_proj/worldsim_v3/drivestudio_compat.py",
        PROJECT / "motion_proj/dynamic_editing_v2/drivestudio_registry.py",
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
    train_command, checkpoint = build_train_command(args.run_dir)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": "scene-0255 A0 native 30k held-out baseline",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "actors": {
            "high-support": HIGH_TOKEN,
            "boundary-support": BOUNDARY_TOKEN,
        },
        "seed": 0,
        "test_image_stride": 10,
        "num_iters": NUM_ITERS,
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "patched_drivestudio": compatibility,
        "train_command": train_command,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    environment = common_environment()

    train_stage = run_stage(
        run_dir=args.run_dir,
        stage="train_heldout_30000",
        command=train_command,
        cwd=PATCHED_DRIVESTUDIO,
        environment=environment,
        validate=lambda: (
            checkpoint.is_file() and checkpoint.stat().st_size > 0,
            {
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size
                if checkpoint.is_file()
                else 0,
            },
        ),
        timeout_seconds=args.train_timeout_seconds,
    )
    checkpoint_info = checkpoint_contract(checkpoint)
    if checkpoint_info.get("step") != NUM_ITERS:
        raise RuntimeError(f"checkpoint step mismatch: {checkpoint_info}")
    train_stage["checkpoint_contract"] = checkpoint_info
    atomic_json(args.run_dir / "stages/train_heldout_30000.json", train_stage)

    registry = args.run_dir / "artifacts/actor_registry.json"
    registry_command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/build_dr_v2_drivestudio_registry.py"),
        "--checkpoint",
        str(checkpoint),
        "--drivestudio-root",
        str(PATCHED_DRIVESTUDIO),
        "--scene-name",
        SCENE_NAME,
        "--selected-token",
        HIGH_TOKEN,
        "--output",
        str(registry),
    ]
    run_stage(
        run_dir=args.run_dir,
        stage="actor_registry",
        command=registry_command,
        cwd=PROJECT,
        environment=environment,
        validate=lambda: (
            registry.is_file() and registry.stat().st_size > 0,
            {"registry": str(registry)},
        ),
        timeout_seconds=900,
    )
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    by_token = {
        row["instance_token"]: row
        for row in registry_payload.get("actors", [])
    }
    selected = {
        "high-support": compact_actor(by_token.get(HIGH_TOKEN)),
        "boundary-support": compact_actor(by_token.get(BOUNDARY_TOKEN)),
    }
    if selected["high-support"] is None:
        raise RuntimeError("high-support actor is missing from registry")
    atomic_json(
        args.run_dir / "stages/selected_actors.json",
        {
            "status": "done",
            "registry_sha256": sha256_file(registry),
            "selected": selected,
        },
    )

    eval_command = build_eval_command(checkpoint)
    eval_dir = checkpoint.parent / "metrics_eval"
    video_dir = checkpoint.parent / "videos_eval"
    eval_stage = run_stage(
        run_dir=args.run_dir,
        stage="eval_heldout",
        command=eval_command,
        cwd=PATCHED_DRIVESTUDIO,
        environment=environment,
        validate=lambda: (
            bool(list(eval_dir.glob("images_test_*.json")))
            and bool(list(video_dir.glob("test_set_*.mp4"))),
            {
                "metric_files": [str(path) for path in sorted(eval_dir.glob("*.json"))],
                "video_files": [str(path) for path in sorted(video_dir.glob("*.mp4"))],
            },
        ),
        timeout_seconds=args.eval_timeout_seconds,
    )
    metric_files = sorted(eval_dir.glob("images_test_*.json"))
    heldout_metrics = json.loads(metric_files[-1].read_text(encoding="utf-8"))
    eval_stage["heldout_metrics"] = heldout_metrics
    atomic_json(args.run_dir / "stages/eval_heldout.json", eval_stage)

    summary = {
        "status": "done",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "checkpoint": checkpoint_info,
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "selected_actors": selected,
        "heldout_metrics": heldout_metrics,
        "train_resources": {
            key: train_stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        },
        "eval_resources": {
            key: eval_stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        },
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


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
                        "code": "A0_SCENE0255_FORMAL_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

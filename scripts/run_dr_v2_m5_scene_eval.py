#!/usr/bin/env python3
"""带资源守卫的单场景 M5 渲染与感知评测控制器。"""
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

import yaml


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, payload: object, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def int_file(path: str) -> int | None:
    raw = Path(path).read_text().strip()
    return None if raw == "max" else int(raw)


def events() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split()
            for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )
    }


def gpu() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    return {
        "name": values[0],
        "driver": values[1],
        "memory_total_mib": int(values[2]),
        "memory_used_mib": int(values[3]),
        "utilization_percent": int(values[4]),
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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_stage(
    *,
    run_dir: Path,
    stage: str,
    command: list[str],
    environment: dict[str, str],
    expected_report: Path,
    minimum_disk_gib: int,
) -> dict:
    pre = sample(stage, "preflight")
    append(run_dir / "resource.jsonl", pre)
    if pre["disk_free_bytes"] < minimum_disk_gib * 2**30:
        raise RuntimeError(f"{stage}: disk below {minimum_disk_gib} GiB")
    if pre["gpu"]["memory_used_mib"] > 2048:
        raise RuntimeError(f"{stage}: GPU not idle at start")
    baseline_events = pre["memory_events"]
    log_path = run_dir / "logs" / f"{stage}.log"
    started = time.monotonic()
    peak_gpu = pre["gpu"]["memory_used_mib"]
    peak_memory = int(pre["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd="/root/autodl-tmp/motion_proj",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(10)
            current = sample(stage, "running")
            append(run_dir / "resource.jsonl", current)
            peak_gpu = max(peak_gpu, current["gpu"]["memory_used_mib"])
            peak_memory = max(peak_memory, int(current["memory_current_bytes"] or 0))
            maximum = current["memory_max_bytes"]
            used = current["memory_current_bytes"]
            over_memory = over_memory + 1 if maximum and used and used / maximum >= 0.90 else 0
            current_events = current["memory_events"]
            if over_memory >= 2:
                stop_reason = "memory.current/memory.max >= 0.90 twice"
            elif (
                current_events.get("oom", 0) > baseline_events.get("oom", 0)
                or current_events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0)
            ):
                stop_reason = "cgroup oom event increased"
            elif current["disk_free_bytes"] < 12 * 2**30:
                stop_reason = "disk below 12 GiB"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()
    report = None
    if expected_report.is_file():
        report = json.loads(expected_report.read_text(encoding="utf-8"))
    output_ok = report is not None and report.get("status") == "done"
    if stage == "stress_render" and output_ok:
        output_ok = all(report.get("checks", {}).values())
    if stage == "perception" and output_ok:
        output_ok = bool(report.get("checks")) and all(
            report["checks"].values()
        )
    payload = {
        "stage": stage,
        "status": "done" if return_code == 0 and stop_reason is None and output_ok else "blocked",
        "return_code": return_code,
        "stop_reason": stop_reason,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "log": str(log_path),
        "report": str(expected_report) if expected_report.is_file() else None,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
    }
    atomic_json(run_dir / "stages" / f"{stage}.json", payload)
    append(run_dir / "resource.jsonl", sample(stage, "completed"))
    if payload["status"] != "done":
        raise RuntimeError(f"{stage} blocked: {payload}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("/root/autodl-tmp/motion_proj/configs/dynamic_editing_v2/m5_protocol_v1.yaml"),
    )
    parser.add_argument("--frame-count", type=int)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    for directory in ("artifacts", "logs", "source_snapshot", "stages"):
        (args.run_dir / directory).mkdir(parents=True, exist_ok=True)
    terminal = args.run_dir / "terminal.json"
    atomic_json(terminal, {"status": "running", "updated_at": now(), "failure": None})
    project_root = Path("/root/autodl-tmp/motion_proj")
    for source in (
        "scripts/run_dr_v2_m5_scene_eval.py",
        "scripts/run_dr_v2_m5_stress_scene.py",
        "scripts/run_dr_v2_m5_perception.py",
        "motion_proj/dynamic_editing_v2/stress_metrics.py",
        "configs/dynamic_editing_v2/m5_protocol_v1.yaml",
    ):
        src = project_root / source
        dst = args.run_dir / "source_snapshot" / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    try:
        training_terminal = json.loads(
            (args.training_run / "terminal.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            (args.training_run / "summary.json").read_text(encoding="utf-8")
        )
        if training_terminal["status"] != "done" or training["status"] != "done":
            raise RuntimeError("training run 未完成")
        protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
        scene = training["scene_name"]
        scene_protocol = protocol["scenes"][scene]
        manifest = {
            "schema_version": 1,
            "task_id": "DR-V2-M5-STRESS-3SCENE-01",
            "component": "scene edit stress and frozen perception evaluation",
            "scene": scene,
            "training_run": str(args.training_run),
            "checkpoint": training["checkpoint"],
            "registry": training["registry"],
            "protocol": str(args.protocol),
            "frame_count": args.frame_count or protocol["frame_count"],
            "started_at": now(),
            "project_commit": subprocess.check_output(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        atomic_json(args.run_dir / "manifest.json", manifest)
        atomic_json(args.run_dir / "resolved.json", manifest)
        common = os.environ.copy()
        common.update(
            {
                "PYTHONPATH": f"{project_root}:/root/autodl-tmp/third_party/drivestudio",
                "WANDB_MODE": "disabled",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "TMPDIR": "/root/autodl-tmp/tmp",
            }
        )
        stress_output = args.run_dir / "artifacts/stress"
        render_command = [
            "/root/autodl-tmp/envs/drivestudio/bin/python",
            str(project_root / "scripts/run_dr_v2_m5_stress_scene.py"),
            "--checkpoint",
            training["checkpoint"],
            "--registry",
            training["registry"],
            "--scene-name",
            scene,
            "--high-token",
            scene_protocol["high-support"],
            "--boundary-token",
            scene_protocol["boundary-support"],
            "--output-dir",
            str(stress_output),
            "--protocol",
            str(args.protocol),
        ]
        if args.frame_count is not None:
            render_command.extend(["--frame-count", str(args.frame_count)])
        render_stage = run_stage(
            run_dir=args.run_dir,
            stage="stress_render",
            command=render_command,
            environment=common,
            expected_report=stress_output / "report.json",
            minimum_disk_gib=30,
        )
        perception_output = args.run_dir / "artifacts/perception"
        perception_env = common.copy()
        perception_env.update(
            {
                "HF_HOME": "/root/autodl-tmp/hf_cache",
                "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        perception_command = [
            "/root/autodl-tmp/envs/adgs-sam/bin/python",
            str(project_root / "scripts/run_dr_v2_m5_perception.py"),
            "--scene-output",
            str(stress_output),
            "--protocol",
            str(args.protocol),
            "--output-dir",
            str(perception_output),
        ]
        if args.frame_count is not None:
            available_samples = [
                frame
                for frame in protocol["perception"]["sample_frames"]
                if frame < args.frame_count
            ]
            if not available_samples:
                available_samples = [0]
            perception_command.extend(
                ["--sample-frames", *[str(frame) for frame in available_samples]]
            )
        perception_stage = run_stage(
            run_dir=args.run_dir,
            stage="perception",
            command=perception_command,
            environment=perception_env,
            expected_report=perception_output / "report.json",
            minimum_disk_gib=20,
        )
        summary = {
            "status": "done",
            "scene": scene,
            "training_run": str(args.training_run),
            "stress_output": str(stress_output),
            "stress_report": str(stress_output / "report.json"),
            "perception_report": str(perception_output / "report.json"),
            "render_stage": render_stage,
            "perception_stage": perception_stage,
        }
        atomic_json(args.run_dir / "summary.json", summary)
        atomic_json(
            terminal,
            {"status": "done", "updated_at": now(), "failure": None},
            replace=True,
        )
        print(json.dumps(summary, sort_keys=True))
    except BaseException as error:
        atomic_json(
            terminal,
            {
                "status": "blocked",
                "updated_at": now(),
                "failure": {
                    "code": "M5_SCENE_EVALUATION_FAILED",
                    "detail": f"{type(error).__name__}: {error}",
                },
            },
            replace=True,
        )
        raise


if __name__ == "__main__":
    main()

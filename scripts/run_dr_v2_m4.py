#!/usr/bin/env python3
"""资源受控地执行并封存 M4 单场景编辑闭环。"""
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


TASK_ID = "DR-V2-M4-EDIT-PILOT-01"
DEFAULT_TOKEN = "af663976db5e412e83db033d309c5c29"
SOURCE_FILES = (
    "motion_proj/dynamic_editing_v2/drivestudio_registry.py",
    "motion_proj/dynamic_editing_v2/pilot_metrics.py",
    "scripts/run_dr_v2_m4.py",
    "scripts/run_dr_v2_m4_pilot.py",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖运行产物: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def replace_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def int_file(path: str) -> int | None:
    value = Path(path).read_text().strip()
    return None if value == "max" else int(value)


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
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
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
        "utilization_percent": int(fields[4]) if len(fields) > 4 else None,
    }


def sample(stage: str, event: str) -> dict:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": now(),
        "stage": stage,
        "event": event,
        "memory_current_bytes": int_file("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": int_file("/sys/fs/cgroup/memory.max"),
        "memory_events": memory_events(),
        "disk_free_bytes": disk.free,
        "gpu": gpu_sample(),
    }


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def validate_report(report: dict) -> None:
    if report.get("status") != "done" or report.get("task_id") != TASK_ID:
        raise RuntimeError("M4 report 状态或 task id 不正确")
    if report.get("scene") != "scene-0230":
        raise RuntimeError("M4 report 场景不正确")
    if report.get("instance_token") != DEFAULT_TOKEN:
        raise RuntimeError("M4 report actor 不是 M2 冻结 high-support token")
    if len(report.get("frames", [])) != 196 or report.get("cameras") != [0, 1, 2]:
        raise RuntimeError("M4 report frame/camera coverage 不完整")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RuntimeError(f"M4 report checks 失败: {checks}")
    if report.get("mask_truth_claim") is not False:
        raise RuntimeError("counterfactual effect mask 不得声称为真实观测")
    if report.get("quality_claim") is not False:
        raise RuntimeError("M4 自动运行不得预写质量结论")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--instance-token", default=DEFAULT_TOKEN)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/root/autodl-tmp/motion_proj")
    )
    parser.add_argument(
        "--environment", type=Path, default=Path("/root/autodl-tmp/envs/drivestudio")
    )
    args = parser.parse_args()

    if args.run_dir.exists():
        raise FileExistsError(f"M4 run 已存在: {args.run_dir}")
    if args.instance_token != DEFAULT_TOKEN:
        raise RuntimeError("M4 actor 已冻结，禁止替换 token")
    if not args.checkpoint.is_file() or not args.registry.is_file():
        raise FileNotFoundError("M4 checkpoint 或 registry 缺失")
    for directory in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / directory).mkdir(parents=True, exist_ok=True)
    for relative in SOURCE_FILES:
        source = args.project_root / relative
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    commit = subprocess.check_output(
        ["git", "-C", str(args.project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(args.project_root), "status", "--short"], text=True
    ).splitlines()
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": args.run_dir.name,
        "status": "running",
        "started_at": now(),
        "project_commit": commit,
        "project_git_status": git_status,
        "scene": "scene-0230",
        "scene_index": 179,
        "instance_token": args.instance_token,
        "baseline": "DriveStudio/StreetGS actor-aware native baseline",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "seed": 0,
        "edits": ["original", "actor_local_lateral_plus_1m", "delete"],
    }
    write_new_json(args.run_dir / "manifest.json", manifest)
    write_new_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    write_new_json(
        args.run_dir / "resolved.json",
        {
            "task_id": TASK_ID,
            "scene": "scene-0230",
            "scene_index": 179,
            "instance_token": args.instance_token,
            "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
            "frames": [0, 195],
            "fps": 10,
            "mask_provenance": "paired baseline counterfactual render difference",
            "depth_max_m": 120.0,
            "quality_thresholds": None,
        },
    )

    stage = "edit_pilot"
    resource_path = args.run_dir / "resource.jsonl"
    preflight = sample(stage, "preflight")
    append_jsonl(resource_path, preflight)
    if int(preflight["gpu"]["memory_used_mib"] or 0) > 2048:
        raise RuntimeError(f"M4 启动前 GPU 必须空闲: {preflight['gpu']}")
    if preflight["disk_free_bytes"] < 45 * 2**30:
        raise RuntimeError("M4 启动前数据盘可用空间必须不少于 45 GiB")
    baseline_events = preflight["memory_events"]
    output_dir = args.run_dir / "artifacts" / "pilot"
    command = [
        str(args.environment / "bin" / "python"),
        str(args.project_root / "scripts" / "run_dr_v2_m4_pilot.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--registry",
        str(args.registry),
        "--instance-token",
        args.instance_token,
        "--output-dir",
        str(output_dir),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{args.project_root}:/root/autodl-tmp/third_party/drivestudio",
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TMPDIR": "/root/autodl-tmp/tmp",
        }
    )
    started = time.monotonic()
    peak_gpu = 0
    peak_memory = int(preflight["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    log_path = args.run_dir / "logs" / "edit_pilot.log"
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=args.project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(10)
            current = sample(stage, "running")
            append_jsonl(resource_path, current)
            peak_gpu = max(peak_gpu, int(current["gpu"]["memory_used_mib"] or 0))
            peak_memory = max(peak_memory, int(current["memory_current_bytes"] or 0))
            maximum = current["memory_max_bytes"]
            used = current["memory_current_bytes"]
            over_memory = over_memory + 1 if maximum and used and used / maximum >= 0.90 else 0
            events = current["memory_events"]
            if over_memory >= 2:
                stop_reason = "memory.current/memory.max >= 0.90 twice"
            elif (
                events.get("oom", 0) > baseline_events.get("oom", 0)
                or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0)
            ):
                stop_reason = "cgroup oom event increased"
            elif current["disk_free_bytes"] < 15 * 2**30:
                stop_reason = "disk free below 15 GiB"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()
    append_jsonl(resource_path, sample(stage, "completed"))

    report_path = output_dir / "report.json"
    report = {}
    validation_error = None
    if return_code == 0 and stop_reason is None and report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_report(report)
        except Exception as error:  # noqa: BLE001 - 必须把 validator 原因写入 run。
            validation_error = f"{type(error).__name__}: {error}"
    else:
        validation_error = "child process, resource guard, or report existence check failed"
    status = "done" if validation_error is None else "blocked"
    stage_payload = {
        "stage": stage,
        "status": status,
        "return_code": return_code,
        "stop_reason": stop_reason,
        "validation_error": validation_error,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
        "command": command,
        "log": str(log_path),
        "output": str(output_dir),
        "report": str(report_path) if report_path.is_file() else None,
        "report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "metric_rows": report.get("metric_rows"),
        "checks": report.get("checks"),
    }
    write_new_json(args.run_dir / "stages" / "edit_pilot.json", stage_payload)
    if status != "done":
        replace_json(
            args.run_dir / "terminal.json",
            {
                "status": "blocked",
                "updated_at": now(),
                "failure": {
                    "code": "M4_PILOT_RUNTIME_OR_VALIDATION_FAILED",
                    "detail": validation_error or stop_reason,
                    "evidence": [str(log_path), str(args.run_dir / "stages/edit_pilot.json")],
                },
            },
        )
        raise SystemExit(return_code or 2)

    videos = sorted((output_dir / "variants").glob("*/videos/*.mp4"))
    artifacts = {
        "report": {
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "metrics_jsonl": {
            "path": str(output_dir / "metrics.jsonl"),
            "bytes": (output_dir / "metrics.jsonl").stat().st_size,
            "sha256": sha256_file(output_dir / "metrics.jsonl"),
        },
        "qa_html": {
            "path": str(output_dir / "qa/index.html"),
            "bytes": (output_dir / "qa/index.html").stat().st_size,
            "sha256": sha256_file(output_dir / "qa/index.html"),
        },
        "videos": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in videos
        ],
    }
    write_new_json(args.run_dir / "artifacts.json", artifacts)
    (args.run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# DR-V2 M4 单场景编辑闭环",
                "",
                "- status: `done`",
                "- scene: `scene-0230`",
                f"- actor: `{args.instance_token}`",
                "- coverage: `196 frames × 3 cameras × 3 variants`",
                f"- report: `{report_path}`",
                f"- QA: `{output_dir / 'qa/index.html'}`",
                "- 自动检查只证明协议与产物完整，不代表视觉质量通过。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    replace_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    print(json.dumps({"status": "done", "run_dir": str(args.run_dir), "report": str(report_path)}))


if __name__ == "__main__":
    main()

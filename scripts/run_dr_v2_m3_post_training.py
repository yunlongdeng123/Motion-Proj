#!/usr/bin/env python3
"""Build the M3 actor registry and execute the reversible edit smoke."""
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


SELECTED_TOKEN = "af663976db5e412e83db033d309c5c29"
SUCCESS = {"available", "done"}
REQUIRED_INPUT_STAGES = (
    "raw_prepare",
    "preprocess",
    "sky_masks",
    "train_profile100",
    "native_actor_mapping_probe",
    "train_profile1000",
    "train_formal",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refuse to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_input_stages(run_dir: Path) -> dict[str, dict]:
    result = {}
    for name in REQUIRED_INPUT_STAGES:
        path = run_dir / "stages" / f"{name}.json"
        if not path.is_file():
            raise RuntimeError(f"required input stage is missing: {path}")
        row = load_json(path)
        if row.get("status") not in SUCCESS:
            raise RuntimeError(f"required input stage failed: {name}: {row}")
        result[name] = row
    probe = result["native_actor_mapping_probe"]
    if probe.get("instance_token") != SELECTED_TOKEN:
        raise RuntimeError("native actor probe token does not match the M2-frozen actor")
    if int(probe.get("checkpoint_gaussian_count", 0)) <= 0:
        raise RuntimeError("native actor probe resolved an empty checkpoint slice")
    return result


def int_file(path: str) -> int | None:
    raw = Path(path).read_text().strip()
    return None if raw == "max" else int(raw)


def memory_events() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split()
            for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
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


def run_guarded(
    *,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    resource_path: Path,
    stage: str,
) -> dict:
    if log_path.exists():
        raise FileExistsError(f"refuse to overwrite log: {log_path}")
    pre = sample(stage, "preflight")
    append_jsonl(resource_path, pre)
    if int(pre["gpu"]["memory_used_mib"] or 0) > 2048:
        raise RuntimeError(f"GPU must be idle before {stage}: {pre['gpu']}")
    if pre["disk_free_bytes"] < 60 * 2**30:
        raise RuntimeError(f"disk free is below 60 GiB before {stage}")
    baseline_events = pre["memory_events"]
    peak_gpu = int(pre["gpu"]["memory_used_mib"] or 0)
    peak_memory = int(pre["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    started = time.monotonic()
    with log_path.open("wb") as log:
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
            current = sample(stage, "running")
            append_jsonl(resource_path, current)
            peak_gpu = max(peak_gpu, int(current["gpu"]["memory_used_mib"] or 0))
            peak_memory = max(peak_memory, int(current["memory_current_bytes"] or 0))
            maximum = current["memory_max_bytes"]
            used = current["memory_current_bytes"]
            over_memory = (
                over_memory + 1
                if maximum and used and used / maximum >= 0.90
                else 0
            )
            events = current["memory_events"]
            if over_memory >= 2:
                stop_reason = "memory ratio >= 0.90 twice"
            elif (
                events.get("oom", 0) > baseline_events.get("oom", 0)
                or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0)
            ):
                stop_reason = "cgroup oom event increased"
            elif current["disk_free_bytes"] < 20 * 2**30:
                stop_reason = "disk free below 20 GiB"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()
    append_jsonl(resource_path, sample(stage, "completed"))
    return {
        "return_code": return_code,
        "stop_reason": stop_reason,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
        "command": command,
        "log": str(log_path),
    }


def validate_smoke_report(report: dict, expected_images: int = 27) -> None:
    if report.get("status") != "done":
        raise RuntimeError(f"edit smoke is not done: {report.get('status')!r}")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RuntimeError(f"edit smoke checks failed: {checks}")
    images = report.get("images")
    if not isinstance(images, list) or len(images) != expected_images:
        raise RuntimeError(
            f"edit smoke image count mismatch: {len(images or [])} != {expected_images}"
        )
    variants = {row.get("variant") for row in images}
    if variants != {"original", "lateral_plus_1m", "remove"}:
        raise RuntimeError(f"edit smoke variants mismatch: {variants}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/root/autodl-tmp/motion_proj")
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    parser.add_argument(
        "--environment", type=Path, default=Path("/root/autodl-tmp/envs/drivestudio")
    )
    args = parser.parse_args()

    stages = require_input_stages(args.run_dir)
    checkpoint = Path(str(stages["train_formal"]["checkpoint"]))
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"formal checkpoint is missing: {checkpoint}")

    registry_path = args.run_dir / "artifacts" / "actor_registry.json"
    smoke_dir = args.run_dir / "artifacts" / "edit_smoke"
    readiness_path = args.run_dir / "baseline_readiness.json"
    for path in (registry_path, smoke_dir, readiness_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite post-training output: {path}")
    (args.run_dir / "artifacts").mkdir(exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{args.project_root}:{args.upstream_root}",
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TMPDIR": "/root/autodl-tmp/tmp",
        }
    )
    resource_path = args.run_dir / "resource.jsonl"
    registry_command = [
        str(args.environment / "bin" / "python"),
        str(args.project_root / "scripts" / "build_dr_v2_drivestudio_registry.py"),
        "--checkpoint",
        str(checkpoint),
        "--selected-token",
        SELECTED_TOKEN,
        "--output",
        str(registry_path),
    ]
    registry_run = run_guarded(
        command=registry_command,
        cwd=args.project_root,
        environment=environment,
        log_path=args.run_dir / "logs" / "actor_registry.log",
        resource_path=resource_path,
        stage="actor_registry",
    )
    registry_ok = registry_run["return_code"] == 0 and registry_path.is_file()
    registry = load_json(registry_path) if registry_ok else {}
    selected = registry.get("selected_smoke_actor", {})
    registry_ok = registry_ok and selected.get("instance_token") == SELECTED_TOKEN
    registry_stage = {
        "stage": "actor_registry",
        "status": "done" if registry_ok else "blocked",
        **registry_run,
        "registry": str(registry_path) if registry_path.is_file() else None,
        "registry_sha256": sha256_file(registry_path) if registry_path.is_file() else None,
        "actor_count": registry.get("actor_count"),
        "selected_actor": selected or None,
    }
    atomic_json(args.run_dir / "stages" / "actor_registry.json", registry_stage)
    if not registry_ok:
        raise SystemExit(registry_run["return_code"] or 2)

    smoke_command = [
        str(args.environment / "bin" / "python"),
        str(args.project_root / "scripts" / "run_dr_v2_drivestudio_edit_smoke.py"),
        "--checkpoint",
        str(checkpoint),
        "--registry",
        str(registry_path),
        "--instance-token",
        SELECTED_TOKEN,
        "--output-dir",
        str(smoke_dir),
        "--frames",
        "3",
        "--lateral-meters",
        "1.0",
    ]
    smoke_run = run_guarded(
        command=smoke_command,
        cwd=args.project_root,
        environment=environment,
        log_path=args.run_dir / "logs" / "edit_smoke.log",
        resource_path=resource_path,
        stage="edit_smoke",
    )
    report_path = smoke_dir / "edit_smoke_report.json"
    smoke_ok = smoke_run["return_code"] == 0 and report_path.is_file()
    report = load_json(report_path) if smoke_ok else {}
    if smoke_ok:
        try:
            validate_smoke_report(report)
        except RuntimeError:
            smoke_ok = False
    smoke_stage = {
        "stage": "edit_smoke",
        "status": "done" if smoke_ok else "blocked",
        **smoke_run,
        "report": str(report_path) if report_path.is_file() else None,
        "report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "image_count": len(report.get("images", [])),
        "checks": report.get("checks"),
        "diffs": report.get("diffs"),
    }
    atomic_json(args.run_dir / "stages" / "edit_smoke.json", smoke_stage)
    if not smoke_ok:
        raise SystemExit(smoke_run["return_code"] or 2)

    audit_command = [
        "/root/autodl-tmp/envs/motionproj/bin/python",
        str(args.project_root / "scripts" / "audit_dr_v2_m3_baseline.py"),
        "--output",
        str(readiness_path),
        "--checkpoint",
        str(checkpoint),
    ]
    audit_run = run_guarded(
        command=audit_command,
        cwd=args.project_root,
        environment=environment,
        log_path=args.run_dir / "logs" / "baseline_readiness_final.log",
        resource_path=resource_path,
        stage="baseline_readiness_final",
    )
    readiness = load_json(readiness_path) if readiness_path.is_file() else {}
    counts = readiness.get("status_counts", {})
    readiness_ok = (
        audit_run["return_code"] == 0
        and counts.get("missing") == 0
        and counts.get("incompatible") == 0
    )
    readiness_stage = {
        "stage": "baseline_readiness_final",
        "status": "done" if readiness_ok else "blocked",
        **audit_run,
        "readiness": str(readiness_path) if readiness_path.is_file() else None,
        "readiness_sha256": sha256_file(readiness_path) if readiness_path.is_file() else None,
        "status_counts": counts,
    }
    atomic_json(
        args.run_dir / "stages" / "baseline_readiness_final.json", readiness_stage
    )
    if not readiness_ok:
        raise SystemExit(audit_run["return_code"] or 2)

    metric_rows = [
        {
            "metric": "formal_checkpoint_bytes",
            "value": checkpoint.stat().st_size,
            "scene": "scene-0230",
            "instance_token": SELECTED_TOKEN,
        },
        {
            "metric": "lateral_mean_abs_rgb_diff",
            "value": report["diffs"]["lateral_plus_1m"]["mean_abs_rgb_diff"],
            "scene": "scene-0230",
            "instance_token": SELECTED_TOKEN,
        },
        {
            "metric": "remove_mean_abs_rgb_diff",
            "value": report["diffs"]["remove"]["mean_abs_rgb_diff"],
            "scene": "scene-0230",
            "instance_token": SELECTED_TOKEN,
        },
    ]
    metrics_path = args.run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refuse to overwrite metrics: {metrics_path}")
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "done",
                "checkpoint": str(checkpoint),
                "registry": str(registry_path),
                "edit_smoke": str(report_path),
                "readiness": str(readiness_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

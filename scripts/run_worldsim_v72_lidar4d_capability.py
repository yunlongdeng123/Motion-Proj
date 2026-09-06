#!/usr/bin/env python3
"""封装 LiDAR4D 官方命令，并写出可追溯的能力运行凭证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--runs-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v72")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()


def monitor_gpu(path: Path, stop: threading.Event, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                ).strip()
                timestamp, memory, utilization, power = [
                    item.strip() for item in output.split(",")
                ]
                sample = {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "nvidia_timestamp": timestamp,
                    "memory_used_mib": float(memory),
                    "utilization_percent": float(utilization),
                    "power_watts": float(power),
                }
                samples.append(sample)
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.flush()
            except Exception as error:  # 资源采样失败不能中断主训练。
                handle.write(
                    json.dumps(
                        {
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "monitor_error": repr(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
            stop.wait(5)


def parse_metrics(log_path: Path, output_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        output_path.write_text("", encoding="utf-8")
        return []
    patterns = {
        "raydrop": re.compile(r"Rdrop_error .* = \[([^]]+)\]"),
        "intensity": re.compile(r"Inten_error = \[([^]]+)\]"),
        "depth": re.compile(r"Depth_error = \[([^]]+)\]"),
        "points": re.compile(r"Point_error .* = \[([^]]+)\]"),
    }
    epoch = None
    current: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"Evaluate at epoch (\d+)", line)
        if match:
            epoch = int(match.group(1))
            current = {"epoch": epoch}
        for name, pattern in patterns.items():
            metric_match = pattern.search(line)
            if not metric_match:
                continue
            current[name] = [float(value) for value in metric_match.group(1).split()]
            if name == "points":
                records.append(dict(current))
                current = {"epoch": epoch}
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = config["task_id"]
    run_dir = (args.runs_root / task_id / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run 目录已存在：{run_dir}")
    run_dir.mkdir(parents=True)

    upstream = Path(config["upstream"]["checkout"]).resolve()
    expected_upstream_commit = config["upstream"]["commit"]
    actual_upstream_commit = git_commit(upstream)
    if actual_upstream_commit != expected_upstream_commit:
        raise RuntimeError(
            f"LiDAR4D commit 不匹配：{actual_upstream_commit} != {expected_upstream_commit}"
        )
    tcnn_checkout = upstream / "tiny-cuda-nn"
    actual_tcnn_commit = git_commit(tcnn_checkout)
    if actual_tcnn_commit != config["upstream"]["tiny_cuda_nn_commit"]:
        raise RuntimeError("tiny-cuda-nn commit 与冻结配置不一致")

    dataset_manifest = Path(config["dataset"]["manifest"]).resolve()
    upstream_config = upstream / config["upstream"]["config"]
    shutil.copy2(config_path, run_dir / "resolved_config.yaml")
    shutil.copy2(dataset_manifest, run_dir / "dataset_manifest.json")
    shutil.copy2(upstream_config, run_dir / "upstream_config.txt")

    motionproj_commit = git_commit(REPO_ROOT)
    python_path = Path(config["environment"]["python"]).resolve()
    freeze = subprocess.check_output(
        [str(python_path), "-m", "pip", "freeze"], text=True
    )
    (run_dir / "environment_freeze.txt").write_text(freeze, encoding="utf-8")

    training = config["training"]
    workspace = run_dir / "upstream_workspace"
    command = [
        str(python_path),
        "main_lidar4d.py",
        "--config",
        str(upstream_config),
        "--workspace",
        str(workspace),
        "--seed",
        str(config["seed"]),
        "--lr",
        str(training["learning_rate"]),
        "--num_rays_lidar",
        str(training["num_rays_lidar"]),
        "--iters",
        str(training["iterations"]),
        "--alpha_d",
        str(training["alpha_depth"]),
        "--alpha_i",
        str(training["alpha_intensity"]),
        "--alpha_r",
        str(training["alpha_raydrop"]),
    ]
    fingerprint = {
        "motionproj_commit": motionproj_commit,
        "upstream_commit": actual_upstream_commit,
        "tiny_cuda_nn_commit": actual_tcnn_commit,
        "resolved_config_sha256": sha256(run_dir / "resolved_config.yaml"),
        "dataset_manifest_sha256": sha256(dataset_manifest),
        "upstream_config_sha256": sha256(upstream_config),
        "environment_freeze_sha256": sha256(run_dir / "environment_freeze.txt"),
        "command": command,
    }
    write_json(run_dir / "fingerprint.json", fingerprint)

    started_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "worldsim_v72.formal_run.v1",
        "task_id": task_id,
        "run_id": args.run_id,
        "run_uri": f"run://worldsim_v72/{task_id}/{args.run_id}",
        "status": "running",
        "started_at": started_at.isoformat(),
        "evidence_role": config["evidence_role"],
        "route_selection_eligible": config["route_selection_eligible"],
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": None,
        "motionproj_source_test_read": config["motionproj_source_test_read"],
        "motionproj_external_test_read": config["motionproj_external_test_read"],
        "command": command,
    }
    write_json(run_dir / "manifest.json", manifest)

    environment = os.environ.copy()
    cuda_home = config["environment"]["cuda_home"]
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": config["environment"]["cuda_visible_devices"],
            "CUDA_HOME": cuda_home,
            "CMAKE_CUDA_COMPILER": f"{cuda_home}/bin/nvcc",
            "TORCH_HOME": config["environment"]["torch_home"],
            "TCNN_CUDA_ARCHITECTURES": "86",
            "PYTHONUNBUFFERED": "1",
            "PATH": f"{cuda_home}/bin:{python_path.parent}:{environment.get('PATH', '')}",
            "LD_LIBRARY_PATH": (
                f"{cuda_home}/lib:{cuda_home}/lib64:{python_path.parent.parent}/lib:"
                f"{environment.get('LD_LIBRARY_PATH', '')}"
            ),
        }
    )

    stop = threading.Event()
    samples: list[dict[str, Any]] = []
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(run_dir / "resource_samples.jsonl", stop, samples),
        daemon=True,
    )
    monitor.start()
    stdout_path = run_dir / "stdout.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=upstream,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
    finally:
        stop.set()
        monitor.join(timeout=15)

    finished_at = datetime.now(timezone.utc)
    upstream_log = workspace / "log_lidar4d.txt"
    metrics = parse_metrics(upstream_log, run_dir / "metrics.jsonl")
    checkpoints = []
    for path in sorted((workspace / "checkpoints").glob("*.pth")):
        checkpoints.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    summary = {
        "status": "done" if return_code == 0 else "failed",
        "return_code": return_code,
        "wall_seconds": (finished_at - started_at).total_seconds(),
        "gpu_sample_count": len(samples),
        "peak_gpu_memory_mib": max(
            (sample["memory_used_mib"] for sample in samples), default=None
        ),
        "mean_gpu_utilization_percent": (
            sum(sample["utilization_percent"] for sample in samples) / len(samples)
            if samples
            else None
        ),
        "last_metrics": metrics[-1] if metrics else None,
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
    }
    write_json(run_dir / "summary.json", summary)
    manifest.update(
        {
            "status": summary["status"],
            "finished_at": finished_at.isoformat(),
            "return_code": return_code,
            "summary": "summary.json",
            "failure_ledger_delta": config["failure_ledger_delta"]
            if return_code == 0
            else "pending_failure_ledger_update",
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()

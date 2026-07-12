#!/usr/bin/env python3
"""一次性健康检查；由 cron/tmux 每 5 分钟调用，不负责修改实验参数。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


TERMINAL_MARKERS = {
    "COMPLETE": "completed",
    "FAILED": "failed",
    "REJECTED": "rejected",
}


def inspect_run(
    root: Path,
    *,
    max_heartbeat_age: int = 180,
    max_log_age: int = 600,
    max_checkpoint_age: int = 1800,
    min_free_gb: float = 10,
    check_gpu: bool = True,
    now: float | None = None,
) -> dict[str, object]:
    """检查运行目录，并将已结束 run 与活跃性告警解耦。"""
    now = time.time() if now is None else now
    present_markers = [name for name in TERMINAL_MARKERS if (root / name).exists()]
    free_gb = shutil.disk_usage(root if root.exists() else root.parent).free / 2**30

    if len(present_markers) > 1:
        problems = ["conflicting terminal markers: " + ",".join(sorted(present_markers))]
        return {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "invalid",
            "terminal": True,
            "healthy": False,
            "problems": problems,
            "free_gb": round(free_gb, 2),
        }

    if present_markers:
        marker = present_markers[0]
        status = TERMINAL_MARKERS[marker]
        problems = ["run marked FAILED"] if marker == "FAILED" else []
        return {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "terminal": True,
            "healthy": not problems,
            "problems": problems,
            "free_gb": round(free_gb, 2),
        }

    problems: list[str] = []
    heartbeat = root / "heartbeat.json"
    if not heartbeat.exists() or now - heartbeat.stat().st_mtime > max_heartbeat_age:
        problems.append("heartbeat stale")
    logs = list((root / "logs").glob("*.log")) if (root / "logs").exists() else []
    if logs and now - max(path.stat().st_mtime for path in logs) > max_log_age:
        problems.append("log stale")
    checkpoints = list((root / "ckpts").glob("step_*/COMPLETE")) if (root / "ckpts").exists() else []
    if free_gb < min_free_gb:
        problems.append(f"disk low: {free_gb:.1f}GB")
    metrics = root / "metrics.jsonl"
    if metrics.exists() and checkpoints and now - max(path.stat().st_mtime for path in checkpoints) > max_checkpoint_age:
        problems.append("checkpoint stale")
    if metrics.exists():
        tail = metrics.read_text(encoding="utf-8", errors="replace")[-65536:].lower()
        if '"nan"' in tail or ": nan" in tail or '"inf"' in tail:
            problems.append("non-finite metric")
    if check_gpu:
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                capture_output=True,
                timeout=10,
            )
            if query.returncode != 0:
                problems.append("GPU unavailable")
        except (OSError, subprocess.TimeoutExpired):
            problems.append("nvidia-smi unavailable")

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "terminal": False,
        "healthy": not problems,
        "problems": problems,
        "free_gb": round(free_gb, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--max-heartbeat-age", type=int, default=180)
    parser.add_argument("--max-log-age", type=int, default=600)
    parser.add_argument("--max-checkpoint-age", type=int, default=1800)
    parser.add_argument("--min-free-gb", type=float, default=10)
    args = parser.parse_args()
    root = Path(args.run_dir)
    report = inspect_run(
        root,
        max_heartbeat_age=args.max_heartbeat_age,
        max_log_age=args.max_log_age,
        max_checkpoint_age=args.max_checkpoint_age,
        min_free_gb=args.min_free_gb,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

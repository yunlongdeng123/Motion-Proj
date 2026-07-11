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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--max-heartbeat-age", type=int, default=180)
    parser.add_argument("--max-log-age", type=int, default=600)
    parser.add_argument("--max-checkpoint-age", type=int, default=1800)
    parser.add_argument("--min-free-gb", type=float, default=10)
    args = parser.parse_args()
    root = Path(args.run_dir)
    problems: list[str] = []
    now = time.time()

    heartbeat = root / "heartbeat.json"
    if not heartbeat.exists() or now - heartbeat.stat().st_mtime > args.max_heartbeat_age:
        problems.append("heartbeat stale")
    logs = list((root / "logs").glob("*.log")) if (root / "logs").exists() else []
    if logs and now - max(path.stat().st_mtime for path in logs) > args.max_log_age:
        problems.append("log stale")
    checkpoints = list((root / "ckpts").glob("step_*/COMPLETE")) if (root / "ckpts").exists() else []
    free_gb = shutil.disk_usage(root if root.exists() else root.parent).free / 2**30
    if free_gb < args.min_free_gb:
        problems.append(f"disk low: {free_gb:.1f}GB")
    metrics = root / "metrics.jsonl"
    if metrics.exists() and checkpoints and now - max(path.stat().st_mtime for path in checkpoints) > args.max_checkpoint_age:
        problems.append("checkpoint stale")
    if metrics.exists():
        tail = metrics.read_text(encoding="utf-8", errors="replace")[-65536:].lower()
        if '"nan"' in tail or ": nan" in tail or '"inf"' in tail:
            problems.append("non-finite metric")
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10,
        )
        if query.returncode != 0:
            problems.append("GPU unavailable")
    except (OSError, subprocess.TimeoutExpired):
        problems.append("nvidia-smi unavailable")

    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "healthy": not problems,
              "problems": problems, "free_gb": round(free_gb, 2)}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

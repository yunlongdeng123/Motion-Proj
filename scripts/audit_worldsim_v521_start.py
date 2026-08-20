#!/usr/bin/env python3
"""执行 WorldSim V5.2.1 P0 provenance/resource/protocol freeze。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, inventory_files, sha256_file


class P0AuditError(RuntimeError):
    """P0 门禁失败。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: Sequence[str], *, check: bool = True) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise P0AuditError(f"命令失败 {command}: {completed.stderr.strip()}")
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _git_text(*args: str) -> str:
    return _run(["git", *args])["stdout"].strip()


def _read_text(path: str) -> str:
    target = PROJECT_ROOT / path
    if not target.is_file():
        raise P0AuditError(f"事实源缺失：{path}")
    return target.read_text(encoding="utf-8")


def _assert_contains(text: str, markers: Sequence[str], label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise P0AuditError(f"{label} 缺少冻结事实：{missing}")


def _cgroup() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in ("memory.max", "memory.current", "cpu.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        rows[name] = path.read_text(encoding="utf-8").strip() if path.is_file() else None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldsim_v521/p0_scope_freeze_v1.yaml")
    parser.add_argument("--run-root", default="/root/autodl-tmp/runs/worldsim_v521")
    parser.add_argument("--run-id")
    arguments = parser.parse_args()

    config_path = PROJECT_ROOT / arguments.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan_path = PROJECT_ROOT / config["plan"]["path"]
    if plan_path.stat().st_size != int(config["plan"]["bytes"]):
        raise P0AuditError("计划 bytes 漂移")
    if sha256_file(plan_path) != config["plan"]["sha256"]:
        raise P0AuditError("计划 SHA-256 漂移")

    branch = _git_text("branch", "--show-current")
    if branch != config["execution"]["required_branch"]:
        raise P0AuditError(f"执行分支错误：{branch}")
    dirty = _git_text("status", "--porcelain=v1")
    if dirty:
        raise P0AuditError("P0 正式 run 要求 clean worktree")

    status_text = _read_text("docs/RESEARCH_STATUS.md")
    experiments_text = _read_text("docs/EXPERIMENTS.md")
    failures_text = _read_text("docs/RESEARCH_FAILURES.md")
    closeout = yaml.safe_load(_read_text("configs/worldsim_v51/m1_closeout_v1.yaml"))
    archive_text = _read_text("docs/archive/2026-08/worldsim-v51-m1-closeout/README.md")

    _assert_contains(
        status_text,
        [
            "V5.2.1 Base Badcase Census 已授权、待执行",
            "executed=`false`",
            "fresh validation/test/KITTI method-tuning quality 全程 unread",
        ],
        "RESEARCH_STATUS",
    )
    _assert_contains(experiments_text, ["V5.2.1 Base Badcase Census 预注册"], "EXPERIMENTS")
    _assert_contains(failures_text, config["failure_ledger_refs"], "RESEARCH_FAILURES")
    _assert_contains(archive_text, ["Stage H", "execution=`false`"], "V5.1 closeout archive")

    expected = config["v51_terminal"]
    actual = {
        "terminal": closeout["outcome"],
        "stage_h_status": closeout["route_dispositions"]["stage_h_bki_graph_free"]["status"],
        "stage_h_executed": closeout["route_dispositions"]["stage_h_bki_graph_free"]["executed"],
        "stage_h_disposition": closeout["route_dispositions"]["stage_h_bki_graph_free"]["disposition"],
        "retained_comparator": "/".join(closeout["frozen_baseline"]["arms"]),
        "m2_status": closeout["scope"]["m2_status"],
        "m3_status": closeout["scope"]["m3_status"],
    }
    if actual != expected:
        raise P0AuditError(f"V5.1 terminal lock 漂移：{actual}")
    if closeout["scope"]["validation_read"] or closeout["scope"]["test_read"]:
        raise P0AuditError("V5.1 fresh quality lock 已被破坏")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = arguments.run_id or f"{stamp}__p0-base-census-freeze-s0-r001"
    run_root = Path(arguments.run_root)
    target = run_root / run_id
    partial = run_root / f"{run_id}.partial"
    if target.exists() or partial.exists():
        raise P0AuditError(f"run ID 已存在：{run_id}")
    partial.mkdir(parents=True)

    events = [{"event": "run_started", "at_utc": _utc_now()}]
    atomic_json(partial / "status.json", {"task_id": config["task_id"], "status": "running"})
    atomic_jsonl(partial / "events.jsonl", events)
    shutil.copy2(config_path, partial / "resolved_config.yaml")

    repo = {
        "head": _git_text("rev-parse", "HEAD"),
        "branch": branch,
        "status_porcelain": dirty,
        "last_commit": _git_text("log", "-1", "--decorate", "--oneline"),
        "upstream": _git_text("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False)["returncode"] == 0
        else None,
    }
    resource_commands = {
        "nvidia_smi": _run(["nvidia-smi"]),
        "gpu_query": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,temperature.gpu,pstate",
                "--format=csv",
            ]
        ),
        "disk": _run(["df", "-h", "/root/autodl-tmp"]),
        "memory": _run(["free", "-h"]),
    }
    resources = {"commands": resource_commands, "cgroup": _cgroup()}
    atomic_json(partial / "resource_telemetry.json", resources)

    inputs = [
        config["plan"]["path"],
        "docs/RESEARCH_STATUS.md",
        "docs/RESEARCH_FAILURES.md",
        "docs/EXPERIMENTS.md",
        "configs/worldsim_v51/m1_closeout_v1.yaml",
        "docs/archive/2026-08/worldsim-v51-m1-closeout/README.md",
    ]
    fingerprint = {
        "source_head": repo["head"],
        "inputs": [
            {
                "path": name,
                "bytes": (PROJECT_ROOT / name).stat().st_size,
                "sha256": sha256_file(PROJECT_ROOT / name),
            }
            for name in inputs
        ],
        "quality_bytes_decoded": 0,
        "quality_read_roles": [],
    }
    atomic_json(partial / "input_fingerprint.json", fingerprint)

    summary = {
        "task_id": config["task_id"],
        "status": "done",
        "outcome": "p0_gate_pass",
        "repo": repo,
        "plan": config["plan"],
        "v51_terminal": actual,
        "fresh_validation_read": False,
        "fresh_test_read": False,
        "kitti_method_tuning_quality_read": False,
        "stage_h_executed": False,
        "partition_membership_frozen": False,
        "next_task": "WS-V521-P1-BASE-ASSET-CENSUS-01",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
    }
    atomic_json(partial / "summary.json", summary)
    events.append({"event": "run_completed", "at_utc": _utc_now(), "outcome": "p0_gate_pass"})
    atomic_jsonl(partial / "events.jsonl", events)
    atomic_json(partial / "status.json", {"task_id": config["task_id"], "status": "done", "outcome": "p0_gate_pass"})
    manifest = {
        "schema": "worldsim_v521_run_manifest_v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "source_head": repo["head"],
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
        "inventory_before_manifest": inventory_files(partial),
    }
    atomic_json(partial / "run_manifest.json", manifest)
    partial.rename(target)
    print(json.dumps({"run_dir": str(target), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

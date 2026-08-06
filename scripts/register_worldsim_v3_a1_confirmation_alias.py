#!/usr/bin/env python
"""登记 A1 确认场景中 C*=C0 的只读 exact alias。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts.run_worldsim_v3_a0_smoke import atomic_json, command_output, now, sha256_file


PROJECT = Path("/root/autodl-tmp/motion_proj")
TASK_RUNS = Path("/root/autodl-tmp/runs/worldsim_v3/WS-V3-A1-CALIBRATION-01")
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_done(run_dir: Path) -> dict[str, Any]:
    terminal = load_json(run_dir / "terminal.json")
    summary = load_json(run_dir / "summary.json")
    if terminal.get("status") != "done" or summary.get("status") != "done":
        raise RuntimeError(f"source run 不是 done：{run_dir}")
    return summary


def resource_guard(run_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(run_root)
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if usage.free < 10 * 1024**3 or processes:
        raise RuntimeError(f"alias resource guard 失败：free={usage.free}, gpu={processes}")
    return {"free_disk_bytes": usage.free, "active_gpu_compute_processes": processes}


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("scene-0242", "scene-0255"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmation-config",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a1_confirmation_v1.yaml",
    )
    parser.add_argument("--task-runs-root", type=Path, default=TASK_RUNS)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    contract = yaml.safe_load(args.confirmation_config.read_text(encoding="utf-8"))
    scene = contract["scenes"][args.scene]
    if args.run_dir.name != scene["cstar_alias_run"]:
        raise RuntimeError("alias run ID 与冻结配置不一致")
    if contract["development_selection"]["selected_variant"] != "c0-off":
        raise RuntimeError("本 alias registrar 只允许冻结的 C*=C0")
    guard = resource_guard(args.run_dir.parent)

    train_dir = args.task_runs_root / scene["training_runs"]["c0-off"]
    endpoint_dir = args.task_runs_root / scene["endpoint_runs"]["c0-off"]
    selection_dir = args.task_runs_root / contract["development_selection"]["run_id"]
    train = require_done(train_dir)
    endpoint = require_done(endpoint_dir)
    selection = require_done(selection_dir)
    checkpoint = train["checkpoint"]
    endpoint_metrics = endpoint["endpoint_metrics"]
    if (
        train.get("scene_name") != args.scene
        or train.get("variant") != "c0-off"
        or train.get("formal") is not True
        or train.get("num_iters") != 30000
        or train.get("initialization_provenance", {}).get("sha256")
        != scene["expected_initialization_sha256"]
    ):
        raise RuntimeError("C0 training contract 不匹配")
    if (
        endpoint.get("variant") != "c0-off"
        or Path(endpoint["source_run_dir"]).name != train_dir.name
        or endpoint.get("endpoint_config_sha256") != contract["endpoint_config_sha256"]
        or endpoint_metrics.get("checkpoint_sha256_before") != checkpoint["sha256"]
        or endpoint_metrics.get("checkpoint_sha256_after") != checkpoint["sha256"]
    ):
        raise RuntimeError("C0 endpoint/checkpoint contract 不匹配")
    if (
        selection.get("selected_variant") != "c0-off"
        or selection.get("decision_status") != "done_off"
        or selection.get("selection_config_sha256")
        != contract["development_selection"]["config_sha256"]
    ):
        raise RuntimeError("development selection contract 不匹配")

    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.confirmation_config, args.run_dir / "resolved.yaml")
    source = PROJECT / "scripts/register_worldsim_v3_a1_confirmation_alias.py"
    snapshot = args.run_dir / "source_snapshot/scripts" / source.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)
    atomic_json(args.run_dir / "terminal.json", {"status": "running", "updated_at": now(), "failure": None})
    commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    evidence = {
        "training_summary_sha256": sha256_file(train_dir / "summary.json"),
        "endpoint_summary_sha256": sha256_file(endpoint_dir / "summary.json"),
        "selection_summary_sha256": sha256_file(selection_dir / "summary.json"),
    }
    fingerprint = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "project_commit": commit,
        "confirmation_config_sha256": sha256_file(args.run_dir / "resolved.yaml"),
        "source_sha256": sha256_file(source),
        "evidence": evidence,
    }
    atomic_json(args.run_dir / "fingerprint.json", fingerprint)
    manifest = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "component": "A1 confirmation exact alias registration",
        "status": "running",
        "scene_name": args.scene,
        "scene_index": scene["scene_index"],
        "seed": 0,
        "logical_variant": "c-star",
        "alias_of": "c0-off",
        "source_training_run": str(train_dir),
        "source_endpoint_run": str(endpoint_dir),
        "development_selection_run": str(selection_dir),
        "resource_guard": guard,
        "project_commit": commit,
        "project_status": command_output("git", "status", "--short", cwd=PROJECT).splitlines(),
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    alias = {
        "status": "done",
        "truth_tier": "exact_logical_alias_no_new_training_or_evaluation",
        "scene_name": args.scene,
        "scene_index": scene["scene_index"],
        "logical_variant": "c-star",
        "selected_variant": "c0-off",
        "alias_of": "c0-off",
        "exact_alias": True,
        "new_training_performed": False,
        "new_evaluation_performed": False,
        "source_training_run": str(train_dir),
        "source_endpoint_run": str(endpoint_dir),
        "source_checkpoint": checkpoint,
        "endpoint_config_sha256": endpoint["endpoint_config_sha256"],
        "development_selection_run": str(selection_dir),
        "development_selection_config_sha256": selection["selection_config_sha256"],
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "metrics.json", alias)
    atomic_json(args.run_dir / "artifacts.json", {"source_training_run": str(train_dir), "source_endpoint_run": str(endpoint_dir)})
    summary = dict(alias)
    summary["task_id"] = contract["task_id"]
    summary["component"] = "A1 confirmation C* exact alias"
    summary["confirmation_config_sha256"] = fingerprint["confirmation_config_sha256"]
    atomic_json(args.run_dir / "summary.json", summary)
    (args.run_dir / "summary.md").write_text(
        f"# {args.scene} A1 C* alias\n\nC* 是 C0 checkpoint 的 exact alias；未执行新训练或评测。\n",
        encoding="utf-8",
    )
    atomic_json(args.run_dir / "terminal.json", {"status": "done", "updated_at": now(), "failure": None})
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.is_dir() and not _TERMINAL_FINAL:
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {"status": "blocked", "updated_at": now(), "failure": {"code": "A1_ALIAS_FAILED", "detail": f"{type(error).__name__}: {error}"}},
            )
        raise

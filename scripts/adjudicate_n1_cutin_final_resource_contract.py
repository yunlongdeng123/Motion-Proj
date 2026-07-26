#!/usr/bin/env python
"""将 final cut-in 的关键资源合同失败登记为独立的拒绝裁决 run。

该工具只冻结父 formal run 已经记录的 preflight 和失败事实；它不修改父 run、
不读取 evaluation scene、不填写人工 verdict，也不会授权或启动 N2。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now


_TERMINALS = ("COMPLETE", "REJECTED", "FAILED", "AWAITING_HUMAN_REVIEW")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是 object: {path}")
    return value


def _unique_parent_terminal(parent: Path) -> str:
    markers = [name for name in _TERMINALS if (parent / name).is_file()]
    if len(markers) != 1:
        raise RuntimeError(f"父 run 终态必须唯一: {markers}")
    return markers[0]


def _parent_snapshot(parent: Path) -> dict[str, str]:
    required = (
        "preflight.json",
        "failure.json",
        "resolved.yaml",
        "stages/K4_REGRESSION.json",
    )
    snapshot: dict[str, str] = {}
    for relative in required:
        path = parent / relative
        if not path.is_file():
            raise RuntimeError(f"缺少父 formal 证据: {path}")
        snapshot[relative] = file_fingerprint(str(path))
    return snapshot


def _validate_parent(config: dict[str, Any], parent: Path) -> dict[str, Any]:
    terminal = _unique_parent_terminal(parent)
    if terminal != str(config["parent_expected_terminal"]):
        raise RuntimeError(f"父 formal 终态不匹配: {terminal}")

    snapshot = _parent_snapshot(parent)
    expected_snapshot = {
        str(key): str(value)
        for key, value in dict(config["parent_artifact_sha256"]).items()
    }
    if snapshot != expected_snapshot:
        raise RuntimeError("父 formal 证据 SHA256 不匹配")

    preflight = _load_json(parent / "preflight.json")
    failure = _load_json(parent / "failure.json")
    k4 = _load_json(parent / "stages" / "K4_REGRESSION.json")
    if preflight.get("formal") is not True:
        raise RuntimeError("父 run 不是 formal")
    if preflight.get("code_dirty") is not False:
        raise RuntimeError("父 formal run 代码状态不干净")
    if str(preflight.get("code_commit")) != str(config["expected_parent_code_commit"]):
        raise RuntimeError("父 formal code commit 不匹配")
    if failure.get("terminal_status") != "FAILED":
        raise RuntimeError("父 failure 不是 FAILED")
    if failure.get("exit_reason") != str(config["expected_parent_failure_exit_reason"]):
        raise RuntimeError("父 failure exit reason 不匹配")
    if k4.get("passed") is not True or k4.get("n2_authorized") is not False:
        raise RuntimeError("父 run 的 K4 regression 未通过或错误授权 N2")

    memory = preflight.get("memory")
    if not isinstance(memory, dict):
        raise RuntimeError("父 preflight 缺少 memory")
    observed = int(memory["cgroup_memory_current_bytes"])
    expected_observed = int(
        config["resource_contract"]["expected_parent_preflight_cgroup_current_bytes"]
    )
    limit = int(config["resource_contract"]["max_start_cgroup_current_bytes"])
    if observed != expected_observed:
        raise RuntimeError("父 preflight cgroup current 与冻结值不匹配")
    if observed <= limit:
        raise RuntimeError("父 preflight 并未违反启动资源合同，拒绝裁决 fail-closed")

    return {
        "parent_terminal": terminal,
        "parent_snapshot": snapshot,
        "preflight": preflight,
        "failure": failure,
        "k4_regression": k4,
        "max_start_cgroup_current_bytes": limit,
        "observed_preflight_cgroup_current_bytes": observed,
        "start_contract_excess_bytes": observed - limit,
    }


def adjudicate(config_path: Path, output_root: Path | None = None) -> Path:
    config = _load_yaml(config_path)
    parent = Path(config["parent_run"])
    parent_evidence = _validate_parent(config, parent)
    decision = dict(config["decision"])
    if decision.get("terminal_status") != "REJECTED":
        raise RuntimeError("resource-contract adjudication 只能登记 REJECTED")
    if decision.get("research_verdict") != "stop_nuscenes_cutin_mining":
        raise RuntimeError("resource-contract adjudication 的终局必须停止 cut-in mining")
    if decision.get("n2_authorized") is not False:
        raise RuntimeError("resource-contract adjudication 必须保持 N2 未授权")

    code = git_state(str(PROJECT_ROOT))
    if bool(config.get("require_clean_git", True)) and code["dirty"]:
        raise RuntimeError("正式 resource-contract adjudication 必须在 clean git worktree 运行")

    config_sha = file_fingerprint(str(config_path))
    run_id = generate_run_id(
        str(config["task_id"]), "resource-contract-reject-v1", int(config["seed"]), config_sha
    )
    run_dir = (output_root or Path(config["run_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    data_fingerprint = canonical_sha256(
        {
            "parent_run": str(parent),
            "parent_snapshot": parent_evidence["parent_snapshot"],
            "preflight_memory": parent_evidence["preflight"]["memory"],
            "k4_report_sha256": parent_evidence["parent_snapshot"]["stages/K4_REGRESSION.json"],
            "decision": decision,
        }
    )
    summary = {
        "schema_version": str(config["schema_version"]),
        "task_id": str(config["task_id"]),
        "run_id": run_id,
        "parent_run": str(parent),
        "parent_terminal": parent_evidence["parent_terminal"],
        "parent_snapshot": parent_evidence["parent_snapshot"],
        "parent_k4_regression_passed": True,
        "parent_formal_code_clean": True,
        "evaluation_scene_count_started": 0,
        "strict_pass_candidate_count": 0,
        "human_audit_created": False,
        "human_verdict_filled": False,
        "resource_contract": {
            "max_start_cgroup_current_bytes": parent_evidence[
                "max_start_cgroup_current_bytes"
            ],
            "observed_preflight_cgroup_current_bytes": parent_evidence[
                "observed_preflight_cgroup_current_bytes"
            ],
            "start_contract_excess_bytes": parent_evidence["start_contract_excess_bytes"],
            "parent_runner_process_rss_bytes": int(
                parent_evidence["preflight"]["memory"]["process_rss_bytes"]
            ),
        },
        "terminal_status": decision["terminal_status"],
        "research_verdict": decision["research_verdict"],
        "decision_source": "frozen formal preflight resource contract; no human verdict inferred",
        "n2_authorized": False,
    }
    ended_at = utc_now()
    manifest = {
        "schema_version": 1,
        "task_id": str(config["task_id"]),
        "run_id": run_id,
        "command": list(sys.argv),
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_sha,
        "data_fingerprint": data_fingerprint,
        "parent_run": str(parent),
        "parent_code_commit": parent_evidence["preflight"]["code_commit"],
        "parent_snapshot": parent_evidence["parent_snapshot"],
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": started_at,
        "ended_at": ended_at,
        "terminal_status": decision["terminal_status"],
        "exit_reason": decision["research_verdict"],
        "n2_authorized": False,
    }
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    atomic_write_json(str(run_dir / "parent_evidence.json"), parent_evidence)
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(str(run_dir / "REJECTED"), str(decision["research_verdict"]) + "\n")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/n1_cutin_final_resource_rejection_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    adjudicate(args.config, args.output_root)


if __name__ == "__main__":
    main()

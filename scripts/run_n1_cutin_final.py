#!/usr/bin/env python
"""编排 final cut-in 的 mining worker 与独立 audit worker。"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.io_memory import memory_snapshot, page_cache_control_available
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _terminal(run_dir: Path, status: str, reason: str) -> None:
    running = run_dir / "RUNNING"
    if running.is_file():
        running.unlink()
    atomic_write_text(str(run_dir / status), reason + "\n")


def _run_worker(command: list[str], run_dir: Path) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"worker 失败，exit code={completed.returncode}")


def _load_k4_report(run_dir: Path) -> dict:
    path = run_dir / "stages" / "K4_REGRESSION.json"
    if not path.is_file():
        raise RuntimeError("缺少已冻结的 K4 regression report")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("passed") is not True:
        raise RuntimeError("K4 regression 未通过")
    return value


def run(
    config_path: Path,
    output_root: Path | None = None,
    *,
    allow_dirty_development: bool = False,
    max_evaluation_scenes: int | None = None,
    skip_audit: bool = False,
    skip_k4_regression: bool = False,
) -> Path:
    config = _load_config(config_path)
    formal = not allow_dirty_development
    # final config 是冻结的、可在 clean worktree 复用的研究输入；因此真正要
    # 检查的代码树是当前 orchestrator 所在 worktree，而不是配置里历史写死的
    # shared checkout 路径。这样不会为绕过 dirty worktree 而改写 config。
    configured_repo_root = Path(config["repo_root"]).resolve()
    code = git_state(str(PROJECT_ROOT))
    if formal and bool(config.get("require_clean_git", True)) and code["dirty"]:
        raise RuntimeError("final formal 必须在 clean git worktree 运行")
    if formal and max_evaluation_scenes is not None:
        raise RuntimeError("formal 不允许截断 evaluation scenes")
    if formal and skip_audit:
        raise RuntimeError("formal 不允许跳过 audit worker")
    if formal and skip_k4_regression:
        raise RuntimeError("formal 不允许跳过 K4 regression")
    if config["runtime"].get("require_posix_page_cache_control") and not page_cache_control_available():
        raise RuntimeError("final runtime 需要 POSIX page-cache control")

    config_sha = file_fingerprint(str(config_path))
    run_id = generate_run_id(
        config["task_id"],
        "receiver-cutin-final-v1" if formal else "receiver-cutin-final-v1-dev",
        int(config["seed"]),
        config_sha,
    )
    run_dir = (output_root or Path(config["run_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    atomic_write_text(str(run_dir / "RUNNING"), "n1_cutin_final_orchestrator\n")
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    preflight = {
        "memory": memory_snapshot(),
        "page_cache_control_available": page_cache_control_available(),
        "formal": formal,
        "configured_repo_root": str(configured_repo_root),
        "executing_repo_root": str(PROJECT_ROOT),
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
    }
    atomic_write_json(str(run_dir / "preflight.json"), preflight)
    try:
        if not skip_k4_regression:
            k4_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "replay_n1_cutin_k4_evidence.py"),
                "--fixture",
                str(PROJECT_ROOT / config["calibration"]["fourth_review"]["fixture_dir"]),
                "--output-root",
                str(run_dir / "stages" / "k4-evidence"),
                "--report",
                str(run_dir / "stages" / "K4_REGRESSION.json"),
            ]
            _run_worker(k4_command, run_dir)
            _load_k4_report(run_dir)
        mining_command = [
            sys.executable,
            str(PROJECT_ROOT / "resim" / "event_first_n1_cutin.py"),
            "--internal-final-mining",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ]
        if max_evaluation_scenes is not None:
            mining_command.extend(["--max-evaluation-scenes", str(max_evaluation_scenes)])
        _run_worker(mining_command, run_dir)
        mining = json.loads((run_dir / "mining_summary.json").read_text(encoding="utf-8"))
        readiness = dict(mining["machine_readiness_checks"])
        readiness["k4_regression"] = not skip_k4_regression
        machine_ready = all(readiness.values())
        if not machine_ready:
            terminal = "REJECTED"
            reason = "stop_nuscenes_cutin_mining_too_sparse"
            audit_manifest = None
        elif skip_audit:
            terminal = "REJECTED"
            reason = "development_mining_only_no_human_audit"
            audit_manifest = None
        else:
            audit_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_n1_cutin_audit.py"),
                "--run-dir",
                str(run_dir),
                "--config",
                str(config_path),
            ]
            _run_worker(audit_command, run_dir)
            audit_manifest = json.loads(
                (run_dir / "audit" / "audit_manifest.json").read_text(encoding="utf-8")
            )
            atomic_write_text(str(run_dir / "stages" / "AUDIT_COMPLETE"), "audit_complete\n")
            terminal = "AWAITING_HUMAN_REVIEW"
            reason = "await_human_review_n1_receiver_cutin_final"
        summary = {
            "schema_version": "n1-cutin-final-summary-v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "seed": int(config["seed"]),
            "formal": formal,
            "calibration_scene_count": mining["calibration_scene_count"],
            "evaluation_scene_count": mining["evaluation_scene_count"],
            "strict_pass_candidate_count": mining["strict_pass_candidate_count"],
            "strict_pass_scene_count": mining["strict_pass_scene_count"],
            "machine_readiness_checks": readiness,
            "machine_readiness_passed": machine_ready,
            "audit_ready": audit_manifest is not None,
            "human_verdict_filled": False,
            "n2_authorized": False,
            "terminal_status": terminal,
            "research_verdict": reason,
            "strict_event_pool_sha256": json.loads(
                (run_dir / "strict_event_pool.json").read_text(encoding="utf-8")
            )["strict_event_pool_sha256"],
        }
        artifacts = {
            relative: file_fingerprint(str(run_dir / relative))
            for relative in (
                "calibration_audit.json",
                "scene_metrics.jsonl",
                "transition_diagnostics.jsonl",
                "strict_candidates.jsonl",
                "strict_event_pool.json",
                "mining_summary.json",
            )
        }
        if audit_manifest is not None:
            artifacts["audit/audit_manifest.json"] = file_fingerprint(
                str(run_dir / "audit" / "audit_manifest.json")
            )
        manifest = {
            "schema_version": 1,
            "task_id": config["task_id"],
            "run_id": run_id,
            "command": list(sys.argv),
            "formal": formal,
            "code_commit": code["commit"],
            "code_dirty": code["dirty"],
            "configured_repo_root": str(configured_repo_root),
            "executing_repo_root": str(PROJECT_ROOT),
            "dirty_diff_hash": code["dirty_diff_hash"],
            "config_fingerprint": config_sha,
            "data_fingerprint": canonical_sha256(
                {
                    "calibration_provenance": json.loads(
                        (run_dir / "calibration_audit.json").read_text(encoding="utf-8")
                    )["provenance"],
                    "evaluation_scene_count": mining["evaluation_scene_count"],
                }
            ),
            "artifact_hashes": artifacts,
            "artifact_set_sha256": canonical_sha256(artifacts),
            "seed": int(config["seed"]),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "started_at": started_at,
            "ended_at": utc_now(),
            "terminal_status": terminal,
            "exit_reason": reason,
            "n2_authorized": False,
        }
        atomic_write_json(str(run_dir / "summary.json"), summary)
        atomic_write_json(str(run_dir / "manifest.json"), manifest)
        _terminal(run_dir, terminal, reason)
        print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
        return run_dir
    except BaseException as exc:
        failure = {
            "schema_version": "n1-cutin-final-engineering-failure-v1",
            "task_id": config["task_id"],
            "terminal_status": "FAILED",
            "exit_reason": "orchestrator_or_worker_failure",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "n2_authorized": False,
        }
        atomic_write_json(str(run_dir / "failure.json"), failure)
        _terminal(run_dir, "FAILED", failure["exit_reason"])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n1_cutin_final_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-dirty-development", action="store_true")
    parser.add_argument("--max-evaluation-scenes", type=int)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-k4-regression", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.output_root,
        allow_dirty_development=args.allow_dirty_development,
        max_evaluation_scenes=args.max_evaluation_scenes,
        skip_audit=args.skip_audit,
        skip_k4_regression=args.skip_k4_regression,
    )


if __name__ == "__main__":
    main()

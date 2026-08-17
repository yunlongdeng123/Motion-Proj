#!/usr/bin/env python3
"""核对 V5.1 M1-only 的 P0、开发角色与 V5 canonical 输入。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
    validate_development_roles,
    validate_scope,
    verify_canonical_run,
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def run(scope_path: Path, roles_path: Path, baselines_path: Path) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行，当前为 {branch}")
    if status:
        raise ProtocolError("P0 formal audit 要求 clean worktree")
    scope = load_yaml(scope_path)
    roles = load_yaml(roles_path)
    baselines = load_yaml(baselines_path)
    scope_report = validate_scope(PROJECT, scope)
    roles_report = validate_development_roles(PROJECT, roles)
    if baselines.get("schema_version") != "worldsim_v51_m1_unary_baselines_v1":
        raise ProtocolError("A0 baseline schema 漂移")
    if not baselines.get("failure_ledger_refs"):
        raise ProtocolError("A0 baseline 缺 failure_ledger_refs")
    runs = [
        verify_canonical_run(scene, spec)
        for scene, spec in baselines["canonical_runs"].items()
    ]
    return {
        "schema_version": "worldsim_v51_start_audit_v1",
        "task_ids": [scope["task_id"], roles["task_id"]],
        "status": "done",
        "conclusion": "v51_m1_scope_roles_and_v5_inputs_frozen",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "scope_config_sha256": sha256_file(scope_path),
        "development_roles_config_sha256": sha256_file(roles_path),
        "baseline_config_sha256": sha256_file(baselines_path),
        "scope": scope_report,
        "development_roles": roles_report,
        "canonical_runs": runs,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "method_inference_started": False,
        "parameter_search_performed": False,
        "failure_ledger_refs": sorted(
            set(scope["failure_ledger_refs"])
            | set(roles["failure_ledger_refs"])
            | set(baselines["failure_ledger_refs"])
        ),
        "failure_ledger_delta": "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/p0_m1_scope_v1.yaml",
    )
    parser.add_argument(
        "--roles",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/development_roles_v1.yaml",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/m1_unary_baselines_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_json(run_dir / "events.json", {"events": events})
    try:
        result = run(
            args.scope.resolve(), args.roles.resolve(), args.baselines.resolve()
        )
        result["created_at_utc"] = _utc_now()
        _write_json(run_dir / "summary.json", result)
        fingerprint = {
            "schema_version": "worldsim_v51_start_audit_fingerprint_v1",
            "task_ids": result["task_ids"],
            "source_commit": result["source_commit"],
            "source_branch": result["source_branch"],
            "worktree_clean": result["worktree_clean"],
            "scope_config": {
                "path": str(args.scope.resolve()),
                "sha256": result["scope_config_sha256"],
            },
            "development_roles_config": {
                "path": str(args.roles.resolve()),
                "sha256": result["development_roles_config_sha256"],
            },
            "baseline_config": {
                "path": str(args.baselines.resolve()),
                "sha256": result["baseline_config_sha256"],
            },
        }
        _write_json(run_dir / "fingerprint.json", fingerprint)
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_json(run_dir / "events.json", {"events": events})
        manifest = {
            "schema_version": "worldsim_v51_start_audit_manifest_v1",
            "task_ids": result["task_ids"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        status = {
            "schema_version": "worldsim_v51_start_audit_status_v1",
            "task_ids": result["task_ids"],
            "status": "done",
            "source_commit": result["source_commit"],
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "finished_at_utc": _utc_now(),
        }
        _write_json(run_dir / "status.json", status)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(run_dir / "events.json", {"events": events})
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_start_audit_status_v1",
                "task_ids": [
                    "WS-V51-P0-M1-SCOPE-FREEZE-01",
                    "WS-V51-D0-DEV-ROLE-FREEZE-01",
                ],
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()

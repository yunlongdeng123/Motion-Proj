#!/usr/bin/env python
"""WorldSim V3 A1-E0 正式只读回填控制器。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v3.calibration_endpoints import validate_endpoint_contract
from scripts.run_worldsim_v3_a0_scene import common_environment, run_stage
from scripts.run_worldsim_v3_a0_smoke import (
    DRIVESTUDIO_ENV,
    PATCHED_DRIVESTUDIO,
    PROJECT,
    atomic_json,
    command_output,
    now,
    sha256_file,
)


_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False
ROLE_NAMES = ("high-support", "boundary-support")


def resource_guard(run_root: Path) -> dict[str, object]:
    usage = shutil.disk_usage(run_root)
    if usage.free < 10 * 1024**3:
        raise RuntimeError(f"less than 10 GiB free below {run_root}")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f"GPU has active compute processes: {processes}")
    return {
        "free_disk_bytes": usage.free,
        "minimum_free_disk_bytes": 10 * 1024**3,
        "active_gpu_compute_processes": processes,
    }


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint-config",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a1_endpoints_v1.yaml",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument(
        "--max-heldout-frames",
        type=int,
        help="仅供工程 smoke；正式回填必须省略。",
    )
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    source_summary_path = args.source_run_dir / "summary.json"
    source_terminal_path = args.source_run_dir / "terminal.json"
    if not source_summary_path.is_file() or not source_terminal_path.is_file():
        raise FileNotFoundError("source A1 run is missing summary.json or terminal.json")
    source_terminal = json.loads(source_terminal_path.read_text(encoding="utf-8"))
    if source_terminal.get("status") != "done":
        raise RuntimeError("source A1 run is not terminal done")
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    endpoint_contract = yaml.safe_load(args.endpoint_config.read_text(encoding="utf-8"))
    validate_endpoint_contract(endpoint_contract)
    guard = resource_guard(args.run_dir.parent)

    _ACTIVE_RUN_DIR = args.run_dir
    for name in (
        "artifacts",
        "environment",
        "logs",
        "source_snapshot",
        "stages",
    ):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.endpoint_config, args.run_dir / "resolved_config.yaml")
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    sources = (
        PROJECT / "scripts/run_worldsim_v3_a1_endpoints.py",
        PROJECT / "scripts/eval_worldsim_v3_a1_endpoints.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "motion_proj/worldsim_v3/calibration_endpoints.py",
        args.endpoint_config,
    )
    source_hashes: dict[str, str] = {}
    for source_path in sources:
        relative = source_path.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        source_hashes[str(relative)] = sha256_file(source_path)

    output_dir = args.run_dir / "artifacts" / "a1_endpoints"
    command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/eval_worldsim_v3_a1_endpoints.py"),
        "--source-summary",
        str(source_summary_path),
        "--endpoint-config",
        str(args.run_dir / "resolved_config.yaml"),
        "--output-dir",
        str(output_dir),
        "--drivestudio-root",
        str(PATCHED_DRIVESTUDIO),
    ]
    if args.max_heldout_frames is not None:
        command.extend(["--max-heldout-frames", str(args.max_heldout_frames)])
    commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    fingerprint = {
        "schema_version": 1,
        "task_id": endpoint_contract["task_id"],
        "project_commit": commit,
        "source_hashes": source_hashes,
        "source_summary_sha256": sha256_file(source_summary_path),
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "endpoint_config_sha256": sha256_file(args.run_dir / "resolved_config.yaml"),
    }
    atomic_json(args.run_dir / "fingerprint.json", fingerprint)
    manifest = {
        "schema_version": 1,
        "task_id": endpoint_contract["task_id"],
        "component": "A1-E0 cross-camera and actor-boundary endpoint backfill",
        "status": "running",
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "variant": source.get("variant"),
        "seed": 0,
        "split": "heldout test split",
        "formal_full_split": args.max_heldout_frames is None,
        "source_run_dir": str(args.source_run_dir),
        "source_summary_sha256": sha256_file(source_summary_path),
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "project_commit": commit,
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "resource_guard": guard,
        "command": command,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    def validate() -> tuple[bool, dict[str, object]]:
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file() or summary_path.stat().st_size == 0:
            return False, {"summary": str(summary_path), "exists": False}
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        roles = payload.get("e2", {}).get("roles", {})
        valid_statuses = {"done", "ABSTAIN"}
        ok = (
            payload.get("status") == "done"
            and payload.get("heldout_split", {}).get("formal_full_split")
            == (args.max_heldout_frames is None)
            and payload.get("e1", {}).get("status") in valid_statuses
            and set(roles) == set(ROLE_NAMES)
            and all(row.get("status") in valid_statuses for row in roles.values())
            and payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after")
        )
        return ok, {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "e1_status": payload.get("e1", {}).get("status"),
            "e2_role_statuses": {
                name: row.get("status") for name, row in roles.items()
            },
            "checkpoint_unchanged": payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after"),
        }

    stage = run_stage(
        run_dir=args.run_dir,
        stage="a1_e0_endpoints",
        command=command,
        cwd=PROJECT,
        environment=common_environment(),
        validate=validate,
        timeout_seconds=args.timeout_seconds,
    )
    endpoint_summary_path = output_dir / "summary.json"
    endpoint_summary = json.loads(endpoint_summary_path.read_text(encoding="utf-8"))
    shutil.copy2(endpoint_summary_path, args.run_dir / "metrics.json")
    summary = {
        "status": "done",
        "task_id": endpoint_contract["task_id"],
        "component": "A1-E0 endpoint backfill",
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "variant": source.get("variant"),
        "seed": 0,
        "split": "heldout test split",
        "source_run_dir": str(args.source_run_dir),
        "source_checkpoint": source["checkpoint"],
        "endpoint_config_sha256": fingerprint["endpoint_config_sha256"],
        "endpoint_metrics": endpoint_summary,
        "resources": {
            key: stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        },
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    manifest["status"] = "done"
    manifest["completed_at"] = summary["completed_at"]
    atomic_json(args.run_dir / "manifest.json", manifest)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.is_dir() and not _TERMINAL_FINAL:
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "code": "A1_E0_ENDPOINTS_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

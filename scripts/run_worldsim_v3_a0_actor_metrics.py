#!/usr/bin/env python
"""Formal run controller for WorldSim V3 A0 actor-region diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.run_worldsim_v3_a0_scene import common_environment, run_stage
from scripts.run_worldsim_v3_a0_smoke import (
    DRIVESTUDIO_ENV,
    PATCHED_DRIVESTUDIO,
    PROJECT,
    TASK_ID,
    atomic_json,
    command_output,
    now,
    sha256_file,
)


_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()

    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    source_summary_path = args.source_run_dir / "summary.json"
    source_terminal_path = args.source_run_dir / "terminal.json"
    if not source_summary_path.is_file() or not source_terminal_path.is_file():
        raise FileNotFoundError("source A0 run is missing summary.json or terminal.json")
    source_terminal = json.loads(source_terminal_path.read_text(encoding="utf-8"))
    if source_terminal.get("status") != "done":
        raise RuntimeError("source A0 run is not terminal done")
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))

    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    sources = (
        PROJECT / "scripts/run_worldsim_v3_a0_actor_metrics.py",
        PROJECT / "scripts/eval_worldsim_v3_a0_actor_metrics.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "motion_proj/worldsim_v3/actor_metrics.py",
    )
    for source_path in sources:
        destination = args.run_dir / "source_snapshot" / source_path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    output_dir = args.run_dir / "artifacts" / "actor_metrics"
    command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/eval_worldsim_v3_a0_actor_metrics.py"),
        "--source-summary",
        str(source_summary_path),
        "--output-dir",
        str(output_dir),
        "--drivestudio-root",
        str(PATCHED_DRIVESTUDIO),
    ]
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": "A0 actor-region and boundary held-out diagnostic",
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "source_run_dir": str(args.source_run_dir),
        "source_summary_sha256": sha256_file(source_summary_path),
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "source_registry_sha256": source["registry_sha256"],
        "selected_actors": source["selected_actors"],
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": command_output("git", "status", "--short", cwd=PROJECT).splitlines(),
        "command": command,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    def validate() -> tuple[bool, dict[str, object]]:
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file() or summary_path.stat().st_size == 0:
            return False, {"summary": str(summary_path), "exists": False}
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        roles = payload.get("roles", {})
        valid_statuses = {"done", "ABSTAIN"}
        ok = (
            payload.get("status") == "done"
            and set(roles) == {"high-support", "boundary-support"}
            and all(row.get("status") in valid_statuses for row in roles.values())
            and payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after")
        )
        return ok, {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "role_statuses": {
                name: row.get("status") for name, row in roles.items()
            },
            "checkpoint_unchanged": payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after"),
        }

    stage = run_stage(
        run_dir=args.run_dir,
        stage="actor_metrics_heldout",
        command=command,
        cwd=PROJECT,
        environment=common_environment(),
        validate=validate,
        timeout_seconds=args.timeout_seconds,
    )
    actor_summary_path = output_dir / "summary.json"
    actor_summary = json.loads(actor_summary_path.read_text(encoding="utf-8"))
    summary = {
        "status": "done",
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "source_run_dir": str(args.source_run_dir),
        "source_summary_sha256": sha256_file(source_summary_path),
        "actor_metrics": actor_summary,
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
                        "code": "A0_ACTOR_METRICS_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

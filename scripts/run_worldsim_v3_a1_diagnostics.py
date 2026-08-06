#!/usr/bin/env python
"""WorldSim V3 A1-D0 正式只读诊断控制器。"""

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

from motion_proj.worldsim_v3.calibration_diagnostics import (
    VARIANTS,
    sha256_file,
    validate_diagnostic_contract,
)
from scripts.run_worldsim_v3_a0_scene import common_environment, run_stage
from scripts.run_worldsim_v3_a0_smoke import atomic_json, command_output, now


PROJECT = Path("/root/autodl-tmp/motion_proj")
MOTIONPROJ_PYTHON = Path("/root/autodl-tmp/envs/motionproj/bin/python")
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def parse_source_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        variant, separator, path = value.partition("=")
        if not separator or variant not in VARIANTS or not path:
            raise ValueError(f"invalid --source-run value: {value}")
        if variant in result:
            raise ValueError(f"duplicate source variant: {variant}")
        result[variant] = Path(path)
    if set(result) != set(VARIANTS):
        raise ValueError(f"source variants must be exactly {VARIANTS}")
    return result


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
    parser.add_argument("--source-run", action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-config",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a1_diagnostics_v1.yaml",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    source_runs = parse_source_runs(args.source_run)
    contract = yaml.safe_load(args.diagnostic_config.read_text(encoding="utf-8"))
    validate_diagnostic_contract(contract)
    guard = resource_guard(args.run_dir.parent)
    source_summaries: dict[str, dict[str, str]] = {}
    for variant, source_run in source_runs.items():
        summary = source_run / "summary.json"
        terminal = source_run / "terminal.json"
        if not summary.is_file() or not terminal.is_file():
            raise FileNotFoundError(f"source run is incomplete: {source_run}")
        terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
        if terminal_payload.get("status") != "done":
            raise RuntimeError(f"source run is not terminal done: {source_run}")
        source_summaries[variant] = {
            "source_run_dir": str(source_run),
            "summary_sha256": sha256_file(summary),
            "terminal_sha256": sha256_file(terminal),
        }

    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.diagnostic_config, args.run_dir / "resolved_config.yaml")
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    sources = (
        PROJECT / "scripts/run_worldsim_v3_a1_diagnostics.py",
        PROJECT / "scripts/eval_worldsim_v3_a1_diagnostics.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "motion_proj/worldsim_v3/calibration.py",
        PROJECT / "motion_proj/worldsim_v3/calibration_diagnostics.py",
        args.diagnostic_config,
    )
    source_hashes: dict[str, str] = {}
    for source in sources:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)

    output_dir = args.run_dir / "artifacts/a1_diagnostics"
    command = [
        str(MOTIONPROJ_PYTHON),
        str(PROJECT / "scripts/eval_worldsim_v3_a1_diagnostics.py"),
    ]
    for variant in VARIANTS:
        command.extend(["--source-run", f"{variant}={source_runs[variant]}"])
    command.extend(
        [
            "--diagnostic-config",
            str(args.run_dir / "resolved_config.yaml"),
            "--output-dir",
            str(output_dir),
        ]
    )
    commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    fingerprint = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "project_commit": commit,
        "source_hashes": source_hashes,
        "source_summaries": source_summaries,
        "diagnostic_config_sha256": sha256_file(args.run_dir / "resolved_config.yaml"),
    }
    atomic_json(args.run_dir / "fingerprint.json", fingerprint)
    manifest = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "component": "A1 ISP, pose, temporal, and input-speed diagnostics",
        "status": "running",
        "scene_name": contract["scene"]["name"],
        "scene_index": contract["scene"]["index"],
        "variants": list(VARIANTS),
        "seed": 0,
        "split": "all processed frames; held-out policy applied to ISP where defined",
        "source_runs": {key: str(value) for key, value in source_runs.items()},
        "project_commit": commit,
        "project_status": command_output("git", "status", "--short", cwd=PROJECT).splitlines(),
        "resource_guard": guard,
        "command": command,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    def validate() -> tuple[bool, dict[str, object]]:
        summary_path = output_dir / "summary.json"
        csv_path = output_dir / "diagnostics_matrix.csv"
        if not summary_path.is_file() or not csv_path.is_file():
            return False, {"summary": str(summary_path), "csv": str(csv_path)}
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        variants = payload.get("variants", {})
        ok = (
            payload.get("status") == "done"
            and set(variants) == set(VARIANTS)
            and all(
                row.get("diagnostic", {}).get("checkpoint_unchanged") is True
                for row in variants.values()
            )
            and payload.get("input_motion", {}).get("tier_frame_counts")
            == contract["speed_tiers"]["expected_input_only_counts"]
        )
        return ok, {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "matrix_csv": str(csv_path),
            "variant_count": len(variants),
            "checkpoint_unchanged": {
                key: row.get("diagnostic", {}).get("checkpoint_unchanged")
                for key, row in variants.items()
            },
        }

    stage = run_stage(
        run_dir=args.run_dir,
        stage="a1_diagnostics",
        command=command,
        cwd=PROJECT,
        environment=common_environment(),
        validate=validate,
        timeout_seconds=args.timeout_seconds,
    )
    diagnostic_summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    shutil.copy2(output_dir / "diagnostics_matrix.json", args.run_dir / "metrics.json")
    summary = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A1-D0 calibration diagnostic matrix",
        "scene_name": contract["scene"]["name"],
        "scene_index": contract["scene"]["index"],
        "seed": 0,
        "diagnostic_config_sha256": fingerprint["diagnostic_config_sha256"],
        "diagnostic_metrics": diagnostic_summary,
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
        if (
            _ACTIVE_RUN_DIR is not None
            and _ACTIVE_RUN_DIR.is_dir()
            and not _TERMINAL_FINAL
        ):
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "code": "A1_DIAGNOSTIC_RUN_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

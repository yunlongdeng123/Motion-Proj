"""WorldSim V6 R4 fresh-process deterministic runtime 正式实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TASK_ID = "WS-V6-R4-DETERMINISTIC-RUNTIME-01"


class R4ExperimentError(RuntimeError):
    """R4 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R4ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R4ExperimentError("正式 R4 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R4ExperimentError("R4 task_id 漂移")
    package = _resolve_runs_uri(config["source"]["sceneir_package"])
    if not package.is_dir():
        raise R4ExperimentError("R4 SceneIR package 缺失")
    if _sha256(package / "MANIFEST.json") != config["source"]["package_manifest_sha256"]:
        raise R4ExperimentError("SceneIR package manifest 漂移")
    if _sha256(package / "sceneir.json") != config["source"]["sceneir_json_sha256"]:
        raise R4ExperimentError("SceneIR document 漂移")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R4ExperimentError("R4 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__deterministic-runtime-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        process_rows = []
        python = Path("/root/autodl-tmp/envs/motionproj/bin/python")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        process_count = int(config["fresh_process_count"])
        for index in range(1, process_count + 1):
            output = run_dir / f"fresh_process_{index}"
            command = [
                str(python),
                str(repo_root / "scripts/worldsim_v6/r4_runtime_worker.py"),
                "--package",
                str(package),
                "--config",
                str(config_path.resolve()),
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(config["resources"]["maximum_wall_seconds_per_process"]),
            )
            (run_dir / f"fresh_process_{index}.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise R4ExperimentError(
                    f"fresh process {index} 失败：{(completed.stdout + completed.stderr)[-4000:]}"
                )
            process_rows.append(
                {
                    "index": index,
                    "pid_isolated": True,
                    "returncode": completed.returncode,
                    "audit_sha256": _sha256(output / "RUNTIME_AUDIT.json"),
                }
            )
        comparisons = {}
        for name in config["determinism_gate"]["exact_files"]:
            paths = [
                run_dir / f"fresh_process_{index}" / name
                for index in range(1, process_count + 1)
            ]
            hashes = [_sha256(path) for path in paths]
            comparisons[name] = {
                "sha256_by_process": hashes,
                "exact": len(set(hashes)) == 1,
            }
        rgb_arrays = [
            np.load(run_dir / f"fresh_process_{index}/RGB.npy", allow_pickle=False)
            for index in range(1, process_count + 1)
        ]
        comparisons["RGB.npy"]["array_equal"] = bool(
            np.array_equal(rgb_arrays[0], rgb_arrays[1])
            and np.array_equal(rgb_arrays[0], rgb_arrays[2])
        )
        comparisons["RGB.npy"]["shape"] = list(rgb_arrays[0].shape)
        exact = all(row["exact"] for row in comparisons.values())
        gate = {
            "schema_version": "worldsim_v6.r4_determinism_gate.v1",
            "fresh_process_count": len(process_rows),
            "comparisons": comparisons,
            "world_state_exact": comparisons["WORLD_STATE.json"]["exact"],
            "labels_exact": comparisons["LABELS.json"]["exact"],
            "asset_chunk_selection_exact": comparisons["CHUNK_SELECTION.json"]["exact"],
            "actor_trajectory_exact": comparisons["ACTOR_TRAJECTORY.json"]["exact"],
            "rgb_exact": comparisons["RGB.npy"]["exact"],
            "passed": exact,
        }
        _write_json(run_dir / "DETERMINISM_GATE.json", gate)
        _write_json(
            run_dir / "PROCESS_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r4_process_audit.v1",
                "processes": process_rows,
                "gpu_used": False,
                "training_started": False,
                "confirmation_content_read": False,
                "disk_free_gib_at_start": free_gib,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r4_summary.v1",
            "task_id": TASK_ID,
            "status": "done" if exact else "rejected",
            "source_commit": source_commit,
            "sceneir_content_sha256": config["source"]["sceneir_content_sha256"],
            "fresh_process_count": len(process_rows),
            "determinism_passed": exact,
            "rgb_kind": config["sensor"]["rgb_semantics"],
            "visual_float_tolerance_used": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["DETERMINISM_GATE.json", "PROCESS_AUDIT.json", "SUMMARY.json"]
        for index in range(1, process_count + 1):
            tracked.append(f"fresh_process_{index}/RUNTIME_AUDIT.json")
            tracked.extend(
                f"fresh_process_{index}/{name}" for name in config["determinism_gate"]["exact_files"]
            )
        manifest = {
            "schema_version": "worldsim_v6.r4_run_manifest.v1",
            "files": {
                relative: {
                    "bytes": (run_dir / relative).stat().st_size,
                    "sha256": _sha256(run_dir / relative),
                }
                for relative in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "done" if exact else "rejected",
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/worldsim_v6/r4_deterministic_runtime_v0.yaml")
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

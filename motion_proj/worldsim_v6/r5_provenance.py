"""WorldSim V6 R5 provenance field 正式实验。"""

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

import yaml

from motion_proj.worldsim_v6.provenance import (
    build_provenance_document,
    verify_provenance_package,
    write_provenance_package,
)


TASK_ID = "WS-V6-R5-PROVENANCE-01"


class R5ExperimentError(RuntimeError):
    """R5 正式合同失败。"""


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
    if not uri.startswith("runs://") or ".." in Path(uri[7:]).parts:
        raise R5ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[7:]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R5ExperimentError("正式 R5 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R5ExperimentError("R5 task_id 漂移")
    source_package = _resolve_runs_uri(config["source"]["sceneir_package"])
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R5ExperimentError("R5 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__provenance-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package = run_dir / "provenance_package"
        document = build_provenance_document(source_package)
        if document["source_sceneir_content_sha256"] != config["source"]["sceneir_content_sha256"]:
            raise R5ExperimentError("source SceneIR content 漂移")
        write_provenance_package(package, document)
        local_verification = verify_provenance_package(package, source_package)
        fresh_rows = []
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        python = Path("/root/autodl-tmp/envs/motionproj/bin/python")
        for index in range(1, int(config["gate"]["fresh_reload_processes"]) + 1):
            completed = subprocess.run(
                [
                    str(python),
                    str(repo_root / "scripts/worldsim_v6/verify_provenance.py"),
                    "--package",
                    str(package),
                    "--sceneir-package",
                    str(source_package),
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise R5ExperimentError(f"fresh provenance reload {index} 失败：{completed.stderr[-3000:]}")
            result = json.loads(completed.stdout)
            fresh_rows.append(result)
        exact_reload = all(row == local_verification for row in fresh_rows)
        gate = {
            "schema_version": "worldsim_v6.r5_provenance_gate.v1",
            "local_verification": local_verification,
            "fresh_reload_count": len(fresh_rows),
            "fresh_reload_exact": exact_reload,
            "chunk_coverage": document["coverage"]["chunk_covered"] / document["coverage"]["chunk_total"],
            "actor_coverage": document["coverage"]["actor_covered"] / document["coverage"]["actor_total"],
            "primitive_coverage": document["coverage"]["primitive_covered"] / document["coverage"]["primitive_total"],
            "global_primitive_identity_unique": document["coverage"]["global_primitive_identity_unique"],
            "observed_reconstructed_generated_disjoint": local_verification["type_separation_passed"],
        }
        gate["passed"] = bool(
            exact_reload
            and gate["chunk_coverage"] == float(config["gate"]["chunk_coverage"])
            and gate["actor_coverage"] == float(config["gate"]["actor_coverage"])
            and gate["primitive_coverage"] == float(config["gate"]["primitive_coverage"])
            and gate["global_primitive_identity_unique"]
            and gate["observed_reconstructed_generated_disjoint"]
        )
        _write_json(run_dir / "PROVENANCE_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r5_summary.v1",
            "task_id": TASK_ID,
            "status": "done" if gate["passed"] else "rejected",
            "source_commit": source_commit,
            "source_sceneir_content_sha256": document["source_sceneir_content_sha256"],
            "provenance_content_sha256": document["content_sha256"],
            "coverage_passed": gate["passed"],
            "source_type_primitive_counts": local_verification["source_type_primitive_counts"],
            "sensor_support_status": "unknown_source_sceneir_has_no_sensor",
            "view_support_status": "unknown_source_sceneir_has_no_observed_views",
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "PROVENANCE_GATE.json",
            "SUMMARY.json",
            "provenance_package/PROVENANCE.json",
            "provenance_package/MANIFEST.json",
        ]
        manifest = {
            "schema_version": "worldsim_v6.r5_run_manifest.v1",
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
                "status": "done" if gate["passed"] else "rejected",
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r5_provenance_v0.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

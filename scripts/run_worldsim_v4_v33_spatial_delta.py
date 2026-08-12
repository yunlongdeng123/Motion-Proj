#!/usr/bin/env python3
"""运行 V4 V3.3 fail-safe spatial package 与 development-only 实渲染。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
RUN_ROOT = Path(f"/root/autodl-tmp/runs/worldsim_v4/{TASK_ID}")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_commands(
    *, config_path: Path, run_dir: Path, project_root: Path, config: Mapping[str, Any]
) -> list[list[str]]:
    python = str(config["runtimes"]["drivestudio_python"])
    package_run = run_dir / "package"
    eval_run = run_dir / "evaluation"
    package_manifest = package_run / "artifacts" / "worldsim_asset" / "package_manifest.json"
    return [
        [
            python,
            str(project_root / "scripts/build_worldsim_v33_s4_spatial_delta.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(package_run),
        ],
        [
            python,
            str(project_root / "scripts/evaluate_worldsim_v33_s4_spatial_delta.py"),
            "--config",
            str(config_path),
            "--package-manifest",
            str(package_manifest),
            "--package-manifest-sha256",
            "__RESOLVE_AFTER_PACKAGE__",
            "--run-dir",
            str(eval_run),
        ],
    ]


def run(*, config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run 目录已存在，禁止覆盖：{run_dir}")
    if RUN_ROOT.resolve() not in run_dir.resolve().parents:
        raise V33ReplayError(f"spatial run 必须位于冻结根目录：{RUN_ROOT}")
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v4_v33_spatial_delta_v1":
        raise V33ReplayError("spatial config schema 漂移")
    provenance = config.get("provenance", {})
    if provenance.get("heldout_content_read") is not False or provenance.get(
        "test_quality_read"
    ) is not False:
        raise V33ReplayError("spatial config 未证明 heldout/test 未读")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if gpu.strip():
        raise V33ReplayError("GPU preflight 非空闲")
    git_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    ).strip():
        raise V33ReplayError("formal spatial run 要求 project git clean")

    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    status = {
        "schema_version": "worldsim_v4_v33_spatial_status_v1",
        "task_id": TASK_ID,
        "scene": config["scene"]["name"],
        "stage": "spatial_delta",
        "status": "running",
        "project_git_head": git_head,
        "development_content_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "status.json", status)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{project_root}:{config['runtimes']['drivestudio_checkout']}"
    )
    commands = build_commands(
        config_path=config_path,
        run_dir=run_dir,
        project_root=project_root,
        config=config,
    )
    with (run_dir / "logs" / "spatial.log").open("xb") as log:
        package = subprocess.run(
            commands[0], cwd=project_root, env=environment,
            stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False,
        )
        if package.returncode != 0:
            raise V33ReplayError(f"spatial package failed: exit={package.returncode}")
        manifest_path = Path(
            commands[1][commands[1].index("--package-manifest") + 1]
        )
        commands[1][commands[1].index("--package-manifest-sha256") + 1] = sha256_file(
            manifest_path
        )
        evaluation = subprocess.run(
            commands[1], cwd=project_root, env=environment,
            stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False,
        )
        if evaluation.returncode != 0:
            raise V33ReplayError(f"spatial evaluation failed: exit={evaluation.returncode}")
    package_summary = run_dir / "package" / "summary.json"
    eval_summary = run_dir / "evaluation" / "summary.json"
    decision = run_dir / "evaluation" / "artifacts" / "decision.json"
    eval_payload = json.loads(eval_summary.read_text(encoding="utf-8"))
    if eval_payload.get("state") != "completed" or not eval_payload["decision"]["accepted"]:
        raise V33ReplayError("spatial evaluation 未通过")
    stage_summary = {
        "schema_version": "worldsim_v4_v33_spatial_stage_v1",
        "task_id": TASK_ID,
        "scene": config["scene"]["name"],
        "status": "done",
        "stage": "spatial_delta",
        "project_git_head": git_head,
        "config_sha256": sha256_file(config_path),
        "package_summary_sha256": sha256_file(package_summary),
        "evaluation_summary_sha256": sha256_file(eval_summary),
        "decision_sha256": sha256_file(decision),
        "available_stacks": eval_payload["package_manifest"].get(
            "available_stacks", config["composition"]["stacks"]
        ),
        "stage_abstentions": config["stage_abstentions"],
        "development_content_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "stage_summary.json", stage_summary)
    status.update(
        status="done",
        stage_summary_sha256=sha256_file(run_dir / "stage_summary.json"),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(run_dir / "status.json", status)
    return stage_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    preexisted = run_dir.exists()
    try:
        summary = run(
            config_path=args.config.resolve(), run_dir=run_dir,
            project_root=args.project_root.resolve(),
        )
    except Exception as error:
        if not preexisted and run_dir.is_dir():
            try:
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            except Exception:
                status = {"task_id": TASK_ID, "stage": "spatial_delta"}
            status.update(
                status="failed", reason=type(error).__name__, error=str(error),
                heldout_content_read=False, test_quality_read=False,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(run_dir / "status.json", status)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

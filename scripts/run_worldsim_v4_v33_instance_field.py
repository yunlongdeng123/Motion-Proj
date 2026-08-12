#!/usr/bin/env python3
"""运行 V4 development-only pseudo target 与冻结 O1 instance field。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
RUN_ROOT = Path(f"/root/autodl-tmp/runs/worldsim_v4/{TASK_ID}")
SNAPSHOT_FILES = (
    "motion_proj/worldsim_v33/evaluation_partition.py",
    "scripts/prepare_worldsim_v33_s1_eval_prompts.py",
    "scripts/build_worldsim_v33_s1_eval_masks.py",
    "scripts/finalize_worldsim_v33_s1_eval_targets.py",
    "scripts/run_worldsim_v33_s1_instance_field.py",
    "scripts/finalize_worldsim_v33_s1.py",
    "scripts/run_worldsim_v4_v33_instance_field.py",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_root_for(config: Mapping[str, Any]) -> Path:
    task_id = str(config.get("task_id", TASK_ID))
    return Path(
        str(
            config.get("provenance", {}).get(
                "run_root", f"/root/autodl-tmp/runs/worldsim_v4/{task_id}"
            )
        )
    )


def build_commands(
    *, config_path: Path, run_dir: Path, project_root: Path, config: Mapping[str, Any]
) -> list[list[str]]:
    eval_dir = run_dir / "eval_targets"
    instance_dir = run_dir / "instance_field"
    prompt_dir = eval_dir / "artifacts" / "prompts"
    mask_dir = eval_dir / "artifacts" / "masks"
    mask_manifest = mask_dir / "mask_manifest.json"
    drivestudio_python = str(config["runtimes"]["drivestudio_python"])
    sam_python = str(config["runtimes"]["sam_python"])
    partition = str(config["provenance"]["evaluation_partition"])
    return [
        [
            drivestudio_python,
            str(project_root / "scripts/prepare_worldsim_v33_s1_eval_prompts.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(prompt_dir),
            "--partition",
            partition,
        ],
        [
            sam_python,
            str(project_root / "scripts/build_worldsim_v33_s1_eval_masks.py"),
            "--config",
            str(config_path),
            "--prompt-manifest",
            str(prompt_dir / "prompt_manifest.json"),
            "--output-dir",
            str(mask_dir),
            "--partition",
            partition,
        ],
        [
            drivestudio_python,
            str(project_root / "scripts/finalize_worldsim_v33_s1_eval_targets.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(eval_dir),
            "--partition",
            partition,
        ],
        [
            drivestudio_python,
            str(project_root / "scripts/run_worldsim_v33_s1_instance_field.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(instance_dir),
            "--phase",
            "formal",
            "--eval-mask-manifest",
            str(mask_manifest),
            "--eval-partition",
            partition,
        ],
        [
            drivestudio_python,
            str(project_root / "scripts/finalize_worldsim_v33_s1.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(instance_dir),
            "--phase",
            "formal",
            "--evaluation-partition",
            partition,
        ],
    ]


def gpu_compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def manifest(run_dir: Path, *, task_id: str = TASK_ID) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"wrapper_manifest.json", "status.json"}:
            continue
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "worldsim_v4_v33_instance_wrapper_manifest_v1",
        "task_id": task_id,
        "files": rows,
    }


def run(*, config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run 目录已存在，禁止覆盖：{run_dir}")
    config = load_yaml(config_path)
    task_id = str(config.get("task_id", TASK_ID))
    run_root = run_root_for(config)
    if run_root.resolve() not in run_dir.resolve().parents:
        raise V33ReplayError(f"instance run 必须位于冻结根目录：{run_root}")
    if config.get("schema_version") != "worldsim_v4_v33_instance_field_v1":
        raise V33ReplayError("instance config schema 漂移")
    provenance = config.get("provenance", {})
    if provenance.get("evaluation_partition") != "development":
        raise V33ReplayError("V4 B0 instance evaluation 必须是 development")
    if any(
        provenance.get(key) is not False
        for key in (
            "development_content_read",
            "heldout_content_read",
            "test_quality_read",
        )
    ):
        raise V33ReplayError("instance config 未证明 evaluation/test 内容未读")
    if gpu_compute_processes():
        raise V33ReplayError("GPU preflight 非空闲")
    git_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    )
    if git_status.strip():
        raise V33ReplayError("formal instance run 要求 project git clean")

    started = time.monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "eval_targets" / "artifacts").mkdir(parents=True)
    (run_dir / "instance_field" / "logs").mkdir(parents=True)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    for relative in SNAPSHOT_FILES:
        source = project_root / relative
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    status = {
        "schema_version": "worldsim_v4_v33_instance_status_v1",
        "task_id": task_id,
        "scene": config["scene"]["name"],
        "stage": "instance_field",
        "status": "running",
        "evaluation_partition": "development",
        "project_git_head": git_head,
        "development_optimization_read": False,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "status.json", status)
    atomic_json(
        run_dir / "instance_field" / "status.json",
        {"status": "running", "phase": "formal", "task_id": task_id},
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{project_root}:{config['runtimes']['drivestudio_checkout']}"
    )
    with (run_dir / "logs" / "instance_pipeline.log").open("xb") as log:
        for index, command in enumerate(
            build_commands(
                config_path=config_path,
                run_dir=run_dir,
                project_root=project_root,
                config=config,
            )
        ):
            atomic_json(
                run_dir / "active_command.json",
                {"index": index, "argv": command, "started_at_utc": datetime.now(timezone.utc).isoformat()},
            )
            process = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=6 * 60 * 60,
                check=False,
            )
            if process.returncode != 0:
                raise V33ReplayError(
                    f"instance child failed: index={index} exit={process.returncode}"
                )
    if gpu_compute_processes():
        raise V33ReplayError("instance run 结束后仍有 GPU compute process")
    eval_summary = run_dir / "eval_targets" / "summary.json"
    instance_summary = run_dir / "instance_field" / "summary.json"
    acceptance = run_dir / "instance_field" / "acceptance.json"
    instance = json.loads(instance_summary.read_text(encoding="utf-8"))
    if instance.get("evaluation_partition") != "development":
        raise V33ReplayError("instance terminal summary partition 漂移")
    stage_summary = {
        "schema_version": "worldsim_v4_v33_instance_stage_v1",
        "task_id": task_id,
        "scene": config["scene"]["name"],
        "status": "done",
        "stage": "instance_field",
        "evaluation_partition": "development",
        "project_git_head": git_head,
        "config_sha256": sha256_file(config_path),
        "eval_summary_sha256": sha256_file(eval_summary),
        "instance_summary_sha256": sha256_file(instance_summary),
        "acceptance_sha256": sha256_file(acceptance),
        "selected_arm": instance["recommended_arm"],
        "duration_seconds": time.monotonic() - started,
        "development_optimization_read": False,
        "development_content_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "stage_summary.json", stage_summary)
    atomic_json(
        run_dir / "wrapper_manifest.json", manifest(run_dir, task_id=task_id)
    )
    status.update(
        status="done",
        development_content_read=True,
        stage_summary_sha256=sha256_file(run_dir / "stage_summary.json"),
        wrapper_manifest_sha256=sha256_file(run_dir / "wrapper_manifest.json"),
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
            config_path=args.config.resolve(),
            run_dir=run_dir,
            project_root=args.project_root.resolve(),
        )
    except Exception as error:
        if not preexisted and run_dir.is_dir():
            try:
                failure_task_id = str(load_yaml(args.config.resolve()).get("task_id", TASK_ID))
            except Exception:
                failure_task_id = TASK_ID
            try:
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            except Exception:
                status = {
                    "schema_version": "worldsim_v4_v33_instance_status_v1",
                    "task_id": failure_task_id,
                    "stage": "instance_field",
                }
            status.update(
                status="failed",
                reason=type(error).__name__,
                error=str(error),
                development_optimization_read=False,
                development_content_read=False,
                heldout_content_read=False,
                test_quality_read=False,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(run_dir / "status.json", status)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

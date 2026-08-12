#!/usr/bin/env python3
"""运行单场景 V4 V3.3 semantic lift，并保持 development/heldout 封存。"""

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
    "motion_proj/worldsim_v4/semantic_split.py",
    "scripts/prepare_worldsim_v32_s1_prompts.py",
    "scripts/validate_worldsim_v32_s1.py",
    "scripts/build_worldsim_v32_sam_masks.py",
    "scripts/lift_worldsim_v32_semantics.py",
    "scripts/finalize_worldsim_v32_s1.py",
    "scripts/run_worldsim_v4_v33_semantic_lift.py",
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


def build_commands(
    *, config_path: Path, run_dir: Path, project_root: Path, config: Mapping[str, Any]
) -> list[list[str]]:
    prompt_dir = run_dir / "artifacts" / "prompts"
    mask_dir = run_dir / "artifacts" / "sam2"
    semantic_dir = run_dir / "artifacts" / "semantic_sidecar"
    drivestudio_python = str(config["runtimes"]["drivestudio_python"])
    sam_python = str(config["runtimes"]["sam_python"])
    return [
        [
            drivestudio_python,
            str(project_root / "scripts/validate_worldsim_v32_s1.py"),
            "--config",
            str(config_path),
        ],
        [
            drivestudio_python,
            str(project_root / "scripts/prepare_worldsim_v32_s1_prompts.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(prompt_dir),
        ],
        [
            sam_python,
            str(project_root / "scripts/build_worldsim_v32_sam_masks.py"),
            "--config",
            str(config_path),
            "--prompt-manifest",
            str(prompt_dir / "prompt_manifest.json"),
            "--output-dir",
            str(mask_dir),
        ],
        [
            drivestudio_python,
            str(project_root / "scripts/lift_worldsim_v32_semantics.py"),
            "--config",
            str(config_path),
            "--mask-manifest",
            str(mask_dir / "mask_manifest.json"),
            "--output-dir",
            str(semantic_dir),
        ],
        [
            drivestudio_python,
            str(project_root / "scripts/finalize_worldsim_v32_s1.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--run-root",
            str(RUN_ROOT),
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


def manifest(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"run_manifest.json", "status.json"}:
            continue
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "worldsim_v4_v33_semantic_run_manifest_v1",
        "task_id": TASK_ID,
        "files": rows,
    }


def run(
    *, config_path: Path, run_dir: Path, project_root: Path
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run 目录已存在，禁止覆盖：{run_dir}")
    if RUN_ROOT.resolve() not in run_dir.resolve().parents:
        raise V33ReplayError(f"semantic run 必须位于冻结根目录：{RUN_ROOT}")
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v4_v33_semantic_lift_v1":
        raise V33ReplayError("semantic run config schema 漂移")
    provenance = config.get("provenance", {})
    if any(
        provenance.get(key) is not False
        for key in (
            "development_content_read",
            "heldout_content_read",
            "test_quality_read",
        )
    ):
        raise V33ReplayError("semantic config 未证明 evaluation/test 内容未读")
    if gpu_compute_processes():
        raise V33ReplayError("GPU preflight 非空闲")
    git_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    )
    if git_status.strip():
        raise V33ReplayError("formal semantic run 要求 project git clean")

    started = time.monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    for relative in SNAPSHOT_FILES:
        source = project_root / relative
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    status = {
        "schema_version": "worldsim_v4_v33_semantic_status_v1",
        "task_id": TASK_ID,
        "scene": config["scene"]["name"],
        "stage": "semantic_lift",
        "status": "running",
        "project_git_head": git_head,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "status.json", status)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{project_root}:{config['runtimes']['drivestudio_checkout']}"
    )
    with (run_dir / "logs" / "semantic_lift.log").open("xb") as log:
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
                timeout=4 * 60 * 60,
                check=False,
            )
            if process.returncode != 0:
                raise V33ReplayError(
                    f"semantic child failed: index={index} exit={process.returncode}"
                )
    if gpu_compute_processes():
        raise V33ReplayError("semantic run 结束后仍有 GPU compute process")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stage_summary = {
        "schema_version": "worldsim_v4_v33_semantic_stage_v1",
        "task_id": TASK_ID,
        "scene": config["scene"]["name"],
        "status": "done",
        "stage": "semantic_lift",
        "project_git_head": git_head,
        "config_sha256": sha256_file(config_path),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "actor_count": len(summary["actors"]),
        "actor_abstentions": config["provenance"]["actor_abstentions"],
        "duration_seconds": time.monotonic() - started,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "stage_summary.json", stage_summary)
    atomic_json(run_dir / "run_manifest.json", manifest(run_dir))
    status.update(
        status="done",
        stage_summary_sha256=sha256_file(run_dir / "stage_summary.json"),
        run_manifest_sha256=sha256_file(run_dir / "run_manifest.json"),
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
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            except Exception:
                status = {
                    "schema_version": "worldsim_v4_v33_semantic_status_v1",
                    "task_id": TASK_ID,
                    "stage": "semantic_lift",
                }
            status.update(
                status="failed",
                reason=type(error).__name__,
                error=str(error),
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

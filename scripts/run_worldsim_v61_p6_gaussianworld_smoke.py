#!/usr/bin/env python3
"""WorldSim V6.1 P6：GaussianWorld 官方权重 3090 capability smoke。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V61-P6-GAUSSIANWORLD-3090-SMOKE-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class P6SmokeError(RuntimeError):
    """P6 source、环境、输入或能力合同失败。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://"):
        raise P6SmokeError("只接受 runs URI")
    relative = Path(uri[len("runs://") :])
    if ".." in relative.parts:
        raise P6SmokeError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def verify_files(root: Path, files: Mapping[str, str]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise P6SmokeError(f"冻结 authority 漂移: {path}")


def frozen_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if git(repo_root, "status", "--porcelain"):
        raise P6SmokeError("正式 P6 要求 motion_proj 工作树干净")
    source_commit = git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise P6SmokeError("P6 task_id 漂移")

    me1_root = resolve_runs_uri(config["sources"]["me1_run"])
    me2_root = resolve_runs_uri(config["sources"]["me2_run"])
    verify_files(me1_root, config["sources"]["me1_files"])
    verify_files(me2_root, config["sources"]["me2_files"])
    me1_gate = json.loads((me1_root / "ME1_GATE.json").read_text(encoding="utf-8"))
    me2_summary = json.loads((me2_root / "SUMMARY.json").read_text(encoding="utf-8"))
    me2_terminal = json.loads((me2_root / "TERMINAL.json").read_text(encoding="utf-8"))
    if not me1_gate.get("passed"):
        raise P6SmokeError("ME1 oracle authority 未通过")
    if (
        me2_summary.get("hypothesis_outcome") != "rejected_voxel_control_actor_upper_bound"
        or me2_terminal.get("status") != "rejected"
        or not me2_terminal.get("canonical")
    ):
        raise P6SmokeError("ME2 Hunyuan stop authority 漂移")

    official_repo = Path(config["sources"]["official_repo"])
    if git(official_repo, "rev-parse", "HEAD") != config["sources"]["official_git_commit"]:
        raise P6SmokeError("GaussianWorld 官方 commit 漂移")
    if git(official_repo, "status", "--porcelain"):
        raise P6SmokeError("GaussianWorld 官方 checkout 不是 clean")

    model_root = Path(config["sources"]["model_root"])
    frozen_files: list[dict[str, Any]] = []
    for name, expected in config["sources"]["model_files"].items():
        path = model_root / name
        record = frozen_record(path)
        if record["bytes"] != int(expected["bytes"]) or record["sha256"] != expected["sha256"]:
            raise P6SmokeError(f"GaussianWorld 模型输入漂移: {path}")
        frozen_files.append(record)
    official_config = official_repo / config["sources"]["official_config"]
    frozen_files.append(frozen_record(official_config))

    scene_root = Path(config["input"]["processed_root"])
    frame = int(config["input"]["frame"])
    camera_ids = [int(value) for value in config["input"]["drivestudio_camera_ids"]]
    for camera_id in camera_ids:
        frozen_files.extend(
            frozen_record(path)
            for path in (
                scene_root / f"images/{frame:03d}_{camera_id}.jpg",
                scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt",
                scene_root / f"intrinsics/{camera_id}.txt",
            )
        )
    frozen_files.append(frozen_record(scene_root / f"lidar_pose/{frame:03d}.txt"))

    run_root.mkdir(parents=True, exist_ok=True)
    disk_free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__gaussianworld-smoke-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        plan = {
            "schema_version": "worldsim_v61.p6_gaussianworld_worker_plan.v1",
            "seed": int(config["seed"]),
            "gpu": int(config["resources"]["gpu"]),
            "official_repo": str(official_repo),
            "official_config": str(official_config),
            "checkpoint_path": str(model_root / "ckpt_stream.pth"),
            "backbone_path": str(model_root / "r101_dcn_fcos3d_pretrain.pth"),
            "scene": config["input"]["scene"],
            "scene_root": str(scene_root),
            "frame": frame,
            "camera_names": config["input"]["camera_names"],
            "camera_ids": camera_ids,
            "native_shape": [
                int(config["input"]["expected_native_height"]),
                int(config["input"]["expected_native_width"]),
            ],
            "final_shape": [
                int(config["input"]["final_height"]),
                int(config["input"]["final_width"]),
            ],
            **config["output_contract"],
            "frozen_files": frozen_files,
        }
        plan_path = run_dir / "WORKER_PLAN.json"
        output_path = run_dir / "PREDICTED_OCCUPANCY.npz"
        report_path = run_dir / "WORKER_REPORT.json"
        log_path = run_dir / "WORKER.log"
        write_json(plan_path, plan)
        command = [
            str(Path(config["environment"]["prefix"]) / "bin/python"),
            str(repo_root / "scripts/run_worldsim_v61_p6_gaussianworld_worker.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(config["resources"]["gpu"]),
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
                timeout=float(config["resources"]["maximum_wall_seconds"]),
            )
        if result.returncode != 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
            raise P6SmokeError(f"GaussianWorld worker rc={result.returncode}:\n{tail}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_versions = config["environment"]
        actual_versions = report["environment_versions"]
        environment_exact = bool(
            report["python"] == str(expected_versions["python"])
            and report["torch"] == str(expected_versions["torch"])
            and actual_versions["mmcv"] == str(expected_versions["mmcv"])
            and actual_versions["mmdet"] == str(expected_versions["mmdet"])
            and actual_versions["mmsegmentation"] == str(expected_versions["mmsegmentation"])
            and actual_versions["mmdet3d"] == str(expected_versions["mmdet3d"])
            and actual_versions["spconv-cu117"] == str(expected_versions["spconv"])
        )
        checks = {
            "me1_oracle_authority_passed": True,
            "me2_hy3d_route_stopped": True,
            "official_source_and_weights_exact": True,
            "environment_versions_exact": environment_exact,
            "model_state_exact": not report["model_load_missing_keys"]
            and not report["model_load_unexpected_keys"],
            "six_camera_input_complete": report["camera_count"] == 6,
            "output_shape_exact": report["logits_shape"]
            == [1, int(config["output_contract"]["class_count"]), *config["output_contract"]["grid_shape"]],
            "finite_logits": bool(report["finite_logits"]),
            "occupied_nonempty": report["occupied_voxel_count"]
            >= int(config["output_contract"]["minimum_occupied_voxels"]),
            "empty_nonempty": report["empty_voxel_count"]
            >= int(config["output_contract"]["minimum_empty_voxels"]),
            "history_anchor_present": bool(report["history_anchor_present"]),
            "dummy_label_shape_only": report["dummy_label_role"] == "shape_only_not_truth",
            "no_surroundocc_truth_read": not report["surroundocc_label_read"],
            "no_training_or_confirmation": not report["training_started"]
            and not report["confirmation_content_read"],
            "gpu_memory_within_budget": report["peak_gpu_memory_gib"]
            <= float(config["resources"]["maximum_gpu_memory_gib"]),
            "worker_wall_within_budget": report["wall_seconds"]
            <= float(config["resources"]["maximum_wall_seconds"]),
            "disk_free_within_budget": disk_free_gib
            >= float(config["resources"]["minimum_disk_free_gib"]),
        }
        gate = {
            "schema_version": "worldsim_v61.p6_gaussianworld_gate.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "checks": checks,
            "passed": all(checks.values()),
        }
        write_json(run_dir / "P6_GATE.json", gate)
        resource = {
            "schema_version": "worldsim_v61.p6_gaussianworld_resource_audit.v1",
            "gpu_name": report["gpu_name"],
            "peak_gpu_memory_gib": report["peak_gpu_memory_gib"],
            "worker_inference_seconds": report["inference_seconds"],
            "worker_wall_seconds": report["wall_seconds"],
            "runner_wall_seconds": time.monotonic() - started,
            "disk_free_gib_at_start": disk_free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        }
        write_json(run_dir / "RESOURCE_AUDIT.json", resource)
        summary = {
            "schema_version": "worldsim_v61.p6_gaussianworld_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_commit": source_commit,
            "status": "done" if gate["passed"] else "rejected",
            "hypothesis_outcome": "accepted_gaussianworld_3090_capability"
            if gate["passed"]
            else "rejected_gaussianworld_3090_capability",
            "occupied_voxel_count": report["occupied_voxel_count"],
            "empty_voxel_count": report["empty_voxel_count"],
            "next": "WS-V61-ME3-PREDICTED-OCC-01"
            if gate["passed"]
            else "audit_occworld_once_without_gaussianworld_parameter_tuning",
            "failure_ledger_delta": "none" if gate["passed"] else "required",
            "claim_boundary": config["claim_boundary"],
        }
        write_json(run_dir / "SUMMARY.json", summary)
        manifest = {
            "schema_version": "worldsim_v61.p6_gaussianworld_manifest.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "official_git_commit": config["sources"]["official_git_commit"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "frozen_files": frozen_files,
            "artifacts": {
                name: sha256_file(run_dir / name)
                for name in (
                    "WORKER_PLAN.json",
                    "WORKER_REPORT.json",
                    "WORKER.log",
                    "PREDICTED_OCCUPANCY.npz",
                    "P6_GATE.json",
                    "RESOURCE_AUDIT.json",
                    "SUMMARY.json",
                )
            },
        }
        write_json(run_dir / "MANIFEST.json", manifest)
        terminal = {
            "schema_version": "worldsim_v61.p6_gaussianworld_terminal.v1",
            "task_id": TASK_ID,
            "status": summary["status"],
            "canonical": bool(gate["passed"]),
            "run_uri": f"run://worldsim_v61/{TASK_ID}/{run_dir.name}",
        }
        write_json(run_dir / "TERMINAL.json", terminal)
        return run_dir
    except Exception as error:
        write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.p6_gaussianworld_terminal.v1",
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "python": platform.python_version(),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v61")
    )
    args = parser.parse_args()
    print(run(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

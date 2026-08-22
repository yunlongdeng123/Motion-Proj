#!/usr/bin/env python3
"""WorldSim V6.1 P7：IR-WM truth-free current-state 3090 capability smoke。"""

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


TASK_ID = "WS-V61-P7-IRWM-3090-SMOKE-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class P7SmokeError(RuntimeError):
    """P7 source、环境、输入或 capability 合同失败。"""


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
        raise P7SmokeError("只接受 runs URI")
    relative = Path(uri[len("runs://") :])
    if ".." in relative.parts:
        raise P7SmokeError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def verify_files(root: Path, files: Mapping[str, str]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise P7SmokeError(f"冻结 authority 漂移: {path}")


def frozen_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def add_frozen_record(
    records: list[dict[str, Any]], seen: set[Path], path: Path
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        return next(record for record in records if Path(record["path"]).resolve() == resolved)
    record = frozen_record(resolved)
    records.append(record)
    seen.add(resolved)
    return record


def verify_expected(record: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if record["bytes"] != int(expected["bytes"]) or record["sha256"] != expected["sha256"]:
        raise P7SmokeError(f"{label} 漂移: {record['path']}")


def run(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if git(repo_root, "status", "--porcelain"):
        raise P7SmokeError("正式 P7 要求 motion_proj 工作树干净")
    source_commit = git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise P7SmokeError("P7 task_id 漂移")

    me3_root = resolve_runs_uri(config["sources"]["me3_run"])
    verify_files(me3_root, config["sources"]["me3_files"])
    me3_gate = json.loads((me3_root / "ME3_GATE.json").read_text(encoding="utf-8"))
    me3_summary = json.loads((me3_root / "SUMMARY.json").read_text(encoding="utf-8"))
    me3_terminal = json.loads((me3_root / "TERMINAL.json").read_text(encoding="utf-8"))
    if (
        me3_gate.get("passed")
        or me3_summary.get("primary_false_safe_count") != 10
        or me3_summary.get("status") != "rejected"
        or me3_terminal.get("status") != "rejected"
        or not me3_terminal.get("canonical")
    ):
        raise P7SmokeError("ME3 GaussianWorld stop authority 漂移")

    frozen_files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    official_repo = Path(config["sources"]["official_repo"]).resolve()
    archive = official_repo / config["sources"]["source_archive"]["path"]
    verify_expected(
        add_frozen_record(frozen_files, seen, archive),
        config["sources"]["source_archive"],
        "IR-WM source archive",
    )
    for relative, expected_hash in config["sources"]["official_files"].items():
        record = add_frozen_record(frozen_files, seen, official_repo / relative)
        if record["sha256"] != expected_hash:
            raise P7SmokeError(f"IR-WM official source 漂移: {relative}")
    official_config = official_repo / config["sources"]["official_config"]

    model_root = Path(config["sources"]["model_root"])
    checkpoint = model_root / config["sources"]["checkpoint"]["path"]
    verify_expected(
        add_frozen_record(frozen_files, seen, checkpoint),
        config["sources"]["checkpoint"],
        "IR-WM checkpoint",
    )
    metadata = Path(config["sources"]["temporal_metadata"]["path"])
    verify_expected(
        add_frozen_record(frozen_files, seen, metadata),
        config["sources"]["temporal_metadata"],
        "temporal metadata",
    )

    scene_root = Path(config["input"]["processed_root"])
    frames = [int(value) for value in config["input"]["frames"]]
    camera_ids = [int(value) for value in config["input"]["drivestudio_camera_ids"]]
    for frame in frames:
        add_frozen_record(frozen_files, seen, scene_root / f"lidar_pose/{frame:03d}.txt")
        for camera_id in camera_ids:
            for path in (
                scene_root / f"images/{frame:03d}_{camera_id}.jpg",
                scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt",
                scene_root / f"intrinsics/{camera_id}.txt",
            ):
                add_frozen_record(frozen_files, seen, path)

    run_root.mkdir(parents=True, exist_ok=True)
    disk_free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__irwm-current-smoke-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        plan = {
            "schema_version": "worldsim_v61.p7_irwm_worker_plan.v1",
            "seed": int(config["seed"]),
            "gpu": int(config["resources"]["gpu"]),
            "official_repo": str(official_repo),
            "official_git_commit": config["sources"]["official_git_commit"],
            "official_config": str(official_config),
            "checkpoint_path": str(checkpoint),
            "temporal_metadata_path": str(metadata),
            "scene": config["input"]["scene"],
            "scene_root": str(scene_root),
            "frames": frames,
            "metadata_indices": config["input"]["metadata_indices"],
            "camera_names": config["input"]["camera_names"],
            "camera_ids": camera_ids,
            "native_shape": [
                int(config["input"]["expected_native_height"]),
                int(config["input"]["expected_native_width"]),
            ],
            "pad_size_divisor": int(config["input"]["pad_size_divisor"]),
            "image_mean_bgr": config["input"]["image_mean_bgr"],
            "image_std": config["input"]["image_std"],
            **config["output_contract"],
            "frozen_files": frozen_files,
        }
        plan_path = run_dir / "WORKER_PLAN.json"
        output_path = run_dir / "CURRENT_PREDICTED_OCCUPANCY.npz"
        report_path = run_dir / "WORKER_REPORT.json"
        log_path = run_dir / "WORKER.log"
        write_json(plan_path, plan)
        command = [
            str(Path(config["environment"]["prefix"]) / "bin/python"),
            str(repo_root / "scripts/run_worldsim_v61_p7_irwm_worker.py"),
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
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
            raise P7SmokeError(f"IR-WM worker rc={result.returncode}:\n{tail}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_versions = config["environment"]
        actual_versions = report["environment_versions"]
        environment_exact = bool(
            report["python"] == str(expected_versions["python"])
            and report["torch"] == str(expected_versions["torch"])
            and all(
                actual_versions[name] == str(expected_versions[name])
                for name in ("mmcv-full", "mmdet", "mmsegmentation", "mmdet3d", "detectron2")
            )
        )
        checks = {
            "me3_gaussianworld_scientific_stop_authority_exact": True,
            "official_source_archive_and_checkpoint_exact": True,
            "environment_versions_exact": environment_exact,
            "model_state_exact": not report["model_load_missing_keys"]
            and not report["model_load_unexpected_keys"],
            "three_frame_six_camera_input_complete": report["history_frame_count"] == 2
            and report["camera_count"] == 6,
            "raw_output_shape_exact": report["raw_logits_shape"]
            == config["output_contract"]["raw_logits_shape"],
            "class_grid_shape_exact": report["class_label_shape"]
            == config["output_contract"]["grid_shape"],
            "finite_logits": bool(report["finite_logits"]),
            "occupied_nonempty": report["occupied_voxel_count"]
            >= int(config["output_contract"]["minimum_occupied_voxels"]),
            "free_nonempty": report["free_voxel_count"]
            >= int(config["output_contract"]["minimum_free_voxels"]),
            "official_current_state_path_only": report["current_state_extraction"]
            == "official_scene_encoder_plus_official_final_decoder_occupancy_head",
            "no_future_decoder_or_planning": not report["future_decoder_started"]
            and not report["planning_head_started"],
            "no_truth_method_eval_or_confirmation": not report["occupancy_ground_truth_read"]
            and not report["o_method_or_o_eval_read"]
            and not report["confirmation_content_read"],
            "no_training": not report["training_started"],
            "gpu_memory_within_budget": report["peak_gpu_memory_gib"]
            <= float(config["resources"]["maximum_gpu_memory_gib"]),
            "worker_wall_within_budget": report["wall_seconds"]
            <= float(config["resources"]["maximum_wall_seconds"]),
            "disk_free_within_budget": disk_free_gib
            >= float(config["resources"]["minimum_disk_free_gib"]),
        }
        gate = {
            "schema_version": "worldsim_v61.p7_irwm_gate.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "checks": checks,
            "passed": all(checks.values()),
        }
        write_json(run_dir / "P7_GATE.json", gate)
        resource = {
            "schema_version": "worldsim_v61.p7_irwm_resource_audit.v1",
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
            "schema_version": "worldsim_v61.p7_irwm_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_commit": source_commit,
            "status": "done" if gate["passed"] else "rejected",
            "hypothesis_outcome": "accepted_irwm_current_occupancy_3090_capability"
            if gate["passed"]
            else "rejected_irwm_current_occupancy_3090_capability",
            "occupied_voxel_count": report["occupied_voxel_count"],
            "free_voxel_count": report["free_voxel_count"],
            "next": "pre_register_single_ME3_IRWM_recovery"
            if gate["passed"]
            else "stop_learned_occupancy_and_close_v61_minimum_experiment_negative",
            "failure_ledger_delta": "none" if gate["passed"] else "required",
            "claim_boundary": config["claim_boundary"],
        }
        write_json(run_dir / "SUMMARY.json", summary)
        manifest = {
            "schema_version": "worldsim_v61.p7_irwm_manifest.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "official_git_commit": config["sources"]["official_git_commit"],
            "model_revision": config["sources"]["model_revision"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "frozen_files": frozen_files,
            "artifacts": {
                name: sha256_file(run_dir / name)
                for name in (
                    "WORKER_PLAN.json",
                    "WORKER_REPORT.json",
                    "WORKER.log",
                    "CURRENT_PREDICTED_OCCUPANCY.npz",
                    "P7_GATE.json",
                    "RESOURCE_AUDIT.json",
                    "SUMMARY.json",
                )
            },
        }
        write_json(run_dir / "MANIFEST.json", manifest)
        terminal = {
            "schema_version": "worldsim_v61.p7_irwm_terminal.v1",
            "task_id": TASK_ID,
            "status": summary["status"],
            "canonical": True,
            "run_uri": f"run://worldsim_v61/{TASK_ID}/{run_dir.name}",
        }
        write_json(run_dir / "TERMINAL.json", terminal)
        return run_dir
    except Exception as error:
        write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.p7_irwm_terminal.v1",
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

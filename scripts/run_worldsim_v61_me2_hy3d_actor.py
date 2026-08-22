#!/usr/bin/env python3
"""WorldSim V6.1 ME-2：四臂 Hunyuan actor proposal 正式实验。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml
from PIL import Image

from motion_proj.worldsim_v61.me2_actor import (
    actor_projection_mask,
    actor_state,
    build_actor_controls,
    evaluate_mesh,
    prepare_compiled_asset,
)


TASK_ID = "WS-V61-ME2-HY3D-OCC-ACTOR-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class ME2ExperimentError(RuntimeError):
    """ME-2 source、worker 或 gate contract 失败。"""


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


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://"):
        raise ME2ExperimentError("只接受 runs URI")
    relative = Path(uri.removeprefix("runs://"))
    if ".." in relative.parts:
        raise ME2ExperimentError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def verify_files(root: Path, files: Mapping[str, str]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ME2ExperimentError(f"冻结文件漂移: {path}")


def _case_mask(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    return np.asarray(values["mask"], dtype=bool)


def _mean_actor_rgb(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3] > 0
    if not np.any(alpha):
        raise ME2ExperimentError(f"actor RGBA alpha 为空: {path}")
    return rgba[..., :3][alpha].mean(axis=0).astype(np.float32) / 255.0


def _binding_factors(
    scene_root: Path,
    frame: int,
    actor_id: int,
    camera_id: int,
    case_mask: np.ndarray,
) -> dict[str, float | int]:
    instances = json.loads(
        (scene_root / "instances/instances_info.json").read_text(encoding="utf-8")
    )
    pose, size, _ = actor_state(instances, actor_id, frame)
    t_global_camera = np.loadtxt(scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt")
    values = np.loadtxt(scene_root / f"intrinsics/{camera_id}.txt").reshape(-1)
    fx, fy, cx, cy = values[:4]
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    with Image.open(scene_root / f"images/{frame:03d}_{camera_id}.jpg") as image:
        width, height = image.size
    native, _ = actor_projection_mask(
        pose, size, t_global_camera, intrinsics, width=width, height=height
    )
    resized = cv2.resize(
        native.astype(np.uint8),
        (case_mask.shape[1], case_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    intersection = int(np.count_nonzero(resized & case_mask))
    union = int(np.count_nonzero(resized | case_mask))
    return {
        "intersection_pixels": intersection,
        "hole_coverage": intersection / max(int(case_mask.sum()), 1),
        "actor_hull_coverage": intersection / max(int(resized.sum()), 1),
        "iou": intersection / max(union, 1),
    }


def _worker(
    python: Path,
    worker_script: Path,
    plan_path: Path,
    output_root: Path,
    report_path: Path,
    log_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        str(python),
        str(worker_script),
        "--plan",
        str(plan_path),
        "--output-root",
        str(output_root),
        "--report",
        str(report_path),
    ]
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **environment},
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise ME2ExperimentError(f"GPU worker rc={result.returncode}:\n{tail}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def run(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if git(repo_root, "status", "--porcelain"):
        raise ME2ExperimentError("正式 ME-2 要求 motion_proj 工作树干净")
    source_commit = git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise ME2ExperimentError("ME-2 task_id 漂移")
    if importlib.metadata.version("pymeshlab") != str(config["environment"]["pymeshlab"]):
        raise ME2ExperimentError("Hunyuan3D-2.1 pymeshlab dependency 漂移")

    p4_root = resolve_runs_uri(config["sources"]["p4_run"])
    me0_root = resolve_runs_uri(config["sources"]["me0_run"])
    verify_files(p4_root, config["sources"]["p4_files"])
    verify_files(me0_root, config["sources"]["me0_files"])
    p4_gate = json.loads((p4_root / "P4_GATE.json").read_text(encoding="utf-8"))
    me0_gate = json.loads((me0_root / "ME0_GATE.json").read_text(encoding="utf-8"))
    if not p4_gate.get("passed") or not me0_gate.get("checks", {}).get("passed"):
        raise ME2ExperimentError("P4 或 ME-0 authority 未通过")

    r9_root = Path(config["sources"]["r9_root"])
    verify_files(r9_root, config["sources"]["r9_files"])
    all_cases = read_jsonl(r9_root / "CASES.jsonl")
    cases = [row for row in all_cases if row["hole_type"] == "actor_removal_hole"]
    if len(cases) != int(config["cohort"]["expected_case_count"]):
        raise ME2ExperimentError("actor case denominator 漂移")
    for case in cases:
        path = r9_root / "generator_inputs" / f"{case['case_id']}.npz"
        if sha256_file(path) != case["generator_input_sha256"]:
            raise ME2ExperimentError(f"generator input 漂移: {case['case_id']}")

    omni_repo = Path(config["sources"]["omni_repo"])
    base_repo = Path(config["sources"]["image_base_repo"])
    if git(omni_repo, "rev-parse", "HEAD") != config["sources"]["omni_git_commit"]:
        raise ME2ExperimentError("Omni official commit 漂移")
    if git(base_repo, "rev-parse", "HEAD") != config["sources"]["image_base_git_commit"]:
        raise ME2ExperimentError("image-only official commit 漂移")
    verify_files(omni_repo, config["sources"]["omni_files"])
    verify_files(base_repo, config["sources"]["image_base_files"])
    omni_model = Path(config["sources"]["omni_model_root"])
    base_model = Path(config["sources"]["image_base_model_root"])
    verify_files(omni_model, config["sources"]["omni_model_files"])
    verify_files(base_model, config["sources"]["image_base_model_files"])
    dino_ref = Path(config["environment"]["hf_home"]) / config["environment"]["dino_cache_ref_path"]
    if dino_ref.read_text(encoding="utf-8") != config["sources"]["dino_revision"]:
        raise ME2ExperimentError("DINO offline ref 漂移")

    run_root.mkdir(parents=True, exist_ok=True)
    disk_free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__hy3d-actor-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        unit_rows: list[dict[str, Any]] = []
        unit_index: dict[str, dict[str, Any]] = {}
        for unit in config["units"]:
            unit_id = str(unit["unit_id"])
            output_dir = run_dir / "inputs" / unit_id
            scene_root = Path(config["raw_evidence"][unit["scene"]])
            method_grid = me0_root / unit["method_grid"]
            audit = build_actor_controls(
                scene=unit["scene"],
                scene_root=scene_root,
                target_frame=int(unit["target_frame"]),
                actor_id=int(unit["actor_id"]),
                source_frames=unit["method_source_frames"],
                method_grid_path=method_grid,
                output_dir=output_dir,
                camera_id=int(config["input_contract"]["camera_id"]),
                point_count=int(config["input_contract"]["point_control_count"]),
                voxel_count=int(config["input_contract"]["voxel_control_count"]),
                lidar_record_width=int(config["input_contract"]["lidar_record_width"]),
                box_margin_m=float(config["input_contract"]["box_margin_m"]),
                crop_border_fraction=float(config["input_contract"]["crop_border_fraction"]),
            )
            row = {
                **unit,
                **audit,
                "seed": int(unit["seed"]),
                "actor_rgba_sha256": sha256_file(Path(audit["actor_rgba_path"])),
                "controls_sha256": sha256_file(Path(audit["controls_path"])),
            }
            write_json(output_dir / "INPUT_AUDIT.json", row)
            unit_rows.append(row)
            unit_index[unit_id] = row
        write_jsonl(run_dir / "INPUT_INDEX.jsonl", unit_rows)

        case_bindings: list[dict[str, Any]] = []
        case_to_unit = config["case_to_unit"]
        for case in cases:
            unit = unit_index[case_to_unit[case["case_id"]]]
            scene_root = Path(config["raw_evidence"][case["scene"]])
            mask_path = r9_root / "generator_inputs" / f"{case['case_id']}.npz"
            mask = _case_mask(mask_path)
            binding = _binding_factors(
                scene_root,
                int(case["frame_index"]),
                int(unit["actor_id"]),
                int(config["input_contract"]["camera_id"]),
                mask,
            )
            if binding["hole_coverage"] < float(config["input_contract"]["minimum_target_hole_coverage"]):
                raise ME2ExperimentError(f"target actor binding 不足: {case['case_id']}")
            case_bindings.append(
                {
                    "case_id": case["case_id"],
                    "unit_id": unit["unit_id"],
                    "actor_id": unit["actor_id"],
                    "binding": binding,
                }
            )
        write_jsonl(run_dir / "CASE_BINDINGS.jsonl", case_bindings)

        worker_units = [
            {
                "unit_id": row["unit_id"],
                "seed": row["seed"],
                "image_path": row["actor_rgba_path"],
                "controls_path": row["controls_path"],
            }
            for row in unit_rows
        ]
        base_plan = {
            "backend": "image_base",
            "repo": str(base_repo),
            "model_root": str(base_model),
            "global_seed": int(config["seed"]),
            "units": worker_units,
            **config["generation"]["A0-image"],
        }
        omni_plan = {
            "backend": "omni",
            "repo": str(omni_repo),
            "model_root": str(omni_model),
            "global_seed": int(config["seed"]),
            "units": worker_units,
            **config["generation"]["omni_controls"],
        }
        write_json(run_dir / "A0_WORKER_PLAN.json", base_plan)
        write_json(run_dir / "OMNI_WORKER_PLAN.json", omni_plan)
        worker_script = repo_root / "scripts/run_worldsim_v61_me2_hy3d_worker.py"
        worker_python = Path(config["environment"]["python"])
        worker_environment = {
            "HF_HOME": config["environment"]["hf_home"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        base_report = _worker(
            worker_python,
            worker_script,
            run_dir / "A0_WORKER_PLAN.json",
            run_dir / "raw_assets",
            run_dir / "A0_WORKER_REPORT.json",
            run_dir / "A0_WORKER.log",
            worker_environment,
        )
        omni_report = _worker(
            worker_python,
            worker_script,
            run_dir / "OMNI_WORKER_PLAN.json",
            run_dir / "raw_assets",
            run_dir / "OMNI_WORKER_REPORT.json",
            run_dir / "OMNI_WORKER.log",
            worker_environment,
        )
        raw_assets = base_report["assets"] + omni_report["assets"]
        expected_assets = len(unit_rows) * len(config["arms"])
        if len(raw_assets) != expected_assets:
            raise ME2ExperimentError(f"raw asset 数漂移: {len(raw_assets)} != {expected_assets}")

        raw_asset_index = {(row["arm"], row["unit_id"]): row for row in raw_assets}
        compiled_assets: list[dict[str, Any]] = []
        for arm in config["arms"]:
            for unit in unit_rows:
                raw = raw_asset_index[(arm, unit["unit_id"])]
                output_root = run_dir / "compiled_assets" / arm
                prepared = prepare_compiled_asset(
                    mesh_path=Path(raw["path"]),
                    native_size_lwh=np.asarray(unit["native_size_lwh"], dtype=np.float64),
                    surface_count=int(config["compiler"]["surface_gaussian_count"]),
                    seed=int(unit["seed"]),
                    mesh_output_path=output_root / f"{unit['unit_id']}.glb",
                    surface_output_path=output_root / f"{unit['unit_id']}.npz",
                    audit_output_path=output_root / f"{unit['unit_id']}.json",
                    mean_rgb=_mean_actor_rgb(Path(unit["actor_rgba_path"])),
                )
                compiled_assets.append(
                    {
                        **raw,
                        **prepared,
                        "canonical_mesh_sha256": sha256_file(Path(prepared["canonical_mesh_path"])),
                        "surface_gaussians_sha256": sha256_file(
                            Path(prepared["surface_gaussians_path"])
                        ),
                    }
                )
        write_jsonl(run_dir / "ASSET_INDEX.jsonl", compiled_assets)
        compiled_index = {(row["arm"], row["unit_id"]): row for row in compiled_assets}

        legal_by_scene = {
            scene: {
                frozenset((int(pair[0]), int(pair[1])))
                for pair in pairs
            }
            for scene, pairs in config["kinematic_relations"]["legal_collision_pairs"].items()
        }
        method_rows: list[dict[str, Any]] = []
        for case in cases:
            unit = unit_index[case_to_unit[case["case_id"]]]
            scene_root = Path(config["raw_evidence"][case["scene"]])
            mask = _case_mask(r9_root / "generator_inputs" / f"{case['case_id']}.npz")
            for arm in config["arms"]:
                asset = compiled_index[(arm, unit["unit_id"])]
                factors = evaluate_mesh(
                    mesh_path=Path(asset["path"]),
                    scene_root=scene_root,
                    scene=case["scene"],
                    target_frame=int(case["frame_index"]),
                    actor_id=int(unit["actor_id"]),
                    evidence_grid_path=me0_root / unit["method_grid"],
                    case_mask=mask,
                    thresholds=config["method_and_eval_gate"],
                    interpolation_samples=int(config["compiler"]["interpolation_samples"]),
                    legal_pairs=legal_by_scene.get(case["scene"], set()),
                    surface_count=int(config["compiler"]["surface_gaussian_count"]),
                    seed=int(unit["seed"]),
                    camera_id=int(config["input_contract"]["camera_id"]),
                    prepared_asset=asset,
                )
                method_rows.append(
                    {
                        "schema_version": "worldsim_v61.me2_method_decision.v1",
                        "case_id": case["case_id"],
                        "scene": case["scene"],
                        "frame_index": int(case["frame_index"]),
                        "frontend": case["frontend"],
                        "arm": arm,
                        "unit_id": unit["unit_id"],
                        "actor_id": int(unit["actor_id"]),
                        "raw_asset_sha256": asset["sha256"],
                        "canonical_mesh_sha256": asset["canonical_mesh_sha256"],
                        "method_factors": factors,
                        "method_decision": "ACCEPT" if factors["passed"] else "REJECT",
                        "decision_inputs": [
                            "raw_actor_image",
                            "native_actor_metadata",
                            "O_method",
                            "generated_mesh",
                        ],
                    }
                )
        method_path = run_dir / "METHOD_DECISIONS.jsonl"
        write_jsonl(method_path, method_rows)
        method_sha = sha256_file(method_path)

        per_case: list[dict[str, Any]] = []
        for method in method_rows:
            case = next(row for row in cases if row["case_id"] == method["case_id"])
            unit = unit_index[method["unit_id"]]
            asset = compiled_index[(method["arm"], method["unit_id"])]
            scene_root = Path(config["raw_evidence"][case["scene"]])
            mask = _case_mask(r9_root / "generator_inputs" / f"{case['case_id']}.npz")
            evaluation = evaluate_mesh(
                mesh_path=Path(asset["path"]),
                scene_root=scene_root,
                scene=case["scene"],
                target_frame=int(case["frame_index"]),
                actor_id=int(unit["actor_id"]),
                evidence_grid_path=me0_root / unit["eval_grid"],
                case_mask=mask,
                thresholds=config["method_and_eval_gate"],
                interpolation_samples=int(config["compiler"]["interpolation_samples"]),
                legal_pairs=legal_by_scene.get(case["scene"], set()),
                surface_count=int(config["compiler"]["surface_gaussian_count"]),
                seed=int(unit["seed"]),
                camera_id=int(config["input_contract"]["camera_id"]),
                prepared_asset=asset,
            )
            truth_safe = bool(evaluation["passed"])
            false_safe = method["method_decision"] == "ACCEPT" and not truth_safe
            per_case.append(
                {
                    **method,
                    "decision_artifact_frozen_before_eval_sha256": method_sha,
                    "independent_eval_factors": evaluation,
                    "independent_truth_safe": truth_safe,
                    "false_safe": false_safe,
                }
            )
        write_jsonl(run_dir / "PER_CASE.jsonl", per_case)

        arm_summaries = []
        for arm in config["arms"]:
            rows = [row for row in per_case if row["arm"] == arm]
            accepted = [row for row in rows if row["method_decision"] == "ACCEPT"]
            arm_summaries.append(
                {
                    "schema_version": "worldsim_v61.me2_arm_summary.v1",
                    "arm": arm,
                    "case_count": len(rows),
                    "accepted_count": len(accepted),
                    "rejected_count": len(rows) - len(accepted),
                    "false_safe_count": sum(bool(row["false_safe"]) for row in rows),
                    "accepted_free_space_conflict_count": sum(
                        row["method_factors"]["factors"]["free_space_conflict_count"]
                        for row in accepted
                    ),
                    "accepted_swept_collision_count": sum(
                        len(row["method_factors"]["factors"]["collision"]["unfiltered_collisions"])
                        for row in accepted
                    ),
                    "accepted_case_ids": [row["case_id"] for row in accepted],
                }
            )
        write_jsonl(run_dir / "ARM_SUMMARIES.jsonl", arm_summaries)
        primary = next(row for row in arm_summaries if row["arm"] == "A3-voxel")
        elapsed = time.monotonic() - started
        peak_gpu = max(
            float(base_report["peak_gpu_memory_gib"]), float(omni_report["peak_gpu_memory_gib"])
        )
        checks = {
            "p4_and_me0_authority_passed": True,
            "matched_four_arms_complete": all(
                row["case_count"] == int(config["cohort"]["expected_case_count"])
                for row in arm_summaries
            ),
            "primary_voxel_minimum_accept_count": primary["accepted_count"]
            >= int(config["primary_gate"]["minimum_accepted_cases"]),
            "primary_false_safe_zero": primary["false_safe_count"]
            <= int(config["primary_gate"]["maximum_false_safe_count"]),
            "primary_free_space_conflict_zero": primary[
                "accepted_free_space_conflict_count"
            ]
            == 0,
            "primary_swept_collision_zero": primary["accepted_swept_collision_count"] == 0,
            "offline_workers": all(value == "1" for key, value in worker_environment.items() if "OFFLINE" in key),
            "gpu_memory_within_budget": peak_gpu
            <= float(config["resources"]["maximum_gpu_memory_gib"]),
            "wall_within_budget": elapsed <= float(config["resources"]["maximum_wall_seconds"]),
            "disk_free_within_budget": disk_free_gib
            >= float(config["resources"]["minimum_disk_free_gib"]),
            "no_training_or_confirmation": True,
        }
        gate = {
            "schema_version": "worldsim_v61.me2_gate.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "checks": checks,
            "passed": all(checks.values()),
        }
        write_json(run_dir / "ME2_GATE.json", gate)
        resource = {
            "schema_version": "worldsim_v61.me2_resource_audit.v1",
            "python": platform.python_version(),
            "gpu": base_report["gpu"],
            "worker_torch": base_report["torch"],
            "worker_cuda": base_report["cuda"],
            "base_worker_wall_seconds": base_report["wall_seconds"],
            "omni_worker_wall_seconds": omni_report["wall_seconds"],
            "peak_gpu_memory_gib": peak_gpu,
            "wall_seconds": elapsed,
            "disk_free_gib_at_start": disk_free_gib,
            "raw_generated_asset_count": len(raw_assets),
            "unique_generation_unit_count": len(unit_rows),
            "evaluated_case_arm_count": len(per_case),
            "training_started": False,
            "confirmation_content_read": False,
        }
        write_json(run_dir / "RESOURCE_AUDIT.json", resource)
        summary = {
            "schema_version": "worldsim_v61.me2_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_commit": source_commit,
            "status": "done" if gate["passed"] else "rejected",
            "hypothesis_outcome": (
                "accepted_voxel_control_actor_upper_bound"
                if gate["passed"]
                else "rejected_voxel_control_actor_upper_bound"
            ),
            "primary_arm": primary,
            "next": (
                "WS-V61-ME3-PREDICTED-OCC-01"
                if gate["passed"]
                else "stop_hy3d_without_prompt_seed_texture_tuning"
            ),
            "failure_ledger_delta": "none" if gate["passed"] else "required",
            "claim_boundary": config["claim_boundary"],
        }
        write_json(run_dir / "SUMMARY.json", summary)
        artifact_names = (
            "ME2_GATE.json",
            "ARM_SUMMARIES.jsonl",
            "PER_CASE.jsonl",
            "METHOD_DECISIONS.jsonl",
            "CASE_BINDINGS.jsonl",
            "INPUT_INDEX.jsonl",
            "ASSET_INDEX.jsonl",
            "A0_WORKER_PLAN.json",
            "OMNI_WORKER_PLAN.json",
            "A0_WORKER_REPORT.json",
            "OMNI_WORKER_REPORT.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        )
        manifest = {
            "schema_version": "worldsim_v61.me2_manifest.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "method_decisions_frozen_before_eval_sha256": method_sha,
            "omni_model_revision": config["sources"]["omni_model_revision"],
            "image_base_model_revision": config["sources"]["image_base_model_revision"],
            "artifacts": {name: sha256_file(run_dir / name) for name in artifact_names},
        }
        write_json(run_dir / "MANIFEST.json", manifest)
        write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.me2_terminal.v1",
                "task_id": TASK_ID,
                "status": summary["status"],
                "canonical": True,
                "run_uri": f"run://worldsim_v61/{TASK_ID}/{run_dir.name}",
            },
        )
        return run_dir
    except Exception as error:
        write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.me2_terminal.v1",
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", default=Path("/root/autodl-tmp/runs/worldsim_v61"), type=Path)
    args = parser.parse_args()
    print(run(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

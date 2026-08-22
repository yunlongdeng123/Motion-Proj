"""WorldSim V6.1 ME-3R：IR-WM predicted occupancy 唯一科学恢复。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v61.me1_oracle import (
    _camera_contract,
    _git,
    _independent_eval_factors,
    _load_grid,
    _raycast,
    _read_jsonl,
    _resolve_runs_uri,
    _verify_files,
    _write_json,
    _write_jsonl,
    method_decision,
    occupancy_gate,
)
from motion_proj.worldsim_v61.me3_predicted import (
    bind_native_actor_identity_without_geometry_fill,
    predicted_method_factors,
    raycast_predicted_conservative,
    resample_irwm_classes,
)
from motion_proj.worldsim_v61.occupancy import UNKNOWN, VoxelGridSpec, sha256_file


TASK_ID = "WS-V61-ME3R-IRWM-PREDICTED-OCC-01"


class ME3ExperimentError(RuntimeError):
    """ME-3R source、预测、适配或 verifier 合同失败。"""


def _frozen_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_scene_files(
    scene_root: Path, frames: list[int], camera_ids: list[int]
) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for camera_id in camera_ids:
        paths.append(scene_root / f"intrinsics/{camera_id}.txt")
    for frame in frames:
        paths.append(scene_root / f"lidar_pose/{frame:03d}.txt")
        for camera_id in camera_ids:
            paths.extend(
                (
                    scene_root / f"images/{frame:03d}_{camera_id}.jpg",
                    scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt",
                )
            )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ME3ExperimentError(f"streaming 输入缺失: {missing[:4]}")
    return [_frozen_record(path) for path in paths]


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    decisions = [row["decisions"][arm] for row in rows]
    accepted = [row for row in rows if row["decisions"][arm] == "ACCEPT"]
    return {
        "schema_version": "worldsim_v61.me3_arm_summary.v1",
        "arm": arm,
        "denominator": len(rows),
        "accept_count": decisions.count("ACCEPT"),
        "abstain_count": decisions.count("ABSTAIN"),
        "reject_count": decisions.count("REJECT"),
        "false_safe_count": sum(bool(row["predicted_false_safe"]) for row in accepted),
        "accepted_mask_pixels": sum(int(row["mask_pixel_count"]) for row in accepted),
        "accepted_case_ids": [row["case_id"] for row in accepted],
    }


def _wait_parallel_workers(
    processes: list[subprocess.Popen[str]], deadline: float
) -> list[int]:
    while any(process.poll() is None for process in processes):
        if time.monotonic() >= deadline:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            raise ME3ExperimentError("IR-WM parallel workers 超时")
        time.sleep(0.5)
    return [int(process.returncode or 0) for process in processes]


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise ME3ExperimentError("正式 ME-3 要求 motion_proj 工作树干净")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise ME3ExperimentError("ME-3 task_id 漂移")
    if not torch.cuda.is_available():
        raise ME3ExperimentError("ME-3 verifier raycast 需要 CUDA")

    p7r_root = _resolve_runs_uri(config["sources"]["p7r_run"])
    me0_root = _resolve_runs_uri(config["sources"]["me0_run"])
    me1_root = _resolve_runs_uri(config["sources"]["me1_run"])
    r9_root = Path(config["sources"]["r9_cross_run"])
    r10_root = Path(config["sources"]["r10_run"])
    for root, files in (
        (p7r_root, config["sources"]["p7r_files"]),
        (me0_root, config["sources"]["me0_files"]),
        (me1_root, config["sources"]["me1_files"]),
        (r9_root, config["sources"]["r9_cross_files"]),
        (r10_root, config["sources"]["r10_files"]),
    ):
        _verify_files(root, files)
    p7r_gate = json.loads((p7r_root / "P7R_GATE.json").read_text(encoding="utf-8"))
    p7r_terminal = json.loads((p7r_root / "TERMINAL.json").read_text(encoding="utf-8"))
    if not p7r_gate.get("passed") or not p7r_terminal.get("canonical"):
        raise ME3ExperimentError("P7R IR-WM capability authority 未通过")

    official_repo = Path(config["model"]["official_repo"])
    archive_cfg = config["model"]["source_archive"]
    archive_record = _frozen_record(official_repo / archive_cfg["path"])
    if (
        archive_record["bytes"] != int(archive_cfg["bytes"])
        or archive_record["sha256"] != archive_cfg["sha256"]
    ):
        raise ME3ExperimentError("IR-WM official source archive 漂移")
    official_records = [archive_record]
    for name, expected_hash in config["model"]["official_files"].items():
        record = _frozen_record(official_repo / name)
        if record["sha256"] != expected_hash:
            raise ME3ExperimentError(f"IR-WM official source 漂移: {name}")
        official_records.append(record)
    model_root = Path(config["model"]["model_root"])
    model_records = []
    for name, expected in config["model"]["files"].items():
        record = _frozen_record(model_root / name)
        if record["bytes"] != int(expected["bytes"]) or record["sha256"] != expected["sha256"]:
            raise ME3ExperimentError(f"IR-WM model 输入漂移: {name}")
        model_records.append(record)
    metadata_cfg = config["model"]["temporal_metadata"]
    metadata_path = Path(metadata_cfg["path"])
    metadata_record = _frozen_record(metadata_path)
    if (
        metadata_record["bytes"] != int(metadata_cfg["bytes"])
        or metadata_record["sha256"] != metadata_cfg["sha256"]
    ):
        raise ME3ExperimentError("IR-WM temporal metadata 漂移")

    cases = _read_jsonl(r9_root / "CASES.jsonl")
    if len(cases) != int(config["cohort"]["expected_case_count"]):
        raise ME3ExperimentError("28-case denominator 漂移")
    r9_rows = {
        row["case_id"]: row
        for row in _read_jsonl(r9_root / "verifier_worker/PER_CASE_ARMS.jsonl")
    }
    r10_rows = {
        row["case_id"]: row for row in _read_jsonl(r10_root / "FACTORIZED_DECISIONS.jsonl")
    }
    me1_rows = {row["case_id"]: row for row in _read_jsonl(me1_root / "PER_CASE.jsonl")}
    case_ids = {row["case_id"] for row in cases}
    if case_ids != set(r9_rows) or case_ids != set(r10_rows) or case_ids != set(me1_rows):
        raise ME3ExperimentError("baseline case identity 漂移")

    run_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__irwm-predicted-occ-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    processes: list[subprocess.Popen[str]] = []
    try:
        camera_ids = [int(value) for value in config["streaming"]["camera_ids"]]
        target_frames = [int(value) for value in config["streaming"]["target_frames"]]
        targets = []
        for frame in target_frames:
            window = config["streaming"]["history_windows"][str(frame)]
            targets.append(
                {
                    "frames": [int(value) for value in window["frames"]],
                    "metadata_indices": [
                        int(value) for value in window["metadata_indices"]
                    ],
                }
            )
        input_frames = sorted(
            {frame for target in targets for frame in target["frames"]}
        )
        worker_plans: list[dict[str, Any]] = []
        worker_logs: list[Any] = []
        worker_script = repo_root / "scripts/run_worldsim_v61_me3_irwm_worker.py"
        worker_python = Path(config["environment"]["prefix"]) / "bin/python"
        for scene in config["cohort"]["scenes"]:
            scene_root = Path(config["raw_evidence"][scene])
            plan = {
                "schema_version": "worldsim_v61.me3r_irwm_worker_plan.v1",
                "seed": int(config["seed"]),
                "gpu": int(config["resources"]["gpu"]),
                "scene": scene,
                "scene_root": str(scene_root),
                "target_frames": target_frames,
                "targets": targets,
                "camera_ids": camera_ids,
                "camera_names": config["streaming"]["camera_names"],
                "native_shape": config["streaming"]["native_shape"],
                "pad_size_divisor": config["streaming"]["pad_size_divisor"],
                "image_mean_bgr": config["streaming"]["image_mean_bgr"],
                "image_std": config["streaming"]["image_std"],
                "official_repo": str(official_repo),
                "official_config": str(official_repo / config["model"]["official_config"]),
                "checkpoint_path": str(model_root / config["model"]["checkpoint_name"]),
                "temporal_metadata_path": str(metadata_path),
                **config["model"]["output_contract"],
                "frozen_files": _worker_scene_files(
                    scene_root, input_frames, camera_ids
                ),
            }
            plan_path = run_dir / f"WORKER_PLAN_{scene}.json"
            _write_json(plan_path, plan)
            output_dir = run_dir / "predictions" / scene
            report_path = run_dir / f"WORKER_REPORT_{scene}.json"
            log_path = run_dir / f"WORKER_{scene}.log"
            log_stream = log_path.open("w", encoding="utf-8")
            worker_logs.append(log_stream)
            command = [
                str(worker_python),
                str(worker_script),
                "--plan",
                str(plan_path),
                "--output-dir",
                str(output_dir),
                "--report",
                str(report_path),
            ]
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(config["resources"]["gpu"]),
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": str(config["resources"]["worker_cpu_threads"]),
                "MKL_NUM_THREADS": str(config["resources"]["worker_cpu_threads"]),
                "PATH": str(Path(config["environment"]["prefix"]) / "bin")
                + os.pathsep
                + os.environ.get("PATH", ""),
                "TORCH_CUDA_ARCH_LIST": "8.6",
            }
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
            )
            worker_plans.append(plan)
        deadline = time.monotonic() + float(config["resources"]["worker_timeout_seconds"])
        return_codes = _wait_parallel_workers(processes, deadline)
        for stream in worker_logs:
            stream.close()
        if any(code != 0 for code in return_codes):
            tails = []
            for scene in config["cohort"]["scenes"]:
                log_path = run_dir / f"WORKER_{scene}.log"
                tails.append("\n".join(log_path.read_text(encoding="utf-8").splitlines()[-20:]))
            raise ME3ExperimentError("IR-WM worker failed:\n" + "\n".join(tails))

        reports = [
            json.loads((run_dir / f"WORKER_REPORT_{scene}.json").read_text(encoding="utf-8"))
            for scene in config["cohort"]["scenes"]
        ]
        expected_versions = config["environment"]
        workers_exact = all(
            sorted(report["model_load_missing_keys"])
            == sorted(config["model"]["allowed_missing_keys"])
            and not report["model_load_unexpected_keys"]
            and report["target_frames"] == target_frames
            and report["python"] == str(expected_versions["python"])
            and report["torch"] == str(expected_versions["torch"])
            and report["environment_versions"]["mmcv-full"]
            == str(expected_versions["mmcv-full"])
            and report["environment_versions"]["mmdet"] == str(expected_versions["mmdet"])
            and report["environment_versions"]["mmsegmentation"]
            == str(expected_versions["mmsegmentation"])
            and report["environment_versions"]["mmdet3d"] == str(expected_versions["mmdet3d"])
            and report["environment_versions"]["detectron2"]
            == str(expected_versions["detectron2"])
            and report["current_state_extraction"]
            == "official_scene_encoder_plus_official_final_decoder_occupancy_head"
            and not report["future_decoder_started"]
            and not report["planning_head_started"]
            and not report["occupancy_ground_truth_read"]
            and not report["o_method_or_o_eval_read"]
            and not report["training_started"]
            and not report["confirmation_content_read"]
            for report in reports
        )
        if not workers_exact:
            raise ME3ExperimentError("IR-WM worker source/environment contract 漂移")

        torch.cuda.set_device(int(config["resources"]["gpu"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        target_spec = VoxelGridSpec(
            frame="target_lidar",
            origin_m=tuple(float(value) for value in config["target_grid"]["origin_m"]),
            voxel_size_m=float(config["target_grid"]["voxel_size_m"]),
            shape=tuple(int(value) for value in config["target_grid"]["shape"]),
        )
        method_rays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        method_grids: dict[tuple[str, int], dict[str, Any]] = {}
        adapter_rows: list[dict[str, Any]] = []
        projection_index: list[dict[str, Any]] = []
        for scene in config["cohort"]["scenes"]:
            scene_root = Path(config["raw_evidence"][scene])
            for frame in target_frames:
                source_path = (
                    run_dir
                    / "predictions"
                    / scene
                    / f"f{frame:03d}/IRWM_CLASS.npz"
                )
                source = np.load(source_path, allow_pickle=False)
                adapted = resample_irwm_classes(
                    np.asarray(source["class_label"], dtype=np.uint8),
                    target_spec,
                    source_origin_m=tuple(
                        float(value) for value in config["model"]["output_contract"]["source_origin_m"]
                    ),
                    source_voxel_size_m=float(
                        config["model"]["output_contract"]["grid_size_m"]
                    ),
                    empty_class=int(config["model"]["output_contract"]["empty_class"]),
                    occupied_class_min=int(
                        config["model"]["output_contract"]["occupied_class_min"]
                    ),
                    occupied_class_max=int(
                        config["model"]["output_contract"]["occupied_class_max"]
                    ),
                )
                actor_grid, identity_rows = bind_native_actor_identity_without_geometry_fill(
                    adapted["semantics"], target_spec, scene_root, frame
                )
                grid = {
                    "semantics": adapted["semantics"],
                    "actor_grid": actor_grid,
                    "origin": target_spec.origin,
                    "voxel_size": target_spec.voxel_size_m,
                }
                method_grids[(scene, frame)] = grid
                adapted_path = run_dir / f"adapted/{scene}/f{frame:03d}/PREDICTED_OCCUPANCY.npz"
                adapted_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    adapted_path,
                    semantics=adapted["semantics"],
                    actor_grid=actor_grid,
                    predicted_class=adapted["predicted_class"],
                    source_valid=adapted["source_valid"],
                    grid_origin_m=target_spec.origin.astype(np.float64),
                    voxel_size_m=np.asarray(target_spec.voxel_size_m, dtype=np.float64),
                )
                t_lidar_camera, intrinsics = _camera_contract(
                    scene_root, frame, config["projection"]
                )
                rays = raycast_predicted_conservative(
                    grid, t_lidar_camera, intrinsics, config["projection"]
                )
                method_rays[(scene, frame)] = rays
                projection_path = run_dir / f"projections/{scene}/f{frame:03d}/PREDICTED.npz"
                projection_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(projection_path, **rays)
                adapter_rows.append(
                    {
                        "schema_version": "worldsim_v61.me3_adapter_record.v1",
                        "scene": scene,
                        "frame": frame,
                        "source_path": str(source_path.relative_to(run_dir)),
                        "source_sha256": sha256_file(source_path),
                        "adapted_path": str(adapted_path.relative_to(run_dir)),
                        "adapted_sha256": sha256_file(adapted_path),
                        "unknown_voxel_count": int(
                            np.count_nonzero(adapted["semantics"] == int(UNKNOWN))
                        ),
                        "identity_rows": identity_rows,
                        "geometry_created_by_identity_binding": False,
                    }
                )
                projection_index.append(
                    {
                        "scene": scene,
                        "frame": frame,
                        "tier": "PREDICTED_IRWM",
                        "path": str(projection_path.relative_to(run_dir)),
                        "sha256": sha256_file(projection_path),
                        "occupied_hit_pixels": int(np.count_nonzero(rays["voxel_linear"] >= 0)),
                        "unknown_blocked_pixels": int(np.count_nonzero(rays["unknown_blocked"])),
                    }
                )
        _write_jsonl(run_dir / "ADAPTER_INDEX.jsonl", adapter_rows)

        method_rows: list[dict[str, Any]] = []
        proposal_root = run_dir / "compiled_proposals"
        proposal_root.mkdir()
        for case in cases:
            case_id = case["case_id"]
            key = (case["scene"], int(case["frame_index"]))
            payload = np.load(
                r9_root / "verifier_inputs" / f"{case_id}.npz", allow_pickle=False
            )
            mask = np.asarray(payload["mask"], dtype=bool)
            factors, candidate_linear, actor_ids = predicted_method_factors(
                mask,
                method_rays[key],
                case,
                float(config["method_gate"]["minimum_surface_coverage"]),
            )
            proposal = np.load(
                r9_root
                / "cross_frontend_reconstruction_proposals"
                / f"{case_id}__repeat1.npy",
                allow_pickle=False,
            )
            hit_linear = method_rays[key]["voxel_linear"]
            selected = mask & (hit_linear >= 0)
            selected_linear = hit_linear[selected]
            selected_rgb = proposal[selected]
            order = np.argsort(selected_linear, kind="stable")
            sorted_linear = selected_linear[order]
            first = (
                np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
                if sorted_linear.size
                else np.asarray([], dtype=bool)
            )
            unique_linear = sorted_linear[first]
            unique_rgb = selected_rgb[order][first]
            if set(unique_linear.tolist()) != set(candidate_linear.tolist()):
                raise ME3ExperimentError(f"predicted surface attachment 漂移: {case_id}")
            relative = Path(f"compiled_proposals/{case_id}.npz")
            compiled_path = run_dir / relative
            np.savez_compressed(
                compiled_path,
                occupied_voxel_linear=unique_linear.astype(np.int64),
                appearance_rgb_uint8=unique_rgb.astype(np.uint8),
                actor_instance_ids=np.asarray(actor_ids, dtype=np.int32),
                geometry_tier=np.asarray("PREDICTED_IRWM"),
            )
            decisions = {
                "B1-R10": r10_rows[case_id]["overall_decision"],
                "O2-ORACLE-UPPER-BOUND": me1_rows[case_id]["decisions"]["O2-OCC-GEOMETRY"],
                "P1-IRWM-PREDICTED": method_decision(
                    r9_rows[case_id]["P1"]["decision"], bool(factors["passed"])
                ),
            }
            method_rows.append(
                {
                    "schema_version": "worldsim_v61.me3_method_decision.v1",
                    "case_id": case_id,
                    "scene": case["scene"],
                    "frame_index": int(case["frame_index"]),
                    "frontend": case["frontend"],
                    "hole_type": case["hole_type"],
                    "mask_pixel_count": int(mask.sum()),
                    "predicted_geometry_factors": factors,
                    "compiled_proposal_path": str(relative),
                    "compiled_proposal_sha256": sha256_file(compiled_path),
                    "decision_inputs": [
                        "frozen_R9_P1_photo",
                        "IRWM_predicted_occupancy",
                        "native_actor_identity_only_without_geometry_fill",
                    ],
                    "decisions": decisions,
                }
            )
        _write_jsonl(run_dir / "METHOD_DECISIONS.jsonl", method_rows)
        method_sha256 = sha256_file(run_dir / "METHOD_DECISIONS.jsonl")

        eval_rays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        eval_grids: dict[tuple[str, int], dict[str, Any]] = {}
        for scene in config["cohort"]["scenes"]:
            scene_root = Path(config["raw_evidence"][scene])
            for frame in target_frames:
                grid = _load_grid(me0_root / f"evidence/{scene}/f{frame:03d}/O_eval.npz")
                t_lidar_camera, intrinsics = _camera_contract(
                    scene_root, frame, config["projection"]
                )
                rays = _raycast(grid, t_lidar_camera, intrinsics, config["projection"])
                eval_grids[(scene, frame)] = grid
                eval_rays[(scene, frame)] = rays
                path = run_dir / f"projections/{scene}/f{frame:03d}/O_eval.npz"
                np.savez_compressed(path, **rays)
                projection_index.append(
                    {
                        "scene": scene,
                        "frame": frame,
                        "tier": "O_eval_hidden",
                        "path": str(path.relative_to(run_dir)),
                        "sha256": sha256_file(path),
                        "occupied_hit_pixels": int(np.count_nonzero(rays["voxel_linear"] >= 0)),
                    }
                )

        def score(row: dict[str, Any]) -> dict[str, Any]:
            key = (row["scene"], int(row["frame_index"]))
            payload = np.load(
                r9_root / "verifier_inputs" / f"{row['case_id']}.npz", allow_pickle=False
            )
            mask = np.asarray(payload["mask"], dtype=bool)
            compiled = np.load(run_dir / row["compiled_proposal_path"], allow_pickle=False)
            candidate_linear = np.asarray(compiled["occupied_voxel_linear"], dtype=np.int64)
            factors = _independent_eval_factors(
                mask,
                candidate_linear,
                method_rays[key],
                eval_grids[key],
                eval_rays[key],
                int(config["projection"]["observed_support_dilation_voxels"]),
            )
            factors["passed"] = occupancy_gate(factors, config["independent_eval_gate"])
            predicted_decision = row["decisions"]["P1-IRWM-PREDICTED"]
            return {
                **row,
                "independent_eval_geometry_factors": factors,
                "predicted_false_safe": predicted_decision == "ACCEPT" and not factors["passed"],
                "decision_artifact_frozen_before_eval_sha256": method_sha256,
            }

        workers = min(int(config["resources"]["maximum_cpu_workers"]), len(method_rows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            final_rows = list(executor.map(score, method_rows))
        _write_jsonl(run_dir / "PER_CASE.jsonl", final_rows)
        _write_jsonl(run_dir / "PROJECTION_INDEX.jsonl", projection_index)

        predicted = _arm_summary(final_rows, "P1-IRWM-PREDICTED")
        total_mask_pixels = sum(int(row["mask_pixel_count"]) for row in final_rows)
        predicted["accept_coverage"] = predicted["accept_count"] / len(final_rows)
        predicted["accepted_mask_area_yield"] = (
            predicted["accepted_mask_pixels"] / total_mask_pixels
        )
        me1_arm_rows = {
            row["arm"]: row for row in _read_jsonl(me1_root / "ARM_SUMMARIES.jsonl")
        }
        oracle = dict(me1_arm_rows["O2-OCC-GEOMETRY"])
        oracle["arm"] = "O2-ORACLE-UPPER-BOUND"
        r10_decisions = [row["overall_decision"] for row in r10_rows.values()]
        r10_summary = {
            "schema_version": "worldsim_v61.me3_arm_summary.v1",
            "arm": "B1-R10",
            "denominator": len(r10_decisions),
            "accept_count": r10_decisions.count("ACCEPT"),
            "abstain_count": r10_decisions.count("ABSTAIN"),
            "reject_count": r10_decisions.count("REJECT"),
            "false_safe_count": 0,
            "accepted_case_ids": [
                case_id
                for case_id, row in r10_rows.items()
                if row["overall_decision"] == "ACCEPT"
            ],
        }
        arm_summaries = [r10_summary, oracle, predicted]
        _write_jsonl(run_dir / "ARM_SUMMARIES.jsonl", arm_summaries)

        oracle_yield = float(oracle["accepted_mask_area_yield"])
        all_unknown_preserved = all(int(row["unknown_voxel_count"]) > 0 for row in adapter_rows)
        no_geometry_fill = all(
            not row["geometry_created_by_identity_binding"] for row in adapter_rows
        )
        worker_peak_sum = sum(float(report["peak_gpu_memory_gib"]) for report in reports)
        raycast_peak = torch.cuda.max_memory_allocated() / 1024**3
        peak_upper_bound = max(worker_peak_sum, raycast_peak)
        elapsed = time.monotonic() - started
        checks = {
            "p7r_irwm_capability_authority_passed": True,
            "case_denominator_exact": len(final_rows)
            == int(config["cohort"]["expected_case_count"]),
            "worker_source_environment_exact": workers_exact,
            "scene_workers_ran_in_parallel": len(processes)
            == int(config["resources"]["parallel_scene_workers"]),
            "two_history_plus_current_windows_exact": all(
                [
                    {
                        "frames": row["input_frames"],
                        "metadata_indices": row["metadata_indices"],
                    }
                    for row in report["target_rows"]
                ]
                == targets
                for report in reports
            ),
            "four_target_outputs_complete": sum(len(report["outputs"]) for report in reports)
            == 4,
            "unknown_preserved": all_unknown_preserved,
            "native_identity_did_not_create_geometry": no_geometry_fill,
            "predicted_free_not_promoted_to_observed_truth": True,
            "method_decisions_frozen_before_eval": all(
                row["decision_artifact_frozen_before_eval_sha256"] == method_sha256
                for row in final_rows
            ),
            "oracle_upper_bound_exact": oracle["accept_count"] == 10
            and oracle["false_safe_count"] == 0,
            "predicted_minimum_eight_accepts": predicted["accept_count"]
            >= int(config["primary_gate"]["minimum_accepted_cases"]),
            "predicted_zero_false_safe": predicted["false_safe_count"]
            <= int(config["primary_gate"]["maximum_false_safe_count"]),
            "predicted_safely_exceeds_v6": predicted["accept_count"]
            > int(config["primary_gate"]["must_exceed_accept_count"]),
            "predicted_retains_eighty_percent_oracle_mask_yield": predicted[
                "accepted_mask_area_yield"
            ]
            >= float(config["primary_gate"]["minimum_oracle_yield_fraction"]) * oracle_yield,
            "no_training_confirmation_or_threshold_selection": True,
            "wall_within_budget": elapsed
            <= float(config["resources"]["maximum_wall_seconds"]),
            "gpu_memory_upper_bound_within_budget": peak_upper_bound
            <= float(config["resources"]["maximum_gpu_memory_gib"]),
            "disk_free_within_budget": free_gib
            >= float(config["resources"]["minimum_disk_free_gib"]),
        }
        gate = {
            "schema_version": "worldsim_v61.me3r_irwm_gate.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "primary_arm": "P1-IRWM-PREDICTED",
            "checks": checks,
            "passed": all(checks.values()),
        }
        _write_json(run_dir / "ME3_GATE.json", gate)
        metrics = {
            "schema_version": "worldsim_v61.me3r_irwm_metrics.v1",
            "task_id": TASK_ID,
            "arm_summaries": arm_summaries,
            "oracle_mask_area_yield": oracle_yield,
            "predicted_oracle_yield_fraction": predicted["accepted_mask_area_yield"]
            / oracle_yield,
            "predicted_improvement_cases_over_r10": predicted["accept_count"] - 3,
        }
        _write_json(run_dir / "METRICS.json", metrics)
        resource = {
            "schema_version": "worldsim_v61.me3r_irwm_resource_audit.v1",
            "wall_seconds": elapsed,
            "gpu_name": reports[0]["gpu_name"],
            "parallel_worker_count": len(processes),
            "worker_peak_gpu_memory_gib_sum_upper_bound": worker_peak_sum,
            "raycast_peak_gpu_memory_gib": raycast_peak,
            "overall_peak_gpu_memory_gib_upper_bound": peak_upper_bound,
            "maximum_cpu_workers": workers,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "RESOURCE_AUDIT.json", resource)
        status = "done" if gate["passed"] else "rejected"
        summary = {
            "schema_version": "worldsim_v61.me3r_irwm_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_commit": source_commit,
            "status": status,
            "hypothesis_outcome": "accepted_irwm_predicted_occupancy_development"
            if gate["passed"]
            else "rejected_irwm_predicted_occupancy_development",
            "primary_accept_count": predicted["accept_count"],
            "primary_false_safe_count": predicted["false_safe_count"],
            "primary_accepted_mask_area_yield": predicted["accepted_mask_area_yield"],
            "oracle_accept_count": oracle["accept_count"],
            "r10_accept_count": r10_summary["accept_count"],
            "next": "WS-V61-ME4-MULTIACTOR-RUNTIME-01"
            if gate["passed"]
            else "close_v61_minimum_experiment_negative_no_more_learned_occupancy_recovery",
            "failure_ledger_delta": "none" if gate["passed"] else "required",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        artifact_names = (
            "ADAPTER_INDEX.jsonl",
            "METHOD_DECISIONS.jsonl",
            "PER_CASE.jsonl",
            "PROJECTION_INDEX.jsonl",
            "ARM_SUMMARIES.jsonl",
            "ME3_GATE.json",
            "METRICS.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        )
        manifest = {
            "schema_version": "worldsim_v61.me3r_irwm_manifest.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": _source_sha256(config_path),
            "official_git_commit": config["model"]["official_git_commit"],
            "official_files": official_records,
            "model_files": model_records,
            "temporal_metadata": metadata_record,
            "source_artifacts": {
                "p7r_run": str(p7r_root),
                "me0_run": str(me0_root),
                "me1_run": str(me1_root),
                "r9_run": str(r9_root),
                "r10_run": str(r10_root),
            },
            "artifacts": {
                name: sha256_file(run_dir / name) for name in artifact_names
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        terminal = {
            "schema_version": "worldsim_v61.me3r_irwm_terminal.v1",
            "task_id": TASK_ID,
            "status": status,
            "canonical": True,
            "run_uri": f"run://worldsim_v61/{TASK_ID}/{run_dir.name}",
        }
        _write_json(run_dir / "TERMINAL.json", terminal)
        return run_dir
    except Exception as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.me3r_irwm_terminal.v1",
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

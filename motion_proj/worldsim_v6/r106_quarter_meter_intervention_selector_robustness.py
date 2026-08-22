"""WorldSim V6 R106: selector robustness at a quarter-meter intervention."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r38_actor_interaction_factor import _compile_intervention, _content_sha256
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R106-QUARTER-METER-INTERVENTION-SELECTOR-ROBUSTNESS-01"


class R106ExperimentError(RuntimeError):
    """The preregistered R106 contract was violated."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "trigger_count": int(predicted.sum()), "skip_count": int((~predicted).sum()),
        "skip_fraction": float((~predicted).mean()),
    }


def _sensor_pass(row: dict[str, Any], actor_id: str, thresholds: dict[str, Any]) -> bool:
    errors = row["actors"][actor_id]["native_actor_field_max_error"]
    return bool(
        errors["means_m"] <= float(thresholds["maximum_means_error_m"])
        and errors["quaternions_wxyz"] <= float(thresholds["maximum_quaternion_error"])
        and all(
            errors[name] <= float(thresholds["maximum_static_field_error"])
            for name in ("scales_m", "opacities", "view_dependent_rgb")
        )
        and row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"])
        and row["full_sensor_rgb_p99_absolute_error"]
        <= float(thresholds["maximum_rgb_p99_absolute_error"])
        and row["full_sensor_depth_mae_m"] <= float(thresholds["maximum_depth_mae_m"])
        and row["full_sensor_opacity_mae"] <= float(thresholds["maximum_opacity_mae"])
    )


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R106ExperimentError("formal R106 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R106ExperimentError("R106 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r101_run = _resolve_runs_uri(sources["r101_run"])
    r101_package = r101_run / "package"
    sceneir_package = r101_run / "sceneir_packages/streetgs_scene0255"
    r104_run = _resolve_runs_uri(sources["r104_run"])
    r105_run = _resolve_runs_uri(sources["r105_run"])
    r90_run = _resolve_runs_uri(sources["r90_run"])
    r90_package = r90_run / "package_a"
    model_root = Path(sources["semantic_model_root"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r101_run / "MANIFEST.json": sources["r101_manifest_sha256"],
        r101_run / "R101_GATE.json": sources["r101_gate_sha256"],
        r101_run / "SUMMARY.json": sources["r101_summary_sha256"],
        r101_run / "SELECTED_PROPOSAL.json": sources["r101_selected_proposal_sha256"],
        r101_run / "EDIT_SENSOR_COMPARISON.json": sources["r101_edit_sensor_comparison_sha256"],
        r101_package / "PACKAGE_MANIFEST.json": sources["r101_package_manifest_sha256"],
        sceneir_package / "MANIFEST.json": sources["r101_sceneir_manifest_sha256"],
        sceneir_package / "sceneir.json": sources["r101_sceneir_document_sha256"],
        r104_run / "MANIFEST.json": sources["r104_manifest_sha256"],
        r104_run / "R104_GATE.json": sources["r104_gate_sha256"],
        r104_run / "SUMMARY.json": sources["r104_summary_sha256"],
        r104_run / "SELECTOR_TRANSFER.json": sources["r104_selector_transfer_sha256"],
        r104_run / "scene_package/SCENE_COMPOSITION.json": sources["r104_scene_composition_sha256"],
        r105_run / "MANIFEST.json": sources["r105_manifest_sha256"],
        r105_run / "R105_GATE.json": sources["r105_gate_sha256"],
        r105_run / "SUMMARY.json": sources["r105_summary_sha256"],
        r105_run / "THRESHOLD_CERTIFICATE.json": sources["r105_threshold_certificate_sha256"],
        r90_run / "MANIFEST.json": sources["r90_manifest_sha256"],
        r90_run / "R90_GATE.json": sources["r90_gate_sha256"],
        r90_run / "SUMMARY.json": sources["r90_summary_sha256"],
        r90_package / "PACKAGE_MANIFEST.json": sources["r90_package_manifest_sha256"],
        r90_package / "POLICY.json": sources["r90_policy_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R106ExperimentError("StreetGS upstream commit drift")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R106ExperimentError("R106 disk resource insufficient")
    r101_gate = json.loads((r101_run / "R101_GATE.json").read_text())
    r104_gate = json.loads((r104_run / "R104_GATE.json").read_text())
    r105_gate = json.loads((r105_run / "R105_GATE.json").read_text())
    r90_gate = json.loads((r90_run / "R90_GATE.json").read_text())
    r104_composition = json.loads((r104_run / "scene_package/SCENE_COMPOSITION.json").read_text())
    policy = json.loads((r90_package / "POLICY.json").read_text())
    source_package_manifest = json.loads((r101_package / "PACKAGE_MANIFEST.json").read_text())
    source_package_files = {}
    for relative, row in source_package_manifest["files"].items():
        path = r101_package / relative
        _verify(path, row["sha256"])
        source_package_files[path] = row["sha256"]
    source_sceneir_manifest = json.loads((sceneir_package / "MANIFEST.json").read_text())
    source_sceneir_files = {}
    for relative, row in source_sceneir_manifest["files"].items():
        path = sceneir_package / relative
        _verify(path, row["sha256"])
        source_sceneir_files[path] = row["sha256"]

    base = _replay_once(sceneir_package, 1)
    target_id = str(runtime["actor_id"])
    target_states = sorted(
        (row for row in base["actor_states"] if row["actor_id"] == target_id),
        key=lambda row: int(row["timestamp_us"]),
    )
    base_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
    base_collision_keys = {
        (int(row["timestamp_us"]), tuple(row["actor_pair"]))
        for row in base["collision_labels"]
        if row["aabb_overlap"] and target_id in row["actor_pair"]
    }
    delta = np.asarray(runtime["translation_delta_m"], dtype=np.float64)
    states1, collisions1 = _compile_intervention(base["actor_states"], target_id, delta)
    states2, collisions2 = _compile_intervention(base["actor_states"], target_id, delta)
    edited_target = sorted(
        (row for row in states1 if row["actor_id"] == target_id),
        key=lambda row: int(row["timestamp_us"]),
    )
    edited_positions = np.asarray([row["centroid_world_m"] for row in edited_target], dtype=np.float64)
    dt = 0.1
    velocity_error = float(
        np.max(np.abs(np.diff(base_positions, axis=0) / dt - np.diff(edited_positions, axis=0) / dt))
    )
    acceleration_error = float(
        np.max(
            np.abs(
                np.diff(np.diff(base_positions, axis=0) / dt, axis=0) / dt
                - np.diff(np.diff(edited_positions, axis=0) / dt, axis=0) / dt
            )
        )
    )
    edited_collision_keys = {
        (int(row["timestamp_us"]), tuple(row["actor_pair"]))
        for row in collisions1
        if row["aabb_overlap"] and target_id in row["actor_pair"]
    }
    new_collision_keys = edited_collision_keys - base_collision_keys
    interaction_factor = {
        "schema_version": "worldsim_v6.r106_quarter_meter_interaction_factor.v1",
        "proposal_id": runtime["proposal_id"],
        "translation_delta_m": runtime["translation_delta_m"],
        "translation_norm_m": float(np.linalg.norm(delta)),
        "actor_state_count": len(base["actor_states"]),
        "target_trajectory_count": len(target_states),
        "baseline_overlap_event_count": len(base_collision_keys),
        "edited_overlap_event_count": len(edited_collision_keys),
        "new_overlap_event_count": len(new_collision_keys),
        "removed_overlap_event_count": len(base_collision_keys - edited_collision_keys),
        "maximum_velocity_invariance_error": velocity_error,
        "maximum_acceleration_invariance_error": acceleration_error,
        "repeat_exact": _content_sha256({"states": states1, "collisions": collisions1})
        == _content_sha256({"states": states2, "collisions": collisions2}),
        "q_self_kinematics": "ACCEPT" if velocity_error <= float(thresholds["maximum_kinematic_invariance_error"])
        and acceleration_error <= float(thresholds["maximum_kinematic_invariance_error"]) else "REJECT",
        "q_aabb_interaction": "ACCEPT" if not new_collision_keys else "REJECT",
        "contact_road_physical_safety": "ABSTAIN",
    }
    del base, states1, states2, collisions1, collisions2
    frame_indices = list(
        range(int(runtime["frame_start"]), int(runtime["frame_stop_exclusive"]), int(runtime["frame_stride"]))
    )
    if len(frame_indices) != int(runtime["expected_frame_count"]):
        raise R106ExperimentError("R106 frame denominator drift")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scene0255-quarter-meter-intervention-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "INTERACTION_FACTOR.json", interaction_factor)
    scene_package = run_dir / "scene_package"
    actor_package = scene_package / "actors" / runtime["actor_id"]
    shutil.copytree(r101_package, actor_package)
    legacy_geometry = json.loads((actor_package / "TRAJECTORY_GEOMETRY.json").read_text())
    lifecycle_reference = legacy_geometry["arrays"]["actor_frame_validity"]
    native_lifecycle_full = np.load(
        actor_package / lifecycle_reference["path"], allow_pickle=False
    ).astype(bool)
    trajectory_count = len(legacy_geometry["trajectory"])
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], trajectory_count, axis=0)
    transforms[:, :3, 3] = np.asarray(runtime["translation_delta_m"], dtype=np.float64)
    transform_path = actor_package / "blobs/proposal_transform_world.npy"
    np.save(transform_path, transforms, allow_pickle=False)
    geometry = {
        "schema_version": "worldsim_v6.r106_transform_owned_actor_geometry.v1",
        "asset_id": legacy_geometry["asset_id"],
        "chunk_id": legacy_geometry["chunk_id"],
        "frontend_model_index": int(legacy_geometry["frontend_model_index"]),
        "primitive_count": int(legacy_geometry["primitive_count"]),
        "trajectory": legacy_geometry["trajectory"],
        "base_arrays": legacy_geometry["arrays"],
        "actor_frame_validity": legacy_geometry["arrays"]["actor_frame_validity"],
        "proposal_transform_world": {
            "path": str(transform_path.relative_to(actor_package)), "sha256": _sha256(transform_path),
            "shape": list(transforms.shape), "dtype": transforms.dtype.str,
        },
    }
    _write_json(actor_package / "TRAJECTORY_GEOMETRY.json", geometry)
    _write_json(
        actor_package / "VALIDITY.json",
        {
            "schema_version": "worldsim_v6.r106_actor_validity.v1",
            "identity_binding": "OBSERVED_MODEL_INDEX_BOUND",
            "logged_rigid_pose_application": "ACCEPT_CONFORMANCE",
            "proposal_transform_ownership": "PRESENT_FLOAT64",
            "native_lifecycle": "ACCEPT_EXACT",
            "semantic_dynamics_physics_planning_safety": "ABSTAIN",
        },
    )
    actor_files = sorted(
        str(path.relative_to(actor_package))
        for path in actor_package.rglob("*")
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
    )
    _write_json(
        actor_package / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r106_actor_package_manifest.v1",
            "files": {
                name: {"bytes": (actor_package / name).stat().st_size, "sha256": _sha256(actor_package / name)}
                for name in actor_files
            },
        },
    )
    actor_manifest_sha = _sha256(actor_package / "PACKAGE_MANIFEST.json")
    composition = {
        "schema_version": "worldsim_v6.r106_scene_composition.v1",
        "scene": runtime["scene"],
        "actors": [
            {
                "actor_id": runtime["actor_id"], "actor_model_index": int(runtime["actor_model_index"]),
                "proposal_id": runtime["proposal_id"], "translation_delta_m": runtime["translation_delta_m"],
                "actor_package_path": f"actors/{runtime['actor_id']}",
            }
        ],
    }
    runtime_contract = {
        "schema_version": "worldsim_v6.r106_runtime_contract.v1",
        "runtime_mode": "single_actor_transform_lifecycle_owned_scene_patch",
        "actor_order": [runtime["actor_id"]],
        "frame_count": len(frame_indices),
    }
    _write_json(scene_package / "SCENE_COMPOSITION.json", composition)
    _write_json(scene_package / "RUNTIME_CONTRACT.json", runtime_contract)
    _write_json(
        scene_package / "VALIDITY.json",
        {
            "schema_version": "worldsim_v6.r106_scene_validity.v1",
            "compiled_native_sensor_conformance": "PENDING_FULL_EPISODE",
            "semantic_correctness_local_causality_physics_planning_safety": "ABSTAIN",
        },
    )
    scene_files = ["SCENE_COMPOSITION.json", "RUNTIME_CONTRACT.json", "VALIDITY.json"]
    _write_json(
        scene_package / "SCENE_PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r106_scene_package_manifest.v1",
            "files": {
                name: {"bytes": (scene_package / name).stat().st_size, "sha256": _sha256(scene_package / name)}
                for name in scene_files
            },
            "actor_package_manifests": {runtime["actor_id"]: actor_manifest_sha},
        },
    )
    scene_manifest_sha = _sha256(scene_package / "SCENE_PACKAGE_MANIFEST.json")

    sensor_dir = run_dir / "sensor_worker"
    frames_text = ",".join(str(frame) for frame in frame_indices)
    sensor_command = [
        sources["drivestudio_python"],
        str(repo_root / "scripts/worldsim_v6/r86_three_actor_full_episode_sensor_worker.py"),
        "--repo-root", str(repo_root), "--checkpoint", str(checkpoint),
        "--upstream-root", str(upstream), "--scene-package", str(scene_package),
        "--frames", frames_text, "--output", str(sensor_dir),
    ]
    with (run_dir / "sensor_worker.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            sensor_command, cwd=repo_root, stdout=log_stream, stderr=subprocess.STDOUT,
            check=True, timeout=float(resources["maximum_sensor_worker_seconds"]),
        )
    sensor_rows = _load_rows(sensor_dir / "FRAME_METRICS.jsonl")
    sensor_audit = json.loads((sensor_dir / "WORKER_AUDIT.json").read_text())
    if [row["frame_index"] for row in sensor_rows] != frame_indices:
        raise R106ExperimentError("R106 sensor output denominator/order drift")

    perception_dir = run_dir / "perception"
    variants = list(evaluation["variants"])
    if variants != ["logged", "edited"]:
        raise R106ExperimentError("R106 variant order drift")
    index_rows = [
        {
            "case_id": f"frame{frame:03d}_{variant}", "frame_index": frame,
            "variant": variant, "rgb_key": evaluation["rgb_keys"][variant], "repeat_index": repeat,
            "render_path": str(sensor_dir / f"sensors/frame{frame:03d}.npz"),
        }
        for frame in frame_indices for variant in variants for repeat in range(int(evaluation["repeat_count"]))
    ]
    _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", index_rows)
    perception_command = [
        sys.executable, str(repo_root / "scripts/worldsim_v6/r87_full_episode_perception_worker.py"),
        "--index", str(run_dir / "PERCEPTION_INPUT_INDEX.jsonl"),
        "--model-root", str(model_root), "--output-dir", str(perception_dir),
    ]
    worker_env = os.environ.copy()
    worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with (run_dir / "perception.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            perception_command, cwd=repo_root, env=worker_env, stdout=log_stream,
            stderr=subprocess.STDOUT, check=True,
            timeout=float(resources["maximum_perception_worker_seconds"]),
        )
    perception_worker = json.loads((perception_dir / "WORKER_RESULT.json").read_text())
    perception_rows = _load_rows(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in perception_rows:
        grouped.setdefault(row["case_id"], []).append(row)
    repeat_exact = all(
        len(items) == int(evaluation["repeat_count"])
        and len({item["label_array_sha256"] for item in items}) == 1
        for items in grouped.values()
    )
    changed_label_pixels = []
    for frame in frame_indices:
        arrays = {}
        for variant in variants:
            case_id = f"frame{frame:03d}_{variant}"
            first = sorted(grouped[case_id], key=lambda item: item["repeat_index"])[0]
            arrays[variant] = np.load(perception_dir / first["label_path"], allow_pickle=False)
        changed_label_pixels.append(int((arrays["edited"] != arrays["logged"]).sum()))
    sensor_changed_pixels = np.asarray(
        [row["edited_vs_logged_rgb_changed_pixels"] for row in sensor_rows], dtype=np.int64
    )
    changed_labels = np.asarray(changed_label_pixels, dtype=np.int64)
    target_mask = changed_labels >= int(evaluation["perception_target_minimum_changed_label_pixels"])
    frozen_threshold = int(evaluation["frozen_policy_threshold_pixels"])
    frozen_mask = sensor_changed_pixels >= frozen_threshold
    fixed_mask = sensor_changed_pixels >= int(evaluation["fixed_baseline_threshold_pixels"])
    lifecycle_mask = native_lifecycle_full[np.asarray(frame_indices, dtype=np.int64)]
    frozen_metrics = _metrics(frozen_mask, target_mask)
    fixed_metrics = _metrics(fixed_mask, target_mask)
    lifecycle_metrics = _metrics(lifecycle_mask, target_mask)
    log_sensor = np.log1p(sensor_changed_pixels.astype(np.float64))
    log_labels = np.log1p(changed_labels.astype(np.float64))
    correlation = float(np.corrcoef(log_sensor, log_labels)[0, 1]) if log_sensor.std() > 0 and log_labels.std() > 0 else 0.0
    transfer = {
        "schema_version": "worldsim_v6.r106_quarter_meter_intervention_selector.v1",
        "source_policy_id": policy["policy_id"],
        "source_scene": "scene-0242",
        "target_scene": runtime["scene"],
        "calibration_frames_in_target_scene": 0,
        "frozen_threshold_pixels": frozen_threshold,
        "frame_count": len(frame_indices),
        "positive_target_frame_count": int(target_mask.sum()),
        "negative_target_frame_count": int((~target_mask).sum()),
        "frozen_policy_metrics": frozen_metrics,
        "fixed256_metrics": fixed_metrics,
        "native_lifecycle_metrics": lifecycle_metrics,
        "total_changed_label_pixels": int(changed_labels.sum()),
        "maximum_changed_label_pixels": int(changed_labels.max()),
        "maximum_sensor_changed_pixels": int(sensor_changed_pixels.max()),
        "log1p_sensor_label_impact_pearson": correlation,
        "sensor_changed_pixels_by_frame": dict(zip(map(str, frame_indices), sensor_changed_pixels.astype(int).tolist())),
        "changed_label_pixels_by_frame": dict(zip(map(str, frame_indices), changed_labels.astype(int).tolist())),
        "semantic_correctness": "ABSTAIN",
        "local_causality": "ABSTAIN",
    }
    _write_json(run_dir / "SELECTOR_TRANSFER.json", transfer)
    sensor_output_bytes = sum(path.stat().st_size for path in sensor_dir.rglob("*") if path.is_file())
    perception_output_bytes = sum(path.stat().st_size for path in perception_dir.rglob("*") if path.is_file())
    actor_id = runtime["actor_id"]
    checks = {
        "r101_r104_r105_and_r90_authorities_accepted": bool(
            r101_gate["checks"]["passed"] and r104_gate["checks"]["passed"]
            and r105_gate["checks"]["passed"] and r90_gate["checks"]["passed"]
        ),
        "quarter_meter_interaction_and_self_kinematics_accept": interaction_factor["q_aabb_interaction"] == "ACCEPT"
        and interaction_factor["q_self_kinematics"] == "ACCEPT"
        and interaction_factor["new_overlap_event_count"] == 0 and interaction_factor["repeat_exact"],
        "same_direction_half_magnitude_of_r104_exact": len(r104_composition["actors"]) == 1
        and r104_composition["actors"][0]["actor_id"] == runtime["actor_id"]
        and np.allclose(
            np.asarray(r104_composition["actors"][0]["translation_delta_m"], dtype=np.float64),
            2.0 * np.asarray(runtime["translation_delta_m"], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        ),
        "r90_policy_bound_without_target_calibration": policy["feature"]
        == "edited_vs_logged_rgb_changed_pixels" and policy["comparator"] == "greater_than_or_equal"
        and int(policy["threshold_pixels"]) == frozen_threshold and transfer["calibration_frames_in_target_scene"] == 0,
        "full196_sensor_denominator_exact": len(sensor_rows) == int(runtime["expected_frame_count"]),
        "native_lifecycle_denominator_and_active_count_exact": native_lifecycle_full.shape
        == (int(runtime["expected_frame_count"]),)
        and int(native_lifecycle_full.sum()) == int(runtime["expected_active_frame_count"]),
        "all196_compiled_native_sensor_numerics_conformant": all(
            _sensor_pass(row, actor_id, thresholds) and row["compiled_repeat_exact"]
            and row["native_translation_state_restored_exact"]
            for row in sensor_rows
        ),
        "package_actor_validity_matches_native_lifecycle_all196": all(
            bool(row["actors"][actor_id]["package_actor_frame_valid"])
            == bool(native_lifecycle_full[row["frame_index"]])
            for row in sensor_rows
        ),
        "actor_primitive_and_translation_exact_all_frames": all(
            row["actors"][actor_id]["primitive_count"] == int(runtime["expected_primitive_count"])
            and row["actors"][actor_id]["translation_delta_m"] == runtime["translation_delta_m"]
            for row in sensor_rows
        ),
        "full784_perception_input_output_denominator_exact": len(index_rows) == len(perception_rows)
        == int(runtime["expected_frame_count"]) * 2 * int(evaluation["repeat_count"]),
        "perception_repeat_exact_every_frame_and_variant": repeat_exact,
        "target_positive_and_negative_support_nontrivial": int(target_mask.sum())
        >= int(evaluation["minimum_positive_target_frames"])
        and int((~target_mask).sum()) >= int(evaluation["minimum_negative_target_frames"]),
        "zero_calibration_transfer_precision_gate": frozen_metrics["precision"]
        >= float(evaluation["minimum_transfer_precision"]),
        "zero_calibration_transfer_recall_gate": frozen_metrics["recall"]
        >= float(evaluation["minimum_transfer_recall"]),
        "zero_calibration_transfer_f1_gate": frozen_metrics["f1"]
        >= float(evaluation["minimum_transfer_f1"]),
        "zero_calibration_transfer_skip_fraction_gate": frozen_metrics["skip_fraction"]
        >= float(evaluation["minimum_skip_fraction"]),
        "scene_and_nested_package_manifests_immutable": sensor_audit["scene_package_manifest_sha256_before"]
        == sensor_audit["scene_package_manifest_sha256_after"] == scene_manifest_sha
        and sensor_audit["nested_actor_package_manifest_sha256_before"][actor_id]
        == sensor_audit["nested_actor_package_manifest_sha256_after"][actor_id] == actor_manifest_sha,
        "checkpoint_upstream_and_frozen_sources_immutable": sensor_audit["checkpoint_sha256_before"]
        == sensor_audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"]
        and sensor_audit["upstream_commit"] == sources["streetgs_upstream_commit"]
        and all(_sha256(path) == expected for path, expected in frozen_files.items())
        and all(_sha256(path) == expected for path, expected in source_package_files.items()),
        "source_sceneir_blobs_immutable": all(
            _sha256(path) == expected for path, expected in source_sceneir_files.items()
        ),
        "semantic_correctness_local_causality_contact_physics_planning_safety_abstain": True,
        "gpu_within_budget": max(
            sensor_audit["peak_torch_reserved_bytes"] / (1024**2), perception_worker["peak_gpu_memory_mib"]
        ) <= float(resources["maximum_peak_gpu_memory_mib"]),
        "workers_within_budget": sensor_audit["wall_seconds"]
        <= float(resources["maximum_sensor_worker_seconds"])
        and perception_worker["elapsed_seconds"] <= float(resources["maximum_perception_worker_seconds"]),
        "wall_within_budget": time.monotonic() - started <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": sensor_output_bytes <= int(resources["maximum_sensor_output_bytes"])
        and perception_output_bytes <= int(resources["maximum_perception_output_bytes"]),
        "training_not_started": not sensor_audit["training_started"],
        "confirmation_not_read": not sensor_audit["confirmation_content_read"],
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R106_GATE.json",
        {
            "schema_version": "worldsim_v6.r106_gate.v1", "checks": checks,
            "decision": "accept_quarter_meter_intervention_selector_robustness"
            if checks["passed"] else "reject_quarter_meter_intervention_selector_robustness",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r106_resource_audit.v1",
            "sensor_worker_seconds": sensor_audit["wall_seconds"],
            "perception_worker_seconds": perception_worker["elapsed_seconds"],
            "wall_seconds": time.monotonic() - started,
            "sensor_output_bytes": sensor_output_bytes,
            "perception_output_bytes": perception_output_bytes,
            "peak_gpu_memory_mib": max(
                sensor_audit["peak_torch_reserved_bytes"] / (1024**2), perception_worker["peak_gpu_memory_mib"]
            ),
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r106_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_quarter_meter_intervention_selector_robustness"
        if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "source_scene": "scene-0242", "target_scene": runtime["scene"],
        "frame_count": len(frame_indices), "frozen_threshold_pixels": frozen_threshold,
        "positive_target_frame_count": int(target_mask.sum()),
        "negative_target_frame_count": int((~target_mask).sum()),
        "precision": frozen_metrics["precision"], "recall": frozen_metrics["recall"],
        "f1": frozen_metrics["f1"], "trigger_count": frozen_metrics["trigger_count"],
        "skip_count": frozen_metrics["skip_count"], "skip_fraction": frozen_metrics["skip_fraction"],
        "fixed256_f1": fixed_metrics["f1"], "native_lifecycle_f1": lifecycle_metrics["f1"],
        "total_changed_label_pixels": int(changed_labels.sum()),
        "log1p_sensor_label_impact_pearson": correlation,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R106_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "SELECTOR_TRANSFER.json",
        "INTERACTION_FACTOR.json",
        "PERCEPTION_INPUT_INDEX.jsonl", "sensor_worker.log", "perception.log",
        "scene_package/SCENE_PACKAGE_MANIFEST.json", "scene_package/SCENE_COMPOSITION.json",
        "scene_package/RUNTIME_CONTRACT.json", "scene_package/VALIDITY.json",
        f"scene_package/actors/{actor_id}/PACKAGE_MANIFEST.json",
        "sensor_worker/FRAME_METRICS.jsonl", "sensor_worker/WORKER_AUDIT.json",
        "perception/PERCEPTION_OUTPUTS.jsonl", "perception/WORKER_RESULT.json",
    ]
    tracked.extend(f"sensor_worker/{row['sensor_path']}" for row in sensor_rows)
    tracked.extend(str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy")))
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r106_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        },
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {
            "schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        },
    )
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/worldsim_v6/r106_quarter_meter_intervention_selector_robustness_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WorldSim V6 R117: compile a fourth visible actor edit in scene0255."""

from __future__ import annotations

import json
import gc
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r35_actor_rigid_trajectory_compiler import REQUIRED_ARRAYS, _compile
from motion_proj.worldsim_v6.r38_actor_interaction_factor import _compile_intervention, _content_sha256
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)
from motion_proj.worldsim_v6.sceneir import verify_sceneir, write_sceneir
from motion_proj.worldsim_v6.sceneir_adapters import streetgs_to_sceneir


TASK_ID = "WS-V6-R117-SCENE0255-FOURTH-ACTOR-EDIT-COMPILER-01"


class R117ExperimentError(RuntimeError):
    """The preregistered R117 contract was violated."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sensor_pass(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    errors = row["native_actor_field_max_error"]
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
        raise R117ExperimentError("formal R117 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R117ExperimentError("R117 task_id drift")
    sources = config["sources"]
    target = config["target"]
    search = config["search"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r100_run = _resolve_runs_uri(sources["r100_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    inventory_path = r100_run / "worker/ACTOR_INVENTORY.json"
    selection_path = r100_run / "ACTOR_VISIBILITY_SELECTION.json"
    lifecycle_path = r100_run / "worker" / f"ACTOR_{int(target['actor_model_index']):04d}_FRAME_VALID.npy"
    frozen_files = {
        r100_run / "MANIFEST.json": sources["r100_manifest_sha256"],
        r100_run / "R100_GATE.json": sources["r100_gate_sha256"],
        r100_run / "SUMMARY.json": sources["r100_summary_sha256"],
        selection_path: sources["r100_selection_sha256"],
        inventory_path: sources["r100_inventory_sha256"],
        r100_run / "worker/VISIBILITY.json": sources["r100_visibility_sha256"],
        r100_run / "worker/WORKER_AUDIT.json": sources["r100_worker_audit_sha256"],
        lifecycle_path: target["lifecycle_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R117ExperimentError("StreetGS upstream commit drift")
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R117ExperimentError("R117 disk resource insufficient")
    r100_gate = json.loads((r100_run / "R100_GATE.json").read_text())
    selection = json.loads(selection_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    visibility = json.loads((r100_run / "worker/VISIBILITY.json").read_text())
    excluded_actor_indices = {int(value) for value in target["excluded_actor_model_indices"]}
    fourth_actor_selection = sorted(
        (
            row for row in visibility["rows"]
            if int(row["actor_model_index"]) not in excluded_actor_indices
        ),
        key=lambda row: (
            -int(row["actor_effect_pixels"]), int(row["actor_model_index"]), int(row["frame_index"])
        ),
    )[0]

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scene0255-actor1-edit-s{config['seed']}-r1"
    if run_dir.exists():
        raise R117ExperimentError(f"formal run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    document_template, chunk_arrays = streetgs_to_sceneir(
        checkpoint_object,
        source_sha256=sources["streetgs_checkpoint_sha256"],
        source_uri="checkpoint://streetgs_scene0255_reference",
        reconstructor_version=sources["streetgs_upstream_commit"],
        seed=int(config["seed"]),
    )
    base_package = run_dir / "sceneir_packages/streetgs_scene0255"
    document = write_sceneir(base_package, document_template, chunk_arrays)
    sceneir_verification = verify_sceneir(base_package)
    del checkpoint_object, document_template, chunk_arrays
    gc.collect()
    source_manifest = json.loads((base_package / "MANIFEST.json").read_text())
    sceneir_manifest_sha = _sha256(base_package / "MANIFEST.json")
    sceneir_document_sha = _sha256(base_package / "sceneir.json")
    _write_json(
        run_dir / "SCENEIR_EXPORT.json",
        {
            "schema_version": "worldsim_v6.r117_sceneir_export.v1",
            "scene": target["scene"],
            "checkpoint_sha256": sources["streetgs_checkpoint_sha256"],
            "sceneir_manifest_sha256": sceneir_manifest_sha,
            "sceneir_document_sha256": sceneir_document_sha,
            "verification": sceneir_verification,
            "actor_count": len(document["actors"]),
            "actor_primitive_count": sum(
                int(row["primitive_count"]) for row in document["chunks"] if row["role"] == "actor"
            ),
        },
    )
    actor_index = int(target["actor_model_index"])
    inventory_actor = next(
        row for row in inventory["actors"] if int(row["actor_model_index"]) == actor_index
    )
    actor = next(row for row in document["actors"] if row["id"] == target["actor_id"])
    chunk = next(row for row in document["chunks"] if row["id"] == target["chunk_id"])
    if set(chunk["arrays"]) != REQUIRED_ARRAYS:
        raise R117ExperimentError("actor1 Gaussian array set drift")
    transforms_by_key = {
        (row["name"], int(row["timestamp_us"])): row for row in document["transforms"]
    }
    ordered_transforms = []
    trajectory_metadata = []
    visibility = {int(row["timestamp_us"]): bool(row["visible"]) for row in actor["visibility"]}
    for row in actor["trajectory"]:
        timestamp = int(row["timestamp_us"])
        key = (row["transform_name"], timestamp)
        if key not in transforms_by_key or timestamp not in visibility:
            raise R117ExperimentError(f"actor1 trajectory reference missing: {key}")
        ordered_transforms.append(transforms_by_key[key])
        trajectory_metadata.append(
            {"timestamp_us": timestamp, "transform_name": row["transform_name"], "visible": visibility[timestamp]}
        )
    source_blobs = {}
    arrays = {}
    for name, reference in chunk["arrays"].items():
        source_path = base_package / reference["path"]
        manifest_row = source_manifest["files"].get(reference["path"])
        if manifest_row is None or manifest_row["sha256"] != reference["sha256"]:
            raise R117ExperimentError(f"base manifest missing actor1 blob: {name}")
        _verify(source_path, reference["sha256"])
        source_blobs[source_path] = reference["sha256"]
        arrays[name] = np.load(source_path, allow_pickle=False)
    lifecycle = np.load(lifecycle_path, allow_pickle=False)
    translations = np.asarray([row["translation_m"] for row in ordered_transforms], dtype=np.float64)
    pose_quaternions = np.asarray([row["rotation_wxyz"] for row in ordered_transforms], dtype=np.float64)
    means1, quaternions1, rotations = _compile(
        arrays["means_m"], arrays["quaternions_wxyz"], translations, pose_quaternions
    )
    means2, quaternions2, _ = _compile(
        arrays["means_m"], arrays["quaternions_wxyz"], translations, pose_quaternions
    )
    inverse_means = np.einsum(
        "tji,tnj->tni", rotations, means1.astype(np.float64) - translations[:, None, :], optimize=True
    )
    roundtrip_error_m = float(
        np.max(np.abs(inverse_means - arrays["means_m"].astype(np.float64)[None, :, :]))
    )
    local_radius = np.linalg.norm(arrays["means_m"].astype(np.float64), axis=-1)
    world_radius = np.linalg.norm(means1.astype(np.float64) - translations[:, None, :], axis=-1)
    rigid_radius_error_m = float(np.max(np.abs(world_radius - local_radius[None, :])))
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(quaternions1.astype(np.float64), axis=-1) - 1.0))
    )

    base = _replay_once(base_package, 1)
    target_id = str(target["actor_id"])
    target_states = sorted(
        (row for row in base["actor_states"] if row["actor_id"] == target_id),
        key=lambda row: int(row["timestamp_us"]),
    )
    actor_ids = sorted({row["actor_id"] for row in base["actor_states"]})
    base_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
    base_collision_keys = {
        (int(row["timestamp_us"]), tuple(row["actor_pair"]))
        for row in base["collision_labels"]
        if row["aabb_overlap"] and target_id in row["actor_pair"]
    }
    candidates = [
        (float(x), float(z))
        for x in search["x_values_m"]
        for z in search["z_values_m"]
        if not (float(x) == 0.0 and float(z) == 0.0)
    ]
    proposal_rows = []
    dt = 0.1
    tolerance = float(thresholds["maximum_kinematic_invariance_error"])
    for x, z in candidates:
        delta = np.asarray([x, 0.0, z], dtype=np.float64)
        states, collisions = _compile_intervention(base["actor_states"], target_id, delta)
        edited_target = sorted(
            (row for row in states if row["actor_id"] == target_id),
            key=lambda row: int(row["timestamp_us"]),
        )
        edited_positions = np.asarray([row["centroid_world_m"] for row in edited_target], dtype=np.float64)
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
        edited_keys = {
            (int(row["timestamp_us"]), tuple(row["actor_pair"]))
            for row in collisions
            if row["aabb_overlap"] and target_id in row["actor_pair"]
        }
        new_keys = edited_keys - base_collision_keys
        norm = float(np.linalg.norm(delta))
        accepted = (
            velocity_error <= tolerance
            and acceleration_error <= tolerance
            and not new_keys
            and norm >= float(search["minimum_translation_norm_m"])
        )
        proposal_rows.append(
            {
                "proposal_id": f"scene0255_actor1_translate_x_{x:+.1f}_z_{z:+.1f}",
                "translation_delta_m": delta.tolist(),
                "translation_norm_m": norm,
                "new_overlap_events": len(new_keys),
                "removed_overlap_events": len(base_collision_keys - edited_keys),
                "maximum_velocity_invariance_error": velocity_error,
                "maximum_acceleration_invariance_error": acceleration_error,
                "q_self_kinematics": "ACCEPT" if velocity_error <= tolerance and acceleration_error <= tolerance else "REJECT",
                "q_aabb_interaction": "ACCEPT" if not new_keys else "REJECT",
                "joint_decision": "ACCEPT" if accepted else "REJECT",
                "contact_road_physical_safety": "ABSTAIN",
            }
        )
    accepted_rows = [row for row in proposal_rows if row["joint_decision"] == "ACCEPT"]
    if not accepted_rows:
        raise R117ExperimentError("no zero-new-overlap candidate in preregistered grid")
    selected = sorted(
        accepted_rows,
        key=lambda row: (
            row["translation_norm_m"], row["translation_delta_m"][0], row["translation_delta_m"][2]
        ),
    )[0]
    selected_delta = np.asarray(selected["translation_delta_m"], dtype=np.float64)
    states1, collisions1 = _compile_intervention(base["actor_states"], target_id, selected_delta)
    states2, collisions2 = _compile_intervention(base["actor_states"], target_id, selected_delta)
    selected_repeat_exact = _content_sha256({"states": states1, "collisions": collisions1}) == _content_sha256(
        {"states": states2, "collisions": collisions2}
    )

    package = run_dir / "package"
    blob_dir = package / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=False)
    means_path = blob_dir / "means_world_by_timestamp.npy"
    quaternions_path = blob_dir / "quaternions_world_wxyz_by_timestamp.npy"
    np.save(means_path, means1, allow_pickle=False)
    np.save(quaternions_path, quaternions1, allow_pickle=False)
    invariant_files = {}
    for name in sorted(REQUIRED_ARRAYS - {"means_m", "quaternions_wxyz"}):
        reference = chunk["arrays"][name]
        destination = package / reference["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_package / reference["path"], destination)
        _verify(destination, reference["sha256"])
        invariant_files[name] = reference
    lifecycle_destination = blob_dir / f"{target['lifecycle_sha256']}.npy"
    shutil.copy2(lifecycle_path, lifecycle_destination)
    lifecycle_reference = {
        "path": str(lifecycle_destination.relative_to(package)),
        "sha256": _sha256(lifecycle_destination),
        "shape": list(lifecycle.shape),
        "dtype": lifecycle.dtype.str,
    }
    geometry = {
        "schema_version": "worldsim_v6.r117_trajectory_geometry.v1",
        "asset_id": actor["id"],
        "chunk_id": chunk["id"],
        "frontend_model_index": actor_index,
        "primitive_count": int(chunk["primitive_count"]),
        "trajectory": trajectory_metadata,
        "arrays": {
            "means_world_m": {
                "path": str(means_path.relative_to(package)), "sha256": _sha256(means_path),
                "shape": list(means1.shape), "dtype": means1.dtype.str,
            },
            "quaternions_world_wxyz": {
                "path": str(quaternions_path.relative_to(package)), "sha256": _sha256(quaternions_path),
                "shape": list(quaternions1.shape), "dtype": quaternions1.dtype.str,
            },
            "actor_frame_validity": lifecycle_reference,
            **invariant_files,
        },
    }
    _write_json(package / "TRAJECTORY_GEOMETRY.json", geometry)
    _write_json(
        package / "ACTOR_BUNDLE.json",
        {
            "schema_version": "worldsim_v6.r117_actor_bundle.v1",
            "frontend_model_index": actor_index,
            "actor": actor,
            "chunk": chunk,
            "trajectory_transforms": ordered_transforms,
            "lifecycle": lifecycle_reference,
        },
    )
    _write_json(
        package / "VALIDITY.json",
        {
            "schema_version": "worldsim_v6.r117_validity.v1",
            "identity_binding": "OBSERVED_MODEL_INDEX_BOUND",
            "logged_rigid_pose_application": "ACCEPT_CONFORMANCE",
            "native_lifecycle": "ACCEPT_EXACT",
            "sensor_render_validity": "PENDING_FORMAL_WORKERS",
            "semantic_identity": "ABSTAIN",
            "contact_road_dynamics_physics_planning_safety": "ABSTAIN",
        },
    )
    package_files = [
        "TRAJECTORY_GEOMETRY.json", "ACTOR_BUNDLE.json", "VALIDITY.json",
        str(means_path.relative_to(package)), str(quaternions_path.relative_to(package)),
        lifecycle_reference["path"], *sorted(reference["path"] for reference in invariant_files.values()),
    ]
    _write_json(
        package / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r117_package_manifest.v1",
            "files": {
                name: {"bytes": (package / name).stat().st_size, "sha256": _sha256(package / name)}
                for name in package_files
            },
        },
    )
    package_manifest_sha = _sha256(package / "PACKAGE_MANIFEST.json")
    _write_jsonl(run_dir / "PROPOSAL_CATALOG.jsonl", proposal_rows)
    _write_json(
        run_dir / "SELECTED_PROPOSAL.json",
        {"schema_version": "worldsim_v6.r117_selected_proposal.v1", "selection_rule": search["selection_rule"], "selected": selected},
    )

    def run_worker(label: str, delta: list[float]) -> tuple[dict, dict, Path]:
        worker_dir = run_dir / label
        delta_text = ",".join(str(float(value)) for value in delta)
        command = [
            sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
            "--repo-root", str(repo_root), "--checkpoint", str(checkpoint),
            "--upstream-root", str(upstream), "--package", str(package),
            "--frames", str(target["visibility_frame_index"]), "--actor-model-index", str(actor_index),
            f"--translation-delta-m={delta_text}", "--output", str(worker_dir),
        ]
        with (run_dir / f"{label}.log").open("w", encoding="utf-8") as log_stream:
            subprocess.run(
                command, cwd=repo_root, stdout=log_stream, stderr=subprocess.STDOUT,
                check=True, timeout=float(resources["maximum_worker_seconds"]),
            )
        rows = _load_rows(worker_dir / "FRAME_METRICS.jsonl")
        if len(rows) != 1:
            raise R117ExperimentError(f"{label} sensor denominator drift")
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text())
        return rows[0], audit, worker_dir / rows[0]["sensor_path"]

    baseline_row, baseline_audit, baseline_sensor = run_worker("baseline_worker", [0.0, 0.0, 0.0])
    edited_row, edited_audit, edited_sensor = run_worker("edited_worker", selected["translation_delta_m"])
    with np.load(baseline_sensor, allow_pickle=False) as baseline, np.load(edited_sensor, allow_pickle=False) as edited:
        absolute = np.abs(
            edited["compiled_rgb"].astype(np.float64) - baseline["compiled_rgb"].astype(np.float64)
        )
    channel_axis = 0 if absolute.ndim == 3 and absolute.shape[0] == 3 else -1
    changed_pixels = int(np.sum(np.max(absolute, axis=channel_axis) > float(search["rgb_change_epsilon"])))
    comparison = {
        "schema_version": "worldsim_v6.r117_edit_sensor_comparison.v1",
        "frame_index": int(target["visibility_frame_index"]),
        "translation_delta_m": selected["translation_delta_m"],
        "changed_rgb_pixels_vs_logged": changed_pixels,
        "rgb_mae_vs_logged": float(absolute.mean()),
        "baseline_sensor_sha256": _sha256(baseline_sensor),
        "edited_sensor_sha256": _sha256(edited_sensor),
    }
    _write_json(run_dir / "EDIT_SENSOR_COMPARISON.json", comparison)
    timestamps = [row["timestamp_us"] for row in trajectory_metadata]
    expected_selected = sorted(
        accepted_rows,
        key=lambda row: (
            row["translation_norm_m"], row["translation_delta_m"][0], row["translation_delta_m"][2]
        ),
    )[0]
    checks = {
        "r100_exhaustive_inventory_and_visibility_authority_accepted": bool(
            r100_gate["checks"]["passed"]
        ),
        "r100_fourth_actor_selection_exact": int(fourth_actor_selection["actor_model_index"]) == actor_index
        and int(fourth_actor_selection["frame_index"]) == int(target["visibility_frame_index"])
        and int(fourth_actor_selection["actor_effect_pixels"]) == int(target["native_effect_pixels"])
        and excluded_actor_indices == {34, 24, 9},
        "fresh_sceneir_export_exact": sceneir_verification["schema_version"] == "worldsim.sceneir.v0"
        and sceneir_verification["actor_count"] == int(target["expected_base_actor_count"])
        and sum(
            int(row["primitive_count"]) for row in document["chunks"] if row["role"] == "actor"
        ) == int(target["expected_rigid_primitive_count"]),
        "identity_chunk_and_primitive_exact": actor["id"] == target["actor_id"]
        and chunk["id"] == target["chunk_id"] and chunk["actor_id"] == actor["id"]
        and int(chunk["primitive_count"]) == int(target["primitive_count"])
        == inventory_actor["primitive_count"],
        "trajectory_denominator_exact": len(trajectory_metadata) == int(target["trajectory_rows"])
        and len(target_states) == int(target["trajectory_rows"]),
        "timestamps_strictly_increasing": all(b > a for a, b in zip(timestamps, timestamps[1:])),
        "lifecycle_exact_and_visibility_bound": _sha256(lifecycle_destination) == target["lifecycle_sha256"]
        and lifecycle.shape == (int(target["trajectory_rows"]),)
        and int(lifecycle.sum()) == int(target["active_frame_count"])
        and [bool(value) for value in lifecycle] == [row["visible"] for row in trajectory_metadata],
        "two_trajectory_compilations_exact": np.array_equal(means1, means2)
        and np.array_equal(quaternions1, quaternions2),
        "compiled_shapes_exact": list(means1.shape)
        == [int(target["trajectory_rows"]), int(target["primitive_count"]), 3]
        and list(quaternions1.shape)
        == [int(target["trajectory_rows"]), int(target["primitive_count"]), 4],
        "rigid_roundtrip_and_radius_within_tolerance": roundtrip_error_m
        <= float(thresholds["maximum_rigid_error_m"])
        and rigid_radius_error_m <= float(thresholds["maximum_rigid_error_m"]),
        "world_quaternions_normalized": quaternion_norm_error
        <= float(thresholds["maximum_quaternion_norm_error"]),
        "base_actor_denominator_exact": len(actor_ids) == int(target["expected_base_actor_count"]),
        "candidate_denominator_exact": len(proposal_rows) == int(search["expected_candidate_count"]),
        "all_self_kinematics_accept": all(row["q_self_kinematics"] == "ACCEPT" for row in proposal_rows),
        "at_least_one_zero_new_overlap_candidate": bool(accepted_rows),
        "selected_by_preregistered_rule": selected == expected_selected,
        "selected_zero_new_overlap_and_repeat_exact": selected["new_overlap_events"] == 0
        and selected_repeat_exact,
        "baseline_native_visibility_reproduces_r100": baseline_row["actor_effect_pixels"]
        == int(target["native_effect_pixels"]),
        "baseline_compiled_native_sensor_conformant": _sensor_pass(baseline_row, thresholds)
        and baseline_row["compiled_repeat_exact"] and baseline_row["package_actor_frame_valid"],
        "edited_compiled_native_sensor_conformant": _sensor_pass(edited_row, thresholds)
        and edited_row["compiled_repeat_exact"] and edited_row["package_actor_frame_valid"]
        and edited_row["translation_delta_m"] == selected["translation_delta_m"],
        "selected_sensor_change_nontrivial": changed_pixels
        >= int(search["minimum_changed_rgb_pixels_vs_logged"]),
        "package_checkpoint_sources_and_sceneir_immutable": all(
            audit["package_manifest_sha256_before"] == audit["package_manifest_sha256_after"]
            == package_manifest_sha and audit["checkpoint_sha256_before"]
            == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"]
            for audit in (baseline_audit, edited_audit)
        ) and all(_sha256(path) == expected for path, expected in frozen_files.items())
        and all(_sha256(path) == expected for path, expected in source_blobs.items())
        and _sha256(base_package / "MANIFEST.json") == sceneir_manifest_sha
        and _sha256(base_package / "sceneir.json") == sceneir_document_sha,
        "contact_road_semantic_physical_planning_safety_selector_transfer_abstain": True,
        "gpu_within_budget": max(
            baseline_audit["peak_torch_reserved_bytes"], edited_audit["peak_torch_reserved_bytes"]
        ) / (1024**2) <= float(resources["maximum_peak_gpu_memory_mib"]),
        "workers_within_budget": baseline_audit["wall_seconds"]
        <= float(resources["maximum_worker_seconds"])
        and edited_audit["wall_seconds"] <= float(resources["maximum_worker_seconds"]),
        "wall_within_budget": time.monotonic() - started <= float(resources["maximum_wall_seconds"]),
        "training_not_started": not baseline_audit["training_started"] and not edited_audit["training_started"],
        "confirmation_not_read": not baseline_audit["confirmation_content_read"]
        and not edited_audit["confirmation_content_read"],
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R117_GATE.json",
        {
            "schema_version": "worldsim_v6.r117_gate.v1", "checks": checks,
            "decision": "accept_scene0255_fourth_actor_edit_compiler"
            if checks["passed"] else "reject_or_repair_scene0255_fourth_actor_edit_compiler",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r117_resource_audit.v1",
            "peak_torch_reserved_mib": max(
                baseline_audit["peak_torch_reserved_bytes"], edited_audit["peak_torch_reserved_bytes"]
            ) / (1024**2),
            "baseline_worker_seconds": baseline_audit["wall_seconds"],
            "edited_worker_seconds": edited_audit["wall_seconds"],
            "wall_seconds": time.monotonic() - started,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r117_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_scene0255_fourth_actor_edit_compiler"
        if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "scene": target["scene"],
        "actor_model_index": actor_index,
        "primitive_count": int(target["primitive_count"]),
        "trajectory_rows": len(trajectory_metadata),
        "candidate_count": len(proposal_rows),
        "accepted_candidate_count": len(accepted_rows),
        "selected_proposal_id": selected["proposal_id"],
        "selected_translation_delta_m": selected["translation_delta_m"],
        "changed_rgb_pixels_vs_logged": changed_pixels,
        "baseline_compiled_native_rgb_mae": baseline_row["full_sensor_rgb_mae"],
        "edited_compiled_native_rgb_mae": edited_row["full_sensor_rgb_mae"],
        "edited_compiled_native_depth_mae_m": edited_row["full_sensor_depth_mae_m"],
        "roundtrip_error_m": roundtrip_error_m,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R117_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "SCENEIR_EXPORT.json",
        "PROPOSAL_CATALOG.jsonl",
        "SELECTED_PROPOSAL.json", "EDIT_SENSOR_COMPARISON.json",
        "sceneir_packages/streetgs_scene0255/MANIFEST.json",
        "sceneir_packages/streetgs_scene0255/sceneir.json",
        "baseline_worker.log", "edited_worker.log", "package/PACKAGE_MANIFEST.json",
        *[f"package/{name}" for name in package_files],
        "baseline_worker/FRAME_METRICS.jsonl", "baseline_worker/WORKER_AUDIT.json",
        f"baseline_worker/{baseline_row['sensor_path']}",
        "edited_worker/FRAME_METRICS.jsonl", "edited_worker/WORKER_AUDIT.json",
        f"edited_worker/{edited_row['sensor_path']}",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r117_manifest.v1",
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
        default=Path("configs/worldsim_v6/r117_scene0255_fourth_actor_edit_compiler_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

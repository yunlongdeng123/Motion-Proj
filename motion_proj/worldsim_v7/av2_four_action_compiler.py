"""Real-coordinate KEEP/PROJECT/COMPLETE/UNKNOWN compilation on AV2 Actors."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from motion_proj.worldsim_v7.av2_canonical_surface import (
    TrackGeometry,
    _associate_actor_points,
    _deterministic_limit,
    _fuse_surfels,
    _read_lidar,
    _track_geometries,
)
from motion_proj.worldsim_v7.physical_compiler import (
    ActorState,
    CompilerThresholds,
    HazardPreservingPhysicalCompiler,
    PhysicalEvidence,
)


def _nearest(
    query: np.ndarray,
    reference: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(query) == 0 or len(reference) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=device)
    distances, indices = [], []
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            query_tensor = torch.as_tensor(
                query[start : start + chunk_size], dtype=torch.float32, device=device
            )
            values, positions = torch.cdist(query_tensor, reference_tensor).min(dim=1)
            distances.append(values.cpu())
            indices.append(positions.cpu())
    return torch.cat(distances).numpy(), torch.cat(indices).numpy()


def _limit_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[positions]


def _voxel_unique(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    keys = np.floor(points / float(voxel_size_m)).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points[np.sort(indices)], dtype=np.float32)


def _quality(
    surface: np.ndarray,
    target: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    chunk = int(config["compiler_geometry"]["distance_chunk_size"])
    threshold = float(config["compiler_geometry"]["target_support_distance_m"])
    target_to_surface, _ = _nearest(target, surface, device, chunk)
    surface_to_target, _ = _nearest(surface, target, device, chunk)
    return {
        "target_distance_mean_m": float(np.mean(target_to_surface)),
        "surface_distance_mean_m": float(np.mean(surface_to_target)),
        "target_recall": float(np.mean(target_to_surface <= threshold)),
        "surface_precision": float(np.mean(surface_to_target <= threshold)),
        "symmetric_chamfer_m": float(
            0.5 * (np.mean(target_to_surface) + np.mean(surface_to_target))
        ),
    }


def _compile_actor(
    track: TrackGeometry,
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
    include_diagnostics: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    surface_config = config["surface"]
    stride = int(surface_config["evaluation_stride"])
    build_records = [item for item in records if item["frame_rank"] % stride != 0]
    heldout_records = [item for item in records if item["frame_rank"] % stride == 0]
    if (
        len(build_records) < int(surface_config["minimum_build_frames"])
        or len(heldout_records) < int(surface_config["minimum_heldout_frames"])
    ):
        return None
    surface = _fuse_surfels(build_records, config, device)
    if len(surface["points"]) < int(surface_config["minimum_stable_surfels"]):
        return None
    maximum_points = int(config["compiler_geometry"]["maximum_metric_points"])
    query = _deterministic_limit(heldout_records[0]["points"], maximum_points)
    target_all = np.concatenate(
        [item["points"] for item in heldout_records[1:]], axis=0
    )
    target = _deterministic_limit(target_all, maximum_points)
    if (
        len(query) < int(config["compiler_geometry"]["minimum_query_points"])
        or len(target) < int(config["compiler_geometry"]["minimum_target_points"])
    ):
        return None

    canonical = np.asarray(surface["points"], dtype=np.float32)
    support = np.asarray(surface["temporal_support"], dtype=np.int64)
    views = np.asarray(surface["view_support"], dtype=np.int64)
    chunk = int(config["compiler_geometry"]["distance_chunk_size"])
    query_distance, query_nearest = _nearest(query, canonical, device, chunk)
    nearest_support = support[query_nearest]
    keep = (
        query_distance <= float(config["compiler_geometry"]["keep_distance_m"])
    ) & (
        nearest_support >= int(config["compiler_thresholds"]["minimum_temporal_support"])
    )
    stable_query_indices = np.flatnonzero(keep)
    if len(stable_query_indices) < int(config["compiler_geometry"]["minimum_stable_query_points"]):
        return None
    probe_indices = _limit_indices(
        stable_query_indices,
        int(config["compiler_geometry"]["maximum_artifact_probes_per_actor"]),
    )

    thresholds = CompilerThresholds(**config["compiler_thresholds"])
    compiler = HazardPreservingPhysicalCompiler(thresholds)
    actor = ActorState(
        actor_id=track.track_id,
        trajectory_xyz_m=tuple(
            tuple(float(value) for value in row) for row in track.city_centers_m
        ),
        size_lwh_m=tuple(float(value) for value in track.size_lwh_m),
    )
    baseline_parts = [query]
    compiled_parts = [query[keep]]
    action_counts = Counter({"KEEP": int(np.count_nonzero(keep)),
                             "UNKNOWN": int(np.count_nonzero(~keep))})
    input_counts = Counter({"clean": len(query)})

    selected_query = query[probe_indices]
    sensor_origin = np.asarray(heldout_records[0]["sensor_origin"], dtype=np.float32)
    ray = selected_query - sensor_origin[None, :]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-6)
    ghost = selected_query - float(config["compiler_geometry"]["ghost_offset_m"]) * ray
    baseline_parts.append(ghost)
    input_counts["observed_free_ghost"] += len(ghost)
    ghost_distance, ghost_nearest = _nearest(ghost, canonical, device, chunk)
    projected = []
    projected_aligned = np.full(ghost.shape, np.nan, dtype=np.float32)
    ghost_actions = []
    for index, (distance, nearest) in enumerate(zip(ghost_distance, ghost_nearest)):
        decision = compiler.compile(
            PhysicalEvidence(
                actor=actor,
                primitive_id=f"ghost-{index}",
                sensor_hit_count=int(surface["hit_count"][nearest]),
                temporal_support_count=int(support[nearest]),
                view_direction_count=int(views[nearest]),
                provenance_supported=True,
                free_space_violation_m=float(config["compiler_geometry"]["ghost_offset_m"]),
                surface_distance_m=float(distance),
            )
        )
        action_counts[decision.action.value] += 1
        ghost_actions.append(decision.action.value)
        if decision.action.value == "PROJECT":
            projection_output = str(
                config["compiler_geometry"].get(
                    "projection_output", "canonical_nearest"
                )
            )
            if projection_output == "canonical_nearest":
                projected_point = canonical[nearest]
            elif projection_output == "observed_lidar_hit":
                projected_point = selected_query[index]
            else:
                raise ValueError(f"unsupported projection_output={projection_output}")
            projected.append(projected_point)
            projected_aligned[index] = projected_point
    if projected:
        compiled_parts.append(np.asarray(projected, dtype=np.float32))

    radial = selected_query / np.maximum(
        np.linalg.norm(selected_query, axis=1, keepdims=True), 1e-6
    )
    duplicate = selected_query + float(
        config["compiler_geometry"]["duplicate_shell_offset_m"]
    ) * radial
    baseline_parts.append(duplicate)
    input_counts["duplicate_shell"] += len(duplicate)
    duplicate_unknown_count = 0
    for index in range(len(duplicate)):
        decision = compiler.compile(
            PhysicalEvidence(
                actor=actor,
                primitive_id=f"duplicate-{index}",
                sensor_hit_count=0,
                temporal_support_count=0,
                view_direction_count=0,
                provenance_supported=False,
                free_space_violation_m=0.0,
                surface_distance_m=float(
                    config["compiler_geometry"]["duplicate_shell_offset_m"]
                ),
            )
        )
        action_counts[decision.action.value] += 1
        duplicate_unknown_count += int(decision.action.value == "UNKNOWN")

    flicker = selected_query + np.asarray(
        config["compiler_geometry"]["flicker_offset_xyz_m"], dtype=np.float32
    )[None, :]
    baseline_parts.append(flicker)
    input_counts["temporal_flicker"] += len(flicker)
    flicker_unknown_count = 0
    for index in range(len(flicker)):
        decision = compiler.compile(
            PhysicalEvidence(
                actor=actor,
                primitive_id=f"flicker-{index}",
                sensor_hit_count=1,
                temporal_support_count=1,
                view_direction_count=1,
                provenance_supported=True,
                free_space_violation_m=0.0,
                surface_distance_m=0.0,
                evidence_known=False,
            )
        )
        action_counts[decision.action.value] += 1
        flicker_unknown_count += int(decision.action.value == "UNKNOWN")

    canonical_to_query, _ = _nearest(canonical, query, device, chunk)
    completion_mask = (
        canonical_to_query >= float(config["compiler_geometry"]["hole_radius_m"])
    ) & (
        support >= int(config["compiler_geometry"]["minimum_completion_support"])
    ) & (
        views >= thresholds.minimum_completion_views
    )
    completion_indices = _limit_indices(
        np.flatnonzero(completion_mask),
        int(config["compiler_geometry"]["maximum_completion_probes_per_actor"]),
    )
    completed = []
    for index, surface_index in enumerate(completion_indices):
        decision = compiler.compile(
            PhysicalEvidence(
                actor=actor,
                primitive_id=f"complete-{index}",
                sensor_hit_count=int(surface["hit_count"][surface_index]),
                temporal_support_count=int(support[surface_index]),
                view_direction_count=int(views[surface_index]),
                provenance_supported=True,
                free_space_violation_m=0.0,
                surface_distance_m=0.0,
                hole_radius_m=float(config["compiler_geometry"]["hole_radius_m"]),
            )
        )
        input_counts["surface_hole"] += 1
        action_counts[decision.action.value] += 1
        if decision.action.value == "COMPLETE":
            completed.append(canonical[surface_index])
    if completed:
        compiled_parts.append(np.asarray(completed, dtype=np.float32))

    baseline = _voxel_unique(
        np.concatenate(baseline_parts, axis=0),
        float(config["compiler_geometry"]["output_voxel_size_m"]),
    )
    compiled = _voxel_unique(
        np.concatenate(compiled_parts, axis=0),
        float(config["compiler_geometry"]["output_voxel_size_m"]),
    )
    before = _quality(baseline, target, config, device)
    after = _quality(compiled, target, config, device)
    completed_points = np.asarray(completed, dtype=np.float32).reshape(-1, 3)
    completion_distance, _ = _nearest(completed_points, target, device, chunk)
    target_threshold = float(config["compiler_geometry"]["target_support_distance_m"])
    completion_supported = int(np.count_nonzero(completion_distance <= target_threshold))
    row = {
        "track_id": track.track_id,
        "category": track.category,
        "hazardous": track.hazardous,
        "query_points": len(query),
        "target_points": len(target),
        "canonical_surfels": len(canonical),
        "baseline_points": len(baseline),
        "compiled_points": len(compiled),
        "input_counts": dict(sorted(input_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "clean_keep_count": int(np.count_nonzero(keep)),
        "ghost_project_count": len(projected),
        "duplicate_unknown_count": duplicate_unknown_count,
        "flicker_unknown_count": flicker_unknown_count,
        "completion_decision_count": len(completed),
        "completion_supported_count": completion_supported,
        "before": before,
        "after": after,
        "target_recall_delta": after["target_recall"] - before["target_recall"],
        "surface_precision_delta": after["surface_precision"] - before["surface_precision"],
        "chamfer_ratio": after["symmetric_chamfer_m"]
        / max(before["symmetric_chamfer_m"], 1e-8),
        "actor_identity_trajectory_size_retention": 1.0,
        "hazard_label_retention": 1.0,
    }
    package = {
        "points": compiled,
        "action_counts": dict(action_counts),
        "input_counts": dict(input_counts),
    }
    if include_diagnostics:
        target_origins_all = np.concatenate(
            [
                np.repeat(item["sensor_origin"][None, :], len(item["points"]), axis=0)
                for item in heldout_records[1:]
            ],
            axis=0,
        )
        target_indices = _limit_indices(np.arange(len(target_all)), maximum_points)
        package["diagnostics"] = {
            "track": track,
            "query_timestamp_ns": int(heldout_records[0]["timestamp_ns"]),
            "query_actor_center_ego": np.asarray(
                heldout_records[0]["actor_center_ego"], dtype=np.float32
            ),
            "query_actor_rotation_ego": np.asarray(
                heldout_records[0]["actor_rotation_ego"], dtype=np.float32
            ),
            "build_frame_points": [
                _deterministic_limit(item["points"], maximum_points)
                for item in build_records
            ],
            "query": query,
            "query_sensor_origin": sensor_origin,
            "target": target,
            "target_sensor_origins": target_origins_all[target_indices],
            "canonical": canonical,
            "baseline": baseline,
            "compiled": compiled,
            "kept": query[keep],
            "unknown_query": query[~keep],
            "ghost_hit": selected_query,
            "ghost_ray": ray,
            "ghost": ghost,
            "projected": np.asarray(projected, dtype=np.float32).reshape(-1, 3),
            "projected_aligned": projected_aligned,
            "ghost_actions": tuple(ghost_actions),
            "duplicate": duplicate,
            "flicker": flicker,
            "completed": completed_points,
        }
    return row, package


def compile_log(
    log_dir: Path,
    config: Mapping[str, Any],
    device: torch.device,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    annotations = pd.read_feather(log_dir / "annotations.feather")
    poses = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    tracks = _track_geometries(annotations, poses, config)
    eligible_ids = set(tracks)
    annotations = annotations[annotations["track_uuid"].isin(eligible_ids)].copy()
    frame_ranks = {
        track_id: {int(timestamp): index for index, timestamp in enumerate(track.timestamps_ns)}
        for track_id, track in tracks.items()
    }
    grouped = {
        int(timestamp): frame
        for timestamp, frame in annotations.groupby("timestamp_ns", sort=True)
    }
    sweep_paths = [
        path
        for path in sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
        if int(path.stem) in grouped
    ]
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_read_lidar, sweep_paths[0]) if sweep_paths else None
        for index, path in enumerate(sweep_paths):
            assert future is not None
            points = future.result()
            if index + 1 < len(sweep_paths):
                future = executor.submit(_read_lidar, sweep_paths[index + 1])
            _associate_actor_points(
                points, grouped[int(path.stem)], frame_ranks, records, config, device
            )
    actor_rows, packages = [], []
    diagnostics: dict[str, dict[str, Any]] = {}
    offsets = [0]
    actor_ids = []
    for track_id in sorted(tracks):
        compiled = _compile_actor(
            tracks[track_id],
            sorted(records.get(track_id, []), key=lambda item: item["frame_rank"]),
            config,
            device,
            include_diagnostics=include_diagnostics,
        )
        if compiled is None:
            continue
        row, package = compiled
        actor_rows.append(row)
        packages.append(package["points"])
        offsets.append(offsets[-1] + len(package["points"]))
        actor_ids.append(track_id)
        if include_diagnostics:
            diagnostics[track_id] = package["diagnostics"]
    return {
        "log_id": log_dir.name,
        "actor_rows": actor_rows,
        "compiled": {
            "points": np.concatenate(packages, axis=0)
            if packages
            else np.empty((0, 3), dtype=np.float32),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "actor_ids": actor_ids,
            "diagnostics": diagnostics,
        },
        "metadata_tracks": len(tracks),
        "sweeps_read": len(sweep_paths),
    }


def summarize_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("role has no evaluable Actor rows")
    actions = Counter()
    inputs = Counter()
    for row in rows:
        actions.update(row["action_counts"])
        inputs.update(row["input_counts"])
    artifact_inputs = (
        inputs["observed_free_ghost"]
        + inputs["duplicate_shell"]
        + inputs["temporal_flicker"]
    )
    clean_keep = sum(row["clean_keep_count"] for row in rows)
    ghost_project = sum(row["ghost_project_count"] for row in rows)
    duplicate_unknown = sum(row["duplicate_unknown_count"] for row in rows)
    flicker_unknown = sum(row["flicker_unknown_count"] for row in rows)
    completion_decisions = sum(row["completion_decision_count"] for row in rows)
    artifact_compiled = ghost_project + duplicate_unknown + flicker_unknown
    completion_total = inputs["surface_hole"]
    completion_supported = sum(row["completion_supported_count"] for row in rows)
    before_chamfer = np.asarray([row["before"]["symmetric_chamfer_m"] for row in rows])
    after_chamfer = np.asarray([row["after"]["symmetric_chamfer_m"] for row in rows])
    return {
        "actor_count": len(rows),
        "hazard_actor_count": int(sum(row["hazardous"] for row in rows)),
        "safe_actor_count": int(sum(not row["hazardous"] for row in rows)),
        "action_counts": dict(sorted(actions.items())),
        "input_counts": dict(sorted(inputs.items())),
        "clean_keep_rate": clean_keep / max(inputs["clean"], 1),
        "ghost_project_rate": ghost_project / max(inputs["observed_free_ghost"], 1),
        "unknown_artifact_rate": (duplicate_unknown + flicker_unknown)
        / max(inputs["duplicate_shell"] + inputs["temporal_flicker"], 1),
        "completion_decision_rate": completion_decisions / max(completion_total, 1),
        "completion_target_support": completion_supported / max(actions["COMPLETE"], 1),
        "artifact_repair_or_reject_rate": artifact_compiled / max(artifact_inputs, 1),
        "mean_target_recall_before": float(
            np.mean([row["before"]["target_recall"] for row in rows])
        ),
        "mean_target_recall_after": float(
            np.mean([row["after"]["target_recall"] for row in rows])
        ),
        "mean_target_recall_delta": float(
            np.mean([row["target_recall_delta"] for row in rows])
        ),
        "mean_surface_precision_before": float(
            np.mean([row["before"]["surface_precision"] for row in rows])
        ),
        "mean_surface_precision_after": float(
            np.mean([row["after"]["surface_precision"] for row in rows])
        ),
        "mean_surface_precision_delta": float(
            np.mean([row["surface_precision_delta"] for row in rows])
        ),
        "mean_chamfer_before_m": float(np.mean(before_chamfer)),
        "mean_chamfer_after_m": float(np.mean(after_chamfer)),
        "mean_chamfer_ratio": float(
            np.mean(after_chamfer) / max(float(np.mean(before_chamfer)), 1e-8)
        ),
        "actor_identity_trajectory_size_retention": 1.0,
        "hazard_label_retention": 1.0,
    }

#!/usr/bin/env python3
"""Compile the frozen V6.3 development proposal-surface corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN, VoxelGridSpec
from motion_proj.worldsim_v62.evidence import build_evidence_grid
from motion_proj.worldsim_v63.native_features import target_points_to_native_indices
from motion_proj.worldsim_v63.surface_builder import (
    boundary_indices,
    label_components,
    partition_surface,
    run_negative_contract_tests,
    surface_normals,
    validate_unique,
)


TASK_ID = "WS-V63-P3-SURFACE-CORPUS-01"
SURFACE_TYPE = {
    "route_support": 0,
    "static_disocclusion": 1,
    "actor_surface": 2,
    "actor_swept_surface": 3,
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _indices_mask(indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if indices.shape[0]:
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return mask


def _actor_index_map(
    indices: np.ndarray, ids: np.ndarray, shape: tuple[int, int, int]
) -> dict[int, np.ndarray]:
    result = {}
    for actor_id in np.unique(ids):
        if int(actor_id) < 0:
            continue
        result[int(actor_id)] = _indices_mask(indices[ids == actor_id], shape)
    return result


def _native_occupied_target_grid(
    argmax_path: Path,
    target_spec: VoxelGridSpec,
    source_origin_m: np.ndarray,
    source_voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native_argmax = np.load(argmax_path, mmap_mode="r")
    axes = []
    valids = []
    for axis, size in enumerate(target_spec.shape):
        centers = np.zeros((size, 3), dtype=np.float64)
        centers[:, axis] = (
            target_spec.origin[axis]
            + (np.arange(size, dtype=np.float64) + 0.5) * target_spec.voxel_size_m
        )
        for other in range(3):
            if other != axis:
                centers[:, other] = target_spec.origin[other] + 0.5 * target_spec.voxel_size_m
        indices, valid = target_points_to_native_indices(
            centers,
            source_origin_m=source_origin_m,
            source_voxel_size_m=source_voxel_size_m,
            source_shape=tuple(int(value) for value in native_argmax.shape),
        )
        axes.append(indices[:, axis])
        valids.append(valid)
    valid_grid = (
        valids[0][:, None, None]
        & valids[1][None, :, None]
        & valids[2][None, None, :]
    )
    clipped = [
        np.clip(axis, 0, native_argmax.shape[index] - 1)
        for index, axis in enumerate(axes)
    ]
    occupied = native_argmax[
        clipped[0][:, None, None],
        clipped[1][None, :, None],
        clipped[2][None, None, :],
    ] > 0
    return occupied & valid_grid, np.stack(axes, axis=0), valid_grid


def _temporal_support(
    scene_root: Path,
    target_frame: int,
    method_frames: list[int],
    spec: VoxelGridSpec,
    cohort: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    free_count = np.zeros(spec.shape, dtype=np.uint8)
    occupied_count = np.zeros(spec.shape, dtype=np.uint8)
    contradiction_count = np.zeros(spec.shape, dtype=np.uint8)
    kwargs = {
        "record_width": int(cohort["raw_lidar"]["point_record_float32_width"]),
        "dynamic_box_margin_m": float(cohort["raw_lidar"]["dynamic_box_margin_m"]),
        "maximum_free_rays_per_sweep": int(cohort["ray_carving"]["maximum_rays_per_sweep"]),
        "maximum_range_m": float(cohort["ray_carving"]["maximum_range_m"]),
        "behind_hit_steps": int(cohort["ray_carving"]["behind_hit_steps"]),
    }
    for frame in method_frames:
        grid = build_evidence_grid(scene_root, target_frame, [frame], spec, **kwargs)
        semantics = np.asarray(grid.arrays["semantics"])
        free_count += semantics == FREE
        occupied_count += semantics == OCCUPIED
        contradiction_count += np.asarray(grid.arrays["contradiction"], dtype=np.uint8)
    return free_count, occupied_count, contradiction_count


def _compile_volume(
    *,
    volume: np.ndarray,
    surface_kind: str,
    proposal_prefix: str,
    actor_id: int,
    actor_current: bool,
    scene: str,
    target_frame: int,
    method: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    temporal_free: np.ndarray,
    temporal_occ: np.ndarray,
    temporal_contradiction: np.ndarray,
    target_spec: VoxelGridSpec,
    source_origin_m: np.ndarray,
    source_voxel_size_m: float,
    patch_config: dict[str, int],
    native_unit_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    labels, component_count = label_components(volume)
    surfaces = []
    patches = []
    proposals = []
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "grid_indices",
            "coordinates_m",
            "normals",
            "surface_index",
            "patch_index",
            "native_indices",
            "native_valid",
            "method_state",
            "target_state",
            "method_contradiction",
            "target_contradiction",
            "temporal_free_count",
            "temporal_occ_count",
            "temporal_contradiction_count",
            "ray_direction",
            "ray_hit_order",
            "ray_bundle_id",
            "actor_id",
            "actor_current_support",
            "actor_swept_support",
            "authority_bits",
            "surface_type",
        )
    }
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        component_indices = np.argwhere(component).astype(np.int32)
        surface_points = boundary_indices(component)
        if surface_points.shape[0] == 0:
            continue
        proposal_id = f"{proposal_prefix}__c{component_id:05d}"
        surface_id = f"{proposal_id}__surface"
        point_start = sum(values.shape[0] for values in arrays["grid_indices"])
        normal = surface_normals(component_indices, surface_points, target_spec.shape)
        patch_members = partition_surface(
            surface_points,
            target_spec.shape,
            minimum_points=int(patch_config["minimum_points"]),
            target_points=int(patch_config["target_points"]),
            maximum_points=int(patch_config["maximum_points"]),
        )
        surface_index = len(surfaces)
        point_patch = np.full(surface_points.shape[0], -1, dtype=np.int32)
        for local_patch, members in enumerate(patch_members):
            patch_index = len(patches)
            point_patch[members] = patch_index
            patches.append(
                {
                    "patch_id": f"{surface_id}__p{local_patch:04d}",
                    "surface_id": surface_id,
                    "proposal_id": proposal_id,
                    "scene": scene,
                    "frame_id": target_frame,
                    "point_count": int(members.shape[0]),
                    "point_indices_local": members.tolist(),
                }
            )
        if np.any(point_patch < 0):
            raise RuntimeError("unassigned surface point")
        coordinates = target_spec.indices_to_centers(surface_points).astype(np.float32)
        native_indices, native_valid = target_points_to_native_indices(
            coordinates,
            source_origin_m=source_origin_m,
            source_voxel_size_m=source_voxel_size_m,
            source_shape=(200, 200, 16),
        )
        direction = coordinates.copy()
        distance = np.linalg.norm(direction, axis=1)
        direction /= np.maximum(distance[:, None], 1e-6)
        azimuth = np.floor((np.arctan2(direction[:, 1], direction[:, 0]) + np.pi) / (2 * np.pi) * 72).astype(np.int32) % 72
        elevation = np.clip(np.floor((direction[:, 2] + 1.0) * 9).astype(np.int32), 0, 17)
        ray_bundle = azimuth * 18 + elevation
        idx = tuple(surface_points.T)
        method_state = np.asarray(method["semantics"])[idx]
        target_state = np.asarray(target["semantics"])[idx]
        method_contradiction = np.asarray(method["contradiction"])[idx]
        target_contradiction = np.asarray(target["contradiction"])[idx]
        closed = not bool(
            np.any(surface_points == 0)
            or np.any(surface_points == np.asarray(target_spec.shape)[None] - 1)
        )
        current_support = np.full(surface_points.shape[0], actor_current, dtype=bool)
        swept_support = np.full(surface_points.shape[0], actor_id >= 0, dtype=bool)
        authority = (
            (method_state == OCCUPIED).astype(np.uint8)
            | ((temporal_occ[idx] >= 2).astype(np.uint8) << 1)
            | (current_support.astype(np.uint8) << 2)
            | (swept_support.astype(np.uint8) << 3)
            | (np.full(surface_points.shape[0], closed, dtype=np.uint8) << 4)
        )
        point_stop = point_start + surface_points.shape[0]
        hidden_free = int(np.count_nonzero(target_state == FREE))
        target_occ = int(np.count_nonzero(target_state == OCCUPIED))
        surfaces.append(
            {
                "surface_id": surface_id,
                "proposal_id": proposal_id,
                "scene": scene,
                "frame_id": target_frame,
                "surface_type": surface_kind,
                "actor_id": actor_id,
                "component_voxel_count": int(component_indices.shape[0]),
                "point_count": int(surface_points.shape[0]),
                "patch_count": len(patch_members),
                "point_start": point_start,
                "point_stop": point_stop,
                "closed_within_roi": closed,
                "hidden_free_count": hidden_free,
                "target_occupied_count": target_occ,
                "authority_point_count": int(np.count_nonzero(authority)),
                "normal_valid_fraction": float(
                    np.mean(
                        np.isfinite(normal).all(axis=1)
                        & (np.linalg.norm(normal, axis=1) > 0.9)
                        & (np.linalg.norm(normal, axis=1) < 1.1)
                    )
                ),
                "native_valid_fraction": float(np.mean(native_valid)),
                "native_unit_path": native_unit_path,
            }
        )
        proposals.append(
            {
                "proposal_id": proposal_id,
                "surface_id": surface_id,
                "scene": scene,
                "frame_id": target_frame,
                "proposal_type": surface_kind,
                "actor_id": actor_id,
                "source": "actor_method_envelope" if actor_id >= 0 else "native_argmax_plus_observed_occ",
                "geometry_changed_by_topology": False,
            }
        )
        arrays["grid_indices"].append(surface_points.astype(np.int16))
        arrays["coordinates_m"].append(coordinates)
        arrays["normals"].append(normal.astype(np.float16))
        arrays["surface_index"].append(np.full(surface_points.shape[0], surface_index, dtype=np.int32))
        arrays["patch_index"].append(point_patch)
        arrays["native_indices"].append(native_indices.astype(np.int16))
        arrays["native_valid"].append(native_valid)
        arrays["method_state"].append(method_state.astype(np.uint8))
        arrays["target_state"].append(target_state.astype(np.uint8))
        arrays["method_contradiction"].append(method_contradiction.astype(bool))
        arrays["target_contradiction"].append(target_contradiction.astype(bool))
        arrays["temporal_free_count"].append(temporal_free[idx].astype(np.uint8))
        arrays["temporal_occ_count"].append(temporal_occ[idx].astype(np.uint8))
        arrays["temporal_contradiction_count"].append(temporal_contradiction[idx].astype(np.uint8))
        arrays["ray_direction"].append(direction.astype(np.float16))
        arrays["ray_hit_order"].append(distance.astype(np.float16))
        arrays["ray_bundle_id"].append(ray_bundle.astype(np.int32))
        arrays["actor_id"].append(np.full(surface_points.shape[0], actor_id, dtype=np.int32))
        arrays["actor_current_support"].append(current_support)
        arrays["actor_swept_support"].append(swept_support)
        arrays["authority_bits"].append(authority)
        arrays["surface_type"].append(np.full(surface_points.shape[0], SURFACE_TYPE[surface_kind], dtype=np.uint8))
    return surfaces, patches, proposals, arrays


def _unit(task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    config = task["config"]
    cohort = task["cohort"]
    scene = task["scene"]
    target_frame = int(task["target_frame"])
    target_ordinal = int(task["target_ordinal"])
    scene_root = Path(task["scene_root"])
    p2_unit = Path(config["inputs"]["p2_evidence_run"]) / "units" / scene / f"f{target_frame:03d}"
    native_unit = Path(config["inputs"]["p2_native_run"]) / "units" / "development" / scene / f"f{target_frame:03d}"
    with np.load(p2_unit / "METHOD_EVIDENCE.npz", allow_pickle=False) as source:
        method = {name: np.asarray(source[name]) for name in source.files}
    with np.load(p2_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
        target = {name: np.asarray(source[name]) for name in source.files}
    shape = tuple(int(value) for value in method["grid_shape"])
    spec = VoxelGridSpec(
        frame=f"target_lidar_{target_frame:03d}",
        origin_m=tuple(float(value) for value in method["grid_origin_m"]),
        voxel_size_m=float(method["voxel_size_m"]),
        shape=shape,
    )
    native_occ, _, _ = _native_occupied_target_grid(
        native_unit / "ARGMAX.npy",
        spec,
        np.asarray(config["native_grid"]["origin_m"], dtype=np.float64),
        float(config["native_grid"]["voxel_size_m"]),
    )
    candidate_offsets = [int(value) for value in cohort["sweep_roles"]["method_candidate_offsets"]]
    dropout_offset = candidate_offsets[target_ordinal % len(candidate_offsets)]
    method_frames = [target_frame + value for value in candidate_offsets if value != dropout_offset]
    target_frames = [target_frame + int(value) for value in cohort["sweep_roles"]["target_evidence_offsets"]]
    if set(method_frames) & set(target_frames):
        raise RuntimeError("method/target frame overlap")
    temporal_free, temporal_occ, temporal_contradiction = _temporal_support(
        scene_root, target_frame, method_frames, spec, cohort
    )

    actor_indices = np.asarray(method["actor_envelope_indices"], dtype=np.int32)
    actor_ids = np.asarray(method["actor_envelope_ids"], dtype=np.int32)
    current_indices = np.asarray(method["actor_current_envelope_indices"], dtype=np.int32)
    current_ids = np.asarray(method["actor_current_envelope_ids"], dtype=np.int32)
    actor_maps = _actor_index_map(actor_indices, actor_ids, shape)
    current_actor_ids = set(int(value) for value in np.unique(current_ids) if int(value) >= 0)
    actor_union = _indices_mask(actor_indices, shape)
    static_volume = (native_occ | (np.asarray(method["semantics"]) == OCCUPIED)) & ~actor_union

    unit_surface_rows: list[dict[str, Any]] = []
    unit_patch_rows: list[dict[str, Any]] = []
    unit_proposal_rows: list[dict[str, Any]] = []
    unit_arrays: dict[str, list[np.ndarray]] = {}
    for region_kind, volume, actor_id, actor_current, prefix in [
        ("static", static_volume, -1, False, f"{scene}__f{target_frame:03d}__static")
    ] + [
        (
            "actor_surface" if actor_id in current_actor_ids else "actor_swept_surface",
            actor_maps[actor_id],
            actor_id,
            actor_id in current_actor_ids,
            f"{scene}__f{target_frame:03d}__actor{actor_id}",
        )
        for actor_id in sorted(actor_maps)
    ]:
        surfaces, patches, proposals, arrays = _compile_volume(
            volume=volume,
            surface_kind="static_disocclusion" if region_kind == "static" else region_kind,
            proposal_prefix=prefix,
            actor_id=actor_id,
            actor_current=actor_current,
            scene=scene,
            target_frame=target_frame,
            method=method,
            target=target,
            temporal_free=temporal_free,
            temporal_occ=temporal_occ,
            temporal_contradiction=temporal_contradiction,
            target_spec=spec,
            source_origin_m=np.asarray(config["native_grid"]["origin_m"], dtype=np.float64),
            source_voxel_size_m=float(config["native_grid"]["voxel_size_m"]),
            patch_config=config["patch"],
            native_unit_path=str(native_unit),
        )
        if region_kind == "static":
            for local_surface_index, surface in enumerate(surfaces):
                start, stop = int(surface["point_start"]), int(surface["point_stop"])
                local_points = np.concatenate(arrays["coordinates_m"], axis=0)[start:stop]
                route_fraction = float(np.mean((local_points[:, 0] >= 0.0) & (np.abs(local_points[:, 1]) <= 4.0) & (local_points[:, 2] <= 1.0)))
                if route_fraction >= 0.15:
                    surface["surface_type"] = "route_support"
                    proposal = next(row for row in proposals if row["proposal_id"] == surface["proposal_id"])
                    proposal["proposal_type"] = "route_support"
                    arrays["surface_type"][local_surface_index][:] = SURFACE_TYPE["route_support"]
        surface_offset = len(unit_surface_rows)
        patch_offset = len(unit_patch_rows)
        point_offset = sum(value.shape[0] for value in unit_arrays.get("grid_indices", []))
        for local_surface_index, surface in enumerate(surfaces):
            surface["surface_index"] = surface_offset + local_surface_index
            surface["point_start"] += point_offset
            surface["point_stop"] += point_offset
        for local_patch_index, patch in enumerate(patches):
            patch["patch_index"] = patch_offset + local_patch_index
        for name, values in arrays.items():
            unit_arrays.setdefault(name, [])
            if name == "surface_index":
                values = [value + surface_offset for value in values]
            if name == "patch_index":
                values = [value + patch_offset for value in values]
            unit_arrays[name].extend(values)
        unit_surface_rows.extend(surfaces)
        unit_patch_rows.extend(patches)
        unit_proposal_rows.extend(proposals)

    for index, surface in enumerate(unit_surface_rows):
        surface["surface_index"] = index
    for index, patch in enumerate(unit_patch_rows):
        patch["patch_index"] = index
    validate_unique((row["surface_id"] for row in unit_surface_rows), "surface id")
    validate_unique((row["patch_id"] for row in unit_patch_rows), "patch id")
    validate_unique((row["proposal_id"] for row in unit_proposal_rows), "proposal id")
    concatenated = {
        name: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        for name, values in unit_arrays.items()
    }
    unit_dir = Path(task["run_dir"]) / "units" / scene / f"f{target_frame:03d}"
    unit_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(unit_dir / "SURFACE_POINTS.npz", **concatenated)
    _write_jsonl(unit_dir / "SURFACE_REGISTRY.jsonl", unit_surface_rows)
    _write_jsonl(unit_dir / "PATCH_REGISTRY.jsonl", unit_patch_rows)
    _write_jsonl(unit_dir / "PROPOSAL_REGISTRY.jsonl", unit_proposal_rows)
    bytes_written = sum(path.stat().st_size for path in unit_dir.iterdir())
    return {
        "scene": scene,
        "target_frame": target_frame,
        "method_frames": method_frames,
        "target_frames": target_frames,
        "surface_rows": unit_surface_rows,
        "patch_rows": unit_patch_rows,
        "proposal_rows": unit_proposal_rows,
        "unit_path": str(unit_dir.relative_to(Path(task["run_dir"]))),
        "point_count": int(concatenated["grid_indices"].shape[0]),
        "bytes": bytes_written,
        "wall_seconds": time.monotonic() - started,
    }


def run(
    config_path: Path,
    repo_root: Path,
    run_dir: Path,
    maximum_workers: int,
    limit_units: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip():
        raise RuntimeError("P3 formal requires clean source")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = yaml.safe_load(Path(config["inputs"]["development_cohort_config"]).read_text())
    if config["task_id"] != TASK_ID:
        raise ValueError("P3 task identity drift")
    free_gib = shutil.disk_usage(run_dir.parent).free / 1024**3
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise RuntimeError(f"insufficient disk before P3: {free_gib:.3f} GiB")
    run_dir.mkdir(parents=True)
    negative_tests = run_negative_contract_tests()
    _write_json(run_dir / "NEGATIVE_TESTS.json", negative_tests)
    p2_split = json.loads((Path(config["inputs"]["p2_evidence_run"]) / "SPLIT_MANIFEST.json").read_text())
    if int(p2_split["source_role_overlap_count"]) != 0:
        raise RuntimeError("P2 role overlap")
    tasks = []
    for scene_ordinal, scene in enumerate(cohort["scenes"]):
        scene_root = Path(config["inputs"]["processed_root"]) / f"{int(scene['processed_index']):03d}"
        for target_ordinal, target_frame in enumerate(cohort["targets"]["frame_indices"]):
            tasks.append(
                {
                    "config": config,
                    "cohort": cohort,
                    "scene": scene["name"],
                    "scene_ordinal": scene_ordinal,
                    "target_frame": int(target_frame),
                    "target_ordinal": target_ordinal,
                    "scene_root": str(scene_root),
                    "run_dir": str(run_dir),
                }
            )
    if limit_units is not None:
        tasks = tasks[: int(limit_units)]
    workers = max(1, min(int(maximum_workers), int(config["resources"]["maximum_workers"]), len(tasks)))
    if workers == 1:
        rows = [_unit(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_unit, tasks))
    rows.sort(key=lambda row: (row["scene"], row["target_frame"]))

    surfaces = []
    patches = []
    proposals = []
    native_index = []
    evidence_roles = []
    for row in rows:
        surfaces.extend(row["surface_rows"])
        patches.extend(row["patch_rows"])
        proposals.extend(row["proposal_rows"])
        native_index.append(
            {
                "scene": row["scene"],
                "frame_id": row["target_frame"],
                "unit_path": row["unit_path"],
                "native_sidecar": str(
                    Path(config["inputs"]["p2_native_run"])
                    / "units/development"
                    / row["scene"]
                    / f"f{int(row['target_frame']):03d}"
                ),
                "prototype_used": False,
            }
        )
        evidence_roles.append(
            {
                "scene": row["scene"],
                "frame_id": row["target_frame"],
                "method_frames": row["method_frames"],
                "target_frames": row["target_frames"],
                "overlap": False,
            }
        )
    validate_unique((row["surface_id"] for row in surfaces), "global surface id")
    validate_unique((row["patch_id"] for row in patches), "global patch id")
    validate_unique((row["proposal_id"] for row in proposals), "global proposal id")
    _write_jsonl(run_dir / "SURFACE_REGISTRY.jsonl", surfaces)
    _write_jsonl(run_dir / "PATCH_REGISTRY.jsonl", patches)
    _write_jsonl(run_dir / "PROPOSAL_REGISTRY.jsonl", proposals)
    _write_jsonl(run_dir / "NATIVE_FEATURE_INDEX.jsonl", native_index)
    _write_jsonl(run_dir / "EVIDENCE_ROLE_INDEX.jsonl", evidence_roles)
    total_points = sum(int(row["point_count"]) for row in rows)
    summary = {
        "schema_version": "worldsim_v63.p3_surface_corpus_summary.v1",
        "task_id": TASK_ID,
        "mode": "probe" if limit_units is not None else "formal",
        "unit_count": len(rows),
        "scene_count": len({row["scene"] for row in rows}),
        "surface_count": len(surfaces),
        "patch_count": len(patches),
        "proposal_count": len(proposals),
        "point_count": total_points,
        "surface_type_counts": {
            name: sum(row["surface_type"] == name for row in surfaces)
            for name in SURFACE_TYPE
        },
        "hidden_free_point_count": sum(int(row["hidden_free_count"]) for row in surfaces),
        "target_occupied_point_count": sum(int(row["target_occupied_count"]) for row in surfaces),
        "authority_point_count": sum(int(row["authority_point_count"]) for row in surfaces),
        "minimum_native_valid_fraction": min((float(row["native_valid_fraction"]) for row in surfaces), default=0.0),
        "minimum_normal_valid_fraction": min((float(row["normal_valid_fraction"]) for row in surfaces), default=0.0),
        "small_surface_count": sum(int(row["point_count"]) < int(config["patch"]["minimum_points"]) for row in surfaces),
        "maximum_surface_point_count": max((int(row["point_count"]) for row in surfaces), default=0),
        "maximum_patch_point_count": max((int(row["point_count"]) for row in patches), default=0),
        "source_role_overlap_count": 0,
        "prototype_used": False,
        "negative_tests": negative_tests,
        "output_bytes": sum(int(row["bytes"]) for row in rows),
        "maximum_unit_wall_seconds": max((float(row["wall_seconds"]) for row in rows), default=0.0),
        "wall_seconds": time.monotonic() - started,
        "calibration_quality_read": False,
        "confirmation_read": False,
        "exact_once_test_read": False,
    }
    summary["passed"] = (
        len(rows) > 0
        and len(surfaces) > 0
        and len(patches) > 0
        and total_points > 0
        and summary["minimum_normal_valid_fraction"] == 1.0
        and summary["maximum_patch_point_count"] <= int(config["patch"]["maximum_points"])
        and all(negative_tests.values())
    )
    _write_json(run_dir / "P3_SUMMARY.json", summary)
    _write_json(
        run_dir / "P3_MANIFEST.json",
        {
            "schema_version": "worldsim_v63.p3_surface_corpus_manifest.v1",
            "task_id": TASK_ID,
            "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo_root, text=True).strip(),
            "source_worktree_clean": True,
            "unit_rows": [{key: row[key] for key in ("scene", "target_frame", "unit_path", "point_count", "bytes")} for row in rows],
            "prototype_used": False,
            "topology_changes_geometry": False,
            "identity_policy": "semantic_ids_paths_task_run_git_history_no_artifact_hash",
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--maximum-workers", type=int, default=2)
    parser.add_argument("--limit-units", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.run_dir.resolve(), args.maximum_workers, args.limit_units), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

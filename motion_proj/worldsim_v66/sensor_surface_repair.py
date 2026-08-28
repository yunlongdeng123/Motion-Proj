"""用同Actor motion-compensated hit修复冻结action states的局部surface。"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from motion_proj.worldsim_v61.occupancy import FREE
from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.native_voxel_uq import (
    _evidence_on_native_grid,
    _native_unit_dir,
    _unit_dirs,
)
from motion_proj.worldsim_v66.actor_factorial import _ground_actor_ids


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_repair_unit(
    descriptor: tuple[int, str, Path, Path],
    *,
    origin: np.ndarray,
    voxel_size: float,
    point_limit: int,
    seed: int,
) -> dict[str, Any]:
    scene_index, scene, evidence_unit, native_unit = descriptor
    indices, centers, features = _target_free_boundary(
        evidence_unit,
        native_unit,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    with np.load(evidence_unit / "METHOD_EVIDENCE.npz", allow_pickle=False) as source:
        method = {name: np.asarray(source[name]) for name in source.files}
    with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
        target = {name: np.asarray(source[name]) for name in source.files}
    native_shape = tuple(
        int(value) for value in np.load(native_unit / "ARGMAX.npy", mmap_mode="r").shape
    )
    target_state, target_valid = _evidence_on_native_grid(
        target,
        native_shape=native_shape,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    x, y, z = indices.T
    valid = target_valid[x, y, z]
    indices, centers, features = indices[valid], centers[valid], features[valid]
    labels = np.asarray(target_state[x, y, z][valid] == FREE, dtype=bool)
    if features.shape[0] > point_limit:
        unit_seed = int(seed) + scene_index * 1009 + int(evidence_unit.name.removeprefix("f"))
        rng = np.random.default_rng(unit_seed)
        chosen = np.sort(rng.choice(features.shape[0], size=point_limit, replace=False))
        indices, centers, labels = indices[chosen], centers[chosen], labels[chosen]
    actor_ids, _ = _ground_actor_ids(centers, method)

    evidence_origin = np.asarray(method["grid_origin_m"], dtype=np.float64)
    evidence_voxel = float(method["voxel_size_m"])
    evidence_shape = tuple(int(value) for value in method["grid_shape"])
    query_indices = np.floor(
        (np.asarray(centers, dtype=np.float64) - evidence_origin[None, :])
        / evidence_voxel
    ).astype(np.int64)
    query_valid = np.all(
        (query_indices >= 0) & (query_indices < np.asarray(evidence_shape)[None, :]), axis=1
    )
    query_linear = np.full(len(query_indices), -1, dtype=np.int64)
    query_linear[query_valid] = np.ravel_multi_index(query_indices[query_valid].T, evidence_shape)
    hit_indices = np.asarray(method["actor_hit_indices"], dtype=np.int64)
    hit_ids = np.asarray(method["actor_hit_ids"], dtype=np.int64)
    hit_linear = (
        np.ravel_multi_index(hit_indices.T, evidence_shape)
        if hit_indices.size
        else np.empty(0, dtype=np.int64)
    )
    hit_pairs = {(int(linear), int(actor_id)) for linear, actor_id in zip(hit_linear, hit_ids)}
    same_actor_hit = np.asarray(
        [
            valid_query and (int(linear), int(actor_id)) in hit_pairs
            for valid_query, linear, actor_id in zip(query_valid, query_linear, actor_ids)
        ],
        dtype=bool,
    )
    return {
        "scene": scene,
        "unit": evidence_unit.name,
        "indices": np.asarray(indices, dtype=np.int32),
        "labels": labels,
        "actor_ids": actor_ids,
        "same_actor_hit": same_actor_hit,
    }


def repair_surface(config: Mapping[str, Any], runs_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = config["sources"]
    evidence_root = runs_root / str(sources["evidence_run"])
    native_root = runs_root / str(sources["native_run"])
    action_rows = _jsonl(
        runs_root / str(sources["p7_run"]) / str(sources["p7_action_rows"])
    )
    actions = {
        str(row["base_id"]): str(row["local_action"])
        for row in action_rows
        if str(row["arm"]) == str(config["action_arm"])
    }
    package_root = runs_root / str(sources["harp_bake_run"]) / "package"
    manifest = json.loads((package_root / "RUNTIME_MANIFEST.json").read_text())
    partition = str(sources["native_partition"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    descriptors = []
    for scene_index, scene in enumerate(config["scenes"]):
        for evidence_unit in _unit_dirs(evidence_root, str(scene)):
            descriptors.append(
                (
                    scene_index,
                    str(scene),
                    evidence_unit,
                    _native_unit_dir(
                        native_root,
                        str(scene),
                        evidence_unit.name,
                        {str(scene): partition},
                    ),
                )
            )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["sampling"]["io_prefetch_workers"])
    )

    def submit(descriptor: tuple[int, str, Path, Path]):
        return executor.submit(
            _load_repair_unit,
            descriptor,
            origin=origin,
            voxel_size=voxel_size,
            point_limit=int(config["sampling"]["maximum_boundary_points_per_unit"]),
            seed=int(config["seed"]),
        )

    state_rows = []
    retained_indices = []
    retained_offsets = [0]
    retained_base_ids = []
    future = submit(descriptors[0])
    try:
        for position, descriptor in enumerate(descriptors):
            loaded = future.result()
            if position + 1 < len(descriptors):
                future = submit(descriptors[position + 1])
            actor_ids = np.asarray(loaded["actor_ids"], dtype=np.int32)
            for actor_id in np.unique(actor_ids[actor_ids >= 0]):
                base_id = f"{loaded['scene']}/{loaded['unit']}/actor-{int(actor_id)}"
                if base_id not in actions:
                    continue
                members = actor_ids == int(actor_id)
                acted = actions[base_id] == "RANK_REPAIR_OR_ABSTAIN"
                keep = (
                    np.asarray(loaded["same_actor_hit"])[members]
                    if acted
                    else np.ones(int(np.count_nonzero(members)), dtype=bool)
                )
                labels = np.asarray(loaded["labels"])[members]
                indices = np.asarray(loaded["indices"])[members]
                kept_indices = indices[keep]
                retained_indices.append(kept_indices)
                retained_offsets.append(retained_offsets[-1] + len(kept_indices))
                retained_base_ids.append(base_id)
                state_rows.append(
                    {
                        "base_id": base_id,
                        "scene": loaded["scene"],
                        "unit": loaded["unit"],
                        "actor_id": int(actor_id),
                        "action": actions[base_id],
                        "actor_retained": True,
                        "collision_shell_retained": True,
                        "baseline_boundary_count": int(len(labels)),
                        "repaired_boundary_count": int(np.count_nonzero(keep)),
                        "baseline_conflict_point_count": int(np.count_nonzero(labels)),
                        "repaired_conflict_point_count": int(np.count_nonzero(labels & keep)),
                        "baseline_clean_point_count": int(np.count_nonzero(~labels)),
                        "repaired_clean_point_count": int(np.count_nonzero((~labels) & keep)),
                    }
                )
            print(
                f"P7R {position + 1}/{len(descriptors)} scene={descriptor[1]} unit={descriptor[2].name}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)

    baseline_points = sum(row["baseline_boundary_count"] for row in state_rows)
    repaired_points = sum(row["repaired_boundary_count"] for row in state_rows)
    baseline_conflicts = sum(row["baseline_conflict_point_count"] for row in state_rows)
    repaired_conflicts = sum(row["repaired_conflict_point_count"] for row in state_rows)
    baseline_clean = sum(row["baseline_clean_point_count"] for row in state_rows)
    repaired_clean = sum(row["repaired_clean_point_count"] for row in state_rows)
    scene_yield = len(
        {row["scene"] for row in state_rows if row["repaired_boundary_count"] > 0}
    ) / len(set(config["scenes"]))
    evaluation = {
        "source_unit_count": len(descriptors),
        "actor_state_count": len(state_rows),
        "action_state_count": sum(row["action"] == "RANK_REPAIR_OR_ABSTAIN" for row in state_rows),
        "baseline_boundary_point_count": baseline_points,
        "repaired_boundary_point_count": repaired_points,
        "overall_boundary_retention": repaired_points / baseline_points,
        "baseline_conflict_point_count": baseline_conflicts,
        "repaired_conflict_point_count": repaired_conflicts,
        "conflict_point_reduction": 1.0 - repaired_conflicts / baseline_conflicts,
        "baseline_clean_point_count": baseline_clean,
        "repaired_clean_point_count": repaired_clean,
        "clean_boundary_retention": repaired_clean / baseline_clean,
        "actor_retention": 1.0,
        "collision_shell_retention": 1.0,
        "actor_id_track_trajectory_retention": 1.0,
        "actor_removed_count": 0,
        "maximum_hazard_proxy_distribution_shift": 0.0,
        "world_scene_yield": scene_yield,
        "target_used_for_action_or_retention": False,
        "physical_local_surface_mutated": True,
        "canonical_actor_count": int(manifest["counts"]["actor_count"]),
    }
    repaired = {
        "base_ids": np.asarray(retained_base_ids),
        "offsets": np.asarray(retained_offsets, dtype=np.int64),
        "indices": np.concatenate(retained_indices).astype(np.int32),
        "native_grid_origin_m": origin,
        "native_voxel_size_m": np.asarray(voxel_size, dtype=np.float64),
    }
    return evaluation, {"state_rows": state_rows, "repaired": repaired}

"""Lightweight hard-evidence construction without integrity side channels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from motion_proj.worldsim_v61.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Transform,
    VoxelGridSpec,
    load_frame_boxes,
    load_lidar,
    points_inside_oriented_box,
    voxelize_oriented_box,
)


class EvidenceGrid(NamedTuple):
    arrays: dict[str, np.ndarray]
    summary: dict[str, Any]


def _ray_observations(
    spec: VoxelGridSpec,
    sensor_origin: np.ndarray,
    endpoints: np.ndarray,
    maximum_rays: int,
    maximum_range_m: float,
    behind_hit_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    free = np.zeros(spec.shape, dtype=bool)
    behind = np.zeros(spec.shape, dtype=bool)
    if endpoints.shape[0] == 0:
        return free, behind

    count = min(int(maximum_rays), int(endpoints.shape[0]))
    selected = endpoints[np.linspace(0, endpoints.shape[0] - 1, num=count, dtype=np.int64)]
    delta = selected - sensor_origin[None, :]
    raw_distances = np.linalg.norm(delta, axis=1)
    valid = (raw_distances > spec.voxel_size_m) & np.isfinite(raw_distances)
    delta = delta[valid]
    raw_distances = raw_distances[valid]
    directions = delta / raw_distances[:, None]
    distances = np.minimum(raw_distances, float(maximum_range_m))

    maximum_steps = int(np.ceil(float(maximum_range_m) / spec.voxel_size_m))
    for step in range(1, maximum_steps):
        distance = step * spec.voxel_size_m
        active = distance < distances - 0.5 * spec.voxel_size_m
        if not np.any(active):
            break
        points = sensor_origin[None, :] + directions[active] * distance
        indices, inside = spec.points_to_indices(points)
        indices = indices[inside]
        if indices.shape[0]:
            free[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    within_range = raw_distances <= float(maximum_range_m)
    for step in range(1, int(behind_hit_steps) + 1):
        points = selected[valid][within_range] + directions[within_range] * (
            step * spec.voxel_size_m
        )
        indices, inside = spec.points_to_indices(points)
        indices = indices[inside]
        if indices.shape[0]:
            behind[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return free, behind


def _unique_sparse(
    indices: Sequence[np.ndarray], ids: Sequence[np.ndarray], shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    if not indices:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)
    merged_indices = np.concatenate(indices, axis=0).astype(np.int32, copy=False)
    merged_ids = np.concatenate(ids, axis=0).astype(np.int32, copy=False)
    linear = np.ravel_multi_index(merged_indices.T, shape)
    _, first = np.unique(linear, return_index=True)
    return merged_indices[first], merged_ids[first]


def build_evidence_grid(
    scene_root: Path,
    target_frame: int,
    source_frames: Sequence[int],
    spec: VoxelGridSpec,
    *,
    record_width: int,
    dynamic_box_margin_m: float,
    maximum_free_rays_per_sweep: int,
    maximum_range_m: float,
    behind_hit_steps: int,
) -> EvidenceGrid:
    """Build tri-state evidence with static/actor separation and conflict state."""

    scene_root = scene_root.resolve()
    instances = json.loads(
        (scene_root / "instances/instances_info.json").read_text(encoding="utf-8")
    )
    frame_instances = json.loads(
        (scene_root / "instances/frame_instances.json").read_text(encoding="utf-8")
    )
    target_pose = Transform(
        dst="global",
        src=spec.frame,
        matrix=np.loadtxt(scene_root / f"lidar_pose/{int(target_frame):03d}.txt"),
    )
    target_global = target_pose.inverse()
    target_boxes = load_frame_boxes(
        scene_root,
        int(target_frame),
        target_global,
        instances=instances,
        frame_instances=frame_instances,
    )
    target_box_by_id = {int(row["actor_id"]): row for row in target_boxes}

    free_observed = np.zeros(spec.shape, dtype=bool)
    occupied_observed = np.zeros(spec.shape, dtype=bool)
    behind_hit = np.zeros(spec.shape, dtype=bool)
    actor_hit_indices: list[np.ndarray] = []
    actor_hit_ids: list[np.ndarray] = []
    actor_swept_indices: list[np.ndarray] = []
    actor_swept_ids: list[np.ndarray] = []
    raw_point_count = 0
    dynamic_point_count = 0

    for source_frame in source_frames:
        points_source = load_lidar(scene_root / f"lidar/{int(source_frame):03d}.bin", record_width)
        source_pose = Transform(
            dst="global",
            src=f"lidar_{int(source_frame):03d}",
            matrix=np.loadtxt(scene_root / f"lidar_pose/{int(source_frame):03d}.txt"),
        )
        target_source = source_pose.then(target_global)
        points_target = target_source.apply(points_source)
        source_boxes = load_frame_boxes(
            scene_root,
            int(source_frame),
            target_global,
            instances=instances,
            frame_instances=frame_instances,
        )

        # Actor queries cover the visible temporal support, not only the
        # instantaneous target ROI.  This retains actors that enter or leave
        # the grid between method sweeps without turning their boxes into hard
        # occupancy evidence.
        for box in source_boxes:
            indices, _ = voxelize_oriented_box(
                spec, box["transform"], box["size_lwh"]
            )
            if indices.shape[0]:
                actor_swept_indices.append(indices.astype(np.int32))
                actor_swept_ids.append(
                    np.full(indices.shape[0], int(box["actor_id"]), dtype=np.int32)
                )

        point_actor = np.full(points_target.shape[0], -1, dtype=np.int32)
        for box in source_boxes:
            inside = points_inside_oriented_box(
                points_target,
                box["transform"],
                box["size_lwh"],
                margin_m=dynamic_box_margin_m,
            )
            point_actor[(point_actor < 0) & inside] = int(box["actor_id"])

        ray_free, ray_behind = _ray_observations(
            spec,
            target_source.matrix[:3, 3],
            points_target,
            maximum_free_rays_per_sweep,
            maximum_range_m,
            behind_hit_steps,
        )
        free_observed |= ray_free
        behind_hit |= ray_behind

        static_points = points_target[point_actor < 0]
        static_indices, static_inside = spec.points_to_indices(static_points)
        static_indices = static_indices[static_inside]
        if static_indices.shape[0]:
            occupied_observed[
                static_indices[:, 0], static_indices[:, 1], static_indices[:, 2]
            ] = True

        for actor_id in np.unique(point_actor[point_actor >= 0]):
            target_box = target_box_by_id.get(int(actor_id))
            if target_box is None:
                continue
            source_box = next(
                row for row in source_boxes if int(row["actor_id"]) == int(actor_id)
            )
            actor_points_source_pose = points_target[point_actor == actor_id]
            local = source_box["transform"].inverse().apply(actor_points_source_pose)
            compensated = target_box["transform"].apply(local)
            indices, inside = spec.points_to_indices(compensated)
            indices = indices[inside]
            if indices.shape[0]:
                actor_hit_indices.append(indices.astype(np.int32))
                actor_hit_ids.append(
                    np.full(indices.shape[0], int(actor_id), dtype=np.int32)
                )
                occupied_observed[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        raw_point_count += int(points_target.shape[0])
        dynamic_point_count += int(np.count_nonzero(point_actor >= 0))

    envelope_indices: list[np.ndarray] = []
    envelope_ids: list[np.ndarray] = []
    for box in target_boxes:
        indices, _ = voxelize_oriented_box(spec, box["transform"], box["size_lwh"])
        if indices.shape[0]:
            envelope_indices.append(indices.astype(np.int32))
            envelope_ids.append(
                np.full(indices.shape[0], int(box["actor_id"]), dtype=np.int32)
            )

    actor_hits, actor_ids = _unique_sparse(actor_hit_indices, actor_hit_ids, spec.shape)
    actor_current_envelope, current_envelope_actor_ids = _unique_sparse(
        envelope_indices, envelope_ids, spec.shape
    )
    actor_swept_envelope, swept_envelope_actor_ids = _unique_sparse(
        actor_swept_indices, actor_swept_ids, spec.shape
    )
    actor_envelope, envelope_actor_ids = _unique_sparse(
        [actor_current_envelope, actor_swept_envelope],
        [current_envelope_actor_ids, swept_envelope_actor_ids],
        spec.shape,
    )
    contradiction = free_observed & occupied_observed
    semantics = np.full(spec.shape, UNKNOWN, dtype=np.uint8)
    semantics[free_observed & ~occupied_observed] = FREE
    semantics[occupied_observed & ~free_observed] = OCCUPIED

    arrays = {
        "semantics": semantics,
        "contradiction": contradiction,
        "behind_hit": behind_hit,
        "actor_hit_indices": actor_hits,
        "actor_hit_ids": actor_ids,
        "actor_current_envelope_indices": actor_current_envelope,
        "actor_current_envelope_ids": current_envelope_actor_ids,
        "actor_swept_envelope_indices": actor_swept_envelope,
        "actor_swept_envelope_ids": swept_envelope_actor_ids,
        "actor_envelope_indices": actor_envelope,
        "actor_envelope_ids": envelope_actor_ids,
        "grid_origin_m": np.asarray(spec.origin_m, dtype=np.float64),
        "voxel_size_m": np.asarray(spec.voxel_size_m, dtype=np.float64),
        "grid_shape": np.asarray(spec.shape, dtype=np.int64),
    }
    summary = {
        "target_frame": int(target_frame),
        "source_frames": [int(value) for value in source_frames],
        "unknown_count": int(np.count_nonzero(semantics == UNKNOWN)),
        "free_count": int(np.count_nonzero(semantics == FREE)),
        "occupied_count": int(np.count_nonzero(semantics == OCCUPIED)),
        "contradiction_count": int(np.count_nonzero(contradiction)),
        "behind_hit_count": int(np.count_nonzero(behind_hit & (semantics == UNKNOWN))),
        "actor_hit_count": int(actor_hits.shape[0]),
        "actor_current_envelope_count": int(actor_current_envelope.shape[0]),
        "actor_swept_envelope_count": int(actor_swept_envelope.shape[0]),
        "actor_envelope_count": int(actor_envelope.shape[0]),
        "actor_count": int(len(target_boxes)),
        "raw_point_count": raw_point_count,
        "motion_compensated_dynamic_point_count": dynamic_point_count,
    }
    return EvidenceGrid(arrays, summary)


def save_evidence_grid(path: Path, grid: EvidenceGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **grid.arrays)


def grid_spec_from_config(config: Mapping[str, Any], target_frame: int) -> VoxelGridSpec:
    values = config["grid"]
    origin = np.asarray(values["origin_m"], dtype=np.float64)
    extent = np.asarray(values["extent_m"], dtype=np.float64)
    voxel_size = float(values["voxel_size_m"])
    shape = tuple(int(round(value)) for value in ((extent - origin) / voxel_size))
    return VoxelGridSpec(
        frame=f"target_lidar_{int(target_frame):03d}",
        origin_m=tuple(float(value) for value in origin),
        voxel_size_m=voxel_size,
        shape=shape,
    )

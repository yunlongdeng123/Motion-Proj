"""V7.1 H1 matched pilot 的确定性轨迹、证书和外部评估原语。"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import atan2
from pathlib import Path
from typing import Iterable

import numpy as np

from .canonical_hash import canonical_json_bytes, canonical_sha256
from .certificates import Verdict
from .coordinates import box_corners_actor, drivestudio_intrinsics
from .safety_geometry import GridSpec, OrientedBox, swept_obb_collision, voxelize_oriented_box


GROUPS = ("A_raw_rigid", "B_kinematic", "C_pairwise", "D1_occgs_certify_only", "D2_occgs_project")


def yaw_of(matrix: np.ndarray) -> float:
    return float(atan2(matrix[1, 0], matrix[0, 0]))


def trajectory_hash(frames: list[dict]) -> str:
    """Hash only realized trajectory bytes; reports/certificates never enter this hash."""
    payload = [
        {
            "frame_index": int(value["frame_index"]),
            "T_world_actor": value["T_world_actor"],
            "dimensions_lwh": value["dimensions_lwh"],
        }
        for value in frames
    ]
    return sha256(canonical_json_bytes(payload)).hexdigest()


def trajectory_bytes(frames: list[dict]) -> bytes:
    return canonical_json_bytes(
        [
            {
                "frame_index": int(value["frame_index"]),
                "T_world_actor": value["T_world_actor"],
                "dimensions_lwh": value["dimensions_lwh"],
            }
            for value in frames
        ]
    )


def _envelope(count: int, timing: str) -> np.ndarray:
    if count < 2:
        return np.zeros(count, dtype=np.float64)
    if timing not in {"early", "centered", "late"}:
        raise ValueError(f"unsupported timing: {timing}")
    center = {"early": 0.38, "centered": 0.50, "late": 0.62}[timing]
    half_width = 0.38
    t = np.linspace(0.0, 1.0, count)
    phase = (t - (center - half_width)) / (2.0 * half_width)
    inside = (phase >= 0.0) & (phase <= 1.0)
    values = np.zeros(count, dtype=np.float64)
    values[inside] = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase[inside]))
    return values


def apply_lateral_proposal(
    source_frames: list[dict],
    ego_by_frame: dict[int, np.ndarray],
    *,
    peak_offset_m: float,
    timing: str,
    scale: float = 1.0,
) -> tuple[list[dict], dict]:
    """Apply a smooth lateral displacement in each frame's LiDAR/ego proxy frame."""
    local_y = []
    for value in source_frames:
        frame = int(value["frame_index"])
        T_world_actor = np.asarray(value["T_world_actor"], dtype=np.float64)
        T_world_ego = np.asarray(ego_by_frame[frame], dtype=np.float64)
        local_y.append(float((np.linalg.inv(T_world_ego) @ T_world_actor)[1, 3]))
    side = float(np.sign(np.median(local_y)))
    if side == 0.0:
        side = 1.0
    offsets = -side * float(peak_offset_m) * float(scale) * _envelope(
        len(source_frames), timing
    )
    output = []
    for value, offset in zip(source_frames, offsets):
        frame = int(value["frame_index"])
        T_world_actor = np.asarray(value["T_world_actor"], dtype=np.float64)
        T_world_ego = np.asarray(ego_by_frame[frame], dtype=np.float64)
        T_ego_actor = np.linalg.inv(T_world_ego) @ T_world_actor
        T_ego_actor[1, 3] += float(offset)
        edited = T_world_ego @ T_ego_actor
        output.append(
            {
                "frame_index": frame,
                "T_world_actor": edited.tolist(),
                "dimensions_lwh": [float(v) for v in value["dimensions_lwh"]],
            }
        )
    return output, {
        "direction_sign": int(-side),
        "peak_requested_offset_m": float(peak_offset_m),
        "realized_peak_offset_m": float(np.max(np.abs(offsets), initial=0.0)),
        "timing": timing,
        "scale": float(scale),
        "offsets_m": offsets.tolist(),
    }


def relative_kinematic_check(
    source_frames: list[dict],
    realized_frames: list[dict],
    ego_by_frame: dict[int, np.ndarray],
    *,
    dt_s: float,
    max_delta_speed_mps: float,
    max_delta_acceleration_mps2: float,
) -> dict:
    source_y, realized_y = [], []
    for source, realized in zip(source_frames, realized_frames):
        frame = int(source["frame_index"])
        inverse = np.linalg.inv(np.asarray(ego_by_frame[frame], dtype=np.float64))
        source_y.append(float((inverse @ np.asarray(source["T_world_actor"]))[1, 3]))
        realized_y.append(float((inverse @ np.asarray(realized["T_world_actor"]))[1, 3]))
    delta = np.asarray(realized_y) - np.asarray(source_y)
    velocity = np.diff(delta) / dt_s
    acceleration = np.diff(velocity) / dt_s
    max_speed = float(np.max(np.abs(velocity), initial=0.0))
    max_acceleration = float(np.max(np.abs(acceleration), initial=0.0))
    passed = max_speed <= max_delta_speed_mps and max_acceleration <= max_delta_acceleration_mps2
    return {
        "verdict": Verdict.PASS.value if passed else Verdict.FAIL.value,
        "max_delta_lateral_speed_mps": max_speed,
        "max_delta_lateral_acceleration_mps2": max_acceleration,
        "reason": "within_threshold" if passed else "relative_kinematic_threshold_exceeded",
    }


def frames_to_boxes(frames: list[dict], *, actor_id: int) -> list[OrientedBox]:
    return [
        OrientedBox(
            tuple(np.asarray(value["T_world_actor"], dtype=float)[:3, 3]),
            tuple(float(v) for v in value["dimensions_lwh"]),
            yaw_of(np.asarray(value["T_world_actor"], dtype=float)),
            actor_id,
        )
        for value in frames
    ]


def aligned_other_boxes(
    frame_indices: list[int], raw_actors: dict, *, excluded_actor_id: int
) -> list[list[OrientedBox]]:
    output = []
    for actor_id, trajectory in sorted(raw_actors.items()):
        if int(actor_id) == int(excluded_actor_id):
            continue
        by_frame = {int(value["frame_index"]): value for value in trajectory}
        if not all(frame in by_frame for frame in frame_indices):
            continue
        output.append(frames_to_boxes([by_frame[frame] for frame in frame_indices], actor_id=int(actor_id)))
    return output


def ego_boxes(frame_indices: list[int], ego_by_frame: dict[int, np.ndarray], dimensions) -> list[OrientedBox]:
    values = []
    for frame in frame_indices:
        matrix = np.asarray(ego_by_frame[frame], dtype=float)
        center = matrix[:3, 3].copy()
        center[2] += float(dimensions[2]) / 2.0
        values.append(OrientedBox(tuple(center), tuple(float(v) for v in dimensions), yaw_of(matrix), "ego"))
    return values


def continuous_safety_check(
    realized_frames: list[dict],
    others: Iterable[list[OrientedBox]],
    *,
    actor_id: int,
    clearance_m: float,
    max_translation_step_m: float = 0.1,
    max_yaw_step_rad: float = 0.02,
) -> dict:
    own = frames_to_boxes(realized_frames, actor_id=actor_id)
    checked = 0
    minimum = float("inf")
    collision_with = None
    for other in others:
        for index in range(min(len(own), len(other)) - 1):
            result = swept_obb_collision(
                own[index],
                own[index + 1],
                other[index],
                other[index + 1],
                clearance_m=clearance_m,
                max_translation_step_m=max_translation_step_m,
                max_yaw_step_rad=max_yaw_step_rad,
            )
            checked += 1
            minimum = min(minimum, float(result["minimum_signed_separation_m"]))
            if result["collision"]:
                collision_with = other[index].actor_id
                return {
                    "verdict": Verdict.FAIL.value,
                    "reason": "continuous_obb_collision",
                    "checked_intervals": checked,
                    "minimum_signed_separation_m": minimum,
                    "collision_with": collision_with,
                }
    if checked == 0:
        return {
            "verdict": Verdict.UNKNOWN.value,
            "reason": "no_comparable_actor",
            "checked_intervals": 0,
        }
    return {
        "verdict": Verdict.PASS.value,
        "reason": "no_continuous_obb_collision",
        "checked_intervals": checked,
        "minimum_signed_separation_m": minimum,
        "collision_with": collision_with,
    }


def _load_sparse_dynamic(npz) -> dict[int, np.ndarray]:
    shape = npz["base_state"].shape
    actor_ids = npz["dynamic_actor_ids"]
    offsets = npz["dynamic_layer_offsets"]
    indices = npz["dynamic_layer_flat_indices"]
    output = {}
    for index, actor_id in enumerate(actor_ids):
        mask = np.zeros(int(np.prod(shape)), dtype=bool)
        mask[indices[offsets[index] : offsets[index + 1]]] = True
        output[int(actor_id)] = mask.reshape(shape)
    return output


def _trim_box_bottom(box: OrientedBox, margin_m: float) -> OrientedBox:
    height = max(float(box.dimensions_lwh[2]) - margin_m, 0.05)
    center = np.asarray(box.center, dtype=float)
    center[2] += margin_m / 2.0
    return OrientedBox(
        tuple(center),
        (box.dimensions_lwh[0], box.dimensions_lwh[1], height),
        box.yaw,
        box.actor_id,
    )


def occupancy_certificate(
    realized_frames: list[dict],
    *,
    actor_id: int,
    ego_by_frame: dict[int, np.ndarray],
    evidence_scene_root: Path,
    grid: GridSpec,
    lower_vertical_margin_m: float,
    static_overlap_fail_voxels: int,
    minimum_known_fraction_for_pass: float,
) -> dict:
    total_voxels = known_voxels = static_overlap = dynamic_overlap = 0
    per_frame = []
    for value, world_box in zip(realized_frames, frames_to_boxes(realized_frames, actor_id=actor_id)):
        frame = int(value["frame_index"])
        T_grid_world = np.linalg.inv(np.asarray(ego_by_frame[frame], dtype=float))
        center = T_grid_world @ np.asarray([*world_box.center, 1.0])
        heading = T_grid_world @ np.asarray([np.cos(world_box.yaw), np.sin(world_box.yaw), 0.0, 0.0])
        grid_box = _trim_box_bottom(
            OrientedBox(
                tuple(center[:3]),
                world_box.dimensions_lwh,
                float(np.arctan2(heading[1], heading[0])),
                actor_id + 1,
            ),
            lower_vertical_margin_m,
        )
        mask = voxelize_oriented_box(grid_box, grid)
        with np.load(evidence_scene_root / f"frame_{frame:03d}.npz", allow_pickle=False) as evidence:
            base = evidence["base_state"]
            layers = _load_sparse_dynamic(evidence)
            other_dynamic = np.zeros(mask.shape, dtype=bool)
            for layer_id, layer in layers.items():
                if layer_id != actor_id + 1:
                    other_dynamic |= layer
            frame_total = int(mask.sum())
            frame_known = int(np.count_nonzero(mask & (base != 0)))
            frame_static = int(np.count_nonzero(mask & (base == 2)))
            frame_dynamic = int(np.count_nonzero(mask & other_dynamic))
        total_voxels += frame_total
        known_voxels += frame_known
        static_overlap += frame_static
        dynamic_overlap += frame_dynamic
        per_frame.append(
            {
                "frame_index": frame,
                "box_voxels": frame_total,
                "known_voxels": frame_known,
                "static_overlap_voxels": frame_static,
                "other_dynamic_overlap_voxels": frame_dynamic,
            }
        )
    known_fraction = known_voxels / total_voxels if total_voxels else 0.0
    failed = static_overlap >= static_overlap_fail_voxels or dynamic_overlap > 0
    if failed:
        verdict, reason = Verdict.FAIL.value, "occupied_evidence_overlap"
    elif known_fraction >= minimum_known_fraction_for_pass:
        verdict, reason = Verdict.PASS.value, "known_evidence_without_overlap"
    else:
        verdict, reason = Verdict.UNKNOWN.value, "insufficient_observation_coverage"
    return {
        "verdict": verdict,
        "reason": reason,
        "total_box_voxels": total_voxels,
        "known_voxels": known_voxels,
        "known_fraction": known_fraction,
        "static_overlap_voxels": static_overlap,
        "other_dynamic_overlap_voxels": dynamic_overlap,
        "per_frame": per_frame,
    }


def clipped_bbox_area_fraction(
    T_world_actor: np.ndarray,
    dimensions_lwh,
    *,
    T_world_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_size_wh,
) -> float:
    corners = box_corners_actor(dimensions_lwh)
    world = (T_world_actor[:3, :3] @ corners.T).T + T_world_actor[:3, 3]
    camera = (np.linalg.inv(T_world_camera) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
    valid = camera[:, 2] > 0.1
    if int(valid.sum()) < 2:
        return 0.0
    pixels_h = (intrinsics @ camera[valid].T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    width, height = (float(v) for v in image_size_wh)
    x0, y0 = np.maximum(pixels.min(0), [0.0, 0.0])
    x1, y1 = np.minimum(pixels.max(0), [width, height])
    return float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0) / (width * height))


def visibility_certificate(
    realized_frames: list[dict],
    *,
    processed_scene_root: Path,
    cameras: list[dict],
    image_size_wh,
    min_bbox_area_fraction: float,
    min_visible_frames: int,
    actor_gaussian_count: int,
) -> dict:
    visible_frames = 0
    max_area_by_frame = []
    for value in realized_frames:
        frame = int(value["frame_index"])
        areas = []
        for camera in cameras:
            index = int(camera["dataset_index"])
            T_world_camera = np.loadtxt(
                processed_scene_root / "extrinsics" / f"{frame:03d}_{index}.txt"
            ).reshape(4, 4)
            intrinsics = drivestudio_intrinsics(
                np.loadtxt(processed_scene_root / "intrinsics" / f"{index}.txt")
            )
            areas.append(
                clipped_bbox_area_fraction(
                    np.asarray(value["T_world_actor"], dtype=float),
                    value["dimensions_lwh"],
                    T_world_camera=T_world_camera,
                    intrinsics=intrinsics,
                    image_size_wh=image_size_wh,
                )
            )
        maximum = max(areas, default=0.0)
        max_area_by_frame.append({"frame_index": frame, "max_bbox_area_fraction": maximum})
        visible_frames += int(maximum >= min_bbox_area_fraction)
    if actor_gaussian_count <= 0:
        verdict, reason = Verdict.UNKNOWN.value, "actor_has_no_gaussian_primitives"
    elif visible_frames < min_visible_frames:
        verdict, reason = Verdict.FAIL.value, "insufficient_projected_visibility"
    else:
        verdict, reason = Verdict.PASS.value, "projected_visibility_and_render_support_present"
    return {
        "verdict": verdict,
        "reason": reason,
        "visible_frame_count": visible_frames,
        "actor_gaussian_count": int(actor_gaussian_count),
        "max_area_by_frame": max_area_by_frame,
    }


def raw_lidar_external_violation(
    source_frames: list[dict],
    realized_frames: list[dict],
    *,
    processed_scene_root: Path,
    ego_by_frame: dict[int, np.ndarray],
    actor_id: int,
    lower_vertical_margin_m: float,
    minimum_points: int,
) -> dict:
    """Independent raw-return check; it does not read occupancy or certificate verdicts."""
    total = 0
    per_frame = []
    source_boxes = frames_to_boxes(source_frames, actor_id=actor_id)
    realized_boxes = frames_to_boxes(realized_frames, actor_id=actor_id)
    for value, source_world, realized_world in zip(realized_frames, source_boxes, realized_boxes):
        frame = int(value["frame_index"])
        raw = np.fromfile(processed_scene_root / "lidar" / f"{frame:03d}.bin", dtype=np.float32)
        width = 4 if raw.size % 4 == 0 else 5 if raw.size % 5 == 0 else 3
        points = raw.reshape(-1, width)[:, :3].astype(np.float64)
        T_grid_world = np.linalg.inv(np.asarray(ego_by_frame[frame], dtype=float))

        def to_grid(box: OrientedBox) -> OrientedBox:
            center = T_grid_world @ np.asarray([*box.center, 1.0])
            heading = T_grid_world @ np.asarray([np.cos(box.yaw), np.sin(box.yaw), 0.0, 0.0])
            return _trim_box_bottom(
                OrientedBox(
                    tuple(center[:3]),
                    box.dimensions_lwh,
                    float(np.arctan2(heading[1], heading[0])),
                    box.actor_id,
                ),
                lower_vertical_margin_m,
            )

        source_grid = to_grid(source_world)
        realized_grid = to_grid(realized_world)
        # Remove returns that belonged to the source actor footprint.
        count = int(np.count_nonzero(realized_grid.contains(points) & ~source_grid.contains(points)))
        total += count
        per_frame.append({"frame_index": frame, "non_source_points_inside": count})
    violation = total >= int(minimum_points)
    return {
        "violation": violation,
        "reason": "raw_lidar_non_source_intrusion" if violation else "no_raw_lidar_intrusion",
        "non_source_points_inside": total,
        "minimum_points": int(minimum_points),
        "per_frame": per_frame,
    }


def select_geometry_audit_frames(
    source_frames: list[dict],
    realized_frames: list[dict],
    visibility: dict,
    *,
    count: int,
) -> list[int]:
    indices = [int(value["frame_index"]) for value in realized_frames]
    if len(indices) <= count:
        return indices
    equal = {
        indices[int(round(value))]
        for value in np.linspace(0, len(indices) - 1, max(count - 2, 1))
    }
    displacement = [
        float(
            np.linalg.norm(
                np.asarray(realized["T_world_actor"])[:3, 3]
                - np.asarray(source["T_world_actor"])[:3, 3]
            )
        )
        for source, realized in zip(source_frames, realized_frames)
    ]
    equal.add(indices[int(np.argmax(displacement))])
    areas = [float(value["max_bbox_area_fraction"]) for value in visibility["max_area_by_frame"]]
    if areas:
        equal.add(indices[int(np.argmax(np.abs(np.asarray(areas) - np.median(areas))))])
    for frame in indices:
        if len(equal) >= count:
            break
        equal.add(frame)
    return sorted(equal)[:count]


def outcome_hash(payload: dict) -> str:
    return canonical_sha256(payload)


"""Low-memory nuScenes Actor-local LiDAR adapter for WorldSim V7 P4."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import ijson
import numpy as np
import torch

from motion_proj.worldsim_v7.av2_canonical_surface import TrackGeometry
from motion_proj.worldsim_v7.av2_four_action_compiler import _compile_actor


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stream_json(path: Path):
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def _rotation(quaternion_wxyz: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_selected_index(
    dataset_root: Path,
    role_scenes: Mapping[str, list[str]],
    allowed_category_prefixes: list[str],
) -> dict[str, Any]:
    """Stream large nuScenes tables and retain only the frozen V7 scenes."""
    metadata_root = dataset_root / "v1.0-trainval"
    scene_rows = _load_json(metadata_root / "scene.json")
    scene_by_name = {str(row["name"]): row for row in scene_rows}
    wanted_scene_tokens = {
        str(scene_by_name[name]["token"]): {"name": name, "role": role}
        for role, names in role_scenes.items()
        for name in names
    }

    samples: dict[str, dict[str, Any]] = {}
    for row in _load_json(metadata_root / "sample.json"):
        scene = wanted_scene_tokens.get(str(row["scene_token"]))
        if scene is None:
            continue
        samples[str(row["token"])] = {
            "sample_token": str(row["token"]),
            "scene_name": str(scene["name"]),
            "role": str(scene["role"]),
            "timestamp_us": int(row["timestamp"]),
        }

    sensors = {
        str(row["token"]): str(row["channel"])
        for row in _load_json(metadata_root / "sensor.json")
    }
    calibrations = {
        str(row["token"]): row
        for row in _load_json(metadata_root / "calibrated_sensor.json")
    }
    lidar_calibrations = {
        token
        for token, row in calibrations.items()
        if sensors[str(row["sensor_token"])] == "LIDAR_TOP"
    }

    frames: dict[str, dict[str, Any]] = {}
    ego_pose_tokens: set[str] = set()
    for row in _stream_json(metadata_root / "sample_data.json"):
        sample_token = str(row["sample_token"])
        if (
            sample_token not in samples
            or not bool(row["is_key_frame"])
            or str(row["calibrated_sensor_token"]) not in lidar_calibrations
        ):
            continue
        frame = {
            **samples[sample_token],
            "lidar_filename": str(row["filename"]),
            "calibrated_sensor_token": str(row["calibrated_sensor_token"]),
            "ego_pose_token": str(row["ego_pose_token"]),
            "annotations": [],
        }
        frames[sample_token] = frame
        ego_pose_tokens.add(str(row["ego_pose_token"]))

    ego_poses = {
        str(row["token"]): row
        for row in _stream_json(metadata_root / "ego_pose.json")
        if str(row["token"]) in ego_pose_tokens
    }
    categories = {
        str(row["token"]): str(row["name"])
        for row in _load_json(metadata_root / "category.json")
    }
    instances = {
        str(row["token"]): categories[str(row["category_token"])]
        for row in _load_json(metadata_root / "instance.json")
    }
    prefixes = tuple(str(value) for value in allowed_category_prefixes)
    for row in _stream_json(metadata_root / "sample_annotation.json"):
        sample_token = str(row["sample_token"])
        frame = frames.get(sample_token)
        if frame is None:
            continue
        category = instances[str(row["instance_token"])]
        if not category.startswith(prefixes):
            continue
        frame["annotations"].append(
            {
                "token": str(row["token"]),
                "instance_token": str(row["instance_token"]),
                "category": category,
                "translation": [float(value) for value in row["translation"]],
                "rotation": [float(value) for value in row["rotation"]],
                "size_wlh": [float(value) for value in row["size"]],
                "num_lidar_pts": int(row["num_lidar_pts"]),
            }
        )

    per_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames.values():
        calibration = calibrations[str(frame["calibrated_sensor_token"])]
        pose = ego_poses[str(frame["ego_pose_token"])]
        frame["sensor_translation_ego"] = [float(value) for value in calibration["translation"]]
        frame["sensor_rotation_ego"] = [float(value) for value in calibration["rotation"]]
        frame["ego_translation_global"] = [float(value) for value in pose["translation"]]
        frame["ego_rotation_global"] = [float(value) for value in pose["rotation"]]
        per_scene[str(frame["scene_name"])].append(frame)
    for scene_frames in per_scene.values():
        scene_frames.sort(key=lambda row: int(row["timestamp_us"]))

    return {
        "schema_version": "worldsim_v7.nuscenes_selected_index.v1",
        "dataset_root": str(dataset_root),
        "role_scenes": {role: list(names) for role, names in role_scenes.items()},
        "scenes": dict(per_scene),
    }


def _make_track(
    instance_token: str,
    rows: list[dict[str, Any]],
    frame_by_sample: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> TrackGeometry | None:
    rows.sort(key=lambda row: int(frame_by_sample[str(row["sample_token"])]["timestamp_us"]))
    if len(rows) < int(config["minimum_track_states"]):
        return None
    timestamps_ns = np.asarray(
        [int(frame_by_sample[str(row["sample_token"])]["timestamp_us"]) * 1000 for row in rows],
        dtype=np.int64,
    )
    global_centers = np.asarray([row["translation"] for row in rows], dtype=np.float64)
    ego_centers = []
    for row, center in zip(rows, global_centers):
        frame = frame_by_sample[str(row["sample_token"])]
        rotation = _rotation(frame["ego_rotation_global"])
        translation = np.asarray(frame["ego_translation_global"], dtype=np.float64)
        ego_centers.append((center - translation) @ rotation)
    ego_centers_array = np.asarray(ego_centers, dtype=np.float64)
    sizes_lwh = np.asarray(
        [[row["size_wlh"][1], row["size_wlh"][0], row["size_wlh"][2]] for row in rows],
        dtype=np.float64,
    )

    hazard = config["hazard"]
    radius = 0.5 * np.linalg.norm(sizes_lwh[:, :2], axis=1)
    clearance = np.maximum(
        np.linalg.norm(ego_centers_array[:, :2], axis=1)
        - radius
        - float(hazard["ego_radius_m"]),
        0.0,
    )
    seconds = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9
    dt = np.diff(seconds)
    valid_dt = dt > 1e-4
    closing = np.zeros_like(dt)
    closing[valid_dt] = np.maximum(-np.diff(clearance)[valid_dt] / dt[valid_dt], 0.0)
    maximum_closing = float(np.max(closing, initial=0.0))
    ttc = np.full_like(dt, float(hazard["maximum_ttc_s"]))
    closing_valid = closing >= float(hazard["minimum_closing_speed_mps"])
    ttc[closing_valid] = clearance[:-1][closing_valid] / np.maximum(closing[closing_valid], 1e-6)
    minimum_ttc = float(
        min(np.min(ttc, initial=float(hazard["maximum_ttc_s"])), float(hazard["maximum_ttc_s"]))
    )
    velocity = np.zeros((0, 3), dtype=np.float64)
    if np.any(valid_dt):
        velocity = np.diff(global_centers, axis=0)[valid_dt] / dt[valid_dt, None]
    speed = np.linalg.norm(velocity[:, :2], axis=1) if len(velocity) else np.empty(0)
    deceleration = 0.0
    if len(speed) >= 2:
        speed_dt = dt[1:][valid_dt[1:]]
        aligned = min(len(speed_dt), len(speed) - 1)
        if aligned:
            acceleration = np.diff(speed[: aligned + 1]) / np.maximum(speed_dt[:aligned], 1e-4)
            deceleration = float(max(-np.min(acceleration, initial=0.0), 0.0))
    hard_brake = float(
        np.clip(deceleration / float(hazard["hard_brake_reference_mps2"]), 0.0, 1.0)
    )
    lateral = ego_centers_array[:, 1]
    forward = ego_centers_array[:, 0]
    crossing = bool(
        float(np.min(lateral)) <= 0.0 <= float(np.max(lateral))
        and float(np.min(np.abs(forward))) <= float(hazard["crossing_forward_window_m"])
    )
    minimum_clearance = float(np.min(clearance, initial=np.inf))
    hazardous = bool(
        minimum_ttc <= float(hazard["hazard_ttc_s"])
        or minimum_clearance <= float(hazard["hazard_clearance_m"])
        or (
            hard_brake >= float(hazard["hazard_hard_brake_score"])
            and minimum_clearance <= float(hazard["hard_brake_interaction_range_m"])
        )
        or (
            crossing
            and minimum_clearance <= float(hazard["crossing_interaction_range_m"])
        )
    )
    return TrackGeometry(
        track_id=instance_token,
        category=str(rows[0]["category"]),
        timestamps_ns=timestamps_ns,
        ego_centers_m=ego_centers_array,
        city_centers_m=global_centers,
        size_lwh_m=np.median(sizes_lwh, axis=0),
        minimum_ttc_s=minimum_ttc,
        minimum_clearance_m=minimum_clearance,
        maximum_closing_speed_mps=maximum_closing,
        hard_brake_score=hard_brake,
        crossing_probability=float(crossing),
        hazardous=hazardous,
    )


def _read_lidar(path: Path) -> np.ndarray:
    scan = np.fromfile(path, dtype=np.float32)
    if len(scan) % 5:
        raise ValueError(f"invalid nuScenes lidar shape: {path}")
    return scan.reshape(-1, 5)[:, :3].copy()


def _associate_frame(
    points_sensor: np.ndarray,
    frame: Mapping[str, Any],
    eligible_tracks: Mapping[str, TrackGeometry],
    frame_ranks: Mapping[str, Mapping[str, int]],
    records: dict[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    sensor_rotation = _rotation(frame["sensor_rotation_ego"])
    sensor_translation = np.asarray(frame["sensor_translation_ego"], dtype=np.float64)
    ego_rotation = _rotation(frame["ego_rotation_global"])
    ego_translation = np.asarray(frame["ego_translation_global"], dtype=np.float64)
    points_ego = points_sensor.astype(np.float64) @ sensor_rotation.T + sensor_translation
    points_global = points_ego @ ego_rotation.T + ego_translation
    sensor_origin_global = sensor_translation @ ego_rotation.T + ego_translation

    annotations = [
        row for row in frame["annotations"] if str(row["instance_token"]) in eligible_tracks
    ]
    if not annotations:
        return
    points = torch.as_tensor(points_global, dtype=torch.float32, device=device)
    batch_size = int(config["actor_batch_size"])
    padding = float(config["box_padding_m"])
    per_frame_limit = int(config["maximum_points_per_actor_frame"])
    with torch.inference_mode():
        for start in range(0, len(annotations), batch_size):
            batch = annotations[start : start + batch_size]
            centers_np = np.asarray([row["translation"] for row in batch], dtype=np.float32)
            sizes_np = np.asarray(
                [[row["size_wlh"][1], row["size_wlh"][0], row["size_wlh"][2]] for row in batch],
                dtype=np.float32,
            )
            rotations_np = np.asarray([_rotation(row["rotation"]) for row in batch], dtype=np.float32)
            centers = torch.as_tensor(centers_np, device=device)
            sizes = torch.as_tensor(sizes_np, device=device)
            rotations = torch.as_tensor(rotations_np, device=device)
            local = torch.matmul(points.unsqueeze(0) - centers[:, None, :], rotations)
            inside = torch.all(torch.abs(local) <= sizes[:, None, :] * 0.5 + padding, dim=-1)
            origins_np = np.einsum(
                "bi,bij->bj",
                sensor_origin_global[None, :] - centers_np,
                rotations_np,
            )
            for offset, annotation in enumerate(batch):
                selected = local[offset, inside[offset]]
                if selected.numel() == 0:
                    continue
                if len(selected) > per_frame_limit:
                    indices = torch.linspace(
                        0, len(selected) - 1, steps=per_frame_limit, device=device
                    ).to(torch.long)
                    selected = selected.index_select(0, indices)
                track_id = str(annotation["instance_token"])
                records[track_id].append(
                    {
                        "timestamp_ns": int(frame["timestamp_us"]) * 1000,
                        "frame_rank": int(frame_ranks[track_id][str(frame["sample_token"])]),
                        "points": selected.cpu().numpy(),
                        "sensor_origin": origins_np[offset].astype(np.float32),
                        "actor_center_ego": np.zeros(3, dtype=np.float32),
                        "actor_rotation_ego": np.eye(3, dtype=np.float32),
                    }
                )


def compile_nuscenes_scene(
    scene_name: str,
    frames: list[dict[str, Any]],
    dataset_root: Path,
    config: Mapping[str, Any],
    compiler_config: Mapping[str, Any],
    device: torch.device,
    include_diagnostics: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    frame_by_sample = {str(frame["sample_token"]): frame for frame in frames}
    annotations_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for row in frame["annotations"]:
            row = {**row, "sample_token": str(frame["sample_token"])}
            annotations_by_instance[str(row["instance_token"])].append(row)
    tracks = {
        track_id: track
        for track_id, rows in annotations_by_instance.items()
        if (track := _make_track(track_id, rows, frame_by_sample, config)) is not None
    }
    frame_ranks = {
        track_id: {
            str(row["sample_token"]): rank
            for rank, row in enumerate(sorted(
                annotations_by_instance[track_id],
                key=lambda item: int(frame_by_sample[str(item["sample_token"])]["timestamp_us"]),
            ))
        }
        for track_id in tracks
    }
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        lidar_path = dataset_root / str(frame["lidar_filename"])
        if not lidar_path.is_file():
            raise FileNotFoundError(lidar_path)
        _associate_frame(
            _read_lidar(lidar_path),
            frame,
            tracks,
            frame_ranks,
            records,
            config,
            device,
        )

    rows = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for track_id in sorted(tracks):
        compiled = _compile_actor(
            tracks[track_id],
            sorted(records.get(track_id, []), key=lambda item: item["frame_rank"]),
            compiler_config,
            device,
            include_diagnostics=include_diagnostics,
        )
        if compiled is None:
            continue
        row, package = compiled
        row["dataset"] = "nuScenes"
        row["scene_name"] = scene_name
        row["role"] = str(frames[0]["role"])
        row["target_supported_repairable"] = bool(
            float(row["after"]["symmetric_chamfer_m"])
            <= float(row["query_only"]["symmetric_chamfer_m"])
        )
        row["clean_query_to_compiled_chamfer_delta_m"] = float(
            row["after"]["symmetric_chamfer_m"]
            - row["query_only"]["symmetric_chamfer_m"]
        )
        rows.append(row)
        if include_diagnostics:
            diagnostics[track_id] = package["diagnostics"]
    if include_diagnostics:
        return rows, diagnostics
    return rows

#!/usr/bin/env python
"""PILOT-3 真实标定、grid/model frame 与 checkpoint pose 审计。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.coordinates import (
    box_corners_actor,
    compose_transform,
    invert_transform,
    project_world_points,
    transform_points,
    validate_transform,
)
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _rotation_angle(T_left, T_right) -> float:
    delta = T_left[:3, :3].T @ T_right[:3, :3]
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _quaternion_wxyz_to_rotation(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 1e-12:
        raise ValueError("checkpoint quaternion 近零")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("world_state config 必须是 object")
    return config


def audit_scene(scene: dict, config: dict, registry_root: Path) -> dict:
    scene_id = scene["scene_id"]
    data_root = Path(config["processed_root"]) / scene_id
    registry_path = registry_root / scene_id / "actor_registry.json"
    with registry_path.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    with (data_root / "instances" / "instances_info.json").open(encoding="utf-8") as handle:
        instances = json.load(handle)
    checkpoint = torch.load(scene["checkpoint"], map_location="cpu")
    rigid = checkpoint["models"]["RigidNodes"]
    checkpoint_trans = rigid["instances_trans"].cpu().numpy()
    checkpoint_quats = rigid["instances_quats"].cpu().numpy()

    T_world_model = validate_transform(
        np.loadtxt(data_root / "extrinsics" / "000_0.txt").reshape(4, 4)
    )
    T_model_world = invert_transform(T_world_model)
    calibration_rows = []
    for camera_id in (0, 1, 2):
        intrinsics = np.loadtxt(data_root / "intrinsics" / f"{camera_id}.txt").reshape(3, 3)
        T_world_camera = validate_transform(
            np.loadtxt(data_root / "extrinsics" / f"000_{camera_id}.txt").reshape(4, 4)
        )
        T_camera_world = invert_transform(T_world_camera)
        test_points = np.asarray(
            [
                T_world_camera[:3, 3] + T_world_camera[:3, 2] * 10.0,
                T_world_camera[:3, 3] + T_world_camera[:3, 2] * 15.0
                + T_world_camera[:3, 0],
            ]
        )
        pixels, depth = project_world_points(
            test_points,
            T_camera_world=T_camera_world,
            intrinsics=intrinsics,
        )
        camera_points = transform_points(T_camera_world, test_points)
        direct = (intrinsics @ camera_points.T).T
        direct = direct[:, :2] / direct[:, 2:3]
        calibration_rows.append(
            {
                "camera_id": camera_id,
                "projection_max_abs_error_px": float(np.max(np.abs(pixels - direct))),
                "minimum_fixture_depth_m": float(np.min(depth)),
            }
        )

    transform_translation_errors = []
    transform_rotation_errors = []
    box_errors = []
    checkpoint_translation_deltas = []
    checkpoint_rotation_deltas = []
    audited_pose_count = 0
    for actor in registry["actors"]:
        true_id = str(actor["true_instance_id"])
        model_index = int(actor["rigid_model_index"])
        annotation = instances[true_id]["frame_annotations"]
        dimensions = actor["canonical_dimensions_lwh"]
        corners_actor = box_corners_actor(dimensions)
        for frame_index, raw_pose in zip(
            annotation["frame_idx"], annotation["obj_to_world"]
        ):
            frame_index = int(frame_index)
            if not 0 <= frame_index < checkpoint_trans.shape[0]:
                continue
            T_world_actor = validate_transform(np.asarray(raw_pose).reshape(4, 4))
            T_model_actor = compose_transform(T_model_world, T_world_actor)
            recovered_world = compose_transform(T_world_model, T_model_actor)
            transform_translation_errors.append(
                float(np.linalg.norm(recovered_world[:3, 3] - T_world_actor[:3, 3]))
            )
            transform_rotation_errors.append(_rotation_angle(recovered_world, T_world_actor))

            T_world_lidar = validate_transform(
                np.loadtxt(data_root / "lidar_pose" / f"{frame_index:03d}.txt").reshape(4, 4)
            )
            T_grid_world = invert_transform(T_world_lidar)
            corners_world = transform_points(T_world_actor, corners_actor)
            corners_grid = transform_points(T_grid_world, corners_world)
            recovered_corners = transform_points(T_world_lidar, corners_grid)
            box_errors.append(float(np.max(np.abs(recovered_corners - corners_world))))

            checkpoint_pose = np.eye(4)
            checkpoint_pose[:3, :3] = _quaternion_wxyz_to_rotation(
                checkpoint_quats[frame_index, model_index]
            )
            checkpoint_pose[:3, 3] = checkpoint_trans[frame_index, model_index]
            checkpoint_translation_deltas.append(
                float(np.linalg.norm(checkpoint_pose[:3, 3] - T_model_actor[:3, 3]))
            )
            checkpoint_rotation_deltas.append(_rotation_angle(checkpoint_pose, T_model_actor))
            audited_pose_count += 1

    return {
        "scene_id": scene_id,
        "cohort_id": scene["cohort_id"],
        "registry_sha256": registry["actor_registry_sha256"],
        "checkpoint_sha256": file_fingerprint(scene["checkpoint"]),
        "coordinate_frames": {
            "annotation": "world",
            "model": "start_CAM_FRONT_sensor_frame",
            "occupancy_grid": "per_frame_LiDAR_sensor_frame",
            "camera_extrinsics": "T_world_camera",
            "lidar_pose": "T_world_lidar",
        },
        "audited_actor_count": registry["actor_count"],
        "audited_pose_count": audited_pose_count,
        "max_world_model_world_translation_error_m": max(
            transform_translation_errors, default=float("inf")
        ),
        "max_world_model_world_rotation_error_rad": max(
            transform_rotation_errors, default=float("inf")
        ),
        "max_world_grid_world_box_error_m": max(box_errors, default=float("inf")),
        "checkpoint_vs_annotation_translation_delta_m": {
            "median": float(np.median(checkpoint_translation_deltas)),
            "max": max(checkpoint_translation_deltas, default=float("inf")),
        },
        "checkpoint_vs_annotation_rotation_delta_rad": {
            "median": float(np.median(checkpoint_rotation_deltas)),
            "max": max(checkpoint_rotation_deltas, default=float("inf")),
        },
        "camera_projection": calibration_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/v71/world_state_v1.yaml"),
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11A/registry_build"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"coordinate audit 已存在，拒绝覆盖: {args.output}")
    config = _load_config(args.config)
    scenes = [audit_scene(scene, config, args.registry_root) for scene in config["scenes"]]
    thresholds = {
        "roundtrip_translation_m": 1e-8,
        "roundtrip_rotation_rad": 1e-7,
        "box_roundtrip_m": 1e-8,
        "projection_px": 1e-8,
    }
    gate = all(
        scene["max_world_model_world_translation_error_m"]
        <= thresholds["roundtrip_translation_m"]
        and scene["max_world_model_world_rotation_error_rad"]
        <= thresholds["roundtrip_rotation_rad"]
        and scene["max_world_grid_world_box_error_m"] <= thresholds["box_roundtrip_m"]
        and all(
            camera["projection_max_abs_error_px"] <= thresholds["projection_px"]
            for camera in scene["camera_projection"]
        )
        for scene in scenes
    )
    summary = {
        "schema_version": 1,
        "task_id": "V7-H1-11A",
        "config_sha256": file_fingerprint(str(args.config)),
        "thresholds": thresholds,
        "coordinate_roundtrip_gate": gate,
        "scenes": scenes,
    }
    atomic_write_json(str(args.output), summary)
    print(json.dumps({"coordinate_roundtrip_gate": gate, "output": str(args.output)}))


if __name__ == "__main__":
    main()

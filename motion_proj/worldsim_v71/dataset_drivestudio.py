"""DriveStudio 10 Hz 资产的 V7.1 纯训练 Actor 适配器。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from motion_proj.worldsim_v7.av2_four_action_compiler import _compile_actor
from motion_proj.worldsim_v7.nuscenes_actor_surface import _associate_frame, _make_track
from motion_proj.worldsim_v71.dataset_nuscenes import RIGID_NUSCENES_PREFIXES


def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    """把右手正交旋转矩阵稳定转换为 ``[w, x, y, z]``。"""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        values = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
            values = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
            values = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
            values = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    values /= max(float(np.linalg.norm(values)), 1.0e-12)
    if values[0] < 0.0:
        values = -values
    return [float(value) for value in values]


def discover_processed_train_scenes(
    roots: Iterable[Path],
    scene_metadata_path: Path,
    excluded_scene_names: Iterable[str],
) -> list[dict[str, Any]]:
    """按root优先级发现纯训练场景，只读取目录与官方scene名称。"""
    scenes = json.loads(scene_metadata_path.read_text(encoding="utf-8"))
    by_index: dict[int, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not path.name.isdigit():
                continue
            if not (path / "instances/instances_info.json").is_file():
                continue
            by_index.setdefault(int(path.name), path)
    excluded = {str(name) for name in excluded_scene_names}
    return [
        {
            "scene_index": index,
            "scene_name": str(scenes[index]["name"]),
            "scene_root": str(path),
        }
        for index, path in sorted(by_index.items())
        if index < len(scenes) and str(scenes[index]["name"]) not in excluded
    ]


def _read_processed_lidar(path: Path, record_width: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if int(record_width) < 3 or len(values) % int(record_width):
        raise ValueError(f"DriveStudio LiDAR宽度不匹配: {path}")
    return values.reshape(-1, int(record_width))[:, :3].copy()


def _scene_frames(
    scene_name: str,
    scene_root: Path,
    *,
    keyframe_stride: int,
    minimum_track_states: int,
) -> list[dict[str, Any]]:
    instances = json.loads(
        (scene_root / "instances/instances_info.json").read_text(encoding="utf-8")
    )
    annotations_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for info in instances.values():
        category = str(info["class_name"])
        if not category.startswith(RIGID_NUSCENES_PREFIXES):
            continue
        annotations = info["frame_annotations"]
        selected: list[tuple[int, int]] = []
        for position, frame_value in enumerate(annotations["frame_idx"]):
            frame = int(frame_value)
            if frame % int(keyframe_stride):
                continue
            if not (scene_root / f"lidar/{frame:03d}.bin").is_file():
                continue
            if not (scene_root / f"lidar_pose/{frame:03d}.txt").is_file():
                continue
            selected.append((position, frame))
        if len(selected) < int(minimum_track_states):
            continue
        track_id = str(info.get("id") or info.get("token") or "")
        if not track_id:
            raise ValueError(f"{scene_root} 的Actor缺少稳定id")
        for position, frame in selected:
            object_world = np.asarray(
                annotations["obj_to_world"][position], dtype=np.float64
            ).reshape(4, 4)
            size_lwh = np.asarray(annotations["box_size"][position], dtype=np.float64)
            annotations_by_frame[frame].append(
                {
                    "instance_token": track_id,
                    "category": category,
                    "translation": object_world[:3, 3].tolist(),
                    "rotation": _rotation_to_quaternion_wxyz(object_world[:3, :3]),
                    "size_wlh": [float(size_lwh[1]), float(size_lwh[0]), float(size_lwh[2])],
                    "num_lidar_pts": 0,
                }
            )

    frames: list[dict[str, Any]] = []
    for frame, annotations in sorted(annotations_by_frame.items()):
        lidar_world = np.loadtxt(scene_root / f"lidar_pose/{frame:03d}.txt").astype(np.float64)
        frames.append(
            {
                "sample_token": f"{scene_name}:{frame:03d}",
                "scene_name": scene_name,
                "role": "train_processed_recovery",
                "timestamp_us": int(frame * 100_000),
                "lidar_path": str(scene_root / f"lidar/{frame:03d}.bin"),
                "sensor_translation_ego": [0.0, 0.0, 0.0],
                "sensor_rotation_ego": [1.0, 0.0, 0.0, 0.0],
                "ego_translation_global": lidar_world[:3, 3].tolist(),
                "ego_rotation_global": _rotation_to_quaternion_wxyz(lidar_world[:3, :3]),
                "annotations": annotations,
            }
        )
    return frames


def compile_processed_scene(
    scene_name: str,
    scene_root: Path,
    actor_config: Mapping[str, Any],
    compiler_config: Mapping[str, Any],
    device: torch.device,
    *,
    keyframe_stride: int = 5,
    lidar_record_width: int = 4,
) -> list[dict[str, Any]]:
    """以2 Hz关键帧重建同一V7.1 build/query/target Actor合同。"""
    frames = _scene_frames(
        scene_name,
        scene_root,
        keyframe_stride=int(keyframe_stride),
        minimum_track_states=int(actor_config["minimum_track_states"]),
    )
    frame_by_sample = {str(frame["sample_token"]): frame for frame in frames}
    annotations_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for annotation in frame["annotations"]:
            annotations_by_instance[str(annotation["instance_token"])].append(
                {**annotation, "sample_token": str(frame["sample_token"])}
            )
    tracks = {
        track_id: track
        for track_id, rows in annotations_by_instance.items()
        if (track := _make_track(track_id, rows, frame_by_sample, actor_config)) is not None
    }
    frame_ranks = {
        track_id: {
            str(row["sample_token"]): rank
            for rank, row in enumerate(
                sorted(
                    annotations_by_instance[track_id],
                    key=lambda item: int(frame_by_sample[str(item["sample_token"])]["timestamp_us"]),
                )
            )
        }
        for track_id in tracks
    }
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        _associate_frame(
            _read_processed_lidar(Path(frame["lidar_path"]), int(lidar_record_width)),
            frame,
            tracks,
            frame_ranks,
            records,
            actor_config,
            device,
        )

    bundles: list[dict[str, Any]] = []
    for track_id in sorted(tracks):
        compiled = _compile_actor(
            tracks[track_id],
            sorted(records.get(track_id, []), key=lambda item: item["frame_rank"]),
            compiler_config,
            device,
            include_diagnostics=True,
        )
        if compiled is None:
            continue
        row, package = compiled
        row.update(
            {
                "dataset": "nuScenes-DriveStudio",
                "scene_name": scene_name,
                "role": "train_processed_recovery",
                "source_cadence_hz": 2.0,
            }
        )
        bundles.append(
            {"scene_name": scene_name, "row": row, "diagnostics": package["diagnostics"]}
        )
    return bundles

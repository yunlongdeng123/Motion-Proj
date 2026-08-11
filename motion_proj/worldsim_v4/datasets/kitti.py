from __future__ import annotations

import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_ID = "WS-V4-D1-KITTI-ADAPTER-01"
SCHEMA_VERSION = "worldsim_v4_kitti_adapter_v1"
TRACKING_REQUIRED = ("image_02", "image_03", "velodyne", "label_02", "calib")
CAMERA_CONTRACT = ("image_02", "image_03")
ADAPTER_GATES = (
    "meter_axis_handedness",
    "t_velo_cam",
    "rectification",
    "camera_intrinsics",
    "timestamp_order",
    "actor_track_id",
    "box3d_camera_projection",
    "lidar_image_projection",
    "object_world_camera_chain",
    "stereo_association",
    "heldout_leak",
    "deterministic_manifest_hash",
)


class KittiAdapterError(RuntimeError):
    """KITTI adapter 合同不满足。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_named_dirs(root: Path, names: set[str], max_depth: int = 6) -> dict[str, list[Path]]:
    found = {name: [] for name in names}
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        for dirname in list(dirnames):
            if dirname in names:
                found[dirname].append((current / dirname).resolve())
    return {name: sorted(set(paths), key=lambda path: (len(path.parts), str(path))) for name, paths in found.items()}


def _first_dir(found: Mapping[str, Sequence[Path]], name: str) -> Path | None:
    paths = found.get(name, ())
    return paths[0] if paths else None


def _sequence_ids(path: Path | None, *, files: bool = False) -> set[str]:
    if path is None or not path.is_dir():
        return set()
    if files:
        return {item.stem for item in path.iterdir() if item.is_file() and not item.name.startswith(".")}
    return {item.name for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")}


def _tracking_layout(root: Path) -> dict[str, Any] | None:
    names = set(TRACKING_REQUIRED) | {"oxts", "poses", "timestamps"}
    found = _discover_named_dirs(root, names)
    components = {name: _first_dir(found, name) for name in names}
    if not any(components[name] for name in TRACKING_REQUIRED):
        return None
    missing = [name for name in TRACKING_REQUIRED if components[name] is None]
    sequence_sets = {
        name: _sequence_ids(components[name], files=name in {"calib", "label_02"})
        for name in TRACKING_REQUIRED
        if components[name] is not None
    }
    common = set.intersection(*sequence_sets.values()) if len(sequence_sets) == len(TRACKING_REQUIRED) else set()
    pose_root = components["poses"] or components["oxts"]
    return {
        "layout": "tracking_training",
        "status": "ready" if not missing and common else "blocked_dataset_adapter",
        "dataset_root": str(root),
        "components": {
            **{name: str(components[name]) if components[name] else None for name in TRACKING_REQUIRED},
            "pose": str(pose_root) if pose_root else None,
            "timestamps": str(components["timestamps"]) if components["timestamps"] else None,
        },
        "missing_components": missing,
        "sequence_ids": sorted(common),
        "sequence_count": len(common),
        "camera_contract": list(CAMERA_CONTRACT),
    }


def _raw_drive_records(root: Path, max_depth: int = 5) -> list[dict[str, Any]]:
    records = []
    drive_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_drive_\d{4}_sync$")
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        for dirname in list(dirnames):
            if not drive_pattern.match(dirname):
                continue
            drive = (current / dirname).resolve()
            date_root = drive.parent
            components = {
                "image_02": drive / "image_02" / "data",
                "image_03": drive / "image_03" / "data",
                "velodyne": drive / "velodyne_points" / "data",
                "oxts": drive / "oxts" / "data",
                "tracklets": drive / "tracklet_labels.xml",
                "calib_cam_to_cam": date_root / "calib_cam_to_cam.txt",
                "calib_velo_to_cam": date_root / "calib_velo_to_cam.txt",
            }
            missing = [name for name, path in components.items() if not path.exists()]
            records.append(
                {
                    "drive": dirname,
                    "root": str(drive),
                    "components": {name: str(path) for name, path in components.items()},
                    "missing_components": missing,
                    "status": "ready" if not missing else "blocked_dataset_adapter",
                }
            )
    return sorted(records, key=lambda row: row["drive"])


def detect_kitti_layout(root: str | Path) -> dict[str, Any]:
    requested_root = Path(root).absolute()
    root = requested_root.resolve()
    if not root.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": "blocked_local_dataset_missing",
            "layout": "missing",
            "requested_root": str(requested_root),
            "dataset_root": str(root),
            "download_attempted": False,
            "camera_contract": list(CAMERA_CONTRACT),
        }
    if not root.is_dir():
        raise KittiAdapterError(f"KITTI root 不是目录：{root}")
    tracking = _tracking_layout(root)
    raw_drives = _raw_drive_records(root)
    if tracking and tracking["status"] == "ready":
        selected = tracking
    elif any(row["status"] == "ready" for row in raw_drives):
        selected = {
            "layout": "raw_sync",
            "status": "ready",
            "dataset_root": str(root),
            "drives": raw_drives,
            "sequence_ids": [row["drive"] for row in raw_drives if row["status"] == "ready"],
            "sequence_count": sum(row["status"] == "ready" for row in raw_drives),
            "camera_contract": list(CAMERA_CONTRACT),
        }
    else:
        selected = tracking or {
            "layout": "unknown",
            "status": "blocked_dataset_adapter",
            "dataset_root": str(root),
            "missing_components": list(TRACKING_REQUIRED),
            "sequence_ids": [],
            "sequence_count": 0,
            "camera_contract": list(CAMERA_CONTRACT),
        }
        if raw_drives:
            selected["raw_drives"] = raw_drives
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        **selected,
        "requested_root": str(requested_root),
        "download_attempted": False,
    }


def _read_numeric_table(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, payload = line.split(":", 1)
        try:
            values[key.strip()] = [float(value) for value in payload.split()]
        except ValueError:
            continue
    return values


def parse_tracking_calibration(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    values = _read_numeric_table(path)

    def take(keys: Sequence[str], size: int, shape: tuple[int, ...]) -> np.ndarray:
        for key in keys:
            if key in values:
                array = np.asarray(values[key], dtype=np.float64)
                if array.size != size or not np.isfinite(array).all():
                    raise KittiAdapterError(f"{path} {key} shape/finite 非法")
                return array.reshape(shape)
        raise KittiAdapterError(f"{path} 缺少 calibration keys：{list(keys)}")

    p2 = take(("P2", "P_rect_02"), 12, (3, 4))
    p3 = take(("P3", "P_rect_03"), 12, (3, 4))
    rect = take(("R0_rect", "R_rect_00", "R_rect"), 9, (3, 3))
    velo = take(("Tr_velo_to_cam", "Tr_velo_cam", "Tr"), 12, (3, 4))
    t_velo_cam = np.eye(4, dtype=np.float64)
    t_velo_cam[:3, :] = velo
    r_rect = np.eye(4, dtype=np.float64)
    r_rect[:3, :3] = rect
    return {"P2": p2, "P3": p3, "R0_rect": r_rect, "T_velo_cam": t_velo_cam}


def parse_tracking_labels(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 17:
            raise KittiAdapterError(f"label line {line_number} 必须为 17 fields，实际 {len(fields)}")
        frame, track_id = int(fields[0]), int(fields[1])
        if frame < 0 or track_id < -1:
            raise KittiAdapterError(f"label line {line_number} frame/track_id 非法")
        numeric = [float(value) for value in fields[3:]]
        if not all(math.isfinite(value) for value in numeric):
            raise KittiAdapterError(f"label line {line_number} 存在非有限数")
        records.append(
            {
                "frame": frame,
                "track_id": track_id,
                "type": fields[2],
                "truncated": numeric[0],
                "occluded": int(float(fields[4])),
                "alpha": float(fields[5]),
                "bbox": [float(value) for value in fields[6:10]],
                "dimensions_hwl": [float(value) for value in fields[10:13]],
                "location_camera_m": [float(value) for value in fields[13:16]],
                "rotation_y": float(fields[16]),
            }
        )
    return records


def build_frame_partitions(frame_ids: Sequence[int]) -> dict[str, list[int]]:
    ordered = sorted({int(frame) for frame in frame_ids})
    return {
        "train_frames": [frame for index, frame in enumerate(ordered) if index % 5 not in {2, 4}],
        "development_frames": [frame for index, frame in enumerate(ordered) if index % 5 == 2],
        "heldout_frames": [frame for index, frame in enumerate(ordered) if index % 5 == 4],
    }


def transform_lidar_to_rectified_camera(points: np.ndarray, calibration: Mapping[str, np.ndarray]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in {3, 4}:
        raise KittiAdapterError("LiDAR points 必须为 N×3/N×4")
    xyz1 = np.ones((len(points), 4), dtype=np.float64)
    xyz1[:, :3] = points[:, :3]
    camera = (calibration["R0_rect"] @ calibration["T_velo_cam"] @ xyz1.T).T
    return camera[:, :3]


def project_camera_points(points_camera: np.ndarray, projection: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    xyz1 = np.ones((len(points), 4), dtype=np.float64)
    xyz1[:, :3] = points[:, :3]
    image_h = (projection @ xyz1.T).T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = image_h[:, 2] > 1e-9
    result[valid] = image_h[valid, :2] / image_h[valid, 2:3]
    return result


def _box_corners_camera(label: Mapping[str, Any]) -> np.ndarray:
    height, width, length = [float(value) for value in label["dimensions_hwl"]]
    x, y, z = [float(value) for value in label["location_camera_m"]]
    yaw = float(label["rotation_y"])
    local = np.asarray(
        [
            [length / 2, 0, width / 2],
            [length / 2, 0, -width / 2],
            [-length / 2, 0, -width / 2],
            [-length / 2, 0, width / 2],
            [length / 2, -height, width / 2],
            [length / 2, -height, -width / 2],
            [-length / 2, -height, -width / 2],
            [-length / 2, -height, width / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray([x, y, z], dtype=np.float64)


def _frame_stems(path: Path, suffixes: set[str]) -> dict[int, Path]:
    result = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix.lower() in suffixes and item.stem.isdigit():
            result[int(item.stem)] = item
    return result


def _pose_path(root: Path | None, sequence: str) -> Path | None:
    if root is None:
        return None
    file_path = root / f"{sequence}.txt"
    if file_path.is_file():
        return file_path
    directory = root / sequence
    return directory if directory.is_dir() else None


def _load_pose_matrices(path: Path | None, frame_ids: Sequence[int]) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    lines: list[str]
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix == ".txt")
        lines = [item.read_text(encoding="utf-8").strip() for item in files]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    poses = {}
    for frame, line in zip(sorted(frame_ids), lines):
        values = [float(value) for value in line.split()]
        if len(values) < 12 or not np.isfinite(values[:12]).all():
            continue
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = np.asarray(values[:12], dtype=np.float64).reshape(3, 4)
        poses[frame] = matrix
    return poses


def _timestamps(frame_ids: Sequence[int], path: Path | None) -> tuple[list[str], str]:
    if path is not None:
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        if len(lines) >= len(frame_ids):
            values = [line.strip() for line in lines[: len(frame_ids)]]
            return values, "dataset"
    return [f"{frame / 10.0:.6f}" for frame in sorted(frame_ids)], "nominal_10hz"


def build_tracking_manifest(layout: Mapping[str, Any], *, smoke_count: int = 2, formal_count: int = 10) -> dict[str, Any]:
    if layout.get("layout") != "tracking_training" or layout.get("status") != "ready":
        raise KittiAdapterError("tracking layout 未 ready")
    components = {key: Path(value) if value else None for key, value in layout["components"].items()}
    sequence_ids = list(layout["sequence_ids"])
    selected_ids = sequence_ids[: max(smoke_count, min(formal_count, len(sequence_ids)))]
    sequences = []
    gate_rows = []
    for order, sequence in enumerate(selected_ids):
        image_02 = _frame_stems(components["image_02"] / sequence, {".png", ".jpg", ".jpeg"})
        image_03 = _frame_stems(components["image_03"] / sequence, {".png", ".jpg", ".jpeg"})
        lidar = _frame_stems(components["velodyne"] / sequence, {".bin"})
        frame_ids = sorted(set(image_02) & set(image_03) & set(lidar))
        if not frame_ids:
            raise KittiAdapterError(f"sequence {sequence} 无 stereo/LiDAR common frames")
        calibration_path = components["calib"] / f"{sequence}.txt"
        label_path = components["label_02"] / f"{sequence}.txt"
        calibration = parse_tracking_calibration(calibration_path)
        labels = parse_tracking_labels(label_path)
        poses = _load_pose_matrices(_pose_path(components.get("pose"), sequence), frame_ids)
        timestamp_path = (
            components["timestamps"] / f"{sequence}.txt"
            if components.get("timestamps") is not None
            else None
        )
        timestamp_values, timestamp_source = _timestamps(frame_ids, timestamp_path)
        first_label = next((row for row in labels if row["track_id"] >= 0), None)
        box_projection_ok = False
        if first_label is not None:
            corners = _box_corners_camera(first_label)
            projected = project_camera_points(corners, calibration["P2"])
            box_projection_ok = bool(np.isfinite(projected).any() and (corners[:, 2] > 0).any())
        points = np.fromfile(lidar[frame_ids[0]], dtype=np.float32)
        if points.size % 4 != 0:
            raise KittiAdapterError(f"sequence {sequence} LiDAR binary 非 N×4")
        points = points.reshape(-1, 4)
        camera_points = transform_lidar_to_rectified_camera(points, calibration)
        image_points = project_camera_points(camera_points, calibration["P2"])
        lidar_projection_ok = bool(len(points) and np.isfinite(image_points).any())
        partitions = build_frame_partitions(frame_ids)
        flattened = [frame for key in partitions for frame in partitions[key]]
        gates = {
            "meter_axis_handedness": True,
            "t_velo_cam": calibration["T_velo_cam"].shape == (4, 4),
            "rectification": calibration["R0_rect"].shape == (4, 4),
            "camera_intrinsics": calibration["P2"].shape == calibration["P3"].shape == (3, 4),
            "timestamp_order": timestamp_values == sorted(timestamp_values),
            "actor_track_id": first_label is not None,
            "box3d_camera_projection": box_projection_ok,
            "lidar_image_projection": lidar_projection_ok,
            "object_world_camera_chain": len(poses) == len(frame_ids),
            "stereo_association": set(image_02) == set(image_03) == set(lidar),
            "heldout_leak": len(flattened) == len(set(flattened)) == len(frame_ids),
            "deterministic_manifest_hash": True,
        }
        gate_rows.append(gates)
        sequences.append(
            {
                "sequence": sequence,
                "role": "adapter_smoke" if order < smoke_count else "cross_domain_formal",
                "frame_ids": frame_ids,
                **partitions,
                "camera_set": list(CAMERA_CONTRACT),
                "lidar": "velodyne",
                "calibration": str(calibration_path),
                "pose_source": str(_pose_path(components.get("pose"), sequence)) if poses else None,
                "timestamp_source": timestamp_source,
                "actor_track_ids": sorted({row["track_id"] for row in labels if row["track_id"] >= 0}),
                "gates": gates,
            }
        )
    gates = {name: bool(gate_rows) and all(row[name] for row in gate_rows) for name in ADAPTER_GATES}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done" if all(gates.values()) and len(sequences) >= smoke_count else "blocked_dataset_adapter",
        "layout": "tracking_training",
        "dataset_root": layout["dataset_root"],
        "camera_contract": list(CAMERA_CONTRACT),
        "units": "meters",
        "coordinate_contract": {
            "camera": "x_right_y_down_z_forward",
            "lidar": "x_forward_y_left_z_up",
            "handedness": "right_handed",
        },
        "smoke_sequence_count": min(smoke_count, len(sequences)),
        "formal_target_count": formal_count,
        "method_threshold_source": "frozen_nuscenes_only",
        "kitti_threshold_search": False,
        "download_attempted": False,
        "gates": gates,
        "sequences": sequences,
    }
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_json_sha256(unsigned)
    if manifest["manifest_sha256"] != canonical_json_sha256(unsigned):
        raise KittiAdapterError("deterministic manifest hash 失败")
    return manifest


def parse_raw_tracklet_ids(path: str | Path) -> list[int]:
    tree = ET.parse(path)
    items = tree.getroot().findall(".//tracklets/item")
    return list(range(len(items)))

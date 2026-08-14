"""KITTI Tracking V5 calibration、OXTS pose 与缺帧 denominator 合同。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EARTH_RADIUS_M = 6378137.0


class KittiV5AdapterError(RuntimeError):
    pass


def _read_numeric_table(path: str | Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, payload = line.split(":", 1)
        else:
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                continue
            key, payload = fields
        try:
            row = [float(value) for value in payload.split()]
        except ValueError:
            continue
        key = key.strip()
        if not key or key in values:
            raise KittiV5AdapterError(
                f"{path}:{line_number} calibration key 重复或为空: {key!r}"
            )
        values[key] = row
    return values


def _take_matrix(
    values: dict[str, list[float]],
    path: Path,
    keys: Sequence[str],
    shape: tuple[int, int],
) -> np.ndarray:
    size = shape[0] * shape[1]
    for key in keys:
        if key not in values:
            continue
        matrix = np.asarray(values[key], dtype=np.float64)
        if matrix.size != size or not np.isfinite(matrix).all():
            raise KittiV5AdapterError(f"{path} {key} shape/finite 非法")
        return matrix.reshape(shape)
    raise KittiV5AdapterError(f"{path} 缺少 calibration keys: {list(keys)}")


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[: matrix.shape[0], : matrix.shape[1]] = matrix
    return result


def _validate_rotation(name: str, rotation: np.ndarray, tolerance: float = 5e-3) -> None:
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise KittiV5AdapterError(f"{name} rotation 非正交")
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > tolerance:
        raise KittiV5AdapterError(f"{name} rotation handedness 非法: det={determinant}")


def parse_tracking_calibration(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    values = _read_numeric_table(source)
    p2 = _take_matrix(values, source, ("P2", "P_rect_02"), (3, 4))
    p3 = _take_matrix(values, source, ("P3", "P_rect_03"), (3, 4))
    rect = _take_matrix(values, source, ("R0_rect", "R_rect_00", "R_rect"), (3, 3))
    velo = _take_matrix(
        values, source, ("Tr_velo_to_cam", "Tr_velo_cam", "Tr"), (3, 4)
    )
    imu = _take_matrix(
        values, source, ("Tr_imu_to_velo", "Tr_imu_velo"), (3, 4)
    )
    _validate_rotation("R_rect", rect)
    _validate_rotation("Tr_velo_cam", velo[:, :3])
    _validate_rotation("Tr_imu_velo", imu[:, :3])
    if p2[0, 0] <= 0 or p2[1, 1] <= 0 or p3[0, 0] <= 0 or p3[1, 1] <= 0:
        raise KittiV5AdapterError("camera focal length 必须为正")
    return {
        "P2": p2,
        "P3": p3,
        "R_rect": _homogeneous(rect),
        "T_velo_cam": _homogeneous(velo),
        "T_imu_velo": _homogeneous(imu),
    }


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _mercator(latitude_deg: float, longitude_deg: float, scale: float) -> tuple[float, float]:
    x = scale * longitude_deg * math.pi * EARTH_RADIUS_M / 180.0
    y = scale * EARTH_RADIUS_M * math.log(
        math.tan((90.0 + latitude_deg) * math.pi / 360.0)
    )
    return x, y


def parse_tracking_oxts(path: str | Path) -> dict[int, np.ndarray]:
    rows = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            values = np.asarray([float(value) for value in raw_line.split()])
        except ValueError as error:
            raise KittiV5AdapterError(f"OXTS line {line_number} 非数字") from error
        if values.shape != (30,) or not np.isfinite(values).all():
            raise KittiV5AdapterError(
                f"OXTS line {line_number} 必须为 30 个有限字段，实际 {values.size}"
            )
        rows.append(values)
    if not rows:
        raise KittiV5AdapterError("OXTS 为空")
    scale = math.cos(math.radians(float(rows[0][0])))
    absolute = []
    for values in rows:
        x, y = _mercator(float(values[0]), float(values[1]), scale)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _rotation_from_rpy(
            float(values[3]), float(values[4]), float(values[5])
        )
        pose[:3, 3] = [x, y, float(values[2])]
        absolute.append(pose)
    origin_inverse = np.linalg.inv(absolute[0])
    relative = {index: origin_inverse @ pose for index, pose in enumerate(absolute)}
    if not np.allclose(relative[0], np.eye(4), atol=1e-8):
        raise KittiV5AdapterError("OXTS origin normalization 失败")
    return relative


def world_from_rectified_camera(
    world_from_imu: np.ndarray, calibration: dict[str, np.ndarray]
) -> np.ndarray:
    imu_to_rectified_camera = (
        calibration["R_rect"]
        @ calibration["T_velo_cam"]
        @ calibration["T_imu_velo"]
    )
    result = np.asarray(world_from_imu, dtype=np.float64) @ np.linalg.inv(
        imu_to_rectified_camera
    )
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise KittiV5AdapterError("world_from_camera 非法")
    return result


def transform_lidar_to_rectified_camera(
    points: np.ndarray, calibration: dict[str, np.ndarray]
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in {3, 4}:
        raise KittiV5AdapterError("LiDAR points 必须为 N×3/N×4")
    homogeneous = np.ones((len(values), 4), dtype=np.float64)
    homogeneous[:, :3] = values[:, :3]
    camera = (
        calibration["R_rect"]
        @ calibration["T_velo_cam"]
        @ homogeneous.T
    ).T
    return camera[:, :3]


def project_camera_points(
    points_camera: np.ndarray, projection: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise KittiV5AdapterError("camera points 必须为 N×3")
    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise KittiV5AdapterError("projection 必须为 3×4")
    homogeneous = np.ones((len(points), 4), dtype=np.float64)
    homogeneous[:, :3] = points
    image_h = (matrix @ homogeneous.T).T
    valid = np.isfinite(image_h).all(axis=1) & (image_h[:, 2] > 1e-9)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid] = image_h[valid, :2] / image_h[valid, 2:3]
    return pixels, valid


def parse_tracking_labels(path: str | Path) -> list[dict[str, object]]:
    records = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 17:
            raise KittiV5AdapterError(
                f"label line {line_number} 必须为 17 fields，实际 {len(fields)}"
            )
        frame, track_id = int(fields[0]), int(fields[1])
        numeric = np.asarray([float(value) for value in fields[3:]], dtype=np.float64)
        if frame < 0 or track_id < -1 or not np.isfinite(numeric).all():
            raise KittiV5AdapterError(f"label line {line_number} identity/numeric 非法")
        records.append(
            {
                "frame": frame,
                "track_id": track_id,
                "type": fields[2],
                "truncated": float(fields[3]),
                "occluded": int(fields[4]),
                "alpha": float(fields[5]),
                "bbox": [float(value) for value in fields[6:10]],
                "dimensions_hwl": [float(value) for value in fields[10:13]],
                "location_camera_m": [float(value) for value in fields[13:16]],
                "rotation_y": float(fields[16]),
            }
        )
    if not records:
        raise KittiV5AdapterError("tracking labels 为空")
    return records


def freeze_sensor_frame_policy(
    image_02_frames: Iterable[int],
    image_03_frames: Iterable[int],
    lidar_frames: Iterable[int],
) -> dict[str, object]:
    left = {int(value) for value in image_02_frames}
    right = {int(value) for value in image_03_frames}
    lidar = {int(value) for value in lidar_frames}
    stereo = left & right
    evaluable = stereo & lidar
    lidar_abstain = stereo - lidar
    camera_abstain = lidar - stereo
    stereo_unpaired = left ^ right
    denominator = sorted(left | right | lidar)
    if not denominator:
        raise KittiV5AdapterError("sensor frame denominator 为空")
    return {
        "policy": "common_frame_with_explicit_sensor_abstain",
        "silent_intersection_forbidden": True,
        "image_02_count": len(left),
        "image_03_count": len(right),
        "lidar_count": len(lidar),
        "full_denominator_count": len(denominator),
        "stereo_common_count": len(stereo),
        "evaluable_multimodal_count": len(evaluable),
        "stereo_unpaired_abstain_frames": sorted(stereo_unpaired),
        "lidar_missing_abstain_frames": sorted(lidar_abstain),
        "stereo_missing_abstain_frames": sorted(camera_abstain),
        "multimodal_coverage": len(evaluable) / len(denominator),
        "complete_alignment": left == right == lidar,
    }

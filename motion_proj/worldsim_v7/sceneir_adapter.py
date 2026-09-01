"""nuScenes/AV2 共用的轻量 SceneIR 与 AV2 元数据适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SE3:
    """将 source 坐标中的点变换到 target 坐标。"""

    rotation: np.ndarray
    translation_m: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("SE3 requires a 3x3 rotation and a 3-vector translation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999999:
            raise ValueError("rotation must be right-handed")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    @classmethod
    def from_quaternion_translation(
        cls, quaternion_wxyz: np.ndarray, translation_m: np.ndarray
    ) -> "SE3":
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        if quaternion.shape != (4,):
            raise ValueError("quaternion must be [w, x, y, z]")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("quaternion must be non-zero")
        w, x, y, z = quaternion / norm
        rotation = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return cls(rotation, np.asarray(translation_m, dtype=np.float64))

    def inverse(self) -> "SE3":
        rotation = self.rotation.T
        return SE3(rotation, -(rotation @ self.translation_m))

    def compose(self, source: "SE3") -> "SE3":
        return SE3(
            self.rotation @ source.rotation,
            self.rotation @ source.translation_m + self.translation_m,
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError("points must end in XYZ coordinates")
        return values @ self.rotation.T + self.translation_m


@dataclass(frozen=True)
class TimedPose:
    timestamp_ns: int
    city_se3_egovehicle: SE3


@dataclass(frozen=True)
class SensorCalibration:
    sensor_name: str
    egovehicle_se3_sensor: SE3


@dataclass(frozen=True)
class ActorStateSample:
    timestamp_ns: int
    city_se3_actor: SE3
    center_ego_m: tuple[float, float, float]
    size_lwh_m: tuple[float, float, float]
    num_interior_points: int


@dataclass(frozen=True)
class ActorTrack:
    track_id: str
    category: str
    states: tuple[ActorStateSample, ...]


@dataclass(frozen=True)
class SceneIR:
    log_id: str
    source_dataset: str
    coordinate_frame: str
    ego_poses: tuple[TimedPose, ...]
    sensor_calibrations: tuple[SensorCalibration, ...]
    actors: tuple[ActorTrack, ...]


class AV2SceneIRAdapter:
    """逐 log 读取 AV2 feather 元数据，不加载图像或整段 LiDAR。"""

    required_paths = (
        "annotations.feather",
        "city_SE3_egovehicle.feather",
        "calibration/egovehicle_SE3_sensor.feather",
        "calibration/intrinsics.feather",
    )

    def __init__(self, maximum_pose_delta_ns: int = 50_000_000) -> None:
        self.maximum_pose_delta_ns = int(maximum_pose_delta_ns)

    def validate_log(self, log_dir: Path) -> None:
        missing = [relative for relative in self.required_paths if not (log_dir / relative).is_file()]
        lidar_dir = log_dir / "sensors" / "lidar"
        if not lidar_dir.is_dir():
            missing.append("sensors/lidar/")
        if missing:
            raise FileNotFoundError(f"AV2 log {log_dir.name} is incomplete: {missing}")

    @staticmethod
    def _read_feather(path: Path):
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("AV2 preprocessing requires pandas and pyarrow") from error
        return pd.read_feather(path)

    @staticmethod
    def _row_se3(row: Any) -> SE3:
        return SE3.from_quaternion_translation(
            np.asarray([row.qw, row.qx, row.qy, row.qz], dtype=np.float64),
            np.asarray([row.tx_m, row.ty_m, row.tz_m], dtype=np.float64),
        )

    def build_scene_ir(self, log_dir: Path) -> SceneIR:
        log_dir = Path(log_dir)
        self.validate_log(log_dir)
        poses_frame = self._read_feather(log_dir / "city_SE3_egovehicle.feather")
        calibration_frame = self._read_feather(
            log_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        )
        annotations_frame = self._read_feather(log_dir / "annotations.feather")

        poses = tuple(
            TimedPose(int(row.timestamp_ns), self._row_se3(row))
            for row in poses_frame.sort_values("timestamp_ns").itertuples(index=False)
        )
        pose_timestamps = np.asarray([pose.timestamp_ns for pose in poses], dtype=np.int64)
        calibrations = tuple(
            SensorCalibration(str(row.sensor_name), self._row_se3(row))
            for row in calibration_frame.sort_values("sensor_name").itertuples(index=False)
        )

        actor_tracks = []
        grouped = annotations_frame.sort_values(["track_uuid", "timestamp_ns"]).groupby(
            "track_uuid", sort=True
        )
        for track_id, track_frame in grouped:
            categories = tuple(sorted(set(str(value) for value in track_frame["category"])))
            if len(categories) != 1:
                raise ValueError(f"AV2 track {track_id} changes category: {categories}")
            states = []
            for row in track_frame.itertuples(index=False):
                timestamp = int(row.timestamp_ns)
                pose_index = self._nearest_pose_index(pose_timestamps, timestamp)
                pose_delta = abs(int(pose_timestamps[pose_index]) - timestamp)
                if pose_delta > self.maximum_pose_delta_ns:
                    raise ValueError(
                        f"AV2 track {track_id} has no ego pose near timestamp {timestamp}"
                    )
                city_se3_actor = self._row_se3(row)
                ego_se3_city = poses[pose_index].city_se3_egovehicle.inverse()
                center_ego = ego_se3_city.transform_points(city_se3_actor.translation_m)
                states.append(
                    ActorStateSample(
                        timestamp_ns=timestamp,
                        city_se3_actor=city_se3_actor,
                        center_ego_m=tuple(float(value) for value in center_ego),
                        size_lwh_m=(float(row.length_m), float(row.width_m), float(row.height_m)),
                        num_interior_points=int(row.num_interior_pts),
                    )
                )
            actor_tracks.append(ActorTrack(str(track_id), categories[0], tuple(states)))

        return SceneIR(
            log_id=log_dir.name,
            source_dataset="argoverse2_sensor",
            coordinate_frame="AV2 city with explicit city_SE3_egovehicle",
            ego_poses=poses,
            sensor_calibrations=calibrations,
            actors=tuple(actor_tracks),
        )

    @staticmethod
    def _nearest_pose_index(timestamps: np.ndarray, target: int) -> int:
        if len(timestamps) == 0:
            raise ValueError("AV2 log has no ego poses")
        right = int(np.searchsorted(timestamps, int(target), side="left"))
        candidates = [min(right, len(timestamps) - 1), max(right - 1, 0)]
        return min(candidates, key=lambda index: abs(int(timestamps[index]) - int(target)))


def scene_ir_to_dict(scene: SceneIR) -> dict[str, Any]:
    """生成不含图像/LiDAR payload 的可追溯 JSON 表示。"""

    def se3_payload(transform: SE3) -> dict[str, Any]:
        return {
            "rotation": transform.rotation.tolist(),
            "translation_m": transform.translation_m.tolist(),
        }

    return {
        "schema_version": "worldsim_v7.sceneir.av2.v1",
        "log_id": scene.log_id,
        "source_dataset": scene.source_dataset,
        "coordinate_frame": scene.coordinate_frame,
        "ego_poses": [
            {"timestamp_ns": pose.timestamp_ns, "city_se3_egovehicle": se3_payload(pose.city_se3_egovehicle)}
            for pose in scene.ego_poses
        ],
        "sensor_calibrations": [
            {"sensor_name": item.sensor_name, "egovehicle_se3_sensor": se3_payload(item.egovehicle_se3_sensor)}
            for item in scene.sensor_calibrations
        ],
        "actors": [
            {
                "track_id": actor.track_id,
                "category": actor.category,
                "states": [
                    {
                        "timestamp_ns": state.timestamp_ns,
                        "city_se3_actor": se3_payload(state.city_se3_actor),
                        "center_ego_m": list(state.center_ego_m),
                        "size_lwh_m": list(state.size_lwh_m),
                        "num_interior_points": state.num_interior_points,
                    }
                    for state in actor.states
                ],
            }
            for actor in scene.actors
        ],
    }

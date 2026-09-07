"""从 nuScenes 官方元数据构造多相机 build 输入。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from motion_proj.worldsim_v72.data.camera_schema import CameraFramePayload, CameraWindow


DEFAULT_CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quaternion_rotation(wxyz: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(list(wxyz), dtype=np.float64)
    norm = float(np.linalg.norm([w, x, y, z]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("四元数必须有限且非零")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(translation: Iterable[float], rotation_wxyz: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_rotation(rotation_wxyz)
    transform[:3, 3] = np.asarray(list(translation), dtype=np.float64)
    return transform


class NuScenesCameraIndex:
    """只读元数据索引；payload 缺失时跳过窗口，不将缺失解释成观测。"""

    def __init__(self, dataset_root: Path, metadata_version: str = "v1.0-trainval") -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.metadata_root = self.dataset_root / metadata_version

        def rows(name: str) -> list[dict]:
            return json.loads((self.metadata_root / f"{name}.json").read_text(encoding="utf-8"))

        self.scenes = rows("scene")
        self.samples = rows("sample")
        self.sample_by_token = {str(row["token"]): row for row in self.samples}
        self.sample_data = {str(row["token"]): row for row in rows("sample_data")}
        self.calibrated = {str(row["token"]): row for row in rows("calibrated_sensor")}
        self.ego_poses = {str(row["token"]): row for row in rows("ego_pose")}
        sensors = {str(row["token"]): row for row in rows("sensor")}
        self.channel_by_calibrated = {
            token: str(sensors[str(row["sensor_token"])]["channel"])
            for token, row in self.calibrated.items()
        }
        self.data_by_sample_channel = {
            (str(row["sample_token"]), self.channel_by_calibrated[str(row["calibrated_sensor_token"])]): str(row["token"])
            for row in self.sample_data.values()
            if bool(row.get("is_key_frame", False))
        }
        self.scene_by_token = {str(row["token"]): row for row in self.scenes}

    def iter_windows(
        self,
        log_ids: Iterable[str],
        *,
        role: str,
        camera_channels: Iterable[str] = DEFAULT_CAMERA_CHANNELS,
        scene_ids: Iterable[str] | None = None,
        maximum_windows: int | None = None,
    ) -> Iterator[CameraWindow]:
        wanted_logs = {str(value) for value in log_ids}
        wanted_scenes = None if scene_ids is None else {str(value) for value in scene_ids}
        channels = tuple(str(value) for value in camera_channels)
        emitted = 0
        ordered_samples = sorted(self.samples, key=lambda row: (int(row["timestamp"]), str(row["token"])))
        for sample in ordered_samples:
            scene = self.scene_by_token[str(sample["scene_token"])]
            if str(scene["log_token"]) not in wanted_logs:
                continue
            if wanted_scenes is not None and str(scene["name"]) not in wanted_scenes:
                continue
            frames: list[CameraFramePayload] = []
            missing = False
            for channel in channels:
                token = self.data_by_sample_channel.get((str(sample["token"]), channel), "")
                row = self.sample_data.get(token)
                if row is None or not bool(row.get("is_key_frame", False)):
                    missing = True
                    break
                calibrated = self.calibrated[str(row["calibrated_sensor_token"])]
                actual_channel = self.channel_by_calibrated[str(row["calibrated_sensor_token"])]
                if actual_channel != channel:
                    raise RuntimeError(f"sample data channel 错配: {actual_channel} != {channel}")
                image_path = (self.dataset_root / str(row["filename"])).resolve()
                if not image_path.is_file() or image_path.stat().st_size <= 0:
                    missing = True
                    break
                ego_pose = self.ego_poses[str(row["ego_pose_token"])]
                world_from_ego = _transform(ego_pose["translation"], ego_pose["rotation"])
                ego_from_camera = _transform(calibrated["translation"], calibrated["rotation"])
                intrinsics = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
                frames.append(
                    CameraFramePayload(
                        frame_id=token,
                        sample_id=str(sample["token"]),
                        camera_id=channel,
                        time_ns=int(row["timestamp"]) * 1000,
                        image_path=str(image_path),
                        image_sha256=_sha256(image_path),
                        original_size_wh=np.asarray([int(row["width"]), int(row["height"])], dtype=np.int32),
                        intrinsics_px=intrinsics,
                        world_from_camera_opencv=world_from_ego @ ego_from_camera,
                        distortion_model="pinhole_rectified",
                        distortion_parameters=np.empty(0, dtype=np.float64),
                        provenance={
                            "sample_data_token": token,
                            "calibrated_sensor_token": str(row["calibrated_sensor_token"]),
                            "ego_pose_token": str(row["ego_pose_token"]),
                            "payload_role": "build_input",
                        },
                    )
                )
            if missing or len(frames) != len(channels):
                continue
            yield CameraWindow(
                dataset="nuScenes",
                role=role,
                log_id=str(scene["log_token"]),
                scene_id=str(scene["name"]),
                window_id=str(sample["token"]),
                frames=tuple(frames),
                provenance={"selection": "metadata_order_and_payload_availability", "target_access": False},
            )
            emitted += 1
            if maximum_windows is not None and emitted >= maximum_windows:
                return

    def sensor_points_world(self, sample_id: str, channel: str = "LIDAR_TOP") -> np.ndarray:
        """读取一个已存在的 keyframe 点云并变换到 world；缺 payload 时显式失败。"""
        token = self.data_by_sample_channel[(str(sample_id), str(channel))]
        row = self.sample_data[token]
        calibrated = self.calibrated[str(row["calibrated_sensor_token"])]
        ego_pose = self.ego_poses[str(row["ego_pose_token"])]
        payload = (self.dataset_root / str(row["filename"])).resolve()
        if not payload.is_file() or payload.stat().st_size <= 0:
            raise FileNotFoundError(payload)
        raw = np.fromfile(payload, dtype=np.float32)
        if len(raw) % 5 != 0:
            raise ValueError(f"nuScenes LiDAR payload 不是 Nx5 float32: {payload}")
        sensor_points = raw.reshape(-1, 5)[:, :3]
        world_from_sensor = _transform(ego_pose["translation"], ego_pose["rotation"]) @ _transform(
            calibrated["translation"], calibrated["rotation"]
        )
        homogeneous = np.concatenate(
            [sensor_points.astype(np.float64), np.ones((len(sensor_points), 1), dtype=np.float64)], axis=1
        )
        return (homogeneous @ world_from_sensor.T)[:, :3].astype(np.float32)

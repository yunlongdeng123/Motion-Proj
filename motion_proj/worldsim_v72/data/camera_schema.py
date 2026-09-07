"""EAS-VGGT 使用的 build-only 多相机输入合同。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CAMERA_WINDOW_SCHEMA_VERSION = "worldsim_v72.camera_window.v1"


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是有限矩阵 {shape}")
    return array


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    transform = _matrix(value, (4, 4), name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"{name} 的齐次末行无效")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{name} 的旋转部分不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError(f"{name} 的旋转部分必须属于 SO(3)")
    return transform


@dataclass(frozen=True)
class CameraFramePayload:
    """单个相机帧；只引用模型输入，不携带监督或评测标签。"""

    frame_id: str
    sample_id: str
    camera_id: str
    time_ns: int
    image_path: str
    image_sha256: str
    original_size_wh: np.ndarray
    intrinsics_px: np.ndarray
    world_from_camera_opencv: np.ndarray
    distortion_model: str
    distortion_parameters: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        size = np.asarray(self.original_size_wh, dtype=np.int32).reshape(-1)
        if len(size) != 2 or np.any(size <= 0):
            raise ValueError("original_size_wh 必须包含正的 width/height")
        intrinsics = _matrix(self.intrinsics_px, (3, 3), "intrinsics_px")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise ValueError("相机焦距必须为正")
        if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1.0e-8):
            raise ValueError("intrinsics_px 的末行无效")
        digest = str(self.image_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("image_sha256 必须是 64 位十六进制摘要")
        path = Path(self.image_path)
        if not path.is_absolute():
            raise ValueError("image_path 必须是绝对路径")
        object.__setattr__(self, "original_size_wh", size)
        object.__setattr__(self, "intrinsics_px", intrinsics)
        object.__setattr__(
            self,
            "world_from_camera_opencv",
            _rigid_transform(self.world_from_camera_opencv, "world_from_camera_opencv"),
        )
        object.__setattr__(
            self,
            "distortion_parameters",
            np.asarray(self.distortion_parameters, dtype=np.float64).reshape(-1),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class CameraWindow:
    """同一 scene 的多视图窗口，角色在日志级冻结。"""

    dataset: str
    role: str
    log_id: str
    scene_id: str
    window_id: str
    frames: tuple[CameraFramePayload, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("CameraWindow 至少需要一个相机帧")
        identities = [(frame.frame_id, frame.camera_id) for frame in frames]
        if len(identities) != len(set(identities)):
            raise ValueError("CameraWindow 中存在重复帧/相机")
        if any(frame.sample_id != frames[0].sample_id for frame in frames):
            raise ValueError("当前 E1 窗口要求相机来自同一 sample")
        forbidden = {"target", "label", "ground_truth", "heldout"}
        if forbidden & set(self.provenance):
            raise ValueError("CameraWindow provenance 不能携带监督字段")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": CAMERA_WINDOW_SCHEMA_VERSION,
            "dataset": self.dataset,
            "role": self.role,
            "log_id": self.log_id,
            "scene_id": self.scene_id,
            "window_id": self.window_id,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "camera_id": frame.camera_id,
                    "time_ns": frame.time_ns,
                    "image_sha256": frame.image_sha256,
                    "intrinsics_px": frame.intrinsics_px.tolist(),
                    "world_from_camera_opencv": frame.world_from_camera_opencv.tolist(),
                }
                for frame in self.frames
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

"""可替换的 flow/depth/track/ego-motion provider。"""
from __future__ import annotations

import abc

import torch

from .boxes_nuscenes import build_tracks
from .depth_anything import DepthEstimator
from .ego_flow import compute_ego_flow
from .flow_raft import RAFTFlow


class FlowProvider(abc.ABC):
    @abc.abstractmethod
    def estimate(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class DepthProvider(abc.ABC):
    @abc.abstractmethod
    def estimate(self, frames: torch.Tensor, sample: dict) -> torch.Tensor: ...


class TrackProvider(abc.ABC):
    @abc.abstractmethod
    def estimate(self, frames: torch.Tensor, sample: dict) -> list: ...


class EgoMotionProvider(abc.ABC):
    @abc.abstractmethod
    def flow(self, depth: torch.Tensor, sample: dict) -> torch.Tensor: ...


class RAFTFlowProvider(FlowProvider):
    def __init__(self, device: str = "cuda"):
        self.model = RAFTFlow(device=device)

    def estimate(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.flow_with_confidence(frames)


class LidarCalibratedDepthProvider(DepthProvider):
    """Depth-Anything 相对深度，存在投影 LiDAR 时逐帧做鲁棒尺度标定。"""

    def __init__(self, device: str = "cuda", enable: bool = True, min_points: int = 16):
        self.model = DepthEstimator(device=device, enable=enable)
        self.min_points = int(min_points)

    def estimate(self, frames: torch.Tensor, sample: dict) -> torch.Tensor:
        prediction = self.model.depth(frames)
        lidar = sample.get("lidar_depth")
        if lidar is None:
            return prediction
        lidar = torch.as_tensor(lidar, device=prediction.device, dtype=prediction.dtype)
        if lidar.shape != prediction.shape:
            raise ValueError(f"lidar_depth shape {tuple(lidar.shape)} != prediction {tuple(prediction.shape)}")
        calibrated = prediction.clone()
        for index in range(prediction.shape[0]):
            valid = torch.isfinite(lidar[index]) & (lidar[index] > 0) & torch.isfinite(prediction[index]) & (prediction[index] > 0)
            if int(valid.sum()) >= self.min_points:
                scale = torch.median(lidar[index][valid] / prediction[index][valid]).clamp(1e-4, 1e4)
                calibrated[index] = prediction[index] * scale
        return calibrated


class NuScenesTrackProvider(TrackProvider):
    def estimate(self, frames: torch.Tensor, sample: dict) -> list:
        return build_tracks(sample["boxes"], frames.shape[0])


class ProvidedEgoMotionProvider(EgoMotionProvider):
    def flow(self, depth: torch.Tensor, sample: dict) -> torch.Tensor:
        return compute_ego_flow(
            depth,
            sample["intrinsics"].to(depth.device),
            sample["cam2ego"].to(depth.device),
            sample["ego2global"].to(depth.device),
        )

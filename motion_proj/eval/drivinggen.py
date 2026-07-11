"""DrivingGen commit 48ed356 的 8 帧适配诊断公式。"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

DRIVINGGEN_COMMIT = "48ed35695855ef17d7a7cbd4adc0e8bd5fcc8223"
PROTOCOL = {
    "name": "DrivingGen-derived-8frame",
    "source_commit": DRIVINGGEN_COMMIT,
    "frames": 8,
    "nuscenes_dt_seconds": 0.5,
    "leaderboard_comparable": False,
}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=1e-6).clamp_min(0)


def scene_consistency(features: torch.Tensor) -> float:
    """对齐官方 v3：运动采样后相邻 DINO 特征 cosine 的均值。"""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("scene features 必须为 [T,D] 且 T>=2")
    return float(_cosine(features[1:], features[:-1]).mean())


def agent_consistency(track_features: list[torch.Tensor]) -> float | None:
    """对齐官方 agent stability：首帧参考与相邻帧 cosine 各占一半。"""
    scores = []
    for features in track_features:
        if features.ndim != 2 or features.shape[0] < 2:
            continue
        reference = _cosine(features[1:], features[0].expand_as(features[1:])).mean()
        adjacent = _cosine(features[1:], features[:-1]).mean()
        scores.append((reference + adjacent) * 0.5)
    return float(torch.stack(scores).mean()) if scores else None


def agent_disappearance_consistency(track_presence: list[torch.Tensor],
                                    track_last_boxes: list[torch.Tensor],
                                    image_size: tuple[int, int], edge_margin: float = 0.05) -> float | None:
    """8 帧诊断：提前结束的 track 仅在最后框接近画面边缘时视为自然离场。"""
    width, height = image_size
    values = []
    for present, last_box in zip(track_presence, track_last_boxes):
        indices = torch.where(present.bool())[0]
        if len(indices) < 2:
            continue
        last = int(indices[-1])
        if last == len(present) - 1:
            values.append(1.0)
            continue
        x1, y1, x2, y2 = [float(value) for value in last_box]
        near_edge = (x1 <= width * edge_margin or y1 <= height * edge_margin
                     or width - x2 <= width * edge_margin
                     or height - y2 <= height * edge_margin)
        values.append(float(near_edge))
    return float(np.mean(values)) if values else None


def trajectory_consistency(xy: np.ndarray, dt: float = 0.5,
                           v_static: float = 0.1, eps: float = 1e-9) -> dict:
    """复现官方速度/加速度一致性公式，但使用 nuScenes 关键帧的 0.5 秒间隔。"""
    points = np.asarray(xy, dtype=float)[..., :2]
    if points.ndim != 2 or points.shape[0] < 3:
        raise ValueError("trajectory 必须为 [T,2] 且 T>=3")
    speed = np.linalg.norm(np.diff(points, axis=0) / dt, axis=-1)
    if float(speed.max()) < v_static:
        return {"speed_consistency": None, "acceleration_consistency": None,
                "trajectory_consistency": None}
    speed_score = math.exp(-float(speed.std()) / (float(speed.mean()) + eps))
    acceleration = np.diff(speed, axis=0) / dt
    acceleration_score = math.exp(
        -float(acceleration.std()) / (float(np.abs(acceleration).mean()) + eps)
    )
    return {
        "speed_consistency": speed_score,
        "acceleration_consistency": acceleration_score,
        "trajectory_consistency": 0.5 * (speed_score + acceleration_score),
    }


def trajectory_from_ego_poses(ego2global: torch.Tensor, timestamps: torch.Tensor) -> dict:
    seconds = timestamps.detach().cpu().numpy().astype(np.float64) / 1e6
    gaps = np.diff(seconds)
    if not np.allclose(gaps, 0.5, atol=0.05):
        raise ValueError(f"nuScenes 时间间隔不是 0.5 秒: {gaps.tolist()}")
    xy = ego2global.detach().cpu().numpy()[:, :2, 3]
    return trajectory_consistency(xy, dt=float(np.median(gaps)))

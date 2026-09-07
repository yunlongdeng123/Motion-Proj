"""把多相机基础模型特征聚合到 Actor canonical surface candidates。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motion_proj.worldsim_v72.data.camera_schema import CameraWindow
from motion_proj.worldsim_v72.eas_vggt.types import BackboneGeometry
from motion_proj.worldsim_v72.eas_vggt.alignment import aligned_camera_center_rmse_m


def _bilinear(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """在 ``(H,W,...)`` 网格上双线性采样。"""
    height, width = values.shape[:2]
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, width - 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    trailing = (1,) * (values.ndim - 2)
    wx = wx.reshape((-1,) + trailing)
    wy = wy.reshape((-1,) + trailing)
    top = values[y0, x0] * (1.0 - wx) + values[y0, x1] * wx
    bottom = values[y1, x0] * (1.0 - wx) + values[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


@dataclass(frozen=True)
class CandidateVisualObservation:
    feature_sum: np.ndarray
    geometry_feature_sum: np.ndarray
    confidence_sum: np.ndarray
    observation_count: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.feature_sum, dtype=np.float32)
        geometry = np.asarray(self.geometry_feature_sum, dtype=np.float32)
        confidence = np.asarray(self.confidence_sum, dtype=np.float32).reshape(-1)
        count = np.asarray(self.observation_count, dtype=np.int32).reshape(-1)
        if (
            features.ndim != 2
            or geometry.shape != (len(features), 5)
            or len(features) != len(confidence)
            or len(features) != len(count)
        ):
            raise ValueError("candidate visual observation 形状不一致")
        if np.any(confidence < 0.0) or np.any(count < 0):
            raise ValueError("confidence/count 必须非负")
        object.__setattr__(self, "feature_sum", features)
        object.__setattr__(self, "geometry_feature_sum", geometry)
        object.__setattr__(self, "confidence_sum", confidence)
        object.__setattr__(self, "observation_count", count)

    @property
    def pooled_features(self) -> np.ndarray:
        denominator = np.maximum(self.confidence_sum, 1.0e-8)[:, None]
        pooled = self.feature_sum / denominator
        pooled[self.observation_count == 0] = 0.0
        return pooled.astype(np.float32)

    @property
    def pooled_geometry_features(self) -> np.ndarray:
        denominator = np.maximum(self.confidence_sum, 1.0e-8)[:, None]
        pooled = self.geometry_feature_sum / denominator
        pooled[self.observation_count == 0] = 0.0
        return pooled.astype(np.float32)


def aligned_backbone_points_world(window: CameraWindow, geometry: BackboneGeometry) -> np.ndarray:
    """Align a backbone point map once per window before pooling multiple actors."""

    if geometry.scale_status == "metric_aligned":
        return np.asarray(geometry.points_reference, dtype=np.float32)
    calibrated_poses = np.stack([frame.world_from_camera_opencv for frame in window.frames])
    similarity, _ = aligned_camera_center_rmse_m(
        geometry.reference_from_camera_opencv,
        calibrated_poses,
    )
    return similarity.apply(geometry.points_reference)


def observe_actor_candidates(
    candidates_actor_m: np.ndarray,
    world_from_actor: np.ndarray,
    window: CameraWindow,
    geometry: BackboneGeometry,
    *,
    minimum_camera_depth_m: float = 0.25,
    metric_points_world: np.ndarray | None = None,
    geometry_consistency_tolerance_m: float | None = None,
) -> CandidateVisualObservation:
    """按已知标定投影 surface candidates，并聚合同一窗口的 patch features。

    本函数只用 build RGB、相机标定和 backbone 输出。可见性目前是相机 frustum，
    因此后续训练必须把遮挡感知版本作为独立开关，不能把背景特征当 Actor 真值。
    """

    if geometry_consistency_tolerance_m is not None and geometry_consistency_tolerance_m <= 0.0:
        raise ValueError("geometry_consistency_tolerance_m must be positive")
    candidates = np.asarray(candidates_actor_m, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(world_from_actor, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_from_actor 必须为 (4,4)")
    if len(window.frames) != geometry.points_reference.shape[0]:
        raise ValueError("window 与 backbone camera 数量不一致")
    world = candidates @ transform[:3, :3].T + transform[:3, 3]
    homogeneous = np.concatenate([world, np.ones((len(world), 1))], axis=1)
    feature_dim = int(geometry.feature_grid.shape[-1])
    feature_sum = np.zeros((len(candidates), feature_dim), dtype=np.float64)
    geometry_feature_sum = np.zeros((len(candidates), 5), dtype=np.float64)
    confidence_sum = np.zeros(len(candidates), dtype=np.float64)
    observation_count = np.zeros(len(candidates), dtype=np.int32)
    model_height, model_width = geometry.points_reference.shape[1:3]
    feature_height, feature_width = geometry.feature_grid.shape[1:3]
    points_world = (
        aligned_backbone_points_world(window, geometry)
        if metric_points_world is None
        else np.asarray(metric_points_world, dtype=np.float32)
    )
    if points_world.shape != geometry.points_reference.shape:
        raise ValueError("metric_points_world shape mismatch")
    for index, frame in enumerate(window.frames):
        camera = homogeneous @ np.linalg.inv(frame.world_from_camera_opencv).T
        projected = camera[:, :3] @ frame.intrinsics_px.T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1.0e-8)
        uv1 = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
        model_uv = uv1 @ geometry.model_from_original_px[index].T
        inside = (
            (camera[:, 2] > minimum_camera_depth_m)
            & (model_uv[:, 0] >= 0.0)
            & (model_uv[:, 0] < model_width)
            & (model_uv[:, 1] >= 0.0)
            & (model_uv[:, 1] < model_height)
        )
        if not np.any(inside):
            continue
        selected = np.flatnonzero(inside)
        pixel_x = model_uv[selected, 0]
        pixel_y = model_uv[selected, 1]
        feature_x = (pixel_x + 0.5) * feature_width / model_width - 0.5
        feature_y = (pixel_y + 0.5) * feature_height / model_height - 0.5
        features = _bilinear(geometry.feature_grid[index].astype(np.float32), feature_x, feature_y)
        confidence = _bilinear(geometry.confidence[index], pixel_x, pixel_y).reshape(-1)
        geometry_valid = _bilinear(geometry.valid_mask[index].astype(np.float32), pixel_x, pixel_y).reshape(-1) >= 0.5
        sampled_world = _bilinear(points_world[index], pixel_x, pixel_y)
        finite = geometry_valid & np.isfinite(sampled_world).all(axis=1) & np.isfinite(confidence)
        selected = selected[finite]
        if not len(selected):
            continue
        features = features[finite]
        confidence = confidence[finite]
        sampled_world = sampled_world[finite]
        confidence = np.maximum(confidence.astype(np.float64), 1.0e-6)
        residual_actor = (sampled_world.astype(np.float64) - world[selected]) @ transform[:3, :3]
        sampled_homogeneous = np.concatenate([sampled_world, np.ones((len(sampled_world), 1))], axis=1)
        sampled_camera = sampled_homogeneous @ np.linalg.inv(frame.world_from_camera_opencv).T
        depth_residual = sampled_camera[:, 2] - camera[selected, 2]
        if geometry_consistency_tolerance_m is not None:
            consistent = np.abs(depth_residual) <= float(geometry_consistency_tolerance_m)
            selected = selected[consistent]
            if not len(selected):
                continue
            features = features[consistent]
            confidence = confidence[consistent]
            sampled_world = sampled_world[consistent]
            depth_residual = depth_residual[consistent]
            residual_actor = residual_actor[consistent]
        geometry_features = np.column_stack(
            [residual_actor, depth_residual, np.log1p(confidence)]
        )
        feature_sum[selected] += features.astype(np.float64) * confidence[:, None]
        geometry_feature_sum[selected] += geometry_features * confidence[:, None]
        confidence_sum[selected] += confidence
        observation_count[selected] += 1
    return CandidateVisualObservation(
        feature_sum=feature_sum.astype(np.float32),
        geometry_feature_sum=geometry_feature_sum.astype(np.float32),
        confidence_sum=confidence_sum.astype(np.float32),
        observation_count=observation_count,
    )

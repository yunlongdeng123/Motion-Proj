"""不同视觉几何基座共享的输出表示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BACKBONE_GEOMETRY_SCHEMA_VERSION = "worldsim_v72.backbone_geometry.v1"
SCALE_STATUSES = frozenset({"arbitrary", "approximate_metric", "metric_aligned"})


@dataclass(frozen=True)
class BackboneGeometry:
    """缓存中的公共几何；名称明确区分原生尺度与 metric 对齐结果。"""

    backbone_id: str
    checkpoint_id: str
    repository_commit: str
    frame_ids: np.ndarray
    image_sha256: np.ndarray
    model_from_original_px: np.ndarray
    points_reference: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    reference_from_camera_opencv: np.ndarray
    intrinsics_model_px: np.ndarray
    feature_grid: np.ndarray
    scale_status: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_reference, dtype=np.float32)
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("points_reference 必须为 (N,H,W,3)")
        count, height, width, _ = points.shape
        confidence = np.asarray(self.confidence, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if confidence.shape != (count, height, width) or valid.shape != confidence.shape:
            raise ValueError("confidence/valid_mask 必须与点图空间尺寸一致")
        if np.any(valid & ~np.isfinite(points).all(axis=-1)):
            raise ValueError("valid 点图位置必须有限")
        if np.any(valid & ~np.isfinite(confidence)):
            raise ValueError("valid confidence 必须有限")
        transforms = np.asarray(self.reference_from_camera_opencv, dtype=np.float64)
        intrinsics = np.asarray(self.intrinsics_model_px, dtype=np.float64)
        image_transforms = np.asarray(self.model_from_original_px, dtype=np.float64)
        if transforms.shape != (count, 4, 4):
            raise ValueError("reference_from_camera_opencv 必须为 (N,4,4)")
        if intrinsics.shape != (count, 3, 3) or image_transforms.shape != (count, 3, 3):
            raise ValueError("intrinsics/model_from_original 必须为逐帧 3x3 矩阵")
        features = np.asarray(self.feature_grid)
        if features.ndim != 4 or features.shape[0] != count or any(size <= 0 for size in features.shape[1:]):
            raise ValueError("feature_grid 必须为通道和空间尺寸均为正的 (N,h,w,c)")
        if self.scale_status not in SCALE_STATUSES:
            raise ValueError(f"未知 scale_status: {self.scale_status}")
        object.__setattr__(self, "frame_ids", np.asarray(self.frame_ids, dtype=str).reshape(count))
        object.__setattr__(self, "image_sha256", np.asarray(self.image_sha256, dtype=str).reshape(count))
        object.__setattr__(self, "model_from_original_px", image_transforms)
        object.__setattr__(self, "points_reference", points)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "reference_from_camera_opencv", transforms)
        object.__setattr__(self, "intrinsics_model_px", intrinsics)
        object.__setattr__(self, "feature_grid", features)
        object.__setattr__(self, "provenance", dict(self.provenance))

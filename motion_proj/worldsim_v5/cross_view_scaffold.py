"""V5 M2 的目标洞内 reference-blind 跨视图几何 scaffold。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from motion_proj.worldsim_v32.depth_guided_unseen_mask import (
    splat_rgbd_to_target,
)


class CrossViewScaffoldError(ValueError):
    """跨视图投影、融合或冻结输入不满足合约。"""


@dataclass(frozen=True)
class ProjectedDepthStack:
    """多个 source view 在同一 target image 上的独立 z-buffer 深度。"""

    depth: np.ndarray
    observed: np.ndarray
    source_views: tuple[tuple[int, int], ...]
    valid_source_pixels: tuple[int, ...]

    def audit(self, target_mask: np.ndarray) -> dict[str, Any]:
        target = np.asarray(target_mask, dtype=bool)
        if target.shape != self.depth.shape[1:]:
            raise CrossViewScaffoldError("投影 stack 与 target mask 尺寸不一致")
        per_source = []
        for index, (frame, camera_id) in enumerate(self.source_views):
            observed = self.observed[index] & target
            per_source.append(
                {
                    "frame": int(frame),
                    "camera_id": int(camera_id),
                    "valid_source_pixels": int(self.valid_source_pixels[index]),
                    "observed_target_pixels": int(observed.sum()),
                }
            )
        support_count = np.sum(self.observed & target[None], axis=0)
        denominator = max(int(target.sum()), 1)
        return {
            "source_view_count": len(self.source_views),
            "source_views": per_source,
            "any_support_pixels": int((support_count >= 1).sum()),
            "multi_support_pixels": int((support_count >= 2).sum()),
            "any_support_fraction": float((support_count >= 1).sum() / denominator),
            "multi_support_fraction": float((support_count >= 2).sum() / denominator),
            "maximum_support_count": int(support_count.max(initial=0)),
        }


@dataclass(frozen=True)
class CrossViewScaffoldResult:
    """融合后的逐像素深度、支持与不确定性。"""

    depth: np.ndarray
    valid: np.ndarray
    direct_support: np.ndarray
    extrapolated_support: np.ndarray
    support_count: np.ndarray
    disagreement_m: np.ndarray
    extrapolation_distance_pixels: np.ndarray

    def audit(self, target_mask: np.ndarray) -> dict[str, Any]:
        target = np.asarray(target_mask, dtype=bool)
        denominator = max(int(target.sum()), 1)
        direct_values = self.disagreement_m[self.direct_support]
        extrapolated_values = self.extrapolation_distance_pixels[
            self.extrapolated_support
        ]
        return {
            "target_pixels": int(target.sum()),
            "direct_support_pixels": int(self.direct_support.sum()),
            "direct_support_fraction": float(self.direct_support.sum() / denominator),
            "extrapolated_support_pixels": int(self.extrapolated_support.sum()),
            "extrapolated_support_fraction": float(
                self.extrapolated_support.sum() / denominator
            ),
            "fallback_pixels": int(
                (target & ~self.direct_support & ~self.extrapolated_support).sum()
            ),
            "fallback_fraction": float(
                (target & ~self.direct_support & ~self.extrapolated_support).sum()
                / denominator
            ),
            "direct_disagreement_mean_m": (
                float(np.mean(direct_values)) if direct_values.size else None
            ),
            "direct_disagreement_p90_m": (
                float(np.quantile(direct_values, 0.9)) if direct_values.size else None
            ),
            "extrapolation_distance_mean_pixels": (
                float(np.mean(extrapolated_values))
                if extrapolated_values.size
                else None
            ),
            "extrapolation_distance_p90_pixels": (
                float(np.quantile(extrapolated_values, 0.9))
                if extrapolated_values.size
                else None
            ),
            "valid_target_pixels": int((self.valid & target).sum()),
        }


def frozen_source_views(
    *,
    target_frame: int,
    camera_id: int,
    temporal_offsets: Sequence[int],
    minimum_frame: int,
    maximum_frame: int,
) -> tuple[tuple[int, int], ...]:
    """按冻结 offset 选择同相机 train-only source views。"""

    if minimum_frame < 0 or maximum_frame < minimum_frame:
        raise CrossViewScaffoldError("source frame 范围非法")
    offsets = tuple(int(value) for value in temporal_offsets)
    if not offsets or 0 in offsets or len(set(offsets)) != len(offsets):
        raise CrossViewScaffoldError("temporal offsets 必须非空、唯一且不含 0")
    views = tuple(
        (target_frame + offset, int(camera_id))
        for offset in offsets
        if minimum_frame <= target_frame + offset <= maximum_frame
    )
    if len(views) < 2:
        raise CrossViewScaffoldError("有效 source views 少于 2")
    return views


def frozen_multicamera_source_views(
    *,
    target_frame: int,
    target_camera_id: int,
    camera_ids: Sequence[int],
    temporal_offsets: Sequence[int],
    minimum_frame: int,
    maximum_frame: int,
    include_same_frame_other_cameras: bool,
) -> tuple[tuple[int, int], ...]:
    """冻结三训练相机的时空 source grid，并严格排除 target view。"""

    cameras = tuple(int(value) for value in camera_ids)
    offsets = tuple(int(value) for value in temporal_offsets)
    if not cameras or len(set(cameras)) != len(cameras):
        raise CrossViewScaffoldError("camera ids 必须非空且唯一")
    if int(target_camera_id) not in cameras:
        raise CrossViewScaffoldError("target camera 必须属于 frozen camera ids")
    if not offsets or 0 in offsets or len(set(offsets)) != len(offsets):
        raise CrossViewScaffoldError("temporal offsets 必须非空、唯一且不含 0")
    views = {
        (int(target_frame + offset), camera_id)
        for offset in offsets
        if minimum_frame <= target_frame + offset <= maximum_frame
        for camera_id in cameras
    }
    if include_same_frame_other_cameras:
        views.update(
            (int(target_frame), camera_id)
            for camera_id in cameras
            if camera_id != int(target_camera_id)
        )
    views.discard((int(target_frame), int(target_camera_id)))
    ordered = tuple(sorted(views))
    if len(ordered) < 2:
        raise CrossViewScaffoldError("有效 multicamera source views 少于 2")
    return ordered


def project_background_depth_stack(
    *,
    supports: Sequence[Mapping[str, Any]],
    target_shape: tuple[int, int],
    target_intrinsics: np.ndarray,
    target_camera_to_world: np.ndarray,
    minimum_source_opacity: float,
    source_stride: int,
    depth_key: str = "background_depth",
    require_background_opacity: bool = True,
) -> ProjectedDepthStack:
    """不读取 target depth，把 source background/LiDAR z-buffer 到 target。"""

    if not 0.0 <= minimum_source_opacity <= 1.0:
        raise CrossViewScaffoldError("minimum source opacity 必须位于 [0, 1]")
    if source_stride <= 0:
        raise CrossViewScaffoldError("source stride 必须为正")
    height, width = (int(target_shape[0]), int(target_shape[1]))
    target_mask = np.ones((height, width), dtype=bool)
    # 明确以全 NaN target depth 禁用 target-interior consistency/read。
    withheld_target_depth = np.full((height, width), np.nan, dtype=np.float32)
    depths: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    views: list[tuple[int, int]] = []
    valid_counts: list[int] = []
    for support in supports:
        source_depth = np.asarray(support[depth_key], dtype=np.float32)
        if source_depth.shape != (height, width):
            raise CrossViewScaffoldError("source depth 与 target 尺寸不一致")
        valid = (
            ~np.asarray(support["dynamic_mask"], dtype=bool)
            & ~np.asarray(support["egocar_mask"], dtype=bool)
            & np.isfinite(source_depth)
            & (source_depth > 1e-4)
        )
        if require_background_opacity:
            valid &= np.asarray(support["background_opacity"], dtype=np.float32) >= float(
                minimum_source_opacity
            )
        splat = splat_rgbd_to_target(
            source_depth=source_depth,
            source_rgb=np.asarray(support["background_rgb"], dtype=np.uint8),
            source_valid=valid,
            source_intrinsics=np.asarray(support["intrinsics"]),
            source_camera_to_world=np.asarray(support["camera_to_world"]),
            target_depth=withheld_target_depth,
            target_mask=target_mask,
            target_intrinsics=np.asarray(target_intrinsics),
            target_camera_to_world=np.asarray(target_camera_to_world),
            absolute_depth_tolerance_m=0.0,
            relative_depth_tolerance=0.0,
            stride=int(source_stride),
        )
        depths.append(np.asarray(splat.depth, dtype=np.float32))
        observed.append(np.asarray(splat.observed, dtype=bool))
        views.append((int(support["frame"]), int(support["camera_id"])))
        valid_counts.append(int(valid.sum()))
    if not depths:
        raise CrossViewScaffoldError("source support 不能为空")
    return ProjectedDepthStack(
        depth=np.stack(depths, axis=0),
        observed=np.stack(observed, axis=0),
        source_views=tuple(views),
        valid_source_pixels=tuple(valid_counts),
    )


def fuse_cross_view_scaffold(
    *,
    fallback_depth: np.ndarray,
    target_mask: np.ndarray,
    projected: ProjectedDepthStack,
    minimum_support_views: int,
    maximum_absolute_disagreement_m: float,
    maximum_relative_disagreement: float,
    maximum_extrapolation_pixels: float,
) -> CrossViewScaffoldResult:
    """融合多视图中值深度，有限外推后回退到 result-blind G0。"""

    fallback = np.asarray(fallback_depth, dtype=np.float32)
    target = np.asarray(target_mask, dtype=bool)
    if fallback.shape != target.shape or projected.depth.shape[1:] != target.shape:
        raise CrossViewScaffoldError("fallback/target/projected 尺寸不一致")
    if minimum_support_views < 2:
        raise CrossViewScaffoldError("minimum support views 必须至少为 2")
    if maximum_absolute_disagreement_m < 0 or maximum_relative_disagreement < 0:
        raise CrossViewScaffoldError("cross-view disagreement 容差不能为负")
    if maximum_extrapolation_pixels < 0:
        raise CrossViewScaffoldError("extrapolation 距离不能为负")
    values = np.where(projected.observed, projected.depth, np.nan).astype(np.float64)
    masked = np.ma.masked_invalid(values)
    median = np.ma.median(masked, axis=0).filled(np.nan).astype(np.float32)
    minimum = np.ma.min(masked, axis=0).filled(np.nan).astype(np.float32)
    maximum = np.ma.max(masked, axis=0).filled(np.nan).astype(np.float32)
    support_count = np.sum(projected.observed, axis=0).astype(np.int16)
    disagreement = maximum - minimum
    tolerance = np.maximum(
        float(maximum_absolute_disagreement_m),
        float(maximum_relative_disagreement) * np.abs(median),
    )
    direct = (
        target
        & (support_count >= int(minimum_support_views))
        & np.isfinite(median)
        & (median > 1e-4)
        & np.isfinite(disagreement)
        & (disagreement <= tolerance)
    )
    depth = fallback.copy()
    extrapolated = np.zeros_like(target)
    extrapolation_distance = np.full(target.shape, np.nan, dtype=np.float32)
    if direct.any():
        depth[direct] = median[direct]
        distance, nearest = distance_transform_edt(
            ~direct, return_distances=True, return_indices=True
        )
        extrapolated = (
            target
            & ~direct
            & (distance <= float(maximum_extrapolation_pixels))
        )
        nearest_depth = median[nearest[0], nearest[1]]
        depth[extrapolated] = nearest_depth[extrapolated]
        extrapolation_distance[direct] = 0.0
        extrapolation_distance[extrapolated] = distance[extrapolated].astype(
            np.float32
        )
    valid = target & np.isfinite(depth) & (depth > 1e-4)
    return CrossViewScaffoldResult(
        depth=depth,
        valid=valid,
        direct_support=direct,
        extrapolated_support=extrapolated,
        support_count=support_count,
        disagreement_m=disagreement.astype(np.float32),
        extrapolation_distance_pixels=extrapolation_distance,
    )


def lidar_agreement_audit(
    *,
    scaffold: CrossViewScaffoldResult,
    lidar_projected: ProjectedDepthStack,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    """只作独立稀疏支持审计；LiDAR 不改变 candidate depth。"""

    target = np.asarray(target_mask, dtype=bool)
    lidar_values = np.where(
        lidar_projected.observed, lidar_projected.depth, np.nan
    ).astype(np.float64)
    lidar_median = np.ma.median(
        np.ma.masked_invalid(lidar_values), axis=0
    ).filled(np.nan)
    supported = (
        target
        & np.isfinite(lidar_median)
        & np.isfinite(scaffold.depth)
        & (lidar_median > 1e-4)
    )
    residual = np.abs(scaffold.depth[supported] - lidar_median[supported])
    return {
        "projected_lidar_pixels": int(supported.sum()),
        "projected_lidar_coverage": float(
            supported.sum() / max(int(target.sum()), 1)
        ),
        "scaffold_lidar_mae_m": float(np.mean(residual)) if residual.size else None,
        "scaffold_lidar_median_m": (
            float(np.median(residual)) if residual.size else None
        ),
        "lidar_used_to_modify_candidate": False,
    }

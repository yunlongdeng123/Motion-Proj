"""Sibling pair 的共同可见 support 与固定 denominator 审计。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .paired_tracks import PairedQuerySet, RawTrackObservation


class CommonSupportError(RuntimeError):
    """共同 support 的身份、形状或来源不合法。"""


def _eligible(observation: RawTrackObservation, *, min_confidence: float, max_fb_error: float) -> torch.Tensor:
    return (
        observation.raw_visibility.bool()
        & torch.isfinite(observation.raw_points).all(dim=-1)
        & torch.isfinite(observation.raw_confidence)
        & (observation.raw_confidence >= min_confidence)
        & torch.isfinite(observation.forward_backward_error)
        & (observation.forward_backward_error <= max_fb_error)
    )


def _spatial_cluster_ids(points: torch.Tensor, *, cell_size: float) -> torch.Tensor:
    if cell_size <= 0:
        raise ValueError("cluster cell_size 必须为正")
    x = torch.floor(points[:, 0] / cell_size).long()
    y = torch.floor(points[:, 1] / cell_size).long()
    width = int(x.max()) + 2 if int(x.numel()) else 1
    return y * width + x


@dataclass(frozen=True)
class CommonSupportWindow:
    """一条 sibling edge 在一个时间窗上的共同测量域。"""

    candidate_a: str
    candidate_b: str
    start_frame: int
    end_frame: int
    common_mask: torch.Tensor
    eligible_a: torch.Tensor
    eligible_b: torch.Tensor
    dynamic_mask: torch.Tensor
    background_mask: torch.Tensor
    spatial_cluster_ids: torch.Tensor
    valid: bool
    reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ValueError("时间窗必须非空")
        if self.common_mask.shape != self.eligible_a.shape or self.common_mask.shape != self.eligible_b.shape:
            raise ValueError("共同与候选 support mask 必须同形")
        query_count = self.common_mask.shape[0]
        if self.dynamic_mask.shape != (query_count,) or self.background_mask.shape != (query_count,):
            raise ValueError("stratum mask 必须是 [N]")
        if bool((self.dynamic_mask & self.background_mask).any()):
            raise ValueError("background/dynamic query 不得重叠")
        if self.spatial_cluster_ids.shape != (query_count,):
            raise ValueError("spatial_cluster_ids 必须是 [N]")

    @property
    def window_mask(self) -> torch.Tensor:
        result = torch.zeros_like(self.common_mask)
        result[:, self.start_frame : self.end_frame] = self.common_mask[:, self.start_frame : self.end_frame]
        return result

    def to_record(self, *, condition_id: str, edge_id: str, query_set: PairedQuerySet) -> dict[str, Any]:
        common_ids_by_frame = []
        for time in range(self.start_frame, self.end_frame):
            indices = torch.nonzero(self.common_mask[:, time], as_tuple=False).flatten().tolist()
            common_ids_by_frame.append({
                "frame": time,
                "query_ids": [query_set.query_ids[index] for index in indices],
            })
        return {
            "condition_id": condition_id,
            "edge_id": edge_id,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "valid": bool(self.valid),
            "reason": self.reason,
            "common_query_ids_by_frame": common_ids_by_frame,
            "diagnostics": dict(self.diagnostics),
            "uses_future_gt": False,
        }


def build_common_support(
    query_set: PairedQuerySet,
    observation_a: RawTrackObservation,
    observation_b: RawTrackObservation,
    settings: Mapping[str, Any],
    *,
    window_starts: Sequence[int] = (0, 2, 4, 6, 8, 10),
    window_length: int = 4,
) -> list[CommonSupportWindow]:
    """构造 paired support；候选 dropout 只能降低 coverage，不能更改另一侧 denominator。"""
    if observation_a.uses_future_gt or observation_b.uses_future_gt or query_set.uses_future_gt:
        raise CommonSupportError("common support 禁止 future GT")
    if observation_a.query_set_hash != query_set.query_set_hash or observation_b.query_set_hash != query_set.query_set_hash:
        raise CommonSupportError("pair 两侧必须引用同一 query_set_hash")
    if observation_a.raw_points.shape != observation_b.raw_points.shape:
        raise CommonSupportError("pair 两侧轨迹形状必须一致")
    query_count, frame_count = observation_a.raw_visibility.shape
    if query_count != len(query_set.query_ids):
        raise CommonSupportError("track query 数与 query set 不一致")
    if window_length < 3:
        raise ValueError("window_length 至少为 3")

    minimum_dynamic = int(settings["minimum_common_dynamic_queries"])
    minimum_background = int(settings["minimum_common_background_queries"])
    minimum_dynamic_observations = int(settings["minimum_common_dynamic_observations"])
    minimum_clusters = int(settings["minimum_spatial_clusters"])
    minimum_frames_per_query = int(settings.get("minimum_frames_per_query", 3))
    min_confidence = float(settings["min_confidence"])
    max_fb_error = float(settings["max_fb_error_px"])
    cluster_cell_size = float(settings.get("cluster_cell_size_px", 48.0))
    eligible_a = _eligible(observation_a, min_confidence=min_confidence, max_fb_error=max_fb_error)
    eligible_b = _eligible(observation_b, min_confidence=min_confidence, max_fb_error=max_fb_error)
    common = eligible_a & eligible_b
    dynamic_mask = query_set.mask("dynamic")
    background_mask = query_set.mask("background")
    clusters = _spatial_cluster_ids(query_set.points, cell_size=cluster_cell_size)
    windows: list[CommonSupportWindow] = []

    for start in window_starts:
        end = int(start) + int(window_length)
        if start < 0 or end > frame_count:
            continue
        local_common = common[:, start:end]
        dynamic_counts = local_common[dynamic_mask].sum(dim=1) if bool(dynamic_mask.any()) else torch.empty(0)
        background_counts = local_common[background_mask].sum(dim=1) if bool(background_mask.any()) else torch.empty(0)
        dynamic_query_valid = dynamic_counts >= minimum_frames_per_query
        background_query_valid = background_counts >= minimum_frames_per_query
        dynamic_indices_all = torch.nonzero(dynamic_mask, as_tuple=False).flatten()
        dynamic_indices = dynamic_indices_all[dynamic_query_valid] if int(dynamic_indices_all.numel()) else dynamic_indices_all
        spatial_clusters = int(torch.unique(clusters[dynamic_indices]).numel()) if int(dynamic_indices.numel()) else 0
        dynamic_observations = int(local_common[dynamic_mask].sum()) if bool(dynamic_mask.any()) else 0
        reasons = []
        if not query_set.valid:
            reasons.append("query_set_invalid")
        if int(dynamic_query_valid.sum()) < minimum_dynamic:
            reasons.append("insufficient_dynamic_queries")
        if int(background_query_valid.sum()) < minimum_background:
            reasons.append("insufficient_background_queries")
        if dynamic_observations < minimum_dynamic_observations:
            reasons.append("insufficient_dynamic_observations")
        if spatial_clusters < minimum_clusters:
            reasons.append("insufficient_spatial_clusters")

        dynamic_total = max(int(dynamic_mask.sum()) * window_length, 1)
        background_total = max(int(background_mask.sum()) * window_length, 1)
        diagnostics = {
            "common_dynamic_queries": int(dynamic_query_valid.sum()),
            "common_background_queries": int(background_query_valid.sum()),
            "common_dynamic_query_frame_observations": dynamic_observations,
            "common_background_query_frame_observations": (
                int(local_common[background_mask].sum()) if bool(background_mask.any()) else 0
            ),
            "spatial_cluster_count": spatial_clusters,
            "coverage_by_frame": [
                int(common[dynamic_mask, time].sum()) if bool(dynamic_mask.any()) else 0
                for time in range(start, end)
            ],
            "missingness_a_dynamic": 1.0 - float(eligible_a[dynamic_mask, start:end].sum()) / dynamic_total,
            "missingness_b_dynamic": 1.0 - float(eligible_b[dynamic_mask, start:end].sum()) / dynamic_total,
            "missingness_a_background": 1.0 - float(eligible_a[background_mask, start:end].sum()) / background_total,
            "missingness_b_background": 1.0 - float(eligible_b[background_mask, start:end].sum()) / background_total,
            "expected_dynamic_denominator": dynamic_total,
            "common_dynamic_denominator": dynamic_observations,
            "invalid_reasons": reasons,
        }
        windows.append(CommonSupportWindow(
            candidate_a=observation_a.candidate_id,
            candidate_b=observation_b.candidate_id,
            start_frame=int(start),
            end_frame=end,
            common_mask=common.clone(),
            eligible_a=eligible_a.clone(),
            eligible_b=eligible_b.clone(),
            dynamic_mask=dynamic_mask.clone(),
            background_mask=background_mask.clone(),
            spatial_cluster_ids=clusters.clone(),
            valid=not reasons,
            reason=None if not reasons else "incomparable_support",
            diagnostics=diagnostics,
        ))
    return windows


def paired_masked_means(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    common_mask: torch.Tensor,
) -> tuple[float | None, float | None, int]:
    """在完全相同的有限 support 上计算两侧均值。"""
    if values_a.shape != values_b.shape or values_a.shape != common_mask.shape:
        raise ValueError("values 与 common_mask 必须同形")
    mask = common_mask & torch.isfinite(values_a) & torch.isfinite(values_b)
    count = int(mask.sum())
    if not count:
        return None, None, 0
    return float(values_a[mask].mean()), float(values_b[mask].mean()), count

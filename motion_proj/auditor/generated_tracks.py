"""无 future-GT 的 generated video 点轨迹 provider。

轨迹不是数据集实例标注的替代品：它们是基于生成帧本身的高置信局部观测，
以小方框承载给现有 object projector，因而修正范围始终受限于可见点附近。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .flow_raft import RAFTFlow
from .generated_geometry import fit_affine_background_flow
from .state import Track


@dataclass
class GeneratedTrackState:
    """provider 输出及可审计的无 GT 使用声明。"""

    tracks: list[Track]
    points: torch.Tensor                 # [N,K,2]，不可见点为 NaN
    visibility: torch.Tensor             # [N,K] bool
    confidence: torch.Tensor             # [N,K] in [0,1]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uses_future_gt: bool = False


def _sample(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """双线性采样 ``[H,W,C]`` field 于 ``[N,2]`` 像素坐标。"""
    height, width = field.shape[:2]
    gx = 2.0 * points[:, 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * points[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    value = F.grid_sample(
        field.permute(2, 0, 1).unsqueeze(0), grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return value[0, :, 0].transpose(0, 1)


def _in_bounds(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        torch.isfinite(points).all(dim=-1)
        & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
    )


def _image_gradient(frame: torch.Tensor) -> torch.Tensor:
    gray = frame.float().mean(dim=0, keepdim=True).unsqueeze(0)
    dx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy)[0, 0]


def _choose_spaced(score: torch.Tensor, valid: torch.Tensor, count: int,
                   min_distance: float) -> torch.Tensor:
    """按 score 确定性选点，并避免同一局部区域重复主导。"""
    height, width = score.shape
    indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    if not int(indices.numel()):
        return torch.empty(0, 2, device=score.device, dtype=score.dtype)
    values = score.reshape(-1)[indices]
    order = indices[torch.argsort(values, descending=True, stable=True)]
    selected: list[torch.Tensor] = []
    for index in order:
        y = torch.div(index, width, rounding_mode="floor")
        x = index.remainder(width)
        point = torch.stack([x, y]).to(score.dtype)
        if all(bool((point - previous).square().sum() >= min_distance * min_distance) for previous in selected):
            selected.append(point)
        if len(selected) >= count:
            break
    return torch.stack(selected) if selected else torch.empty(0, 2, device=score.device, dtype=score.dtype)


def _robust_smooth(points: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """对完整三帧窗口做中位数平滑，拒绝孤立的 tracking spike。"""
    smoothed = points.clone()
    for time in range(1, points.shape[1] - 1):
        valid = visible[:, time - 1] & visible[:, time] & visible[:, time + 1]
        if bool(valid.any()):
            smoothed[valid, time] = points[valid, time - 1: time + 2].median(dim=1).values
    return smoothed


class GeneratedTrackProvider(abc.ABC):
    """统一 generated track 接口；所有实现不得读取 sample 的 future 标注。"""

    @abc.abstractmethod
    def track(self, frames: torch.Tensor, **kwargs: Any) -> GeneratedTrackState: ...


class RAFTChainGeneratedTrackProvider(GeneratedTrackProvider):
    """RAFT 相邻帧链式传播 + forward/backward consistency 的开发 provider。"""

    def __init__(
        self,
        device: str = "cuda",
        *,
        queries_per_stratum: int = 24,
        min_track_length: int = 3,
        point_box_size: float = 16.0,
        min_distance: float = 8.0,
        fb_alpha: float = 0.05,
        fb_beta: float = 0.5,
        min_confidence: float = 0.5,
    ):
        self.device = device
        self.queries_per_stratum = int(queries_per_stratum)
        self.min_track_length = int(min_track_length)
        self.point_box_size = float(point_box_size)
        self.min_distance = float(min_distance)
        self.fb_alpha = float(fb_alpha)
        self.fb_beta = float(fb_beta)
        self.min_confidence = float(min_confidence)
        self._raft: RAFTFlow | None = None

    def _raft_model(self) -> RAFTFlow:
        if self._raft is None:
            self._raft = RAFTFlow(device=self.device)
        return self._raft

    def _queries(self, frames: torch.Tensor, flow: torch.Tensor, confidence: torch.Tensor,
                 background: torch.Tensor) -> tuple[torch.Tensor, list[str], dict[str, int]]:
        residual = (flow[0] - background[0]).norm(dim=-1)
        valid = torch.isfinite(residual) & torch.isfinite(confidence[0]) & (confidence[0] >= self.min_confidence)
        if not bool(valid.any()):
            return torch.empty(0, 2, device=flow.device, dtype=flow.dtype), [], {
                "background": 0, "dynamic_residual": 0, "foreground_candidate": 0,
            }
        values = residual[valid]
        low = values.median()
        high = torch.quantile(values, 0.75)
        gradient = _image_gradient(frames[0]).to(residual.dtype)
        grad_threshold = torch.quantile(gradient[valid], 0.60)
        strata = {
            "background": (residual <= low) & valid,
            "dynamic_residual": (residual >= high) & valid,
            "foreground_candidate": (residual >= low) & (gradient >= grad_threshold) & valid,
        }
        scores = {
            "background": confidence[0] / (1.0 + residual),
            "dynamic_residual": confidence[0] * residual,
            "foreground_candidate": confidence[0] * (residual + 0.25 * gradient),
        }
        all_points: list[torch.Tensor] = []
        labels: list[str] = []
        counts: dict[str, int] = {}
        for label in ("background", "dynamic_residual", "foreground_candidate"):
            picked = _choose_spaced(scores[label], strata[label], self.queries_per_stratum, self.min_distance)
            # 低纹理或静态 clip 仍保持分层预算，但明确记录为 fallback，而不是引入 GT。
            if picked.shape[0] < self.queries_per_stratum:
                fallback = _choose_spaced(scores[label], valid, self.queries_per_stratum, self.min_distance)
                picked = fallback[:self.queries_per_stratum]
            all_points.append(picked)
            labels.extend([label] * picked.shape[0])
            counts[label] = int(picked.shape[0])
        return torch.cat(all_points, dim=0), labels, counts

    @torch.no_grad()
    def track(
        self,
        frames: torch.Tensor,
        *,
        observed_flow: torch.Tensor | None = None,
        flow_confidence: torch.Tensor | None = None,
        background_flow: torch.Tensor | None = None,
        backward_flow: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        **_: Any,
    ) -> GeneratedTrackState:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames 必须是 [K,3,H,W]")
        frames = frames.to(self.device)
        frame_count, _, height, width = frames.shape
        if observed_flow is None or flow_confidence is None:
            observed_flow, backward_flow, flow_confidence = self._raft_model().flow_with_bidirectional_confidence(frames)
        flow = observed_flow.to(frames.device)
        confidence = flow_confidence.to(frames.device).clamp(0, 1)
        if flow.shape != (frame_count - 1, height, width, 2) or confidence.shape != flow.shape[:-1]:
            raise ValueError("observed_flow/flow_confidence 形状与 frames 不匹配")
        if backward_flow is None:
            backward_flow = self._raft_model().flow(frames[1:], frames[:-1])
        backward = backward_flow.to(frames.device)
        if backward.shape != flow.shape:
            raise ValueError("backward_flow 形状必须与 observed_flow 相同")
        if background_flow is None:
            background = fit_affine_background_flow(flow, confidence).flow
        else:
            background = background_flow.to(frames.device)
        if background.shape != flow.shape:
            raise ValueError("background_flow 形状必须与 observed_flow 相同")

        start, labels, stratum_counts = self._queries(frames, flow, confidence, background)
        query_count = start.shape[0]
        points = torch.full((query_count, frame_count, 2), float("nan"), device=frames.device, dtype=flow.dtype)
        visible = torch.zeros(query_count, frame_count, device=frames.device, dtype=torch.bool)
        point_confidence = torch.zeros(query_count, frame_count, device=frames.device, dtype=flow.dtype)
        if query_count:
            points[:, 0] = start
            visible[:, 0] = True
            point_confidence[:, 0] = _sample(confidence[0, ..., None], start)[:, 0]
        fb_errors: list[torch.Tensor] = []
        for time in range(frame_count - 1):
            active = visible[:, time]
            if not bool(active.any()):
                continue
            source = points[active, time]
            forward = _sample(flow[time], source)
            destination = source + forward
            backward_at_destination = _sample(backward[time], destination)
            local_confidence = _sample(confidence[time, ..., None], source)[:, 0]
            fb_error_sq = (forward + backward_at_destination).square().sum(dim=-1)
            threshold = self.fb_alpha * (forward.square().sum(dim=-1) + backward_at_destination.square().sum(dim=-1)) + self.fb_beta
            valid = _in_bounds(destination, height, width) & torch.isfinite(fb_error_sq) & (fb_error_sq <= threshold)
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            kept = active_indices[valid]
            points[kept, time + 1] = destination[valid]
            visible[kept, time + 1] = True
            point_confidence[kept, time + 1] = local_confidence[valid] * (threshold[valid] / (fb_error_sq[valid] + 1e-6)).clamp(max=1)
            fb_errors.append(fb_error_sq)

        points = _robust_smooth(points, visible)
        tracks: list[Track] = []
        half = self.point_box_size * 0.5
        depth_values = torch.ones(query_count, frame_count, device=frames.device, dtype=flow.dtype)
        if depth is not None:
            source_depth = depth.to(frames.device)
            for time in range(frame_count):
                active = visible[:, time]
                if bool(active.any()):
                    depth_values[active, time] = _sample(source_depth[time, ..., None], points[active, time])[:, 0]
        for index, label in enumerate(labels):
            length = int(visible[index].sum())
            mean_confidence = point_confidence[index, visible[index]].mean() if length else torch.zeros((), device=frames.device)
            if length < self.min_track_length or float(mean_confidence) < self.min_confidence:
                continue
            xy = points[index]
            xyxy = torch.stack([xy[:, 0] - half, xy[:, 1] - half, xy[:, 0] + half, xy[:, 1] + half], dim=-1)
            xyxy[~visible[index]] = float("nan")
            tracks.append(Track(
                instance_token=f"generated_{label}_{index:03d}", category=f"generated_point/{label}",
                xyxy=xyxy, depth=depth_values[index], present=visible[index],
            ))

        lengths = visible.sum(dim=1).float() if query_count else torch.empty(0, device=frames.device)
        velocity, acceleration, jerk = self._dynamics(points, visible, background)
        coverage = self._coverage(tracks, frame_count, height, width)
        diagnostics = {
            "provider": "raft_chain_v1", "uses_future_gt": False,
            "query_count": int(query_count), "stratum_query_count": stratum_counts,
            "valid_track_count": len(tracks),
            "median_track_length": float(lengths.median()) if int(lengths.numel()) else 0.0,
            "survival_rate": float(visible[:, -1].float().mean()) if query_count else 0.0,
            "velocity_rms_px": velocity, "acceleration_rms_px": acceleration, "jerk_rms_px": jerk,
            "correction_coverage": coverage,
            "fb_error_sq_median": float(torch.cat(fb_errors).median()) if fb_errors else None,
            "smoothing": "three_frame_coordinate_median_v1",
            "point_box_size": self.point_box_size,
        }
        return GeneratedTrackState(tracks, points, visible, point_confidence, diagnostics, uses_future_gt=False)

    @staticmethod
    def _dynamics(points: torch.Tensor, visible: torch.Tensor, background: torch.Tensor) -> tuple[float, float, float]:
        if not points.numel():
            return 0.0, 0.0, 0.0
        velocity = torch.full_like(points[:, :-1], float("nan"))
        for time in range(points.shape[1] - 1):
            valid = visible[:, time] & visible[:, time + 1]
            if bool(valid.any()):
                background_at_point = _sample(background[time], points[valid, time])
                velocity[valid, time] = points[valid, time + 1] - points[valid, time] - background_at_point
        def rms(value: torch.Tensor) -> float:
            finite = value[torch.isfinite(value)]
            return float(finite.square().mean().sqrt()) if int(finite.numel()) else 0.0
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        jerk = acceleration[:, 1:] - acceleration[:, :-1]
        return rms(velocity), rms(acceleration), rms(jerk)

    @staticmethod
    def _coverage(tracks: list[Track], frame_count: int, height: int, width: int) -> float:
        mask = torch.zeros(frame_count, height, width, dtype=torch.bool, device=tracks[0].xyxy.device) if tracks else None
        if mask is None:
            return 0.0
        for track in tracks:
            for time in torch.nonzero(track.present, as_tuple=False).flatten().tolist():
                u0, v0, u1, v1 = track.xyxy[time].round().long().tolist()
                mask[time, max(v0, 0):min(v1, height), max(u0, 0):min(u1, width)] = True
        return float(mask.float().mean())


class CoTracker3GeneratedTrackProvider(GeneratedTrackProvider):
    """独立长时 evaluator 的适配器；未安装 CoTracker3 时明确拒绝而不降级伪造结果。"""

    def __init__(self, predictor: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None):
        self.predictor = predictor

    @torch.no_grad()
    def track(self, frames: torch.Tensor, **_: Any) -> GeneratedTrackState:
        if self.predictor is None:
            raise RuntimeError("CoTracker3 未安装或未注入 predictor；它只可作为独立 evaluator，不能静默回退到 RAFT")
        points, visibility, confidence = self.predictor(frames)
        if points.ndim != 3 or points.shape[-1] != 2 or visibility.shape != points.shape[:2] or confidence.shape != points.shape[:2]:
            raise ValueError("CoTracker3 predictor 必须返回 points[N,K,2], visibility[N,K], confidence[N,K]")
        tracks: list[Track] = []
        for index in range(points.shape[0]):
            xy = points[index]
            xyxy = torch.stack([xy[:, 0] - 8, xy[:, 1] - 8, xy[:, 0] + 8, xy[:, 1] + 8], dim=-1)
            xyxy[~visibility[index].bool()] = float("nan")
            tracks.append(Track(f"cotracker3_{index:03d}", "generated_point/cotracker3", xyxy,
                                torch.ones(points.shape[1], device=points.device, dtype=points.dtype), visibility[index].bool()))
        return GeneratedTrackState(tracks, points, visibility.bool(), confidence.clamp(0, 1), {
            "provider": "cotracker3_evaluator_v1", "uses_future_gt": False,
            "valid_track_count": len(tracks),
        }, uses_future_gt=False)

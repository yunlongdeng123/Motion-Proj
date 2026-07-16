"""Sibling 视频的共同首帧 query 与未平滑 pair-mode 轨迹。

该模块与旧 ``RAFTChainGeneratedTrackProvider`` 并存。旧 provider 的候选级选点、
stratum fallback 和原地平滑属于历史 scorer 语义，不能被新的偏序 oracle 复用。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..auditor.flow_raft import RAFTFlow
from ..auditor.generated_geometry import fit_affine_background_flow


class PairedTrackError(RuntimeError):
    """共同 query 或 pair-mode 跟踪违反 fail-closed 契约。"""


def _sample(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """双线性采样 ``[H,W,C]`` field；输入点是 ``[N,2]`` 像素坐标。"""
    height, width = field.shape[:2]
    gx = 2.0 * points[:, 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * points[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    value = F.grid_sample(
        field.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return value[0, :, 0].transpose(0, 1)


def _in_bounds(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        torch.isfinite(points).all(dim=-1)
        & (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )


def _image_gradient(frame: torch.Tensor) -> torch.Tensor:
    gray = frame.float().mean(dim=0, keepdim=True).unsqueeze(0)
    dx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy)[0, 0]


def _choose_spaced(
    score: torch.Tensor,
    valid: torch.Tensor,
    count: int,
    min_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按稳定排序取空间分散点；不会从 ``valid`` 外 fallback。"""
    height, width = score.shape
    indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    if not int(indices.numel()):
        empty_points = torch.empty(0, 2, device=score.device, dtype=score.dtype)
        empty_scores = torch.empty(0, device=score.device, dtype=score.dtype)
        return empty_points, empty_scores
    values = score.reshape(-1)[indices]
    order = indices[torch.argsort(values, descending=True, stable=True)]
    selected: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    for index in order:
        y = torch.div(index, width, rounding_mode="floor")
        x = index.remainder(width)
        point = torch.stack([x, y]).to(score.dtype)
        if all(
            bool((point - previous).square().sum() >= min_distance * min_distance)
            for previous in selected
        ):
            selected.append(point)
            selected_scores.append(score.reshape(-1)[index])
        if len(selected) >= count:
            break
    if not selected:
        empty_points = torch.empty(0, 2, device=score.device, dtype=score.dtype)
        empty_scores = torch.empty(0, device=score.device, dtype=score.dtype)
        return empty_points, empty_scores
    return torch.stack(selected), torch.stack(selected_scores)


def _robust_smooth(points: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """生成独立诊断副本；调用者的 raw points 永不被修改。"""
    output = points.clone()
    for time in range(1, points.shape[1] - 1):
        valid = visible[:, time - 1] & visible[:, time] & visible[:, time + 1]
        if bool(valid.any()):
            window = points[valid, time - 1 : time + 2]
            output[valid, time] = torch.sort(window, dim=1, stable=True).values[:, 1]
    return output


def _tensor_json(value: torch.Tensor) -> list[Any]:
    """把非有限轨迹坐标写成 JSON ``null``，禁止非标准 NaN。"""
    raw = value.detach().cpu().tolist()

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, float) and not torch.isfinite(torch.tensor(item)):
            return None
        return item

    return clean(raw)


def _query_hash(points: torch.Tensor, query_ids: tuple[str, ...], strata: tuple[str, ...]) -> str:
    payload = {
        "points": points.detach().cpu().double().tolist(),
        "query_ids": list(query_ids),
        "strata": list(strata),
        "protocol": "common-first-frame-query-v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PairedQuerySet:
    """一个 condition 内所有 sibling 共用的 query 身份与首帧坐标。"""

    points: torch.Tensor
    query_ids: tuple[str, ...]
    strata: tuple[str, ...]
    selection_scores: torch.Tensor
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uses_future_gt: bool = False

    def __post_init__(self) -> None:
        count = int(self.points.shape[0])
        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError("points 必须是 [N,2]")
        if len(self.query_ids) != count or len(self.strata) != count:
            raise ValueError("query_ids/strata 必须与 points 等长")
        if self.selection_scores.shape != (count,):
            raise ValueError("selection_scores 必须是 [N]")
        if len(set(self.query_ids)) != count:
            raise ValueError("query_id 必须全局唯一")
        if set(self.strata) - {"background", "dynamic"}:
            raise ValueError("strata 只允许 background/dynamic")
        if self.uses_future_gt:
            raise PairedTrackError("共同 query 禁止使用 future GT")

    @property
    def query_set_hash(self) -> str:
        return _query_hash(self.points, self.query_ids, self.strata)

    def mask(self, stratum: str) -> torch.Tensor:
        return torch.tensor([value == stratum for value in self.strata], dtype=torch.bool)

    def to_record(self, *, condition_id: str) -> dict[str, Any]:
        return {
            "condition_id": condition_id,
            "protocol": "common-first-frame-query-v1",
            "query_set_hash": self.query_set_hash,
            "query_ids": list(self.query_ids),
            "xy": _tensor_json(self.points),
            "strata": list(self.strata),
            "selection_scores": _tensor_json(self.selection_scores),
            "valid": bool(self.valid),
            "diagnostics": dict(self.diagnostics),
            "uses_future_gt": False,
        }


@dataclass(frozen=True)
class RawTrackObservation:
    """某个 candidate 对共同 query 的 raw 轨迹观测。"""

    candidate_id: str
    query_set_hash: str
    raw_points: torch.Tensor
    raw_visibility: torch.Tensor
    raw_confidence: torch.Tensor
    forward_backward_error: torch.Tensor
    optional_smoothed_points: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uses_future_gt: bool = False

    def __post_init__(self) -> None:
        if self.raw_points.ndim != 3 or self.raw_points.shape[-1] != 2:
            raise ValueError("raw_points 必须是 [N,K,2]")
        expected = self.raw_points.shape[:2]
        for name, value in (
            ("raw_visibility", self.raw_visibility),
            ("raw_confidence", self.raw_confidence),
            ("forward_backward_error", self.forward_backward_error),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} 必须是 [N,K]")
        if self.optional_smoothed_points is not None and self.optional_smoothed_points.shape != self.raw_points.shape:
            raise ValueError("optional_smoothed_points 必须与 raw_points 同形")
        if self.uses_future_gt:
            raise PairedTrackError("pair-mode track 禁止使用 future GT")

    def to_record(self, *, condition_id: str) -> dict[str, Any]:
        return {
            "condition_id": condition_id,
            "candidate_id": self.candidate_id,
            "provider": "raft-pair-mode-v1",
            "query_set_hash": self.query_set_hash,
            "raw_points": _tensor_json(self.raw_points),
            "raw_visibility": self.raw_visibility.detach().cpu().tolist(),
            "raw_confidence": _tensor_json(self.raw_confidence),
            "forward_backward_error": _tensor_json(self.forward_backward_error),
            "optional_smoothed_points": (
                _tensor_json(self.optional_smoothed_points)
                if self.optional_smoothed_points is not None
                else None
            ),
            "diagnostics": dict(self.diagnostics),
            "uses_future_gt": False,
        }


def select_common_queries(
    first_frame: torch.Tensor,
    observed_flow: torch.Tensor,
    flow_confidence: torch.Tensor,
    background_flow: torch.Tensor,
    settings: Mapping[str, Any],
) -> PairedQuerySet:
    """只从 Base 首个 frame pair 选择一次互斥的背景/动态 query。"""
    if first_frame.ndim != 3 or first_frame.shape[0] != 3:
        raise ValueError("first_frame 必须是 [3,H,W]")
    if observed_flow.ndim != 4 or observed_flow.shape[-1] != 2:
        raise ValueError("observed_flow 必须是 [F,H,W,2]")
    if flow_confidence.shape != observed_flow.shape[:-1] or background_flow.shape != observed_flow.shape:
        raise ValueError("confidence/background 与 observed_flow 形状不匹配")
    if observed_flow.shape[0] < 1:
        raise ValueError("至少需要一个相邻帧 flow")

    confidence_floor = float(settings["min_confidence"])
    budget_background = int(settings["background_queries"])
    budget_dynamic = int(settings["dynamic_queries"])
    minimum_background = int(settings["minimum_background_queries"])
    minimum_dynamic = int(settings["minimum_dynamic_queries"])
    min_distance = float(settings["min_distance"])
    background_quantile = float(settings.get("background_residual_quantile", 0.50))
    dynamic_quantile = float(settings.get("dynamic_residual_quantile", 0.75))
    texture_quantile = float(settings.get("texture_quantile", 0.35))
    minimum_dynamic_residual = float(settings.get("minimum_dynamic_residual_px", 0.25))
    if not (0 <= background_quantile < dynamic_quantile <= 1):
        raise ValueError("background/dynamic residual quantile 无效")
    if min(minimum_background, minimum_dynamic, budget_background, budget_dynamic) <= 0:
        raise ValueError("query 预算与最低数量必须为正")

    residual = (observed_flow[0] - background_flow[0]).norm(dim=-1)
    confidence = flow_confidence[0]
    gradient = _image_gradient(first_frame).to(residual)
    finite = torch.isfinite(residual) & torch.isfinite(confidence) & torch.isfinite(gradient)
    valid = finite & (confidence >= confidence_floor)
    if bool(valid.any()):
        residual_values = residual[valid]
        background_cutoff = torch.quantile(residual_values, background_quantile)
        dynamic_cutoff = torch.quantile(residual_values, dynamic_quantile)
        texture_cutoff = torch.quantile(gradient[valid], texture_quantile)
    else:
        background_cutoff = torch.tensor(float("nan"), device=residual.device)
        dynamic_cutoff = torch.tensor(float("nan"), device=residual.device)
        texture_cutoff = torch.tensor(float("nan"), device=residual.device)

    textured = gradient >= texture_cutoff
    dynamic_mask = (
        valid
        & textured
        & (residual >= dynamic_cutoff)
        & (residual >= background_cutoff + minimum_dynamic_residual)
    )
    background_mask = valid & textured & (residual <= background_cutoff) & ~dynamic_mask
    background_score = confidence * gradient.clamp_min(0) / (1.0 + residual)
    dynamic_score = confidence * gradient.clamp_min(0) * residual
    background_points, background_scores = _choose_spaced(
        background_score, background_mask, budget_background, min_distance
    )
    dynamic_points, dynamic_scores = _choose_spaced(
        dynamic_score, dynamic_mask, budget_dynamic, min_distance
    )

    points = torch.cat([background_points, dynamic_points], dim=0)
    scores = torch.cat([background_scores, dynamic_scores], dim=0)
    query_ids = tuple(
        [f"bg-{index:03d}" for index in range(background_points.shape[0])]
        + [f"dyn-{index:03d}" for index in range(dynamic_points.shape[0])]
    )
    strata = tuple(
        ["background"] * int(background_points.shape[0])
        + ["dynamic"] * int(dynamic_points.shape[0])
    )
    reasons = []
    if int(background_points.shape[0]) < minimum_background:
        reasons.append("insufficient_background_queries")
    if int(dynamic_points.shape[0]) < minimum_dynamic:
        reasons.append("insufficient_dynamic_queries")
    diagnostics = {
        "background_query_count": int(background_points.shape[0]),
        "dynamic_query_count": int(dynamic_points.shape[0]),
        "background_candidate_pixel_count": int(background_mask.sum()),
        "dynamic_candidate_pixel_count": int(dynamic_mask.sum()),
        "minimum_background_queries": minimum_background,
        "minimum_dynamic_queries": minimum_dynamic,
        "background_residual_cutoff_px": float(background_cutoff) if bool(valid.any()) else None,
        "dynamic_residual_cutoff_px": float(dynamic_cutoff) if bool(valid.any()) else None,
        "minimum_dynamic_residual_px": minimum_dynamic_residual,
        "texture_cutoff": float(texture_cutoff) if bool(valid.any()) else None,
        "fallback_used": False,
        "invalid_reasons": reasons,
    }
    return PairedQuerySet(
        points=points.detach().cpu(),
        query_ids=query_ids,
        strata=strata,
        selection_scores=scores.detach().cpu(),
        valid=not reasons,
        diagnostics=diagnostics,
        uses_future_gt=False,
    )


@torch.no_grad()
def propagate_common_queries(
    *,
    candidate_id: str,
    frames: torch.Tensor,
    query_set: PairedQuerySet,
    observed_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    flow_confidence: torch.Tensor,
    fb_alpha: float,
    fb_beta: float,
    include_smoothed_diagnostic: bool = True,
) -> RawTrackObservation:
    """在一个 candidate 上传播固定 query；raw 与平滑结果物理分离。"""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames 必须是 [K,3,H,W]")
    frame_count, _, height, width = frames.shape
    expected_flow_shape = (frame_count - 1, height, width, 2)
    if observed_flow.shape != expected_flow_shape or backward_flow.shape != expected_flow_shape:
        raise ValueError("forward/backward flow 与 frames 形状不匹配")
    if flow_confidence.shape != expected_flow_shape[:-1]:
        raise ValueError("flow_confidence 与 frames 形状不匹配")
    if fb_alpha <= 0 or fb_beta <= 0:
        raise ValueError("F/B threshold 参数必须为正")

    device = observed_flow.device
    dtype = observed_flow.dtype
    start = query_set.points.to(device=device, dtype=dtype)
    query_count = int(start.shape[0])
    points = torch.full((query_count, frame_count, 2), float("nan"), device=device, dtype=dtype)
    visible = torch.zeros(query_count, frame_count, device=device, dtype=torch.bool)
    confidence = torch.zeros(query_count, frame_count, device=device, dtype=dtype)
    fb_error = torch.full((query_count, frame_count), float("nan"), device=device, dtype=dtype)
    if query_count:
        points[:, 0] = start
        visible[:, 0] = True
        confidence[:, 0] = _sample(flow_confidence[0, ..., None], start)[:, 0]
        fb_error[:, 0] = 0.0

    for time in range(frame_count - 1):
        active = visible[:, time]
        if not bool(active.any()):
            continue
        source = points[active, time]
        forward = _sample(observed_flow[time], source)
        destination = source + forward
        backward_at_destination = _sample(backward_flow[time], destination)
        local_confidence = _sample(flow_confidence[time, ..., None], source)[:, 0]
        error_sq = (forward + backward_at_destination).square().sum(dim=-1)
        threshold = fb_alpha * (
            forward.square().sum(dim=-1) + backward_at_destination.square().sum(dim=-1)
        ) + fb_beta
        keep = (
            _in_bounds(destination, height, width)
            & torch.isfinite(error_sq)
            & torch.isfinite(local_confidence)
            & (error_sq <= threshold)
        )
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        kept = active_indices[keep]
        points[kept, time + 1] = destination[keep]
        visible[kept, time + 1] = True
        confidence[kept, time + 1] = local_confidence[keep] * (
            threshold[keep] / (error_sq[keep] + 1.0e-6)
        ).clamp(max=1.0)
        fb_error[kept, time + 1] = error_sq[keep].sqrt()

    raw_points = points.detach().cpu().clone()
    raw_visibility = visible.detach().cpu().clone()
    raw_confidence = confidence.detach().cpu().clone()
    raw_fb_error = fb_error.detach().cpu().clone()
    smoothed = _robust_smooth(raw_points, raw_visibility) if include_smoothed_diagnostic else None
    if not torch.allclose(points.detach().cpu(), raw_points, rtol=0.0, atol=0.0, equal_nan=True):
        raise PairedTrackError("raw points 在诊断平滑前后发生变化")
    stratum_survival = {}
    for stratum in ("background", "dynamic"):
        mask = query_set.mask(stratum)
        stratum_survival[stratum] = (
            float(raw_visibility[mask, -1].float().mean()) if bool(mask.any()) else 0.0
        )
    return RawTrackObservation(
        candidate_id=candidate_id,
        query_set_hash=query_set.query_set_hash,
        raw_points=raw_points,
        raw_visibility=raw_visibility,
        raw_confidence=raw_confidence,
        forward_backward_error=raw_fb_error,
        optional_smoothed_points=smoothed,
        diagnostics={
            "provider": "raft-pair-mode-v1",
            "query_count": query_count,
            "frame_count": frame_count,
            "survival_rate": float(raw_visibility[:, -1].float().mean()) if query_count else 0.0,
            "stratum_survival": stratum_survival,
            "raw_preserved": True,
            "smoothing": "separate-three-frame-coordinate-median-v1" if smoothed is not None else None,
        },
        uses_future_gt=False,
    )


class PairModeRAFTTracker:
    """复用单个 frozen RAFT，在 condition 内生成共同 query 的 sibling 轨迹。"""

    def __init__(self, settings: Mapping[str, Any], *, device: str = "cuda"):
        self.settings = dict(settings)
        self.device = device
        self._raft: RAFTFlow | None = None

    def _model(self) -> RAFTFlow:
        if self._raft is None:
            self._raft = RAFTFlow(device=self.device)
        return self._raft

    @torch.no_grad()
    def _flows(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._model().flow_with_bidirectional_confidence(frames.to(self.device))

    @torch.no_grad()
    def track_fixed_queries(
        self,
        *,
        candidate_id: str,
        frames: torch.Tensor,
        query_set: PairedQuerySet,
    ) -> RawTrackObservation:
        """对 measurement perturbation 或新增 sibling 复用冻结 query set。"""
        flow, backward, confidence = self._flows(frames)
        return propagate_common_queries(
            candidate_id=candidate_id,
            frames=frames,
            query_set=query_set,
            observed_flow=flow,
            backward_flow=backward,
            flow_confidence=confidence,
            fb_alpha=float(self.settings["fb_alpha"]),
            fb_beta=float(self.settings["fb_beta"]),
        )

    @torch.no_grad()
    def track_condition(
        self,
        *,
        base_candidate_id: str,
        base_frames: torch.Tensor,
        sibling_frames: Mapping[str, torch.Tensor],
    ) -> tuple[PairedQuerySet, dict[str, RawTrackObservation]]:
        """Base 负责选点，Base 与所有 sibling 都按同一 query ID 跟踪。"""
        if not sibling_frames:
            raise ValueError("sibling_frames 不能为空")
        base_flow, base_backward, base_confidence = self._flows(base_frames)
        background = fit_affine_background_flow(
            base_flow,
            base_confidence,
            min_confidence=float(self.settings["min_confidence"]),
            min_points=int(self.settings.get("dense_affine_min_points", 256)),
        )
        query_set = select_common_queries(
            base_frames[0].to(base_flow.device),
            base_flow,
            base_confidence,
            background.flow,
            self.settings,
        )
        observations = {
            base_candidate_id: propagate_common_queries(
                candidate_id=base_candidate_id,
                frames=base_frames,
                query_set=query_set,
                observed_flow=base_flow,
                backward_flow=base_backward,
                flow_confidence=base_confidence,
                fb_alpha=float(self.settings["fb_alpha"]),
                fb_beta=float(self.settings["fb_beta"]),
            )
        }
        for candidate_id in sorted(sibling_frames):
            frames = sibling_frames[candidate_id]
            observations[candidate_id] = self.track_fixed_queries(
                candidate_id=candidate_id,
                frames=frames,
                query_set=query_set,
            )
        return query_set, observations

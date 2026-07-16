"""共同轨迹上的 robust affine 背景补偿与成对运动证据。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .common_support import CommonSupportWindow
from .paired_tracks import PairedQuerySet, RawTrackObservation


class ResidualMotionError(RuntimeError):
    """背景拟合或 paired component 不能安全计算。"""


def _json_tensor(value: torch.Tensor) -> list[Any]:
    raw = value.detach().cpu().tolist()

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    return clean(raw)


def _robust_median(value: torch.Tensor) -> float | None:
    finite = value[torch.isfinite(value)]
    return float(finite.median()) if int(finite.numel()) else None


@dataclass(frozen=True)
class SparseAffineField:
    """逐相邻帧的稀疏背景 affine flow。"""

    coefficients: torch.Tensor
    valid: torch.Tensor
    image_hw: tuple[int, int]
    diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 3 or self.coefficients.shape[1:] != (3, 2):
            raise ValueError("coefficients 必须是 [F,3,2]")
        if self.valid.shape != self.coefficients.shape[:1]:
            raise ValueError("valid 必须是 [F]")

    def predict(self, points: torch.Tensor, time: int) -> torch.Tensor:
        if time < 0 or time >= self.coefficients.shape[0] or not bool(self.valid[time]):
            return torch.full_like(points, float("nan"))
        height, width = self.image_hw
        x = 2.0 * points[..., 0] / max(width - 1, 1) - 1.0
        y = 2.0 * points[..., 1] / max(height - 1, 1) - 1.0
        design = torch.stack([torch.ones_like(x), x, y], dim=-1)
        return design @ self.coefficients[time].to(design)

    def to_record(self) -> dict[str, Any]:
        return {
            "model": "robust-sparse-affine-irls-v1",
            "coefficients": _json_tensor(self.coefficients),
            "valid": self.valid.tolist(),
            "diagnostics": [dict(row) for row in self.diagnostics],
            "uses_future_gt": False,
        }


def _weighted_affine_fit(
    design: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    root = weight.sqrt().unsqueeze(1)
    weighted = design * root
    normal = weighted.T @ weighted
    normal = normal + torch.eye(3, dtype=normal.dtype, device=normal.device) * ridge
    rhs = weighted.T @ (target * root)
    return torch.linalg.solve(normal, rhs)


def fit_sparse_affine_background(
    observation: RawTrackObservation,
    query_set: PairedQuerySet,
    support_mask: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    min_points: int,
    irls_iterations: int = 5,
    huber_delta: float = 2.5,
    min_residual_px: float = 0.25,
    ridge: float = 1.0e-4,
) -> SparseAffineField:
    """仅在共同 background anchors 上拟合 candidate-specific affine field。"""
    if observation.uses_future_gt or query_set.uses_future_gt:
        raise ResidualMotionError("背景补偿禁止 future GT")
    if observation.query_set_hash != query_set.query_set_hash:
        raise ResidualMotionError("observation/query set 身份不一致")
    if support_mask.shape != observation.raw_visibility.shape:
        raise ValueError("support_mask 必须是 [N,K]")
    if min_points < 3 or irls_iterations <= 0:
        raise ValueError("affine fit 参数无效")

    points = observation.raw_points.double()
    background = query_set.mask("background")
    frame_pairs = points.shape[1] - 1
    coefficients = torch.zeros(frame_pairs, 3, 2, dtype=torch.float64)
    valid_rows = torch.zeros(frame_pairs, dtype=torch.bool)
    diagnostics = []
    height, width = image_hw
    for time in range(frame_pairs):
        valid = (
            background
            & support_mask[:, time]
            & support_mask[:, time + 1]
            & torch.isfinite(points[:, time]).all(dim=-1)
            & torch.isfinite(points[:, time + 1]).all(dim=-1)
        )
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if int(indices.numel()) < min_points:
            diagnostics.append({
                "frame_pair": time,
                "valid": False,
                "reason": "insufficient_common_background_anchors",
                "fit_points": int(indices.numel()),
            })
            continue
        source = points[indices, time]
        target = points[indices, time + 1] - source
        x = 2.0 * source[:, 0] / max(width - 1, 1) - 1.0
        y = 2.0 * source[:, 1] / max(height - 1, 1) - 1.0
        design = torch.stack([torch.ones_like(x), x, y], dim=-1)
        weight = torch.ones(indices.numel(), dtype=torch.float64)
        cutoff = torch.tensor(min_residual_px, dtype=torch.float64)
        theta = torch.zeros(3, 2, dtype=torch.float64)
        for _ in range(irls_iterations):
            theta = _weighted_affine_fit(design, target, weight, ridge=ridge)
            residual = torch.linalg.vector_norm(target - design @ theta, dim=-1)
            median = residual.median()
            scale = 1.4826 * (residual - median).abs().median().clamp_min(1.0e-6)
            cutoff = torch.maximum(torch.tensor(min_residual_px, dtype=torch.float64), huber_delta * scale)
            weight = torch.where(
                residual <= cutoff,
                torch.ones_like(residual),
                cutoff / residual.clamp_min(1.0e-8),
            )
        residual = torch.linalg.vector_norm(target - design @ theta, dim=-1)
        inliers = residual <= cutoff
        coefficients[time] = theta
        valid_rows[time] = True
        diagnostics.append({
            "frame_pair": time,
            "valid": True,
            "fit_points": int(indices.numel()),
            "inlier_ratio": float(inliers.float().mean()),
            "residual_median_px": float(residual.median()),
            "residual_mean_px": float(residual.mean()),
            "huber_cutoff_px": float(cutoff),
        })
    return SparseAffineField(
        coefficients=coefficients.float(),
        valid=valid_rows,
        image_hw=image_hw,
        diagnostics=tuple(diagnostics),
    )


def camera_field_distance(
    field_a: SparseAffineField,
    field_b: SparseAffineField,
    observation_a: RawTrackObservation,
    observation_b: RawTrackObservation,
    support: CommonSupportWindow,
) -> float | None:
    """在共同 background anchors 的平均位置比较两侧 nuisance field。"""
    values = []
    for time in range(support.start_frame, support.end_frame - 1):
        valid = (
            support.background_mask
            & support.common_mask[:, time]
            & support.common_mask[:, time + 1]
            & torch.isfinite(observation_a.raw_points[:, time]).all(dim=-1)
            & torch.isfinite(observation_b.raw_points[:, time]).all(dim=-1)
        )
        if not bool(valid.any()) or not bool(field_a.valid[time] & field_b.valid[time]):
            continue
        midpoint = 0.5 * (
            observation_a.raw_points[valid, time] + observation_b.raw_points[valid, time]
        )
        delta = torch.linalg.vector_norm(
            field_a.predict(midpoint, time) - field_b.predict(midpoint, time), dim=-1
        )
        values.append(delta)
    if not values:
        return None
    return float(torch.cat(values).median())


def _residual_velocity(
    observation: RawTrackObservation,
    field: SparseAffineField,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    points = observation.raw_points.float()
    result = torch.full_like(points[:, :-1], float("nan"))
    for time in range(points.shape[1] - 1):
        valid = (
            support_mask[:, time]
            & support_mask[:, time + 1]
            & torch.isfinite(points[:, time]).all(dim=-1)
            & torch.isfinite(points[:, time + 1]).all(dim=-1)
            & field.valid[time]
        )
        if bool(valid.any()):
            displacement = points[valid, time + 1] - points[valid, time]
            result[valid, time] = displacement - field.predict(points[valid, time], time)
    return result


def _punc_energy(
    observation: RawTrackObservation,
    common_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """在相同 visible segment 上计算 P-UNC 修正量与 SNR eligibility。"""
    from ..diagnostics.projector_validity import constrained_smooth_points

    points = observation.raw_points.float()
    output = torch.full(points.shape[:2], float("nan"), dtype=torch.float32)
    accepted = torch.zeros(points.shape[:2], dtype=torch.bool)
    confidence_floor = float(settings["confidence_floor"])
    uncertainty_floor = float(settings["uncertainty_floor_px"])
    uncertainty_scale = float(settings["uncertainty_confidence_scale_px"])
    snr_threshold = float(settings["snr_threshold"])
    for index in torch.nonzero(dynamic_mask, as_tuple=False).flatten().tolist():
        visible = common_mask[index].bool()
        confidence = observation.raw_confidence[index].float().clamp(0.0, 1.0)
        allowed = visible & (confidence >= confidence_floor)
        if int(allowed.sum()) < 3:
            continue
        projected = constrained_smooth_points(
            points[index],
            visible,
            confidence,
            allowed,
            lambda_acceleration=float(settings["lambda_acceleration"]),
            lambda_jerk=float(settings["lambda_jerk"]),
            robust_delta_px=float(settings["robust_delta_px"]),
            robust_iterations=int(settings["robust_iterations"]),
        )
        delta = torch.linalg.vector_norm(projected - points[index], dim=-1)
        uncertainty = uncertainty_floor + uncertainty_scale * (1.0 - confidence)
        snr = delta / uncertainty.clamp_min(1.0e-8)
        eligible = allowed & torch.isfinite(snr) & (snr >= snr_threshold)
        output[index, visible] = 0.0
        output[index, eligible] = snr[eligible].square()
        accepted[index] = eligible
    return output, accepted


def _acceleration(velocity: torch.Tensor) -> torch.Tensor:
    output = torch.full_like(velocity[:, 1:], float("nan"))
    valid = torch.isfinite(velocity[:, 1:]).all(dim=-1) & torch.isfinite(velocity[:, :-1]).all(dim=-1)
    delta = velocity[:, 1:] - velocity[:, :-1]
    magnitude = torch.linalg.vector_norm(delta, dim=-1)
    output[..., 0] = torch.where(valid, magnitude, torch.full_like(magnitude, float("nan")))
    return output[..., 0]


def _curvature(velocity: torch.Tensor, *, speed_floor: float) -> torch.Tensor:
    first, second = velocity[:, :-1], velocity[:, 1:]
    speed_first = torch.linalg.vector_norm(first, dim=-1)
    speed_second = torch.linalg.vector_norm(second, dim=-1)
    valid = (
        torch.isfinite(first).all(dim=-1)
        & torch.isfinite(second).all(dim=-1)
        & (speed_first >= speed_floor)
        & (speed_second >= speed_floor)
    )
    cosine = (first * second).sum(dim=-1) / (speed_first * speed_second).clamp_min(1.0e-8)
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    return torch.where(valid, angle, torch.full_like(angle, float("nan")))


def _coherence(
    velocity: torch.Tensor,
    first_frame_points: torch.Tensor,
    dynamic_mask: torch.Tensor,
    *,
    radius: float,
    minimum_neighbors: int,
) -> torch.Tensor:
    query_count, frame_pairs, _ = velocity.shape
    output = torch.full((query_count, frame_pairs), float("nan"), dtype=velocity.dtype)
    distance = torch.cdist(first_frame_points.float(), first_frame_points.float())
    dynamic_indices = torch.nonzero(dynamic_mask, as_tuple=False).flatten()
    for index in dynamic_indices.tolist():
        neighbors = dynamic_mask & (distance[index] <= radius) & (distance[index] > 0)
        for time in range(frame_pairs):
            valid_neighbors = neighbors & torch.isfinite(velocity[:, time]).all(dim=-1)
            if int(valid_neighbors.sum()) < minimum_neighbors or not bool(torch.isfinite(velocity[index, time]).all()):
                continue
            center = velocity[valid_neighbors, time].median(dim=0).values
            output[index, time] = torch.linalg.vector_norm(velocity[index, time] - center)
    return output


def _activity(
    observation: RawTrackObservation,
    velocity: torch.Tensor,
    support: CommonSupportWindow,
) -> dict[str, float | None]:
    dynamic = support.dynamic_mask
    start, end = support.start_frame, support.end_frame
    expected = max(int(dynamic.sum()) * (end - start), 1)
    eligible = support.eligible_a if observation.candidate_id == support.candidate_a else support.eligible_b
    active_fraction = float(eligible[dynamic, start:end].sum()) / expected
    local_velocity = velocity[dynamic, start : end - 1]
    speed = torch.linalg.vector_norm(local_velocity, dim=-1)
    residual_speed = _robust_median(speed)
    displacement = []
    for row in local_velocity:
        finite = torch.isfinite(row).all(dim=-1)
        if int(finite.sum()) >= 2:
            displacement.append(torch.linalg.vector_norm(row[finite].sum(dim=0)))
    net = float(torch.stack(displacement).median()) if displacement else None
    survival = float(eligible[dynamic, end - 1].float().mean()) if bool(dynamic.any()) else 0.0
    return {
        "active_fraction": active_fraction,
        "residual_speed": residual_speed,
        "net_residual_displacement": net,
        "survival": survival,
    }


@dataclass(frozen=True)
class MotionComponentEvidence:
    """一个 edge/window 的 paired component、背景与 activity 证据。"""

    differences: dict[str, torch.Tensor]
    values_a: dict[str, torch.Tensor]
    values_b: dict[str, torch.Tensor]
    field_a: SparseAffineField
    field_b: SparseAffineField
    camera_distance_px: float | None
    activity_a: dict[str, float | None]
    activity_b: dict[str, float | None]
    valid: bool
    reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def component_record(self, *, condition_id: str, edge_id: str, support: CommonSupportWindow) -> dict[str, Any]:
        return {
            "condition_id": condition_id,
            "edge_id": edge_id,
            "candidate_a": support.candidate_a,
            "candidate_b": support.candidate_b,
            "start_frame": support.start_frame,
            "end_frame": support.end_frame,
            "differences_b_minus_a": {name: _json_tensor(value) for name, value in self.differences.items()},
            "summary_a": {name: _robust_median(value) for name, value in self.values_a.items()},
            "summary_b": {name: _robust_median(value) for name, value in self.values_b.items()},
            "activity_a": dict(self.activity_a),
            "activity_b": dict(self.activity_b),
            "camera_distance_px": self.camera_distance_px,
            "valid": bool(self.valid),
            "reason": self.reason,
            "diagnostics": dict(self.diagnostics),
            "uses_future_gt": False,
        }

    def background_record(self, *, condition_id: str, edge_id: str, support: CommonSupportWindow) -> dict[str, Any]:
        return {
            "condition_id": condition_id,
            "edge_id": edge_id,
            "candidate_a": support.candidate_a,
            "candidate_b": support.candidate_b,
            "start_frame": support.start_frame,
            "end_frame": support.end_frame,
            "field_a": self.field_a.to_record(),
            "field_b": self.field_b.to_record(),
            "camera_distance_px": self.camera_distance_px,
            "uses_future_gt": False,
        }


def compute_motion_component_evidence(
    query_set: PairedQuerySet,
    observation_a: RawTrackObservation,
    observation_b: RawTrackObservation,
    support: CommonSupportWindow,
    settings: Mapping[str, Any],
    *,
    image_hw: tuple[int, int],
) -> MotionComponentEvidence:
    """计算 lower-is-better components；差值统一为 ``B - A``。"""
    if observation_a.uses_future_gt or observation_b.uses_future_gt:
        raise ResidualMotionError("motion evidence 禁止 future GT")
    window_mask = support.window_mask
    field_settings = settings["background_affine"]
    field_a = fit_sparse_affine_background(
        observation_a,
        query_set,
        window_mask,
        image_hw=image_hw,
        min_points=int(field_settings["minimum_points"]),
        irls_iterations=int(field_settings["irls_iterations"]),
        huber_delta=float(field_settings["huber_delta"]),
        min_residual_px=float(field_settings["minimum_residual_px"]),
        ridge=float(field_settings["ridge"]),
    )
    field_b = fit_sparse_affine_background(
        observation_b,
        query_set,
        window_mask,
        image_hw=image_hw,
        min_points=int(field_settings["minimum_points"]),
        irls_iterations=int(field_settings["irls_iterations"]),
        huber_delta=float(field_settings["huber_delta"]),
        min_residual_px=float(field_settings["minimum_residual_px"]),
        ridge=float(field_settings["ridge"]),
    )
    needed_pairs = range(support.start_frame, support.end_frame - 1)
    background_valid = all(bool(field_a.valid[index] & field_b.valid[index]) for index in needed_pairs)
    camera_distance = camera_field_distance(field_a, field_b, observation_a, observation_b, support)
    velocity_a = _residual_velocity(observation_a, field_a, window_mask)
    velocity_b = _residual_velocity(observation_b, field_b, window_mask)
    acceleration_a = _acceleration(velocity_a)
    acceleration_b = _acceleration(velocity_b)
    curvature_a = _curvature(velocity_a, speed_floor=float(settings["curve_speed_floor_px"]))
    curvature_b = _curvature(velocity_b, speed_floor=float(settings["curve_speed_floor_px"]))
    coherence_a = _coherence(
        velocity_a,
        query_set.points,
        support.dynamic_mask,
        radius=float(settings["coherence_radius_px"]),
        minimum_neighbors=int(settings["coherence_minimum_neighbors"]),
    )
    coherence_b = _coherence(
        velocity_b,
        query_set.points,
        support.dynamic_mask,
        radius=float(settings["coherence_radius_px"]),
        minimum_neighbors=int(settings["coherence_minimum_neighbors"]),
    )
    punc_a, punc_accept_a = _punc_energy(
        observation_a, window_mask, support.dynamic_mask, settings["punc"]
    )
    punc_b, punc_accept_b = _punc_energy(
        observation_b, window_mask, support.dynamic_mask, settings["punc"]
    )
    punc_union = punc_accept_a | punc_accept_b
    punc_a = torch.where(punc_union, torch.nan_to_num(punc_a, nan=0.0), torch.full_like(punc_a, float("nan")))
    punc_b = torch.where(punc_union, torch.nan_to_num(punc_b, nan=0.0), torch.full_like(punc_b, float("nan")))

    start, end = support.start_frame, support.end_frame
    dynamic = support.dynamic_mask
    values_a = {
        "punc": punc_a[dynamic, start:end],
        "acceleration": acceleration_a[dynamic, start : end - 2],
        "curvature": curvature_a[dynamic, start : end - 2],
        "coherence": coherence_a[dynamic, start : end - 1],
        "survival": 1.0 - support.eligible_a[dynamic, start:end].float(),
    }
    values_b = {
        "punc": punc_b[dynamic, start:end],
        "acceleration": acceleration_b[dynamic, start : end - 2],
        "curvature": curvature_b[dynamic, start : end - 2],
        "coherence": coherence_b[dynamic, start : end - 1],
        "survival": 1.0 - support.eligible_b[dynamic, start:end].float(),
    }
    differences = {}
    counts = {}
    for name in values_a:
        finite = torch.isfinite(values_a[name]) & torch.isfinite(values_b[name])
        difference = torch.full_like(values_a[name], float("nan"))
        difference[finite] = values_b[name][finite] - values_a[name][finite]
        differences[name] = difference
        counts[name] = int(finite.sum())
    reasons = []
    if not support.valid:
        reasons.append("support_invalid")
    if not background_valid:
        reasons.append("background_fit_invalid")
    if camera_distance is None:
        reasons.append("camera_distance_invalid")
    minimum_component_observations = int(settings["minimum_component_observations"])
    if not any(counts[name] >= minimum_component_observations for name in ("punc", "acceleration", "curvature", "coherence")):
        reasons.append("insufficient_component_observations")
    return MotionComponentEvidence(
        differences=differences,
        values_a=values_a,
        values_b=values_b,
        field_a=field_a,
        field_b=field_b,
        camera_distance_px=camera_distance,
        activity_a=_activity(observation_a, velocity_a, support),
        activity_b=_activity(observation_b, velocity_b, support),
        valid=not reasons,
        reason=None if not reasons else reasons[0],
        diagnostics={
            "component_common_counts": counts,
            "punc_union_count": int(punc_union[dynamic, start:end].sum()),
            "invalid_reasons": reasons,
        },
    )

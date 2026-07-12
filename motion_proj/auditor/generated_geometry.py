"""生成视频的条件安全几何模式与背景运动估计。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ..utils.geometry import pixel_grid
from .ego_flow import compute_ego_flow


GENERATED_GEOMETRY_MODES = {
    "gt_ego_debug",
    "identity_ego",
    "estimated_background_motion",
    "controlled_ego",
}


@dataclass
class BackgroundMotionEstimate:
    flow: torch.Tensor
    confidence: torch.Tensor
    uses_future_gt_ego: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _weighted_affine_fit(
    design: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    weighted = design * weight.sqrt().unsqueeze(1)
    normal = weighted.T @ weighted
    normal = normal + torch.eye(3, device=normal.device, dtype=normal.dtype) * ridge
    rhs = weighted.T @ (target * weight.sqrt().unsqueeze(1))
    return torch.linalg.solve(normal, rhs)


@torch.no_grad()
def fit_affine_background_flow(
    observed_flow: torch.Tensor,
    flow_confidence: torch.Tensor,
    *,
    min_confidence: float = 0.5,
    min_points: int = 256,
    max_points: int = 32768,
    irls_iterations: int = 5,
    huber_delta: float = 2.5,
    min_residual_px: float = 0.5,
    ridge: float = 1e-4,
) -> BackgroundMotionEstimate:
    """以确定性 IRLS 拟合逐帧全局 affine 背景光流。"""
    if observed_flow.ndim != 4 or observed_flow.shape[-1] != 2:
        raise ValueError("observed_flow 必须是 [F,H,W,2]")
    if flow_confidence.shape != observed_flow.shape[:-1]:
        raise ValueError("flow_confidence 必须是 [F,H,W]")

    flow = observed_flow.float()
    confidence = flow_confidence.float()
    frames, height, width, _ = flow.shape
    grid = pixel_grid(height, width, device=flow.device, dtype=flow.dtype)
    x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    design_full = torch.stack([torch.ones_like(x), x, y], dim=-1).reshape(-1, 3)

    fitted: list[torch.Tensor] = []
    fit_confidences: list[torch.Tensor] = []
    pair_diagnostics: list[dict[str, Any]] = []
    for frame_index in range(frames):
        target_full = flow[frame_index].reshape(-1, 2)
        conf_full = confidence[frame_index].reshape(-1)
        valid = (
            torch.isfinite(target_full).all(dim=1)
            & torch.isfinite(conf_full)
            & (conf_full >= min_confidence)
        )
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() > max_points:
            positions = torch.linspace(
                0,
                valid_indices.numel() - 1,
                max_points,
                device=valid_indices.device,
            ).round().long()
            valid_indices = valid_indices[positions]

        if valid_indices.numel() < min_points:
            fitted.append(torch.zeros_like(flow[frame_index]))
            fit_confidences.append(torch.zeros_like(confidence[frame_index]))
            pair_diagnostics.append(
                {
                    "frame_pair": frame_index,
                    "valid": False,
                    "reason": "insufficient_points",
                    "fit_points": int(valid_indices.numel()),
                }
            )
            continue

        design = design_full[valid_indices]
        target = target_full[valid_indices]
        base_weight = conf_full[valid_indices].clamp_min(1e-4)
        weight = base_weight.clone()
        theta = torch.zeros(3, 2, device=flow.device, dtype=flow.dtype)
        cutoff = torch.tensor(min_residual_px, device=flow.device, dtype=flow.dtype)
        for _ in range(max(1, irls_iterations)):
            theta = _weighted_affine_fit(design, target, weight, ridge)
            residual = (target - design @ theta).norm(dim=1)
            median = residual.median()
            scale = 1.4826 * (residual - median).abs().median().clamp_min(1e-4)
            cutoff = torch.maximum(
                torch.as_tensor(min_residual_px, device=flow.device, dtype=flow.dtype),
                huber_delta * scale,
            )
            robust_weight = torch.where(
                residual <= cutoff,
                torch.ones_like(residual),
                cutoff / residual.clamp_min(1e-6),
            )
            weight = base_weight * robust_weight

        predicted = (design_full @ theta).reshape(height, width, 2)
        residual_full = (flow[frame_index] - predicted).norm(dim=-1)
        inlier_confidence = (cutoff / residual_full.clamp_min(cutoff)).clamp(0, 1)
        fitted_confidence = confidence[frame_index] * inlier_confidence
        fitted_confidence = torch.where(
            torch.isfinite(residual_full), fitted_confidence, torch.zeros_like(fitted_confidence)
        )
        fitted.append(predicted)
        fit_confidences.append(fitted_confidence)

        valid_residual = residual_full[valid.reshape(height, width)]
        pair_diagnostics.append(
            {
                "frame_pair": frame_index,
                "valid": True,
                "fit_points": int(valid_indices.numel()),
                "coefficients": theta.cpu().tolist(),
                "huber_cutoff_px": float(cutoff),
                "residual_median_px": float(valid_residual.median()),
                "residual_mean_px": float(valid_residual.mean()),
                "confidence_mean": float(fitted_confidence.mean()),
            }
        )

    fitted_flow = torch.stack(fitted).to(observed_flow.dtype)
    fitted_confidence = torch.stack(fit_confidences).to(flow_confidence.dtype)
    return BackgroundMotionEstimate(
        flow=fitted_flow,
        confidence=fitted_confidence,
        uses_future_gt_ego=False,
        diagnostics={
            "provider": "robust_affine_irls_v1",
            "fit_valid_fraction": sum(bool(row["valid"]) for row in pair_diagnostics)
            / max(len(pair_diagnostics), 1),
            "pairs": pair_diagnostics,
        },
    )


@torch.no_grad()
def estimate_generated_geometry(
    mode: str,
    observed_flow: torch.Tensor,
    flow_confidence: torch.Tensor,
    depth: torch.Tensor,
    sample: dict[str, Any],
    *,
    fit_options: dict[str, Any] | None = None,
) -> BackgroundMotionEstimate:
    """按显式 mode 构造期望背景运动，并标注未来 GT 使用情况。"""
    if mode not in GENERATED_GEOMETRY_MODES:
        allowed = ", ".join(sorted(GENERATED_GEOMETRY_MODES))
        raise ValueError(f"未知 generated geometry mode: {mode!r}；允许 {allowed}")

    if mode == "estimated_background_motion":
        return fit_affine_background_flow(
            observed_flow,
            flow_confidence,
            **(fit_options or {}),
        )

    if mode == "identity_ego":
        return BackgroundMotionEstimate(
            flow=torch.zeros_like(observed_flow),
            confidence=flow_confidence.clone(),
            uses_future_gt_ego=False,
            diagnostics={"provider": "identity_ego_v1", "fit_valid_fraction": 1.0},
        )

    pose_key = "ego2global" if mode == "gt_ego_debug" else "control_ego2global"
    if pose_key not in sample:
        raise ValueError(f"{mode} 缺少必需条件 {pose_key}")
    expected, valid = compute_ego_flow(
        depth,
        sample["intrinsics"].to(depth.device),
        sample["cam2ego"].to(depth.device),
        sample[pose_key].to(depth.device),
        return_valid=True,
    )
    return BackgroundMotionEstimate(
        flow=expected.to(observed_flow.dtype),
        confidence=flow_confidence * valid.to(flow_confidence.dtype),
        uses_future_gt_ego=mode == "gt_ego_debug",
        diagnostics={
            "provider": "gt_ego_debug_v1" if mode == "gt_ego_debug" else "controlled_ego_v1",
            "fit_valid_fraction": float(valid.float().mean()),
        },
    )


def _sample_field(field: torch.Tensor, pixel_coordinates: torch.Tensor) -> torch.Tensor:
    height, width = field.shape[-3:-1]
    gx = 2.0 * pixel_coordinates[..., 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * pixel_coordinates[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    channels_first = field.permute(2, 0, 1).unsqueeze(0)
    return F.grid_sample(
        channels_first,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0].permute(1, 2, 0)


@torch.no_grad()
def render_pairwise_background_correction(
    frames: torch.Tensor,
    expected_flow: torch.Tensor,
    confidence: torch.Tensor,
    *,
    inverse_iterations: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按期望 pairwise flow 重绘背景；首帧及低置信区域保持 Base。"""
    frame_count, _, height, width = frames.shape
    if expected_flow.shape != (frame_count - 1, height, width, 2):
        raise ValueError("expected_flow 与 frames 形状不匹配")
    if confidence.shape != expected_flow.shape[:-1]:
        raise ValueError("confidence 与 expected_flow 形状不匹配")

    target = frames.clone()
    masks = torch.zeros(frame_count, 1, height, width, device=frames.device, dtype=frames.dtype)
    destination = pixel_grid(height, width, device=frames.device, dtype=expected_flow.dtype)
    for frame_index in range(frame_count - 1):
        source = destination.clone()
        pair_flow = expected_flow[frame_index]
        for _ in range(max(1, inverse_iterations)):
            source = destination - _sample_field(pair_flow, source)
        gx = 2.0 * source[..., 0] / max(width - 1, 1) - 1.0
        gy = 2.0 * source[..., 1] / max(height - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            frames[frame_index : frame_index + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0]
        sampled_confidence = _sample_field(confidence[frame_index, ..., None], source)[..., 0]
        in_bounds = (
            (source[..., 0] >= 0)
            & (source[..., 0] <= width - 1)
            & (source[..., 1] >= 0)
            & (source[..., 1] <= height - 1)
        )
        weight = (sampled_confidence * in_bounds.to(sampled_confidence.dtype)).clamp(0, 1)
        target[frame_index + 1] = (
            weight.unsqueeze(0) * warped
            + (1.0 - weight.unsqueeze(0)) * frames[frame_index + 1]
        )
        masks[frame_index + 1, 0] = weight.to(frames.dtype)

    return target.clamp(-1, 1), masks


def masked_flow_statistics(
    observed_flow: torch.Tensor,
    expected_flow: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float | None]:
    residual = (observed_flow.float() - expected_flow.float()).norm(dim=-1)
    valid = torch.isfinite(residual) & torch.isfinite(confidence) & (confidence > 0)
    if not bool(valid.any()):
        return {"mean": None, "median": None, "rms": None, "coverage": 0.0}
    values = residual[valid]
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "rms": float(values.square().mean().sqrt()),
        "coverage": float(valid.float().mean()),
    }

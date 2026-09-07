"""Appearance-owned Gaussian colors with an explicit stop at the physical boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AppearanceOutput:
    rgb: torch.Tensor
    opacity: torch.Tensor


class DetachedAppearanceAdapter(nn.Module):
    """Predict appearance parameters from visual tokens without consuming physical state."""

    def __init__(self, visual_feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(visual_feature_dim),
            nn.Linear(visual_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, visual_features: torch.Tensor, observed: torch.Tensor) -> AppearanceOutput:
        raw = self.network(visual_features)
        mask = observed.float().unsqueeze(-1)
        rgb = torch.sigmoid(raw[..., :3]) * mask
        opacity = torch.sigmoid(raw[..., 3]) * observed.float()
        return AppearanceOutput(rgb=rgb, opacity=opacity)


@dataclass(frozen=True)
class RenderedAppearance:
    rgb: torch.Tensor
    alpha: torch.Tensor
    valid_point_count: int


def render_detached_gaussian_appearance(
    world_points_m: torch.Tensor,
    colors_rgb: torch.Tensor,
    opacity: torch.Tensor,
    world_from_camera_opencv: torch.Tensor,
    intrinsics_px: torch.Tensor,
    *,
    height: int,
    width: int,
    sigma_px: float = 1.5,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RenderedAppearance:
    """Front-to-back Gaussian splat; RGB gradients cannot update points or camera pose."""

    if world_points_m.shape[-1] != 3 or colors_rgb.shape != world_points_m.shape:
        raise ValueError("world points and colors must have matching (N,3) shapes")
    if opacity.shape != world_points_m.shape[:-1]:
        raise ValueError("opacity must have shape (N,)")
    if height <= 0 or width <= 0 or sigma_px <= 0.0:
        raise ValueError("render dimensions and sigma must be positive")
    points = world_points_m.detach()
    camera_from_world = torch.linalg.inv(world_from_camera_opencv.detach())
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    camera = homogeneous @ camera_from_world.T
    projected = camera[:, :3] @ intrinsics_px.detach().T
    uv = projected[:, :2] / projected[:, 2:3].clamp_min(1.0e-8)
    valid = (
        (camera[:, 2] > 0.05)
        & (uv[:, 0] > -3.0 * sigma_px)
        & (uv[:, 0] < width + 3.0 * sigma_px)
        & (uv[:, 1] > -3.0 * sigma_px)
        & (uv[:, 1] < height + 3.0 * sigma_px)
    )
    background = torch.as_tensor(background_rgb, dtype=colors_rgb.dtype, device=colors_rgb.device)
    if not torch.any(valid):
        zero = (colors_rgb.sum() + opacity.sum()) * 0.0
        image = background[:, None, None].expand(3, height, width) + zero
        return RenderedAppearance(rgb=image, alpha=torch.zeros((height, width), device=image.device), valid_point_count=0)
    order = torch.argsort(camera[valid, 2].detach())
    uv = uv[valid][order]
    colors = colors_rgb[valid][order]
    opacity_valid = opacity[valid][order].clamp(0.0, 1.0)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=uv.dtype, device=uv.device),
        torch.arange(width, dtype=uv.dtype, device=uv.device),
        indexing="ij",
    )
    distance_squared = (xx[None] - uv[:, 0, None, None]).square() + (yy[None] - uv[:, 1, None, None]).square()
    footprint = torch.exp(-0.5 * distance_squared / float(sigma_px) ** 2)
    alpha = (opacity_valid[:, None, None] * footprint).clamp(0.0, 1.0 - 1.0e-6)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:1]), 1.0 - alpha[:-1]], dim=0),
        dim=0,
    )
    contribution = transmittance * alpha
    accumulated_alpha = contribution.sum(dim=0).clamp(0.0, 1.0)
    image = torch.sum(contribution[:, None] * colors[:, :, None, None], dim=0)
    image = image + (1.0 - accumulated_alpha)[None] * background[:, None, None]
    return RenderedAppearance(rgb=image, alpha=accumulated_alpha, valid_point_count=int(valid.sum().item()))

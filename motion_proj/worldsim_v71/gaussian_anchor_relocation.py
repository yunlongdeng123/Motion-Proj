"""GT-supervised Gaussian center residuals for Actor-local completion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class GaussianAnchorTargets:
    centers: torch.Tensor
    normals: torch.Tensor
    scales: torch.Tensor


class GaussianAnchorResidualMLP(nn.Module):
    """Predict a bounded 3D correction and one physical support scale per center."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(features)
        pooled = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        return self.head(torch.cat([encoded, pooled], dim=-1))


def initialize_from_relocation(
    model: GaussianAnchorResidualMLP,
    relocation: nn.Module,
    *,
    minimum_scale_m: float,
    maximum_scale_m: float,
    initial_scale_m: float,
) -> None:
    """Reuse M5 build-evidence features while starting the center correction at zero."""

    model.point_encoder.load_state_dict(relocation.point_encoder.state_dict())
    model.head[0].load_state_dict(relocation.head[0].state_dict())
    low = math.log(float(minimum_scale_m))
    high = math.log(float(maximum_scale_m))
    fraction = (math.log(float(initial_scale_m)) - low) / max(high - low, 1.0e-8)
    fraction = min(max(fraction, 1.0e-4), 1.0 - 1.0e-4)
    scale_bias = math.log(fraction / (1.0 - fraction))
    with torch.no_grad():
        model.head[2].weight.zero_()
        model.head[2].bias.zero_()
        model.head[2].bias[3] = scale_bias


def apply_gaussian_anchor_residual(
    base_centers: torch.Tensor,
    raw_prediction: torch.Tensor,
    *,
    maximum_residual_xyz_m: Sequence[float],
    actor_half_size_m: torch.Tensor,
    cuboid_padding_m: float,
    minimum_scale_m: float,
    maximum_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_prediction.shape[-1] != 4:
        raise ValueError("Gaussian anchor prediction must have xyz residual + scale")
    maximum = torch.as_tensor(
        maximum_residual_xyz_m,
        dtype=base_centers.dtype,
        device=base_centers.device,
    ).reshape(1, 3)
    residual = torch.tanh(raw_prediction[:, :3]) * maximum
    bounds = actor_half_size_m.reshape(1, 3) + float(cuboid_padding_m)
    centers = torch.maximum(torch.minimum(base_centers + residual, bounds), -bounds)
    log_minimum = math.log(float(minimum_scale_m))
    log_maximum = math.log(float(maximum_scale_m))
    log_scale = log_minimum + torch.sigmoid(raw_prediction[:, 3]) * (
        log_maximum - log_minimum
    )
    return centers, residual, log_scale.exp()


def build_gaussian_anchor_targets(
    candidates: torch.Tensor,
    target_surface: torch.Tensor,
    *,
    neighbors: int,
    minimum_scale_m: float,
    maximum_scale_m: float,
) -> GaussianAnchorTargets:
    """Construct deterministic actor-canonical center, tangent-plane and scale labels."""

    if len(candidates) == 0 or len(target_surface) == 0:
        raise ValueError("Gaussian target construction requires candidates and target surface")
    match_indices = torch.cdist(candidates, target_surface).argmin(dim=1)
    centers = target_surface.index_select(0, match_indices)
    count = min(max(int(neighbors), 1), len(target_surface))
    neighbor_indices = torch.cdist(centers, target_surface).topk(
        count, largest=False
    ).indices
    local = target_surface[neighbor_indices]
    centered = local - local.mean(dim=1, keepdim=True)
    covariance = torch.matmul(centered.transpose(1, 2), centered) / max(count, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    distances = torch.linalg.vector_norm(local - centers[:, None, :], dim=-1)
    scales = distances.median(dim=1).values.clamp(
        min=float(minimum_scale_m), max=float(maximum_scale_m)
    )
    return GaussianAnchorTargets(centers=centers, normals=normals, scales=scales)

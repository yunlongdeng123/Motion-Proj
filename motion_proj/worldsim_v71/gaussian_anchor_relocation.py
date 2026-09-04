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


class GaussianSeedExpansionMLP(nn.Module):
    """Expand every build-conditioned M5 seed into a fixed set of child centers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        branch_factor: int = 4,
        slot_dim: int = 16,
    ) -> None:
        super().__init__()
        self.branch_factor = int(branch_factor)
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.branch_factor, int(slot_dim)) * 0.02
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + int(slot_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(features)
        pooled = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        parent = torch.cat([encoded, pooled], dim=-1)
        parent = parent[:, None, :].expand(-1, self.branch_factor, -1)
        slots = self.slot_embeddings[None, :, :].expand(len(features), -1, -1)
        return self.head(torch.cat([parent, slots], dim=-1))


class OrientedGaussianSeedExpansionMLP(nn.Module):
    """Expand each seed into an oriented oblate Gaussian surface primitive."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        branch_factor: int = 4,
        slot_dim: int = 16,
    ) -> None:
        super().__init__()
        self.branch_factor = int(branch_factor)
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.branch_factor, int(slot_dim)) * 0.02
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + int(slot_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(features)
        pooled = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        parent = torch.cat([encoded, pooled], dim=-1)
        parent = parent[:, None, :].expand(-1, self.branch_factor, -1)
        slots = self.slot_embeddings[None, :, :].expand(len(features), -1, -1)
        return self.head(torch.cat([parent, slots], dim=-1))


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


def initialize_expansion_from_relocation(
    model: GaussianSeedExpansionMLP,
    relocation: nn.Module,
    *,
    minimum_scale_m: float,
    maximum_scale_m: float,
    initial_scale_m: float,
) -> None:
    """Reuse M5 features and start all child coordinates exactly at their parent."""

    model.point_encoder.load_state_dict(relocation.point_encoder.state_dict())
    source_weight = relocation.head[0].weight
    with torch.no_grad():
        model.head[0].weight[:, : source_weight.shape[1]].copy_(source_weight)
        model.head[0].bias.copy_(relocation.head[0].bias)
        model.head[2].weight.zero_()
        model.head[2].bias.zero_()
        low = math.log(float(minimum_scale_m))
        high = math.log(float(maximum_scale_m))
        fraction = (math.log(float(initial_scale_m)) - low) / max(high - low, 1.0e-8)
        fraction = min(max(fraction, 1.0e-4), 1.0 - 1.0e-4)
        model.head[2].bias[3] = math.log(fraction / (1.0 - fraction))


def initialize_oriented_from_expansion(
    model: OrientedGaussianSeedExpansionMLP,
    expansion: GaussianSeedExpansionMLP,
    *,
    minimum_thickness_m: float,
    maximum_thickness_m: float,
    initial_thickness_m: float,
) -> None:
    """Preserve M8 centers/scales and initialize normals from parent PCA evidence."""

    model.point_encoder.load_state_dict(expansion.point_encoder.state_dict())
    model.slot_embeddings.data.copy_(expansion.slot_embeddings.data)
    model.head[0].load_state_dict(expansion.head[0].state_dict())
    low = math.log(float(minimum_thickness_m))
    high = math.log(float(maximum_thickness_m))
    fraction = (math.log(float(initial_thickness_m)) - low) / max(high - low, 1.0e-8)
    fraction = min(max(fraction, 1.0e-4), 1.0 - 1.0e-4)
    with torch.no_grad():
        model.head[2].weight.zero_()
        model.head[2].bias.zero_()
        model.head[2].weight[:4].copy_(expansion.head[2].weight)
        model.head[2].bias[:4].copy_(expansion.head[2].bias)
        model.head[2].bias[7] = math.log(fraction / (1.0 - fraction))


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


def apply_gaussian_seed_expansion(
    base_centers: torch.Tensor,
    raw_prediction: torch.Tensor,
    *,
    maximum_residual_xyz_m: Sequence[float],
    actor_half_size_m: torch.Tensor,
    cuboid_padding_m: float,
    minimum_scale_m: float,
    maximum_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_prediction.ndim != 3 or raw_prediction.shape[-1] != 4:
        raise ValueError("Seed expansion prediction must be [parent, child, xyz+scale]")
    maximum = torch.as_tensor(
        maximum_residual_xyz_m,
        dtype=base_centers.dtype,
        device=base_centers.device,
    ).reshape(1, 1, 3)
    residual = torch.tanh(raw_prediction[..., :3]) * maximum
    children = base_centers[:, None, :] + residual
    bounds = actor_half_size_m.reshape(1, 1, 3) + float(cuboid_padding_m)
    children = torch.maximum(torch.minimum(children, bounds), -bounds)
    log_minimum = math.log(float(minimum_scale_m))
    log_maximum = math.log(float(maximum_scale_m))
    log_scale = log_minimum + torch.sigmoid(raw_prediction[..., 3]) * (
        log_maximum - log_minimum
    )
    return (
        children.reshape(-1, 3),
        residual.reshape(-1, 3),
        log_scale.exp().reshape(-1),
    )


def apply_oriented_gaussian_seed_expansion(
    base_centers: torch.Tensor,
    parent_normals: torch.Tensor,
    raw_prediction: torch.Tensor,
    *,
    maximum_residual_xyz_m: Sequence[float],
    normal_residual_bound: float,
    actor_half_size_m: torch.Tensor,
    cuboid_padding_m: float,
    minimum_scale_m: float,
    maximum_scale_m: float,
    minimum_thickness_m: float,
    maximum_thickness_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_prediction.ndim != 3 or raw_prediction.shape[-1] != 8:
        raise ValueError(
            "Oriented expansion must be [parent, child, xyz+scale+normal+thickness]"
        )
    if len(parent_normals) != len(base_centers):
        raise ValueError("Parent centers and normals must align")
    maximum = torch.as_tensor(
        maximum_residual_xyz_m,
        dtype=base_centers.dtype,
        device=base_centers.device,
    ).reshape(1, 1, 3)
    residual = torch.tanh(raw_prediction[..., :3]) * maximum
    children = base_centers[:, None, :] + residual
    bounds = actor_half_size_m.reshape(1, 1, 3) + float(cuboid_padding_m)
    children = torch.maximum(torch.minimum(children, bounds), -bounds)

    scale_low = math.log(float(minimum_scale_m))
    scale_high = math.log(float(maximum_scale_m))
    log_scale = scale_low + torch.sigmoid(raw_prediction[..., 3]) * (
        scale_high - scale_low
    )
    normal_delta = torch.tanh(raw_prediction[..., 4:7]) * float(normal_residual_bound)
    normals = torch.nn.functional.normalize(
        parent_normals[:, None, :] + normal_delta, dim=-1, eps=1.0e-6
    )
    thickness_low = math.log(float(minimum_thickness_m))
    thickness_high = math.log(float(maximum_thickness_m))
    log_thickness = thickness_low + torch.sigmoid(raw_prediction[..., 7]) * (
        thickness_high - thickness_low
    )
    return (
        children.reshape(-1, 3),
        residual.reshape(-1, 3),
        log_scale.exp().reshape(-1),
        normals.reshape(-1, 3),
        log_thickness.exp().reshape(-1),
    )


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

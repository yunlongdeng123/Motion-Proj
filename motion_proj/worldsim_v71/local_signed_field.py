"""Query-specific local signed field guided by frozen M8 surface iso-points."""

from __future__ import annotations

import torch
from torch import nn

from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP


class LocalAnchorSignedField(nn.Module):
    """Blend learned local implicit patches instead of pooling an Actor to one latent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        branch_factor: int = 4,
        slot_dim: int = 16,
        latent_dim: int = 64,
        neighbor_count: int = 4,
        maximum_residual_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self.branch_factor = int(branch_factor)
        self.neighbor_count = int(neighbor_count)
        self.maximum_residual_fraction = float(maximum_residual_fraction)
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.branch_factor, int(slot_dim)) * 0.02
        )
        self.child_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + int(slot_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(latent_dim)),
            nn.ReLU(),
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(int(latent_dim) + 4, hidden_dim),
            nn.Softplus(beta=20.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=20.0),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            self.query_decoder[-1].weight.zero_()
            self.query_decoder[-1].bias.zero_()

    def child_features(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(features)
        pooled = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        parent = torch.cat([encoded, pooled], dim=-1)
        parent = parent[:, None, :].expand(-1, self.branch_factor, -1)
        slots = self.slot_embeddings[None, :, :].expand(len(features), -1, -1)
        return self.child_encoder(torch.cat([parent, slots], dim=-1)).reshape(
            -1, self.child_encoder[2].out_features
        )

    def outward_child_normals(
        self, parent_normals: torch.Tensor, parent_ray_directions: torch.Tensor
    ) -> torch.Tensor:
        normals = torch.nn.functional.normalize(parent_normals, dim=1, eps=1.0e-6)
        outward = -torch.nn.functional.normalize(
            parent_ray_directions, dim=1, eps=1.0e-6
        )
        aligned = torch.where(
            torch.sum(normals * outward, dim=1, keepdim=True) >= 0.0,
            normals,
            -normals,
        )
        return aligned[:, None, :].expand(-1, self.branch_factor, -1).reshape(-1, 3)

    def forward(
        self,
        features: torch.Tensor,
        child_centers: torch.Tensor,
        child_scales: torch.Tensor,
        parent_normals: torch.Tensor,
        parent_ray_directions: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        child_latents = self.child_features(features)
        child_normals = self.outward_child_normals(
            parent_normals, parent_ray_directions
        )
        child_scales = child_scales.reshape(-1).clamp_min(1.0e-4)
        if not (
            len(child_centers)
            == len(child_scales)
            == len(child_normals)
            == len(child_latents)
        ):
            raise ValueError("Local-field child attributes must align")
        count = min(max(self.neighbor_count, 1), len(child_centers))
        normalized_distance = torch.cdist(queries, child_centers) / child_scales[None, :]
        distances, indices = normalized_distance.topk(count, largest=False)
        centers = child_centers[indices]
        scales = child_scales[indices]
        normals = child_normals[indices]
        latents = child_latents[indices]
        relative = (queries[:, None, :] - centers) / scales[:, :, None]
        base_normalized = torch.sum(relative * normals, dim=-1)
        decoder_input = torch.cat(
            [latents, relative, base_normalized[:, :, None]], dim=-1
        )
        residual = (
            torch.tanh(self.query_decoder(decoder_input).squeeze(-1))
            * scales
            * self.maximum_residual_fraction
        )
        local_field = base_normalized * scales + residual
        weights = torch.softmax(-distances, dim=1)
        return torch.sum(weights * local_field, dim=1)


class CompactLocalOccupancyField(LocalAnchorSignedField):
    """Union compact half-ball patches so unanchored space remains explicitly free."""

    def forward(
        self,
        features: torch.Tensor,
        child_centers: torch.Tensor,
        child_scales: torch.Tensor,
        parent_normals: torch.Tensor,
        parent_ray_directions: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        child_latents = self.child_features(features)
        child_normals = self.outward_child_normals(
            parent_normals, parent_ray_directions
        )
        child_scales = child_scales.reshape(-1).clamp_min(1.0e-4)
        if not (
            len(child_centers)
            == len(child_scales)
            == len(child_normals)
            == len(child_latents)
        ):
            raise ValueError("Compact-field child attributes must align")
        count = min(max(self.neighbor_count, 1), len(child_centers))
        metric_distance = torch.cdist(queries, child_centers)
        normalized_distance = metric_distance / child_scales[None, :]
        _, indices = normalized_distance.topk(count, largest=False)
        centers = child_centers[indices]
        scales = child_scales[indices]
        normals = child_normals[indices]
        latents = child_latents[indices]
        relative_metric = queries[:, None, :] - centers
        relative = relative_metric / scales[:, :, None]
        plane = torch.sum(relative_metric * normals, dim=-1)
        decoder_input = torch.cat(
            [latents, relative, (plane / scales)[:, :, None]], dim=-1
        )
        residual = (
            torch.tanh(self.query_decoder(decoder_input).squeeze(-1))
            * scales
            * self.maximum_residual_fraction
        )
        radial_support = torch.linalg.vector_norm(relative_metric, dim=-1) - scales
        local_patch = torch.maximum(plane + residual, radial_support)
        return local_patch.min(dim=1).values


class OneSidedLocalOccupancyField(LocalAnchorSignedField):
    """Union finite surface cells with no support thickness on the free side."""

    def __init__(
        self,
        *args: object,
        maximum_log_radius_delta: float = 0.6931471805599453,
        back_support_depth_m: float = 0.10,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_log_radius_delta = float(maximum_log_radius_delta)
        self.back_support_depth_m = float(back_support_depth_m)
        latent_dim = int(self.child_encoder[2].out_features)
        self.radius_head = nn.Linear(latent_dim, 1)
        with torch.no_grad():
            self.radius_head.weight.zero_()
            self.radius_head.bias.zero_()

    def radii_from_latents(
        self, child_latents: torch.Tensor, child_scales: torch.Tensor
    ) -> torch.Tensor:
        base = child_scales.reshape(-1).clamp_min(1.0e-4)
        log_delta = self.maximum_log_radius_delta * torch.tanh(
            self.radius_head(child_latents).squeeze(-1)
        )
        return base * torch.exp(log_delta)

    def child_radii(
        self, features: torch.Tensor, child_scales: torch.Tensor
    ) -> torch.Tensor:
        return self.radii_from_latents(self.child_features(features), child_scales)

    def forward(
        self,
        features: torch.Tensor,
        child_centers: torch.Tensor,
        child_scales: torch.Tensor,
        parent_normals: torch.Tensor,
        parent_ray_directions: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        child_latents = self.child_features(features)
        child_normals = self.outward_child_normals(
            parent_normals, parent_ray_directions
        )
        child_scales = child_scales.reshape(-1).clamp_min(1.0e-4)
        child_radii = self.radii_from_latents(child_latents, child_scales)
        if not (
            len(child_centers)
            == len(child_scales)
            == len(child_radii)
            == len(child_normals)
            == len(child_latents)
        ):
            raise ValueError("One-sided field child attributes must align")
        count = min(max(self.neighbor_count, 1), len(child_centers))
        normalized_distance = torch.cdist(queries, child_centers) / child_scales[None, :]
        _, indices = normalized_distance.topk(count, largest=False)
        centers = child_centers[indices]
        scales = child_scales[indices]
        radii = child_radii[indices]
        normals = child_normals[indices]
        latents = child_latents[indices]
        relative_metric = queries[:, None, :] - centers
        relative = relative_metric / scales[:, :, None]
        plane = torch.sum(relative_metric * normals, dim=-1)
        decoder_input = torch.cat(
            [latents, relative, (plane / scales)[:, :, None]], dim=-1
        )
        residual = (
            torch.tanh(self.query_decoder(decoder_input).squeeze(-1))
            * scales
            * self.maximum_residual_fraction
        )
        shifted_plane = plane + residual
        tangent = torch.linalg.vector_norm(
            relative_metric - plane[:, :, None] * normals, dim=-1
        )
        tangent_boundary = tangent - radii
        back_boundary = -shifted_plane - self.back_support_depth_m
        local_cell = torch.maximum(
            torch.maximum(shifted_plane, tangent_boundary), back_boundary
        )
        return local_cell.min(dim=1).values


def initialize_local_field_from_expansion(
    model: LocalAnchorSignedField, expansion: GaussianSeedExpansionMLP
) -> None:
    """Reuse M8 local evidence features while keeping the initial signed residual zero."""

    model.point_encoder.load_state_dict(expansion.point_encoder.state_dict())
    with torch.no_grad():
        model.slot_embeddings.copy_(expansion.slot_embeddings)
        model.child_encoder[0].load_state_dict(expansion.head[0].state_dict())
        model.query_decoder[-1].weight.zero_()
        model.query_decoder[-1].bias.zero_()

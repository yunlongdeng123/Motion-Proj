"""GT-supervised continuous authority masses for frozen physical Gaussians."""

from __future__ import annotations

import torch
from torch import nn


def primitive_authority_features(
    centers: torch.Tensor,
    scales: torch.Tensor,
    is_anchor: torch.Tensor,
    actor_size_lwh_m: torch.Tensor,
) -> torch.Tensor:
    """Build geometry-only features; trajectory, image, hazard, and target are absent."""
    size = actor_size_lwh_m.reshape(3).clamp_min(0.10)
    half = 0.5 * size
    normalized = centers / half.reshape(1, 3)
    scale_fraction = scales.reshape(-1, 1) / size.mean()
    anchor = is_anchor.reshape(-1, 1).to(dtype=centers.dtype)
    primitive_type = torch.cat([anchor, 1.0 - anchor], dim=1)
    actor_size = torch.log1p(size).reshape(1, 3).expand(len(centers), -1)
    return torch.cat(
        [normalized, normalized.abs(), scale_fraction, primitive_type, actor_size],
        dim=1,
    )


class EvidentialGaussianAuthority(nn.Module):
    """PointNet-style primitive scorer with FREE/OCCUPIED/UNKNOWN output masses."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.primitive_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.authority_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.primitive_encoder(features)
        context = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        return self.authority_head(torch.cat([encoded, context], dim=1))


def occupied_masses(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("authority logits must be [N,3] FREE/OCCUPIED/UNKNOWN")
    return torch.softmax(logits, dim=1)[:, 1]


def weighted_gaussian_energy(
    queries: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    occupied: torch.Tensor,
) -> torch.Tensor:
    """Continuous occupied mass weights metric Gaussian energy without deletion."""
    normalized_distance = torch.cdist(queries, centers) / scales.reshape(1, -1)
    log_components = (
        -0.5 * normalized_distance.square()
        + torch.log(occupied.clamp_min(1.0e-8)).reshape(1, -1)
    )
    return torch.logsumexp(log_components, dim=1)

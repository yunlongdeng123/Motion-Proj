"""单一ray-signed Actor表面level-set。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RaySignedSurfaceField(nn.Module):
    """PointNet evidence encoder与单一signed-distance decoder。"""

    def __init__(
        self,
        evidence_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        frequency_count: int = 4,
        maximum_distance_m: float = 0.48,
    ) -> None:
        super().__init__()
        self.maximum_distance_m = float(maximum_distance_m)
        self.register_buffer(
            "frequencies",
            torch.pow(2.0, torch.arange(int(frequency_count), dtype=torch.float32)) * torch.pi,
        )
        self.point_encoder = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        position_dim = 3 + 6 * int(frequency_count)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + position_dim + 3, hidden_dim),
            nn.Softplus(beta=100.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100.0),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, evidence_features: torch.Tensor) -> torch.Tensor:
        return self.point_encoder(evidence_features).max(dim=0).values

    def _position(self, normalized_xyz: torch.Tensor) -> torch.Tensor:
        angles = normalized_xyz[:, :, None] * self.frequencies[None, None, :]
        return torch.cat(
            [normalized_xyz, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)],
            dim=1,
        )

    def decode(
        self,
        latent: torch.Tensor,
        query_normalized_xyz: torch.Tensor,
        actor_size_lwh_m: torch.Tensor,
    ) -> torch.Tensor:
        count = len(query_normalized_xyz)
        features = torch.cat(
            [
                latent.reshape(1, -1).expand(count, -1),
                self._position(query_normalized_xyz),
                actor_size_lwh_m.reshape(1, 3).expand(count, -1),
            ],
            dim=1,
        )
        return torch.tanh(self.decoder(features).squeeze(1)) * self.maximum_distance_m

    def forward(
        self,
        evidence_features: torch.Tensor,
        query_normalized_xyz: torch.Tensor,
        actor_size_lwh_m: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(
            self.encode(evidence_features), query_normalized_xyz, actor_size_lwh_m
        )


class RayOccupancyBoundaryField(RaySignedSurfaceField):
    """以单一无界occupancy logit的零等值面表示surface。"""

    def __init__(
        self,
        evidence_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        frequency_count: int = 4,
    ) -> None:
        super().__init__(
            evidence_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            frequency_count=frequency_count,
            maximum_distance_m=1.0,
        )

    def decode(
        self,
        latent: torch.Tensor,
        query_normalized_xyz: torch.Tensor,
        actor_size_lwh_m: torch.Tensor,
    ) -> torch.Tensor:
        count = len(query_normalized_xyz)
        features = torch.cat(
            [
                latent.reshape(1, -1).expand(count, -1),
                self._position(query_normalized_xyz),
                actor_size_lwh_m.reshape(1, 3).expand(count, -1),
            ],
            dim=1,
        )
        return self.decoder(features).squeeze(1)


def extract_signed_zero_crossings(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    signed_distance: torch.Tensor,
) -> torch.Tensor:
    """在线性grid edge上插值SDF零点，不使用额外evidence gate。"""

    shape = tuple(len(axis) for axis in axes)
    if int(np.prod(shape)) != len(signed_distance):
        raise ValueError("grid shape与SDF数量不一致")
    device = signed_distance.device
    dtype = signed_distance.dtype
    coordinates = torch.stack(
        torch.meshgrid(
            *[torch.as_tensor(axis, dtype=dtype, device=device) for axis in axes],
            indexing="ij",
        ),
        dim=-1,
    )
    field = signed_distance.reshape(shape)
    crossings: list[torch.Tensor] = []
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        left = field[tuple(left_slice)]
        right = field[tuple(right_slice)]
        mask = (left <= 0.0) != (right <= 0.0)
        if not torch.any(mask):
            continue
        left_points = coordinates[tuple(left_slice)][mask]
        right_points = coordinates[tuple(right_slice)][mask]
        left_values = left[mask]
        right_values = right[mask]
        weight = left_values.abs() / (left_values.abs() + right_values.abs()).clamp_min(1.0e-6)
        crossings.append(left_points + weight[:, None] * (right_points - left_points))
    if not crossings:
        return torch.empty((0, 3), dtype=dtype, device=device)
    return torch.cat(crossings, dim=0)

"""M0：低容量 Actor-local 射线/法向表面位移网络。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v71.actor_canonical import normalized_actor_coordinates


M0_EXTRA_FEATURE_NAMES = (
    "actor_normalized_x",
    "actor_normalized_y",
    "actor_normalized_z",
    "free_evidence_mass",
    "occupied_evidence_mass",
    "unknown_evidence_mass",
    "normal_x",
    "normal_y",
    "normal_z",
    "ray_x",
    "ray_y",
    "ray_z",
    "local_neighbor_radius_over_diagonal",
    "local_normal_variation",
)


def estimate_surface_normals(points: torch.Tensor, reference: torch.Tensor | None = None, neighbors: int = 8) -> torch.Tensor:
    if len(points) == 0:
        return torch.empty_like(points)
    reference = points if reference is None else reference
    k = min(max(int(neighbors), 3), len(reference))
    distances = torch.cdist(points, reference)
    indices = distances.topk(k, largest=False).indices
    neighborhoods = reference[indices]
    centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / max(k - 1, 1)
    normals = torch.linalg.eigh(covariance).eigenvectors[:, :, 0]
    sign = torch.where((normals * points).sum(dim=1, keepdim=True) >= 0.0, 1.0, -1.0)
    return normals * sign


def local_surface_summary(points: torch.Tensor, normals: torch.Tensor, neighbors: int = 8) -> torch.Tensor:
    if len(points) == 0:
        return torch.empty((0, 2), dtype=points.dtype, device=points.device)
    k = min(max(int(neighbors), 2), len(points))
    distances = torch.cdist(points, points)
    values, indices = distances.topk(k, largest=False)
    neighbor_normals = normals[indices]
    alignment = torch.abs((neighbor_normals * normals[:, None, :]).sum(dim=-1))
    radius = values[:, -1]
    variation = 1.0 - alignment.mean(dim=1)
    return torch.stack([radius, variation], dim=1)


def build_m0_features(
    base_features: np.ndarray,
    candidates: np.ndarray,
    size_lwh_m: np.ndarray,
    evidence_masses: np.ndarray,
    query_sensor_origin: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32, device=device).reshape(-1, 3)
    normals = estimate_surface_normals(candidate_tensor)
    origin = torch.as_tensor(query_sensor_origin, dtype=torch.float32, device=device).reshape(1, 3)
    rays = candidate_tensor - origin
    rays = rays / torch.linalg.vector_norm(rays, dim=1, keepdim=True).clamp_min(1.0e-6)
    local = local_surface_summary(candidate_tensor, normals)
    diagonal = max(float(np.linalg.norm(size_lwh_m)), 1.0e-6)
    local[:, 0] /= diagonal
    extras = np.concatenate(
        [
            normalized_actor_coordinates(candidates, size_lwh_m),
            np.asarray(evidence_masses, dtype=np.float32),
            normals.detach().cpu().numpy(),
            rays.detach().cpu().numpy(),
            local.detach().cpu().numpy(),
        ],
        axis=1,
    )
    features = np.concatenate([np.asarray(base_features, dtype=np.float32), extras], axis=1)
    return features.astype(np.float32), rays.detach().cpu().numpy(), normals.detach().cpu().numpy()


class RaySurfaceDisplacementMLP(nn.Module):
    """共享 Actor pooled context 的 PointNet residual。"""

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
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(features)
        pooled = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        return self.head(torch.cat([encoded, pooled], dim=-1))


def apply_predicted_displacement(
    candidates: torch.Tensor,
    ray_directions: torch.Tensor,
    normals: torch.Tensor,
    prediction: torch.Tensor,
    *,
    maximum_ray_displacement_m: float,
    maximum_normal_displacement_m: float,
    actor_half_size_m: torch.Tensor,
    cuboid_padding_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ray_delta = torch.tanh(prediction[:, 0]) * float(maximum_ray_displacement_m)
    normal_delta = torch.tanh(prediction[:, 1]) * float(maximum_normal_displacement_m)
    displacement = ray_delta[:, None] * ray_directions + normal_delta[:, None] * normals
    moved = candidates + displacement
    bounds = actor_half_size_m.reshape(1, 3) + float(cuboid_padding_m)
    moved = torch.maximum(torch.minimum(moved, bounds), -bounds)
    unknown_probability = torch.sigmoid(prediction[:, 2])
    return moved, unknown_probability


def hard_collision_surface(
    observed_anchors: torch.Tensor,
    moved_candidates: torch.Tensor,
    unknown_probability: torch.Tensor,
    *,
    unknown_threshold: float = 0.5,
) -> torch.Tensor:
    accepted = moved_candidates[unknown_probability < float(unknown_threshold)]
    return torch.cat([observed_anchors, accepted], dim=0)


def displacement_loss_weights(config: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        float(config.get("first_return", 1.0)),
        float(config.get("surface", 1.0)),
        float(config.get("anchor", 0.05)),
        float(config.get("temporal", 0.05)),
    )

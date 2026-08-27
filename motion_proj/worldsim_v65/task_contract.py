"""Continuous trajectory task contract used by the first V6.5 research slice."""

from __future__ import annotations

import numpy as np
import torch


TRAJECTORY_FEATURE_NAMES = (
    "distance_to_trajectory_m",
    "signed_lateral_offset_m",
    "along_trajectory_m",
    "nearest_horizon_fraction",
    "local_tangent_x",
    "local_tangent_y",
    "trajectory_length_m",
    "height_m",
    "radial_distance_m",
    "route_relevance_soft",
)


def continuous_trajectory_features(
    centers_m: np.ndarray, route_xy: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Encode point-to-polyline geometry without a hard route-corridor feature."""
    centers = torch.as_tensor(centers_m, dtype=torch.float32, device=device)
    future = torch.as_tensor(route_xy, dtype=torch.float32, device=device)
    origin = torch.zeros((1, 2), dtype=torch.float32, device=device)
    polyline = torch.cat((origin, future), dim=0)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    lengths = torch.linalg.vector_norm(vectors, dim=1).clamp_min(1e-4)
    relative = centers[:, None, :2] - starts[None]
    fractions = (
        (relative * vectors[None]).sum(dim=-1) / vectors.square().sum(dim=-1).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    projections = starts[None] + fractions[..., None] * vectors[None]
    squared_distance = (centers[:, None, :2] - projections).square().sum(dim=-1)
    nearest = squared_distance.argmin(dim=1)
    rows = torch.arange(centers.shape[0], device=device)
    nearest_fraction = fractions[rows, nearest]
    nearest_vector = vectors[nearest]
    nearest_length = lengths[nearest]
    offset = centers[:, :2] - projections[rows, nearest]
    signed_lateral = (
        nearest_vector[:, 0] * offset[:, 1] - nearest_vector[:, 1] * offset[:, 0]
    ) / nearest_length
    cumulative = torch.cat(
        (torch.zeros(1, device=device), torch.cumsum(lengths, dim=0)[:-1])
    )
    along = cumulative[nearest] + nearest_fraction * nearest_length
    horizon_fraction = (nearest.float() + nearest_fraction) / max(1, vectors.shape[0])
    tangent = nearest_vector / nearest_length[:, None]
    distance = squared_distance[rows, nearest].sqrt()
    route_length = lengths.sum().expand_as(distance)
    radial = torch.linalg.vector_norm(centers[:, :2], dim=1)
    relevance = torch.exp(-distance / 3.0)
    return torch.stack(
        (
            distance,
            signed_lateral,
            along,
            horizon_fraction,
            tangent[:, 0],
            tangent[:, 1],
            route_length,
            centers[:, 2],
            radial,
            relevance,
        ),
        dim=1,
    )

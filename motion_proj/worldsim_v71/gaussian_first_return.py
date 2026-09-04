"""Order-aware first-return distributions from analytic Gaussian ray integrals."""

from __future__ import annotations

import math

import torch


def gaussian_segment_optical_thickness(
    origins: torch.Tensor,
    directions: torch.Tensor,
    depth_edges: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    occupied_masses: torch.Tensor,
) -> torch.Tensor:
    """Integrate normalized isotropic Gaussian line densities in each ray segment."""
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins must be [R,3]")
    if directions.shape != origins.shape:
        raise ValueError("directions must match origins")
    if depth_edges.ndim != 2 or depth_edges.shape[0] != len(origins):
        raise ValueError("depth_edges must be [R,K+1]")
    centers = centers.reshape(-1, 3)
    scales = scales.reshape(-1).clamp_min(1.0e-4)
    occupied_masses = occupied_masses.reshape(-1).clamp_min(0.0)
    if len(centers) != len(scales) or len(centers) != len(occupied_masses):
        raise ValueError("center, scale, and occupied arrays must align")

    relative = centers[None, :, :] - origins[:, None, :]
    axial_mean = torch.sum(relative * directions[:, None, :], dim=-1)
    squared_range = torch.sum(relative.square(), dim=-1)
    perpendicular_squared = (squared_range - axial_mean.square()).clamp_min(0.0)
    amplitude = occupied_masses[None, :] * torch.exp(
        -0.5 * perpendicular_squared / scales.square()[None, :]
    )
    normalized_edges = (
        depth_edges[:, :, None] - axial_mean[:, None, :]
    ) / scales[None, None, :]
    cdf = 0.5 * (1.0 + torch.erf(normalized_edges / math.sqrt(2.0)))
    segment_integrals = (cdf[:, 1:, :] - cdf[:, :-1, :]).clamp_min(0.0)
    return torch.sum(amplitude[:, None, :] * segment_integrals, dim=-1)


def conditional_termination_log_probabilities(
    optical_thickness: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered T*alpha log probabilities conditioned on a return in the box."""
    optical_thickness = optical_thickness.clamp_min(0.0)
    prefix = torch.cat(
        [
            torch.zeros(
                (len(optical_thickness), 1),
                dtype=optical_thickness.dtype,
                device=optical_thickness.device,
            ),
            torch.cumsum(optical_thickness[:, :-1], dim=1),
        ],
        dim=1,
    )
    log_alpha = torch.log(
        (-torch.expm1(-optical_thickness)).clamp_min(1.0e-12)
    )
    log_weights = -prefix + log_alpha
    log_probabilities = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
    no_return_probability = torch.exp(-torch.sum(optical_thickness, dim=1))
    return log_probabilities, no_return_probability


def gaussian_conditional_termination_log_probabilities(
    origins: torch.Tensor,
    directions: torch.Tensor,
    depth_edges: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    occupied_masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    optical_thickness = gaussian_segment_optical_thickness(
        origins,
        directions,
        depth_edges,
        centers,
        scales,
        occupied_masses,
    )
    log_probabilities, no_return_probability = conditional_termination_log_probabilities(
        optical_thickness
    )
    return log_probabilities, no_return_probability, optical_thickness

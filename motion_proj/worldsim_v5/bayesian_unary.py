"""V5 reliability-aware effective-count Bayesian unary。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .evidence_schema import validate_observation_chunk


def observation_reliability(
    observations: Mapping[str, np.ndarray],
    *,
    sam_confidence_floor: float,
    boundary_distance_scale_px: float,
    depth_residual_scale_m: float,
) -> np.ndarray:
    for name, value in {
        "sam_confidence_floor": sam_confidence_floor,
        "boundary_distance_scale_px": boundary_distance_scale_px,
        "depth_residual_scale_m": depth_residual_scale_m,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须有限且大于 0")
    if sam_confidence_floor > 1.0:
        raise ValueError("sam_confidence_floor 必须不大于 1")
    sam_probability = np.asarray(observations["sam_probability"], dtype=np.float64)
    sam_confidence = sam_confidence_floor + (1.0 - sam_confidence_floor) * np.abs(
        2.0 * sam_probability - 1.0
    )
    visibility = np.clip(np.asarray(observations["visibility"], dtype=np.float64), 0.0, 1.0)
    boundary = 1.0 - np.exp(
        -np.abs(np.asarray(observations["mask_boundary_distance"], dtype=np.float64))
        / boundary_distance_scale_px
    )
    depth = np.exp(
        -np.abs(np.asarray(observations["depth_residual"], dtype=np.float64))
        / depth_residual_scale_m
    )
    view_angle = np.clip(
        np.asarray(observations["view_angle_cosine"], dtype=np.float64), 0.0, 1.0
    )
    factors = np.stack((sam_confidence, visibility, boundary, depth, view_angle), axis=0)
    reliability = np.prod(np.clip(factors, 0.0, 1.0), axis=0) ** (1.0 / factors.shape[0])
    return reliability.astype(np.float32)


def effective_count_unary(
    *,
    prior_probability: np.ndarray,
    prior_strength: float,
    observations: Mapping[str, np.ndarray],
    sam_confidence_floor: float,
    boundary_distance_scale_px: float,
    depth_residual_scale_m: float,
) -> dict[str, np.ndarray]:
    prior = np.asarray(prior_probability, dtype=np.float64)
    if prior.ndim != 1 or not np.isfinite(prior).all() or np.any((prior <= 0.0) | (prior >= 1.0)):
        raise ValueError("prior_probability 必须是一维有限 open-interval probability")
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength 必须有限且大于 0")
    validate_observation_chunk(observations, gaussian_count=prior.size)
    gaussian_id = np.asarray(observations["gaussian_id"], dtype=np.int64)
    sam = np.asarray(observations["sam_probability"], dtype=np.float64)
    reliability = observation_reliability(
        observations,
        sam_confidence_floor=sam_confidence_floor,
        boundary_distance_scale_px=boundary_distance_scale_px,
        depth_residual_scale_m=depth_residual_scale_m,
    ).astype(np.float64)
    positive = reliability * sam
    negative = reliability * (1.0 - sam)
    positive_mass = np.bincount(gaussian_id, weights=positive, minlength=prior.size)
    negative_mass = np.bincount(gaussian_id, weights=negative, minlength=prior.size)
    effective_count = np.bincount(gaussian_id, weights=reliability, minlength=prior.size)
    weighted_square = np.bincount(
        gaussian_id, weights=reliability * np.square(sam), minlength=prior.size
    )
    weighted_mean = np.divide(
        positive_mass,
        effective_count,
        out=prior.copy(),
        where=effective_count > 0.0,
    )
    variance = np.divide(
        weighted_square,
        effective_count,
        out=np.zeros_like(prior),
        where=effective_count > 0.0,
    ) - np.square(weighted_mean)
    alpha = prior * prior_strength + positive_mass
    beta = (1.0 - prior) * prior_strength + negative_mass
    strength = alpha + beta
    posterior = alpha / strength
    uncertainty = 1.0 / (strength + 1.0)
    boundary_weight = np.exp(
        -np.abs(np.asarray(observations["mask_boundary_distance"], dtype=np.float64))
        / boundary_distance_scale_px
    )
    boundary_mass = np.bincount(
        gaussian_id, weights=reliability * boundary_weight, minlength=prior.size
    )
    boundary_ambiguity = np.divide(
        boundary_mass,
        effective_count,
        out=np.zeros_like(prior),
        where=effective_count > 0.0,
    )
    return {
        "alpha": alpha.astype(np.float32),
        "beta": beta.astype(np.float32),
        "unary_posterior": posterior.astype(np.float32),
        "unary_uncertainty": uncertainty.astype(np.float32),
        "effective_evidence_count": effective_count.astype(np.float32),
        "multi_view_disagreement": np.clip(variance, 0.0, 1.0).astype(np.float32),
        "boundary_ambiguity": np.clip(boundary_ambiguity, 0.0, 1.0).astype(np.float32),
        "observation_reliability": reliability.astype(np.float32),
    }

"""Beta--Bernoulli evidence fusion for WorldSim V4 Gaussian ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


_EPSILON = 1e-6


def _finite_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


@dataclass(frozen=True)
class BetaEvidence:
    """Independent Beta posterior parameters for a fixed Gaussian ordering."""

    alpha: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        alpha = _finite_vector(self.alpha, name="alpha")
        beta = _finite_vector(self.beta, name="beta")
        if alpha.shape != beta.shape:
            raise ValueError("alpha/beta shapes differ")
        if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
            raise ValueError("alpha/beta must be strictly positive")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def posterior(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> np.ndarray:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (np.square(total) * (total + 1.0))

    @property
    def strength(self) -> np.ndarray:
        return self.alpha + self.beta


def prior_from_probability(
    probability: np.ndarray,
    *,
    strength: float | np.ndarray,
    minimum_parameter: float = _EPSILON,
) -> BetaEvidence:
    """Convert a frozen V3.3 O1 ownership probability into a Beta prior."""

    value = _finite_vector(probability, name="probability")
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    concentration = np.asarray(strength, dtype=np.float64)
    try:
        concentration = np.broadcast_to(concentration, value.shape)
    except ValueError as error:
        raise ValueError("strength is not broadcastable to probability") from error
    if not np.isfinite(concentration).all() or np.any(concentration <= 0.0):
        raise ValueError("strength must be finite and positive")
    floor = float(minimum_parameter)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_parameter must be finite and positive")
    return BetaEvidence(
        alpha=np.maximum(value * concentration, floor),
        beta=np.maximum((1.0 - value) * concentration, floor),
    )


def product_weight(
    factors: Mapping[str, np.ndarray | float], *, shape: tuple[int, ...]
) -> np.ndarray:
    """Multiply named confidence factors after validating every factor in [0, 1]."""

    if len(shape) != 1:
        raise ValueError("evidence arrays must be one-dimensional")
    result = np.ones(shape, dtype=np.float64)
    if not factors:
        raise ValueError("at least one evidence factor is required")
    for name, value in factors.items():
        factor = np.asarray(value, dtype=np.float64)
        try:
            factor = np.broadcast_to(factor, shape)
        except ValueError as error:
            raise ValueError(f"factor {name} is not broadcastable") from error
        if not np.isfinite(factor).all() or np.any((factor < 0.0) | (factor > 1.0)):
            raise ValueError(f"factor {name} must lie in [0, 1]")
        result *= factor
    return result


def update_beta_evidence(
    prior: BetaEvidence,
    *,
    evidence: np.ndarray,
    weight: np.ndarray | float,
) -> BetaEvidence:
    """Apply alpha += w*e and beta += w*(1-e) elementwise."""

    observation = _finite_vector(evidence, name="evidence")
    if observation.shape != prior.alpha.shape:
        raise ValueError("evidence/prior shapes differ")
    if np.any((observation < 0.0) | (observation > 1.0)):
        raise ValueError("evidence must lie in [0, 1]")
    confidence = np.asarray(weight, dtype=np.float64)
    try:
        confidence = np.broadcast_to(confidence, observation.shape)
    except ValueError as error:
        raise ValueError("weight is not broadcastable to evidence") from error
    if not np.isfinite(confidence).all() or np.any(confidence < 0.0):
        raise ValueError("weight must be finite and non-negative")
    return BetaEvidence(
        alpha=prior.alpha + confidence * observation,
        beta=prior.beta + confidence * (1.0 - observation),
    )


def update_from_counts(
    prior: BetaEvidence,
    *,
    positive_count: np.ndarray,
    negative_count: np.ndarray,
    factors: Mapping[str, np.ndarray | float],
) -> BetaEvidence:
    """Fuse aggregated multi-view Bernoulli counts with a factored confidence."""

    positive = _finite_vector(positive_count, name="positive_count")
    negative = _finite_vector(negative_count, name="negative_count")
    if positive.shape != prior.alpha.shape or negative.shape != prior.alpha.shape:
        raise ValueError("count/prior shapes differ")
    if np.any(positive < 0.0) or np.any(negative < 0.0):
        raise ValueError("evidence counts must be non-negative")
    confidence = product_weight(factors, shape=prior.alpha.shape)
    return BetaEvidence(
        alpha=prior.alpha + confidence * positive,
        beta=prior.beta + confidence * negative,
    )

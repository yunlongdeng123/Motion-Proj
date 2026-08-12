"""Temporal memory for WorldSim V4 Beta evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beta_fusion import BetaEvidence


def temporal_update(
    previous: BetaEvidence, observed: BetaEvidence, *, retention: float
) -> BetaEvidence:
    """Blend consecutive posterior parameters using the frozen retention rho."""

    if previous.alpha.shape != observed.alpha.shape:
        raise ValueError("previous/observed evidence shapes differ")
    rho = float(retention)
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    return BetaEvidence(
        alpha=rho * previous.alpha + (1.0 - rho) * observed.alpha,
        beta=rho * previous.beta + (1.0 - rho) * observed.beta,
    )


@dataclass
class TemporalEvidenceMemory:
    """Stateful fail-closed wrapper that also protects Gaussian identity order."""

    gaussian_id: np.ndarray
    retention: float
    state: BetaEvidence | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.gaussian_id, dtype=np.int64)
        if ids.ndim != 1 or np.unique(ids).size != ids.size:
            raise ValueError("gaussian_id must be unique and one-dimensional")
        if not np.isfinite(self.retention) or not 0.0 <= self.retention <= 1.0:
            raise ValueError("retention must lie in [0, 1]")
        self.gaussian_id = ids.copy()

    def update(self, gaussian_id: np.ndarray, observed: BetaEvidence) -> BetaEvidence:
        ids = np.asarray(gaussian_id, dtype=np.int64)
        if not np.array_equal(ids, self.gaussian_id):
            raise ValueError("Gaussian identity/order drift in temporal evidence")
        if observed.alpha.shape != ids.shape:
            raise ValueError("observed evidence/Gaussian identity shapes differ")
        self.state = (
            observed
            if self.state is None
            else temporal_update(self.state, observed, retention=self.retention)
        )
        return self.state

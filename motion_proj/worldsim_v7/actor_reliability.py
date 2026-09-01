"""Actor 残差分布与轨迹边界投影。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActorResidualDistribution:
    """每个 horizon 的二维高斯 Actor 位置残差。"""

    actor_id: str
    horizons_s: np.ndarray
    mean_xy_m: np.ndarray
    covariance_xy_m2: np.ndarray

    def __post_init__(self) -> None:
        horizons = np.asarray(self.horizons_s, dtype=np.float64)
        mean = np.asarray(self.mean_xy_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_xy_m2, dtype=np.float64)
        if not self.actor_id:
            raise ValueError("actor_id must be non-empty")
        if horizons.ndim != 1 or len(horizons) == 0 or np.any(np.diff(horizons) <= 0.0):
            raise ValueError("horizons_s must be a non-empty strictly increasing vector")
        if mean.shape != (len(horizons), 2):
            raise ValueError("mean_xy_m must have shape (H, 2)")
        if covariance.shape != (len(horizons), 2, 2):
            raise ValueError("covariance_xy_m2 must have shape (H, 2, 2)")
        if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-7):
            raise ValueError("covariance matrices must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) < -1e-8):
            raise ValueError("covariance matrices must be positive semidefinite")
        object.__setattr__(self, "horizons_s", horizons)
        object.__setattr__(self, "mean_xy_m", mean)
        object.__setattr__(self, "covariance_xy_m2", covariance)

    def project_to_boundary(self, normals_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回边界法向上的均值与标准差。"""

        normals = np.asarray(normals_xy, dtype=np.float64)
        if normals.shape != self.mean_xy_m.shape:
            raise ValueError("normals_xy must have shape (H, 2)")
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("boundary normals must be non-zero")
        unit = normals / norms
        projected_mean = np.sum(unit * self.mean_xy_m, axis=1)
        projected_variance = np.einsum(
            "hi,hij,hj->h", unit, self.covariance_xy_m2, unit
        )
        return projected_mean, np.sqrt(np.maximum(projected_variance, 0.0))

    def sample(self, sample_count: int, seed: int = 0) -> np.ndarray:
        """按 horizon 独立采样，供确定性 Monte Carlo 评测使用。"""

        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        generator = np.random.default_rng(int(seed))
        draws = [
            generator.multivariate_normal(mean, covariance, size=int(sample_count))
            for mean, covariance in zip(self.mean_xy_m, self.covariance_xy_m2)
        ]
        return np.stack(draws, axis=1)

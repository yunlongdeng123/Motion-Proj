"""V5 per-view chunk 到 per-Gaussian structured unary 的流式收口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .bayesian_unary import (
    accumulate_effective_count_statistics,
    empty_effective_count_statistics,
    finalize_effective_count_unary,
)
from .evidence_schema import (
    atomic_save_npz,
    sha256_file,
    validate_gaussian_table,
    validate_observation_chunk,
)


@dataclass
class StructuredEvidenceAccumulator:
    scene: str
    role: str
    base_model: np.ndarray
    base_index: np.ndarray
    geometry: Mapping[str, np.ndarray]
    prior_probability: np.ndarray
    prior_strength: float
    sam_confidence_floor: float
    boundary_distance_scale_px: float
    depth_residual_scale_m: float
    lidar_support: np.ndarray | None = None
    motion_consistency: np.ndarray | None = None
    statistics: dict[str, np.ndarray] = field(init=False)
    observation_count: int = field(init=False, default=0)
    chunk_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.prior_probability = np.asarray(self.prior_probability, dtype=np.float64)
        if self.prior_probability.ndim != 1:
            raise ValueError("prior_probability 必须为一维")
        count = self.prior_probability.size
        if np.asarray(self.base_model).shape != (count,):
            raise ValueError("base_model 必须与 prior 对齐")
        if np.asarray(self.base_index).shape != (count,):
            raise ValueError("base_index 必须与 prior 对齐")
        for name, shape in (
            ("center", (count, 3)),
            ("covariance", (count, 3, 3)),
            ("normal_proxy", (count, 3)),
            ("normal_available", (count,)),
        ):
            if np.asarray(self.geometry[name]).shape != shape:
                raise ValueError(f"geometry/{name} shape 必须为 {shape}")
        for name, value in (
            ("lidar_support", self.lidar_support),
            ("motion_consistency", self.motion_consistency),
        ):
            if value is not None and np.asarray(value).shape != (count,):
                raise ValueError(f"{name} 必须与 prior 对齐")
        self.statistics = empty_effective_count_statistics(count)

    @property
    def gaussian_count(self) -> int:
        return int(self.prior_probability.size)

    def add_chunk(self, chunk: Mapping[str, np.ndarray]) -> np.ndarray:
        validate_observation_chunk(chunk, gaussian_count=self.gaussian_count)
        if str(np.asarray(chunk["scene"]).item()) != self.scene:
            raise ValueError("observation scene 漂移")
        if str(np.asarray(chunk["role"]).item()) != self.role:
            raise ValueError("observation role 漂移")
        reliability = accumulate_effective_count_statistics(
            self.statistics,
            observations=chunk,
            gaussian_count=self.gaussian_count,
            sam_confidence_floor=self.sam_confidence_floor,
            boundary_distance_scale_px=self.boundary_distance_scale_px,
            depth_residual_scale_m=self.depth_residual_scale_m,
        )
        self.observation_count += int(np.asarray(chunk["gaussian_id"]).size)
        self.chunk_count += 1
        return reliability

    def gaussian_table(self) -> dict[str, np.ndarray]:
        unary = finalize_effective_count_unary(
            prior_probability=self.prior_probability,
            prior_strength=self.prior_strength,
            statistics=self.statistics,
        )
        count = self.gaussian_count
        if self.lidar_support is None:
            lidar = np.zeros(count, dtype=np.float32)
            lidar_available = np.zeros(count, dtype=np.int8)
        else:
            lidar = np.asarray(self.lidar_support, dtype=np.float32)
            lidar_available = np.ones(count, dtype=np.int8)
        if self.motion_consistency is None:
            motion = np.zeros(count, dtype=np.float32)
            motion_available = np.zeros(count, dtype=np.int8)
        else:
            motion = np.asarray(self.motion_consistency, dtype=np.float32)
            motion_available = np.ones(count, dtype=np.int8)
        table = {
            "scene": np.asarray(self.scene),
            "role": np.asarray(self.role),
            "gaussian_id": np.arange(count, dtype=np.int64),
            "base_model": np.asarray(self.base_model),
            "base_index": np.asarray(self.base_index, dtype=np.int64),
            "center": np.asarray(self.geometry["center"], dtype=np.float32),
            "covariance": np.asarray(self.geometry["covariance"], dtype=np.float32),
            "normal_proxy": np.asarray(self.geometry["normal_proxy"], dtype=np.float32),
            "normal_available": np.asarray(
                self.geometry["normal_available"], dtype=np.int8
            ),
            "prior": self.prior_probability.astype(np.float32),
            "unary_posterior": unary["unary_posterior"],
            "unary_uncertainty": unary["unary_uncertainty"],
            "effective_evidence_count": unary["effective_evidence_count"],
            "multi_view_disagreement": unary["multi_view_disagreement"],
            "boundary_ambiguity": unary["boundary_ambiguity"],
            "depth_support": unary["depth_support"],
            "lidar_support": lidar,
            "lidar_support_available": lidar_available,
            "motion_consistency": motion,
            "motion_consistency_available": motion_available,
        }
        validate_gaussian_table(table)
        return table

    def save_gaussian_table(self, path: str | Path) -> dict[str, object]:
        output = Path(path)
        atomic_save_npz(output, self.gaussian_table())
        return {
            "path": str(output),
            "sha256": sha256_file(output),
            "gaussian_count": self.gaussian_count,
            "observation_count": self.observation_count,
            "chunk_count": self.chunk_count,
        }

"""不修改 base 的连续时间 Gaussian delta 状态。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .repair_candidates import GaussianAssetBinding
from .se3_bspline import UniformCubicSE3Spline


TEMPORAL_OPERATIONS = ("REMOVE", "LATERAL", "INSERT")


@dataclass(frozen=True)
class TemporalDeltaState:
    active: bool
    erase_gaussian_ids: tuple[int, ...]
    insert_asset: GaussianAssetBinding | None
    insert_transform: np.ndarray | None


@dataclass(frozen=True)
class TemporalGaussianDelta:
    delta_id: str
    scene: str
    operation: str
    base_checkpoint_sha256: str
    start_time: float
    end_time: float
    erase_gaussian_ids: tuple[int, ...]
    insert_asset: GaussianAssetBinding | None
    trajectory: UniformCubicSE3Spline | None
    base_mutated: bool = False
    test_quality_read: bool = False

    def __post_init__(self) -> None:
        if not self.delta_id or not self.scene:
            raise ValueError("temporal delta identity must be non-empty")
        if self.operation not in TEMPORAL_OPERATIONS:
            raise ValueError(f"unsupported temporal operation: {self.operation}")
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("temporal interval must be finite")
        if self.end_time <= self.start_time:
            raise ValueError("temporal interval must have positive duration")
        if len(self.base_checkpoint_sha256) != 64:
            raise ValueError("base checkpoint SHA must have 64 characters")
        if len(set(self.erase_gaussian_ids)) != len(self.erase_gaussian_ids):
            raise ValueError("erase Gaussian IDs must be unique")
        if self.operation == "REMOVE" and self.insert_asset is not None:
            raise ValueError("REMOVE cannot insert an asset")
        if self.operation in {"LATERAL", "INSERT"} and (
            self.insert_asset is None or self.trajectory is None
        ):
            raise ValueError("LATERAL/INSERT require an asset and a trajectory")
        if self.base_mutated or self.test_quality_read:
            raise ValueError("temporal delta must keep base immutable and test unread")

    def state_at(self, time: float) -> TemporalDeltaState:
        value = float(time)
        if not math.isfinite(value):
            raise ValueError("time must be finite")
        active = self.start_time <= value <= self.end_time
        if not active:
            return TemporalDeltaState(False, (), None, None)
        transform = self.trajectory.transform(value) if self.trajectory else None
        return TemporalDeltaState(
            True,
            self.erase_gaussian_ids,
            self.insert_asset,
            transform,
        )

    def rollback_checkpoint_sha256(self) -> str:
        return self.base_checkpoint_sha256

    def to_dict(self) -> dict[str, Any]:
        trajectory = None
        if self.trajectory is not None:
            trajectory = {
                "control_twists": self.trajectory.control_twists.tolist(),
                "start_time": self.trajectory.start_time,
                "knot_interval": self.trajectory.knot_interval,
                "end_time": self.trajectory.end_time,
            }
        return {
            "schema_version": "worldsim_v4_m3_temporal_delta_v1",
            "delta_id": self.delta_id,
            "scene": self.scene,
            "operation": self.operation,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "erase_gaussian_ids": list(self.erase_gaussian_ids),
            "insert_asset": self.insert_asset.to_dict() if self.insert_asset else None,
            "trajectory": trajectory,
            "composition_order": ["BASE", "ERASE", "INSERT"],
            "base_mutated": False,
            "rollback_checkpoint_sha256": self.base_checkpoint_sha256,
            "test_quality_read": False,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()

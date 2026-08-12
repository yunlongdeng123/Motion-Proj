"""M3 remove/lateral/insert 请求的 fail-closed compiler。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .repair_candidates import GaussianAssetBinding
from .se3_bspline import UniformCubicSE3Spline
from .temporal_delta import TEMPORAL_OPERATIONS, TemporalGaussianDelta


@dataclass(frozen=True)
class TemporalEditRequest:
    request_id: str
    scene: str
    operation: str
    base_checkpoint_sha256: str
    start_time: float
    end_time: float
    erase_gaussian_ids: tuple[int, ...]
    insert_asset: GaussianAssetBinding | None = None
    control_twists: np.ndarray | None = None
    test_quality_read: bool = False

    def __post_init__(self) -> None:
        if self.operation not in TEMPORAL_OPERATIONS:
            raise ValueError(f"unsupported temporal operation: {self.operation}")
        duration = float(self.end_time) - float(self.start_time)
        if not math.isfinite(duration) or duration < 2.0 or duration > 4.0:
            raise ValueError("formal temporal clip duration must lie in [2, 4] seconds")
        if self.test_quality_read:
            raise ValueError("M3 request cannot consume test quality")


def compile_temporal_delta(request: TemporalEditRequest) -> TemporalGaussianDelta:
    trajectory = None
    if request.operation in {"LATERAL", "INSERT"}:
        if request.control_twists is None or request.insert_asset is None:
            raise ValueError("LATERAL/INSERT require control twists and an insert asset")
        controls = np.asarray(request.control_twists, dtype=np.float64)
        if controls.ndim != 2 or controls.shape[0] < 4:
            raise ValueError("temporal control twists require at least four rows")
        interval = (request.end_time - request.start_time) / (controls.shape[0] - 3)
        trajectory = UniformCubicSE3Spline(
            controls, start_time=request.start_time, knot_interval=interval
        )
    elif request.insert_asset is not None or request.control_twists is not None:
        raise ValueError("REMOVE cannot carry an insert asset or trajectory")
    return TemporalGaussianDelta(
        delta_id=request.request_id,
        scene=request.scene,
        operation=request.operation,
        base_checkpoint_sha256=request.base_checkpoint_sha256,
        start_time=request.start_time,
        end_time=request.end_time,
        erase_gaussian_ids=request.erase_gaussian_ids,
        insert_asset=request.insert_asset,
        trajectory=trajectory,
    )

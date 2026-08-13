"""真实连续片段使用的 M3 时序轨迹拟合协议。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .se3_bspline import UniformCubicSE3Spline, cubic_basis, se3_exp, se3_log


TEMPORAL_ARMS = (
    "FRAME_INDEPENDENT",
    "LINEAR",
    "CUBIC_BSPLINE",
    "CUBIC_BSPLINE_TEMPORAL_EVIDENCE",
    "FULL_WARP_REGULARIZED",
)


@dataclass(frozen=True)
class FittedTrajectory:
    transforms: np.ndarray
    control_twists: np.ndarray | None
    acceleration_energy: float
    source_rmse_m: float


def _transforms(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 4 or array.shape[1:] != (4, 4):
        raise ValueError("transforms must have shape (T>=4, 4, 4)")
    if not np.isfinite(array).all():
        raise ValueError("transforms must be finite")
    return array


def uniform_cubic_design(sample_count: int, control_point_count: int) -> np.ndarray:
    if sample_count < 4:
        raise ValueError("sample_count must be at least four")
    if control_point_count < 4 or control_point_count > sample_count:
        raise ValueError("control_point_count must lie in [4, sample_count]")
    segments = control_point_count - 3
    positions = np.linspace(0.0, float(segments), sample_count)
    rows = np.zeros((sample_count, control_point_count), dtype=np.float64)
    for row, position in enumerate(positions):
        if row == sample_count - 1:
            segment, unit = segments - 1, 1.0
        else:
            segment = min(int(math.floor(position)), segments - 1)
            unit = position - segment
        rows[row, segment : segment + 4] = cubic_basis(float(unit))
    return rows


def evidence_memory_weights(
    support: np.ndarray, *, retention: float, minimum_weight: float = 0.05
) -> np.ndarray:
    values = np.asarray(support, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.isfinite(values).all():
        raise ValueError("support must be a finite vector with at least four values")
    if np.any(values < 0.0):
        raise ValueError("support must be non-negative")
    if not 0.0 <= retention < 1.0:
        raise ValueError("retention must lie in [0, 1)")
    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError("minimum_weight must lie in (0, 1]")
    maximum = float(values.max())
    normalized = np.ones_like(values) if maximum == 0.0 else values / maximum
    forward = normalized.copy()
    backward = normalized.copy()
    for index in range(1, len(values)):
        forward[index] = max(forward[index], retention * forward[index - 1])
    for index in range(len(values) - 2, -1, -1):
        backward[index] = max(backward[index], retention * backward[index + 1])
    return np.maximum(minimum_weight, np.maximum(forward, backward))


def _relative_twists(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _transforms(transforms)
    base = values[0]
    inverse = np.linalg.inv(base)
    twists = np.asarray([se3_log(inverse @ transform) for transform in values])
    return base, twists


def _translation_rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left)[:, :3, 3] - np.asarray(right)[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


def _translation_acceleration_energy(transforms: np.ndarray) -> float:
    translations = np.asarray(transforms, dtype=np.float64)[:, :3, 3]
    acceleration = np.diff(translations, n=2, axis=0)
    return float(np.mean(np.sum(acceleration**2, axis=1)))


def linear_se3_trajectory(transforms: np.ndarray) -> FittedTrajectory:
    values = _transforms(transforms)
    base = values[0]
    final_twist = se3_log(np.linalg.inv(base) @ values[-1])
    fractions = np.linspace(0.0, 1.0, len(values))
    fitted = np.asarray([base @ se3_exp(fraction * final_twist) for fraction in fractions])
    return FittedTrajectory(
        transforms=fitted,
        control_twists=None,
        acceleration_energy=_translation_acceleration_energy(fitted),
        source_rmse_m=_translation_rmse(fitted, values),
    )


def fit_uniform_cubic_se3(
    transforms: np.ndarray,
    *,
    control_point_count: int,
    weights: np.ndarray | None = None,
    acceleration_regularization: float = 0.0,
) -> FittedTrajectory:
    values = _transforms(transforms)
    if not math.isfinite(acceleration_regularization) or acceleration_regularization < 0:
        raise ValueError("acceleration_regularization must be finite and non-negative")
    base, samples = _relative_twists(values)
    design = uniform_cubic_design(len(values), control_point_count)
    if weights is None:
        weight = np.ones(len(values), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (len(values),) or not np.isfinite(weight).all():
            raise ValueError("weights must match the transform sequence")
        if np.any(weight <= 0.0):
            raise ValueError("weights must be positive")
    second_difference = np.zeros(
        (control_point_count - 2, control_point_count), dtype=np.float64
    )
    for row in range(control_point_count - 2):
        second_difference[row, row : row + 3] = (1.0, -2.0, 1.0)
    weighted = design * np.sqrt(weight)[:, None]
    target = samples * np.sqrt(weight)[:, None]
    system = weighted.T @ weighted
    if acceleration_regularization > 0.0:
        system += acceleration_regularization * (
            second_difference.T @ second_difference
        )
    system += np.eye(control_point_count) * 1e-10
    controls = np.linalg.solve(system, weighted.T @ target)
    spline = UniformCubicSE3Spline(
        controls,
        start_time=0.0,
        knot_interval=1.0 / (control_point_count - 3),
    )
    fitted = np.asarray(
        [
            spline.transform(time, base_transform=base)
            for time in np.linspace(0.0, 1.0, len(values))
        ]
    )
    return FittedTrajectory(
        transforms=fitted,
        control_twists=controls,
        acceleration_energy=spline.acceleration_energy(samples_per_segment=64),
        source_rmse_m=_translation_rmse(fitted, values),
    )


def apply_actor_local_translation(
    transforms: np.ndarray, translation_xyz: np.ndarray
) -> np.ndarray:
    values = _transforms(transforms)
    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation_xyz must be a finite three-vector")
    local = np.eye(4, dtype=np.float64)
    local[:3, 3] = translation
    return np.asarray([transform @ local for transform in values])


def resample_se3_transforms(transforms: np.ndarray, sample_count: int) -> np.ndarray:
    values = _transforms(transforms)
    if sample_count < len(values):
        raise ValueError("sample_count cannot reduce the source sequence")
    source_positions = np.linspace(0.0, 1.0, len(values))
    target_positions = np.linspace(0.0, 1.0, sample_count)
    result = []
    for position in target_positions:
        if position >= 1.0:
            result.append(values[-1])
            continue
        right = int(np.searchsorted(source_positions, position, side="right"))
        left = max(0, right - 1)
        right = min(right, len(values) - 1)
        span = source_positions[right] - source_positions[left]
        fraction = 0.0 if span == 0.0 else (position - source_positions[left]) / span
        relative = se3_log(np.linalg.inv(values[left]) @ values[right])
        result.append(values[left] @ se3_exp(float(fraction) * relative))
    return np.asarray(result)


def build_arm_trajectories(
    source_transforms: np.ndarray,
    *,
    support: np.ndarray,
    control_point_count: int,
    evidence_retention: float,
    acceleration_regularization: float,
) -> Mapping[str, FittedTrajectory]:
    source = _transforms(source_transforms)
    independent = FittedTrajectory(
        transforms=source.copy(),
        control_twists=None,
        acceleration_energy=_translation_acceleration_energy(source),
        source_rmse_m=0.0,
    )
    cubic = fit_uniform_cubic_se3(
        source,
        control_point_count=control_point_count,
        acceleration_regularization=acceleration_regularization,
    )
    memory = evidence_memory_weights(support, retention=evidence_retention)
    evidence = fit_uniform_cubic_se3(
        source,
        control_point_count=control_point_count,
        weights=memory,
        acceleration_regularization=acceleration_regularization,
    )
    return {
        "FRAME_INDEPENDENT": independent,
        "LINEAR": linear_se3_trajectory(source),
        "CUBIC_BSPLINE": cubic,
        "CUBIC_BSPLINE_TEMPORAL_EVIDENCE": evidence,
        # FULL 先共享 evidence trajectory，图像 delta 的 warp 正则由 renderer 执行。
        "FULL_WARP_REGULARIZED": evidence,
    }

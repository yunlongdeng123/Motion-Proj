"""连续时间 `SE(3)` 三次均匀 B-spline。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """把 `[translation, rotation]` twist 映射到齐次变换。"""

    value = np.asarray(twist, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("twist must be a finite vector with shape (6,)")
    translation = value[:3]
    rotation = value[3:]
    theta = float(np.linalg.norm(rotation))
    omega = _skew(rotation)
    omega_squared = omega @ omega
    if theta < 1e-8:
        rotation_matrix = np.eye(3) + omega + 0.5 * omega_squared
        left_jacobian = np.eye(3) + 0.5 * omega + omega_squared / 6.0
    else:
        theta_squared = theta * theta
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta_squared
        c = (theta - math.sin(theta)) / (theta_squared * theta)
        rotation_matrix = np.eye(3) + a * omega + b * omega_squared
        left_jacobian = np.eye(3) + b * omega + c * omega_squared
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = left_jacobian @ translation
    return transform


def cubic_basis(unit_time: float) -> np.ndarray:
    u = float(unit_time)
    if not math.isfinite(u) or u < 0.0 or u > 1.0:
        raise ValueError("unit_time must lie in [0, 1]")
    return np.asarray(
        [
            (1.0 - u) ** 3 / 6.0,
            (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0,
            (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0,
            u**3 / 6.0,
        ],
        dtype=np.float64,
    )


def cubic_basis_second_derivative(unit_time: float) -> np.ndarray:
    u = float(unit_time)
    if not math.isfinite(u) or u < 0.0 or u > 1.0:
        raise ValueError("unit_time must lie in [0, 1]")
    return np.asarray([1.0 - u, 3.0 * u - 2.0, 1.0 - 3.0 * u, u])


@dataclass(frozen=True)
class UniformCubicSE3Spline:
    control_twists: np.ndarray
    start_time: float
    knot_interval: float

    def __post_init__(self) -> None:
        controls = np.asarray(self.control_twists, dtype=np.float64)
        if controls.ndim != 2 or controls.shape[0] < 4 or controls.shape[1] != 6:
            raise ValueError("control_twists must have shape (N>=4, 6)")
        if not np.isfinite(controls).all():
            raise ValueError("control_twists must be finite")
        if not math.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if not math.isfinite(self.knot_interval) or self.knot_interval <= 0.0:
            raise ValueError("knot_interval must be finite and positive")
        object.__setattr__(self, "control_twists", controls.copy())

    @property
    def segment_count(self) -> int:
        return int(self.control_twists.shape[0] - 3)

    @property
    def end_time(self) -> float:
        return self.start_time + self.segment_count * self.knot_interval

    def _segment(self, time: float) -> tuple[int, float]:
        value = float(time)
        if not math.isfinite(value) or value < self.start_time or value > self.end_time:
            raise ValueError("time lies outside the spline domain")
        normalized = (value - self.start_time) / self.knot_interval
        if value == self.end_time:
            return self.segment_count - 1, 1.0
        segment = int(math.floor(normalized))
        return segment, normalized - segment

    def twist(self, time: float) -> np.ndarray:
        segment, unit_time = self._segment(time)
        return cubic_basis(unit_time) @ self.control_twists[segment : segment + 4]

    def acceleration(self, time: float) -> np.ndarray:
        segment, unit_time = self._segment(time)
        weights = cubic_basis_second_derivative(unit_time) / self.knot_interval**2
        return weights @ self.control_twists[segment : segment + 4]

    def transform(self, time: float, *, base_transform: np.ndarray | None = None) -> np.ndarray:
        base = np.eye(4) if base_transform is None else np.asarray(base_transform, dtype=np.float64)
        if base.shape != (4, 4) or not np.isfinite(base).all():
            raise ValueError("base_transform must be a finite 4x4 matrix")
        return base @ se3_exp(self.twist(time))

    def acceleration_energy(self, *, samples_per_segment: int = 32) -> float:
        if samples_per_segment < 2:
            raise ValueError("samples_per_segment must be at least two")
        times = np.linspace(
            self.start_time,
            self.end_time,
            self.segment_count * samples_per_segment + 1,
        )
        squared = np.asarray(
            [float(self.acceleration(time) @ self.acceleration(time)) for time in times]
        )
        return float(np.trapz(squared, times))

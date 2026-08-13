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


def se3_log(transform: np.ndarray) -> np.ndarray:
    """把齐次变换映射回 `[translation, rotation]` twist。"""

    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-10):
        raise ValueError("transform must have a homogeneous final row")
    rotation_matrix = value[:3, :3]
    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-8):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-8):
        raise ValueError("transform rotation must have determinant one")

    cosine = float(np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1e-8:
        omega = 0.5 * (rotation_matrix - rotation_matrix.T)
    elif math.pi - theta < 1e-5:
        # 近 pi 时反对称项病态，直接从对角项恢复旋转轴。
        axis = np.sqrt(np.maximum((np.diag(rotation_matrix) + 1.0) / 2.0, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] < 1e-8:
            raise ValueError("rotation logarithm is numerically singular")
        for index in range(3):
            if index != largest:
                axis[index] = (
                    rotation_matrix[largest, index]
                    + rotation_matrix[index, largest]
                ) / (4.0 * axis[largest])
        axis /= np.linalg.norm(axis)
        omega = _skew(axis * theta)
    else:
        omega = theta / (2.0 * math.sin(theta)) * (
            rotation_matrix - rotation_matrix.T
        )
    rotation = np.asarray([omega[2, 1], omega[0, 2], omega[1, 0]])
    theta = float(np.linalg.norm(rotation))
    omega_squared = omega @ omega
    if theta < 1e-8:
        inverse_left_jacobian = np.eye(3) - 0.5 * omega + omega_squared / 12.0
    else:
        coefficient = (
            1.0 / theta**2
            - (1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta))
        )
        inverse_left_jacobian = np.eye(3) - 0.5 * omega + coefficient * omega_squared
    translation = inverse_left_jacobian @ value[:3, 3]
    return np.concatenate([translation, rotation])


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

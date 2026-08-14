"""V5 M3 平面车辆轨迹的 minimum-jerk 与物理约束投影。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from motion_proj.worldsim_v4.temporal_protocol import fit_uniform_cubic_se3


@dataclass(frozen=True)
class KinematicLimits:
    maximum_speed_mps: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    maximum_yaw_rate_radps: float
    maximum_lateral_acceleration_mps2: float
    maximum_heading_velocity_mismatch_rad: float
    maximum_contact_error_m: float
    collision_overlap_tolerance_m2: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.maximum_deceleration_mps2,
            self.maximum_yaw_rate_radps,
            self.maximum_lateral_acceleration_mps2,
            self.maximum_heading_velocity_mismatch_rad,
            self.maximum_contact_error_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("kinematic limits 必须为有限正数")
        if (
            not math.isfinite(self.collision_overlap_tolerance_m2)
            or self.collision_overlap_tolerance_m2 < 0.0
        ):
            raise ValueError("collision overlap tolerance 必须有限且非负")


@dataclass(frozen=True)
class PlanarTrajectory:
    times: np.ndarray
    positions: np.ndarray
    yaws: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        yaws = np.asarray(self.yaws, dtype=np.float64)
        if times.ndim != 1 or len(times) < 7 or not np.isfinite(times).all():
            raise ValueError("times 必须是至少七帧的有限向量")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times 必须严格递增")
        if positions.shape != (len(times), 3) or not np.isfinite(positions).all():
            raise ValueError("positions 必须为有限的 [T,3]")
        if yaws.shape != (len(times),) or not np.isfinite(yaws).all():
            raise ValueError("yaws 必须为有限的 [T]")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "yaws", np.unwrap(yaws).copy())

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "positions": self.positions.tolist(),
            "yaws": self.yaws.tolist(),
        }


@dataclass(frozen=True)
class CircularObstacle:
    center_xy: tuple[float, float]
    radius_m: float

    def __post_init__(self) -> None:
        if (
            len(self.center_xy) != 2
            or not all(math.isfinite(value) for value in self.center_xy)
            or not math.isfinite(self.radius_m)
            or self.radius_m <= 0.0
        ):
            raise ValueError("circular obstacle 非法")


def planar_to_transforms(trajectory: PlanarTrajectory) -> np.ndarray:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(trajectory.times), axis=0)
    cosine = np.cos(trajectory.yaws)
    sine = np.sin(trajectory.yaws)
    transforms[:, 0, 0] = cosine
    transforms[:, 0, 1] = -sine
    transforms[:, 1, 0] = sine
    transforms[:, 1, 1] = cosine
    transforms[:, :3, 3] = trajectory.positions
    return transforms


def transforms_to_planar(times: np.ndarray, transforms: np.ndarray) -> PlanarTrajectory:
    values = np.asarray(transforms, dtype=np.float64)
    time_values = np.asarray(times, dtype=np.float64)
    if values.shape != (len(time_values), 4, 4) or not np.isfinite(values).all():
        raise ValueError("transforms 必须为与 times 对齐的有限 [T,4,4]")
    yaws = np.arctan2(values[:, 1, 0], values[:, 0, 0])
    return PlanarTrajectory(time_values, values[:, :3, 3], yaws)


def v4_frozen_bspline_comparator(
    desired: PlanarTrajectory,
    *,
    control_point_count: int = 4,
    acceleration_regularization: float = 0.10,
) -> PlanarTrajectory:
    """T2：严格复用 V4 cubic SE(3) 拟合器，不复用 V4 汇总统计。"""

    fitted = fit_uniform_cubic_se3(
        planar_to_transforms(desired),
        control_point_count=control_point_count,
        acceleration_regularization=acceleration_regularization,
    )
    return transforms_to_planar(desired.times, fitted.transforms)


def _third_difference(sample_count: int) -> np.ndarray:
    if sample_count < 4:
        raise ValueError("third difference 至少需要四帧")
    operator = np.zeros((sample_count - 3, sample_count), dtype=np.float64)
    for row in range(sample_count - 3):
        operator[row, row : row + 4] = (-1.0, 3.0, -3.0, 1.0)
    return operator


def _fixed_endpoint_smooth(values: np.ndarray, jerk_weight: float) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim not in {1, 2} or samples.shape[0] < 7:
        raise ValueError("minimum-jerk 输入至少七帧")
    if not np.isfinite(samples).all() or not math.isfinite(jerk_weight) or jerk_weight < 0:
        raise ValueError("minimum-jerk 输入或权重非法")
    matrix = samples[:, None] if samples.ndim == 1 else samples
    difference = _third_difference(len(matrix))
    system = np.eye(len(matrix)) + jerk_weight * (difference.T @ difference)
    interior = np.arange(1, len(matrix) - 1)
    boundary = np.asarray([0, len(matrix) - 1])
    right = matrix[interior] - system[np.ix_(interior, boundary)] @ matrix[boundary]
    result = matrix.copy()
    result[interior] = np.linalg.solve(system[np.ix_(interior, interior)], right)
    return result[:, 0] if samples.ndim == 1 else result


def minimum_jerk_smooth(
    desired: PlanarTrajectory, *, jerk_weight: float
) -> PlanarTrajectory:
    """拟合全部 desired samples，同时精确保留首尾状态。"""

    return PlanarTrajectory(
        times=desired.times,
        positions=_fixed_endpoint_smooth(desired.positions, jerk_weight),
        yaws=_fixed_endpoint_smooth(desired.yaws, jerk_weight),
    )


def _angle_delta(left: float | np.ndarray, right: float | np.ndarray) -> np.ndarray:
    return (np.asarray(left) - np.asarray(right) + np.pi) % (2.0 * np.pi) - np.pi


def _velocity(trajectory: PlanarTrajectory) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(trajectory.times)
    velocity = np.diff(trajectory.positions[:, :2], axis=0) / dt[:, None]
    return velocity, dt


def trajectory_metrics(
    trajectory: PlanarTrajectory,
    *,
    limits: KinematicLimits,
    road_z: np.ndarray | None = None,
    obstacles: Sequence[CircularObstacle] = (),
    actor_radius_m: float = 1.25,
) -> dict[str, Any]:
    velocity, dt = _velocity(trajectory)
    speed = np.linalg.norm(velocity, axis=1)
    midpoint_dt = (dt[:-1] + dt[1:]) * 0.5
    acceleration = np.diff(velocity, axis=0) / midpoint_dt[:, None]
    speed_mid = (speed[:-1] + speed[1:]) * 0.5
    longitudinal = np.sum(acceleration * velocity[1:], axis=1) / np.maximum(
        speed[1:], 1e-9
    )
    yaw_rate = np.abs(np.diff(trajectory.yaws) / dt)
    segment_heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading_mismatch = np.abs(
        _angle_delta(trajectory.yaws[:-1], segment_heading)
    )
    lateral_acceleration = speed * yaw_rate
    if len(acceleration) >= 2:
        jerk_dt = (midpoint_dt[:-1] + midpoint_dt[1:]) * 0.5
        jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
        jerk_energy = float(np.mean(np.sum(jerk**2, axis=1)))
    else:
        jerk_energy = 0.0
    if road_z is None:
        contact_error = np.zeros(len(trajectory.times), dtype=np.float64)
    else:
        road = np.asarray(road_z, dtype=np.float64)
        if road.shape != trajectory.times.shape or not np.isfinite(road).all():
            raise ValueError("road_z 必须与 trajectory times 对齐")
        contact_error = np.abs(trajectory.positions[:, 2] - road)
    collision_depth = []
    if not math.isfinite(actor_radius_m) or actor_radius_m <= 0.0:
        raise ValueError("actor_radius_m 必须为有限正数")
    for position in trajectory.positions[:, :2]:
        maximum = 0.0
        for obstacle in obstacles:
            distance = float(np.linalg.norm(position - np.asarray(obstacle.center_xy)))
            maximum = max(maximum, actor_radius_m + obstacle.radius_m - distance)
        collision_depth.append(maximum)
    collision_depth_array = np.asarray(collision_depth, dtype=np.float64)
    counts = {
        "speed": int(np.count_nonzero(speed > limits.maximum_speed_mps + 1e-9)),
        "acceleration": int(
            np.count_nonzero(longitudinal > limits.maximum_acceleration_mps2 + 1e-9)
        ),
        "deceleration": int(
            np.count_nonzero(longitudinal < -limits.maximum_deceleration_mps2 - 1e-9)
        ),
        "yaw_rate": int(
            np.count_nonzero(yaw_rate > limits.maximum_yaw_rate_radps + 1e-9)
        ),
        "lateral_acceleration": int(
            np.count_nonzero(
                lateral_acceleration > limits.maximum_lateral_acceleration_mps2 + 1e-9
            )
        ),
        "heading_velocity": int(
            np.count_nonzero(
                (speed > 0.1)
                & (heading_mismatch > limits.maximum_heading_velocity_mismatch_rad + 1e-9)
            )
        ),
        "contact": int(
            np.count_nonzero(contact_error > limits.maximum_contact_error_m + 1e-9)
        ),
        "collision": int(np.count_nonzero(collision_depth_array > 1e-9)),
    }
    return {
        "violation_counts": counts,
        "total_violation_count": int(sum(counts.values())),
        "maximum_speed_mps": float(speed.max(initial=0.0)),
        "maximum_acceleration_mps2": float(longitudinal.max(initial=0.0)),
        "maximum_deceleration_mps2": float(max(0.0, -longitudinal.min(initial=0.0))),
        "maximum_yaw_rate_radps": float(yaw_rate.max(initial=0.0)),
        "maximum_lateral_acceleration_mps2": float(
            lateral_acceleration.max(initial=0.0)
        ),
        "maximum_heading_velocity_mismatch_rad": float(
            heading_mismatch[speed > 0.1].max(initial=0.0)
        ),
        "maximum_contact_error_m": float(contact_error.max(initial=0.0)),
        "maximum_collision_depth_m": float(collision_depth_array.max(initial=0.0)),
        "mean_jerk_energy": jerk_energy,
    }


def _correct_pair(
    values: np.ndarray,
    index: int,
    target_delta: np.ndarray,
    fixed: np.ndarray,
) -> None:
    current = values[index + 1] - values[index]
    error = current - target_delta
    left_fixed = bool(fixed[index])
    right_fixed = bool(fixed[index + 1])
    if left_fixed and right_fixed:
        return
    if left_fixed:
        values[index + 1] -= error
    elif right_fixed:
        values[index] += error
    else:
        values[index] += 0.5 * error
        values[index + 1] -= 0.5 * error


def project_road_contact(
    trajectory: PlanarTrajectory, *, road_z: np.ndarray
) -> PlanarTrajectory:
    road = np.asarray(road_z, dtype=np.float64)
    if road.shape != trajectory.times.shape or not np.isfinite(road).all():
        raise ValueError("road_z 必须与 trajectory times 对齐")
    positions = trajectory.positions.copy()
    positions[:, 2] = road
    return PlanarTrajectory(trajectory.times, positions, trajectory.yaws)


def project_vehicle_kinematics(
    trajectory: PlanarTrajectory,
    *,
    limits: KinematicLimits,
    road_z: np.ndarray,
    obstacles: Sequence[CircularObstacle] = (),
    actor_radius_m: float = 1.25,
    maximum_iterations: int = 200,
    convergence_tolerance: float = 1e-6,
) -> tuple[PlanarTrajectory, dict[str, Any]]:
    """确定性 POCS 投影；首尾 xy/yaw 固定，road contact 每轮精确执行。"""

    if maximum_iterations <= 0 or not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0:
        raise ValueError("projection iteration/tolerance 非法")
    road = np.asarray(road_z, dtype=np.float64)
    if road.shape != trajectory.times.shape or not np.isfinite(road).all():
        raise ValueError("road_z 必须与 trajectory times 对齐")
    positions = trajectory.positions.copy()
    yaws = trajectory.yaws.copy()
    fixed = np.zeros(len(positions), dtype=bool)
    fixed[[0, -1]] = True
    endpoint_xy = positions[[0, -1], :2].copy()
    endpoint_yaw = yaws[[0, -1]].copy()
    dt = np.diff(trajectory.times)
    converged = False
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        before_positions = positions.copy()
        before_yaws = yaws.copy()
        positions[:, 2] = road
        for index, delta_t in enumerate(dt):
            delta = positions[index + 1, :2] - positions[index, :2]
            distance = float(np.linalg.norm(delta))
            maximum = limits.maximum_speed_mps * delta_t
            if distance > maximum and distance > 0.0:
                _correct_pair(
                    positions[:, :2], index, delta * (maximum / distance), fixed
                )
        for index in range(1, len(positions) - 1):
            left_dt, right_dt = dt[index - 1], dt[index]
            mean_dt = 0.5 * (left_dt + right_dt)
            left_velocity = (positions[index, :2] - positions[index - 1, :2]) / left_dt
            right_velocity = (positions[index + 1, :2] - positions[index, :2]) / right_dt
            acceleration = (right_velocity - left_velocity) / mean_dt
            reference_velocity = 0.5 * (left_velocity + right_velocity)
            longitudinal = float(
                np.dot(acceleration, reference_velocity)
                / max(np.linalg.norm(reference_velocity), 1e-9)
            )
            bound = (
                limits.maximum_deceleration_mps2
                if longitudinal < 0.0
                else limits.maximum_acceleration_mps2
            )
            magnitude = float(np.linalg.norm(acceleration))
            if magnitude > bound:
                target = acceleration * (bound / magnitude)
                coefficient = (1.0 / left_dt + 1.0 / right_dt) / mean_dt
                positions[index, :2] += (acceleration - target) / coefficient
        velocity = np.diff(positions[:, :2], axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        for index, delta_t in enumerate(dt):
            delta_yaw = float(yaws[index + 1] - yaws[index])
            yaw_bound = limits.maximum_yaw_rate_radps * delta_t
            if speed[index] > 1e-6:
                yaw_bound = min(
                    yaw_bound,
                    limits.maximum_lateral_acceleration_mps2
                    * delta_t
                    / speed[index],
                )
            if abs(delta_yaw) > yaw_bound:
                _correct_pair(
                    yaws[:, None],
                    index,
                    np.asarray([math.copysign(yaw_bound, delta_yaw)]),
                    fixed,
                )
        velocity = np.diff(positions[:, :2], axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        headings = np.arctan2(velocity[:, 1], velocity[:, 0])
        for index in range(1, len(yaws) - 1):
            if speed[index - 1] <= 0.1:
                continue
            mismatch = float(_angle_delta(yaws[index], headings[index - 1]))
            if abs(mismatch) > limits.maximum_heading_velocity_mismatch_rad:
                yaws[index] = headings[index - 1] + math.copysign(
                    limits.maximum_heading_velocity_mismatch_rad, mismatch
                )
        for index in range(1, len(positions) - 1):
            point = positions[index, :2]
            for obstacle in obstacles:
                center = np.asarray(obstacle.center_xy, dtype=np.float64)
                delta = point - center
                distance = float(np.linalg.norm(delta))
                required = actor_radius_m + obstacle.radius_m
                if distance < required:
                    direction = (
                        delta / distance
                        if distance > 1e-9
                        else np.asarray([1.0, 0.0], dtype=np.float64)
                    )
                    point[:] = center + direction * required
        positions[[0, -1], :2] = endpoint_xy
        yaws[[0, -1]] = endpoint_yaw
        change = max(
            float(np.max(np.abs(positions - before_positions))),
            float(np.max(np.abs(yaws - before_yaws))),
        )
        if change <= convergence_tolerance:
            converged = True
            break
    projected = PlanarTrajectory(trajectory.times, positions, yaws)
    metrics = trajectory_metrics(
        projected,
        limits=limits,
        road_z=road,
        obstacles=obstacles,
        actor_radius_m=actor_radius_m,
    )
    diagnostics = {
        "iterations": iterations,
        "converged": converged,
        "convergence_tolerance": convergence_tolerance,
        "endpoint_translation_error_m": float(
            np.max(np.linalg.norm(projected.positions[[0, -1], :2] - endpoint_xy, axis=1))
        ),
        "endpoint_yaw_error_rad": float(
            np.max(np.abs(projected.yaws[[0, -1]] - endpoint_yaw))
        ),
        "metrics": metrics,
    }
    return projected, diagnostics


def desired_trajectory_rmse_m(
    candidate: PlanarTrajectory, desired: PlanarTrajectory
) -> float:
    if not np.array_equal(candidate.times, desired.times):
        raise ValueError("candidate/desired times 必须完全一致")
    delta = candidate.positions - desired.positions
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))

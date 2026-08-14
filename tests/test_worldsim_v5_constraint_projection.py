from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v5.constraint_projection import (
    CircularObstacle,
    KinematicLimits,
    PlanarTrajectory,
    desired_trajectory_rmse_m,
    heading_velocity_mismatch,
    minimum_jerk_smooth,
    project_road_contact,
    project_vehicle_kinematics,
    trajectory_metrics,
    v4_frozen_bspline_comparator,
)


def _limits() -> KinematicLimits:
    return KinematicLimits(
        maximum_speed_mps=8.0,
        maximum_acceleration_mps2=4.0,
        maximum_deceleration_mps2=6.0,
        maximum_yaw_rate_radps=0.7,
        maximum_lateral_acceleration_mps2=4.0,
        maximum_heading_velocity_mismatch_rad=0.4,
        maximum_contact_error_m=0.2,
    )


def _trajectory() -> PlanarTrajectory:
    times = np.linspace(0.0, 3.0, 7)
    positions = np.stack(
        [
            np.asarray([0.0, 1.0, 2.0, 8.0, 8.5, 9.0, 10.0]),
            np.asarray([0.0, 0.2, -0.2, 3.0, -1.0, 0.0, 0.0]),
            np.asarray([0.5, 0.4, 0.7, 0.6, 0.3, 0.2, 0.5]),
        ],
        axis=1,
    )
    yaws = np.asarray([0.0, 0.1, 0.2, 1.8, -1.0, 0.1, 0.0])
    return PlanarTrajectory(times, positions, yaws)


def test_minimum_jerk_preserves_endpoints_and_reduces_jerk() -> None:
    desired = _trajectory()
    limits = _limits()
    before = trajectory_metrics(desired, limits=limits)["mean_jerk_energy"]
    smoothed = minimum_jerk_smooth(desired, jerk_weight=5.0)
    after = trajectory_metrics(smoothed, limits=limits)["mean_jerk_energy"]
    assert np.array_equal(smoothed.positions[[0, -1]], desired.positions[[0, -1]])
    assert np.array_equal(smoothed.yaws[[0, -1]], desired.yaws[[0, -1]])
    assert after < before
    assert desired_trajectory_rmse_m(smoothed, desired) > 0.0


def test_contact_projection_is_exact_without_changing_xy_or_yaw() -> None:
    desired = _trajectory()
    road = np.linspace(0.0, 0.1, len(desired.times))
    projected = project_road_contact(desired, road_z=road)
    assert np.array_equal(projected.positions[:, :2], desired.positions[:, :2])
    assert np.array_equal(projected.yaws, desired.yaws)
    assert np.array_equal(projected.positions[:, 2], road)


def test_t2_reuses_v4_frozen_bspline_on_the_same_desired_samples() -> None:
    desired = _trajectory()
    comparator = v4_frozen_bspline_comparator(
        desired, control_point_count=4, acceleration_regularization=0.10
    )
    assert np.array_equal(comparator.times, desired.times)
    assert comparator.positions.shape == desired.positions.shape
    assert np.isfinite(comparator.positions).all()
    assert desired_trajectory_rmse_m(comparator, desired) > 0.0


def test_vehicle_projection_reduces_violation_count_and_keeps_endpoints() -> None:
    desired = minimum_jerk_smooth(_trajectory(), jerk_weight=5.0)
    road = np.zeros(len(desired.times))
    obstacle = CircularObstacle(center_xy=(5.0, 0.0), radius_m=0.5)
    limits = _limits()
    before = trajectory_metrics(
        desired,
        limits=limits,
        road_z=road,
        obstacles=[obstacle],
        actor_radius_m=1.0,
    )
    projected, diagnostics = project_vehicle_kinematics(
        desired,
        limits=limits,
        road_z=road,
        obstacles=[obstacle],
        actor_radius_m=1.0,
        maximum_iterations=200,
        convergence_tolerance=1e-6,
    )
    after = diagnostics["metrics"]
    assert after["total_violation_count"] < before["total_violation_count"]
    assert diagnostics["endpoint_translation_error_m"] == 0.0
    assert diagnostics["endpoint_yaw_error_rad"] == 0.0
    assert np.array_equal(projected.positions[:, 2], road)


def test_metrics_detect_each_major_violation_family() -> None:
    metrics = trajectory_metrics(
        _trajectory(),
        limits=_limits(),
        road_z=np.zeros(7),
        obstacles=[CircularObstacle(center_xy=(8.0, 3.0), radius_m=0.5)],
        actor_radius_m=1.0,
    )
    counts = metrics["violation_counts"]
    assert counts["speed"] > 0
    assert counts["yaw_rate"] > 0
    assert counts["lateral_acceleration"] > 0
    assert counts["contact"] > 0
    assert counts["collision"] > 0


def test_heading_metric_ignores_low_speed_noise_and_allows_reverse() -> None:
    mismatch = heading_velocity_mismatch(
        np.asarray([0.0, 0.0]),
        np.asarray([np.pi, np.pi / 2.0]),
        allow_reverse=True,
    )
    assert mismatch[0] < 1e-12
    assert np.isclose(mismatch[1], np.pi / 2.0)


def test_convergence_never_claims_an_infeasible_projection() -> None:
    desired = _trajectory()
    road = np.zeros(len(desired.times))
    projected, diagnostics = project_vehicle_kinematics(
        desired,
        limits=_limits(),
        road_z=road,
        maximum_iterations=2,
        convergence_tolerance=1e9,
    )
    assert diagnostics["converged"] is (
        diagnostics["metrics"]["total_violation_count"] == 0
    )
    assert diagnostics["feasible"] is (
        diagnostics["metrics"]["total_violation_count"] == 0
    )
    assert projected.positions.shape == desired.positions.shape

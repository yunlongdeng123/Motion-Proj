from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v32.actor_asset_schema import (
    fit_gaussians_to_actor_box,
    gaussian_support_bounds,
    project_actor_points,
    quaternion_to_matrix_wxyz,
    transform_gaussians,
    validate_actor_asset,
)


def test_anisotropic_affine_preserves_covariance_as_scale_quaternion() -> None:
    means = np.asarray([[1.0, 2.0, 3.0]])
    scales = np.asarray([[0.1, 0.2, 0.3]])
    quats = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    linear = np.asarray([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0], [-4.0, 0.0, 0.0]])
    transformed = transform_gaussians(
        means=means,
        scales=scales,
        quats=quats,
        linear=linear,
        translation=np.asarray([1.0, -2.0, 0.5]),
    )
    out_means, out_scales, out_quats = transformed
    np.testing.assert_allclose(out_means, [[7.0, 4.0, -3.5]])
    rotations = quaternion_to_matrix_wxyz(out_quats)
    covariance = (rotations * np.square(out_scales)[:, None, :]) @ np.swapaxes(
        rotations, 1, 2
    )
    expected = linear @ np.diag(np.square(scales[0])) @ linear.T
    np.testing.assert_allclose(covariance[0], expected, atol=1e-10)


def test_box_fit_matches_three_sigma_lwh() -> None:
    means = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scales = np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    quats = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)
    target = np.asarray([5.2, 1.9, 1.8])
    fitted = fit_gaussians_to_actor_box(
        means=means,
        scales=scales,
        quats=quats,
        target_lwh=target,
    )
    lower, upper = gaussian_support_bounds(
        fitted["means"], fitted["scales"], fitted["quats"], 3.0
    )
    np.testing.assert_allclose(upper - lower, target, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose((upper + lower) / 2.0, np.zeros(3), atol=1e-6)


def test_box_fit_uses_official_nurec_positive_y_rotation_direction() -> None:
    means = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    scales = np.asarray([[0.1, 0.1, 0.1]] * 2)
    quats = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)
    fitted = fit_gaussians_to_actor_box(
        means=means,
        scales=scales,
        quats=quats,
        target_lwh=np.asarray([4.0, 2.0, 1.5]),
        orientation_y_degrees=90.0,
    )
    delta = fitted["means"][1] - fitted["means"][0]
    assert abs(float(delta[0])) < 1e-6
    assert float(delta[2]) > 0


def test_actor_asset_schema_rejects_out_of_range_rgb() -> None:
    arrays = {
        "means": np.zeros((1, 3)),
        "scales": np.ones((1, 3)),
        "quats": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        "rgb": np.asarray([[1.1, 0.0, 0.0]]),
        "opacity": np.asarray([0.5]),
    }
    try:
        validate_actor_asset(arrays)
    except ValueError as error:
        assert "rgb" in str(error)
    else:
        raise AssertionError("out-of-range rgb 应被拒绝")


def test_actor_local_world_camera_projection_and_lateral_shift() -> None:
    means = np.asarray([[0.0, 0.0, 0.0]])
    world_actor = np.eye(4)
    world_actor[2, 3] = 10.0
    intrinsics = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    original, depth = project_actor_points(
        means_actor=means,
        T_world_actor=world_actor,
        T_camera_world=np.eye(4),
        intrinsics=intrinsics,
    )
    np.testing.assert_allclose(original, [[50.0, 50.0]])
    np.testing.assert_allclose(depth, [10.0])
    lateral = world_actor.copy()
    lateral[1, 3] += 1.0
    shifted, _ = project_actor_points(
        means_actor=means,
        T_world_actor=lateral,
        T_camera_world=np.eye(4),
        intrinsics=intrinsics,
    )
    np.testing.assert_allclose(shifted, [[50.0, 60.0]])

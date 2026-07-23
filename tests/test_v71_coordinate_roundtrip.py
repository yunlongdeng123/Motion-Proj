import numpy as np
import pytest

from motion_proj.resim.coordinates import (
    CoordinateError,
    box_corners_actor,
    compose_transform,
    invert_transform,
    project_world_points,
    transform_points,
    validate_transform,
)


def _yaw_transform(yaw, translation):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = translation
    return transform


def test_translation_yaw_and_box_corners_round_trip():
    T_world_actor = _yaw_transform(0.37, [12.0, -3.0, 1.2])
    corners_actor = box_corners_actor([4.0, 2.0, 1.5])
    corners_world = transform_points(T_world_actor, corners_actor)
    recovered = transform_points(invert_transform(T_world_actor), corners_world)
    np.testing.assert_allclose(recovered, corners_actor, atol=1e-9)
    np.testing.assert_allclose(
        compose_transform(invert_transform(T_world_actor), T_world_actor),
        np.eye(4),
        atol=1e-9,
    )


def test_camera_projection_uses_explicit_T_camera_world():
    points_world = np.asarray([[0.0, 0.0, 10.0], [1.0, 2.0, 5.0]])
    intrinsics = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pixels, depth = project_world_points(
        points_world,
        T_camera_world=np.eye(4),
        intrinsics=intrinsics,
    )
    np.testing.assert_allclose(pixels, [[50.0, 40.0], [70.0, 80.0]])
    np.testing.assert_allclose(depth, [10.0, 5.0])


def test_non_rigid_transform_fails_closed():
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(CoordinateError, match="非正交"):
        validate_transform(bad)

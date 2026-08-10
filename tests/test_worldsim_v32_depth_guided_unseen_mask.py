from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v32.depth_guided_unseen_mask import (
    project_world,
    splat_rgbd_to_target,
    unproject_rgbd,
)


def camera() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64)


def test_unproject_and_project_roundtrip() -> None:
    k, c2w = camera()
    depth = np.array([[2.0, 3.0], [4.0, 5.0]])
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    points, colors, pixels = unproject_rgbd(
        depth=depth,
        rgb=rgb,
        valid=np.ones((2, 2), dtype=bool),
        intrinsics=k,
        camera_to_world=c2w,
    )
    uv, z = project_world(
        points_world=points, intrinsics=k, camera_to_world=c2w
    )
    np.testing.assert_allclose(uv, pixels)
    np.testing.assert_allclose(z, depth[pixels[:, 1], pixels[:, 0]])
    np.testing.assert_array_equal(colors, rgb[pixels[:, 1], pixels[:, 0]])


def test_splat_obeys_target_mask_depth_gate_and_zbuffer() -> None:
    k, c2w = camera()
    source_depth = np.array([[2.0, 3.0], [2.0, 2.0]])
    source_rgb = np.array(
        [[[10, 0, 0], [20, 0, 0]], [[30, 0, 0], [40, 0, 0]]], dtype=np.uint8
    )
    target_depth = np.full((2, 2), 2.0)
    target_mask = np.array([[True, True], [False, True]])
    splat = splat_rgbd_to_target(
        source_depth=source_depth,
        source_rgb=source_rgb,
        source_valid=np.ones((2, 2), dtype=bool),
        source_intrinsics=k,
        source_camera_to_world=c2w,
        target_depth=target_depth,
        target_mask=target_mask,
        target_intrinsics=k,
        target_camera_to_world=c2w,
        absolute_depth_tolerance_m=0.1,
        relative_depth_tolerance=0.0,
    )
    assert splat.observed.tolist() == [[True, False], [False, True]]
    assert splat.rgb[0, 0, 0] == 10
    assert splat.rgb[1, 1, 0] == 40


def test_invalid_shapes_and_tolerance_fail_closed() -> None:
    k, c2w = camera()
    with pytest.raises(ValueError, match="尺寸"):
        unproject_rgbd(
            depth=np.ones((2, 2)),
            rgb=np.ones((2, 2)),
            valid=np.ones((2, 2)),
            intrinsics=k,
            camera_to_world=c2w,
        )
    with pytest.raises(ValueError, match="容差"):
        splat_rgbd_to_target(
            source_depth=np.ones((1, 1)),
            source_rgb=np.ones((1, 1, 3), dtype=np.uint8),
            source_valid=np.ones((1, 1), dtype=bool),
            source_intrinsics=k,
            source_camera_to_world=c2w,
            target_depth=np.ones((1, 1)),
            target_mask=np.ones((1, 1), dtype=bool),
            target_intrinsics=k,
            target_camera_to_world=c2w,
            absolute_depth_tolerance_m=-1,
            relative_depth_tolerance=0,
        )

from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_worldsim_v4_adgs import (
    ADGSAdapterError,
    aligned_world_to_camera,
    build_partition_flags,
    partition_name,
    project_visible_colors,
    scaled_intrinsic,
)


def test_partition_contract_for_196_frames_and_three_cameras() -> None:
    flags, rows = build_partition_flags()
    assert flags.shape == (588,)
    counts = {
        split: sum(row["partition"] == split for row in rows)
        for split in ("train", "development", "heldout")
    }
    assert counts == {"train": 354, "development": 117, "heldout": 117}
    assert int(flags.sum()) == 234
    assert [partition_name(i) for i in range(5)] == [
        "train",
        "train",
        "development",
        "train",
        "heldout",
    ]


def test_intrinsic_and_aligned_extrinsic_are_half_resolution() -> None:
    intrinsic = scaled_intrinsic(
        np.asarray([1200.0, 1000.0, 800.0, 450.0]), (800, 450)
    )
    np.testing.assert_allclose(
        intrinsic,
        np.asarray([[600.0, 0.0, 400.0], [0.0, 500.0, 225.0], [0.0, 0.0, 1.0]]),
    )
    origin = np.eye(4)
    origin[0, 3] = 10.0
    camera = origin.copy()
    camera[1, 3] = 2.0
    world_to_camera = aligned_world_to_camera(camera, origin)
    np.testing.assert_allclose(world_to_camera[:3, 3], [0.0, -2.0, 0.0])


def test_projection_keeps_visible_point_and_samples_rgb() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [8.0, 0.0, 1.0]], dtype=np.float32)
    intrinsic = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[1, 1] = [10, 20, 30]
    visible, colors = project_visible_colors(
        points, [intrinsic], [np.eye(4, dtype=np.float32)], [image]
    )
    assert visible.tolist() == [True, False]
    assert colors[0].tolist() == [10, 20, 30]


def test_projection_rejects_camera_count_mismatch() -> None:
    with pytest.raises(ADGSAdapterError, match="相机数"):
        project_visible_colors(np.zeros((1, 3)), [], [np.eye(4)], [])

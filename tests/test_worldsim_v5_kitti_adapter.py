from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v5.datasets.kitti import (
    KittiV5AdapterError,
    freeze_sensor_frame_policy,
    parse_tracking_labels,
    parse_tracking_calibration,
    parse_tracking_oxts,
    project_camera_points,
    transform_lidar_to_rectified_camera,
    world_from_rectified_camera,
)


def test_tracking_calibration_accepts_colon_and_whitespace_rows(tmp_path) -> None:
    calibration = tmp_path / "0000.txt"
    calibration.write_text(
        "P2: 700 0 600 0 0 700 180 0 0 0 1 0\n"
        "P3: 700 0 600 -350 0 700 180 0 0 0 1 0\n"
        "R_rect 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_cam 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "Tr_imu_velo 1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )
    result = parse_tracking_calibration(calibration)
    assert result["P2"].shape == (3, 4)
    assert np.allclose(result["T_imu_velo"], np.eye(4))


def _oxts_row(latitude: float, longitude: float, yaw: float = 0.0) -> str:
    values = [latitude, longitude, 10.0, 0.0, 0.0, yaw] + [0.0] * 24
    return " ".join(str(value) for value in values)


def test_tracking_oxts_uses_30_field_geodetic_pose(tmp_path) -> None:
    path = tmp_path / "0000.txt"
    path.write_text(
        _oxts_row(49.0, 8.0) + "\n" + _oxts_row(49.0, 8.00001, 0.1) + "\n",
        encoding="utf-8",
    )
    poses = parse_tracking_oxts(path)
    assert np.allclose(poses[0], np.eye(4))
    assert np.linalg.norm(poses[1][:3, 3]) > 0
    calibration = {
        "R_rect": np.eye(4),
        "T_velo_cam": np.eye(4),
        "T_imu_velo": np.eye(4),
    }
    assert np.allclose(world_from_rectified_camera(poses[1], calibration), poses[1])
    path.write_text(" ".join(["0"] * 12), encoding="utf-8")
    with pytest.raises(KittiV5AdapterError, match="30"):
        parse_tracking_oxts(path)


def test_missing_lidar_is_explicit_abstain_not_silent_intersection() -> None:
    images = range(447)
    lidar = [frame for frame in images if frame not in {177, 178, 179, 180}]
    policy = freeze_sensor_frame_policy(images, images, lidar)
    assert policy["full_denominator_count"] == 447
    assert policy["evaluable_multimodal_count"] == 443
    assert policy["lidar_missing_abstain_frames"] == [177, 178, 179, 180]
    assert policy["complete_alignment"] is False

    camera_mismatch = freeze_sensor_frame_policy([0, 1], [0], [0, 1])
    assert camera_mismatch["full_denominator_count"] == 2
    assert camera_mismatch["stereo_unpaired_abstain_frames"] == [1]
    assert camera_mismatch["stereo_missing_abstain_frames"] == [1]


def test_projection_and_track_identity_contract(tmp_path) -> None:
    calibration = {
        "R_rect": np.eye(4),
        "T_velo_cam": np.eye(4),
    }
    camera = transform_lidar_to_rectified_camera(
        np.asarray([[1.0, 2.0, 5.0, 0.5]]), calibration
    )
    projection = np.asarray(
        [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    pixels, valid = project_camera_points(camera, projection)
    assert valid.tolist() == [True]
    assert np.allclose(pixels, [[70.0, 80.0]])
    labels = tmp_path / "0000.txt"
    labels.write_text(
        "0 7 Car 0 0 0 10 20 30 40 1.5 1.6 4.0 1.0 2.0 20.0 0.1\n",
        encoding="utf-8",
    )
    record = parse_tracking_labels(labels)[0]
    assert record["frame"] == 0
    assert record["track_id"] == 7

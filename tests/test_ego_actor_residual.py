import math

import pytest
import torch

from motion_proj.data.real_motion_targets import actor_residual_target


K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
I = torch.eye(4, dtype=torch.float64)


def _box(center, *, token="actor", attribute="vehicle.moving"):
    return {
        "annotation_token": f"ann-{token}",
        "instance_token": token,
        "category": "vehicle.car",
        "attributes": [attribute],
        "visibility": 4,
        "xyxy": [0.0, 0.0, 100.0, 80.0],
        "center_cam": center,
        "velocity_global": [2.0, 0.0, 0.0],
    }


def _target(first, second, *, dt=0.5, ego_next=None):
    return actor_residual_target(
        first,
        second,
        frame_index=0,
        dt_s=dt,
        intrinsics_t=K,
        intrinsics_tp1=K,
        cam2ego_t=I,
        cam2ego_tp1=I,
        ego2global_t=I,
        ego2global_tp1=I if ego_next is None else ego_next,
    )


def test_world_stationary_actor_has_zero_residual_under_ego_translation():
    ego_next = I.clone()
    ego_next[0, 3] = 1.0
    row = _target(
        _box([0.0, 0.0, 10.0], attribute="vehicle.parked"),
        _box([-1.0, 0.0, 10.0], attribute="vehicle.parked"),
        ego_next=ego_next,
    )
    assert row["finite"]
    assert row["motion_label"] == "stationary"
    assert row["residual_px"] == pytest.approx([0.0, 0.0], abs=1e-8)


def test_lateral_actor_motion_sign_and_variable_dt_units():
    first = _box([0.0, 0.0, 10.0])
    second = _box([1.0, 0.0, 10.0])
    half_second = _target(first, second, dt=0.5)
    one_second = _target(first, second, dt=1.0)
    assert half_second["residual_px"] == pytest.approx([10.0, 0.0])
    assert half_second["residual_speed_px_per_s"] == pytest.approx(20.0)
    assert one_second["residual_speed_px_per_s"] == pytest.approx(10.0)
    assert half_second["velocity_direction_cosine"] == pytest.approx(1.0)


def test_longitudinal_motion_changes_projection_at_off_axis_point():
    row = _target(_box([2.0, 0.0, 10.0]), _box([2.0, 0.0, 12.0]))
    assert row["finite"]
    assert row["residual_px"][0] == pytest.approx(-100.0 / 30.0)
    assert abs(row["residual_px"][1]) < 1e-8


def test_behind_camera_actor_is_invalid_not_zero_filled():
    row = _target(_box([0.0, 0.0, 10.0]), _box([0.0, 0.0, -1.0]))
    assert not row["finite"]
    assert math.isnan(row["residual_px"][0])


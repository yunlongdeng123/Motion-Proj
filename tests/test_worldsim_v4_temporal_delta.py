from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v4.repair_candidates import GaussianAssetBinding
from motion_proj.worldsim_v4.se3_bspline import (
    UniformCubicSE3Spline,
    cubic_basis,
    se3_exp,
)
from motion_proj.worldsim_v4.temporal_compiler import (
    TemporalEditRequest,
    compile_temporal_delta,
)
from motion_proj.worldsim_v4.temporal_metrics import (
    centroid_acceleration_jitter,
    frame_flicker_l1,
    identity_switch_count,
    mask_iou_jitter,
    relative_error_improvement,
    warp_l1,
)


def asset() -> GaussianAssetBinding:
    return GaussianAssetBinding("/asset/delta.npz", "c" * 64, 64, 2)


def test_cubic_basis_is_partition_of_unity() -> None:
    for time in np.linspace(0.0, 1.0, 21):
        assert cubic_basis(float(time)).sum() == pytest.approx(1.0)


def test_se3_exp_zero_is_identity() -> None:
    assert np.array_equal(se3_exp(np.zeros(6)), np.eye(4))


def test_se3_exp_rotation_is_rigid() -> None:
    transform = se3_exp(np.asarray([1.0, 2.0, 3.0, 0.1, -0.2, 0.3]))
    rotation = transform[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_constant_control_twist_has_zero_acceleration_energy() -> None:
    controls = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.1]), (6, 1))
    spline = UniformCubicSE3Spline(controls, start_time=0.0, knot_interval=1.0)
    assert spline.acceleration_energy() == pytest.approx(0.0, abs=1e-20)
    assert np.allclose(spline.twist(0.0), controls[0])
    assert np.allclose(spline.twist(spline.end_time), controls[0])


def test_spline_rejects_time_outside_domain() -> None:
    spline = UniformCubicSE3Spline(np.zeros((4, 6)), 1.0, 2.0)
    with pytest.raises(ValueError, match="outside"):
        spline.transform(0.0)


def test_compile_lateral_delta_and_rollback() -> None:
    controls = np.zeros((6, 6))
    controls[:, 0] = np.linspace(0.0, 1.0, 6)
    delta = compile_temporal_delta(
        TemporalEditRequest(
            request_id="lateral",
            scene="scene-0994",
            operation="LATERAL",
            base_checkpoint_sha256="d" * 64,
            start_time=0.0,
            end_time=3.0,
            erase_gaussian_ids=(1, 2),
            insert_asset=asset(),
            control_twists=controls,
        )
    )
    assert delta.state_at(-1.0).active is False
    assert delta.state_at(1.5).active is True
    assert delta.state_at(1.5).insert_transform.shape == (4, 4)
    assert delta.rollback_checkpoint_sha256() == "d" * 64
    assert not delta.base_mutated
    assert len(delta.sha256) == 64


def test_compile_remove_has_temporal_erase_without_insert() -> None:
    delta = compile_temporal_delta(
        TemporalEditRequest(
            request_id="remove",
            scene="scene-0139",
            operation="REMOVE",
            base_checkpoint_sha256="e" * 64,
            start_time=5.0,
            end_time=7.0,
            erase_gaussian_ids=(9,),
        )
    )
    state = delta.state_at(6.0)
    assert state.erase_gaussian_ids == (9,)
    assert state.insert_asset is None
    assert state.insert_transform is None


def test_formal_clip_duration_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="duration"):
        TemporalEditRequest(
            request_id="short",
            scene="scene-0139",
            operation="REMOVE",
            base_checkpoint_sha256="e" * 64,
            start_time=0.0,
            end_time=1.0,
            erase_gaussian_ids=(9,),
        )


def test_temporal_metrics_known_values() -> None:
    frames = np.asarray([[[0.0]], [[1.0]], [[1.0]]])
    assert frame_flicker_l1(frames) == pytest.approx(0.5)
    assert warp_l1(frames[1:], frames[:-1]) == pytest.approx(0.5)
    masks = np.asarray(
        [
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[1, 1], [0, 0]],
        ]
    )
    assert mask_iou_jitter(masks) == pytest.approx(0.25)


def test_centroid_and_identity_metrics() -> None:
    linear = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert centroid_acceleration_jitter(linear) == pytest.approx(0.0)
    assert identity_switch_count([1, 1, None, 2, 2, 3]) == 2
    assert relative_error_improvement(1.0, 0.85) == pytest.approx(0.15)

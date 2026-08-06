from __future__ import annotations

import numpy as np
import pytest
import yaml

from motion_proj.worldsim_v3.calibration_endpoints import (
    bidirectional_boundary_distances,
    canonicalize_observed_rgb,
    coverage_status,
    cross_camera_residuals,
    depth_continuity_mask,
    inner_boundary,
    validate_endpoint_contract,
)


def test_endpoint_contract_is_frozen() -> None:
    payload = yaml.safe_load(
        open("configs/worldsim_v3/a1_endpoints_v1.yaml", encoding="utf-8")
    )
    validate_endpoint_contract(payload)
    payload["e1"]["camera_pairs"].append(["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"])
    with pytest.raises(ValueError, match="camera_pairs"):
        validate_endpoint_contract(payload)


def test_canonicalize_observed_rgb_inverts_affine() -> None:
    canonical = np.array([[[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]])
    affine = np.array(
        [[1.2, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, -0.1], [0.0, 0.0, 1.1, 0.05]]
    )
    observed = canonical @ affine[:, :3].T + affine[:, 3]
    np.testing.assert_allclose(
        canonicalize_observed_rgb(observed, affine), canonical, atol=1e-10
    )


def test_canonicalize_observed_rgb_inverts_per_pixel_affine() -> None:
    canonical = np.array([[[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]])
    affine = np.zeros((1, 2, 3, 4), dtype=np.float64)
    affine[..., :3] = np.eye(3)
    affine[0, 0, :, 3] = [0.1, 0.0, -0.1]
    affine[0, 1, :, 3] = [-0.1, 0.2, 0.0]
    observed = canonical + affine[..., 3]
    np.testing.assert_allclose(
        canonicalize_observed_rgb(observed, affine), canonical, atol=1e-10
    )


def test_depth_continuity_rejects_discontinuity_neighborhood() -> None:
    depth = np.ones((7, 7), dtype=np.float64) * 10.0
    depth[:, 4:] = 20.0
    valid = depth_continuity_mask(depth, maximum_relative_edge=0.05, dilation_radius=1)
    assert valid[:, 1].all()
    assert not valid[:, 2:6].any()


def test_cross_camera_identity_has_zero_residual() -> None:
    height, width = 12, 16
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((xx / width, yy / height, np.full_like(xx, 0.5)), axis=-1)
    rgb = rgb.astype(np.float64)
    depth = np.ones((height, width), dtype=np.float64) * 10.0
    support = np.ones((height, width), dtype=bool)
    c2w = np.eye(4)
    intrinsics = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]])
    result = cross_camera_residuals(
        source_rgb=rgb,
        source_depth=depth,
        source_support=support,
        source_camera_to_world=c2w,
        source_intrinsics=intrinsics,
        target_rgb=rgb,
        target_depth=depth,
        target_support=support,
        target_camera_to_world=c2w,
        target_intrinsics=intrinsics,
        grid_stride_pixels=2,
        maximum_relative_occlusion_error=0.01,
    )
    assert result["candidate_count"] == 48
    np.testing.assert_allclose(result["residuals"], 0.0, atol=1e-12)


def test_cross_camera_occlusion_filter_can_abstain() -> None:
    shape = (8, 8)
    result = cross_camera_residuals(
        source_rgb=np.zeros((*shape, 3)),
        source_depth=np.ones(shape) * 10.0,
        source_support=np.ones(shape, dtype=bool),
        source_camera_to_world=np.eye(4),
        source_intrinsics=np.eye(3),
        target_rgb=np.zeros((*shape, 3)),
        target_depth=np.ones(shape) * 3.0,
        target_support=np.ones(shape, dtype=bool),
        target_camera_to_world=np.eye(4),
        target_intrinsics=np.eye(3),
        grid_stride_pixels=2,
        maximum_relative_occlusion_error=0.01,
    )
    assert result["candidate_count"] > 0
    assert result["residuals"].size == 0


def test_boundary_distance_is_zero_for_equal_masks() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:8, 3:9] = True
    assert inner_boundary(mask).sum() == 20
    result = bidirectional_boundary_distances(mask, mask)
    np.testing.assert_allclose(result["combined"], 0.0)


def test_coverage_status_freezes_abstain_reasons() -> None:
    result = coverage_status(
        valid_count=2,
        candidate_count=100,
        minimum_valid_count=3,
        minimum_coverage=0.01,
        zero_reason="ZERO_CROSS_CAMERA_SUPPORT",
    )
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "INSUFFICIENT_VALID_SUPPORT"

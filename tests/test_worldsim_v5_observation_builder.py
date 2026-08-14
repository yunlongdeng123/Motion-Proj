from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v5.observation_builder import (
    build_observation_chunk,
    signed_boundary_distance,
    sparse_contribution_selection,
    stable_sigmoid,
)


def test_signed_boundary_distance_preserves_inside_outside_sign() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    distance = signed_boundary_distance(mask)
    assert distance[2, 2] > distance[1, 1] > 0
    assert distance[0, 0] < 0


def test_sparse_selection_reports_dropped_mass() -> None:
    selected, report = sparse_contribution_selection(
        np.asarray([0.5, 1e-5, 0.25]), minimum_weight=1e-4
    )
    assert selected.tolist() == [True, False, True]
    assert report["kept_count"] == 2
    assert np.isclose(report["dropped_contribution_mass"], 1e-5)


def test_observation_chunk_uses_logits_and_quality_gate() -> None:
    logits = np.asarray([[4.0, -4.0], [4.0, -4.0]], dtype=np.float32)
    binary = logits > 0
    chunk = build_observation_chunk(
        scene="scene-0471",
        role="actor",
        view_id=3,
        frame_id=10,
        camera_id=0,
        gaussian_count=2,
        gaussian_id=np.asarray([0, 1]),
        pixel_id=np.asarray([0, 1]),
        projected_pixel=np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float32),
        contribution_weight=np.asarray([0.8, 0.7], dtype=np.float32),
        projected_depth=np.asarray([5.1, 5.2], dtype=np.float32),
        first_hit_depth=np.asarray([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32),
        first_hit_valid=np.ones((2, 2), dtype=bool),
        mask_logits=logits,
        mask_binary=binary,
        mask_quality_accepted=True,
        view_angle_cosine=np.asarray([1.0, 0.5], dtype=np.float32),
        lidar_support=None,
        depth_absolute_tolerance_m=0.25,
        depth_relative_tolerance=0.02,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    assert chunk["positive_observation"].tolist() == [1, 0]
    assert chunk["negative_observation"].tolist() == [0, 1]
    assert np.allclose(chunk["sam_probability"], stable_sigmoid(np.asarray([4.0, -4.0])))
    assert chunk["lidar_support_available"].tolist() == [0, 0]
    assert np.all(chunk["reliability"] > 0)

    rejected = build_observation_chunk(
        scene="scene-0471",
        role="actor",
        view_id=3,
        frame_id=10,
        camera_id=0,
        gaussian_count=2,
        gaussian_id=np.asarray([0, 1]),
        pixel_id=np.asarray([0, 1]),
        projected_pixel=np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float32),
        contribution_weight=np.asarray([0.8, 0.7], dtype=np.float32),
        projected_depth=np.asarray([5.1, 5.2], dtype=np.float32),
        first_hit_depth=np.asarray([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32),
        first_hit_valid=np.ones((2, 2), dtype=bool),
        mask_logits=logits,
        mask_binary=binary,
        mask_quality_accepted=False,
        view_angle_cosine=np.asarray([1.0, 0.5], dtype=np.float32),
        lidar_support=None,
        depth_absolute_tolerance_m=0.25,
        depth_relative_tolerance=0.02,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    assert rejected["positive_observation"].tolist() == [0, 0]
    assert rejected["negative_observation"].tolist() == [0, 0]
    assert rejected["reliability"].tolist() == [0.0, 0.0]

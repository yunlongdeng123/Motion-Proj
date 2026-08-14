from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v5.evidence_schema import validate_observation_chunk
from motion_proj.worldsim_v5.observation_aggregation import (
    aggregate_intersection_observations,
)


def _intersections() -> dict[str, np.ndarray]:
    count = 5
    return {
        "scene": np.asarray("scene-0001"),
        "role": np.asarray("actor_union"),
        "sam_probability_source": np.asarray("sigmoid_sam2_logit"),
        "gaussian_id": np.asarray([0, 0, 1, 1, 2], dtype=np.int64),
        "view_id": np.full(count, 7, dtype=np.int32),
        "frame_id": np.full(count, 12, dtype=np.int32),
        "camera_id": np.full(count, 1, dtype=np.int8),
        "projected_pixel": np.asarray(
            [[0, 0], [2, 0], [3, 4], [5, 4], [8, 9]], dtype=np.float32
        ),
        "visibility": np.asarray([0.2, 0.8, 0.4, 0.6, 1e-6], dtype=np.float32),
        "sam_probability": np.asarray([0.2, 0.8, 0.9, 0.7, 0.5], dtype=np.float32),
        "sam_logit": np.asarray([-1.386, 1.386, 2.197, 0.847, 0.0], dtype=np.float32),
        "sam_probability_available": np.ones(count, dtype=np.int8),
        "mask_quality_accepted": np.ones(count, dtype=np.int8),
        "mask_boundary_distance": np.asarray([-2, 2, 5, 7, 0], dtype=np.float32),
        "depth_residual": np.asarray([0.1, 0.3, 0.0, 0.2, 0.0], dtype=np.float32),
        "depth_consistent": np.asarray([1, 0, 1, 1, 1], dtype=np.int8),
        "lidar_support": np.zeros(count, dtype=np.float32),
        "lidar_support_available": np.zeros(count, dtype=np.int8),
        "view_angle_cosine": np.asarray([1, 1, 0.5, 0.5, 1], dtype=np.float32),
        "positive_observation": np.asarray([0, 1, 1, 1, 0], dtype=np.int8),
        "negative_observation": np.asarray([1, 0, 0, 0, 1], dtype=np.int8),
        "reliability": np.ones(count, dtype=np.float32),
        "contribution_weight": np.asarray(
            [0.2, 0.8, 0.4, 0.6, 1e-6], dtype=np.float32
        ),
    }


def test_intersections_aggregate_to_one_row_per_gaussian_view() -> None:
    source = _intersections()
    validate_observation_chunk(source, gaussian_count=3)
    chunk, report = aggregate_intersection_observations(
        source,
        gaussian_count=3,
        minimum_contribution_mass=1e-4,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    validate_observation_chunk(chunk, gaussian_count=3)
    assert chunk["gaussian_id"].tolist() == [0, 1]
    assert np.allclose(chunk["projected_pixel"][0], [1.6, 0.0])
    assert np.isclose(chunk["sam_probability"][0], 0.68)
    assert np.isclose(chunk["positive_observation"][0], 0.8)
    assert np.isclose(chunk["negative_observation"][0], 0.2)
    assert np.isclose(chunk["depth_consistent"][0], 0.2)
    assert np.isclose(chunk["visibility"][0], 1.0 - np.exp(-1.0))
    assert report["input_intersection_count"] == 5
    assert report["input_unique_gaussian_count"] == 3
    assert report["kept_gaussian_count"] == 2
    assert report["dropped_gaussian_count"] == 1


def test_intersection_aggregation_is_permutation_invariant() -> None:
    source = _intersections()
    order = np.asarray([4, 2, 0, 3, 1])
    shuffled = {
        name: value if np.asarray(value).shape == () else np.asarray(value)[order]
        for name, value in source.items()
    }
    kwargs = {
        "gaussian_count": 3,
        "minimum_contribution_mass": 1e-4,
        "sam_confidence_floor": 0.1,
        "boundary_distance_scale_px": 4.0,
        "depth_residual_scale_m": 0.5,
    }
    first, first_report = aggregate_intersection_observations(source, **kwargs)
    second, second_report = aggregate_intersection_observations(shuffled, **kwargs)
    for name in first:
        assert np.allclose(first[name], second[name]) if np.asarray(first[name]).dtype.kind not in "US" else np.array_equal(first[name], second[name])
    assert first_report == second_report

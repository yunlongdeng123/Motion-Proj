from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v5.structured_evidence import StructuredEvidenceAccumulator


def _observations() -> dict[str, np.ndarray]:
    count = 4
    return {
        "scene": np.asarray("scene-0001"),
        "role": np.asarray("high_support"),
        "sam_probability_source": np.asarray("sigmoid_sam2_logit"),
        "gaussian_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "view_id": np.arange(count, dtype=np.int32),
        "frame_id": np.asarray([2, 7, 2, 7], dtype=np.int32),
        "camera_id": np.asarray([0, 1, 0, 1], dtype=np.int8),
        "projected_pixel": np.asarray(
            [[10, 20], [11, 20], [4, 5], [5, 5]], dtype=np.float32
        ),
        "visibility": np.asarray([1.0, 0.8, 1.0, 1.0], dtype=np.float32),
        "sam_probability": np.asarray([0.95, 0.9, 0.8, 0.2], dtype=np.float32),
        "sam_logit": np.asarray([2.94, 2.20, 1.39, -1.39], dtype=np.float32),
        "sam_probability_available": np.ones(count, dtype=np.int8),
        "mask_quality_accepted": np.ones(count, dtype=np.int8),
        "mask_boundary_distance": np.asarray([8.0, 6.0, 0.1, -0.1], dtype=np.float32),
        "depth_residual": np.asarray([0.1, 0.2, 0.1, 0.1], dtype=np.float32),
        "depth_consistent": np.ones(count, dtype=np.int8),
        "lidar_support": np.asarray([1, 1, 0, 0], dtype=np.int8),
        "lidar_support_available": np.ones(count, dtype=np.int8),
        "view_angle_cosine": np.asarray([1.0, 0.9, 1.0, 1.0], dtype=np.float32),
        "positive_observation": np.asarray([1, 1, 1, 0], dtype=np.int8),
        "negative_observation": np.asarray([0, 0, 0, 1], dtype=np.int8),
        "reliability": np.ones(count, dtype=np.float32),
        "contribution_weight": np.ones(count, dtype=np.float32),
    }


def test_streaming_accumulator_builds_valid_gaussian_table(tmp_path) -> None:
    accumulator = StructuredEvidenceAccumulator(
        scene="scene-0001",
        role="high_support",
        base_model=np.asarray([0, 1], dtype=np.int8),
        base_index=np.asarray([0, 0], dtype=np.int64),
        geometry={
            "center": np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            "covariance": np.stack([np.eye(3), np.eye(3)]).astype(np.float32),
            "normal_proxy": np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
            "normal_available": np.ones(2, dtype=np.int8),
        },
        prior_probability=np.asarray([0.1, 0.1]),
        prior_strength=2.0,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    observations = _observations()
    for indices in (np.asarray([0, 1]), np.asarray([2, 3])):
        chunk = {
            name: value if np.asarray(value).shape == () else np.asarray(value)[indices]
            for name, value in observations.items()
        }
        accumulator.add_chunk(chunk)
    table = accumulator.gaussian_table()
    assert table["unary_posterior"][0] > table["unary_posterior"][1]
    assert table["lidar_support_available"].tolist() == [0, 0]
    result = accumulator.save_gaussian_table(tmp_path / "gaussians.npz")
    assert result["observation_count"] == 4
    assert result["chunk_count"] == 2

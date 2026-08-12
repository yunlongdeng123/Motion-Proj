from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v32.inpainting_adapter import CompletionPoints
from motion_proj.worldsim_v4.repair_builders import (
    completion_points_to_repair_asset,
    normalized_repair_risks,
    roadpatch_delta_to_repair_asset,
)


def _points() -> CompletionPoints:
    return CompletionPoints(
        means=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        rgb=np.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32),
        scales=np.full((2, 3), 0.05, dtype=np.float32),
        confidence=np.asarray([0.9, 0.8], dtype=np.float32),
        observed_cross_view=np.ones(2, dtype=bool),
        source_pixels_xy=np.asarray([[10, 20], [30, 40]], dtype=np.int64),
    )


def test_completion_points_preserve_provenance_and_render_shapes() -> None:
    asset = completion_points_to_repair_asset(
        _points(),
        candidate_id="observed",
        method="OBSERVED",
        provenance="observed_cross_view",
        features_rest_shape=(15, 3),
        opacity=0.3,
        target_frame=7,
        target_camera_id=2,
    )
    assert asset["features_rest"].shape == (2, 15, 3)
    assert asset["point_provenance"].tolist() == [1, 1]
    assert asset["source_frames"].tolist() == [7, 7]
    assert np.allclose(np.exp(asset["raw_scales"]), 0.05)


def test_roadpatch_conversion_keeps_native_gaussian_lineage() -> None:
    delta = {
        "means": np.zeros((2, 3), np.float32),
        "raw_scales": np.zeros((2, 3), np.float32),
        "quats": np.tile(np.asarray([[1, 0, 0, 0]], np.float32), (2, 1)),
        "features_dc": np.zeros((2, 3), np.float32),
        "features_rest": np.zeros((2, 15, 3), np.float32),
        "raw_opacities": np.zeros((2, 1), np.float32),
        "source_gaussian_ids": np.asarray([41, 42], np.int64),
    }
    asset = roadpatch_delta_to_repair_asset(
        delta,
        candidate_id="donor",
        confidence=0.75,
        target_frame=7,
        target_camera_id=0,
    )
    assert asset["source_gaussian_ids"].tolist() == [41, 42]
    assert asset["point_provenance"].tolist() == [2, 2]
    assert np.allclose(asset["confidence"], 0.75)


def test_risk_normalization_is_bounded_and_fail_closed() -> None:
    risks = normalized_repair_risks(
        photo_l1_uint8=12.5,
        geometry_mae_m=float("nan"),
        temporal_std_uint8=None,
        uncertainty=1.2,
        gaussian_count=250,
        normalization={
            "photo_l1_uint8": 25,
            "geometry_mae_m": 0.5,
            "temporal_std_uint8": 50,
            "gaussian_count": 500,
        },
    )
    assert risks == {
        "photo_risk": 0.5,
        "geometry_risk": 1.0,
        "temporal_risk": 1.0,
        "uncertainty": 1.0,
        "compute_cost": 0.5,
    }

import torch

from motion_proj.diagnostics.feature_discriminability import (
    _sample_features,
    feature_route_decision,
    relation_metrics,
)


def test_pixel_to_feature_sampling_respects_cell_centers():
    features = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]]],
        [[[5.0, 6.0], [7.0, 8.0]]],
    ])
    points = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    sampled = _sample_features(features, points, (2, 2))
    torch.testing.assert_close(sampled[0, :, 0], torch.tensor([2.0, 7.0]))


def test_relation_metrics_tracks_unique_frozen_features():
    features = torch.zeros(2, 4, 2, 2)
    for channel in range(4):
        y, x = divmod(channel, 2)
        features[:, channel, y, x] = 1.0
    observed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    metrics, _ = relation_metrics(
        features, observed, observed.clone(), valid, ["dynamic_residual"],
        image_hw=(2, 2), temperature=0.01, pck_radius_cells=1.0,
    )
    assert metrics["argmax_observed_pck_1cell"] == 1.0
    assert metrics["argmax_projected_pck_1cell"] == 1.0
    assert metrics["correction_below_half_cell_fraction"] == 1.0
    assert metrics["observed_projected_heatmap_tv"] == 0.0


def test_feature_decision_stops_when_all_corrections_are_subcell():
    rows = [
        {
            "layer": "fine",
            "correction_below_half_cell_fraction": 0.8,
            "argmax_observed_pck_1cell": 0.9,
            "argmax_projected_pck_1cell": 0.9,
            "softargmax_projected_error_cells": 0.1,
        },
        {
            "layer": "coarse",
            "correction_below_half_cell_fraction": 1.0,
            "argmax_observed_pck_1cell": 0.1,
            "argmax_projected_pck_1cell": 0.1,
            "softargmax_projected_error_cells": 2.0,
        },
    ]
    decision = feature_route_decision(
        rows, max_fraction_below_half_cell=0.5, min_observed_pck_1cell=0.2
    )
    assert decision["classification"] == "feature_resolution_failure"
    assert not decision["passed"]
    assert decision["best_tracking_layer_diagnostic_only"] == "fine"

from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v51.superprimitive_probe import (
    edge_length_quantile_voxel_sizes,
    evaluate_e0a_density_gate,
    observation_density_report,
    voxel_assignments,
)


def test_edge_quantile_sizes_and_voxel_assignment_are_deterministic() -> None:
    centers = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.2, 0.1, 0.1], [2.5, 0.1, 0.1]],
        dtype=np.float64,
    )
    source = np.array([0, 1, 2], dtype=np.int64)
    target = np.array([1, 2, 3], dtype=np.int64)
    sizes, report = edge_length_quantile_voxel_sizes(
        centers, source, target, (0.25, 0.5, 0.75)
    )
    assert np.all(np.diff(sizes) > 0.0)
    assert report["edge_count"] == 3
    left, left_keys = voxel_assignments(centers, 1.0)
    right, right_keys = voxel_assignments(centers, 1.0)
    assert np.array_equal(left, right)
    assert np.array_equal(left_keys, right_keys)
    assert left.tolist() == [0, 0, 1, 2]


def test_edge_quantiles_exclude_zero_length_without_dropping_nodes() -> None:
    centers = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    sizes, report = edge_length_quantile_voxel_sizes(
        centers,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
        (0.25, 0.5, 0.75),
    )
    assert report["edge_count"] == 3
    assert report["positive_edge_count"] == 2
    assert report["zero_edge_count"] == 1
    assert sizes.tolist() == [1.25, 1.5, 1.75]
    assignment, _ = voxel_assignments(centers, float(sizes[0]))
    assert assignment.shape == (4,)
    assert assignment[0] == assignment[1]


def test_density_report_measures_union_rescue_and_seed_conflict() -> None:
    assignment = np.array([0, 0, 1, 1], dtype=np.int64)
    visibility = np.array(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
    )
    unary = np.array([0.95, 0.05, 0.5, 0.5], dtype=np.float32)
    report = observation_density_report(assignment, visibility, unary)
    assert report["node_count"] == 2
    assert report["raw_zero_observation_gaussian_count"] == 2
    assert report["rescued_zero_observation_gaussian_count"] == 2
    assert report["observation_union_gain_per_gaussian"] == 0.5
    assert report["conflicting_seed_node_count"] == 1
    assert report["conflicting_seed_gaussian_count"] == 2


def test_e0a_gate_requires_a_strict_level_in_every_scene() -> None:
    scenes = []
    for scene in ("a", "b", "c"):
        scenes.append(
            {
                "scene": scene,
                "raw": {
                    "node_count": 4,
                    "raw_mean_observed_views_per_gaussian": 0.5,
                },
                "levels": [
                    {
                        "level": "fine_q50",
                        "node_count": 2,
                        "gaussian_weighted_node_union_observed_views": 1.0,
                        "rescued_zero_observation_gaussian_count": 1,
                    }
                ],
            }
        )
    assert evaluate_e0a_density_gate(scenes, expected_scene_count=3)["pass"] is True
    scenes[-1]["levels"][0]["rescued_zero_observation_gaussian_count"] = 0
    assert evaluate_e0a_density_gate(scenes, expected_scene_count=3)["pass"] is False

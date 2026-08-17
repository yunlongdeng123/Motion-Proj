from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.progressive_propagation import (
    ACTOR,
    BACKGROUND,
    DEFAULT_THRESHOLDS,
    UNKNOWN,
    affinity_matrices_from_topologies,
    build_exact_logical_topologies,
    progressive_region_growing,
    soft_binary_cosine_affinity,
)


def test_soft_binary_cosine_uses_common_visibility_weighting() -> None:
    probability = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    visibility = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    affinity, report = soft_binary_cosine_affinity(
        probability,
        visibility,
        np.asarray([0, 0, 1]),
        np.asarray([1, 2, 2]),
        chunk_size=2,
    )
    np.testing.assert_allclose(affinity, [1.0, 0.0, 0.0], atol=1e-7)
    assert report["pair_count"] == 3
    assert report["jointly_visible_pair_count"] == 3
    assert report["mean_common_view_count"] == 4 / 3


def test_logical_topology_is_symmetric_and_exact_distance_two() -> None:
    direct, second = build_exact_logical_topologies(
        np.asarray([0, 1, 2]),
        np.asarray([1, 2, 3]),
        node_count=4,
        maximum_logical_distance=2,
    )
    np.testing.assert_array_equal(
        direct.toarray(),
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        second.toarray(),
        np.asarray(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_progressive_levels_expand_seeds_then_stop_unknown() -> None:
    unary = np.asarray([0.99, 0.85, 0.70, 0.50, 0.01], dtype=np.float32)
    topology = build_exact_logical_topologies(
        np.asarray([0, 1]),
        np.asarray([1, 2]),
        node_count=5,
        maximum_logical_distance=1,
    )
    probability_by_view = unary[:, None]
    visibility_by_view = np.ones_like(probability_by_view)
    affinity, _ = affinity_matrices_from_topologies(
        topology, probability_by_view, visibility_by_view
    )
    result = progressive_region_growing(
        unary,
        topologies=topology,
        affinities=affinity,
        progressive_thresholds=DEFAULT_THRESHOLDS,
    )
    np.testing.assert_array_equal(
        result["labels"], [ACTOR, ACTOR, ACTOR, UNKNOWN, BACKGROUND]
    )
    np.testing.assert_array_equal(result["posterior"], [1.0, 1.0, 1.0, 0.5, 0.0])
    assert result["report"]["actor_seed_count"] == 1
    assert result["report"]["background_seed_count"] == 1
    assert result["report"]["final_unknown_count"] == 1


def test_exact_conflict_tie_remains_unknown() -> None:
    unary = np.asarray([0.99, 0.50, 0.01], dtype=np.float32)
    topology = build_exact_logical_topologies(
        np.asarray([0, 1]),
        np.asarray([1, 2]),
        node_count=3,
        maximum_logical_distance=1,
    )
    affinity, _ = affinity_matrices_from_topologies(
        topology, unary[:, None], np.ones((3, 1), dtype=np.float32)
    )
    result = progressive_region_growing(
        unary,
        topologies=topology,
        affinities=affinity,
        progressive_thresholds=(0.5,),
    )
    np.testing.assert_array_equal(result["labels"], [ACTOR, UNKNOWN, BACKGROUND])


def test_edge_order_does_not_change_topology_affinity_or_result() -> None:
    source = np.asarray([0, 1, 2, 3, 1, 2])
    target = np.asarray([1, 2, 3, 4, 0, 1])
    order = np.asarray([5, 1, 4, 0, 3, 2])
    unary = np.asarray([0.99, 0.85, 0.70, 0.15, 0.01], dtype=np.float32)
    probability_by_view = np.stack((unary, unary), axis=1)
    visibility_by_view = np.ones_like(probability_by_view)
    outputs = []
    for left, right in ((source, target), (source[order], target[order])):
        topology = build_exact_logical_topologies(
            left, right, node_count=5, maximum_logical_distance=2
        )
        affinity, _ = affinity_matrices_from_topologies(
            topology, probability_by_view, visibility_by_view
        )
        outputs.append(
            progressive_region_growing(
                unary,
                topologies=topology,
                affinities=affinity,
            )
        )
    np.testing.assert_array_equal(outputs[0]["labels"], outputs[1]["labels"])
    np.testing.assert_array_equal(outputs[0]["posterior"], outputs[1]["posterior"])
    assert outputs[0]["report"] == outputs[1]["report"]

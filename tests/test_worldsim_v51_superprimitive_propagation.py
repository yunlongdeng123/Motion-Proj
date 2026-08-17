from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v51.superprimitive_propagation import (
    aggregate_node_evidence,
    broadcast_node_result,
    quotient_directed_edges,
)


def test_quotient_edges_drop_self_and_deduplicate() -> None:
    node_id = np.asarray([0, 0, 1, 2], dtype=np.int64)
    source = np.asarray([0, 1, 0, 1, 2, 3, 3], dtype=np.int64)
    target = np.asarray([1, 0, 2, 2, 3, 2, 0], dtype=np.int64)

    quotient_source, quotient_target, report = quotient_directed_edges(
        source, target, node_id
    )

    assert np.array_equal(quotient_source, np.asarray([0, 1, 2, 2], dtype=np.int64))
    assert np.array_equal(quotient_target, np.asarray([1, 2, 0, 1], dtype=np.int64))
    assert report == {
        "raw_directed_edge_count": 7,
        "intra_node_edge_count": 2,
        "cross_node_edge_count_before_deduplication": 5,
        "quotient_directed_edge_count": 4,
        "duplicate_cross_node_edge_count": 1,
    }


def test_node_evidence_uses_frozen_mean_weighted_mean_and_max_rules() -> None:
    node_id = np.asarray([0, 0, 1], dtype=np.int64)
    unary = np.asarray([0.1, 0.9, 0.8], dtype=np.float32)
    probability = np.asarray(
        [[0.2, 0.9], [0.8, 0.4], [0.7, 0.3]], dtype=np.float32
    )
    visibility = np.asarray(
        [[0.25, 0.0], [0.75, 0.5], [0.4, 0.0]], dtype=np.float32
    )

    node_unary, node_probability, node_visibility, report = aggregate_node_evidence(
        unary, probability, visibility, node_id
    )

    assert np.allclose(node_unary, [0.5, 0.8])
    assert np.allclose(node_probability, [[0.65, 0.4], [0.7, 0.5]])
    assert np.allclose(node_visibility, [[0.75, 0.5], [0.4, 0.0]])
    assert report["observed_node_view_count"] == 3
    assert report["member_count_maximum"] == 2


def test_broadcast_node_result_is_constant_within_node() -> None:
    node_id = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    label, posterior, assignment = broadcast_node_result(
        node_id,
        np.asarray([0, 1, -1], dtype=np.int8),
        np.asarray([0.0, 1.0, 0.5], dtype=np.float32),
        np.asarray([-2, 1, -1], dtype=np.int8),
    )

    assert np.array_equal(label, [0, 0, 1, -1, -1])
    assert np.array_equal(posterior, [0.0, 0.0, 1.0, 0.5, 0.5])
    assert np.array_equal(assignment, [-2, -2, 1, -1, -1])

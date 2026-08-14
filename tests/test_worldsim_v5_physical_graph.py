from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v5.physical_graph import (
    build_physical_graph,
    diffuse_graph_probability,
    graph_affinity_diagnostics,
)


def _table() -> dict[str, np.ndarray]:
    center = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    count = center.shape[0]
    identity = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0) * 0.01
    zeros = np.zeros(count, dtype=np.float32)
    unavailable = np.zeros(count, dtype=np.int8)
    return {
        "scene": np.asarray("scene-test"),
        "role": np.asarray("moving_rigid_union"),
        "gaussian_id": np.arange(count, dtype=np.int64),
        "base_model": np.asarray(["Background"] * 3 + ["RigidNodes"] * 2),
        "base_index": np.arange(count, dtype=np.int64),
        "center": center,
        "covariance": identity,
        "normal_proxy": np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        "normal_available": np.ones(count, dtype=np.int8),
        "prior": np.asarray([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "unary_posterior": np.asarray([0.05, 0.10, 0.20, 0.80, 0.95], dtype=np.float32),
        "unary_uncertainty": np.full(count, 0.25, dtype=np.float32),
        "effective_evidence_count": np.asarray([0.0, 1.0, 4.0, 1.0, 0.0], dtype=np.float32),
        "multi_view_disagreement": zeros.copy(),
        "boundary_ambiguity": np.asarray([0.0, 0.0, 0.8, 0.8, 0.0], dtype=np.float32),
        "depth_support": zeros.copy(),
        "lidar_support": zeros.copy(),
        "lidar_support_available": unavailable.copy(),
        "motion_consistency": zeros.copy(),
        "motion_consistency_available": unavailable.copy(),
    }


def test_graph_is_deterministic_and_does_not_consume_proxy_membership() -> None:
    table = _table()
    graph_a, metadata_a = build_physical_graph(
        table, candidate_k=2, normal_affinity_power=2.0
    )
    changed = dict(table)
    changed["base_model"] = np.asarray(["RigidNodes"] * 5)
    graph_b, metadata_b = build_physical_graph(
        changed, candidate_k=2, normal_affinity_power=2.0
    )
    assert metadata_a == metadata_b
    assert metadata_a["edge_count"] == 10
    for key in graph_a:
        np.testing.assert_array_equal(graph_a[key], graph_b[key])
    assert not np.any(graph_a["source_gaussian_id"] == graph_a["target_gaussian_id"])


def test_boundary_barrier_only_reduces_physical_affinity() -> None:
    graph, _ = build_physical_graph(
        _table(), candidate_k=2, normal_affinity_power=2.0
    )
    assert np.all(graph["edge_affinity"] <= graph["physical_affinity"])
    assert np.any(graph["edge_affinity"] < graph["physical_affinity"])
    assert np.all((graph["euclidean_affinity"] >= 0.0) & (graph["euclidean_affinity"] <= 1.0))


def test_diffusion_is_anchored_by_effective_evidence_count() -> None:
    table = _table()
    graph, _ = build_physical_graph(
        table, candidate_k=2, normal_affinity_power=2.0
    )
    initial = table["unary_posterior"].copy()
    output, report = diffuse_graph_probability(
        initial,
        source_gaussian_id=graph["source_gaussian_id"],
        target_gaussian_id=graph["target_gaussian_id"],
        edge_affinity=graph["euclidean_affinity"],
        effective_evidence_count=table["effective_evidence_count"],
        diffusion_rate=0.25,
        iterations=2,
        minimum_affinity_sum=1e-8,
    )
    np.testing.assert_array_equal(initial, table["unary_posterior"])
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert report["changed_node_count"] > 0
    assert abs(float(output[2] - initial[2])) < abs(float(output[0] - initial[0]))


def test_affinity_diagnostic_reports_proxy_crossing_without_feedback() -> None:
    table = _table()
    graph, _ = build_physical_graph(
        table, candidate_k=2, normal_affinity_power=2.0
    )
    report = graph_affinity_diagnostics(
        source_gaussian_id=graph["source_gaussian_id"],
        target_gaussian_id=graph["target_gaussian_id"],
        edge_affinity=graph["edge_affinity"],
        probability=table["unary_posterior"],
        proxy_target=table["base_model"] == "RigidNodes",
    )
    assert report["edge_count"] == 10
    assert 0.0 <= report["cross_proxy_affinity_ratio"] <= 1.0
    assert 0.0 <= report["weighted_neighbor_consistency"] <= 1.0

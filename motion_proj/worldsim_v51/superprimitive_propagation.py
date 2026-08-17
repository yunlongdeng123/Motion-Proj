"""E0b voxel node 的证据聚合、quotient topology 与 Gaussian readout。"""

from __future__ import annotations

from typing import Any

import numpy as np


class SuperPrimitivePropagationError(RuntimeError):
    """E0b 输入不满足冻结的 node-elevation 合同。"""


def _dense_node_ids(node_id: np.ndarray) -> tuple[np.ndarray, int]:
    node = np.asarray(node_id, dtype=np.int64)
    if node.ndim != 1 or node.size < 2 or np.any(node < 0):
        raise SuperPrimitivePropagationError("node_id must be a nonnegative N>=2 vector")
    node_count = int(node.max()) + 1
    if not np.array_equal(np.unique(node), np.arange(node_count, dtype=np.int64)):
        raise SuperPrimitivePropagationError("node ids must be dense from zero")
    return node, node_count


def quotient_directed_edges(
    raw_source_gaussian_id: np.ndarray,
    raw_target_gaussian_id: np.ndarray,
    node_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """把 frozen directed KNN 映射为去自环、去重的 directed quotient edges。"""

    node, node_count = _dense_node_ids(node_id)
    source = np.asarray(raw_source_gaussian_id, dtype=np.int64)
    target = np.asarray(raw_target_gaussian_id, dtype=np.int64)
    if source.ndim != 1 or target.shape != source.shape or source.size == 0:
        raise SuperPrimitivePropagationError("raw edge ids must be nonempty aligned vectors")
    if np.any(source < 0) or np.any(source >= node.size):
        raise SuperPrimitivePropagationError("raw source Gaussian id is out of range")
    if np.any(target < 0) or np.any(target >= node.size):
        raise SuperPrimitivePropagationError("raw target Gaussian id is out of range")

    quotient_source = node[source]
    quotient_target = node[target]
    retained = quotient_source != quotient_target
    encoded = quotient_source[retained] * np.int64(node_count) + quotient_target[retained]
    unique = np.unique(encoded)
    result_source = np.floor_divide(unique, np.int64(node_count)).astype(
        np.int64, copy=False
    )
    result_target = np.remainder(unique, np.int64(node_count)).astype(
        np.int64, copy=False
    )
    return result_source, result_target, {
        "raw_directed_edge_count": int(source.size),
        "intra_node_edge_count": int(np.count_nonzero(~retained)),
        "cross_node_edge_count_before_deduplication": int(np.count_nonzero(retained)),
        "quotient_directed_edge_count": int(unique.size),
        "duplicate_cross_node_edge_count": int(np.count_nonzero(retained) - unique.size),
    }


def aggregate_node_evidence(
    unary_posterior: np.ndarray,
    probability_by_view: np.ndarray,
    visibility_by_view: np.ndarray,
    node_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """按预注册规则将 Gaussian unary 与多视图 SAM evidence 聚合到 voxel node。"""

    node, node_count = _dense_node_ids(node_id)
    unary = np.asarray(unary_posterior, dtype=np.float32)
    probability = np.asarray(probability_by_view, dtype=np.float32)
    visibility = np.asarray(visibility_by_view, dtype=np.float32)
    if unary.shape != node.shape:
        raise SuperPrimitivePropagationError("unary posterior must align with node ids")
    if probability.ndim != 2 or probability.shape[0] != node.size:
        raise SuperPrimitivePropagationError("probability matrix must align with node ids")
    if visibility.shape != probability.shape:
        raise SuperPrimitivePropagationError("visibility matrix must align with probability")
    for value, label in (
        (unary, "unary"),
        (probability, "probability"),
        (visibility, "visibility"),
    ):
        if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise SuperPrimitivePropagationError(f"{label} must be finite within [0,1]")

    member_count = np.bincount(node, minlength=node_count).astype(np.int64)
    node_unary = (
        np.bincount(node, weights=unary, minlength=node_count) / member_count
    ).astype(np.float32)
    node_probability = np.full(
        (node_count, probability.shape[1]), 0.5, dtype=np.float32
    )
    node_visibility = np.zeros_like(node_probability)
    observed_node_views = 0
    for view_index in range(probability.shape[1]):
        view_visibility = visibility[:, view_index]
        visibility_sum = np.bincount(
            node, weights=view_visibility, minlength=node_count
        )
        weighted_probability_sum = np.bincount(
            node,
            weights=probability[:, view_index] * view_visibility,
            minlength=node_count,
        )
        observed = visibility_sum > 0.0
        node_probability[observed, view_index] = (
            weighted_probability_sum[observed] / visibility_sum[observed]
        ).astype(np.float32)
        np.maximum.at(node_visibility[:, view_index], node, view_visibility)
        observed_node_views += int(np.count_nonzero(observed))

    if np.any((node_probability < 0.0) | (node_probability > 1.0)) or np.any(
        (node_visibility < 0.0) | (node_visibility > 1.0)
    ):
        raise SuperPrimitivePropagationError("aggregated node evidence escaped [0,1]")
    return node_unary, node_probability, node_visibility, {
        "gaussian_count": int(node.size),
        "node_count": node_count,
        "view_count": int(probability.shape[1]),
        "member_count_minimum": int(member_count.min()),
        "member_count_maximum": int(member_count.max()),
        "singleton_node_count": int(np.count_nonzero(member_count == 1)),
        "observed_node_view_count": observed_node_views,
        "actor_seed_count": int(np.count_nonzero(node_unary >= 0.9)),
        "background_seed_count": int(np.count_nonzero(node_unary <= 0.1)),
        "unknown_seed_count": int(
            np.count_nonzero((node_unary > 0.1) & (node_unary < 0.9))
        ),
        "probability_aggregation": "member_visibility_weighted_mean",
        "visibility_aggregation": "maximum_valid_member_visibility",
        "unary_aggregation": "unweighted_member_arithmetic_mean",
    }


def broadcast_node_result(
    node_id: np.ndarray,
    node_label: np.ndarray,
    node_posterior: np.ndarray,
    node_assignment_level: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 node-level propagation 结果原样广播给全部 member Gaussian。"""

    node, node_count = _dense_node_ids(node_id)
    label = np.asarray(node_label)
    posterior = np.asarray(node_posterior)
    assignment = np.asarray(node_assignment_level)
    if label.shape != (node_count,) or posterior.shape != (node_count,) or assignment.shape != (
        node_count,
    ):
        raise SuperPrimitivePropagationError("node result arrays must align with node count")
    return label[node], posterior[node], assignment[node]

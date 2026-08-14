"""V5 M1 的 result-blind Gaussian physical graph 与稀疏扩散。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .evidence_schema import validate_edge_table, validate_gaussian_table


GRAPH_AFFINITY_NAMES = ("G1", "G2", "G3")


def _positive_median(value: np.ndarray, label: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    positive = array[np.isfinite(array) & (array > 0.0)]
    if positive.size == 0:
        raise ValueError(f"{label} 没有有限正值")
    return float(np.median(positive))


def build_physical_graph(
    gaussian_table: Mapping[str, np.ndarray],
    *,
    candidate_k: int,
    normal_affinity_power: float,
    workers: int = 1,
    chunk_size: int = 500_000,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    """先冻结 Euclidean KNN 候选，再计算 Gaussian-native affinity。

    ``base_model`` 只存在于输入 schema，不参与候选或 affinity，避免用
    Background/RigidNodes membership 泄漏 proxy target。
    """

    validate_gaussian_table(gaussian_table)
    centers = np.asarray(gaussian_table["center"], dtype=np.float64)
    covariance = np.asarray(gaussian_table["covariance"], dtype=np.float64)
    normals = np.asarray(gaussian_table["normal_proxy"], dtype=np.float64)
    normal_available = np.asarray(gaussian_table["normal_available"], dtype=bool)
    ambiguity = np.asarray(gaussian_table["boundary_ambiguity"], dtype=np.float64)
    motion = np.asarray(gaussian_table["motion_consistency"], dtype=np.float64)
    motion_available = np.asarray(
        gaussian_table["motion_consistency_available"], dtype=bool
    )
    count = centers.shape[0]
    if not 1 <= int(candidate_k) < count:
        raise ValueError("candidate_k 必须位于 [1, gaussian_count)")
    if not np.isfinite(normal_affinity_power) or normal_affinity_power <= 0.0:
        raise ValueError("normal_affinity_power 必须有限为正")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size 必须为正")

    tree = cKDTree(centers)
    distances, neighbors = tree.query(
        centers, k=int(candidate_k) + 1, workers=int(workers)
    )
    row_ids = np.arange(count, dtype=np.int64)
    non_self = neighbors != row_ids[:, None]
    if not np.all(non_self.sum(axis=1) == int(candidate_k)):
        raise ValueError("KNN self exclusion 分母漂移")
    targets = neighbors[non_self].reshape(count, int(candidate_k)).astype(
        np.int64, copy=False
    )
    euclidean_distance = distances[non_self].reshape(count, int(candidate_k))
    source = np.repeat(row_ids, int(candidate_k))
    target = targets.reshape(-1)
    euclidean_distance = euclidean_distance.reshape(-1).astype(np.float32)
    edge_count = source.size

    inverse_covariance = np.linalg.inv(covariance)
    mahalanobis = np.empty(edge_count, dtype=np.float32)
    normal_distance = np.empty(edge_count, dtype=np.float32)
    motion_distance = np.zeros(edge_count, dtype=np.float32)
    boundary_barrier = np.empty(edge_count, dtype=np.float32)
    for start in range(0, edge_count, int(chunk_size)):
        stop = min(start + int(chunk_size), edge_count)
        left = source[start:stop]
        right = target[start:stop]
        delta = centers[right] - centers[left]
        left_squared = np.einsum(
            "ni,nij,nj->n", delta, inverse_covariance[left], delta
        )
        right_squared = np.einsum(
            "ni,nij,nj->n", delta, inverse_covariance[right], delta
        )
        mahalanobis[start:stop] = np.sqrt(
            np.maximum(0.5 * (left_squared + right_squared), 0.0)
        ).astype(np.float32)
        dot = np.abs(np.sum(normals[left] * normals[right], axis=1))
        both_normal = normal_available[left] & normal_available[right]
        normal_distance[start:stop] = np.where(
            both_normal, 1.0 - np.clip(dot, 0.0, 1.0), 0.0
        ).astype(np.float32)
        both_motion = motion_available[left] & motion_available[right]
        motion_distance[start:stop] = np.where(
            both_motion, np.abs(motion[left] - motion[right]), 0.0
        ).astype(np.float32)
        boundary_barrier[start:stop] = np.clip(
            1.0 - np.maximum(ambiguity[left], ambiguity[right]), 0.0, 1.0
        ).astype(np.float32)

    euclidean_bandwidth = _positive_median(euclidean_distance, "Euclidean distance")
    mahalanobis_bandwidth = _positive_median(mahalanobis, "Mahalanobis distance")
    euclidean_affinity = np.exp(
        -0.5 * np.square(euclidean_distance / euclidean_bandwidth)
    ).astype(np.float32)
    gaussian_affinity = np.exp(
        -0.5 * np.square(mahalanobis / mahalanobis_bandwidth)
    ).astype(np.float32)
    normal_affinity = np.power(
        np.clip(1.0 - normal_distance, 0.0, 1.0), normal_affinity_power
    ).astype(np.float32)
    physical_affinity = (gaussian_affinity * normal_affinity).astype(np.float32)
    full_affinity = (physical_affinity * boundary_barrier).astype(np.float32)
    table = {
        "scene": np.asarray(str(np.asarray(gaussian_table["scene"]).item())),
        "role": np.asarray(str(np.asarray(gaussian_table["role"]).item())),
        "source_gaussian_id": source,
        "target_gaussian_id": target,
        "mahalanobis_distance": mahalanobis,
        "normal_distance": normal_distance,
        "motion_distance": motion_distance,
        "boundary_barrier": boundary_barrier,
        "edge_affinity": full_affinity,
        "euclidean_distance": euclidean_distance,
        "euclidean_affinity": euclidean_affinity,
        "physical_affinity": physical_affinity,
    }
    validate_edge_table(table, gaussian_count=count)
    metadata: dict[str, float | int | str] = {
        "candidate_policy": "directed_euclidean_knn_without_base_model",
        "candidate_k": int(candidate_k),
        "gaussian_count": int(count),
        "edge_count": int(edge_count),
        "euclidean_bandwidth": euclidean_bandwidth,
        "mahalanobis_bandwidth": mahalanobis_bandwidth,
        "normal_affinity_power": float(normal_affinity_power),
        "boundary_barrier_policy": "one_minus_max_endpoint_boundary_ambiguity",
    }
    return table, metadata


def diffuse_graph_probability(
    probability: np.ndarray,
    *,
    source_gaussian_id: np.ndarray,
    target_gaussian_id: np.ndarray,
    edge_affinity: np.ndarray,
    effective_evidence_count: np.ndarray,
    diffusion_rate: float,
    iterations: int,
    minimum_affinity_sum: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """以 evidence count 锚定 unary，执行固定步数的非训练稀疏扩散。"""

    initial = np.asarray(probability, dtype=np.float64)
    source = np.asarray(source_gaussian_id, dtype=np.int64)
    target = np.asarray(target_gaussian_id, dtype=np.int64)
    affinity = np.asarray(edge_affinity, dtype=np.float64)
    evidence = np.asarray(effective_evidence_count, dtype=np.float64)
    count = initial.size
    if initial.ndim != 1 or evidence.shape != (count,):
        raise ValueError("probability/effective_evidence_count shape 漂移")
    if source.shape != target.shape or affinity.shape != source.shape:
        raise ValueError("graph edge shape 漂移")
    if np.any(source < 0) or np.any(source >= count) or np.any(target < 0) or np.any(target >= count):
        raise ValueError("graph edge 索引越界")
    if not np.isfinite(initial).all() or np.any((initial < 0.0) | (initial > 1.0)):
        raise ValueError("probability 必须为有限 [0,1]")
    if not np.isfinite(evidence).all() or np.any(evidence < 0.0):
        raise ValueError("effective_evidence_count 必须有限非负")
    if not np.isfinite(affinity).all() or np.any((affinity < 0.0) | (affinity > 1.0)):
        raise ValueError("edge_affinity 必须为有限 [0,1]")
    if not 0.0 <= float(diffusion_rate) <= 1.0:
        raise ValueError("diffusion_rate 必须位于 [0,1]")
    if int(iterations) < 0 or float(minimum_affinity_sum) <= 0.0:
        raise ValueError("iterations/minimum_affinity_sum 非法")

    update_strength = float(diffusion_rate) / (1.0 + evidence)
    current = initial.copy()
    active_nodes = 0
    for _ in range(int(iterations)):
        weight_sum = np.bincount(source, weights=affinity, minlength=count)
        weighted_probability = np.bincount(
            source, weights=affinity * current[target], minlength=count
        )
        active = weight_sum >= float(minimum_affinity_sum)
        active_nodes = int(active.sum())
        neighbor = current.copy()
        neighbor[active] = weighted_probability[active] / weight_sum[active]
        current += update_strength * (neighbor - current)
        np.clip(current, 0.0, 1.0, out=current)
    movement = np.abs(current - initial)
    return current.astype(np.float32), {
        "iterations": int(iterations),
        "active_node_count": active_nodes,
        "mean_absolute_update": float(movement.mean()),
        "maximum_absolute_update": float(movement.max(initial=0.0)),
        "changed_node_count": int(np.count_nonzero(movement > 1e-8)),
    }


def graph_affinity_diagnostics(
    *,
    source_gaussian_id: np.ndarray,
    target_gaussian_id: np.ndarray,
    edge_affinity: np.ndarray,
    probability: np.ndarray,
    proxy_target: np.ndarray,
) -> dict[str, float | int]:
    """报告 affinity leakage；proxy target 只用于审计，不回流 graph。"""

    source = np.asarray(source_gaussian_id, dtype=np.int64)
    target = np.asarray(target_gaussian_id, dtype=np.int64)
    affinity = np.asarray(edge_affinity, dtype=np.float64)
    posterior = np.asarray(probability, dtype=np.float64)
    proxy = np.asarray(proxy_target, dtype=bool)
    total = float(affinity.sum())
    if total <= 0.0:
        raise ValueError("graph affinity 总质量必须为正")
    cross = proxy[source] != proxy[target]
    difference = np.abs(posterior[source] - posterior[target])
    return {
        "edge_count": int(source.size),
        "affinity_mass": total,
        "cross_proxy_affinity_ratio": float(affinity[cross].sum() / total),
        "weighted_mean_posterior_difference": float(
            np.sum(affinity * difference) / total
        ),
        "weighted_neighbor_consistency": float(
            1.0 - np.sum(affinity * difference) / total
        ),
    }

"""Clean-room SAI3D-style progressive propagation for V5.1 D0.

The implementation follows the equations and mechanism frozen in the D0
preregistration.  It does not import or copy code from the upstream SAI3D
repository.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


UNKNOWN = np.int8(-1)
BACKGROUND = np.int8(0)
ACTOR = np.int8(1)
SEED_LEVEL = np.int8(-2)
DEFAULT_THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5)


class ProgressivePropagationError(RuntimeError):
    """The frozen D0 input or propagation contract is invalid."""


def _finite_probability(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ProgressivePropagationError(f"{label} must be finite and within [0, 1]")
    return array


def build_exact_logical_topologies(
    source_node_id: np.ndarray,
    target_node_id: np.ndarray,
    *,
    node_count: int,
    maximum_logical_distance: int = 2,
) -> tuple[sparse.csr_matrix, ...]:
    """Build symmetric exact-distance sparse topology from frozen directed KNN.

    Output element zero contains direct neighbors.  Element one, when enabled,
    contains nodes at exactly two graph hops and excludes self/direct neighbors.
    All matrices are binary float32 CSR with sorted indices.
    """

    count = int(node_count)
    maximum_distance = int(maximum_logical_distance)
    source = np.asarray(source_node_id, dtype=np.int64)
    target = np.asarray(target_node_id, dtype=np.int64)
    if count <= 1:
        raise ProgressivePropagationError("node_count must be greater than one")
    if source.ndim != 1 or target.shape != source.shape or source.size == 0:
        raise ProgressivePropagationError("edge source/target must be nonempty aligned vectors")
    if np.any(source < 0) or np.any(source >= count):
        raise ProgressivePropagationError("source node id is out of range")
    if np.any(target < 0) or np.any(target >= count):
        raise ProgressivePropagationError("target node id is out of range")
    if maximum_distance not in (1, 2):
        raise ProgressivePropagationError("D0 supports maximum logical distance one or two")

    direct = sparse.csr_matrix(
        (np.ones(source.size, dtype=np.float32), (source, target)),
        shape=(count, count),
        dtype=np.float32,
    )
    direct.sum_duplicates()
    direct.data.fill(1.0)
    direct = direct.maximum(direct.transpose()).tocsr()
    direct.setdiag(0.0)
    direct.eliminate_zeros()
    direct.sort_indices()
    if direct.nnz == 0:
        raise ProgressivePropagationError("symmetrized direct topology is empty")
    if maximum_distance == 1:
        return (direct,)

    second = (direct @ direct).tocsr()
    second.data.fill(1.0)
    second.setdiag(0.0)
    second.eliminate_zeros()
    second = (second - second.multiply(direct)).tocsr()
    second.eliminate_zeros()
    if second.nnz:
        second.data.fill(1.0)
    second.sort_indices()
    return direct, second


def soft_binary_cosine_affinity(
    probability_by_view: np.ndarray,
    visibility_by_view: np.ndarray,
    source_node_id: np.ndarray,
    target_node_id: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Compute SAI3D-style visibility-weighted multi-view label similarity.

    Each retained binary SAM probability is represented by the L2-normalized
    vector ``[1-p, p]``.  Pair similarity is cosine, and its cross-view mean is
    weighted by the product of endpoint visibility.  A pair without a jointly
    valid view receives zero affinity and remains a geometric neighbor only.
    """

    probability = _finite_probability(probability_by_view, "probability_by_view")
    visibility = _finite_probability(visibility_by_view, "visibility_by_view")
    source = np.asarray(source_node_id, dtype=np.int64)
    target = np.asarray(target_node_id, dtype=np.int64)
    if probability.ndim != 2 or visibility.shape != probability.shape:
        raise ProgressivePropagationError("probability/visibility view matrices must align")
    if source.ndim != 1 or target.shape != source.shape:
        raise ProgressivePropagationError("affinity edge vectors must align")
    if source.size == 0:
        return np.empty(0, dtype=np.float32), {
            "pair_count": 0,
            "jointly_visible_pair_count": 0,
            "mean_common_view_count": 0.0,
            "mean_affinity": 0.0,
        }
    count = probability.shape[0]
    if np.any(source < 0) or np.any(source >= count) or np.any(target < 0) or np.any(target >= count):
        raise ProgressivePropagationError("affinity node id is out of range")
    if int(chunk_size) <= 0:
        raise ProgressivePropagationError("chunk_size must be positive")

    result = np.zeros(source.size, dtype=np.float32)
    jointly_visible = 0
    common_view_total = 0
    affinity_total = 0.0
    for start in range(0, source.size, int(chunk_size)):
        stop = min(start + int(chunk_size), source.size)
        left = probability[source[start:stop]]
        right = probability[target[start:stop]]
        left_visibility = visibility[source[start:stop]]
        right_visibility = visibility[target[start:stop]]
        confidence = left_visibility * right_visibility
        common = confidence > 0.0

        left_norm = np.sqrt(np.square(1.0 - left) + np.square(left))
        right_norm = np.sqrt(np.square(1.0 - right) + np.square(right))
        similarity = (
            (1.0 - left) * (1.0 - right) + left * right
        ) / (left_norm * right_norm)
        np.clip(similarity, 0.0, 1.0, out=similarity)
        numerator = np.sum(similarity * confidence, axis=1, dtype=np.float64)
        denominator = np.sum(confidence, axis=1, dtype=np.float64)
        chunk = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        ).astype(np.float32)
        result[start:stop] = chunk
        common_count = np.count_nonzero(common, axis=1)
        jointly_visible += int(np.count_nonzero(denominator > 0.0))
        common_view_total += int(common_count.sum(dtype=np.int64))
        affinity_total += float(chunk.sum(dtype=np.float64))

    return result, {
        "pair_count": int(source.size),
        "jointly_visible_pair_count": jointly_visible,
        "mean_common_view_count": float(common_view_total / source.size),
        "mean_affinity": float(affinity_total / source.size),
    }


def affinity_matrices_from_topologies(
    topologies: Sequence[sparse.csr_matrix],
    probability_by_view: np.ndarray,
    visibility_by_view: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> tuple[tuple[sparse.csr_matrix, ...], list[dict[str, float | int]]]:
    """Evaluate multi-view affinity on every exact logical-distance pair."""

    probability = np.asarray(probability_by_view)
    matrices: list[sparse.csr_matrix] = []
    reports: list[dict[str, float | int]] = []
    for distance, topology in enumerate(topologies, start=1):
        matrix = topology.tocsr(copy=False)
        if matrix.shape != (probability.shape[0], probability.shape[0]):
            raise ProgressivePropagationError("topology/probability node count drift")
        row = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr)
        )
        affinity, report = soft_binary_cosine_affinity(
            probability_by_view,
            visibility_by_view,
            row,
            matrix.indices,
            chunk_size=chunk_size,
        )
        result = sparse.csr_matrix(
            (affinity, matrix.indices.copy(), matrix.indptr.copy()),
            shape=matrix.shape,
            dtype=np.float32,
        )
        result.sort_indices()
        report = dict(report)
        report["logical_distance"] = distance
        matrices.append(result)
        reports.append(report)
    return tuple(matrices), reports


def _validate_sparse_levels(
    topologies: Sequence[sparse.csr_matrix],
    affinities: Sequence[sparse.csr_matrix],
    node_count: int,
) -> tuple[tuple[sparse.csr_matrix, ...], tuple[sparse.csr_matrix, ...]]:
    if not topologies or len(topologies) != len(affinities):
        raise ProgressivePropagationError("topology/affinity levels must be nonempty and aligned")
    checked_topologies = []
    checked_affinities = []
    for topology, affinity in zip(topologies, affinities):
        topology = topology.tocsr(copy=False)
        affinity = affinity.tocsr(copy=False)
        if topology.shape != (node_count, node_count) or affinity.shape != topology.shape:
            raise ProgressivePropagationError("sparse propagation shape drift")
        if not np.array_equal(topology.indptr, affinity.indptr) or not np.array_equal(
            topology.indices, affinity.indices
        ):
            raise ProgressivePropagationError("topology/affinity sparse structure drift")
        if not np.isfinite(affinity.data).all() or np.any(
            (affinity.data < 0.0) | (affinity.data > 1.0)
        ):
            raise ProgressivePropagationError("sparse affinity must be within [0, 1]")
        checked_topologies.append(topology)
        checked_affinities.append(affinity)
    return tuple(checked_topologies), tuple(checked_affinities)


def progressive_region_growing(
    unary_posterior: np.ndarray,
    *,
    topologies: Sequence[sparse.csr_matrix],
    affinities: Sequence[sparse.csr_matrix],
    progressive_thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    logical_distance_decay: float = 0.5,
    actor_seed_minimum: float = 0.9,
    background_seed_maximum: float = 0.1,
    unknown_probability: float = 0.5,
    maximum_rounds_per_threshold: int | None = None,
) -> dict[str, object]:
    """Grow immutable high-confidence binary seeds with dynamic region affinity."""

    unary = _finite_probability(unary_posterior, "unary_posterior")
    if unary.ndim != 1:
        raise ProgressivePropagationError("unary_posterior must be one dimensional")
    topology_levels, affinity_levels = _validate_sparse_levels(
        topologies, affinities, unary.size
    )
    thresholds = tuple(float(value) for value in progressive_thresholds)
    if not thresholds or not all(np.isfinite(thresholds)):
        raise ProgressivePropagationError("progressive thresholds must be finite")
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ProgressivePropagationError("progressive thresholds must be within [0, 1]")
    if any(left <= right for left, right in zip(thresholds, thresholds[1:])):
        raise ProgressivePropagationError("progressive thresholds must be strictly decreasing")
    if not 0.0 < float(logical_distance_decay) <= 1.0:
        raise ProgressivePropagationError("logical_distance_decay must be within (0, 1]")
    if not 0.5 < float(actor_seed_minimum) <= 1.0:
        raise ProgressivePropagationError("actor seed minimum must be within (0.5, 1]")
    if not 0.0 <= float(background_seed_maximum) < 0.5:
        raise ProgressivePropagationError("background seed maximum must be within [0, 0.5)")
    if not 0.0 <= float(unknown_probability) <= 1.0:
        raise ProgressivePropagationError("unknown probability must be within [0, 1]")
    if maximum_rounds_per_threshold is not None and int(maximum_rounds_per_threshold) <= 0:
        raise ProgressivePropagationError("maximum rounds must be positive when specified")

    labels = np.full(unary.size, UNKNOWN, dtype=np.int8)
    labels[unary <= float(background_seed_maximum)] = BACKGROUND
    labels[unary >= float(actor_seed_minimum)] = ACTOR
    assignment_level = np.full(unary.size, UNKNOWN, dtype=np.int8)
    assignment_level[labels != UNKNOWN] = SEED_LEVEL
    background_seed_count = int(np.count_nonzero(labels == BACKGROUND))
    actor_seed_count = int(np.count_nonzero(labels == ACTOR))
    stage_reports: list[dict[str, float | int]] = []

    for level, threshold in enumerate(thresholds):
        background_assigned = 0
        actor_assigned = 0
        rounds = 0
        while True:
            if maximum_rounds_per_threshold is not None and rounds >= int(
                maximum_rounds_per_threshold
            ):
                raise ProgressivePropagationError(
                    f"progressive level {level} did not reach a fixed point"
                )
            members = np.stack(
                (labels == BACKGROUND, labels == ACTOR), axis=1
            ).astype(np.float32)
            numerator = np.zeros((unary.size, 2), dtype=np.float32)
            denominator = np.zeros((unary.size, 2), dtype=np.float32)
            for distance, (topology, affinity) in enumerate(
                zip(topology_levels, affinity_levels), start=1
            ):
                weight = np.float32(float(logical_distance_decay) ** (distance - 1))
                numerator += weight * (affinity @ members)
                denominator += weight * (topology @ members)
            scores = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0.0,
            )
            unknown = labels == UNKNOWN
            background_pass = unknown & (scores[:, 0] >= threshold)
            actor_pass = unknown & (scores[:, 1] >= threshold)
            choose_background = background_pass & (
                ~actor_pass | (scores[:, 0] > scores[:, 1])
            )
            choose_actor = actor_pass & (
                ~background_pass | (scores[:, 1] > scores[:, 0])
            )
            newly_assigned = choose_background | choose_actor
            if not np.any(newly_assigned):
                break
            labels[choose_background] = BACKGROUND
            labels[choose_actor] = ACTOR
            assignment_level[newly_assigned] = np.int8(level)
            background_assigned += int(np.count_nonzero(choose_background))
            actor_assigned += int(np.count_nonzero(choose_actor))
            rounds += 1
        stage_reports.append(
            {
                "level": level,
                "threshold": threshold,
                "round_count": rounds,
                "background_assignment_count": background_assigned,
                "actor_assignment_count": actor_assigned,
                "remaining_unknown_count": int(np.count_nonzero(labels == UNKNOWN)),
            }
        )

    posterior = np.full(unary.size, float(unknown_probability), dtype=np.float32)
    posterior[labels == BACKGROUND] = 0.0
    posterior[labels == ACTOR] = 1.0
    return {
        "labels": labels,
        "posterior": posterior,
        "assignment_level": assignment_level,
        "report": {
            "node_count": int(unary.size),
            "background_seed_count": background_seed_count,
            "actor_seed_count": actor_seed_count,
            "initial_unknown_count": int(
                unary.size - background_seed_count - actor_seed_count
            ),
            "final_background_count": int(np.count_nonzero(labels == BACKGROUND)),
            "final_actor_count": int(np.count_nonzero(labels == ACTOR)),
            "final_unknown_count": int(np.count_nonzero(labels == UNKNOWN)),
            "final_unknown_ratio": float(np.mean(labels == UNKNOWN)),
            "logical_distance_count": len(topology_levels),
            "logical_distance_decay": float(logical_distance_decay),
            "progressive_thresholds": list(thresholds),
            "stage_reports": stage_reports,
            "global_one_shot_smoothing": False,
            "parameter_search": False,
        },
    }

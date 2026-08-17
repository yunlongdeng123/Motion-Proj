"""Deterministic E0a voxel super-primitive structural probe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class SuperPrimitiveProbeError(RuntimeError):
    """The frozen E0a structural probe contract is invalid."""


def _centers(value: np.ndarray) -> np.ndarray:
    centers = np.asarray(value, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 2:
        raise SuperPrimitiveProbeError("centers must have shape [N,3] with N>=2")
    if not np.isfinite(centers).all():
        raise SuperPrimitiveProbeError("centers must be finite")
    return centers


def edge_length_quantile_voxel_sizes(
    centers: np.ndarray,
    source_node_id: np.ndarray,
    target_node_id: np.ndarray,
    quantiles: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive scene-adaptive voxel sizes only from frozen KNN edge lengths."""

    xyz = _centers(centers)
    source = np.asarray(source_node_id, dtype=np.int64)
    target = np.asarray(target_node_id, dtype=np.int64)
    q = np.asarray(tuple(float(value) for value in quantiles), dtype=np.float64)
    if source.ndim != 1 or target.shape != source.shape or source.size == 0:
        raise SuperPrimitiveProbeError("edge ids must be nonempty aligned vectors")
    if np.any(source < 0) or np.any(source >= xyz.shape[0]):
        raise SuperPrimitiveProbeError("source id is out of range")
    if np.any(target < 0) or np.any(target >= xyz.shape[0]):
        raise SuperPrimitiveProbeError("target id is out of range")
    if q.ndim != 1 or q.size == 0 or not np.isfinite(q).all():
        raise SuperPrimitiveProbeError("quantiles must be finite and nonempty")
    if np.any((q <= 0.0) | (q >= 1.0)) or np.any(q[1:] <= q[:-1]):
        raise SuperPrimitiveProbeError("quantiles must be strictly increasing within (0,1)")

    edge_lengths = np.linalg.norm(xyz[source] - xyz[target], axis=1).astype(np.float32)
    positive = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
    if positive.size != edge_lengths.size:
        raise SuperPrimitiveProbeError("frozen KNN contains nonpositive/nonfinite edge length")
    sizes = np.quantile(positive, q, method="linear").astype(np.float64)
    if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0.0) or np.any(sizes[1:] <= sizes[:-1]):
        raise SuperPrimitiveProbeError("edge quantiles do not define distinct positive levels")
    return sizes, {
        "edge_count": int(edge_lengths.size),
        "minimum_edge_length_m": float(positive.min()),
        "maximum_edge_length_m": float(positive.max()),
        "mean_edge_length_m": float(positive.mean(dtype=np.float64)),
        "quantiles": [float(value) for value in q],
        "voxel_sizes_m": [float(value) for value in sizes],
    }


def voxel_assignments(
    centers: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign Gaussians to a world-origin voxel grid in lexicographic order."""

    xyz = _centers(centers)
    size = float(voxel_size_m)
    if not np.isfinite(size) or size <= 0.0:
        raise SuperPrimitiveProbeError("voxel_size_m must be positive and finite")
    scaled = np.floor(xyz / size)
    limits = np.iinfo(np.int64)
    if np.any(scaled < limits.min) or np.any(scaled > limits.max):
        raise SuperPrimitiveProbeError("voxel coordinate exceeds int64")
    keys = scaled.astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    assignment = inverse.astype(np.int64, copy=False)
    if unique_keys.shape[0] < 1 or assignment.shape != (xyz.shape[0],):
        raise SuperPrimitiveProbeError("voxel assignment denominator drift")
    return assignment, unique_keys


def observation_density_report(
    assignment: np.ndarray,
    visibility_by_view: np.ndarray,
    unary_posterior: np.ndarray,
    *,
    actor_seed_minimum: float = 0.9,
    background_seed_maximum: float = 0.1,
) -> dict[str, Any]:
    """Measure observation union and seed mixing without reading quality labels."""

    node_id = np.asarray(assignment, dtype=np.int64)
    visibility = np.asarray(visibility_by_view, dtype=np.float32)
    unary = np.asarray(unary_posterior, dtype=np.float32)
    if node_id.ndim != 1 or node_id.size < 2:
        raise SuperPrimitiveProbeError("assignment must be a one-dimensional N>=2 vector")
    if visibility.ndim != 2 or visibility.shape[0] != node_id.size:
        raise SuperPrimitiveProbeError("visibility matrix must align with assignments")
    if unary.shape != node_id.shape:
        raise SuperPrimitiveProbeError("unary posterior must align with assignments")
    if np.any(node_id < 0) or not np.isfinite(visibility).all() or not np.isfinite(unary).all():
        raise SuperPrimitiveProbeError("assignment/visibility/unary contains invalid values")
    if np.any((visibility < 0.0) | (visibility > 1.0)):
        raise SuperPrimitiveProbeError("visibility must be within [0,1]")
    if np.any((unary < 0.0) | (unary > 1.0)):
        raise SuperPrimitiveProbeError("unary posterior must be within [0,1]")
    unique = np.unique(node_id)
    if not np.array_equal(unique, np.arange(unique.size, dtype=np.int64)):
        raise SuperPrimitiveProbeError("node ids must be dense from zero")

    node_count = int(unique.size)
    member_count = np.bincount(node_id, minlength=node_count).astype(np.int64)
    observed = visibility > 0.0
    raw_view_count = np.count_nonzero(observed, axis=1).astype(np.int64)
    node_view_count = np.zeros(node_count, dtype=np.int64)
    for view in range(observed.shape[1]):
        present = np.bincount(
            node_id, weights=observed[:, view].astype(np.int8), minlength=node_count
        )
        node_view_count += present > 0
    assigned_union_view_count = node_view_count[node_id]
    raw_zero = raw_view_count == 0
    rescued = raw_zero & (assigned_union_view_count > 0)

    actor_seed = unary >= float(actor_seed_minimum)
    background_seed = unary <= float(background_seed_maximum)
    actor_nodes = (
        np.bincount(node_id, weights=actor_seed.astype(np.int8), minlength=node_count) > 0
    )
    background_nodes = (
        np.bincount(node_id, weights=background_seed.astype(np.int8), minlength=node_count)
        > 0
    )
    conflict_nodes = actor_nodes & background_nodes
    conflict_gaussian_count = int(member_count[conflict_nodes].sum(dtype=np.int64))

    posterior_sum = np.bincount(node_id, weights=unary, minlength=node_count)
    posterior_square_sum = np.bincount(
        node_id, weights=np.square(unary, dtype=np.float32), minlength=node_count
    )
    posterior_mean = posterior_sum / member_count
    posterior_variance = np.maximum(
        posterior_square_sum / member_count - np.square(posterior_mean), 0.0
    )
    raw_zero_count = int(np.count_nonzero(raw_zero))
    return {
        "gaussian_count": int(node_id.size),
        "node_count": node_count,
        "node_reduction_ratio": float(1.0 - node_count / node_id.size),
        "singleton_node_fraction": float(np.mean(member_count == 1, dtype=np.float64)),
        "member_count_percentiles": {
            str(percentile): float(np.percentile(member_count, percentile))
            for percentile in (0, 50, 90, 99, 100)
        },
        "raw_mean_observed_views_per_gaussian": float(
            raw_view_count.mean(dtype=np.float64)
        ),
        "node_mean_union_observed_views": float(node_view_count.mean(dtype=np.float64)),
        "gaussian_weighted_node_union_observed_views": float(
            assigned_union_view_count.mean(dtype=np.float64)
        ),
        "observation_union_gain_per_gaussian": float(
            (assigned_union_view_count - raw_view_count).mean(dtype=np.float64)
        ),
        "raw_zero_observation_gaussian_count": raw_zero_count,
        "rescued_zero_observation_gaussian_count": int(np.count_nonzero(rescued)),
        "rescued_zero_observation_gaussian_fraction": float(
            np.count_nonzero(rescued) / raw_zero_count if raw_zero_count else 0.0
        ),
        "actor_seed_node_count": int(np.count_nonzero(actor_nodes)),
        "background_seed_node_count": int(np.count_nonzero(background_nodes)),
        "conflicting_seed_node_count": int(np.count_nonzero(conflict_nodes)),
        "conflicting_seed_gaussian_count": conflict_gaussian_count,
        "conflicting_seed_gaussian_fraction": float(conflict_gaussian_count / node_id.size),
        "gaussian_weighted_within_node_unary_variance": float(
            posterior_variance[node_id].mean(dtype=np.float64)
        ),
    }


def evaluate_e0a_density_gate(
    scenes: Sequence[Mapping[str, Any]],
    *,
    expected_scene_count: int,
) -> dict[str, Any]:
    """Pass only when at least one preregistered level strictly helps every scene."""

    if len(scenes) != int(expected_scene_count):
        raise SuperPrimitiveProbeError("E0a scene denominator drift")
    scene_checks = []
    for scene in scenes:
        raw = scene["raw"]
        passing_levels = []
        level_checks = []
        for level in scene["levels"]:
            checks = {
                "node_count_reduced": int(level["node_count"]) < int(raw["node_count"]),
                "observation_union_strictly_improved": float(
                    level["gaussian_weighted_node_union_observed_views"]
                )
                > float(raw["raw_mean_observed_views_per_gaussian"]),
                "unobserved_gaussians_strictly_rescued": int(
                    level["rescued_zero_observation_gaussian_count"]
                )
                > 0,
            }
            passed = bool(all(checks.values()))
            if passed:
                passing_levels.append(level["level"])
            level_checks.append(
                {"level": level["level"], "pass": passed, "checks": checks}
            )
        scene_checks.append(
            {
                "scene": scene["scene"],
                "pass": bool(passing_levels),
                "passing_levels": passing_levels,
                "levels": level_checks,
            }
        )
    return {
        "pass": bool(all(scene["pass"] for scene in scene_checks)),
        "scene_count": len(scene_checks),
        "passing_scene_count": sum(scene["pass"] for scene in scene_checks),
        "scenes": scene_checks,
    }

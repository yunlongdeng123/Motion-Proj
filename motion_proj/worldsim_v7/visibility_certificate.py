"""Visibility-conditioned LiDAR certificates for WorldSim V7.

The evaluator keeps three physically distinct states separate: a surface point can
be supported by an observed return, contradicted by observed free space, or remain
unknown because it is occluded/off-ray.  Unknown evidence is never relabelled as
free or occupied.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from motion_proj.worldsim_v7.av2_four_action_compiler import _nearest


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _fscore(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / max(precision + recall, 1.0e-12))


def visibility_evidence(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Partition target rays and surface points into observable physical states."""

    chunk = int(config["distance_chunk_size"])
    lateral_tolerance = float(config["ray_lateral_tolerance_m"])
    depth_tolerance = float(config["depth_tolerance_m"])

    _, target_nearest = _nearest(target, surface, device, chunk)
    nearest_surface = surface[target_nearest].astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    origins64 = origins.astype(np.float64, copy=False)
    target_vectors = target64 - origins64
    target_depth = np.linalg.norm(target_vectors, axis=1)
    target_directions = target_vectors / np.maximum(target_depth[:, None], 1.0e-9)
    predicted_vectors = nearest_surface - origins64
    predicted_depth = np.sum(predicted_vectors * target_directions, axis=1)
    target_lateral = np.linalg.norm(
        predicted_vectors - predicted_depth[:, None] * target_directions,
        axis=1,
    )
    target_observable = (predicted_depth > 0.0) & (
        target_lateral <= lateral_tolerance
    )
    target_early = target_observable & (
        predicted_depth < target_depth - depth_tolerance
    )
    target_hit = target_observable & (
        np.abs(predicted_depth - target_depth) <= depth_tolerance
    )
    target_late = target_observable & (
        predicted_depth > target_depth + depth_tolerance
    )
    target_unmatched = ~target_observable

    _, surface_nearest = _nearest(surface, target, device, chunk)
    nearest_target = target64[surface_nearest]
    nearest_origins = origins64[surface_nearest]
    nearest_vectors = nearest_target - nearest_origins
    nearest_depth = np.linalg.norm(nearest_vectors, axis=1)
    nearest_directions = nearest_vectors / np.maximum(
        nearest_depth[:, None], 1.0e-9
    )
    surface_vectors = surface.astype(np.float64, copy=False) - nearest_origins
    surface_depth = np.sum(surface_vectors * nearest_directions, axis=1)
    surface_lateral = np.linalg.norm(
        surface_vectors - surface_depth[:, None] * nearest_directions,
        axis=1,
    )
    surface_observable = (surface_depth > 0.0) & (
        surface_lateral <= lateral_tolerance
    )
    surface_contradicted = surface_observable & (
        surface_depth < nearest_depth - depth_tolerance
    )
    surface_supported = surface_observable & (
        np.abs(surface_depth - nearest_depth) <= depth_tolerance
    )
    surface_occluded = surface_observable & (
        surface_depth > nearest_depth + depth_tolerance
    )
    surface_unknown = ~surface_observable

    target_counts = {
        "early": int(np.count_nonzero(target_early)),
        "hit": int(np.count_nonzero(target_hit)),
        "late": int(np.count_nonzero(target_late)),
        "unmatched": int(np.count_nonzero(target_unmatched)),
        "total": int(len(target)),
    }
    surface_counts = {
        "contradicted": int(np.count_nonzero(surface_contradicted)),
        "supported": int(np.count_nonzero(surface_supported)),
        "occluded": int(np.count_nonzero(surface_occluded)),
        "unknown": int(np.count_nonzero(surface_unknown)),
        "total": int(len(surface)),
    }
    falsifiable = surface_counts["supported"] + surface_counts["contradicted"]
    precision = _safe_fraction(surface_counts["supported"], falsifiable)
    recall = _safe_fraction(target_counts["hit"], target_counts["total"])
    return {
        "target_ray_counts": target_counts,
        "surface_counts": surface_counts,
        "target_hit_recall": recall,
        "target_early_termination_rate": _safe_fraction(
            target_counts["early"], target_counts["total"]
        ),
        "visible_surface_precision": precision,
        "visible_surface_falsifiable_fraction": _safe_fraction(
            falsifiable, surface_counts["total"]
        ),
        "surface_occluded_fraction": _safe_fraction(
            surface_counts["occluded"], surface_counts["total"]
        ),
        "surface_unknown_fraction": _safe_fraction(
            surface_counts["unknown"], surface_counts["total"]
        ),
        "visibility_fscore": _fscore(precision, recall),
        "partition_complete": (
            sum(target_counts[key] for key in ("early", "hit", "late", "unmatched"))
            == target_counts["total"]
            and sum(
                surface_counts[key]
                for key in ("contradicted", "supported", "occluded", "unknown")
            )
            == surface_counts["total"]
        ),
    }


def compile_actor_certificate(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    target = diagnostics["target"]
    origins = diagnostics["target_sensor_origins"]
    query = visibility_evidence(diagnostics["query"], target, origins, config, device)
    baseline = visibility_evidence(
        diagnostics["baseline"], target, origins, config, device
    )
    compiled = visibility_evidence(
        diagnostics["compiled"], target, origins, config, device
    )
    query_early = int(query["target_ray_counts"]["early"])
    compiled_early = int(compiled["target_ray_counts"]["early"])
    query_contradicted = int(query["surface_counts"]["contradicted"])
    compiled_contradicted = int(compiled["surface_counts"]["contradicted"])
    chamfer_query = float(actor_row["query_only"]["symmetric_chamfer_m"])
    chamfer_compiled = float(actor_row["after"]["symmetric_chamfer_m"])
    return {
        "track_id": str(actor_row["track_id"]),
        "category": str(actor_row["category"]),
        "hazardous": bool(actor_row["hazardous"]),
        "query_only": query,
        "paired_baseline": baseline,
        "compiled": compiled,
        "added_target_early_count_vs_query": compiled_early - query_early,
        "added_surface_contradiction_count_vs_query": (
            compiled_contradicted - query_contradicted
        ),
        "nonnew_visible_violation": (
            compiled_early <= query_early
            and compiled_contradicted <= query_contradicted
        ),
        "exact_no_visible_contradiction": (
            compiled_early == 0 and compiled_contradicted == 0
        ),
        "target_hit_noninferior_to_query": (
            float(compiled["target_hit_recall"]) >= float(query["target_hit_recall"])
        ),
        "visibility_fscore_noninferior_to_query": (
            float(compiled["visibility_fscore"])
            >= float(query["visibility_fscore"])
        ),
        "query_chamfer_m": chamfer_query,
        "compiled_chamfer_m": chamfer_compiled,
        "chamfer_worsened_vs_query": chamfer_compiled > chamfer_query,
        "actor_identity_trajectory_hazard_retention": 1.0,
    }


def summarize_certificates(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("visibility certificate has no Actor rows")

    surfaces = ("query_only", "paired_baseline", "compiled")
    aggregate: dict[str, Any] = {}
    for label in surfaces:
        target_counts = {
            key: sum(int(row[label]["target_ray_counts"][key]) for row in rows)
            for key in ("early", "hit", "late", "unmatched", "total")
        }
        surface_counts = {
            key: sum(int(row[label]["surface_counts"][key]) for row in rows)
            for key in ("contradicted", "supported", "occluded", "unknown", "total")
        }
        falsifiable = surface_counts["supported"] + surface_counts["contradicted"]
        precision = _safe_fraction(surface_counts["supported"], falsifiable)
        recall = _safe_fraction(target_counts["hit"], target_counts["total"])
        aggregate[label] = {
            "target_ray_counts": target_counts,
            "surface_counts": surface_counts,
            "target_hit_recall": recall,
            "target_early_termination_rate": _safe_fraction(
                target_counts["early"], target_counts["total"]
            ),
            "visible_surface_precision": precision,
            "visible_surface_falsifiable_fraction": _safe_fraction(
                falsifiable, surface_counts["total"]
            ),
            "surface_occluded_fraction": _safe_fraction(
                surface_counts["occluded"], surface_counts["total"]
            ),
            "surface_unknown_fraction": _safe_fraction(
                surface_counts["unknown"], surface_counts["total"]
            ),
            "visibility_fscore": _fscore(precision, recall),
        }

    chamfer_worsened = [row for row in rows if bool(row["chamfer_worsened_vs_query"])]

    def rate(key: str, selected: list[Mapping[str, Any]] = rows) -> float:
        return _safe_fraction(sum(bool(row[key]) for row in selected), len(selected))

    return {
        "actor_count": len(rows),
        "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in rows)),
        "aggregate": aggregate,
        "actor_certificate_rates": {
            "nonnew_visible_violation": rate("nonnew_visible_violation"),
            "exact_no_visible_contradiction": rate(
                "exact_no_visible_contradiction"
            ),
            "target_hit_noninferior_to_query": rate(
                "target_hit_noninferior_to_query"
            ),
            "visibility_fscore_noninferior_to_query": rate(
                "visibility_fscore_noninferior_to_query"
            ),
            "chamfer_worsened_vs_query": rate("chamfer_worsened_vs_query"),
        },
        "chamfer_worsened_stratum": {
            "actor_count": len(chamfer_worsened),
            "nonnew_visible_violation_rate": rate(
                "nonnew_visible_violation", chamfer_worsened
            ),
            "exact_no_visible_contradiction_rate": rate(
                "exact_no_visible_contradiction", chamfer_worsened
            ),
            "visibility_fscore_noninferior_rate": rate(
                "visibility_fscore_noninferior_to_query", chamfer_worsened
            ),
        },
        "partition_complete": all(
            bool(row[label]["partition_complete"])
            for row in rows
            for label in surfaces
        ),
        "actor_identity_trajectory_hazard_retention": float(
            min(float(row["actor_identity_trajectory_hazard_retention"]) for row in rows)
        ),
    }

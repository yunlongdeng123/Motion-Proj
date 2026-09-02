"""Output-provenance attribution for V7 visibility-certificate failures."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from motion_proj.worldsim_v7.av2_four_action_compiler import _nearest


PROVENANCE = ("KEEP", "PROJECT", "COMPLETE")


def _target_partition(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    _, nearest_indices = _nearest(
        target, surface, device, int(config["distance_chunk_size"])
    )
    nearest = surface[nearest_indices].astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    origins64 = origins.astype(np.float64, copy=False)
    vectors = target64 - origins64
    target_depth = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(target_depth[:, None], 1.0e-9)
    predicted_vectors = nearest - origins64
    predicted_depth = np.sum(predicted_vectors * directions, axis=1)
    lateral = np.linalg.norm(
        predicted_vectors - predicted_depth[:, None] * directions, axis=1
    )
    observable = (predicted_depth > 0.0) & (
        lateral <= float(config["ray_lateral_tolerance_m"])
    )
    tolerance = float(config["depth_tolerance_m"])
    return {
        "nearest_indices": nearest_indices,
        "early": observable & (predicted_depth < target_depth - tolerance),
        "hit": observable & (np.abs(predicted_depth - target_depth) <= tolerance),
    }


def _surface_contradiction(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    _, nearest_indices = _nearest(
        surface, target, device, int(config["distance_chunk_size"])
    )
    target64 = target.astype(np.float64, copy=False)[nearest_indices]
    origins64 = origins.astype(np.float64, copy=False)[nearest_indices]
    target_vectors = target64 - origins64
    target_depth = np.linalg.norm(target_vectors, axis=1)
    directions = target_vectors / np.maximum(target_depth[:, None], 1.0e-9)
    surface_vectors = surface.astype(np.float64, copy=False) - origins64
    surface_depth = np.sum(surface_vectors * directions, axis=1)
    lateral = np.linalg.norm(
        surface_vectors - surface_depth[:, None] * directions, axis=1
    )
    observable = (surface_depth > 0.0) & (
        lateral <= float(config["ray_lateral_tolerance_m"])
    )
    return observable & (
        surface_depth < target_depth - float(config["depth_tolerance_m"])
    )


def _compiled_provenance(
    diagnostics: Mapping[str, Any], voxel_size_m: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    layers = [
        np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
        np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
        np.asarray(diagnostics["completed"], dtype=np.float32).reshape(-1, 3),
    ]
    points = np.concatenate(layers, axis=0)
    labels = np.concatenate(
        [np.full(len(layer), index, dtype=np.int8) for index, layer in enumerate(layers)]
    )
    keys = np.floor(points / float(voxel_size_m)).astype(np.int32)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    selected = np.sort(first_indices)
    reconstructed = np.asarray(points[selected], dtype=np.float32)
    compiled = np.asarray(diagnostics["compiled"], dtype=np.float32)
    aligned = reconstructed.shape == compiled.shape and np.array_equal(
        reconstructed, compiled
    )
    return reconstructed, labels[selected], bool(aligned)


def _counts_by_provenance(mask: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(mask & (labels == index)))
        for index, name in enumerate(PROVENANCE)
    }


def attribute_actor(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    voxel_size_m: float,
    device: torch.device,
) -> dict[str, Any]:
    target = diagnostics["target"]
    origins = diagnostics["target_sensor_origins"]
    query_partition = _target_partition(
        diagnostics["query"], target, origins, config, device
    )
    compiled, labels, aligned = _compiled_provenance(diagnostics, voxel_size_m)
    compiled_partition = _target_partition(compiled, target, origins, config, device)
    nearest_labels = labels[compiled_partition["nearest_indices"]]
    new_early = compiled_partition["early"] & ~query_partition["early"]
    new_hit = compiled_partition["hit"] & ~query_partition["hit"]
    resolved_early = query_partition["early"] & ~compiled_partition["early"]
    surface_contradicted = _surface_contradiction(
        compiled, target, origins, config, device
    )
    return {
        "track_id": str(actor_row["track_id"]),
        "category": str(actor_row["category"]),
        "hazardous": bool(actor_row["hazardous"]),
        "compiled_provenance_aligned": aligned,
        "output_point_counts": {
            name: int(np.count_nonzero(labels == index))
            for index, name in enumerate(PROVENANCE)
        },
        "compiled_early_by_provenance": _counts_by_provenance(
            compiled_partition["early"], nearest_labels
        ),
        "new_early_by_provenance": _counts_by_provenance(new_early, nearest_labels),
        "compiled_hit_by_provenance": _counts_by_provenance(
            compiled_partition["hit"], nearest_labels
        ),
        "new_hit_by_provenance": _counts_by_provenance(new_hit, nearest_labels),
        "surface_contradiction_by_provenance": _counts_by_provenance(
            surface_contradicted, labels
        ),
        "query_early_count": int(np.count_nonzero(query_partition["early"])),
        "compiled_early_count": int(np.count_nonzero(compiled_partition["early"])),
        "new_early_count": int(np.count_nonzero(new_early)),
        "resolved_query_early_count": int(np.count_nonzero(resolved_early)),
        "query_hit_count": int(np.count_nonzero(query_partition["hit"])),
        "compiled_hit_count": int(np.count_nonzero(compiled_partition["hit"])),
        "new_hit_count": int(np.count_nonzero(new_hit)),
        "target_ray_count": int(len(target)),
    }


def summarize_attributions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("visible-failure attribution has no Actor rows")

    def summed(field: str) -> dict[str, int]:
        return {
            name: sum(int(row[field][name]) for row in rows) for name in PROVENANCE
        }

    new_early_total = sum(int(row["new_early_count"]) for row in rows)
    new_hit_total = sum(int(row["new_hit_count"]) for row in rows)
    target_total = sum(int(row["target_ray_count"]) for row in rows)
    return {
        "actor_count": len(rows),
        "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in rows)),
        "output_point_counts": summed("output_point_counts"),
        "compiled_early_by_provenance": summed("compiled_early_by_provenance"),
        "new_early_by_provenance": summed("new_early_by_provenance"),
        "compiled_hit_by_provenance": summed("compiled_hit_by_provenance"),
        "new_hit_by_provenance": summed("new_hit_by_provenance"),
        "surface_contradiction_by_provenance": summed(
            "surface_contradiction_by_provenance"
        ),
        "query_early_count": sum(int(row["query_early_count"]) for row in rows),
        "compiled_early_count": sum(int(row["compiled_early_count"]) for row in rows),
        "new_early_count": new_early_total,
        "resolved_query_early_count": sum(
            int(row["resolved_query_early_count"]) for row in rows
        ),
        "query_hit_count": sum(int(row["query_hit_count"]) for row in rows),
        "compiled_hit_count": sum(int(row["compiled_hit_count"]) for row in rows),
        "new_hit_count": new_hit_total,
        "target_ray_count": target_total,
        "new_early_fraction_of_target_rays": float(new_early_total / target_total),
        "new_hit_to_new_early_ratio": float(new_hit_total / max(new_early_total, 1)),
        "all_compiled_provenance_aligned": all(
            bool(row["compiled_provenance_aligned"]) for row in rows
        ),
    }

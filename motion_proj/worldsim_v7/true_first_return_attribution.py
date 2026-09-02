"""Literal first-return ray attribution for WorldSim V7 surfaces."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from motion_proj.worldsim_v7.av2_four_action_compiler import _quality, _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import (
    COMPLETION_STATES,
    OCCUPIED,
)
from motion_proj.worldsim_v7.visible_failure_attribution import (
    PROVENANCE,
    _compiled_provenance,
    _counts_by_provenance,
    _surface_contradiction,
)


def true_first_return_partition(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    surface = np.asarray(surface, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    origins = np.asarray(origins, dtype=np.float32).reshape(-1, 3)
    if not len(surface):
        return {
            "first_indices": np.zeros(len(target), dtype=np.int64),
            "first_depth": np.full(len(target), np.inf, dtype=np.float32),
            "early": np.zeros(len(target), dtype=np.bool_),
            "hit": np.zeros(len(target), dtype=np.bool_),
        }
    surface_tensor = torch.as_tensor(surface, dtype=torch.float32, device=device)
    target_depth_chunks = []
    first_depth_chunks = []
    first_index_chunks = []
    tolerance = float(config["ray_lateral_tolerance_m"])
    chunk_size = int(config["distance_chunk_size"])
    with torch.inference_mode():
        for start in range(0, len(target), chunk_size):
            target_tensor = torch.as_tensor(
                target[start : start + chunk_size], dtype=torch.float32, device=device
            )
            origin_tensor = torch.as_tensor(
                origins[start : start + chunk_size], dtype=torch.float32, device=device
            )
            target_vectors = target_tensor - origin_tensor
            target_depth = torch.linalg.vector_norm(target_vectors, dim=1)
            directions = target_vectors / target_depth[:, None].clamp_min(1.0e-9)
            surface_vectors = surface_tensor[None, :, :] - origin_tensor[:, None, :]
            depth = torch.sum(surface_vectors * directions[:, None, :], dim=-1)
            lateral = torch.linalg.vector_norm(
                surface_vectors - depth[:, :, None] * directions[:, None, :], dim=-1
            )
            valid = (depth > 0.0) & (lateral <= tolerance)
            first_depth, first_index = torch.min(
                torch.where(valid, depth, torch.full_like(depth, torch.inf)), dim=1
            )
            target_depth_chunks.append(target_depth.cpu())
            first_depth_chunks.append(first_depth.cpu())
            first_index_chunks.append(first_index.cpu())
    target_depth = torch.cat(target_depth_chunks).numpy()
    first_depth = torch.cat(first_depth_chunks).numpy()
    first_indices = torch.cat(first_index_chunks).numpy().astype(np.int64)
    observable = np.isfinite(first_depth)
    depth_tolerance = float(config["depth_tolerance_m"])
    return {
        "first_indices": first_indices,
        "first_depth": first_depth.astype(np.float32),
        "early": observable & (first_depth < target_depth - depth_tolerance),
        "hit": observable & (np.abs(first_depth - target_depth) <= depth_tolerance),
    }


def attribute_actor_true_first_return(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    voxel_size_m: float,
    device: torch.device,
) -> dict[str, Any]:
    target = np.asarray(diagnostics["target"], dtype=np.float32)
    origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32)
    query_partition = true_first_return_partition(
        diagnostics["query"], target, origins, config, device
    )
    compiled, labels, aligned = _compiled_provenance(diagnostics, voxel_size_m)
    compiled_partition = true_first_return_partition(
        compiled, target, origins, config, device
    )
    first_labels = labels[compiled_partition["first_indices"]]
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
        "ray_operator": "minimum_positive_depth_within_lateral_tolerance",
        "compiled_provenance_aligned": aligned,
        "output_point_counts": {
            name: int(np.count_nonzero(labels == index))
            for index, name in enumerate(PROVENANCE)
        },
        "compiled_early_by_provenance": _counts_by_provenance(
            compiled_partition["early"], first_labels
        ),
        "new_early_by_provenance": _counts_by_provenance(new_early, first_labels),
        "compiled_hit_by_provenance": _counts_by_provenance(
            compiled_partition["hit"], first_labels
        ),
        "new_hit_by_provenance": _counts_by_provenance(new_hit, first_labels),
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


def apply_completion_policy_true_first_return(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    predicted_states: np.ndarray,
    probabilities: np.ndarray,
    compiler_config: Mapping[str, Any],
    attribution_config: Mapping[str, Any],
    device: torch.device,
    baseline_attribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32)
    completed = candidates[predicted_states == OCCUPIED]
    compiled = _voxel_unique(
        np.concatenate(
            [
                np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
                np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
                completed.reshape(-1, 3),
            ],
            axis=0,
        ),
        float(compiler_config["compiler_geometry"]["output_voxel_size_m"]),
    )
    quality = _quality(
        compiled,
        np.asarray(diagnostics["target"], dtype=np.float32),
        compiler_config,
        device,
    )
    policy_diagnostics = dict(diagnostics)
    policy_diagnostics["compiled"] = compiled
    policy_diagnostics["completed"] = completed
    voxel_size = float(compiler_config["compiler_geometry"]["output_voxel_size_m"])
    if baseline_attribution is None:
        baseline_attribution = attribute_actor_true_first_return(
            actor_row, diagnostics, attribution_config, voxel_size, device
        )
    policy_attribution = attribute_actor_true_first_return(
        actor_row, policy_diagnostics, attribution_config, voxel_size, device
    )
    query_chamfer = float(actor_row["query_only"]["symmetric_chamfer_m"])
    baseline_chamfer = float(actor_row["after"]["symmetric_chamfer_m"])
    state_counts = {
        name: int(np.count_nonzero(predicted_states == index))
        for index, name in enumerate(COMPLETION_STATES)
    }
    return {
        "track_id": str(actor_row["track_id"]),
        "category": str(actor_row["category"]),
        "hazardous": bool(actor_row["hazardous"]),
        "candidate_count": int(len(candidates)),
        "predicted_state_counts": state_counts,
        "mean_unknown_score": (
            float(np.mean(probabilities[:, 2])) if len(probabilities) else 0.0
        ),
        "query_chamfer_m": query_chamfer,
        "baseline_chamfer_m": baseline_chamfer,
        "p16_chamfer_m": float(quality["symmetric_chamfer_m"]),
        "baseline_composite_gain_m": query_chamfer - baseline_chamfer,
        "p16_composite_gain_m": query_chamfer - float(quality["symmetric_chamfer_m"]),
        "p16_quality": quality,
        "baseline_attribution": dict(baseline_attribution),
        "p16_attribution": policy_attribution,
    }

"""S1 candidate displacement oracle。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v71.evaluate_surface import (
    differentiable_symmetric_chamfer,
    evaluate_actor_surface,
)
from motion_proj.worldsim_v71.first_return_renderer import differentiable_first_return_depth
from motion_proj.worldsim_v71.ray_displacement import estimate_surface_normals


@dataclass(frozen=True)
class OracleResult:
    row: dict[str, Any]
    moved_candidates: np.ndarray
    displacement: np.ndarray
    ray_directions: np.ndarray
    normals: np.ndarray


def _neighbor_pairs(points: torch.Tensor, neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(points) < 2:
        empty = torch.empty(0, dtype=torch.long, device=points.device)
        return empty, empty
    k = min(int(neighbors) + 1, len(points))
    indices = torch.cdist(points, points).topk(k, largest=False).indices[:, 1:]
    left = torch.arange(len(points), device=points.device)[:, None].expand_as(indices).reshape(-1)
    return left, indices.reshape(-1)


def optimize_candidate_displacement(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> OracleResult | None:
    candidates_np = np.asarray(diagnostics["completion_candidates"], dtype=np.float32).reshape(-1, 3)
    targets_np = np.asarray(diagnostics["target"], dtype=np.float32).reshape(-1, 3)
    origins_np = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32).reshape(-1, 3)
    if len(candidates_np) == 0 or len(targets_np) < 4:
        return None
    maximum_rays = int(config.get("maximum_total_rays", len(targets_np)))
    if len(targets_np) > maximum_rays:
        selected = np.linspace(0, len(targets_np) - 1, num=maximum_rays, dtype=np.int64)
        targets_np = targets_np[selected]
        origins_np = origins_np[selected]

    candidates = torch.as_tensor(candidates_np, dtype=torch.float32, device=device)
    anchors = torch.as_tensor(
        np.concatenate(
            [
                np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
                np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
            ],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    origins = torch.as_tensor(origins_np, dtype=torch.float32, device=device)
    fit_indices = torch.arange(len(targets), device=device) % 2 == 0
    check_indices = ~fit_indices
    fit_targets = targets[fit_indices]
    fit_origins = origins[fit_indices]
    check_targets_np = targets_np[check_indices.cpu().numpy()]
    check_origins_np = origins_np[check_indices.cpu().numpy()]

    normals = estimate_surface_normals(candidates, torch.cat([anchors, candidates], dim=0), int(config["normal_neighbors"]))
    query_origin = torch.as_tensor(
        diagnostics["query_sensor_origin"], dtype=torch.float32, device=device
    ).reshape(1, 3)
    ray_directions = candidates - query_origin
    ray_directions = ray_directions / torch.linalg.vector_norm(
        ray_directions, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    left, right = _neighbor_pairs(candidates, int(config["smooth_neighbors"]))
    raw = torch.nn.Parameter(torch.zeros((len(candidates), 2), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([raw], lr=float(config["learning_rate"]))
    half_size = torch.as_tensor(
        np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32) * 0.5,
        dtype=torch.float32,
        device=device,
    )
    maximum = torch.as_tensor(
        [float(config["maximum_ray_displacement_m"]), float(config["maximum_normal_displacement_m"])],
        dtype=torch.float32,
        device=device,
    )

    baseline_surface = torch.cat([anchors, candidates], dim=0)
    with torch.no_grad():
        reference_depth = differentiable_first_return_depth(
            baseline_surface, fit_origins, fit_targets, **config["renderer"]
        )
        target_depth = torch.linalg.vector_norm(fit_targets - fit_origins, dim=1)
        reference_first = F.smooth_l1_loss(reference_depth, target_depth).clamp_min(1.0e-3)
        reference_surface = differentiable_symmetric_chamfer(baseline_surface, fit_targets).clamp_min(1.0e-3)
    history = []
    for step in range(int(config["steps"])):
        delta_components = torch.tanh(raw) * maximum[None, :]
        displacement = (
            delta_components[:, :1] * ray_directions
            + delta_components[:, 1:] * normals
        )
        moved = candidates + displacement
        bounds = half_size[None, :] + float(config["cuboid_padding_m"])
        moved = torch.maximum(torch.minimum(moved, bounds), -bounds)
        surface = torch.cat([anchors, moved], dim=0)
        predicted_depth = differentiable_first_return_depth(
            surface, fit_origins, fit_targets, **config["renderer"]
        )
        target_depth = torch.linalg.vector_norm(fit_targets - fit_origins, dim=1)
        first_loss = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
        surface_loss = differentiable_symmetric_chamfer(surface, fit_targets) / reference_surface
        anchor_loss = displacement.square().mean() / maximum.square().mean().clamp_min(1.0e-6)
        smooth_loss = (
            (displacement[left] - displacement[right]).square().mean()
            if len(left)
            else torch.zeros((), dtype=surface.dtype, device=device)
        ) / maximum.square().mean().clamp_min(1.0e-6)
        loss = (
            first_loss
            + surface_loss
            + float(config["anchor_weight"]) * anchor_loss
            + float(config["smooth_weight"]) * smooth_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, int(config["steps"]) - 1}:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "first_return": float(first_loss.detach()),
                    "surface": float(surface_loss.detach()),
                }
            )

    with torch.no_grad():
        delta_components = torch.tanh(raw) * maximum[None, :]
        displacement = delta_components[:, :1] * ray_directions + delta_components[:, 1:] * normals
        moved = candidates + displacement
        bounds = half_size[None, :] + float(config["cuboid_padding_m"])
        moved = torch.maximum(torch.minimum(moved, bounds), -bounds)
    output_np = _voxel_unique(
        np.concatenate([anchors.cpu().numpy(), moved.cpu().numpy()], axis=0),
        float(config["output_voxel_size_m"]),
    )
    baseline_np = np.asarray(diagnostics["compiled"], dtype=np.float32).reshape(-1, 3)
    metrics = evaluate_actor_surface(
        baseline_np,
        output_np,
        check_targets_np,
        check_origins_np,
        hazardous=bool(actor_row["hazardous"]),
        device=device,
        lateral_tolerance_m=float(config["literal_lateral_tolerance_m"]),
        depth_tolerance_m=float(config["literal_depth_tolerance_m"]),
        distance_chunk_size=int(config["distance_chunk_size"]),
    )
    metrics.update(
        {
            "track_id": str(actor_row["track_id"]),
            "category": str(actor_row["category"]),
            "candidate_count": len(candidates_np),
            "anchor_count": len(anchors),
            "fit_ray_count": int(fit_indices.sum()),
            "check_ray_count": int(check_indices.sum()),
            "maximum_displacement_m": float(torch.linalg.vector_norm(displacement, dim=1).max()),
            "mean_displacement_m": float(torch.linalg.vector_norm(displacement, dim=1).mean()),
            "history": history,
        }
    )
    return OracleResult(
        row=metrics,
        moved_candidates=moved.cpu().numpy(),
        displacement=displacement.cpu().numpy(),
        ray_directions=ray_directions.cpu().numpy(),
        normals=normals.cpu().numpy(),
    )

"""Joint first-return rendering objective for WorldSim V7 completion sets."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v7.av2_four_action_compiler import (
    COMPLETION_FEATURE_NAMES,
    _nearest,
)
from motion_proj.worldsim_v7.completion_responsibility import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    FeatureStandardizer,
)


class RaySetCompletionMLP(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(COMPLETION_FEATURE_NAMES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _ray_point_geometry(
    points: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        shape = (len(origins), 0)
        return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
    point_tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
    depth_chunks, lateral_chunks = [], []
    with torch.inference_mode():
        for start in range(0, len(origins), chunk_size):
            origin = torch.as_tensor(
                origins[start : start + chunk_size], dtype=torch.float32, device=device
            )
            direction = torch.as_tensor(
                directions[start : start + chunk_size], dtype=torch.float32, device=device
            )
            vectors = point_tensor[None, :, :] - origin[:, None, :]
            depth = torch.sum(vectors * direction[:, None, :], dim=-1)
            lateral = torch.linalg.vector_norm(
                vectors - depth[:, :, None] * direction[:, None, :], dim=-1
            )
            depth_chunks.append(depth.cpu())
            lateral_chunks.append(lateral.cpu())
    return torch.cat(depth_chunks).numpy(), torch.cat(lateral_chunks).numpy()


def build_ray_package(
    diagnostics: Mapping[str, Any],
    attribution_config: Mapping[str, Any],
    maximum_rays: int,
    device: torch.device,
) -> dict[str, np.ndarray] | None:
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32)
    features = np.asarray(diagnostics["completion_features"], dtype=np.float32)
    if not len(candidates):
        return None
    target = np.asarray(diagnostics["target"], dtype=np.float32)
    origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32)
    vectors = target.astype(np.float64) - origins.astype(np.float64)
    target_depth = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(target_depth[:, None], 1.0e-9)
    core = np.concatenate(
        [
            np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
            np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
        ],
        axis=0,
    )
    chunk = int(attribution_config["distance_chunk_size"])
    tolerance = float(attribution_config["ray_lateral_tolerance_m"])
    core_depth, core_lateral = _ray_point_geometry(
        core, origins, directions, device, chunk
    )
    core_observable = (core_depth > 0.0) & (core_lateral <= tolerance)
    fixed_depth = np.min(np.where(core_observable, core_depth, np.inf), axis=1)
    candidate_depth, candidate_lateral = _ray_point_geometry(
        candidates, origins, directions, device, chunk
    )
    candidate_valid = (
        (candidate_depth > 0.0)
        & (candidate_lateral <= tolerance)
        & (candidate_depth < fixed_depth[:, None])
    )
    influential = np.flatnonzero(np.isfinite(fixed_depth) & np.any(candidate_valid, axis=1))
    if not len(influential):
        return None
    if len(influential) > int(maximum_rays):
        influential = influential[
            np.linspace(0, len(influential) - 1, int(maximum_rays), dtype=np.int64)
        ]
    depths = candidate_depth[influential]
    valid = candidate_valid[influential]
    order = np.argsort(np.where(valid, depths, np.inf), axis=1, kind="stable")
    sorted_depths = np.take_along_axis(depths, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)
    distance_chunk = int(attribution_config["distance_chunk_size"])
    core_to_target, _ = _nearest(core, target, device, distance_chunk)
    candidate_to_target, _ = _nearest(candidates, target, device, distance_chunk)
    target_to_core, _ = _nearest(target, core, device, distance_chunk)
    target_candidate_chunks = []
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in range(0, len(target), distance_chunk):
            target_tensor = torch.as_tensor(
                target[start : start + distance_chunk], dtype=torch.float32, device=device
            )
            target_candidate_chunks.append(torch.cdist(target_tensor, candidate_tensor).cpu())
    target_candidate = torch.cat(target_candidate_chunks).numpy()
    target_candidate_valid = target_candidate < target_to_core[:, None]
    target_candidate_order = np.argsort(
        np.where(target_candidate_valid, target_candidate, np.inf), axis=1, kind="stable"
    )
    target_candidate_sorted = np.take_along_axis(
        target_candidate, target_candidate_order, axis=1
    )
    target_candidate_valid_sorted = np.take_along_axis(
        target_candidate_valid, target_candidate_order, axis=1
    )
    baseline_ray_depth = np.min(np.where(sorted_valid, sorted_depths, np.inf), axis=1)
    ray_error = np.abs(baseline_ray_depth - target_depth[influential])
    baseline_ray_loss = float(np.mean(np.where(ray_error < 1.0, 0.5 * ray_error**2, ray_error - 0.5)))
    baseline_surface_distance = float(
        (np.sum(core_to_target) + np.sum(candidate_to_target))
        / max(len(core_to_target) + len(candidate_to_target), 1)
    )
    baseline_target_distance = float(
        np.mean(np.minimum(target_to_core, np.min(target_candidate, axis=1)))
    )
    baseline_chamfer = 0.5 * (baseline_surface_distance + baseline_target_distance)
    return {
        "features": features,
        "order": order.astype(np.int64),
        "candidate_depth": np.where(sorted_valid, sorted_depths, 0.0).astype(np.float32),
        "candidate_valid": sorted_valid.astype(np.bool_),
        "fixed_depth": fixed_depth[influential].astype(np.float32),
        "target_depth": target_depth[influential].astype(np.float32),
        "core_to_target_distance_sum": np.asarray(np.sum(core_to_target), dtype=np.float32),
        "core_point_count": np.asarray(len(core_to_target), dtype=np.float32),
        "candidate_to_target_distance": candidate_to_target.astype(np.float32),
        "target_candidate_order": target_candidate_order.astype(np.int64),
        "target_candidate_distance": np.where(
            target_candidate_valid_sorted, target_candidate_sorted, 0.0
        ).astype(np.float32),
        "target_candidate_valid": target_candidate_valid_sorted.astype(np.bool_),
        "target_core_distance": target_to_core.astype(np.float32),
        "baseline_ray_loss": np.asarray(max(baseline_ray_loss, 1.0e-4), dtype=np.float32),
        "baseline_chamfer": np.asarray(max(baseline_chamfer, 1.0e-4), dtype=np.float32),
    }


def package_to_device(
    package: Mapping[str, np.ndarray],
    standardizer: FeatureStandardizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(
            standardizer.transform(package["features"]), dtype=torch.float32, device=device
        ),
        "order": torch.as_tensor(package["order"], dtype=torch.long, device=device),
        "candidate_depth": torch.as_tensor(
            package["candidate_depth"], dtype=torch.float32, device=device
        ),
        "candidate_valid": torch.as_tensor(
            package["candidate_valid"], dtype=torch.bool, device=device
        ),
        "fixed_depth": torch.as_tensor(
            package["fixed_depth"], dtype=torch.float32, device=device
        ),
        "target_depth": torch.as_tensor(
            package["target_depth"], dtype=torch.float32, device=device
        ),
        "core_to_target_distance_sum": torch.as_tensor(
            package["core_to_target_distance_sum"], dtype=torch.float32, device=device
        ),
        "core_point_count": torch.as_tensor(
            package["core_point_count"], dtype=torch.float32, device=device
        ),
        "candidate_to_target_distance": torch.as_tensor(
            package["candidate_to_target_distance"], dtype=torch.float32, device=device
        ),
        "target_candidate_order": torch.as_tensor(
            package["target_candidate_order"], dtype=torch.long, device=device
        ),
        "target_candidate_distance": torch.as_tensor(
            package["target_candidate_distance"], dtype=torch.float32, device=device
        ),
        "target_candidate_valid": torch.as_tensor(
            package["target_candidate_valid"], dtype=torch.bool, device=device
        ),
        "target_core_distance": torch.as_tensor(
            package["target_core_distance"], dtype=torch.float32, device=device
        ),
        "baseline_ray_loss": torch.as_tensor(
            package["baseline_ray_loss"], dtype=torch.float32, device=device
        ),
        "baseline_chamfer": torch.as_tensor(
            package["baseline_chamfer"], dtype=torch.float32, device=device
        ),
    }


def rendered_actor_loss(
    model: RaySetCompletionMLP,
    package: Mapping[str, torch.Tensor],
    threshold: float,
    hybrid_chamfer: bool = False,
) -> torch.Tensor:
    soft = torch.sigmoid(model(package["features"]))
    hard = (soft >= float(threshold)).to(soft.dtype)
    alpha = hard.detach() - soft.detach() + soft
    ordered_alpha = alpha[package["order"]]
    ordered_alpha = torch.where(
        package["candidate_valid"], ordered_alpha, torch.zeros_like(ordered_alpha)
    )
    transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones((len(ordered_alpha), 1), device=ordered_alpha.device),
                1.0 - ordered_alpha,
            ],
            dim=1,
        ),
        dim=1,
    )
    point_weights = transmittance[:, :-1] * ordered_alpha
    rendered_depth = torch.sum(
        point_weights * package["candidate_depth"], dim=1
    ) + transmittance[:, -1] * package["fixed_depth"]
    ray_loss = torch.nn.functional.smooth_l1_loss(
        rendered_depth, package["target_depth"], reduction="mean"
    )
    if not hybrid_chamfer:
        return ray_loss
    surface_to_target = (
        package["core_to_target_distance_sum"]
        + torch.sum(alpha * package["candidate_to_target_distance"])
    ) / (package["core_point_count"] + torch.sum(alpha)).clamp_min(1.0)
    target_alpha = alpha[package["target_candidate_order"]]
    target_alpha = torch.where(
        package["target_candidate_valid"], target_alpha, torch.zeros_like(target_alpha)
    )
    target_transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones((len(target_alpha), 1), device=target_alpha.device),
                1.0 - target_alpha,
            ],
            dim=1,
        ),
        dim=1,
    )
    target_weights = target_transmittance[:, :-1] * target_alpha
    target_to_surface = torch.mean(
        torch.sum(target_weights * package["target_candidate_distance"], dim=1)
        + target_transmittance[:, -1] * package["target_core_distance"]
    )
    expected_chamfer = 0.5 * (surface_to_target + target_to_surface)
    return ray_loss / package["baseline_ray_loss"] + expected_chamfer / package["baseline_chamfer"]


def predict_ray_set(
    model: RaySetCompletionMLP,
    standardizer: FeatureStandardizer,
    features: np.ndarray,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(features):
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    tensor = torch.as_tensor(
        standardizer.transform(features), dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        occupancy = torch.sigmoid(model(tensor)).cpu().numpy().astype(np.float32)
    selected = occupancy >= float(threshold)
    states = np.where(selected, OCCUPIED, UNKNOWN).astype(np.int64)
    probabilities = np.zeros((len(features), 3), dtype=np.float32)
    probabilities[:, OCCUPIED] = occupancy
    probabilities[:, UNKNOWN] = 1.0 - occupancy
    probabilities[:, FREE] = 0.0
    return states, probabilities, occupancy

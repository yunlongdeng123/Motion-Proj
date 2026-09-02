"""Joint first-return rendering objective for WorldSim V7 completion sets."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v7.av2_four_action_compiler import COMPLETION_FEATURE_NAMES
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
    return {
        "features": features,
        "order": order.astype(np.int64),
        "candidate_depth": np.where(sorted_valid, sorted_depths, 0.0).astype(np.float32),
        "candidate_valid": sorted_valid.astype(np.bool_),
        "fixed_depth": fixed_depth[influential].astype(np.float32),
        "target_depth": target_depth[influential].astype(np.float32),
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
    }


def rendered_actor_loss(
    model: RaySetCompletionMLP,
    package: Mapping[str, torch.Tensor],
    threshold: float,
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
    return torch.nn.functional.smooth_l1_loss(
        rendered_depth, package["target_depth"], reduction="mean"
    )


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

"""Train an M8-guided query-local signed field from GT ray narrow bands."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m6_gt_supervised_gaussian_relocation as m6_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import (
    GaussianSeedExpansionMLP,
    build_gaussian_anchor_targets,
)
from motion_proj.worldsim_v71.local_signed_field import (
    CompactLocalOccupancyField,
    DirectQuerySignedField,
    LocalAnchorSignedField,
    OneSidedLocalOccupancyField,
    RaySurvivalDensityField,
    RayTerminationLogitField,
    initialize_local_field_from_expansion,
)
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _field(
    model: LocalAnchorSignedField,
    actor: Mapping[str, Any],
    queries: torch.Tensor,
) -> torch.Tensor:
    return model(
        actor["features"],
        actor["m8_children_t"],
        actor["m8_scales_t"],
        actor["normals_t"],
        actor["ray_directions_t"],
        queries,
    )


def _ray_training_points(
    actor: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), device
    )
    directions = F.normalize(targets - origins, dim=1, eps=1.0e-6)
    front_offsets = torch.as_tensor(
        config["front_offsets_m"], dtype=targets.dtype, device=device
    )
    back_offsets = torch.as_tensor(
        config["back_offsets_m"], dtype=targets.dtype, device=device
    )
    fronts = (
        targets[:, None, :] - directions[:, None, :] * front_offsets[None, :, None]
    ).reshape(-1, 3)
    backs = (
        targets[:, None, :] + directions[:, None, :] * back_offsets[None, :, None]
    ).reshape(-1, 3)
    front_labels = front_offsets[None, :].expand(len(targets), -1).reshape(-1)
    back_labels = -back_offsets[None, :].expand(len(targets), -1).reshape(-1)
    return targets, fronts, backs, front_labels, back_labels


def _geometry_losses(
    model: LocalAnchorSignedField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), actor["features"].device
    )
    hit_field = _field(model, actor, targets)
    with torch.no_grad():
        target_normals = build_gaussian_anchor_targets(
            targets,
            targets,
            neighbors=int(config["target_neighbors"]),
            minimum_scale_m=float(config["minimum_scale_m"]),
            maximum_scale_m=float(config["maximum_scale_m"]),
        ).normals
        outward = F.normalize(origins - targets, dim=1, eps=1.0e-6)
        target_normals = torch.where(
            torch.sum(target_normals * outward, dim=1, keepdim=True) >= 0.0,
            target_normals,
            -target_normals,
        )
    probe_offset = float(config["normal_probe_offset_m"])
    outside_field = _field(
        model, actor, targets + probe_offset * target_normals
    )
    inside_field = _field(
        model, actor, targets - probe_offset * target_normals
    )
    hit = hit_field.abs().mean()
    normal = 0.5 * (
        F.smooth_l1_loss(outside_field, torch.full_like(outside_field, probe_offset))
        + F.smooth_l1_loss(inside_field, torch.full_like(inside_field, -probe_offset))
    )
    radius = hit_field.new_zeros(())
    if isinstance(model, OneSidedLocalOccupancyField):
        predicted_radii = model.child_radii(
            actor["features"], actor["m8_scales_t"]
        )
        with torch.no_grad():
            child_normals = model.outward_child_normals(
                actor["normals_t"], actor["ray_directions_t"]
            )
            child_centers = actor["m8_children_t"]
            relative = targets[None, :, :] - child_centers[:, None, :]
            neighbor_count = min(int(config["target_neighbors"]), len(targets))
            indices = torch.linalg.vector_norm(relative, dim=-1).topk(
                neighbor_count, dim=1, largest=False
            ).indices
            nearest = torch.gather(
                relative,
                1,
                indices[:, :, None].expand(-1, -1, 3),
            )
            normal_coordinate = torch.sum(
                nearest * child_normals[:, None, :], dim=-1, keepdim=True
            )
            tangent = nearest - normal_coordinate * child_normals[:, None, :]
            target_radii = torch.linalg.vector_norm(tangent, dim=-1).max(dim=1).values
            target_radii = target_radii.clamp(
                min=float(config["minimum_scale_m"]),
                max=float(config["maximum_radius_m"]),
            )
        radius = F.smooth_l1_loss(
            predicted_radii / target_radii,
            torch.ones_like(predicted_radii),
        )
    geometry = float(config["hit_weight"]) * hit + float(
        config["normal_weight"]
    ) * normal + float(config.get("radius_weight", 0.0)) * radius
    return {"geometry": geometry, "hit": hit, "normal": normal, "radius": radius}


def _physics_losses(
    model: LocalAnchorSignedField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    targets, fronts, backs, front_labels, back_labels = _ray_training_points(
        actor, config, actor["features"].device
    )
    front_field = _field(model, actor, fronts)
    back_field = _field(model, actor, backs)
    hit_field = _field(model, actor, targets)
    free_fields = [front_field]
    free_labels = [front_labels]
    full_ray_free = front_field.new_zeros(())
    if int(config.get("full_ray_free_samples", 0)) > 0:
        full_queries, full_labels = _full_ray_free_training_points(
            actor, config, actor["features"].device
        )
        if len(full_queries) > 0:
            full_field = _field(model, actor, full_queries)
            free_fields.append(full_field)
            free_labels.append(full_labels)
            full_ray_free = F.softplus(
                -full_field / float(config["sign_temperature_m"])
            ).mean()
    combined_free_field = torch.cat(free_fields, dim=0)
    combined_free_labels = torch.cat(free_labels, dim=0)
    predictions = torch.cat([combined_free_field, hit_field, back_field], dim=0)
    labels = torch.cat(
        [combined_free_labels, torch.zeros_like(hit_field), back_labels], dim=0
    )
    signed_regression = F.smooth_l1_loss(predictions, labels)
    temperature = float(config["sign_temperature_m"])
    front_free = F.softplus(-combined_free_field / temperature).mean()
    back_occupied = F.softplus(back_field / temperature).mean()
    physics = float(config["signed_regression_weight"]) * signed_regression + float(
        config["physics_weight"]
    ) * (front_free + back_occupied)
    return {
        "physics": physics,
        "signed_regression": signed_regression,
        "front_free": front_free,
        "full_ray_free": full_ray_free,
        "back_occupied": back_occupied,
    }


def _full_ray_free_training_points(
    actor: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, _, valid_box = _ray_box_intervals(origins, directions, bounds)
    free_end = target_depth - float(config["minimum_front_clearance_m"])
    valid = valid_box & (free_end > entry + 1.0e-4)
    sample_count = int(config["full_ray_free_samples"])
    fractions = torch.linspace(
        0.0, 1.0, sample_count, dtype=targets.dtype, device=device
    )
    depths = entry[:, None] + (free_end - entry)[:, None] * fractions[None, :]
    queries = origins[:, None, :] + depths[:, :, None] * directions[:, None, :]
    labels = (target_depth[:, None] - depths).clamp_max(
        float(config["maximum_field_distance_m"])
    )
    return queries[valid].reshape(-1, 3), labels[valid].reshape(-1)


def _ray_survival_values(
    model: RaySurvivalDensityField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = _ray_box_intervals(origins, directions, bounds)
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    fractions = torch.linspace(
        0.0, 1.0, sample_count, dtype=targets.dtype, device=device
    )
    depths = entry[valid, None] + (
        exit_depth[valid] - entry[valid]
    )[:, None] * fractions[None, :]
    queries = (
        origins[valid, None, :]
        + depths[:, :, None] * directions[valid, None, :]
    )
    density = _field(model, actor, queries.reshape(-1, 3)).reshape(
        len(depths), sample_count
    )
    spacing = (exit_depth[valid] - entry[valid]).clamp_min(1.0e-4) / max(
        sample_count - 1, 1
    )
    optical_thickness = density * spacing[:, None]
    cumulative = torch.cumsum(optical_thickness, dim=1)
    cdf = 1.0 - torch.exp(-cumulative)
    transmittance_before = torch.exp(-cumulative + optical_thickness)
    weights = (1.0 - torch.exp(-optical_thickness)) * transmittance_before
    return depths, target_depth[valid], cdf, weights, density


def _survival_losses(
    model: RaySurvivalDensityField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, cdf, weights, density = _ray_survival_values(
        model,
        actor,
        config,
        actor["features"].device,
        int(config["survival_train_samples"]),
    )
    epsilon = 1.0e-6
    before = depths < target_depth[:, None]
    after = ~before
    free_survival = -torch.log((1.0 - cdf).clamp_min(epsilon))[before].mean()
    hit_termination = -torch.log(cdf.clamp_min(epsilon))[after].mean()
    normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(epsilon)
    expected_depth = torch.sum(normalized_weights * depths, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    loss = (
        free_survival
        + hit_termination
        + float(config["survival_depth_weight"]) * depth_l1
    )
    return {
        "loss": loss,
        "free_survival": free_survival,
        "hit_termination": hit_termination,
        "depth_l1": depth_l1,
        "mean_density": density.mean(),
        "terminal_opacity": cdf[:, -1].mean(),
    }


def _train_survival(
    model: RaySurvivalDensityField,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "loss",
        "free_survival",
        "hit_termination",
        "depth_l1",
        "mean_density",
        "terminal_opacity",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["fine_tune_epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [_survival_losses(model, actors[index], config) for index in indices]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / len(actors) for name, value in totals.items()},
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "stage": str(config["training_stage"]),
                    "actors": len(actors),
                    **row,
                }
            ),
            flush=True,
        )
    return history


def _ray_categorical_values(
    model: RayTerminationLogitField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = _ray_box_intervals(origins, directions, bounds)
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    fractions = torch.linspace(
        0.0, 1.0, sample_count, dtype=targets.dtype, device=device
    )
    depths = entry[valid, None] + (
        exit_depth[valid] - entry[valid]
    )[:, None] * fractions[None, :]
    queries = (
        origins[valid, None, :]
        + depths[:, :, None] * directions[valid, None, :]
    )
    logits = _field(model, actor, queries.reshape(-1, 3)).reshape(
        len(depths), sample_count
    )
    return depths, target_depth[valid], logits


def _categorical_losses(
    model: RayTerminationLogitField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, logits = _ray_categorical_values(
        model,
        actor,
        config,
        actor["features"].device,
        int(config["categorical_train_bins"]),
    )
    target_bins = torch.abs(depths - target_depth[:, None]).argmin(dim=1)
    categorical_nll = F.cross_entropy(logits, target_bins)
    probabilities = torch.softmax(logits, dim=1)
    expected_depth = torch.sum(probabilities * depths, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1.0e-8)), dim=1
    ).mean()
    loss = categorical_nll + float(config["categorical_depth_weight"]) * depth_l1
    return {
        "loss": loss,
        "categorical_nll": categorical_nll,
        "depth_l1": depth_l1,
        "entropy": entropy,
        "target_probability": probabilities.gather(
            1, target_bins[:, None]
        ).mean(),
    }


def _train_categorical(
    model: RayTerminationLogitField,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "loss",
        "categorical_nll",
        "depth_l1",
        "entropy",
        "target_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["fine_tune_epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [_categorical_losses(model, actors[index], config) for index in indices]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / len(actors) for name, value in totals.items()},
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "stage": str(config["training_stage"]),
                    "actors": len(actors),
                    **row,
                }
            ),
            flush=True,
        )
    return history


def _train(
    model: LocalAnchorSignedField,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    if isinstance(model, RayTerminationLogitField):
        return _train_categorical(model, actors, config, optimizer)
    if isinstance(model, RaySurvivalDensityField):
        return _train_survival(model, actors, config, optimizer)
    history: list[dict[str, float | int]] = []
    names = (
        "geometry",
        "hit",
        "normal",
        "radius",
        "physics",
        "signed_regression",
        "front_free",
        "full_ray_free",
        "back_occupied",
    )
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["fine_tune_epochs"])):
        totals = {name: 0.0 for name in names}
        conflicts = 0
        cosines: list[float] = []
        batches = 0
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            geometry_items = [_geometry_losses(model, actors[i], config) for i in indices]
            physics_items = [_physics_losses(model, actors[i], config) for i in indices]
            means = {
                **{
                    name: torch.stack([item[name] for item in geometry_items]).mean()
                    for name in ("geometry", "hit", "normal", "radius")
                },
                **{
                    name: torch.stack([item[name] for item in physics_items]).mean()
                    for name in (
                        "physics",
                        "signed_regression",
                        "front_free",
                        "full_ray_free",
                        "back_occupied",
                    )
                },
            }
            conflict, cosine = m6_runner._pcgrad_step(
                model, optimizer, means["geometry"], means["physics"]
            )
            conflicts += int(conflict)
            cosines.append(cosine)
            batches += 1
            for name in totals:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / len(actors) for name, value in totals.items()},
            "conflict_batch_fraction": conflicts / max(batches, 1),
            "mean_preprojection_cosine": float(np.mean(cosines)),
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "stage": str(config.get("training_stage", "m13_local_signed_field")),
                    "actors": len(actors),
                    **row,
                }
            ),
            flush=True,
        )
    return history


def _ray_box_intervals(
    origins: torch.Tensor, directions: torch.Tensor, bounds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_parallel = directions.abs() > 1.0e-7
    safe = torch.where(non_parallel, directions, torch.ones_like(directions))
    first = (-bounds.reshape(1, 3) - origins) / safe
    second = (bounds.reshape(1, 3) - origins) / safe
    lower = torch.minimum(first, second)
    upper = torch.maximum(first, second)
    inside = origins.abs() <= bounds.reshape(1, 3)
    lower = torch.where(
        non_parallel,
        lower,
        torch.where(inside, torch.full_like(lower, -torch.inf), torch.full_like(lower, torch.inf)),
    )
    upper = torch.where(
        non_parallel,
        upper,
        torch.where(inside, torch.full_like(upper, torch.inf), torch.full_like(upper, -torch.inf)),
    )
    entry = lower.max(dim=1).values.clamp_min(0.0)
    exit_depth = upper.min(dim=1).values
    return entry, exit_depth, exit_depth > entry


def _field_first_return_partition(
    model: LocalAnchorSignedField,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = _ray_box_intervals(origins, directions, bounds)
    fractions = torch.linspace(
        0.0,
        1.0,
        int(evaluation["field_sample_count"]),
        dtype=targets.dtype,
        device=device,
    )
    first_depths = []
    observables = []
    ray_chunk = int(evaluation["ray_chunk_size"])
    with torch.inference_mode():
        for start in range(0, len(targets), ray_chunk):
            local_entry = entry[start : start + ray_chunk]
            local_exit = exit_depth[start : start + ray_chunk]
            local_valid = valid_box[start : start + ray_chunk]
            depths = local_entry[:, None] + (
                local_exit - local_entry
            )[:, None] * fractions[None, :]
            queries = (
                origins[start : start + ray_chunk, None, :]
                + depths[:, :, None] * directions[start : start + ray_chunk, None, :]
            )
            values = _field(model, actor, queries.reshape(-1, 3)).reshape(
                len(local_entry), -1
            )
            starts_inside = values[:, 0] <= 0.0
            crossings = (values[:, :-1] > 0.0) & (values[:, 1:] <= 0.0)
            has_crossing = crossings.any(dim=1)
            indices = crossings.to(torch.int64).argmax(dim=1)
            left_value = values.gather(1, indices[:, None]).squeeze(1)
            right_value = values.gather(1, (indices + 1)[:, None]).squeeze(1)
            left_depth = depths.gather(1, indices[:, None]).squeeze(1)
            right_depth = depths.gather(1, (indices + 1)[:, None]).squeeze(1)
            ratio = left_value / (left_value - right_value).clamp_min(1.0e-6)
            crossing_depth = left_depth + ratio.clamp(0.0, 1.0) * (
                right_depth - left_depth
            )
            observable = local_valid & (starts_inside | has_crossing)
            selected = torch.where(starts_inside, local_entry, crossing_depth)
            first_depths.append(
                torch.where(observable, selected, torch.full_like(selected, torch.inf)).cpu()
            )
            observables.append(observable.cpu())
    first_depth = torch.cat(first_depths).numpy()
    observable = torch.cat(observables).numpy().astype(bool)
    target_depth_np = target_depth.cpu().numpy()
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "first_depth": first_depth.astype(np.float32),
        "target_depth": target_depth_np.astype(np.float32),
        "observable": observable,
        "early": observable & (first_depth < target_depth_np - tolerance),
        "hit": observable & (np.abs(first_depth - target_depth_np) <= tolerance),
    }


def _survival_first_return_partition(
    model: RaySurvivalDensityField,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = _ray_box_intervals(origins, directions, bounds)
    fractions = torch.linspace(
        0.0,
        1.0,
        int(evaluation["field_sample_count"]),
        dtype=targets.dtype,
        device=device,
    )
    first_depths = []
    observables = []
    ray_chunk = int(evaluation["ray_chunk_size"])
    threshold = float(evaluation["survival_median_threshold"])
    with torch.inference_mode():
        for start in range(0, len(targets), ray_chunk):
            local_entry = entry[start : start + ray_chunk]
            local_exit = exit_depth[start : start + ray_chunk]
            local_valid = valid_box[start : start + ray_chunk]
            depths = local_entry[:, None] + (
                local_exit - local_entry
            )[:, None] * fractions[None, :]
            queries = (
                origins[start : start + ray_chunk, None, :]
                + depths[:, :, None] * directions[start : start + ray_chunk, None, :]
            )
            density = _field(model, actor, queries.reshape(-1, 3)).reshape(
                len(local_entry), -1
            )
            spacing = (local_exit - local_entry).clamp_min(1.0e-4) / max(
                density.shape[1] - 1, 1
            )
            cdf = 1.0 - torch.exp(
                -torch.cumsum(density * spacing[:, None], dim=1)
            )
            reached = cdf >= threshold
            has_termination = reached.any(dim=1)
            indices = reached.to(torch.int64).argmax(dim=1)
            previous_indices = (indices - 1).clamp_min(0)
            right_cdf = cdf.gather(1, indices[:, None]).squeeze(1)
            gathered_left_cdf = cdf.gather(
                1, previous_indices[:, None]
            ).squeeze(1)
            left_cdf = torch.where(
                indices > 0, gathered_left_cdf, torch.zeros_like(gathered_left_cdf)
            )
            right_depth = depths.gather(1, indices[:, None]).squeeze(1)
            gathered_left_depth = depths.gather(
                1, previous_indices[:, None]
            ).squeeze(1)
            left_depth = torch.where(
                indices > 0, gathered_left_depth, local_entry
            )
            ratio = (threshold - left_cdf) / (right_cdf - left_cdf).clamp_min(1.0e-6)
            selected = left_depth + ratio.clamp(0.0, 1.0) * (
                right_depth - left_depth
            )
            observable = local_valid & has_termination
            first_depths.append(
                torch.where(
                    observable, selected, torch.full_like(selected, torch.inf)
                ).cpu()
            )
            observables.append(observable.cpu())
    first_depth = torch.cat(first_depths).numpy()
    observable = torch.cat(observables).numpy().astype(bool)
    target_depth_np = target_depth.cpu().numpy()
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "first_depth": first_depth.astype(np.float32),
        "target_depth": target_depth_np.astype(np.float32),
        "observable": observable,
        "early": observable & (first_depth < target_depth_np - tolerance),
        "hit": observable & (np.abs(first_depth - target_depth_np) <= tolerance),
    }


def _categorical_first_return_partition(
    model: RayTerminationLogitField,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = _ray_box_intervals(origins, directions, bounds)
    fractions = torch.linspace(
        0.0,
        1.0,
        int(evaluation["field_sample_count"]),
        dtype=targets.dtype,
        device=device,
    )
    first_depths = []
    observables = []
    ray_chunk = int(evaluation["ray_chunk_size"])
    threshold = float(evaluation["categorical_median_threshold"])
    with torch.inference_mode():
        for start in range(0, len(targets), ray_chunk):
            local_entry = entry[start : start + ray_chunk]
            local_exit = exit_depth[start : start + ray_chunk]
            local_valid = valid_box[start : start + ray_chunk]
            depths = local_entry[:, None] + (
                local_exit - local_entry
            )[:, None] * fractions[None, :]
            queries = (
                origins[start : start + ray_chunk, None, :]
                + depths[:, :, None] * directions[start : start + ray_chunk, None, :]
            )
            logits = _field(model, actor, queries.reshape(-1, 3)).reshape(
                len(local_entry), -1
            )
            cdf = torch.softmax(logits, dim=1).cumsum(dim=1)
            reached = cdf >= threshold
            indices = reached.to(torch.int64).argmax(dim=1)
            previous_indices = (indices - 1).clamp_min(0)
            right_cdf = cdf.gather(1, indices[:, None]).squeeze(1)
            gathered_left_cdf = cdf.gather(
                1, previous_indices[:, None]
            ).squeeze(1)
            left_cdf = torch.where(
                indices > 0, gathered_left_cdf, torch.zeros_like(gathered_left_cdf)
            )
            right_depth = depths.gather(1, indices[:, None]).squeeze(1)
            gathered_left_depth = depths.gather(
                1, previous_indices[:, None]
            ).squeeze(1)
            left_depth = torch.where(
                indices > 0, gathered_left_depth, local_entry
            )
            ratio = (threshold - left_cdf) / (right_cdf - left_cdf).clamp_min(1.0e-6)
            selected = left_depth + ratio.clamp(0.0, 1.0) * (
                right_depth - left_depth
            )
            first_depths.append(
                torch.where(
                    local_valid, selected, torch.full_like(selected, torch.inf)
                ).cpu()
            )
            observables.append(local_valid.cpu())
    first_depth = torch.cat(first_depths).numpy()
    observable = torch.cat(observables).numpy().astype(bool)
    target_depth_np = target_depth.cpu().numpy()
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "first_depth": first_depth.astype(np.float32),
        "target_depth": target_depth_np.astype(np.float32),
        "observable": observable,
        "early": observable & (first_depth < target_depth_np - tolerance),
        "hit": observable & (np.abs(first_depth - target_depth_np) <= tolerance),
    }


def _field_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["target_ray_count"]) for row in selected)
        before_early = sum(int(row["output_early_count"]) for row in selected)
        after_early = sum(int(row["field_early_count"]) for row in selected)
        before_hit = sum(int(row["output_hit_count"]) for row in selected)
        after_hit = sum(int(row["field_hit_count"]) for row in selected)
        before_rate = before_early / rays if rays else None
        after_rate = after_early / rays if rays else None
        before_hit_rate = before_hit / rays if rays else None
        after_hit_rate = after_hit / rays if rays else None
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "m8_early_rate": before_rate,
            "field_early_rate": after_rate,
            "relative_early_reduction": (
                (before_rate - after_rate) / before_rate
                if before_rate is not None and before_rate > 0.0
                else None
            ),
            "m8_hit_recall": before_hit_rate,
            "field_hit_recall": after_hit_rate,
            "hit_recall_delta": (
                after_hit_rate - before_hit_rate
                if before_hit_rate is not None
                else None
            ),
            "field_observable_rate": sum(
                int(row["field_observable_count"]) for row in selected
            )
            / rays
            if rays
            else None,
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    m6_runner._write_json(
        run_dir / "status.json", {"status": "running", "phase": "loading_train"}
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M13 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        m8_run = Path(config["m8_run"])
        checkpoint = torch.load(m8_run / "MODEL.pt", map_location=device, weights_only=False)
        m8_config = yaml.safe_load((m8_run / "resolved.yaml").read_text(encoding="utf-8"))
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        m5_run = Path(checkpoint["m5_run"])
        m5_checkpoint = torch.load(
            m5_run / "MODEL.pt", map_location=device, weights_only=False
        )
        m5_config = yaml.safe_load((m5_run / "resolved.yaml").read_text(encoding="utf-8"))
        base = RaySurfaceRelocationMLP(
            int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
        ).to(device)
        base.load_state_dict(m5_checkpoint["state_dict"])
        base.eval()
        base.requires_grad_(False)
        m8_model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        m8_model.load_state_dict(checkpoint["state_dict"])
        m8_model.eval()
        m8_model.requires_grad_(False)
        field_variant = str(
            config["model"].get("field_variant", "blended_plane")
        )
        if field_variant == "compact_occupancy_union":
            field_class = CompactLocalOccupancyField
        elif field_variant == "one_sided_surface_cell":
            field_class = OneSidedLocalOccupancyField
        elif field_variant == "direct_query_signed":
            field_class = DirectQuerySignedField
        elif field_variant == "ray_survival_density":
            field_class = RaySurvivalDensityField
        elif field_variant == "categorical_first_return":
            field_class = RayTerminationLogitField
        else:
            field_class = LocalAnchorSignedField
        field_kwargs: dict[str, Any] = {}
        if field_class is OneSidedLocalOccupancyField:
            field_kwargs.update(
                maximum_log_radius_delta=float(
                    config["model"]["maximum_log_radius_delta"]
                ),
                back_support_depth_m=float(
                    config["model"]["back_support_depth_m"]
                ),
            )
        elif field_class is DirectQuerySignedField:
            field_kwargs.update(
                maximum_field_distance_m=float(
                    config["model"]["maximum_field_distance_m"]
                )
            )
        elif field_class is RaySurvivalDensityField:
            field_kwargs.update(
                density_scale_per_m=float(
                    config["model"]["density_scale_per_m"]
                ),
                initial_density_bias=float(
                    config["model"]["initial_density_bias"]
                ),
            )
        model = field_class(
            int(checkpoint["input_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            branch_factor=int(config["model"]["branch_factor"]),
            slot_dim=int(config["model"]["slot_dim"]),
            latent_dim=int(config["model"]["latent_dim"]),
            neighbor_count=int(config["model"]["neighbor_count"]),
            maximum_residual_fraction=float(
                config["model"]["maximum_residual_fraction"]
            ),
            **field_kwargs,
        ).to(device)
        initialize_local_field_from_expansion(model, m8_model)
        if isinstance(model, RaySurvivalDensityField):
            with torch.no_grad():
                model.query_decoder[-1].bias.fill_(
                    float(config["model"]["initial_density_bias"])
                )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["model"]["maximum_training_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, centers = m5_runner._move(base, actor, m5_config["model"])
                actor["m5_centers_t"] = centers.detach()
                children, _, scales = m7_runner._predict(
                    m8_model, actor, m8_config["model"]
                )
                actor["m8_children_t"] = children.detach().clone()
                actor["m8_scales_t"] = scales.detach().clone()
        for actor in actors:
            actor["m8_children_t"] = actor["m8_children_t"].clone()
            actor["m8_scales_t"] = actor["m8_scales_t"].clone()
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        model.train()
        m6_runner._write_json(
            run_dir / "status.json", {"status": "running", "phase": "local_field_finetune"}
        )
        history = _train(model, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "latent_dim": int(config["model"]["latent_dim"]),
                "branch_factor": int(config["model"]["branch_factor"]),
                "slot_dim": int(config["model"]["slot_dim"]),
                "neighbor_count": int(config["model"]["neighbor_count"]),
                "maximum_residual_fraction": float(
                    config["model"]["maximum_residual_fraction"]
                ),
                "seed": int(config["model"]["seed"]),
                "m5_run": str(m5_run),
                "m8_run": str(m8_run),
                "field": str(
                    config["model"].get("field_variant", "blended_plane")
                ),
                "maximum_log_radius_delta": config["model"].get(
                    "maximum_log_radius_delta"
                ),
                "back_support_depth_m": config["model"].get(
                    "back_support_depth_m"
                ),
                "maximum_field_distance_m": config["model"].get(
                    "maximum_field_distance_m"
                ),
                "density_scale_per_m": config["model"].get(
                    "density_scale_per_m"
                ),
                "initial_density_bias": config["model"].get(
                    "initial_density_bias"
                ),
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                output = _voxel_unique(
                    torch.cat([actor["anchors_t"], actor["m8_children_t"]], dim=0)
                    .cpu()
                    .numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                row = evaluate_actor_surface(
                    baseline,
                    output,
                    actor["target"],
                    actor["target_sensor_origins"],
                    hazardous=bool(actor["hazardous"]),
                    device=device,
                    lateral_tolerance_m=float(
                        config["evaluation"]["literal_lateral_tolerance_m"]
                    ),
                    depth_tolerance_m=float(
                        config["evaluation"]["literal_depth_tolerance_m"]
                    ),
                    distance_chunk_size=int(
                        config["evaluation"]["distance_chunk_size"]
                    ),
                )
                if isinstance(model, RayTerminationLogitField):
                    partition = _categorical_first_return_partition(
                        model, actor, config["evaluation"], device
                    )
                elif isinstance(model, RaySurvivalDensityField):
                    partition = _survival_first_return_partition(
                        model, actor, config["evaluation"], device
                    )
                else:
                    partition = _field_first_return_partition(
                        model, actor, config["evaluation"], device
                    )
                row.update(
                    {
                        "scene_name": str(actor["scene_name"]),
                        "track_id": str(actor["track_id"]),
                        "field_early_count": int(np.count_nonzero(partition["early"])),
                        "field_hit_count": int(np.count_nonzero(partition["hit"])),
                        "field_observable_count": int(
                            np.count_nonzero(partition["observable"])
                        ),
                    }
                )
                if isinstance(model, OneSidedLocalOccupancyField):
                    row["mean_field_radius_m"] = float(
                        model.child_radii(
                            actor["features"], actor["m8_scales_t"]
                        ).mean()
                    )
                rows.append(row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m13_train_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        field_metrics = _field_summary(rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        hazard_reduction = field_metrics["hazard"]["relative_early_reduction"]
        decisions["hazard_local_field_relative_reduction"] = (
            hazard_reduction is not None
            and float(hazard_reduction)
            >= float(config["decision"]["minimum_field_hazard_relative_reduction"])
        )
        hit_delta = field_metrics["all"]["hit_recall_delta"]
        decisions["local_field_hit_recall_retained"] = (
            hit_delta is not None
            and float(hit_delta)
            >= float(config["decision"]["minimum_field_hit_recall_delta"])
        )
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_LOCAL_FIELD.jsonl", rows)
        result_label = str(config.get("result_label", "m13"))
        summary = {
            "schema_version": str(
                config.get(
                    "schema_version", "worldsim_v71.m13_local_signed_field.v1"
                )
            ),
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": f"{result_label}_development_passed"
            if passed
            else f"{result_label}_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "m8_point_surface": metrics,
            "local_signed_field": field_metrics,
            "ray_survival": field_metrics
            if isinstance(model, RaySurvivalDensityField)
            else None,
            "categorical_first_return": field_metrics
            if isinstance(model, RayTerminationLogitField)
            else None,
            "decisions": decisions,
            "m8_guidance_frozen": True,
            "field_inputs": "query_local_build_geometry_no_trajectory_time_hazard_image",
            "surface_supervision": str(
                config.get(
                    "surface_supervision",
                    "gt_front_hit_narrow_back_and_hit_gradient_normal",
                )
            ),
            "deployment": str(
                config.get(
                    "deployment_description",
                    "first_positive_to_nonpositive_zero_crossing_in_actor_aabb",
                )
            ),
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        m6_runner._write_json(run_dir / "summary.json", summary)
        m6_runner._write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "train_holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        m6_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m13", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

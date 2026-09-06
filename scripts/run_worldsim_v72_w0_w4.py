"""在冻结 G0/G2 几何上比较 W0--W4 categorical return 权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m13_local_signed_field as field_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
import run_worldsim_v71_m37_supervised_child_transmittance as m37_runner
from scripts.run_worldsim_v72_g0_raw_fusion import _moving


ARM_NAMES = (
    "w0_unit",
    "w1_build_support",
    "w2_density_opportunity",
    "w3_scalar_response",
    "w4_fou_response_only",
    "w4_fou_auxiliary",
)

FEATURE_NAMES = (
    "local_x_over_half_length",
    "local_y_over_half_width",
    "local_z_over_half_height",
    "abs_local_x_over_half_length",
    "abs_local_y_over_half_width",
    "abs_local_z_over_half_height",
    "build_free_mass",
    "build_occupied_mass",
    "build_unknown_mass",
    "log_build_opportunity_fraction",
    "log_local_neighbor_fraction",
    "type_raw",
    "type_anchor",
    "type_child",
    "log_actor_length",
    "log_actor_width",
    "log_actor_height",
    "scale_over_actor_diagonal",
)


class PrimitiveWeightMLP(nn.Module):
    """相同 primitive/context 主干；W3 输出标量，W4 输出 F/O/U。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        context = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        return self.head(torch.cat([encoded, context], dim=1))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else value)


def _load_cohort(path: Path) -> set[tuple[str, str]]:
    return {
        (str(row["scene_name"]), str(row["track_id"]))
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _load_build_actor(
    path: Path,
    standardizer: Any,
    device: torch.device,
) -> dict[str, Any] | None:
    keys = (
        "base_features",
        "candidates",
        "size_lwh_m",
        "evidence_masses",
        "evidence_opportunities",
        "query_sensor_origin",
        "anchors",
        "canonical",
        "scene_name",
        "track_id",
        "category",
        "hazardous",
        "trajectory_xyz_m",
    )
    with np.load(path, allow_pickle=False) as payload:
        build = {key: payload[key] for key in keys}
    if len(build["candidates"]) == 0 or len(build["canonical"]) == 0:
        return None
    features, rays, normals = m0_runner._raw_features(build, device)
    return {
        **build,
        "path": str(path),
        "features": torch.as_tensor(
            standardizer.transform(features), dtype=torch.float32, device=device
        ),
        "candidates_t": torch.as_tensor(
            build["candidates"], dtype=torch.float32, device=device
        ),
        "anchors_t": torch.as_tensor(
            build["anchors"], dtype=torch.float32, device=device
        ),
        "ray_directions_t": torch.as_tensor(rays, dtype=torch.float32, device=device),
        "normals_t": torch.as_tensor(normals, dtype=torch.float32, device=device),
        "size_t": torch.as_tensor(
            build["size_lwh_m"], dtype=torch.float32, device=device
        ),
    }


def _load_sidecar_build(actor: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    path = (
        root
        / "train"
        / _scalar_text(actor["scene_name"])
        / f"{_scalar_text(actor['track_id'])}.npz"
    )
    keys = (
        "anchors",
        "input_canonical_surface_indices",
        "input_build_evidence_masses",
        "input_build_evidence_opportunities",
    )
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in keys}


def _load_sidecar_supervision(
    actor: Mapping[str, Any], root: Path
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        root
        / "train"
        / _scalar_text(actor["scene_name"])
        / f"{_scalar_text(actor['track_id'])}.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["supervision_evidence_masses"], dtype=np.float32),
            np.asarray(payload["input_canonical_surface_indices"], dtype=np.int64),
        )


def _aggregate_by_index(
    values: torch.Tensor,
    indices: torch.Tensor,
    output_count: int,
    default: torch.Tensor,
) -> torch.Tensor:
    values = values.reshape(len(indices), -1)
    output = torch.zeros(
        (output_count, values.shape[1]), dtype=values.dtype, device=values.device
    )
    counts = torch.zeros((output_count, 1), dtype=values.dtype, device=values.device)
    output.index_add_(0, indices, values)
    counts.index_add_(
        0, indices, torch.ones((len(indices), 1), dtype=values.dtype, device=values.device)
    )
    averaged = output / counts.clamp_min(1.0)
    return torch.where(counts > 0, averaged, default.reshape(1, -1))


def _build_geometry(
    actor: dict[str, Any],
    geometry_source: str,
    sidecar_root: Path,
    surface: nn.Module,
    base: nn.Module,
    surface_config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    sidecar = _load_sidecar_build(actor, sidecar_root)
    sidecar_anchors = torch.as_tensor(
        sidecar["anchors"], dtype=torch.float32, device=device
    )
    if sidecar_anchors.shape != actor["anchors_t"].shape or not torch.allclose(
        sidecar_anchors, actor["anchors_t"], atol=1.0e-5, rtol=0.0
    ):
        raise RuntimeError("sidecar anchors 与 Actor cache 不一致")
    anchor_masses = torch.as_tensor(
        sidecar["input_build_evidence_masses"], dtype=torch.float32, device=device
    )
    anchor_opportunities = torch.as_tensor(
        sidecar["input_build_evidence_opportunities"],
        dtype=torch.float32,
        device=device,
    )
    if geometry_source == "g0":
        centers = torch.as_tensor(
            actor["canonical"], dtype=torch.float32, device=device
        )
        indices = torch.as_tensor(
            sidecar["input_canonical_surface_indices"], dtype=torch.long, device=device
        )
        actor["canonical_mapping_fallback"] = False
        actor["canonical_mapping_fallback_max_distance_m"] = 0.0
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= len(centers))):
            nearest = torch.cdist(
                sidecar_anchors,
                centers,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            distances, indices = nearest.min(dim=1)
            actor["canonical_mapping_fallback"] = True
            actor["canonical_mapping_fallback_max_distance_m"] = float(
                distances.max()
            )
        masses = _aggregate_by_index(
            anchor_masses,
            indices,
            len(centers),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
        )
        opportunities = _aggregate_by_index(
            anchor_opportunities[:, None],
            indices,
            len(centers),
            torch.zeros(1, dtype=torch.float32, device=device),
        ).reshape(-1)
        scales = torch.full(
            (len(centers),),
            float(config["geometry"]["anchor_scale_m"]),
            dtype=torch.float32,
            device=device,
        )
        primitive_types = torch.zeros((len(centers), 3), device=device)
        primitive_types[:, 0] = 1.0
    elif geometry_source == "g2":
        actor["canonical_mapping_fallback"] = False
        actor["canonical_mapping_fallback_max_distance_m"] = 0.0
        with torch.inference_mode():
            _, moved = m5_runner._move(base, actor, base_config)
            actor["m5_centers_t"] = moved
            children, residuals, child_scales = m7_runner._predict(
                surface, actor, surface_config
            )
        parent_count = len(actor["candidates_t"])
        branch_factor = len(children) // max(parent_count, 1)
        if branch_factor != 4 or parent_count * branch_factor != len(children):
            raise RuntimeError("G2 W0--W4 需要冻结 M8 四子点顺序")
        child_masses = torch.as_tensor(
            actor["evidence_masses"], dtype=torch.float32, device=device
        ).repeat_interleave(branch_factor, dim=0)
        child_opportunities = torch.as_tensor(
            actor["evidence_opportunities"], dtype=torch.float32, device=device
        ).repeat_interleave(branch_factor)
        centers = torch.cat([actor["anchors_t"], children], dim=0)
        scales = torch.cat(
            [
                torch.full(
                    (len(actor["anchors_t"]),),
                    float(config["geometry"]["anchor_scale_m"]),
                    dtype=torch.float32,
                    device=device,
                ),
                child_scales.reshape(-1).clamp_min(1.0e-4),
            ]
        )
        masses = torch.cat([anchor_masses, child_masses], dim=0)
        opportunities = torch.cat([anchor_opportunities, child_opportunities], dim=0)
        primitive_types = torch.zeros((len(centers), 3), device=device)
        primitive_types[: len(actor["anchors_t"]), 1] = 1.0
        primitive_types[len(actor["anchors_t"]) :, 2] = 1.0
        actor["m8_children_t"] = children
        actor["m8_residuals_t"] = residuals
        actor["m8_scales_t"] = child_scales
    else:
        raise ValueError(f"未知 geometry_source={geometry_source}")

    actor["authority_centers_t"] = centers
    actor["authority_scales_t"] = scales
    actor["build_masses_t"] = masses
    actor["build_opportunities_t"] = opportunities
    actor["primitive_types_t"] = primitive_types
    actor["sidecar_canonical_indices"] = sidecar["input_canonical_surface_indices"]
    _attach_features_and_fixed_weights(actor, config)


def _attach_features_and_fixed_weights(
    actor: dict[str, Any], config: Mapping[str, Any]
) -> None:
    centers = actor["authority_centers_t"]
    size = actor["size_t"].reshape(3).clamp_min(0.10)
    half = 0.5 * size
    diagonal = torch.linalg.vector_norm(size).clamp_min(1.0e-6)
    normalized = centers / half.reshape(1, 3)
    distance = torch.cdist(
        centers,
        centers,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    local_neighbors = (
        distance <= float(config["features"]["local_density_radius_m"])
    ).sum(dim=1).to(torch.float32)
    maximum_opportunities = float(config["features"]["maximum_build_opportunities"])
    actor_size = torch.log1p(size).reshape(1, 3).expand(len(centers), -1)
    features = torch.cat(
        [
            normalized,
            normalized.abs(),
            actor["build_masses_t"],
            torch.log1p(actor["build_opportunities_t"][:, None])
            / np.log1p(maximum_opportunities),
            torch.log1p(local_neighbors[:, None]) / np.log1p(max(len(centers), 2)),
            actor["primitive_types_t"],
            actor_size,
            actor["authority_scales_t"][:, None] / diagonal,
        ],
        dim=1,
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("W0--W4 feature dimension 不符合冻结 schema")
    w1 = actor["build_masses_t"][:, 1].clamp_min(1.0e-4)
    confidence = torch.sqrt(
        actor["build_opportunities_t"]
        / (actor["build_opportunities_t"] + float(config["features"]["opportunity_prior"]))
    )
    w2 = w1 * confidence / torch.sqrt(local_neighbors.clamp_min(1.0))
    w2 = w2 * (w1.mean() / w2.mean().clamp_min(1.0e-6))
    actor["raw_weight_features_t"] = features
    actor["fixed_weights"] = {
        "w0_unit": torch.ones_like(w1),
        "w1_build_support": w1,
        "w2_density_opportunity": w2.clamp_min(1.0e-4),
    }


def _fit_feature_standardizer(
    train_actors: list[dict[str, Any]], actors: list[dict[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    training = torch.cat(
        [actor["raw_weight_features_t"] for actor in train_actors], dim=0
    )
    mean = training.mean(dim=0)
    std = training.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    for actor in actors:
        actor["weight_features_t"] = (actor["raw_weight_features_t"] - mean) / std
    return mean, std


def _attach_target(actor: dict[str, Any]) -> None:
    with np.load(actor["path"], allow_pickle=False) as payload:
        actor["target"] = np.asarray(payload["target"], dtype=np.float32)
        actor["target_sensor_origins"] = np.asarray(
            payload["target_sensor_origins"], dtype=np.float32
        )


def _attach_auxiliary_targets(
    actor: dict[str, Any],
    geometry_source: str,
    sidecar_root: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    anchor_targets, canonical_indices = _load_sidecar_supervision(actor, sidecar_root)
    anchor_targets_t = torch.as_tensor(
        anchor_targets, dtype=torch.float32, device=device
    )
    if geometry_source == "g0":
        targets = _aggregate_by_index(
            anchor_targets_t,
            torch.as_tensor(canonical_indices, dtype=torch.long, device=device),
            len(actor["authority_centers_t"]),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
        )
    else:
        child_targets, stats = m37_runner._child_evidence_targets(
            actor, config["auxiliary_evidence"]
        )
        actor["child_target_stats"] = stats
        targets = torch.cat([anchor_targets_t, child_targets], dim=0)
    actor["auxiliary_target_masses_t"] = targets


def _training_context(
    actor: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    maximum_rays = int(config["training"]["maximum_rays_per_actor"])
    if len(targets) > maximum_rays:
        indices = torch.linspace(0, len(targets) - 1, steps=maximum_rays, device=device)
        indices = torch.unique(indices.round().long())
        targets = targets.index_select(0, indices)
        origins = origins.index_select(0, indices)
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["reader"]["cuboid_padding_m"])
    entry, exit_depth, valid = field_runner._ray_box_intervals(origins, directions, bounds)
    valid = valid & (target_depth >= entry) & (target_depth <= exit_depth)
    origins = origins[valid]
    directions = directions[valid]
    target_depth = target_depth[valid]
    entry = entry[valid]
    exit_depth = exit_depth[valid]
    if len(target_depth) == 0:
        raise RuntimeError("训练 Actor 没有有效 actor-box target rays")
    samples = int(config["training"]["categorical_samples"])
    fractions = torch.linspace(0.0, 1.0, samples, device=device)
    depths = entry[:, None] + (exit_depth - entry)[:, None] * fractions[None, :]
    queries = origins[:, None, :] + depths[:, :, None] * directions[:, None, :]
    normalized_distance = torch.cdist(
        queries.reshape(-1, 3),
        actor["authority_centers_t"],
        compute_mode="donot_use_mm_for_euclid_dist",
    ) / actor["authority_scales_t"].reshape(1, -1)
    kernel = (-0.5 * normalized_distance.square()).reshape(
        len(target_depth), samples, -1
    )
    return {
        "kernel": kernel.to(torch.float16),
        "depths": depths,
        "target_depth": target_depth,
        "target_bins": torch.abs(depths - target_depth[:, None]).argmin(dim=1),
    }


def _model_weights(model: nn.Module, actor: Mapping[str, Any], output_dim: int) -> torch.Tensor:
    logits = model(actor["weight_features_t"])
    if output_dim == 1:
        return torch.sigmoid(logits[:, 0]).clamp_min(1.0e-4)
    return torch.softmax(logits, dim=1)[:, 1].clamp_min(1.0e-4)


def _response_losses(
    model: nn.Module,
    actor: Mapping[str, Any],
    output_dim: int,
    auxiliary_weight: float,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    context = actor["train_context"]
    logits = model(actor["weight_features_t"])
    weights = (
        torch.sigmoid(logits[:, 0])
        if output_dim == 1
        else torch.softmax(logits, dim=1)[:, 1]
    ).clamp_min(1.0e-4)
    energy = torch.logsumexp(
        context["kernel"].to(torch.float32)
        + torch.log(weights)[None, None, :],
        dim=2,
    )
    log_probabilities = F.log_softmax(energy, dim=1)
    categorical = F.nll_loss(log_probabilities, context["target_bins"])
    probabilities = torch.exp(log_probabilities)
    expected_depth = torch.sum(probabilities * context["depths"], dim=1)
    depth_l1 = torch.abs(expected_depth - context["target_depth"]).mean()
    auxiliary = torch.zeros((), dtype=categorical.dtype, device=categorical.device)
    if auxiliary_weight > 0.0:
        if output_dim != 3:
            raise RuntimeError("F/O/U auxiliary 只能用于三输出 head")
        target_masses = actor["auxiliary_target_masses_t"]
        auxiliary = -torch.sum(
            target_masses * torch.log_softmax(logits, dim=1), dim=1
        ).mean()
    loss = (
        categorical
        + float(config["training"]["depth_l1_weight"]) * depth_l1
        + auxiliary_weight * auxiliary
    )
    return {
        "loss": loss,
        "categorical_nll": categorical,
        "depth_l1_m": depth_l1,
        "auxiliary_cross_entropy": auxiliary,
    }


def _train_models(
    models: Mapping[str, nn.Module],
    train_actors: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output_dims = {
        "w3_scalar_response": 1,
        "w4_fou_response_only": 3,
        "w4_fou_auxiliary": 3,
    }
    auxiliary_weights = {
        "w3_scalar_response": 0.0,
        "w4_fou_response_only": 0.0,
        "w4_fou_auxiliary": float(config["training"]["fou_auxiliary_weight"]),
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        for name, model in models.items()
    }
    history: list[dict[str, Any]] = []
    batch_size = int(config["training"]["actor_batch_size"])
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        permutation = torch.randperm(len(train_actors)).tolist()
        totals = {
            name: {
                "loss": 0.0,
                "categorical_nll": 0.0,
                "depth_l1_m": 0.0,
                "auxiliary_cross_entropy": 0.0,
            }
            for name in models
        }
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            for name, model in models.items():
                optimizer = optimizers[name]
                items = [
                    _response_losses(
                        model,
                        train_actors[index],
                        output_dims[name],
                        auxiliary_weights[name],
                        config,
                    )
                    for index in indices
                ]
                means = {
                    key: torch.stack([item[key] for item in items]).mean()
                    for key in totals[name]
                }
                optimizer.zero_grad(set_to_none=True)
                means["loss"].backward()
                optimizer.step()
                for key, value in means.items():
                    totals[name][key] += float(value.detach()) * len(indices)
        for name in models:
            row = {
                "stage": "w0_w4_train",
                "epoch": epoch,
                "arm": name,
                **{
                    key: value / len(train_actors)
                    for key, value in totals[name].items()
                },
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return history


def _predict_holdout_weights(
    actor: dict[str, Any], models: Mapping[str, nn.Module]
) -> dict[str, torch.Tensor]:
    weights = dict(actor["fixed_weights"])
    with torch.inference_mode():
        weights["w3_scalar_response"] = _model_weights(
            models["w3_scalar_response"], actor, 1
        )
        weights["w4_fou_response_only"] = _model_weights(
            models["w4_fou_response_only"], actor, 3
        )
        weights["w4_fou_auxiliary"] = _model_weights(
            models["w4_fou_auxiliary"], actor, 3
        )
    return weights


def _evaluate_actor(
    actor: Mapping[str, Any],
    weights_by_arm: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["reader"]["cuboid_padding_m"])
    entry, exit_depth, valid = field_runner._ray_box_intervals(origins, directions, bounds)
    predicted = {
        name: torch.empty_like(target_depth) for name in weights_by_arm
    }
    samples = int(config["evaluation"]["categorical_samples"])
    fractions = torch.linspace(0.0, 1.0, samples, device=device)
    chunk = int(config["evaluation"]["ray_chunk_size"])
    for start in range(0, len(targets), chunk):
        local_entry = entry[start : start + chunk]
        local_exit = exit_depth[start : start + chunk]
        depths = local_entry[:, None] + (
            local_exit - local_entry
        )[:, None] * fractions[None, :]
        queries = (
            origins[start : start + chunk, None, :]
            + depths[:, :, None] * directions[start : start + chunk, None, :]
        )
        normalized_distance = torch.cdist(
            queries.reshape(-1, 3),
            actor["authority_centers_t"],
            compute_mode="donot_use_mm_for_euclid_dist",
        ) / actor["authority_scales_t"].reshape(1, -1)
        kernel = (-0.5 * normalized_distance.square()).reshape(
            len(local_entry), samples, -1
        )
        for name, weights in weights_by_arm.items():
            energy = torch.logsumexp(
                kernel + torch.log(weights.clamp_min(1.0e-4))[None, None, :],
                dim=2,
            )
            probabilities = torch.softmax(energy, dim=1)
            cdf = probabilities.cumsum(dim=1)
            indices = (cdf >= float(config["reader"]["median_threshold"])).to(
                torch.int64
            ).argmax(dim=1)
            predicted[name][start : start + chunk] = depths.gather(
                1, indices[:, None]
            ).squeeze(1)
    tolerance = float(config["evaluation"]["depth_tolerance_m"])
    output: dict[str, dict[str, Any]] = {}
    for name, depth in predicted.items():
        early = valid & (depth < target_depth - tolerance)
        hit = valid & (torch.abs(depth - target_depth) <= tolerance)
        late = valid & (depth > target_depth + tolerance)
        output[name] = {
            "ray_count": int(len(targets)),
            "observable_count": int(torch.count_nonzero(valid)),
            "early_count": int(torch.count_nonzero(early)),
            "hit_count": int(torch.count_nonzero(hit)),
            "late_count": int(torch.count_nonzero(late)),
            "miss_count": int(torch.count_nonzero(~valid)),
            "absolute_depth_error_sum_m": float(
                torch.abs(depth[valid] - target_depth[valid]).sum()
            ),
        }
    return output


def _stratum(rows: list[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rays = sum(int(row["arms"][arm]["ray_count"]) for row in rows)
    observable = sum(int(row["arms"][arm]["observable_count"]) for row in rows)
    result = {
        "actor_count": len(rows),
        "log_count": len({str(row["log_id"]) for row in rows}),
        "ray_count": rays,
        "mean_surface_point_count": float(
            np.mean([int(row["surface_point_count"]) for row in rows])
        ),
    }
    for key, label in (
        ("observable_count", "observable_rate"),
        ("early_count", "early_rate"),
        ("hit_count", "hit_recall"),
        ("late_count", "late_rate"),
        ("miss_count", "miss_rate"),
    ):
        result[label] = sum(int(row["arms"][arm][key]) for row in rows) / max(rays, 1)
    result["mean_absolute_depth_error_m"] = sum(
        float(row["arms"][arm]["absolute_depth_error_sum_m"]) for row in rows
    ) / max(observable, 1)
    return result


def _log_macro(
    rows: list[Mapping[str, Any]], arm: str, seed: int, samples: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["log_id"])].append(row)
    logs = [
        {"log_id": log_id, **_stratum(items, arm)}
        for log_id, items in sorted(grouped.items())
    ]
    rng = np.random.default_rng(int(seed))
    output: dict[str, Any] = {"log_count": len(logs), "cluster_unit": "driving_log"}
    output["metrics"] = {}
    for key in ("early_rate", "hit_recall", "mean_absolute_depth_error_m"):
        values = np.asarray([float(row[key]) for row in logs], dtype=np.float64)
        indices = rng.integers(0, len(values), size=(int(samples), len(values)))
        estimates = values[indices].mean(axis=1)
        output["metrics"][key] = {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
        }
    return logs, output


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("data_role") != "legacy_diagnostic":
        raise ValueError("W0--W4 当前只允许 legacy_diagnostic")
    if bool(config.get("source_test_read")) or bool(config.get("external_test_read")):
        raise PermissionError("W0--W4 不允许读取 source/external final")
    geometry_source = str(config["geometry_source"])
    if geometry_source not in {"g0", "g2"}:
        raise ValueError("geometry_source 必须是 g0 或 g2")
    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("W0--W4 需要 CUDA")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "setup"})
    started = time.monotonic()
    try:
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
        surface, base, standardizer, surface_config, base_config = m22_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_source_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := _load_build_actor(path, standardizer, device)) is not None
        ]
        sidecar_root = Path(config["sidecar_root"])
        for index, actor in enumerate(actors):
            _build_geometry(
                actor,
                geometry_source,
                sidecar_root,
                surface,
                base,
                surface_config,
                base_config,
                config,
                device,
            )
            if (index + 1) % 100 == 0 or index + 1 == len(actors):
                print(
                    json.dumps(
                        {"stage": "build_geometry", "progress": f"{index + 1}/{len(actors)}"}
                    ),
                    flush=True,
                )
        stride = int(config["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        expected = _load_cohort(Path(config["cohort_rows"]))
        actual = {
            (_scalar_text(actor["scene_name"]), _scalar_text(actor["track_id"]))
            for actor in holdout_actors
        }
        if actual != expected or len(holdout_actors) != int(config["expected_actor_count"]):
            raise RuntimeError("W0--W4 holdout identity 与冻结 G0--G3 cohort 不一致")
        feature_mean, feature_std = _fit_feature_standardizer(train_actors, actors)

        hidden_dim = int(config["training"]["hidden_dim"])
        torch.manual_seed(int(config["seed"]) + 1)
        models = {
            "w3_scalar_response": PrimitiveWeightMLP(len(FEATURE_NAMES), hidden_dim, 1).to(device),
            "w4_fou_response_only": PrimitiveWeightMLP(len(FEATURE_NAMES), hidden_dim, 3).to(device),
            "w4_fou_auxiliary": PrimitiveWeightMLP(len(FEATURE_NAMES), hidden_dim, 3).to(device),
        }
        for model in models.values():
            model.train()

        _write_json(run_dir / "status.json", {"status": "running", "phase": "train_context"})
        for index, actor in enumerate(train_actors):
            _attach_target(actor)
            _attach_auxiliary_targets(
                actor, geometry_source, sidecar_root, config, device
            )
            actor["train_context"] = _training_context(actor, config, device)
            if (index + 1) % 100 == 0 or index + 1 == len(train_actors):
                print(
                    json.dumps(
                        {"stage": "precompute_train_rays", "progress": f"{index + 1}/{len(train_actors)}"}
                    ),
                    flush=True,
                )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
        history = _train_models(models, train_actors, config)
        _write_jsonl(run_dir / "TRAIN.jsonl", history)
        for name, model in models.items():
            model.eval()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": len(FEATURE_NAMES),
                    "hidden_dim": hidden_dim,
                    "output_dim": 1 if name == "w3_scalar_response" else 3,
                    "feature_names": FEATURE_NAMES,
                    "feature_mean": feature_mean.cpu(),
                    "feature_std": feature_std.cpu(),
                    "geometry_source": geometry_source,
                    "seed": int(config["seed"]),
                    "training_loss": "categorical_response"
                    + ("+fou_auxiliary" if name == "w4_fou_auxiliary" else ""),
                },
                run_dir / f"MODEL_{name.upper()}.pt",
            )
        for actor in train_actors:
            actor.pop("train_context", None)
        torch.cuda.empty_cache()

        # 先完成 target-free 权重推理，再挂载 holdout target 进入 evaluator。
        holdout_weights = [
            _predict_holdout_weights(actor, models) for actor in holdout_actors
        ]
        target_free_holdout_inference_complete = True
        scene_to_log = {
            str(row["name"]): str(row["log_token"])
            for row in json.loads(
                Path(config["scene_metadata"]).read_text(encoding="utf-8")
            )
        }
        rows: list[dict[str, Any]] = []
        _write_json(run_dir / "status.json", {"status": "running", "phase": "holdout"})
        with torch.inference_mode():
            for index, (actor, weights) in enumerate(zip(holdout_actors, holdout_weights)):
                _attach_target(actor)
                moving, displacement = _moving(actor["trajectory_xyz_m"])
                scene_name = _scalar_text(actor["scene_name"])
                rows.append(
                    {
                        "scene_name": scene_name,
                        "log_id": scene_to_log[scene_name],
                        "track_id": _scalar_text(actor["track_id"]),
                        "category": _scalar_text(actor["category"]),
                        "hazardous": bool(actor["hazardous"]),
                        "moving": moving,
                        "trajectory_max_displacement_m": displacement,
                        "geometry_source": geometry_source,
                        "surface_point_count": int(len(actor["authority_centers_t"])),
                        "target_used_by_weight_inference": False,
                        "arms": _evaluate_actor(actor, weights, config, device),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "w0_w4_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )

        _write_jsonl(run_dir / "ACTORS.jsonl", rows)
        metrics: dict[str, Any] = {}
        log_rows: list[dict[str, Any]] = []
        for offset, arm in enumerate(ARM_NAMES):
            arm_logs, log_macro = _log_macro(
                rows,
                arm,
                int(config["bootstrap"]["seed"]) + offset,
                int(config["bootstrap"]["samples"]),
            )
            log_rows.extend({"arm": arm, **row} for row in arm_logs)
            metrics[arm] = {
                "all": _stratum(rows, arm),
                "hazard": _stratum([row for row in rows if bool(row["hazardous"])], arm),
                "clear": _stratum([row for row in rows if not bool(row["hazardous"])], arm),
                "moving": _stratum([row for row in rows if bool(row["moving"])], arm),
                "quasi_static": _stratum([row for row in rows if not bool(row["moving"])], arm),
                "log_macro_bootstrap": log_macro,
            }
        _write_jsonl(run_dir / "LOGS.jsonl", log_rows)

        git_commit = _git_commit()
        resolved = {
            **config,
            "run_id": run_id,
            "git_commit": git_commit,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
        }
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema_version": "worldsim_v72.run_manifest.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "git_commit": git_commit,
            "data_role": config["data_role"],
            "geometry_source": geometry_source,
            "training": True,
            "train_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "arms": list(ARM_NAMES),
            "same_feature_schema_w3_w4": True,
            "same_response_loss_w3_w4": True,
            "w4_auxiliary_is_separate_arm": True,
            "canonical_mapping_fallback_actor_count": sum(
                int(bool(actor["canonical_mapping_fallback"])) for actor in actors
            ),
            "target_free_holdout_weight_inference": target_free_holdout_inference_complete,
            "pretrained_holdout_exposure": True,
            "source_test_read": False,
            "external_test_read": False,
            "route_decision_allowed": False,
            "cohort_rows_sha256": _sha256(Path(config["cohort_rows"])),
            "m8_checkpoint_sha256": _sha256(Path(config["m8_run"]) / "MODEL.pt"),
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
        }
        _write_json(run_dir / "manifest.json", manifest)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "config": hashlib.sha256(config_text.encode()).hexdigest(),
                    "git": git_commit,
                    "cohort": manifest["cohort_rows_sha256"],
                    "checkpoint": manifest["m8_checkpoint_sha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        _write_json(
            run_dir / "fingerprint.json",
            {"algorithm": "sha256", "value": fingerprint, "scope": "config+git+cohort+m8"},
        )
        parameter_counts = {
            name: int(sum(parameter.numel() for parameter in model.parameters()))
            for name, model in models.items()
        }
        comparison = {
            "w4_response_minus_w3_early": metrics["w4_fou_response_only"]["all"]["early_rate"]
            - metrics["w3_scalar_response"]["all"]["early_rate"],
            "w4_response_minus_w3_hit": metrics["w4_fou_response_only"]["all"]["hit_recall"]
            - metrics["w3_scalar_response"]["all"]["hit_recall"],
            "w4_aux_minus_response_early": metrics["w4_fou_auxiliary"]["all"]["early_rate"]
            - metrics["w4_fou_response_only"]["all"]["early_rate"],
            "w4_aux_minus_response_hit": metrics["w4_fou_auxiliary"]["all"]["hit_recall"]
            - metrics["w4_fou_response_only"]["all"]["hit_recall"],
        }
        summary = {
            "schema_version": "worldsim_v72.w0_w4.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "git_commit": git_commit,
            "status": "done",
            "verdict": "legacy_weight_ablation_complete_no_route_decision",
            "data_role": config["data_role"],
            "geometry_source": geometry_source,
            "train_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "arm_names": list(ARM_NAMES),
            "parameter_counts": parameter_counts,
            "canonical_mapping_fallback_actor_count": sum(
                int(bool(actor["canonical_mapping_fallback"])) for actor in actors
            ),
            "maximum_canonical_mapping_fallback_distance_m": max(
                float(actor["canonical_mapping_fallback_max_distance_m"])
                for actor in actors
            ),
            "final_train": {name: [row for row in history if row["arm"] == name][-1] for name in models},
            "minimum_train_loss": {
                name: min(float(row["loss"]) for row in history if row["arm"] == name)
                for name in models
            },
            "metrics": metrics,
            "comparison": comparison,
            "target_free_holdout_weight_inference": target_free_holdout_inference_complete,
            "full_return_metrics_supported": False,
            "full_return_metrics_reason": "legacy v1 cohort has positive held-out returns only",
            "pretrained_holdout_exposure": True,
            "source_test_read": False,
            "external_test_read": False,
            "route_decision_allowed": False,
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
            "resources": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "w0_w4",
                "error": f"{type(error).__name__}: {error}",
            },
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

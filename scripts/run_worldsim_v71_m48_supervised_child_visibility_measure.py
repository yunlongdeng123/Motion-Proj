"""Train a ray-conditioned child visibility factor inside the return measure."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m8_runner
import run_worldsim_v71_m11_exact_support_supervision as oriented_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as energy_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as loader_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as anchor_runner
import run_worldsim_v71_m37_supervised_child_transmittance as child_runner
import run_worldsim_v71_m38_prehit_free_space_survival as authority_runner
import run_worldsim_v71_m39_categorical_authority_composition as composition_runner
import run_worldsim_v71_m45_oriented_categorical_surface_measure as m45_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import occupied_masses
from motion_proj.worldsim_v71.gaussian_anchor_relocation import OrientedGaussianSeedExpansionMLP


class ChildVisibilityMLP(torch.nn.Module):
    """Bounded per-ray child visibility from physical ray/component geometry."""

    def __init__(self, hidden_dim: int, initial_visibility: float) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.constant_(
            self.network[-1].bias,
            math.log(initial_visibility / (1.0 - initial_visibility)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features)).squeeze(-1)


def _ray_geometry(
    actor: Mapping[str, Any],
    origins: torch.Tensor,
    directions: torch.Tensor,
    child_normals: torch.Tensor,
    child_thickness: torch.Tensor,
    child_occupied: torch.Tensor,
) -> torch.Tensor:
    displacement = actor["m8_children_t"][None, :, :] - origins[:, None, :]
    longitudinal = torch.sum(displacement * directions[:, None, :], dim=-1)
    lateral = (
        torch.sum(displacement.square(), dim=-1) - longitudinal.square()
    ).clamp_min(0.0).sqrt()
    tangent_scale = actor["m8_scales_t"].reshape(1, -1).clamp_min(1.0e-4)
    incidence = torch.abs(directions @ child_normals.T)
    box_diagonal = torch.linalg.vector_norm(actor["size_t"]).clamp_min(1.0)
    return torch.stack(
        [
            incidence,
            (lateral / tangent_scale).clamp(0.0, 8.0) / 8.0,
            (longitudinal / box_diagonal).clamp(-1.0, 3.0) / 3.0,
            (child_thickness.reshape(1, -1) / tangent_scale)
            .clamp(0.0, 1.0)
            .expand(len(origins), -1),
            child_occupied.reshape(1, -1).expand(len(origins), -1),
        ],
        dim=-1,
    )


def _visibility(
    model: ChildVisibilityMLP,
    actor: Mapping[str, Any],
    origins: torch.Tensor,
    directions: torch.Tensor,
    child_normals: torch.Tensor,
    child_thickness: torch.Tensor,
    child_occupied: torch.Tensor,
) -> torch.Tensor:
    return model(
        _ray_geometry(
            actor,
            origins,
            directions,
            child_normals,
            child_thickness,
            child_occupied,
        )
    )


def _component_energy(
    queries: torch.Tensor,
    actor: Mapping[str, Any],
    anchor_occupied: torch.Tensor,
    child_occupied: torch.Tensor,
    child_normals: torch.Tensor,
    child_thickness: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_scales = actor["authority_scales_t"][: len(actor["anchors_t"])]
    anchor_components = (
        -0.5
        * (torch.cdist(queries, actor["anchors_t"]) / anchor_scales.reshape(1, -1)).square()
        + torch.log(anchor_occupied.clamp_min(1.0e-8)).reshape(1, -1)
    )
    displacement = queries[:, None, :] - actor["m8_children_t"][None, :, :]
    normal_coordinate = torch.sum(displacement * child_normals[None, :, :], dim=-1)
    tangent_sq = (
        torch.sum(displacement.square(), dim=-1) - normal_coordinate.square()
    ).clamp_min(0.0)
    child_components = -0.5 * (
        tangent_sq / actor["m8_scales_t"].square().reshape(1, -1)
        + normal_coordinate.square() / child_thickness.square().reshape(1, -1)
    ) + torch.log(child_occupied.clamp_min(1.0e-8)).reshape(1, -1)
    return anchor_components, child_components


def _training_values(
    model: ChildVisibilityMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = actor["features"].device
    targets, origins = energy_runner.m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = anchor_runner.field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    origins = origins[valid]
    directions = directions[valid]
    target_depth = target_depth[valid]
    fractions = torch.linspace(
        0.0,
        1.0,
        int(config["categorical_train_bins"]),
        dtype=torch.float32,
        device=device,
    )
    depths = entry[valid, None] + (exit_depth[valid] - entry[valid])[:, None] * fractions[None, :]
    queries = origins[:, None, :] + depths[:, :, None] * directions[:, None, :]
    ray_count, bin_count = depths.shape
    with torch.no_grad():
        anchor_components, child_components = _component_energy(
            queries.reshape(-1, 3),
            actor,
            actor["fixed_anchor_occupied_t"],
            actor["fixed_child_occupied_t"],
            actor["fixed_child_normals_t"],
            actor["fixed_child_thickness_t"],
        )
        anchor_components = anchor_components.reshape(ray_count, bin_count, -1)
        child_components = child_components.reshape(ray_count, bin_count, -1)
    visibility = _visibility(
        model,
        actor,
        origins,
        directions,
        actor["fixed_child_normals_t"],
        actor["fixed_child_thickness_t"],
        actor["fixed_child_occupied_t"],
    )
    energy = torch.logsumexp(
        torch.cat(
            [
                anchor_components,
                child_components + torch.log(visibility.clamp_min(1.0e-8))[:, None, :],
            ],
            dim=2,
        ),
        dim=2,
    )
    return depths, target_depth, energy, visibility


def _losses(
    model: ChildVisibilityMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, energy, visibility = _training_values(model, actor, config)
    probabilities = torch.softmax(energy, dim=1)
    tolerance = float(config["interval_tolerance_m"])
    early_probability = torch.sum(
        probabilities * (depths < target_depth[:, None] - tolerance), dim=1
    )
    hit_probability = torch.sum(
        probabilities * (torch.abs(depths - target_depth[:, None]) <= tolerance), dim=1
    )
    safe_nll = -torch.log((1.0 - early_probability).clamp_min(1.0e-8)).mean()
    hit_nll = -torch.log(hit_probability.clamp_min(1.0e-8)).mean()
    identity = torch.mean((1.0 - visibility).square())
    loss = (
        float(config["not_early_weight"]) * safe_nll
        + float(config["hit_band_weight"]) * hit_nll
        + float(config["identity_weight"]) * identity
    )
    return {
        "loss": loss,
        "not_early_nll": safe_nll,
        "hit_band_nll": hit_nll,
        "identity_loss": identity,
        "early_probability": early_probability.mean(),
        "hit_probability": hit_probability.mean(),
        "mean_visibility": visibility.mean(),
    }


def _train(
    model: ChildVisibilityMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    names = (
        "loss",
        "not_early_nll",
        "hit_band_nll",
        "identity_loss",
        "early_probability",
        "hit_probability",
        "mean_visibility",
    )
    history = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [_losses(model, actors[index], config) for index in indices]
            means = {name: torch.stack([item[name] for item in items]).mean() for name in names}
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row = {"epoch": epoch + 1, **{name: totals[name] / len(actors) for name in names}}
        history.append(row)
        print(json.dumps({"stage": "m48_visibility_train", **row}), flush=True)
    return history


def _partition(
    model: ChildVisibilityMLP,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    device = actor["features"].device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(actor["target_sensor_origins"], dtype=torch.float32, device=device)
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = anchor_runner.field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    fractions = torch.linspace(
        0.0, 1.0, int(evaluation["field_sample_count"]), dtype=torch.float32, device=device
    )
    first_depths = []
    visibility_means = []
    threshold = float(evaluation["categorical_median_threshold"])
    chunk = int(evaluation["ray_chunk_size"])
    for start in range(0, len(targets), chunk):
        local_origins = origins[start : start + chunk]
        local_directions = directions[start : start + chunk]
        local_entry = entry[start : start + chunk]
        local_exit = exit_depth[start : start + chunk]
        depths = local_entry[:, None] + (local_exit - local_entry)[:, None] * fractions[None, :]
        queries = local_origins[:, None, :] + depths[:, :, None] * local_directions[:, None, :]
        ray_count, bin_count = depths.shape
        anchor_components, child_components = _component_energy(
            queries.reshape(-1, 3),
            actor,
            actor["fixed_anchor_occupied_t"],
            actor["fixed_child_occupied_t"],
            actor["fixed_child_normals_t"],
            actor["fixed_child_thickness_t"],
        )
        visibility = _visibility(
            model,
            actor,
            local_origins,
            local_directions,
            actor["fixed_child_normals_t"],
            actor["fixed_child_thickness_t"],
            actor["fixed_child_occupied_t"],
        )
        energy = torch.logsumexp(
            torch.cat(
                [
                    anchor_components.reshape(ray_count, bin_count, -1),
                    child_components.reshape(ray_count, bin_count, -1)
                    + torch.log(visibility.clamp_min(1.0e-8))[:, None, :],
                ],
                dim=2,
            ),
            dim=2,
        )
        cdf = torch.softmax(energy, dim=1).cumsum(dim=1)
        indices = (cdf >= threshold).to(torch.int64).argmax(dim=1)
        previous = (indices - 1).clamp_min(0)
        right_cdf = cdf.gather(1, indices[:, None]).squeeze(1)
        gathered_left_cdf = cdf.gather(1, previous[:, None]).squeeze(1)
        left_cdf = torch.where(indices > 0, gathered_left_cdf, torch.zeros_like(gathered_left_cdf))
        right_depth = depths.gather(1, indices[:, None]).squeeze(1)
        gathered_left_depth = depths.gather(1, previous[:, None]).squeeze(1)
        left_depth = torch.where(indices > 0, gathered_left_depth, local_entry)
        ratio = (threshold - left_cdf) / (right_cdf - left_cdf).clamp_min(1.0e-6)
        first_depths.append(left_depth + ratio.clamp(0.0, 1.0) * (right_depth - left_depth))
        visibility_means.append(visibility.mean())
    first_depth = torch.cat(first_depths)
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "observable": valid_box.cpu().numpy().astype(bool),
        "early": (valid_box & (first_depth < target_depth - tolerance)).cpu().numpy(),
        "hit": (valid_box & (torch.abs(first_depth - target_depth) <= tolerance)).cpu().numpy(),
        "mean_visibility": np.asarray(float(torch.stack(visibility_means).mean())),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {"actor_count": len(selected), "ray_count": rays}
        for name in ("baseline", "m39", "m45", "m48"):
            output[f"{name}_early_rate"] = sum(int(row[f"{name}_early_count"]) for row in selected) / rays
            output[f"{name}_hit_rate"] = sum(int(row[f"{name}_hit_count"]) for row in selected) / rays
        for reference in ("m39", "m45"):
            output[f"m48_vs_{reference}_early_delta"] = output["m48_early_rate"] - output[f"{reference}_early_rate"]
            output[f"m48_vs_{reference}_hit_delta"] = output["m48_hit_rate"] - output[f"{reference}_hit_rate"]
        return output
    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M48 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = loader_runner._load_m8(config, device)
        actors = [
            actor
            for path in m0_runner._paths(Path(config["cache_root"]), int(config["maximum_training_actors"]))
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, actor["m5_centers_t"] = m5_runner._move(base, actor, base_config)
                actor["m8_children_t"], actor["m8_residuals_t"], actor["m8_scales_t"] = m8_runner._predict(
                    surface, actor, surface_config
                )
        for actor in actors:
            anchor_runner._attach_frozen_authority_state(
                actor, Path(config["sidecar_root"]), float(config["anchor_scale_m"]), config["features"], device
            )
            actor["authority_child_features_t"] = child_runner._child_features(actor)
        anchor_checkpoint = torch.load(Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        anchor_authority = authority_runner._load_authority(anchor_checkpoint, device)
        anchor_authority.eval().requires_grad_(False)
        child_checkpoint = torch.load(Path(config["m38_run"]) / "CHILD_MODEL.pt", map_location=device, weights_only=False)
        child_authority = authority_runner._load_authority(child_checkpoint, device)
        child_authority.eval().requires_grad_(False)
        support_checkpoint = torch.load(Path(config["m11_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        support_model = OrientedGaussianSeedExpansionMLP(
            int(support_checkpoint["input_dim"]), int(support_checkpoint["hidden_dim"]),
            int(support_checkpoint["branch_factor"]), int(support_checkpoint["slot_dim"]),
        ).to(device)
        support_model.load_state_dict(support_checkpoint["state_dict"])
        support_model.eval().requires_grad_(False)
        support_config = yaml.safe_load((Path(config["m11_run"]) / "resolved.yaml").read_text(encoding="utf-8"))["model"]
        with torch.inference_mode():
            for actor in actors:
                actor["fixed_anchor_occupied_t"] = occupied_masses(anchor_authority(actor["authority_anchor_features_t"]))
                actor["fixed_child_occupied_t"] = occupied_masses(child_authority(actor["authority_child_features_t"]))
                _, _, actor["fixed_child_normals_t"], actor["fixed_child_thickness_t"] = oriented_runner._predict_support(
                    support_model, actor, support_config
                )
        stride = int(config["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        model = ChildVisibilityMLP(
            int(config["model"]["hidden_dim"]), float(config["model"]["initial_visibility"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["model"]["learning_rate"]), weight_decay=float(config["model"]["weight_decay"])
        )
        child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
        model.train()
        history = _train(model, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "input_features": ["abs_ray_normal", "lateral_over_scale", "longitudinal_over_box", "thickness_over_scale", "occupied_mass"],
                "frozen_geometry": str(config["m11_run"]),
                "frozen_anchor_authority": str(config["m35_run"]),
                "frozen_child_authority": str(config["m38_run"]),
                "seed": int(config["model"]["seed"]),
            },
            run_dir / "MODEL.pt",
        )
        rows = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = energy_runner._energy_partition(actor, config["evaluation"], float(config["anchor_scale_m"]), device)
                m39 = composition_runner._categorical_partition(
                    actor, torch.cat([actor["fixed_anchor_occupied_t"], actor["fixed_child_occupied_t"]]), config["evaluation"]
                )
                m45 = m45_runner._partition(
                    actor, actor["fixed_anchor_occupied_t"], actor["fixed_child_occupied_t"],
                    actor["fixed_child_normals_t"], actor["fixed_child_thickness_t"], config["evaluation"]
                )
                m48 = _partition(model, actor, config["evaluation"])
                row = {
                    "scene_name": anchor_runner._scalar_text(actor["scene_name"]),
                    "track_id": anchor_runner._scalar_text(actor["track_id"]),
                    "hazardous": bool(actor["hazardous"]),
                    "ray_count": int(len(actor["target"])),
                    "mean_visibility": float(m48["mean_visibility"]),
                }
                for name, partition in (("baseline", baseline), ("m39", m39), ("m45", m45), ("m48", m48)):
                    row[f"{name}_early_count"] = int(np.count_nonzero(partition["early"]))
                    row[f"{name}_hit_count"] = int(np.count_nonzero(partition["hit"]))
                rows.append(row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(json.dumps({"stage": "m48_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}), flush=True)
        metrics = _summarize(rows)
        worst_delta = max(
            float(metrics["hazard"]["m48_vs_m39_early_delta"]),
            float(metrics["clear"]["m48_vs_m39_early_delta"]),
        )
        decisions = {
            "m48_all_early_nonincrease": float(metrics["all"]["m48_vs_m39_early_delta"]) <= float(config["decision"]["maximum_all_early_delta"]),
            "m48_hazard_and_clear_early_nonincrease": worst_delta <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "m48_all_hit_retained": float(metrics["all"]["m48_vs_m39_hit_delta"]) >= float(config["decision"]["minimum_all_hit_delta"]),
        }
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m48_visibility_measure_supported" if all(decisions.values()) else "m48_visibility_measure_rejected",
            "train_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_delta,
            "mean_holdout_visibility": float(np.mean([row["mean_visibility"] for row in rows])),
            "geometry_and_authority_frozen": True,
            "visibility_inside_joint_return_measure": True,
            "visibility_can_only_attenuate_children": True,
            "hazard_motion_category_or_target_input": False,
            "posthoc_filter": False,
            "pretrained_holdout_exposure": True,
            "external_read": False,
            "m43_partial_quality_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        child_runner._write_jsonl(run_dir / "HOLDOUT_VISIBILITY_ROWS.jsonl", rows)
        child_runner._write_json(run_dir / "summary.json", summary)
        child_runner._write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "holdout", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        child_runner._write_json(
            run_dir / "status.json", {"status": "failed", "phase": "m48", "error": f"{type(error).__name__}: {error}"}
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

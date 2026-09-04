"""Train a supervision-native set-to-set child surface from frozen M5 seeds."""

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
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import (
    differentiable_symmetric_chamfer,
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.first_return_renderer import differentiable_first_return_depth
from motion_proj.worldsim_v71.gaussian_anchor_relocation import (
    GaussianSeedExpansionMLP,
    apply_gaussian_seed_expansion,
    build_gaussian_anchor_targets,
    initialize_expansion_from_relocation,
)
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _predict(
    model: GaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    children, residuals, scales = apply_gaussian_seed_expansion(
        actor["m5_centers_t"],
        model(actor["features"]),
        maximum_residual_xyz_m=config["maximum_residual_xyz_m"],
        actor_half_size_m=actor["size_t"] * 0.5,
        cuboid_padding_m=float(config["cuboid_padding_m"]),
        minimum_scale_m=float(config["minimum_scale_m"]),
        maximum_scale_m=float(config["maximum_scale_m"]),
    )
    return children, residuals, scales


def _geometry_losses(
    model: GaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    children, _, scales = _predict(model, actor, config)
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    target_surface, _ = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), children.device
    )
    surface = torch.cat([anchors, children], dim=0)
    base_surface = torch.cat([anchors, actor["m5_centers_t"]], dim=0)
    with torch.no_grad():
        local_targets = build_gaussian_anchor_targets(
            children.detach(),
            target_surface,
            neighbors=int(config["target_neighbors"]),
            minimum_scale_m=float(config["minimum_scale_m"]),
            maximum_scale_m=float(config["maximum_scale_m"]),
        )
        base_local_targets = build_gaussian_anchor_targets(
            actor["m5_centers_t"],
            target_surface,
            neighbors=int(config["target_neighbors"]),
            minimum_scale_m=float(config["minimum_scale_m"]),
            maximum_scale_m=float(config["maximum_scale_m"]),
        )
        reference_chamfer = differentiable_symmetric_chamfer(
            base_surface, target_surface
        ).clamp_min(1.0e-3)
        reference_plane = torch.abs(
            (
                (actor["m5_centers_t"] - base_local_targets.centers)
                * base_local_targets.normals
            ).sum(dim=1)
        ).mean().clamp_min(1.0e-3)
        initial_scales = torch.full_like(
            base_local_targets.scales, float(config["initial_scale_m"])
        )
        reference_scale = F.smooth_l1_loss(
            initial_scales.log(), base_local_targets.scales.log()
        ).clamp_min(1.0e-3)
    chamfer = differentiable_symmetric_chamfer(surface, target_surface) / reference_chamfer
    point_to_plane = torch.abs(
        ((children - local_targets.centers) * local_targets.normals).sum(dim=1)
    ).mean() / reference_plane
    scale = F.smooth_l1_loss(scales.log(), local_targets.scales.log()) / reference_scale
    geometry = (
        float(config["chamfer_weight"]) * chamfer
        + float(config["point_to_plane_weight"]) * point_to_plane
        + float(config["scale_weight"]) * scale
    )
    return {
        "geometry": geometry,
        "chamfer": chamfer,
        "point_to_plane": point_to_plane,
        "scale": scale,
    }


def _physics_losses(
    model: GaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    children, _, _ = _predict(model, actor, config)
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    surface = torch.cat([anchors, children], dim=0)
    base_surface = torch.cat([anchors, actor["m5_centers_t"]], dim=0)
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), children.device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1)
    with torch.no_grad():
        base_depth = differentiable_first_return_depth(
            base_surface, origins, targets, **config["renderer"]
        )
        reference_first = F.smooth_l1_loss(base_depth, target_depth).clamp_min(1.0e-3)
        reference_free = F.relu(
            target_depth - float(config["free_space_margin_m"]) - base_depth
        ).mean().clamp_min(1.0e-3)
    predicted_depth = differentiable_first_return_depth(
        surface, origins, targets, **config["renderer"]
    )
    first = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    free_space = F.relu(
        target_depth - float(config["free_space_margin_m"]) - predicted_depth
    ).mean() / reference_free
    physics = first + float(config["free_space_weight"]) * free_space
    return {"physics": physics, "first": first, "free_space": free_space}


def _train_geometry(
    model: GaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    names = ("geometry", "chamfer", "point_to_plane", "scale")
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["stage_g_epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            items = [
                _geometry_losses(model, actors[index], config)
                for index in permutation[start : start + batch_size]
            ]
            means = {name: torch.stack([item[name] for item in items]).mean() for name in names}
            optimizer.zero_grad(set_to_none=True)
            means["geometry"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(items)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / len(actors) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps({"stage": "m7_set_geometry", "actors": len(actors), **row}), flush=True)
    return history


def _train_physics(
    model: GaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    names = ("geometry", "chamfer", "point_to_plane", "scale", "physics", "first", "free_space")
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["stage_p_epochs"])):
        totals = {name: 0.0 for name in names}
        conflicts = 0
        cosines: list[float] = []
        batches = 0
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            geometry_items = [_geometry_losses(model, actors[index], config) for index in indices]
            physics_items = [_physics_losses(model, actors[index], config) for index in indices]
            means = {
                **{
                    name: torch.stack([item[name] for item in geometry_items]).mean()
                    for name in ("geometry", "chamfer", "point_to_plane", "scale")
                },
                **{
                    name: torch.stack([item[name] for item in physics_items]).mean()
                    for name in ("physics", "first", "free_space")
                },
            }
            conflict, cosine = m6_runner._pcgrad_step(
                model, optimizer, means["geometry"], means["physics"]
            )
            conflicts += int(conflict)
            cosines.append(cosine)
            batches += 1
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / len(actors) for name, value in totals.items()},
            "conflict_batch_fraction": conflicts / max(batches, 1),
            "mean_preprojection_cosine": float(np.mean(cosines)),
        }
        history.append(row)
        print(json.dumps({"stage": "m7_set_physics", "actors": len(actors), **row}), flush=True)
    return history


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    m6_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading_train"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M7 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        m5_run = Path(config["m5_run"])
        checkpoint = torch.load(m5_run / "MODEL.pt", map_location=device, weights_only=False)
        m5_config = yaml.safe_load((m5_run / "resolved.yaml").read_text(encoding="utf-8"))
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        base = RaySurfaceRelocationMLP(
            int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
        ).to(device)
        base.load_state_dict(checkpoint["state_dict"])
        base.eval()
        base.requires_grad_(False)
        model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(config["model"]["hidden_dim"]),
            int(config["model"]["branch_factor"]),
            int(config["model"]["slot_dim"]),
        ).to(device)
        initialize_expansion_from_relocation(
            model,
            base,
            minimum_scale_m=float(config["model"]["minimum_scale_m"]),
            maximum_scale_m=float(config["model"]["maximum_scale_m"]),
            initial_scale_m=float(config["model"]["initial_scale_m"]),
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["model"]["maximum_training_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        for actor in actors:
            with torch.no_grad():
                _, centers = m5_runner._move(base, actor, m5_config["model"])
            actor["m5_centers_t"] = centers.detach()
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        m6_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "set_geometry"})
        geometry_history = _train_geometry(model, train_actors, config["model"], optimizer)
        m6_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "set_physics"})
        physics_history = _train_physics(model, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "branch_factor": int(config["model"]["branch_factor"]),
                "slot_dim": int(config["model"]["slot_dim"]),
                "seed": int(config["model"]["seed"]),
                "m5_run": str(m5_run),
                "deployment_surface": "immutable_anchors_plus_all_generated_children",
            },
            run_dir / "MODEL.pt",
        )
        m6_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "train_holdout"})
        rows: list[dict[str, Any]] = []
        m5_rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                children, residuals, scales = _predict(model, actor, config["model"])
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                m5_surface = _voxel_unique(
                    torch.cat([actor["anchors_t"], actor["m5_centers_t"]], dim=0).cpu().numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                output = _voxel_unique(
                    torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                kwargs = {
                    "hazardous": bool(actor["hazardous"]),
                    "device": device,
                    "lateral_tolerance_m": float(config["evaluation"]["literal_lateral_tolerance_m"]),
                    "depth_tolerance_m": float(config["evaluation"]["literal_depth_tolerance_m"]),
                    "distance_chunk_size": int(config["evaluation"]["distance_chunk_size"]),
                }
                row = evaluate_actor_surface(
                    baseline, output, actor["target"], actor["target_sensor_origins"], **kwargs
                )
                reference_row = evaluate_actor_surface(
                    baseline, m5_surface, actor["target"], actor["target_sensor_origins"], **kwargs
                )
                identity = {"scene_name": str(actor["scene_name"]), "track_id": str(actor["track_id"])}
                row.update(
                    {
                        **identity,
                        "generated_child_count": int(len(children)),
                        "mean_child_residual_m": float(torch.linalg.vector_norm(residuals, dim=1).mean()),
                        "mean_gaussian_scale_m": float(scales.mean()),
                    }
                )
                reference_row.update(identity)
                rows.append(row)
                m5_rows.append(reference_row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(json.dumps({"stage": "m7_train_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}), flush=True)
        metrics = summarize_surface_rows(rows)
        m5_metrics = summarize_surface_rows(m5_rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_CHILDREN.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m7_gt_supervised_seed_expansion.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m7_development_passed" if passed else "m7_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "geometry_history": geometry_history,
            "physics_history": physics_history,
            "train_holdout": metrics,
            "m5_reference": m5_metrics,
            "decisions": decisions,
            "target_construction": "actor_canonical_symmetric_set_8nn_plane_and_scale",
            "branch_factor": int(config["model"]["branch_factor"]),
            "m5_initialization": str(m5_run),
            "deployment_surface": "immutable_anchors_plus_all_generated_children",
            "unknown_hard_mask": False,
            "image_semantic_motion_inputs": False,
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        m6_runner._write_json(run_dir / "summary.json", summary)
        m6_runner._write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "train_holdout", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        m6_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m7", "error": f"{type(error).__name__}: {error}"},
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

"""Fine-tune M8 so predicted Gaussian scale participates in GT ray physics."""

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
import run_worldsim_v71_m8_temporal_frame_coverage as m8_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.first_return_renderer import (
    differentiable_scaled_first_return_depth,
    literal_spherical_first_return_partition,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _support_surface(
    anchors: torch.Tensor,
    children: torch.Tensor,
    child_scales: torch.Tensor,
    anchor_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.cat(
        [
            torch.full(
                (len(anchors),),
                float(anchor_scale_m),
                dtype=children.dtype,
                device=children.device,
            ),
            child_scales,
        ]
    )
    return torch.cat([anchors, children], dim=0), scales


def _support_physics_losses(
    model: GaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    point = m7_runner._physics_losses(model, actor, config)
    children, _, child_scales = m7_runner._predict(model, actor, config)
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    centers, scales = _support_surface(
        anchors, children, child_scales, float(config["anchor_scale_m"])
    )
    reference_centers, reference_scales = _support_surface(
        anchors,
        actor["m8_children_t"],
        actor["m8_scales_t"],
        float(config["anchor_scale_m"]),
    )
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), children.device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1)
    with torch.no_grad():
        reference_depth = differentiable_scaled_first_return_depth(
            reference_centers,
            reference_scales,
            origins,
            targets,
            **config["support_renderer"],
        )
        reference_first = F.smooth_l1_loss(
            reference_depth, target_depth
        ).clamp_min(1.0e-3)
        reference_free = F.relu(
            target_depth - float(config["free_space_margin_m"]) - reference_depth
        ).mean().clamp_min(1.0e-3)
    predicted_depth = differentiable_scaled_first_return_depth(
        centers, scales, origins, targets, **config["support_renderer"]
    )
    support_first = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    support_free = F.relu(
        target_depth - float(config["free_space_margin_m"]) - predicted_depth
    ).mean() / reference_free
    support_physics = support_first + float(config["free_space_weight"]) * support_free
    combined = 0.5 * (
        point["physics"] + float(config["support_physics_weight"]) * support_physics
    )
    return {
        "physics": combined,
        "point_first": point["first"],
        "point_free_space": point["free_space"],
        "support_first": support_first,
        "support_free_space": support_free,
    }


def _train(
    model: GaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    geometry_names = (
        "geometry",
        "chamfer",
        "point_to_plane",
        "scale",
        "temporal_frame",
    )
    physics_names = (
        "physics",
        "point_first",
        "point_free_space",
        "support_first",
        "support_free_space",
    )
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["fine_tune_epochs"])):
        totals = {name: 0.0 for name in geometry_names + physics_names}
        conflicts = 0
        cosines: list[float] = []
        batches = 0
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            geometry_items = [
                m8_runner._geometry_with_temporal(model, actors[index], config)
                for index in indices
            ]
            physics_items = [
                _support_physics_losses(model, actors[index], config)
                for index in indices
            ]
            means = {
                **{
                    name: torch.stack([item[name] for item in geometry_items]).mean()
                    for name in ("geometry_temporal",) + geometry_names
                },
                **{
                    name: torch.stack([item[name] for item in physics_items]).mean()
                    for name in physics_names
                },
            }
            conflict, cosine = m6_runner._pcgrad_step(
                model, optimizer, means["geometry_temporal"], means["physics"]
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
            json.dumps({"stage": "m9_gaussian_support", "actors": len(actors), **row}),
            flush=True,
        )
    return history


def _support_row(
    actor: Mapping[str, Any],
    m9_children: torch.Tensor,
    m9_scales: torch.Tensor,
    evaluation: Mapping[str, Any],
    anchor_scale_m: float,
    device: torch.device,
) -> dict[str, Any]:
    m8_centers, m8_support = _support_surface(
        actor["anchors_t"],
        actor["m8_children_t"],
        actor["m8_scales_t"],
        anchor_scale_m,
    )
    m9_centers, m9_support = _support_surface(
        actor["anchors_t"], m9_children, m9_scales, anchor_scale_m
    )
    kwargs = {
        "depth_tolerance_m": float(evaluation["literal_depth_tolerance_m"]),
        "device": device,
        "ray_chunk_size": int(evaluation["distance_chunk_size"]),
    }
    m8 = literal_spherical_first_return_partition(
        m8_centers,
        m8_support,
        actor["target"],
        actor["target_sensor_origins"],
        **kwargs,
    )
    m9 = literal_spherical_first_return_partition(
        m9_centers,
        m9_support,
        actor["target"],
        actor["target_sensor_origins"],
        **kwargs,
    )
    return {
        "m8_support_early_count": int(np.count_nonzero(m8["early"])),
        "m9_support_early_count": int(np.count_nonzero(m9["early"])),
        "m8_support_hit_count": int(np.count_nonzero(m8["hit"])),
        "m9_support_hit_count": int(np.count_nonzero(m9["hit"])),
    }


def _support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["target_ray_count"]) for row in selected)
        before_early = sum(int(row["m8_support_early_count"]) for row in selected)
        after_early = sum(int(row["m9_support_early_count"]) for row in selected)
        before_rate = before_early / rays if rays else None
        after_rate = after_early / rays if rays else None
        reduction = (
            (before_rate - after_rate) / before_rate
            if before_rate is not None and before_rate > 0.0 and after_rate is not None
            else None
        )
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "m8_early_count": before_early,
            "m9_early_count": after_early,
            "m8_early_rate": before_rate,
            "m9_early_rate": after_rate,
            "relative_early_reduction": reduction,
            "m8_hit_recall": sum(
                int(row["m8_support_hit_count"]) for row in selected
            )
            / rays
            if rays
            else None,
            "m9_hit_recall": sum(
                int(row["m9_support_hit_count"]) for row in selected
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
    run_dir = (
        Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    m6_runner._write_json(
        run_dir / "status.json", {"status": "running", "phase": "loading_train"}
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M9 requires CUDA")
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
        model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
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
                    model, actor, m8_config["model"]
                )
                actor["m8_children_t"] = children.detach().clone()
                actor["m8_scales_t"] = scales.detach().clone()
                actor["m7_children_t"] = actor["m8_children_t"]
        for actor in actors:
            m8_runner._prepare_temporal_actor(actor, config["model"], device)
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
            run_dir / "status.json", {"status": "running", "phase": "support_finetune"}
        )
        history = _train(model, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(checkpoint["hidden_dim"]),
                "branch_factor": int(checkpoint["branch_factor"]),
                "slot_dim": int(checkpoint["slot_dim"]),
                "seed": int(config["model"]["seed"]),
                "m5_run": str(m5_run),
                "m8_run": str(m8_run),
                "deployment_surface": "immutable_anchors_plus_all_generated_children",
                "collision_support": "one_sigma_isotropic_gaussian",
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        m8_rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                children, _, scales = m7_runner._predict(model, actor, config["model"])
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                m8_surface = _voxel_unique(
                    torch.cat([actor["anchors_t"], actor["m8_children_t"]], dim=0)
                    .cpu()
                    .numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                output = _voxel_unique(
                    torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                kwargs = {
                    "hazardous": bool(actor["hazardous"]),
                    "device": device,
                    "lateral_tolerance_m": float(
                        config["evaluation"]["literal_lateral_tolerance_m"]
                    ),
                    "depth_tolerance_m": float(
                        config["evaluation"]["literal_depth_tolerance_m"]
                    ),
                    "distance_chunk_size": int(
                        config["evaluation"]["distance_chunk_size"]
                    ),
                }
                row = evaluate_actor_surface(
                    baseline,
                    output,
                    actor["target"],
                    actor["target_sensor_origins"],
                    **kwargs,
                )
                reference_row = evaluate_actor_surface(
                    baseline,
                    m8_surface,
                    actor["target"],
                    actor["target_sensor_origins"],
                    **kwargs,
                )
                row.update(
                    _support_row(
                        actor,
                        children,
                        scales,
                        config["evaluation"],
                        float(config["model"]["anchor_scale_m"]),
                        device,
                    )
                )
                row.update(
                    {
                        "scene_name": str(actor["scene_name"]),
                        "track_id": str(actor["track_id"]),
                        "mean_gaussian_scale_m": float(scales.mean()),
                        "m8_mean_gaussian_scale_m": float(actor["m8_scales_t"].mean()),
                    }
                )
                reference_row.update(
                    {"scene_name": str(actor["scene_name"]), "track_id": str(actor["track_id"])}
                )
                rows.append(row)
                m8_rows.append(reference_row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m9_train_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        m8_metrics = summarize_surface_rows(m8_rows)
        support = _support_summary(rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        support_hazard = support["hazard"]["relative_early_reduction"]
        decisions["hazard_gaussian_support_relative_reduction"] = (
            support_hazard is not None
            and float(support_hazard)
            >= float(config["decision"]["minimum_support_hazard_relative_reduction"])
        )
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_GAUSSIAN_SUPPORT.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m9_gaussian_support_physics.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m9_development_passed" if passed else "m9_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "train_holdout": metrics,
            "m8_reference": m8_metrics,
            "gaussian_support": support,
            "decisions": decisions,
            "collision_support": "one_sigma_isotropic_gaussian",
            "all_primitives_retained": True,
            "shape_inputs": "build_geometry_only_no_trajectory_velocity_time_hazard_image",
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
            {"status": "failed", "phase": "m9", "error": f"{type(error).__name__}: {error}"},
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


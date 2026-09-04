"""Learn GT-supervised finite planar charts on frozen M8 center/radius geometry."""

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
import run_worldsim_v71_m10_oriented_planar_gaussian as m10_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.first_return_renderer import (
    differentiable_planar_disc_first_intersection_depth,
    literal_planar_disc_first_return_partition,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import (
    GaussianSeedExpansionMLP,
    OrientedGaussianSeedExpansionMLP,
    build_gaussian_anchor_targets,
    initialize_oriented_from_expansion,
)
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _predict_normals(
    model: OrientedGaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    raw = model(actor["features"])
    delta = torch.tanh(raw[..., 4:7]) * float(config["normal_residual_bound"])
    return F.normalize(
        actor["normals_t"][:, None, :] + delta, dim=-1, eps=1.0e-6
    ).reshape(-1, 3)


def _chart_attributes(
    actor: Mapping[str, Any], child_normals: torch.Tensor, anchor_radius_m: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors = actor["anchors_t"]
    anchor_radii = torch.full(
        (len(anchors),),
        float(anchor_radius_m),
        dtype=actor["m8_scales_t"].dtype,
        device=actor["m8_scales_t"].device,
    )
    return (
        torch.cat([anchors, actor["m8_children_t"]], dim=0),
        torch.cat([actor["anchor_normals_t"], child_normals], dim=0),
        torch.cat([anchor_radii, actor["m8_scales_t"]], dim=0),
    )


def _geometry_losses(
    model: OrientedGaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    centers = actor["m8_children_t"]
    scales = actor["m8_scales_t"]
    normals = _predict_normals(model, actor, config)
    reference_normals = m10_runner._reference_normals(
        actor, int(config["branch_factor"])
    )
    targets, _ = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), centers.device
    )
    with torch.no_grad():
        local_targets = build_gaussian_anchor_targets(
            centers,
            targets,
            neighbors=int(config["target_neighbors"]),
            minimum_scale_m=float(config["minimum_scale_m"]),
            maximum_scale_m=float(config["maximum_scale_m"]),
        )
        offset = centers - local_targets.centers
        reference_projection = torch.sum(offset * reference_normals, dim=1)
        reference_normal = (
            1.0
            - torch.abs(torch.sum(reference_normals * local_targets.normals, dim=1))
        ).mean().clamp_min(1.0e-3)
        reference_plane = reference_projection.abs().mean().clamp_min(1.0e-3)
        reference_tangent = torch.linalg.vector_norm(
            offset - reference_projection[:, None] * reference_normals, dim=1
        )
        reference_in_radius = F.relu(reference_tangent / scales - 1.0).mean().clamp_min(
            1.0e-3
        )
    projection = torch.sum((centers - local_targets.centers) * normals, dim=1)
    tangent = torch.linalg.vector_norm(
        (centers - local_targets.centers) - projection[:, None] * normals, dim=1
    )
    normal = (
        1.0 - torch.abs(torch.sum(normals * local_targets.normals, dim=1))
    ).mean() / reference_normal
    point_to_plane = projection.abs().mean() / reference_plane
    in_radius = F.relu(tangent / scales - 1.0).mean() / reference_in_radius
    geometry = (
        float(config["normal_weight"]) * normal
        + float(config["point_to_plane_weight"]) * point_to_plane
        + float(config["in_radius_weight"]) * in_radius
    )
    return {
        "geometry": geometry,
        "normal": normal,
        "point_to_plane": point_to_plane,
        "in_radius": in_radius,
    }


def _physics_losses(
    model: OrientedGaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    normals = _predict_normals(model, actor, config)
    current = _chart_attributes(actor, normals, float(config["anchor_radius_m"]))
    reference = _chart_attributes(
        actor,
        m10_runner._reference_normals(actor, int(config["branch_factor"])),
        float(config["anchor_radius_m"]),
    )
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), normals.device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1)
    with torch.no_grad():
        reference_depth = differentiable_planar_disc_first_intersection_depth(
            *reference, origins, targets, **config["exact_renderer"]
        )
        reference_first = F.smooth_l1_loss(
            reference_depth, target_depth
        ).clamp_min(1.0e-3)
        reference_free = F.relu(
            target_depth - float(config["free_space_margin_m"]) - reference_depth
        ).mean().clamp_min(1.0e-3)
    predicted_depth = differentiable_planar_disc_first_intersection_depth(
        *current, origins, targets, **config["exact_renderer"]
    )
    first = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    free_space = F.relu(
        target_depth - float(config["free_space_margin_m"]) - predicted_depth
    ).mean() / reference_free
    physics = first + float(config["free_space_weight"]) * free_space
    return {"physics": physics, "first": first, "free_space": free_space}


def _train(
    model: OrientedGaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    names = (
        "geometry",
        "normal",
        "point_to_plane",
        "in_radius",
        "physics",
        "first",
        "free_space",
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
                    for name in ("geometry", "normal", "point_to_plane", "in_radius")
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
            json.dumps({"stage": "m12_finite_chart", "actors": len(actors), **row}),
            flush=True,
        )
    return history


def _chart_row(
    actor: Mapping[str, Any],
    normals: torch.Tensor,
    config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    reference = _chart_attributes(
        actor,
        m10_runner._reference_normals(actor, int(config["branch_factor"])),
        float(config["anchor_radius_m"]),
    )
    predicted = _chart_attributes(actor, normals, float(config["anchor_radius_m"]))
    kwargs = {
        "depth_tolerance_m": float(evaluation["literal_depth_tolerance_m"]),
        "device": device,
        "ray_chunk_size": int(evaluation["distance_chunk_size"]),
    }
    before = literal_planar_disc_first_return_partition(
        *reference,
        actor["target"],
        actor["target_sensor_origins"],
        **kwargs,
    )
    after = literal_planar_disc_first_return_partition(
        *predicted,
        actor["target"],
        actor["target_sensor_origins"],
        **kwargs,
    )
    return {
        "m8_chart_early_count": int(np.count_nonzero(before["early"])),
        "m12_chart_early_count": int(np.count_nonzero(after["early"])),
        "m8_chart_hit_count": int(np.count_nonzero(before["hit"])),
        "m12_chart_hit_count": int(np.count_nonzero(after["hit"])),
    }


def _chart_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["target_ray_count"]) for row in selected)
        before_early = sum(int(row["m8_chart_early_count"]) for row in selected)
        after_early = sum(int(row["m12_chart_early_count"]) for row in selected)
        before_rate = before_early / rays if rays else None
        after_rate = after_early / rays if rays else None
        before_hit = sum(int(row["m8_chart_hit_count"]) for row in selected) / rays if rays else None
        after_hit = sum(int(row["m12_chart_hit_count"]) for row in selected) / rays if rays else None
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "m8_early_rate": before_rate,
            "m12_early_rate": after_rate,
            "relative_early_reduction": (
                (before_rate - after_rate) / before_rate
                if before_rate is not None and before_rate > 0.0
                else None
            ),
            "m8_hit_recall": before_hit,
            "m12_hit_recall": after_hit,
            "hit_recall_delta": after_hit - before_hit if before_hit is not None else None,
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def _prepare_anchor_normals(
    actor: dict[str, Any], config: Mapping[str, Any]
) -> None:
    anchors = actor["anchors_t"]
    if len(anchors) == 0:
        actor["anchor_normals_t"] = torch.empty_like(anchors)
        return
    build_surface = torch.cat([actor["anchors_t"], actor["candidates_t"]], dim=0)
    targets = build_gaussian_anchor_targets(
        anchors,
        build_surface,
        neighbors=int(config["target_neighbors"]),
        minimum_scale_m=float(config["minimum_scale_m"]),
        maximum_scale_m=float(config["maximum_scale_m"]),
    )
    actor["anchor_normals_t"] = targets.normals.detach()


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
        raise RuntimeError("M12 requires CUDA")
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
        model = OrientedGaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        initialize_oriented_from_expansion(
            model,
            m8_model,
            minimum_thickness_m=0.005,
            maximum_thickness_m=0.05,
            initial_thickness_m=0.02,
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
                _prepare_anchor_normals(actor, config["model"])
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
            run_dir / "status.json", {"status": "running", "phase": "finite_chart_finetune"}
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
                "frozen_center_tangent": True,
                "surface_support": "finite_zero_thickness_planar_chart",
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                normals = _predict_normals(model, actor, config["model"])
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
                row.update(
                    _chart_row(
                        actor,
                        normals,
                        config["model"],
                        config["evaluation"],
                        device,
                    )
                )
                row.update(
                    {
                        "scene_name": str(actor["scene_name"]),
                        "track_id": str(actor["track_id"]),
                        "mean_chart_radius_m": float(actor["m8_scales_t"].mean()),
                    }
                )
                rows.append(row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m12_train_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        charts = _chart_summary(rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        chart_hazard = charts["hazard"]["relative_early_reduction"]
        decisions["hazard_finite_chart_relative_reduction"] = (
            chart_hazard is not None
            and float(chart_hazard)
            >= float(config["decision"]["minimum_chart_hazard_relative_reduction"])
        )
        chart_hit_delta = charts["all"]["hit_recall_delta"]
        decisions["finite_chart_hit_recall_retained"] = (
            chart_hit_delta is not None
            and float(chart_hit_delta)
            >= float(config["decision"]["minimum_chart_hit_recall_delta"])
        )
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_FINITE_CHART.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m12_finite_planar_chart.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m12_development_passed" if passed else "m12_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "train_holdout": metrics,
            "finite_planar_chart": charts,
            "decisions": decisions,
            "frozen_center_tangent": True,
            "all_charts_retained": True,
            "shape_inputs": "build_geometry_only_no_trajectory_velocity_time_hazard_image",
            "surface_supervision": "gt_local_normal_plane_radius_and_exact_ray_disc_first_free",
            "claim_boundary": "visible_surface_first_return_not_watertight_or_collision_volume",
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
            {"status": "failed", "phase": "m12", "error": f"{type(error).__name__}: {error}"},
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

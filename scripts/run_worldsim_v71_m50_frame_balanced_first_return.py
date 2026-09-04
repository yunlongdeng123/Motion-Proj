"""Fine-tune frozen M8 geometry with frame-balanced GT first-return supervision."""

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
    differentiable_first_return_depth,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP


def _prepare_frame_actor(
    actor: dict[str, Any], config: Mapping[str, Any], device: torch.device
) -> None:
    targets, groups = m8_runner._frame_groups(
        actor, int(config["maximum_rays_per_frame"]), device
    )
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    actor["frame_targets_t"] = targets
    actor["frame_origins_t"] = origins
    actor["frame_groups_t"] = groups
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    reference_surface = torch.cat(
        [anchors, actor["m8_reference_children_t"]], dim=0
    )
    actor["m8_frame_coverage_reference_t"] = m8_runner._frame_coverage(
        reference_surface, targets, groups
    ).clamp_min(1.0e-3).detach()

    first_terms = []
    free_terms = []
    with torch.inference_mode():
        for indices in groups:
            frame_targets = targets.index_select(0, indices)
            frame_origins = origins.index_select(0, indices)
            target_depth = torch.linalg.vector_norm(
                frame_targets - frame_origins, dim=1
            )
            reference_depth = differentiable_first_return_depth(
                reference_surface,
                frame_origins,
                frame_targets,
                **config["renderer"],
            )
            first_terms.append(F.smooth_l1_loss(reference_depth, target_depth))
            free_terms.append(
                F.relu(
                    target_depth
                    - float(config["free_space_margin_m"])
                    - reference_depth
                ).mean()
            )
    actor["m8_frame_first_reference_t"] = (
        torch.stack(first_terms).mean().clamp_min(1.0e-3).detach()
    )
    actor["m8_frame_free_reference_t"] = (
        torch.stack(free_terms).mean().clamp_min(1.0e-3).detach()
    )


def _geometry_losses(
    model: GaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    losses = m7_runner._geometry_losses(model, actor, config)
    children, _, _ = m7_runner._predict(model, actor, config)
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    surface = torch.cat([anchors, children], dim=0)
    temporal = m8_runner._frame_coverage(
        surface, actor["frame_targets_t"], actor["frame_groups_t"]
    ) / actor["m8_frame_coverage_reference_t"]
    losses["temporal_frame"] = temporal
    losses["geometry_temporal"] = losses["geometry"] + float(
        config["temporal_frame_weight"]
    ) * temporal
    return losses


def _frame_physics_losses(
    model: GaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    children, _, _ = m7_runner._predict(model, actor, config)
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    surface = torch.cat([anchors, children], dim=0)
    first_terms = []
    free_terms = []
    for indices in actor["frame_groups_t"]:
        frame_targets = actor["frame_targets_t"].index_select(0, indices)
        frame_origins = actor["frame_origins_t"].index_select(0, indices)
        target_depth = torch.linalg.vector_norm(frame_targets - frame_origins, dim=1)
        predicted_depth = differentiable_first_return_depth(
            surface, frame_origins, frame_targets, **config["renderer"]
        )
        first_terms.append(F.smooth_l1_loss(predicted_depth, target_depth))
        free_terms.append(
            F.relu(
                target_depth
                - float(config["free_space_margin_m"])
                - predicted_depth
            ).mean()
        )
    first = torch.stack(first_terms).mean() / actor["m8_frame_first_reference_t"]
    free_space = (
        torch.stack(free_terms).mean() / actor["m8_frame_free_reference_t"]
    )
    physics = first + float(config["free_space_weight"]) * free_space
    return {"physics": physics, "first": first, "free_space": free_space}


def _train(
    model: GaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "geometry",
        "chamfer",
        "point_to_plane",
        "scale",
        "temporal_frame",
        "physics",
        "first",
        "free_space",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["fine_tune_epochs"])):
        totals = {name: 0.0 for name in names}
        conflicts = 0
        cosines: list[float] = []
        batches = 0
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            geometry_items = [
                _geometry_losses(model, actors[index], config) for index in indices
            ]
            physics_items = [
                _frame_physics_losses(model, actors[index], config)
                for index in indices
            ]
            means = {
                **{
                    name: torch.stack([item[name] for item in geometry_items]).mean()
                    for name in (
                        "geometry_temporal",
                        "geometry",
                        "chamfer",
                        "point_to_plane",
                        "scale",
                        "temporal_frame",
                    )
                },
                **{
                    name: torch.stack([item[name] for item in physics_items]).mean()
                    for name in ("physics", "first", "free_space")
                },
            }
            conflict, cosine = m6_runner._pcgrad_step(
                model,
                optimizer,
                means["geometry_temporal"],
                means["physics"],
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
        print(
            json.dumps(
                {"stage": "m50_frame_balanced_first_return", "actors": len(actors), **row}
            ),
            flush=True,
        )
    return history


def _per_frame_literal(
    surface: np.ndarray,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    early_rates = []
    hit_rates = []
    targets = np.asarray(actor["target"], dtype=np.float32)
    origins = np.asarray(actor["target_sensor_origins"], dtype=np.float32)
    for group in actor["frame_groups_t"]:
        indices = group.detach().cpu().numpy()
        row = evaluate_actor_surface(
            surface,
            surface,
            targets[indices],
            origins[indices],
            hazardous=bool(actor["hazardous"]),
            device=device,
            lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
            depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
            distance_chunk_size=int(evaluation["distance_chunk_size"]),
        )
        rays = max(int(row["target_ray_count"]), 1)
        early_rates.append(float(row["output_early_count"]) / rays)
        hit_rates.append(float(row["output_hit_count"]) / rays)
    return {
        "frame_count": len(early_rates),
        "mean_early_rate": float(np.mean(early_rates)),
        "worst_early_rate": float(np.max(early_rates)),
        "mean_hit_rate": float(np.mean(hit_rates)),
        "worst_hit_rate": float(np.min(hit_rates)),
    }


def _frame_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"actor_count": 0}
        keys = (
            "mean_early_rate",
            "worst_early_rate",
            "mean_hit_rate",
            "worst_hit_rate",
        )
        result: dict[str, Any] = {"actor_count": len(selected)}
        for key in keys:
            reference = np.asarray([row[f"m8_{key}"] for row in selected])
            current = np.asarray([row[f"m50_{key}"] for row in selected])
            result[f"m8_{key}"] = float(reference.mean())
            result[f"m50_{key}"] = float(current.mean())
            result[f"{key}_delta"] = float((current - reference).mean())
        return result

    return {
        "all": summarize(rows),
        "hazard": summarize([row for row in rows if bool(row["hazardous"])]),
        "clear": summarize([row for row in rows if not bool(row["hazardous"])]),
        "moving": summarize([row for row in rows if bool(row["moving"])]),
        "quasi_static": summarize([row for row in rows if not bool(row["moving"])]),
    }


def _aggregate_comparison(
    rows: list[dict[str, Any]], hazardous: bool | None = None
) -> dict[str, float | int]:
    selected = [
        row
        for row in rows
        if hazardous is None or bool(row["hazardous"]) is hazardous
    ]
    rays = sum(int(row["target_ray_count"]) for row in selected)
    return {
        "actor_count": len(selected),
        "ray_count": rays,
        "m8_early_rate": sum(int(row["m8_output_early_count"]) for row in selected)
        / max(rays, 1),
        "m50_early_rate": sum(int(row["output_early_count"]) for row in selected)
        / max(rays, 1),
        "m8_hit_rate": sum(int(row["m8_output_hit_count"]) for row in selected)
        / max(rays, 1),
        "m50_hit_rate": sum(int(row["output_hit_count"]) for row in selected)
        / max(rays, 1),
        "m8_chamfer_m": float(
            np.mean([row["m8_output_chamfer_m"] for row in selected])
        ),
        "m50_chamfer_m": float(
            np.mean([row["output_chamfer_m"] for row in selected])
        ),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    m6_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M50 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        m8_run = Path(config["m8_run"])
        checkpoint = torch.load(m8_run / "MODEL.pt", map_location=device, weights_only=False)
        m8_config = yaml.safe_load((m8_run / "resolved.yaml").read_text(encoding="utf-8"))
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        actors = [
            actor
            for path in m0_runner._paths(
                Path(config["cache_root"]),
                int(config["model"]["maximum_training_actors"]),
            )
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        model.eval()
        with torch.inference_mode():
            for actor in actors:
                children, _, _ = m7_runner._predict(model, actor, m8_config["model"])
                actor["m8_reference_children_t"] = children.detach().clone()
        for actor in actors:
            _prepare_frame_actor(actor, config["model"], device)

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
            run_dir / "status.json", {"status": "running", "phase": "frame_balanced_finetune"}
        )
        history = _train(model, train_actors, config["model"], optimizer)
        model.eval()

        rows: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        moving_threshold = float(config["evaluation"]["moving_max_displacement_m"])
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                children, _, _ = m7_runner._predict(model, actor, config["model"])
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                reference = _voxel_unique(
                    torch.cat([actor["anchors_t"], actor["m8_reference_children_t"]], dim=0)
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
                    "lateral_tolerance_m": float(config["evaluation"]["literal_lateral_tolerance_m"]),
                    "depth_tolerance_m": float(config["evaluation"]["literal_depth_tolerance_m"]),
                    "distance_chunk_size": int(config["evaluation"]["distance_chunk_size"]),
                }
                row = evaluate_actor_surface(
                    baseline, output, actor["target"], actor["target_sensor_origins"], **kwargs
                )
                reference_row = evaluate_actor_surface(
                    baseline, reference, actor["target"], actor["target_sensor_origins"], **kwargs
                )
                m8_frame = _per_frame_literal(
                    reference, actor, config["evaluation"], device
                )
                m50_frame = _per_frame_literal(output, actor, config["evaluation"], device)
                displacement = m8_runner._motion_displacement(actor)
                identity = {
                    "scene_name": str(actor["scene_name"]),
                    "track_id": str(actor["track_id"]),
                    "hazardous": bool(actor["hazardous"]),
                }
                row.update(
                    {
                        **identity,
                        "moving": displacement > moving_threshold,
                        "trajectory_max_displacement_m": displacement,
                        "m8_output_early_count": int(reference_row["output_early_count"]),
                        "m8_output_hit_count": int(reference_row["output_hit_count"]),
                        "m8_output_chamfer_m": float(reference_row["output_chamfer_m"]),
                        **{f"m8_{key}": value for key, value in m8_frame.items()},
                        **{f"m50_{key}": value for key, value in m50_frame.items()},
                    }
                )
                reference_row.update(identity)
                rows.append(row)
                reference_rows.append(reference_row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps({"stage": "m50_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}),
                        flush=True,
                    )

        metrics = summarize_surface_rows(rows)
        reference_metrics = summarize_surface_rows(reference_rows)
        frame_metrics = _frame_summary(rows)
        comparison = {
            "all": _aggregate_comparison(rows),
            "hazard": _aggregate_comparison(rows, True),
            "clear": _aggregate_comparison(rows, False),
        }
        primary = float(frame_metrics["all"]["worst_early_rate_delta"]) < 0.0
        guard = (
            float(comparison["hazard"]["m50_early_rate"])
            <= float(comparison["hazard"]["m8_early_rate"])
            + float(config["decision"]["maximum_hazard_early_regression"])
            and float(comparison["all"]["m50_hit_rate"])
            >= float(comparison["all"]["m8_hit_rate"])
            - float(config["decision"]["maximum_all_hit_regression"])
            and float(comparison["all"]["m50_chamfer_m"])
            <= float(comparison["all"]["m8_chamfer_m"])
            + float(config["decision"]["maximum_all_chamfer_regression_m"])
        )
        decisions = {
            "actor_mean_worst_frame_early_improved": primary,
            "aggregate_pareto_guard": guard,
        }
        passed = all(decisions.values())
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(checkpoint["hidden_dim"]),
                "branch_factor": int(checkpoint["branch_factor"]),
                "slot_dim": int(checkpoint["slot_dim"]),
                "seed": int(config["model"]["seed"]),
                "m8_run": str(m8_run),
                "shape_inputs": "build_geometry_only",
                "trajectory_authority": "read_only_rigid_pose",
            },
            run_dir / "MODEL.pt",
        )
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_FRAME_BALANCED.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m50_frame_balanced_first_return.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m50_frame_balanced_physics_supported" if passed else "m50_frame_balanced_physics_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "train_holdout": metrics,
            "m8_reference": reference_metrics,
            "frame_balanced_literal": frame_metrics,
            "m50_vs_m8": comparison,
            "decisions": decisions,
            "shape_inputs": "build_geometry_only_no_time_velocity_hazard_image_visibility",
            "trajectory_authority": "read_only_rigid_canonical_to_world",
            "external_read": False,
            "m43_partial_read": False,
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
            {"status": "failed", "phase": "m50", "error": f"{type(error).__name__}: {error}"},
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

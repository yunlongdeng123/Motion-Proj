"""Fine-tune M7 with frame-balanced actor-canonical GT endpoint coverage."""

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
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _frame_groups(
    actor: Mapping[str, Any], maximum_rays_per_frame: int, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = np.asarray(actor["target_sensor_origins"], dtype=np.float32).reshape(-1, 3)
    if len(targets) != len(origins) or len(targets) == 0:
        raise ValueError("frame coverage requires aligned non-empty target rays")
    _, labels = np.unique(origins, axis=0, return_inverse=True)
    groups: list[torch.Tensor] = []
    for label in range(int(labels.max()) + 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) > int(maximum_rays_per_frame):
            positions = np.linspace(
                0, len(indices) - 1, num=int(maximum_rays_per_frame), dtype=np.int64
            )
            indices = indices[positions]
        groups.append(torch.as_tensor(indices, dtype=torch.long, device=device))
    return targets, groups


def _frame_coverage(
    surface: torch.Tensor, targets: torch.Tensor, groups: list[torch.Tensor]
) -> torch.Tensor:
    per_frame = []
    for indices in groups:
        frame = targets.index_select(0, indices)
        per_frame.append(torch.cdist(frame, surface).min(dim=1).values.mean())
    return torch.stack(per_frame).mean()


def _prepare_temporal_actor(
    actor: dict[str, Any], model_config: Mapping[str, Any], device: torch.device
) -> None:
    targets, groups = _frame_groups(
        actor, int(model_config["maximum_rays_per_frame"]), device
    )
    actor["frame_targets_t"] = targets
    actor["frame_groups_t"] = groups
    surface = torch.cat(
        [
            m0_runner._limit_tensor(
                actor["anchors_t"], int(model_config["maximum_training_anchors"])
            ),
            actor["m7_children_t"],
        ],
        dim=0,
    )
    actor["m7_frame_reference_t"] = _frame_coverage(surface, targets, groups).clamp_min(
        1.0e-3
    ).detach()


def _geometry_with_temporal(
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
    temporal = _frame_coverage(
        surface, actor["frame_targets_t"], actor["frame_groups_t"]
    ) / actor["m7_frame_reference_t"]
    losses["temporal_frame"] = temporal
    losses["geometry_temporal"] = losses["geometry"] + float(
        config["temporal_frame_weight"]
    ) * temporal
    return losses


def _train(
    model: GaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
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
                _geometry_with_temporal(model, actors[index], config) for index in indices
            ]
            physics_items = [
                m7_runner._physics_losses(model, actors[index], config) for index in indices
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
                model, optimizer, means["geometry_temporal"], means["physics"]
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
            json.dumps({"stage": "m8_temporal_finetune", "actors": len(actors), **row}),
            flush=True,
        )
    return history


def _surface_frame_metrics(
    surface: torch.Tensor, actor: Mapping[str, Any]
) -> tuple[float, float]:
    values = []
    for indices in actor["frame_groups_t"]:
        frame = actor["frame_targets_t"].index_select(0, indices)
        values.append(float(torch.cdist(frame, surface).min(dim=1).values.mean()))
    return float(np.mean(values)), float(np.max(values))


def _motion_displacement(actor: Mapping[str, Any]) -> float:
    trajectory = np.asarray(actor["trajectory_xyz_m"], dtype=np.float64).reshape(-1, 3)
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory - trajectory[0], axis=1).max(initial=0.0))


def _frame_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"actor_count": 0}
        reference = np.asarray(
            [row["m7_frame_mean_target_distance_m"] for row in selected], dtype=np.float64
        )
        current = np.asarray(
            [row["m8_frame_mean_target_distance_m"] for row in selected], dtype=np.float64
        )
        return {
            "actor_count": len(selected),
            "m7_frame_mean_target_distance_m": float(reference.mean()),
            "m8_frame_mean_target_distance_m": float(current.mean()),
            "frame_mean_target_distance_delta_m": float((current - reference).mean()),
            "m7_mean_worst_frame_distance_m": float(
                np.mean([row["m7_worst_frame_target_distance_m"] for row in selected])
            ),
            "m8_mean_worst_frame_distance_m": float(
                np.mean([row["m8_worst_frame_target_distance_m"] for row in selected])
            ),
        }

    return {
        "all": stratum(rows),
        "moving": stratum([row for row in rows if bool(row["moving"])]),
        "quasi_static": stratum([row for row in rows if not bool(row["moving"])]),
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
        raise RuntimeError("M8 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        m7_run = Path(config["m7_run"])
        checkpoint = torch.load(m7_run / "MODEL.pt", map_location=device, weights_only=False)
        m7_config = yaml.safe_load((m7_run / "resolved.yaml").read_text(encoding="utf-8"))
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
                children, _, _ = m7_runner._predict(model, actor, m7_config["model"])
                actor["m7_children_t"] = children.detach().clone()
                _prepare_temporal_actor(actor, config["model"], device)
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
            run_dir / "status.json", {"status": "running", "phase": "temporal_finetune"}
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
                "m7_run": str(m7_run),
                "deployment_surface": "immutable_anchors_plus_all_generated_children",
                "geometry_motion_factorisation": "canonical_shape_plus_read_only_rigid_trajectory",
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        m7_rows: list[dict[str, Any]] = []
        moving_threshold = float(config["evaluation"]["moving_max_displacement_m"])
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                children, _, _ = m7_runner._predict(model, actor, config["model"])
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                m7_surface = _voxel_unique(
                    torch.cat([actor["anchors_t"], actor["m7_children_t"]], dim=0)
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
                    m7_surface,
                    actor["target"],
                    actor["target_sensor_origins"],
                    **kwargs,
                )
                m7_frame = _surface_frame_metrics(
                    torch.as_tensor(m7_surface, dtype=torch.float32, device=device), actor
                )
                m8_frame = _surface_frame_metrics(
                    torch.as_tensor(output, dtype=torch.float32, device=device), actor
                )
                displacement = _motion_displacement(actor)
                identity = {
                    "scene_name": str(actor["scene_name"]),
                    "track_id": str(actor["track_id"]),
                }
                row.update(
                    {
                        **identity,
                        "target_frame_count": len(actor["frame_groups_t"]),
                        "trajectory_max_displacement_m": displacement,
                        "moving": displacement > moving_threshold,
                        "m7_frame_mean_target_distance_m": m7_frame[0],
                        "m8_frame_mean_target_distance_m": m8_frame[0],
                        "m7_worst_frame_target_distance_m": m7_frame[1],
                        "m8_worst_frame_target_distance_m": m8_frame[1],
                    }
                )
                reference_row.update(identity)
                rows.append(row)
                m7_rows.append(reference_row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m8_train_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        m7_metrics = summarize_surface_rows(m7_rows)
        temporal = _frame_summary(rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        decisions["frame_balanced_coverage_improvement"] = float(
            temporal["all"]["frame_mean_target_distance_delta_m"]
        ) < -float(config["decision"]["minimum_frame_coverage_improvement_m"])
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_TEMPORAL.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m8_temporal_frame_coverage.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m8_development_passed" if passed else "m8_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "train_holdout": metrics,
            "m7_reference": m7_metrics,
            "temporal_frame_coverage": temporal,
            "decisions": decisions,
            "shape_inputs": "build_geometry_only_no_trajectory_velocity_time_hazard_image",
            "trajectory_authority": "read_only_rigid_canonical_to_world",
            "deployment_surface": "immutable_anchors_plus_all_generated_children",
            "unknown_hard_mask": False,
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
            {"status": "failed", "phase": "m8", "error": f"{type(error).__name__}: {error}"},
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


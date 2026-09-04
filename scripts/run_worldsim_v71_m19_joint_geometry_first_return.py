"""Jointly optimize completed Actor geometry and a categorical first-return field."""

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
from torch import nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m6_gt_supervised_gaussian_relocation as m6_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m8_temporal_frame_coverage as m8_runner
import run_worldsim_v71_m13_local_signed_field as field_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.local_signed_field import RayTerminationLogitField
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


class JointGeometryFirstReturn(nn.Module):
    """Expose both heads as one parameter set for two-task PCGrad."""

    def __init__(
        self,
        surface: GaussianSeedExpansionMLP,
        field: RayTerminationLogitField,
    ) -> None:
        super().__init__()
        self.surface = surface
        self.field = field


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _current_geometry(
    surface: GaussianSeedExpansionMLP,
    actor: dict[str, Any],
    surface_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    children, _, scales = m7_runner._predict(surface, actor, surface_config)
    actor["m8_children_t"] = children
    actor["m8_scales_t"] = scales
    return children, scales


def _actor_losses(
    joint: JointGeometryFirstReturn,
    actor: dict[str, Any],
    surface_config: Mapping[str, Any],
    field_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    geometry = m8_runner._geometry_with_temporal(
        joint.surface, actor, surface_config
    )
    _current_geometry(joint.surface, actor, surface_config)
    physics = field_runner._categorical_losses(
        joint.field, actor, field_config
    )
    return {**geometry, **{f"field_{key}": value for key, value in physics.items()}}


def _train(
    joint: JointGeometryFirstReturn,
    actors: list[dict[str, Any]],
    surface_config: Mapping[str, Any],
    field_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "geometry",
        "chamfer",
        "point_to_plane",
        "scale",
        "temporal_frame",
        "field_loss",
        "field_categorical_nll",
        "field_depth_l1",
        "field_entropy",
        "field_target_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(run_config["actor_batch_size"])
    for epoch in range(int(run_config["joint_epochs"])):
        totals = {name: 0.0 for name in names}
        conflicts = 0
        cosines: list[float] = []
        batches = 0
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [
                _actor_losses(
                    joint, actors[index], surface_config, field_config
                )
                for index in indices
            ]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            conflict, cosine = m6_runner._pcgrad_step(
                joint,
                optimizer,
                means["geometry"]
                + float(surface_config["temporal_frame_weight"])
                * means["temporal_frame"],
                means["field_loss"],
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
                {"stage": "m19_joint_native_3d_and_ray", "actors": len(actors), **row}
            ),
            flush=True,
        )
    return history


def _frame_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"actor_count": 0}
        reference = np.asarray(
            [row["m8_frame_mean_target_distance_m"] for row in selected],
            dtype=np.float64,
        )
        current = np.asarray(
            [row["m19_frame_mean_target_distance_m"] for row in selected],
            dtype=np.float64,
        )
        return {
            "actor_count": len(selected),
            "m8_frame_mean_target_distance_m": float(reference.mean()),
            "m19_frame_mean_target_distance_m": float(current.mean()),
            "frame_mean_target_distance_delta_m": float((current - reference).mean()),
        }

    return {
        "all": stratum(rows),
        "moving": stratum([row for row in rows if bool(row["moving"])]),
        "quasi_static": stratum([row for row in rows if not bool(row["moving"])]),
    }


def _load_models(
    config: Mapping[str, Any], device: torch.device
) -> tuple[
    JointGeometryFirstReturn,
    FeatureStandardizer,
    Mapping[str, Any],
    Mapping[str, Any],
    RaySurfaceRelocationMLP,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    m8_run = Path(config["m8_run"])
    m18_run = Path(config["m18_run"])
    m8_checkpoint = torch.load(
        m8_run / "MODEL.pt", map_location=device, weights_only=False
    )
    m18_checkpoint = torch.load(
        m18_run / "MODEL.pt", map_location=device, weights_only=False
    )
    m8_config = yaml.safe_load(
        (m8_run / "resolved.yaml").read_text(encoding="utf-8")
    )
    m18_config = yaml.safe_load(
        (m18_run / "resolved.yaml").read_text(encoding="utf-8")
    )
    standardizer = FeatureStandardizer.from_payload(m8_checkpoint["standardizer"])
    surface = GaussianSeedExpansionMLP(
        int(m8_checkpoint["input_dim"]),
        int(m8_checkpoint["hidden_dim"]),
        int(m8_checkpoint["branch_factor"]),
        int(m8_checkpoint["slot_dim"]),
    ).to(device)
    surface.load_state_dict(m8_checkpoint["state_dict"])
    field = RayTerminationLogitField(
        int(m18_checkpoint["input_dim"]),
        hidden_dim=int(m18_checkpoint["hidden_dim"]),
        branch_factor=int(m18_checkpoint["branch_factor"]),
        slot_dim=int(m18_checkpoint["slot_dim"]),
        latent_dim=int(m18_checkpoint["latent_dim"]),
        neighbor_count=int(m18_checkpoint["neighbor_count"]),
        maximum_residual_fraction=float(
            m18_checkpoint["maximum_residual_fraction"]
        ),
    ).to(device)
    field.load_state_dict(m18_checkpoint["state_dict"])
    m5_run = Path(m8_checkpoint["m5_run"])
    m5_checkpoint = torch.load(
        m5_run / "MODEL.pt", map_location=device, weights_only=False
    )
    m5_config = yaml.safe_load(
        (m5_run / "resolved.yaml").read_text(encoding="utf-8")
    )
    base = RaySurfaceRelocationMLP(
        int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
    ).to(device)
    base.load_state_dict(m5_checkpoint["state_dict"])
    base.eval()
    base.requires_grad_(False)
    reference = json.loads(
        (m18_run / "summary.json").read_text(encoding="utf-8")
    )
    return (
        JointGeometryFirstReturn(surface, field),
        standardizer,
        m8_config["model"],
        m18_config["model"],
        base,
        m5_config["model"],
        reference,
    )


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = (
        Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M19 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        (
            joint,
            standardizer,
            surface_config,
            field_config,
            base,
            m5_config,
            m18_reference,
        ) = _load_models(config, device)
        paths = m0_runner._paths(
            Path(config["cache_root"]),
            int(surface_config["maximum_training_actors"]),
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device))
            is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, centers = m5_runner._move(base, actor, m5_config)
                actor["m5_centers_t"] = centers.detach()
                children, _, scales = m7_runner._predict(
                    joint.surface, actor, surface_config
                )
                actor["m8_reference_children_t"] = children.detach()
                actor["m8_reference_scales_t"] = scales.detach()
        for actor in actors:
            actor["m8_reference_children_t"] = actor[
                "m8_reference_children_t"
            ].clone()
            actor["m8_reference_scales_t"] = actor["m8_reference_scales_t"].clone()
            actor["m7_children_t"] = actor["m8_reference_children_t"]
            m8_runner._prepare_temporal_actor(actor, surface_config, device)
        stride = int(surface_config["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            joint.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        joint.train()
        _write_json(
            run_dir / "status.json", {"status": "running", "phase": "joint_train"}
        )
        history = _train(
            joint,
            train_actors,
            surface_config,
            field_config,
            config["model"],
            optimizer,
        )
        joint.eval()
        torch.save(
            {
                "surface_state_dict": joint.surface.state_dict(),
                "field_state_dict": joint.field.state_dict(),
                "standardizer": standardizer.payload(),
                "m8_run": str(config["m8_run"]),
                "m18_run": str(config["m18_run"]),
                "seed": int(config["model"]["seed"]),
                "deployment_surface": "immutable_anchors_plus_all_generated_children",
                "deployment_first_return": "categorical_cdf_median_in_actor_aabb",
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        m8_rows: list[dict[str, Any]] = []
        evaluation = yaml.safe_load(
            (Path(config["m18_run"]) / "resolved.yaml").read_text(encoding="utf-8")
        )["evaluation"]
        moving_threshold = float(
            yaml.safe_load(
                (Path(config["m8_run"]) / "resolved.yaml").read_text(
                    encoding="utf-8"
                )
            )["evaluation"]["moving_max_displacement_m"]
        )
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                children, _ = _current_geometry(
                    joint.surface, actor, surface_config
                )
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(evaluation["output_voxel_size_m"]),
                )
                reference_surface = _voxel_unique(
                    torch.cat(
                        [actor["anchors_t"], actor["m8_reference_children_t"]], dim=0
                    )
                    .cpu()
                    .numpy(),
                    float(evaluation["output_voxel_size_m"]),
                )
                output = _voxel_unique(
                    torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy(),
                    float(evaluation["output_voxel_size_m"]),
                )
                kwargs = {
                    "hazardous": bool(actor["hazardous"]),
                    "device": device,
                    "lateral_tolerance_m": float(
                        evaluation["literal_lateral_tolerance_m"]
                    ),
                    "depth_tolerance_m": float(
                        evaluation["literal_depth_tolerance_m"]
                    ),
                    "distance_chunk_size": int(evaluation["distance_chunk_size"]),
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
                    reference_surface,
                    actor["target"],
                    actor["target_sensor_origins"],
                    **kwargs,
                )
                partition = field_runner._categorical_first_return_partition(
                    joint.field, actor, evaluation, device
                )
                m8_frame = m8_runner._surface_frame_metrics(
                    torch.as_tensor(reference_surface, dtype=torch.float32, device=device),
                    actor,
                )
                m19_frame = m8_runner._surface_frame_metrics(
                    torch.as_tensor(output, dtype=torch.float32, device=device), actor
                )
                displacement = m8_runner._motion_displacement(actor)
                identity = {
                    "scene_name": str(actor["scene_name"]),
                    "track_id": str(actor["track_id"]),
                }
                row.update(
                    {
                        **identity,
                        "field_early_count": int(np.count_nonzero(partition["early"])),
                        "field_hit_count": int(np.count_nonzero(partition["hit"])),
                        "field_observable_count": int(
                            np.count_nonzero(partition["observable"])
                        ),
                        "moving": displacement > moving_threshold,
                        "trajectory_max_displacement_m": displacement,
                        "m8_frame_mean_target_distance_m": m8_frame[0],
                        "m19_frame_mean_target_distance_m": m19_frame[0],
                        "mean_child_shift_from_m8_m": float(
                            torch.linalg.vector_norm(
                                children - actor["m8_reference_children_t"], dim=1
                            ).mean()
                        ),
                    }
                )
                reference_row.update(identity)
                rows.append(row)
                m8_rows.append(reference_row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m19_train_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        m8_metrics = summarize_surface_rows(m8_rows)
        field_metrics = field_runner._field_summary(rows)
        frame_metrics = _frame_summary(rows)
        chamfer_vs_m8 = float(metrics["output_mean_chamfer_m"]) - float(
            m8_metrics["output_mean_chamfer_m"]
        )
        decisions = {
            "actor_state_retention": float(metrics["minimum_actor_state_retention"])
            >= float(config["decision"]["required_actor_state_retention"]),
            "hazard_state_retention": float(metrics["minimum_hazard_state_retention"])
            >= float(config["decision"]["required_hazard_state_retention"]),
            "chamfer_preserved_vs_m8": chamfer_vs_m8
            <= float(config["decision"]["maximum_chamfer_delta_vs_m8_m"]),
            "field_hazard_early_reduction": float(
                field_metrics["hazard"]["relative_early_reduction"]
            )
            >= float(config["decision"]["minimum_field_hazard_relative_reduction"]),
            "field_all_hit_retained": float(field_metrics["all"]["hit_recall_delta"])
            >= float(config["decision"]["minimum_field_hit_recall_delta"]),
        }
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_JOINT.jsonl", rows)
        summary = {
            "schema_version": str(config["schema_version"]),
            "task_id": str(config["task_id"]),
            "hypothesis_id": str(config["hypothesis_id"]),
            "status": "done",
            "verdict": "m19_development_passed" if passed else "m19_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "joint_point_surface": metrics,
            "m8_point_reference": m8_metrics,
            "chamfer_delta_vs_m8_m": chamfer_vs_m8,
            "joint_categorical_first_return": field_metrics,
            "m18_frozen_reference": m18_reference["categorical_first_return"],
            "frame_coverage_vs_m8": frame_metrics,
            "mean_child_shift_from_m8_m": float(
                np.mean([row["mean_child_shift_from_m8_m"] for row in rows])
            ),
            "decisions": decisions,
            "geometry_supervision": "native_3d_set_plane_scale_and_frame_coverage",
            "physics_supervision": "native_lidar_one_hot_first_return_bin",
            "field_reference": "current_joint_point_surface",
            "trajectory_authority": "read_only_rigid_canonical_to_world",
            "image_semantic_time_velocity_hazard_inputs": False,
            "unknown_hard_mask_or_surface_filter": False,
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "train_holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "m19",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()


"""Train completed Actor geometry through a decoder-free Gaussian ray energy."""

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
import run_worldsim_v71_m13_local_signed_field as field_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _current_geometry(
    model: GaussianSeedExpansionMLP,
    actor: dict[str, Any],
    surface_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    children, _, scales = m7_runner._predict(model, actor, surface_config)
    actor["m8_children_t"] = children
    actor["m8_scales_t"] = scales
    return children, scales


def _gaussian_energy(
    actor: Mapping[str, Any],
    queries: torch.Tensor,
    anchor_scale_m: float,
) -> torch.Tensor:
    child_scales = actor["m8_scales_t"].reshape(-1).clamp_min(1.0e-4)
    anchor_scales = torch.full(
        (len(actor["anchors_t"]),),
        float(anchor_scale_m),
        dtype=queries.dtype,
        device=queries.device,
    )
    centers = torch.cat([actor["anchors_t"], actor["m8_children_t"]], dim=0)
    scales = torch.cat([anchor_scales, child_scales], dim=0)
    normalized_distance = torch.cdist(queries, centers) / scales[None, :]
    return torch.logsumexp(-0.5 * normalized_distance.square(), dim=1)


def _ray_values(
    model: GaussianSeedExpansionMLP,
    actor: dict[str, Any],
    surface_config: Mapping[str, Any],
    field_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _current_geometry(model, actor, surface_config)
    targets, origins = m6_runner._limit_target(
        actor,
        int(field_config["maximum_training_rays"]),
        actor["features"].device,
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(field_config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    fractions = torch.linspace(
        0.0,
        1.0,
        sample_count,
        dtype=targets.dtype,
        device=targets.device,
    )
    depths = entry[valid, None] + (
        exit_depth[valid] - entry[valid]
    )[:, None] * fractions[None, :]
    queries = origins[valid, None, :] + depths[:, :, None] * directions[valid, None, :]
    logits = _gaussian_energy(
        actor, queries.reshape(-1, 3), float(run_config["anchor_scale_m"])
    ).reshape(len(depths), sample_count)
    return depths, target_depth[valid], logits


def _physics_losses(
    model: GaussianSeedExpansionMLP,
    actor: dict[str, Any],
    surface_config: Mapping[str, Any],
    field_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, logits = _ray_values(
        model,
        actor,
        surface_config,
        field_config,
        run_config,
        int(field_config["categorical_train_bins"]),
    )
    target_bins = torch.abs(depths - target_depth[:, None]).argmin(dim=1)
    categorical_nll = F.cross_entropy(logits, target_bins)
    probabilities = torch.softmax(logits, dim=1)
    expected_depth = torch.sum(probabilities * depths, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1.0e-8)), dim=1
    ).mean()
    loss = categorical_nll + float(field_config["categorical_depth_weight"]) * depth_l1
    return {
        "loss": loss,
        "categorical_nll": categorical_nll,
        "depth_l1": depth_l1,
        "entropy": entropy,
        "target_probability": probabilities.gather(1, target_bins[:, None]).mean(),
    }


def _train(
    model: GaussianSeedExpansionMLP,
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
        "energy_loss",
        "energy_categorical_nll",
        "energy_depth_l1",
        "energy_entropy",
        "energy_target_probability",
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
            geometry_items = [
                m8_runner._geometry_with_temporal(
                    model, actors[index], surface_config
                )
                for index in indices
            ]
            physics_items = [
                _physics_losses(
                    model,
                    actors[index],
                    surface_config,
                    field_config,
                    run_config,
                )
                for index in indices
            ]
            means = {
                **{
                    name: torch.stack([item[name] for item in geometry_items]).mean()
                    for name in (
                        "geometry",
                        "chamfer",
                        "point_to_plane",
                        "scale",
                        "temporal_frame",
                    )
                },
                **{
                    f"energy_{name}": torch.stack(
                        [item[name] for item in physics_items]
                    ).mean()
                    for name in (
                        "loss",
                        "categorical_nll",
                        "depth_l1",
                        "entropy",
                        "target_probability",
                    )
                },
            }
            conflict, cosine = m6_runner._pcgrad_step(
                model,
                optimizer,
                means["geometry"]
                + float(surface_config["temporal_frame_weight"])
                * means["temporal_frame"],
                means["energy_loss"],
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
                {"stage": "m20_decoder_free_gaussian_energy", "actors": len(actors), **row}
            ),
            flush=True,
        )
    return history


def _energy_partition(
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    anchor_scale_m: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = field_runner._ray_box_intervals(
        origins, directions, bounds
    )
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
        logits = _gaussian_energy(
            actor, queries.reshape(-1, 3), anchor_scale_m
        ).reshape(len(local_entry), -1)
        cdf = torch.softmax(logits, dim=1).cumsum(dim=1)
        indices = (cdf >= threshold).to(torch.int64).argmax(dim=1)
        previous = (indices - 1).clamp_min(0)
        right_cdf = cdf.gather(1, indices[:, None]).squeeze(1)
        gathered_left_cdf = cdf.gather(1, previous[:, None]).squeeze(1)
        left_cdf = torch.where(
            indices > 0, gathered_left_cdf, torch.zeros_like(gathered_left_cdf)
        )
        right_depth = depths.gather(1, indices[:, None]).squeeze(1)
        gathered_left_depth = depths.gather(1, previous[:, None]).squeeze(1)
        left_depth = torch.where(indices > 0, gathered_left_depth, local_entry)
        ratio = (threshold - left_cdf) / (right_cdf - left_cdf).clamp_min(1.0e-6)
        selected = left_depth + ratio.clamp(0.0, 1.0) * (right_depth - left_depth)
        first_depths.append(
            torch.where(local_valid, selected, torch.full_like(selected, torch.inf)).cpu()
        )
        observables.append(local_valid.cpu())
    first_depth = torch.cat(first_depths).numpy()
    observable = torch.cat(observables).numpy().astype(bool)
    target_depth_np = target_depth.cpu().numpy()
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "observable": observable,
        "early": observable & (first_depth < target_depth_np - tolerance),
        "hit": observable & (np.abs(first_depth - target_depth_np) <= tolerance),
    }


def _frame_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"actor_count": 0}
        reference = np.asarray(
            [row["m8_frame_mean_target_distance_m"] for row in selected]
        )
        current = np.asarray(
            [row["m20_frame_mean_target_distance_m"] for row in selected]
        )
        return {
            "actor_count": len(selected),
            "m8_frame_mean_target_distance_m": float(reference.mean()),
            "m20_frame_mean_target_distance_m": float(current.mean()),
            "frame_mean_target_distance_delta_m": float((current - reference).mean()),
        }

    return {
        "all": stratum(rows),
        "moving": stratum([row for row in rows if bool(row["moving"])]),
        "quasi_static": stratum([row for row in rows if not bool(row["moving"])]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M20 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        m8_run = Path(config["m8_run"])
        m18_run = Path(config["m18_run"])
        checkpoint = torch.load(m8_run / "MODEL.pt", map_location=device, weights_only=False)
        surface_config = yaml.safe_load(
            (m8_run / "resolved.yaml").read_text(encoding="utf-8")
        )["model"]
        m18_resolved = yaml.safe_load(
            (m18_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        field_config = m18_resolved["model"]
        evaluation = m18_resolved["evaluation"]
        m18_reference = json.loads(
            (m18_run / "summary.json").read_text(encoding="utf-8")
        )["categorical_first_return"]
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        m5_run = Path(checkpoint["m5_run"])
        m5_checkpoint = torch.load(
            m5_run / "MODEL.pt", map_location=device, weights_only=False
        )
        m5_config = yaml.safe_load(
            (m5_run / "resolved.yaml").read_text(encoding="utf-8")
        )["model"]
        base = RaySurfaceRelocationMLP(
            int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
        ).to(device)
        base.load_state_dict(m5_checkpoint["state_dict"])
        base.eval()
        base.requires_grad_(False)
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(surface_config["maximum_training_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, centers = m5_runner._move(base, actor, m5_config)
                actor["m5_centers_t"] = centers.detach()
                children, _, scales = m7_runner._predict(model, actor, surface_config)
                actor["m8_reference_children_t"] = children.detach()
                actor["m8_reference_scales_t"] = scales.detach()
        for actor in actors:
            actor["m8_reference_children_t"] = actor["m8_reference_children_t"].clone()
            actor["m8_reference_scales_t"] = actor["m8_reference_scales_t"].clone()
            actor["m7_children_t"] = actor["m8_reference_children_t"]
            m8_runner._prepare_temporal_actor(actor, surface_config, device)
        stride = int(surface_config["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        model.train()
        _write_json(run_dir / "status.json", {"status": "running", "phase": "train"})
        history = _train(
            model,
            train_actors,
            surface_config,
            field_config,
            config["model"],
            optimizer,
        )
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "m8_run": str(m8_run),
                "seed": int(config["model"]["seed"]),
                "ray_energy": "decoder_free_metric_gaussian_logsumexp",
                "anchor_scale_m": float(config["model"]["anchor_scale_m"]),
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        m8_rows: list[dict[str, Any]] = []
        moving_threshold = float(
            yaml.safe_load((m8_run / "resolved.yaml").read_text(encoding="utf-8"))[
                "evaluation"
            ]["moving_max_displacement_m"]
        )
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                children, _ = _current_geometry(model, actor, surface_config)
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
                    "depth_tolerance_m": float(evaluation["literal_depth_tolerance_m"]),
                    "distance_chunk_size": int(evaluation["distance_chunk_size"]),
                }
                row = evaluate_actor_surface(
                    baseline, output, actor["target"], actor["target_sensor_origins"], **kwargs
                )
                reference_row = evaluate_actor_surface(
                    baseline,
                    reference_surface,
                    actor["target"],
                    actor["target_sensor_origins"],
                    **kwargs,
                )
                partition = _energy_partition(
                    actor,
                    evaluation,
                    float(config["model"]["anchor_scale_m"]),
                    device,
                )
                m8_frame = m8_runner._surface_frame_metrics(
                    torch.as_tensor(reference_surface, dtype=torch.float32, device=device), actor
                )
                m20_frame = m8_runner._surface_frame_metrics(
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
                        "field_observable_count": int(np.count_nonzero(partition["observable"])),
                        "moving": displacement > moving_threshold,
                        "m8_frame_mean_target_distance_m": m8_frame[0],
                        "m20_frame_mean_target_distance_m": m20_frame[0],
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
                            {"stage": "m20_train_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        m8_metrics = summarize_surface_rows(m8_rows)
        energy_metrics = field_runner._field_summary(rows)
        frame_metrics = _frame_summary(rows)
        chamfer_vs_m8 = float(metrics["output_mean_chamfer_m"]) - float(
            m8_metrics["output_mean_chamfer_m"]
        )
        hazard_energy_delta = float(energy_metrics["hazard"]["field_early_rate"]) - float(
            m18_reference["hazard"]["field_early_rate"]
        )
        all_energy_hit_delta = float(energy_metrics["all"]["field_hit_recall"]) - float(
            m18_reference["all"]["field_hit_recall"]
        )
        decisions = {
            "actor_state_retention": float(metrics["minimum_actor_state_retention"])
            >= float(config["decision"]["required_actor_state_retention"]),
            "hazard_state_retention": float(metrics["minimum_hazard_state_retention"])
            >= float(config["decision"]["required_hazard_state_retention"]),
            "point_hazard_early_reduction": float(
                metrics["hazard"]["relative_early_reduction"]
            )
            >= float(config["decision"]["minimum_point_hazard_relative_reduction"]),
            "chamfer_preserved_vs_m8": chamfer_vs_m8
            <= float(config["decision"]["maximum_chamfer_delta_vs_m8_m"]),
            "energy_hazard_early_vs_m18": hazard_energy_delta
            <= float(config["decision"]["maximum_energy_hazard_early_delta_vs_m18"]),
            "energy_all_hit_vs_m18": all_energy_hit_delta
            >= float(config["decision"]["minimum_energy_all_hit_delta_vs_m18"]),
        }
        passed = all(decisions.values())
        m6_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_ENERGY.jsonl", rows)
        summary = {
            "schema_version": str(config["schema_version"]),
            "task_id": str(config["task_id"]),
            "hypothesis_id": str(config["hypothesis_id"]),
            "status": "done",
            "verdict": "m20_development_passed" if passed else "m20_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "point_surface": metrics,
            "m8_point_reference": m8_metrics,
            "chamfer_delta_vs_m8_m": chamfer_vs_m8,
            "decoder_free_gaussian_energy": energy_metrics,
            "m18_frozen_reference": m18_reference,
            "energy_hazard_early_delta_vs_m18": hazard_energy_delta,
            "energy_all_hit_delta_vs_m18": all_energy_hit_delta,
            "frame_coverage_vs_m8": frame_metrics,
            "mean_child_shift_from_m8_m": float(
                np.mean([row["mean_child_shift_from_m8_m"] for row in rows])
            ),
            "decisions": decisions,
            "trainable_field_decoder": False,
            "geometry_supervision": "native_3d_set_plane_scale_and_frame_coverage",
            "physics_supervision": "native_lidar_one_hot_first_return_bin",
            "trajectory_authority": "read_only_rigid_canonical_to_world",
            "unknown_hard_mask_or_surface_filter": False,
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
            {"status": "failed", "phase": "m20", "error": f"{type(error).__name__}: {error}"},
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


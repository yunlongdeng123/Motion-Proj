"""Train only oriented child support through the deployed categorical CDF."""

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
import run_worldsim_v71_m6_gt_supervised_gaussian_relocation as target_runner
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


def _interval_losses(
    model: OrientedGaussianSeedExpansionMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    _, _, normals, thickness = oriented_runner._predict_support(model, actor, config)
    targets, origins = target_runner._limit_target(
        actor, int(config["maximum_training_rays"]), actor["features"].device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid = anchor_runner.field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    targets = targets[valid]
    origins = origins[valid]
    target_depth = target_depth[valid]
    directions = directions[valid]
    entry = entry[valid]
    exit_depth = exit_depth[valid]
    fractions = torch.linspace(
        0.0,
        1.0,
        int(config["categorical_train_segments"]),
        dtype=torch.float32,
        device=targets.device,
    )
    safe_losses = []
    hit_losses = []
    early_probabilities = []
    hit_probabilities = []
    chunk = int(config["training_ray_chunk_size"])
    tolerance = float(config["event_tolerance_m"])
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
        energy = m45_runner._oriented_energy(
            queries.reshape(-1, 3),
            actor,
            actor["fixed_anchor_occupied_t"],
            actor["fixed_child_occupied_t"],
            normals,
            thickness,
        ).reshape(len(local_entry), -1)
        probabilities = torch.softmax(energy, dim=1)
        local_target_depth = target_depth[start : start + chunk]
        early = depths < local_target_depth[:, None] - tolerance
        hit = torch.abs(depths - local_target_depth[:, None]) <= tolerance
        early_probability = torch.sum(probabilities * early, dim=1)
        hit_probability = torch.sum(probabilities * hit, dim=1)
        safe_losses.append(-torch.log((1.0 - early_probability).clamp_min(1.0e-8)))
        hit_losses.append(-torch.log(hit_probability.clamp_min(1.0e-8)))
        early_probabilities.append(early_probability)
        hit_probabilities.append(hit_probability)
    safe_nll = torch.cat(safe_losses).mean()
    hit_nll = torch.cat(hit_losses).mean()
    geometry = oriented_runner._geometry_losses(model, actor, config)
    loss = (
        safe_nll
        + hit_nll
        + float(config["geometry_weight"]) * geometry["geometry"]
    )
    return {
        "loss": loss,
        "safe_nll": safe_nll,
        "hit_nll": hit_nll,
        "early_probability": torch.cat(early_probabilities).mean(),
        "hit_probability": torch.cat(hit_probabilities).mean(),
        "normal": geometry["normal"],
        "thickness": geometry["thickness"],
        "boundary": geometry["boundary"],
    }


def _train(
    model: OrientedGaussianSeedExpansionMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "loss",
        "safe_nll",
        "hit_nll",
        "early_probability",
        "hit_probability",
        "normal",
        "thickness",
        "boundary",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [_interval_losses(model, actors[index], config) for index in indices]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {"epoch": epoch + 1}
        row.update({name: totals[name] / len(actors) for name in names})
        history.append(row)
        print(json.dumps({"stage": "m46_train", **row}), flush=True)
    return history


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M46 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = loader_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_training_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = centers
                children, residuals, scales = m8_runner._predict(
                    surface, actor, surface_config
                )
                actor["m8_children_t"] = children
                actor["m8_residuals_t"] = residuals
                actor["m8_scales_t"] = scales
        for actor in actors:
            anchor_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["anchor_scale_m"]),
                config["features"],
                device,
            )
            actor["authority_child_features_t"] = child_runner._child_features(actor)

        anchor_checkpoint = torch.load(
            Path(config["m35_run"]) / "MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        anchor_authority = authority_runner._load_authority(anchor_checkpoint, device)
        anchor_authority.eval().requires_grad_(False)
        child_checkpoint = torch.load(
            Path(config["m38_run"]) / "CHILD_MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        child_authority = authority_runner._load_authority(child_checkpoint, device)
        child_authority.eval().requires_grad_(False)
        with torch.inference_mode():
            for actor in actors:
                actor["fixed_anchor_occupied_t"] = occupied_masses(
                    anchor_authority(actor["authority_anchor_features_t"])
                )
                actor["fixed_child_occupied_t"] = occupied_masses(
                    child_authority(actor["authority_child_features_t"])
                )

        checkpoint = torch.load(
            Path(config["initial_m11_run"]) / "MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        model = OrientedGaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.point_encoder.requires_grad_(False)
        model.slot_embeddings.requires_grad_(False)
        model.head[0].requires_grad_(False)
        stride = int(config["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
        model.train()
        history = _train(model, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(checkpoint["hidden_dim"]),
                "branch_factor": int(checkpoint["branch_factor"]),
                "slot_dim": int(checkpoint["slot_dim"]),
                "initialized_from": str(config["initial_m11_run"]),
                "m8_run": str(config["m8_run"]),
                "frozen_anchor_authority": str(config["m35_run"]),
                "frozen_child_authority": str(config["m38_run"]),
                "seed": int(config["model"]["seed"]),
                "trainable": "normal_and_thickness_output_rows_only",
            },
            run_dir / "MODEL.pt",
        )

        rows = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = energy_runner._energy_partition(
                    actor, config["evaluation"], float(config["anchor_scale_m"]), device
                )
                isotropic = composition_runner._categorical_partition(
                    actor,
                    torch.cat(
                        [actor["fixed_anchor_occupied_t"], actor["fixed_child_occupied_t"]]
                    ),
                    config["evaluation"],
                )
                _, _, normals, thickness = oriented_runner._predict_support(
                    model, actor, config["model"]
                )
                oriented = m45_runner._partition(
                    actor,
                    actor["fixed_anchor_occupied_t"],
                    actor["fixed_child_occupied_t"],
                    normals,
                    thickness,
                    config["evaluation"],
                )
                rows.append(
                    {
                        "scene_name": anchor_runner._scalar_text(actor["scene_name"]),
                        "track_id": anchor_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "m39_early_count": int(np.count_nonzero(isotropic["early"])),
                        "m39_hit_count": int(np.count_nonzero(isotropic["hit"])),
                        "oriented_early_count": int(np.count_nonzero(oriented["early"])),
                        "oriented_hit_count": int(np.count_nonzero(oriented["hit"])),
                        "mean_normal_thickness_m": float(thickness.mean()),
                        "mean_anisotropy_ratio": float(
                            (actor["m8_scales_t"] / thickness).mean()
                        ),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m46_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = m45_runner._summarize(rows)
        worst_delta = max(
            float(metrics["hazard"]["oriented_vs_m39_early_delta"]),
            float(metrics["clear"]["oriented_vs_m39_early_delta"]),
        )
        decisions = {
            "m46_all_early_nonincrease": float(
                metrics["all"]["oriented_vs_m39_early_delta"]
            )
            <= float(config["decision"]["maximum_all_early_delta"]),
            "m46_hazard_and_clear_early_nonincrease": worst_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "m46_all_hit_retained": float(
                metrics["all"]["oriented_vs_m39_hit_delta"]
            )
            >= float(config["decision"]["minimum_all_hit_delta"]),
        }
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m46_cdf_supervised_oriented_support_supported"
            if all(decisions.values())
            else "m46_cdf_supervised_oriented_support_rejected",
            "train_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_delta,
            "mean_normal_thickness_m": float(
                np.mean([row["mean_normal_thickness_m"] for row in rows])
            ),
            "mean_anisotropy_ratio": float(
                np.mean([row["mean_anisotropy_ratio"] for row in rows])
            ),
            "geometry_centers_tangent_and_authority_frozen": True,
            "hazard_input": False,
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
        child_runner._write_jsonl(run_dir / "HOLDOUT_ROWS.jsonl", rows)
        child_runner._write_json(run_dir / "summary.json", summary)
        child_runner._write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        child_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m46", "error": f"{type(error).__name__}: {error}"},
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

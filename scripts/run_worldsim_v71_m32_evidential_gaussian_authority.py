"""Train continuous FREE/OCCUPIED/UNKNOWN authority on frozen M8 Gaussians."""

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
import run_worldsim_v71_m13_local_signed_field as field_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
    primitive_authority_features,
    weighted_gaussian_energy,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _primitive_tensors(
    actor: Mapping[str, Any], anchor_scale_m: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors = actor["anchors_t"]
    children = actor["m8_children_t"]
    centers = torch.cat([anchors, children], dim=0)
    scales = torch.cat(
        [
            torch.full(
                (len(anchors),),
                float(anchor_scale_m),
                dtype=centers.dtype,
                device=centers.device,
            ),
            actor["m8_scales_t"].reshape(-1),
        ],
        dim=0,
    ).clamp_min(1.0e-4)
    is_anchor = torch.cat(
        [
            torch.ones(len(anchors), dtype=torch.bool, device=centers.device),
            torch.zeros(len(children), dtype=torch.bool, device=centers.device),
        ]
    )
    features = primitive_authority_features(
        centers, scales, is_anchor, actor["size_t"]
    )
    return centers, scales, is_anchor, features


def _primitive_evidence_targets(
    actor: Mapping[str, Any], centers: torch.Tensor, config: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, float | int]]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["evidence_maximum_rays"]), centers.device
    )
    free = torch.zeros(len(centers), dtype=torch.float32, device=centers.device)
    occupied = torch.zeros_like(free)
    chunk = int(config["ray_chunk_size"])
    with torch.inference_mode():
        for start in range(0, len(targets), chunk):
            local_targets = targets[start : start + chunk]
            local_origins = origins[start : start + chunk]
            vectors = local_targets - local_origins
            target_depth = torch.linalg.vector_norm(vectors, dim=1).clamp_min(1.0e-6)
            directions = vectors / target_depth[:, None]
            primitive_vectors = centers[None, :, :] - local_origins[:, None, :]
            depths = torch.sum(primitive_vectors * directions[:, None, :], dim=-1)
            lateral = torch.linalg.vector_norm(
                primitive_vectors - depths[:, :, None] * directions[:, None, :], dim=-1
            )
            visible = (depths > 0.0) & (
                lateral <= float(config["lateral_tolerance_m"])
            )
            free += torch.sum(
                visible
                & (depths < target_depth[:, None] - float(config["depth_tolerance_m"])),
                dim=0,
            )
            occupied += torch.sum(
                visible
                & (
                    torch.abs(depths - target_depth[:, None])
                    <= float(config["depth_tolerance_m"])
                ),
                dim=0,
            )
    unknown = torch.full_like(free, float(config["unknown_pseudocount"]))
    denominator = free + occupied + unknown
    masses = torch.stack([free, occupied, unknown], dim=1) / denominator[:, None]
    stats = {
        "primitive_count": int(len(centers)),
        "unsupported_count": int(torch.count_nonzero((free + occupied) == 0.0)),
        "conflicted_count": int(torch.count_nonzero((free > 0.0) & (occupied > 0.0))),
        "mean_free_target_mass": float(masses[:, 0].mean()),
        "mean_occupied_target_mass": float(masses[:, 1].mean()),
        "mean_unknown_target_mass": float(masses[:, 2].mean()),
    }
    return masses, stats


def _ray_values(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), actor["features"].device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    fractions = torch.linspace(
        0.0,
        1.0,
        int(config["categorical_train_bins"]),
        dtype=targets.dtype,
        device=targets.device,
    )
    depths = entry[valid, None] + (
        exit_depth[valid] - entry[valid]
    )[:, None] * fractions[None, :]
    queries = origins[valid, None, :] + depths[:, :, None] * directions[valid, None, :]
    centers = actor["authority_centers_t"]
    scales = actor["authority_scales_t"]
    logits = authority(actor["authority_features_t"])
    energy = weighted_gaussian_energy(
        queries.reshape(-1, 3), centers, scales, occupied_masses(logits)
    ).reshape(len(depths), -1)
    return depths, target_depth[valid], energy, logits


def _losses(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, energy, authority_logits = _ray_values(
        authority, actor, config
    )
    target_bins = torch.abs(depths - target_depth[:, None]).argmin(dim=1)
    categorical = F.cross_entropy(energy, target_bins)
    probabilities = torch.softmax(energy, dim=1)
    expected_depth = torch.sum(probabilities * depths, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    target_masses = actor["authority_target_masses_t"]
    evidential = -torch.sum(
        target_masses * torch.log_softmax(authority_logits, dim=1), dim=1
    ).mean()
    loss = (
        categorical
        + float(config["categorical_depth_weight"]) * depth_l1
        + float(config["evidential_mass_weight"]) * evidential
    )
    return {
        "loss": loss,
        "categorical_nll": categorical,
        "depth_l1": depth_l1,
        "evidential_cross_entropy": evidential,
        "target_probability": probabilities.gather(1, target_bins[:, None]).mean(),
    }


def _train(
    authority: EvidentialGaussianAuthority,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "loss",
        "categorical_nll",
        "depth_l1",
        "evidential_cross_entropy",
        "target_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [_losses(authority, actors[index], config) for index in indices]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: total / len(actors) for name, total in totals.items()},
        }
        history.append(row)
        print(json.dumps({"stage": "m32_authority_train", **row}), flush=True)
    return history


def _weighted_partition(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    device = actor["features"].device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(actor["target_sensor_origins"], dtype=torch.float32, device=device)
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
    authority_logits = authority(actor["authority_features_t"])
    occupied = occupied_masses(authority_logits)
    first_depths = []
    threshold = float(evaluation["categorical_median_threshold"])
    chunk = int(evaluation["ray_chunk_size"])
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
        energy = weighted_gaussian_energy(
            queries.reshape(-1, 3),
            actor["authority_centers_t"],
            actor["authority_scales_t"],
            occupied,
        ).reshape(len(local_entry), -1)
        cdf = torch.softmax(energy, dim=1).cumsum(dim=1)
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
        first_depths.append(
            left_depth + ratio.clamp(0.0, 1.0) * (right_depth - left_depth)
        )
    first_depth = torch.cat(first_depths)
    observable = valid_box
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "observable": observable.cpu().numpy().astype(bool),
        "early": (observable & (first_depth < target_depth - tolerance)).cpu().numpy(),
        "hit": (
            observable & (torch.abs(first_depth - target_depth) <= tolerance)
        ).cpu().numpy(),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        baseline_early = sum(int(row["baseline_early_count"]) for row in selected)
        learned_early = sum(int(row["learned_early_count"]) for row in selected)
        baseline_hit = sum(int(row["baseline_hit_count"]) for row in selected)
        learned_hit = sum(int(row["learned_hit_count"]) for row in selected)
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "baseline_early_rate": baseline_early / rays,
            "learned_early_rate": learned_early / rays,
            "early_rate_delta": (learned_early - baseline_early) / rays,
            "baseline_hit_rate": baseline_hit / rays,
            "learned_hit_rate": learned_hit / rays,
            "hit_rate_delta": (learned_hit - baseline_hit) / rays,
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M32 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = m22_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["model"]["maximum_training_actors"])
        )
        actors = [
            actor for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = centers
                children, _, scales = m7_runner._predict(surface, actor, surface_config)
                actor["m8_children_t"] = children
                actor["m8_scales_t"] = scales
                primitive_centers, primitive_scales, is_anchor, features = _primitive_tensors(
                    actor, float(config["model"]["anchor_scale_m"])
                )
                actor["authority_centers_t"] = primitive_centers
                actor["authority_scales_t"] = primitive_scales
                actor["authority_is_anchor_t"] = is_anchor
                actor["authority_features_t"] = features
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        with torch.inference_mode():
            for actor in train_actors:
                masses, stats = _primitive_evidence_targets(
                    actor, actor["authority_centers_t"], config["model"]
                )
                actor["authority_target_masses_t"] = masses
                actor["authority_target_stats"] = stats
        authority = EvidentialGaussianAuthority(
            hidden_dim=int(config["model"]["hidden_dim"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            authority.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
        authority.train()
        history = _train(authority, train_actors, config["model"], optimizer)
        authority.eval()
        torch.save(
            {
                "state_dict": authority.state_dict(),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "m8_run": str(config["m8_run"]),
                "seed": int(config["model"]["seed"]),
                "states": ["FREE", "OCCUPIED", "UNKNOWN"],
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = m20_runner._energy_partition(
                    actor,
                    config["evaluation"],
                    float(config["model"]["anchor_scale_m"]),
                    device,
                )
                learned = _weighted_partition(authority, actor, config["evaluation"])
                logits = authority(actor["authority_features_t"])
                masses = torch.softmax(logits, dim=1)
                is_anchor = actor["authority_is_anchor_t"]
                target_masses, target_stats = _primitive_evidence_targets(
                    actor, actor["authority_centers_t"], config["model"]
                )
                rows.append(
                    {
                        "scene_name": str(actor["scene_name"]),
                        "track_id": str(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "primitive_count": int(len(masses)),
                        "anchor_count": int(torch.count_nonzero(is_anchor)),
                        "child_count": int(torch.count_nonzero(~is_anchor)),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "learned_early_count": int(np.count_nonzero(learned["early"])),
                        "learned_hit_count": int(np.count_nonzero(learned["hit"])),
                        "learned_observable_count": int(np.count_nonzero(learned["observable"])),
                        "mean_anchor_masses": masses[is_anchor].mean(dim=0).cpu().tolist(),
                        "mean_child_masses": masses[~is_anchor].mean(dim=0).cpu().tolist(),
                        "target_mass_stats": target_stats,
                        "target_mass_cross_entropy": float(
                            -torch.sum(
                                target_masses * torch.log_softmax(logits, dim=1), dim=1
                            ).mean()
                        ),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m32_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        worst_stratum_delta = max(
            float(metrics["hazard"]["early_rate_delta"]),
            float(metrics["clear"]["early_rate_delta"]),
        )
        decisions = {
            "all_early_nonincrease": float(metrics["all"]["early_rate_delta"])
            <= float(config["decision"]["maximum_all_early_delta"]),
            "hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "all_hit_retained": float(metrics["all"]["hit_rate_delta"])
            >= float(config["decision"]["minimum_all_hit_delta"]),
        }
        passed = all(decisions.values())
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m32_development_passed" if passed else "m32_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "training_history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "mean_holdout_anchor_masses": np.mean(
                [row["mean_anchor_masses"] for row in rows], axis=0
            ).tolist(),
            "mean_holdout_child_masses": np.mean(
                [row["mean_child_masses"] for row in rows], axis=0
            ).tolist(),
            "mean_holdout_target_mass_cross_entropy": float(
                np.mean([row["target_mass_cross_entropy"] for row in rows])
            ),
            "geometry_centers_and_scales_frozen": True,
            "geometry_state_retention": 1.0,
            "actor_and_hazard_state_retention": 1.0,
            "physics_supervision": "native_lidar_primitive_evidence_and_one_hot_first_return",
            "unknown_hard_mask_or_surface_filter": False,
            "binary_authority_threshold": False,
            "pretrained_holdout_exposure": True,
            "external_read": False,
            "m21_partial_quality_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "HOLDOUT_AUTHORITY_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "holdout", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m32", "error": f"{type(error).__name__}: {error}"},
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

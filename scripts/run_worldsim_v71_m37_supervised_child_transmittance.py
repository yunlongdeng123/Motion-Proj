"""Train a separate completion-child authority under ordered transmittance."""

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
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
import run_worldsim_v71_m35_transmittance_anchor_authority as m35_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
)


CHILD_EXTRA_FEATURE_NAMES = (
    "child_local_x_over_half_length",
    "child_local_y_over_half_width",
    "child_local_z_over_half_height",
    "child_abs_local_x_over_half_length",
    "child_abs_local_y_over_half_width",
    "child_abs_local_z_over_half_height",
    "child_scale_over_actor_diagonal",
    "child_residual_x_over_length",
    "child_residual_y_over_width",
    "child_residual_z_over_height",
    "child_slot_0",
    "child_slot_1",
    "child_slot_2",
    "child_slot_3",
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


def _child_features(actor: Mapping[str, Any]) -> torch.Tensor:
    parents = actor["features"]
    children = actor["m8_children_t"]
    residuals = actor["m8_residuals_t"]
    scales = actor["m8_scales_t"].reshape(-1, 1)
    parent_count = len(parents)
    branch_factor = len(children) // max(parent_count, 1)
    if branch_factor != 4 or parent_count * branch_factor != len(children):
        raise RuntimeError("M37 expects the frozen M8 four-child parent ordering")
    size = actor["size_t"].reshape(3).clamp_min(0.10)
    half = 0.5 * size
    diagonal = torch.linalg.vector_norm(size).clamp_min(1.0e-6)
    local = children / half.reshape(1, 3)
    slots = F.one_hot(
        torch.arange(branch_factor, device=children.device).repeat(parent_count),
        num_classes=branch_factor,
    ).to(torch.float32)
    return torch.cat(
        [
            parents.repeat_interleave(branch_factor, dim=0),
            local,
            local.abs(),
            scales / diagonal,
            residuals / size.reshape(1, 3),
            slots,
        ],
        dim=1,
    )


def _child_evidence_targets(
    actor: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    centers = actor["m8_children_t"]
    targets, origins = m6_runner._limit_target(
        actor, int(config["evidence_maximum_rays"]), centers.device
    )
    free = torch.zeros(len(centers), dtype=torch.float32, device=centers.device)
    occupied = torch.zeros_like(free)
    opportunities = torch.zeros_like(free)
    ray_chunk = int(config["evidence_ray_chunk_size"])
    with torch.no_grad():
        for start in range(0, len(targets), ray_chunk):
            local_targets = targets[start : start + ray_chunk]
            local_origins = origins[start : start + ray_chunk]
            vectors = local_targets - local_origins
            target_depth = torch.linalg.vector_norm(vectors, dim=1).clamp_min(1.0e-6)
            directions = vectors / target_depth[:, None]
            relative = centers[None, :, :] - local_origins[:, None, :]
            depth = torch.sum(relative * directions[:, None, :], dim=-1)
            lateral = torch.linalg.vector_norm(
                relative - depth[:, :, None] * directions[:, None, :], dim=-1
            )
            opportunity = (depth > 0.0) & (
                lateral <= float(config["evidence_beam_radius_m"])
            )
            opportunities += opportunity.sum(dim=0)
            free += (
                opportunity
                & (depth < target_depth[:, None] - float(config["evidence_endpoint_radius_m"]))
            ).sum(dim=0)
            occupied += (
                opportunity
                & (
                    torch.abs(depth - target_depth[:, None])
                    <= float(config["evidence_endpoint_radius_m"])
                )
            ).sum(dim=0)
        denominator = opportunities.clamp_min(1.0)
        free_mass = free / denominator
        occupied_mass = occupied / denominator
        unknown_mass = torch.where(
            opportunities > 0,
            (1.0 - free_mass - occupied_mass).clamp_min(0.0),
            torch.ones_like(opportunities),
        )
        masses = torch.stack([free_mass, occupied_mass, unknown_mass], dim=1)
        masses = masses / masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return masses, {
        "child_count": int(len(centers)),
        "unsupported_count": int(torch.count_nonzero(opportunities == 0)),
        "conflicted_count": int(torch.count_nonzero((free > 0) & (occupied > 0))),
        "mean_target_masses": masses.mean(dim=0).cpu().tolist(),
    }


def _occupied(
    anchor_authority: EvidentialGaussianAuthority,
    child_authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        anchor_logits = anchor_authority(actor["authority_anchor_features_t"])
        anchor_occupied = occupied_masses(anchor_logits)
    child_logits = child_authority(actor["authority_child_features_t"])
    return torch.cat([anchor_occupied, occupied_masses(child_logits)]), child_logits


def _losses(
    anchor_authority: EvidentialGaussianAuthority,
    child_authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    occupied, child_logits = _occupied(anchor_authority, child_authority, actor)
    midpoints, target_depth, log_probabilities, no_return, _ = m35_runner._ray_distribution(
        actor,
        occupied,
        int(config["maximum_training_rays"]),
        int(config["categorical_train_segments"]),
        float(config["training_cuboid_padding_m"]),
    )
    target_bins = torch.abs(midpoints - target_depth[:, None]).argmin(dim=1)
    categorical = F.nll_loss(log_probabilities, target_bins)
    probabilities = torch.exp(log_probabilities)
    expected_depth = torch.sum(probabilities * midpoints, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    target_masses = actor["authority_child_target_masses_t"]
    evidential = -torch.sum(
        target_masses * torch.log_softmax(child_logits, dim=1), dim=1
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
        "no_return_probability": no_return.mean(),
    }


def _train(
    anchor_authority: EvidentialGaussianAuthority,
    child_authority: EvidentialGaussianAuthority,
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
        "no_return_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [
                _losses(anchor_authority, child_authority, actors[index], config)
                for index in indices
            ]
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
        print(json.dumps({"stage": "m37_child_authority_train", **row}), flush=True)
    return history


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {"actor_count": len(selected), "ray_count": rays}
        for name in ("baseline", "unit_child", "learned_child"):
            early = sum(int(row[f"{name}_early_count"]) for row in selected)
            hit = sum(int(row[f"{name}_hit_count"]) for row in selected)
            output[f"{name}_early_rate"] = early / rays
            output[f"{name}_hit_rate"] = hit / rays
        output["learned_vs_baseline_early_delta"] = (
            output["learned_child_early_rate"] - output["baseline_early_rate"]
        )
        output["learned_vs_baseline_hit_delta"] = (
            output["learned_child_hit_rate"] - output["baseline_hit_rate"]
        )
        output["learned_vs_unit_child_early_delta"] = (
            output["learned_child_early_rate"] - output["unit_child_early_rate"]
        )
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
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M37 requires CUDA")
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
                children, residuals, scales = m7_runner._predict(
                    surface, actor, surface_config
                )
                actor["m8_children_t"] = children
                actor["m8_residuals_t"] = residuals
                actor["m8_scales_t"] = scales
        for actor in actors:
            actor["m8_residuals_t"] = actor["m8_residuals_t"].clone()
            m34_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["model"]["anchor_scale_m"]),
                config["features"],
                device,
            )
            actor["authority_child_features_t"] = _child_features(actor)
            masses, stats = _child_evidence_targets(actor, config["model"])
            actor["authority_child_target_masses_t"] = masses
            actor["authority_child_target_stats"] = stats
        anchor_checkpoint = torch.load(
            Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        anchor_authority = EvidentialGaussianAuthority(
            hidden_dim=int(anchor_checkpoint["hidden_dim"]),
            input_dim=int(anchor_checkpoint["input_dim"]),
        ).to(device)
        anchor_authority.load_state_dict(anchor_checkpoint["state_dict"])
        anchor_authority.eval().requires_grad_(False)
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        child_input_dim = int(train_actors[0]["authority_child_features_t"].shape[1])
        child_authority = EvidentialGaussianAuthority(
            hidden_dim=int(config["model"]["hidden_dim"]), input_dim=child_input_dim
        ).to(device)
        optimizer = torch.optim.AdamW(
            child_authority.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
        child_authority.train()
        history = _train(
            anchor_authority, child_authority, train_actors, config["model"], optimizer
        )
        child_authority.eval()
        torch.save(
            {
                "state_dict": child_authority.state_dict(),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "input_dim": child_input_dim,
                "parent_feature_dim": int(train_actors[0]["features"].shape[1]),
                "extra_feature_names": CHILD_EXTRA_FEATURE_NAMES,
                "m8_run": str(config["m8_run"]),
                "frozen_anchor_run": str(config["m35_run"]),
                "seed": int(config["model"]["seed"]),
                "states": ["FREE", "OCCUPIED", "UNKNOWN"],
            },
            run_dir / "CHILD_MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        predicted_occupied_parts = []
        target_occupied_parts = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = m20_runner._energy_partition(
                    actor,
                    config["evaluation"],
                    float(config["model"]["anchor_scale_m"]),
                    device,
                )
                anchor_logits = anchor_authority(actor["authority_anchor_features_t"])
                anchor_occupied = occupied_masses(anchor_logits)
                unit_occupied = torch.cat(
                    [
                        anchor_occupied,
                        torch.ones(
                            len(actor["m8_children_t"]),
                            dtype=torch.float32,
                            device=device,
                        ),
                    ]
                )
                unit_child = m35_runner._transmittance_partition(
                    actor, unit_occupied, config["evaluation"]
                )
                occupied, child_logits = _occupied(
                    anchor_authority, child_authority, actor
                )
                learned = m35_runner._transmittance_partition(
                    actor, occupied, config["evaluation"]
                )
                child_masses = torch.softmax(child_logits, dim=1)
                target_masses = actor["authority_child_target_masses_t"]
                predicted_occupied_parts.append(child_masses[:, 1].cpu().numpy())
                target_occupied_parts.append(target_masses[:, 1].cpu().numpy())
                rows.append(
                    {
                        "scene_name": m34_runner._scalar_text(actor["scene_name"]),
                        "track_id": m34_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "anchor_count": int(len(actor["anchors_t"])),
                        "child_count": int(len(child_masses)),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "unit_child_early_count": int(np.count_nonzero(unit_child["early"])),
                        "unit_child_hit_count": int(np.count_nonzero(unit_child["hit"])),
                        "learned_child_early_count": int(np.count_nonzero(learned["early"])),
                        "learned_child_hit_count": int(np.count_nonzero(learned["hit"])),
                        "unit_child_no_return_probability": unit_child[
                            "mean_no_return_probability"
                        ],
                        "learned_child_no_return_probability": learned[
                            "mean_no_return_probability"
                        ],
                        "mean_child_masses": child_masses.mean(dim=0).cpu().tolist(),
                        "target_mass_cross_entropy": float(
                            -torch.sum(
                                target_masses * torch.log_softmax(child_logits, dim=1), dim=1
                            ).mean()
                        ),
                        "target_stats": actor["authority_child_target_stats"],
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m37_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        predicted_occupied = np.concatenate(predicted_occupied_parts)
        target_occupied = np.concatenate(target_occupied_parts)
        occupied_correlation = float(
            np.corrcoef(predicted_occupied, target_occupied)[0, 1]
        )
        worst_stratum_delta = max(
            float(metrics["hazard"]["learned_vs_baseline_early_delta"]),
            float(metrics["clear"]["learned_vs_baseline_early_delta"]),
        )
        decisions = {
            "all_early_nonincrease": float(
                metrics["all"]["learned_vs_baseline_early_delta"]
            ) <= float(config["decision"]["maximum_all_early_delta"]),
            "hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "all_hit_retained": float(metrics["all"]["learned_vs_baseline_hit_delta"])
            >= float(config["decision"]["minimum_all_hit_delta"]),
            "child_authority_identifiable": occupied_correlation
            >= float(config["decision"]["minimum_child_occupied_correlation"]),
        }
        passed = all(decisions.values())
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m37_development_passed" if passed else "m37_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "training_history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "holdout_predicted_target_child_occupied_correlation": occupied_correlation,
            "mean_holdout_target_mass_cross_entropy": float(
                np.mean([row["target_mass_cross_entropy"] for row in rows])
            ),
            "mean_holdout_child_masses": np.mean(
                [row["mean_child_masses"] for row in rows], axis=0
            ).tolist(),
            "unit_child_mean_no_return_probability": float(
                np.mean([row["unit_child_no_return_probability"] for row in rows])
            ),
            "learned_child_mean_no_return_probability": float(
                np.mean([row["learned_child_no_return_probability"] for row in rows])
            ),
            "geometry_centers_and_scales_frozen": True,
            "anchor_authority_frozen": True,
            "separate_anchor_child_heads": True,
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
        _write_jsonl(run_dir / "HOLDOUT_CHILD_AUTHORITY_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m37", "error": f"{type(error).__name__}: {error}"},
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

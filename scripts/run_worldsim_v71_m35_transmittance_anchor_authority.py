"""Train producer-evidential anchor authority with ordered Gaussian transmittance."""

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
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
)
from motion_proj.worldsim_v71.gaussian_first_return import (
    gaussian_conditional_termination_log_probabilities,
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


def _occupied_with_unit_children(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_logits = authority(actor["authority_anchor_features_t"])
    anchor_occupied = occupied_masses(anchor_logits)
    child_occupied = torch.ones(
        len(actor["m8_children_t"]),
        dtype=anchor_occupied.dtype,
        device=anchor_occupied.device,
    )
    return torch.cat([anchor_occupied, child_occupied]), anchor_logits


def _ray_distribution(
    actor: Mapping[str, Any],
    occupied: torch.Tensor,
    maximum_rays: int,
    segment_count: int,
    cuboid_padding_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(maximum_rays), actor["features"].device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(cuboid_padding_m)
    entry, exit_depth, valid_box = field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    fractions = torch.linspace(
        0.0,
        1.0,
        int(segment_count) + 1,
        dtype=targets.dtype,
        device=targets.device,
    )
    edges = entry[valid, None] + (
        exit_depth[valid] - entry[valid]
    )[:, None] * fractions[None, :]
    midpoints = 0.5 * (edges[:, :-1] + edges[:, 1:])
    log_probabilities, no_return, optical_thickness = (
        gaussian_conditional_termination_log_probabilities(
            origins[valid],
            directions[valid],
            edges,
            actor["authority_centers_t"],
            actor["authority_scales_t"],
            occupied,
        )
    )
    return midpoints, target_depth[valid], log_probabilities, no_return, optical_thickness


def _losses(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    occupied, anchor_logits = _occupied_with_unit_children(authority, actor)
    midpoints, target_depth, log_probabilities, no_return, _ = _ray_distribution(
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
    target_masses = actor["authority_target_masses_t"]
    evidential = -torch.sum(
        target_masses * torch.log_softmax(anchor_logits, dim=1), dim=1
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
        "no_return_probability",
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
        print(json.dumps({"stage": "m35_transmittance_train", **row}), flush=True)
    return history


def _transmittance_partition(
    actor: Mapping[str, Any],
    occupied: torch.Tensor,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    targets = torch.as_tensor(
        actor["target"], dtype=torch.float32, device=actor["features"].device
    )
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=targets.device
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
        int(evaluation["field_segment_count"]) + 1,
        dtype=targets.dtype,
        device=targets.device,
    )
    first_depth_parts = []
    no_return_parts = []
    chunk = int(evaluation["ray_chunk_size"])
    for start in range(0, len(targets), chunk):
        local_entry = entry[start : start + chunk]
        local_exit = exit_depth[start : start + chunk]
        edges = local_entry[:, None] + (
            local_exit - local_entry
        )[:, None] * fractions[None, :]
        midpoints = 0.5 * (edges[:, :-1] + edges[:, 1:])
        log_probabilities, no_return, _ = (
            gaussian_conditional_termination_log_probabilities(
                origins[start : start + chunk],
                directions[start : start + chunk],
                edges,
                actor["authority_centers_t"],
                actor["authority_scales_t"],
                occupied,
            )
        )
        probabilities = torch.exp(log_probabilities)
        cdf = torch.cumsum(probabilities, dim=1)
        indices = (cdf >= 0.5).to(torch.int64).argmax(dim=1)
        first_depth_parts.append(
            midpoints.gather(1, indices[:, None]).squeeze(1)
        )
        no_return_parts.append(no_return)
    first_depth = torch.cat(first_depth_parts)
    no_return = torch.cat(no_return_parts)
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "observable": valid_box.cpu().numpy().astype(bool),
        "early": (
            valid_box & (first_depth < target_depth - tolerance)
        ).cpu().numpy(),
        "hit": (
            valid_box & (torch.abs(first_depth - target_depth) <= tolerance)
        ).cpu().numpy(),
        "mean_no_return_probability": float(no_return.mean()),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {"actor_count": len(selected), "ray_count": rays}
        for name in ("baseline", "unit_transmittance", "learned_transmittance"):
            early = sum(int(row[f"{name}_early_count"]) for row in selected)
            hit = sum(int(row[f"{name}_hit_count"]) for row in selected)
            output[f"{name}_early_rate"] = early / rays
            output[f"{name}_hit_rate"] = hit / rays
        output["learned_vs_baseline_early_delta"] = (
            output["learned_transmittance_early_rate"] - output["baseline_early_rate"]
        )
        output["learned_vs_baseline_hit_delta"] = (
            output["learned_transmittance_hit_rate"] - output["baseline_hit_rate"]
        )
        output["learned_vs_unit_transmittance_early_delta"] = (
            output["learned_transmittance_early_rate"]
            - output["unit_transmittance_early_rate"]
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
        raise RuntimeError("M35 requires CUDA")
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
        for actor in actors:
            m34_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["model"]["anchor_scale_m"]),
                config["features"],
                device,
            )
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        authority = EvidentialGaussianAuthority(
            hidden_dim=int(config["model"]["hidden_dim"]),
            input_dim=len(m34_runner.FEATURE_NAMES),
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
                "input_dim": len(m34_runner.FEATURE_NAMES),
                "feature_names": m34_runner.FEATURE_NAMES,
                "m8_run": str(config["m8_run"]),
                "sidecar_root": str(config["sidecar_root"]),
                "seed": int(config["model"]["seed"]),
                "states": ["FREE", "OCCUPIED", "UNKNOWN"],
                "ray_composition": "analytic_gaussian_integral_prefix_transmittance",
                "conditional_on_actor_box_return": True,
                "child_authority": "unit",
            },
            run_dir / "MODEL.pt",
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
                unit = torch.ones(
                    len(actor["authority_centers_t"]),
                    dtype=torch.float32,
                    device=device,
                )
                unit_transmittance = _transmittance_partition(
                    actor, unit, config["evaluation"]
                )
                occupied, logits = _occupied_with_unit_children(authority, actor)
                learned = _transmittance_partition(
                    actor, occupied, config["evaluation"]
                )
                masses = torch.softmax(logits, dim=1)
                target_masses = actor["authority_target_masses_t"]
                predicted_occupied_parts.append(masses[:, 1].cpu().numpy())
                target_occupied_parts.append(target_masses[:, 1].cpu().numpy())
                rows.append(
                    {
                        "scene_name": m34_runner._scalar_text(actor["scene_name"]),
                        "track_id": m34_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "anchor_count": int(len(masses)),
                        "child_count": int(len(actor["m8_children_t"])),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "unit_transmittance_early_count": int(
                            np.count_nonzero(unit_transmittance["early"])
                        ),
                        "unit_transmittance_hit_count": int(
                            np.count_nonzero(unit_transmittance["hit"])
                        ),
                        "learned_transmittance_early_count": int(
                            np.count_nonzero(learned["early"])
                        ),
                        "learned_transmittance_hit_count": int(
                            np.count_nonzero(learned["hit"])
                        ),
                        "unit_mean_no_return_probability": unit_transmittance[
                            "mean_no_return_probability"
                        ],
                        "learned_mean_no_return_probability": learned[
                            "mean_no_return_probability"
                        ],
                        "mean_anchor_masses": masses.mean(dim=0).cpu().tolist(),
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
                            {"stage": "m35_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
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
            "all_hit_retained": float(
                metrics["all"]["learned_vs_baseline_hit_delta"]
            ) >= float(config["decision"]["minimum_all_hit_delta"]),
            "anchor_authority_identifiable": occupied_correlation
            >= float(config["decision"]["minimum_occupied_correlation"]),
        }
        passed = all(decisions.values())
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m35_development_passed" if passed else "m35_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "training_history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "holdout_predicted_target_occupied_correlation": occupied_correlation,
            "mean_holdout_target_mass_cross_entropy": float(
                np.mean([row["target_mass_cross_entropy"] for row in rows])
            ),
            "mean_holdout_anchor_masses": np.mean(
                [row["mean_anchor_masses"] for row in rows], axis=0
            ).tolist(),
            "unit_mean_no_return_probability": float(
                np.mean([row["unit_mean_no_return_probability"] for row in rows])
            ),
            "learned_mean_no_return_probability": float(
                np.mean([row["learned_mean_no_return_probability"] for row in rows])
            ),
            "geometry_centers_and_scales_frozen": True,
            "completion_child_authority": "unit",
            "ray_composition": "analytic_gaussian_integral_prefix_transmittance",
            "return_distribution_conditioned_on_actor_box_hit": True,
            "geometry_state_retention": 1.0,
            "actor_and_hazard_state_retention": 1.0,
            "physics_supervision": "producer_build_inputs_to_heldout_lidar_evidence_and_first_return",
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
        _write_jsonl(run_dir / "HOLDOUT_TRANSMITTANCE_ROWS.jsonl", rows)
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
            {"status": "failed", "phase": "m35", "error": f"{type(error).__name__}: {error}"},
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

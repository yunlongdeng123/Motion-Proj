"""Train categorical authority while conserving anchor/child family surface measure."""

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

import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
import run_worldsim_v71_m37_supervised_child_transmittance as m37_runner
import run_worldsim_v71_m38_prehit_free_space_survival as m38_runner
import run_worldsim_v71_m39_categorical_authority_composition as m39_runner
import run_worldsim_v71_m40_joint_categorical_evidential_authority as m40_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
    weighted_gaussian_energy,
)


def _conserved_measure(
    logits: torch.Tensor, reference_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = occupied_masses(logits)
    reference_total = occupied_masses(reference_logits).sum().detach()
    measure = probabilities / probabilities.sum().clamp_min(1.0e-8) * reference_total
    return measure, probabilities


def _ray_values(
    reference_anchor: EvidentialGaussianAuthority,
    reference_child: EvidentialGaussianAuthority,
    anchor_authority: EvidentialGaussianAuthority,
    child_authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    targets, origins = m40_runner.m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), actor["features"].device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = m34_runner.field_runner._ray_box_intervals(
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
    with torch.no_grad():
        reference_anchor_logits = reference_anchor(actor["authority_anchor_features_t"])
        reference_child_logits = reference_child(actor["authority_child_features_t"])
    anchor_logits = anchor_authority(actor["authority_anchor_features_t"])
    child_logits = child_authority(actor["authority_child_features_t"])
    anchor_measure, _ = _conserved_measure(anchor_logits, reference_anchor_logits)
    child_measure, _ = _conserved_measure(child_logits, reference_child_logits)
    energy = weighted_gaussian_energy(
        queries.reshape(-1, 3),
        actor["authority_centers_t"],
        actor["authority_scales_t"],
        torch.cat([anchor_measure, child_measure]),
    ).reshape(len(depths), -1)
    return depths, target_depth[valid], energy, anchor_logits, child_logits


def _losses(
    reference_anchor: EvidentialGaussianAuthority,
    reference_child: EvidentialGaussianAuthority,
    anchor_authority: EvidentialGaussianAuthority,
    child_authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, energy, anchor_logits, child_logits = _ray_values(
        reference_anchor,
        reference_child,
        anchor_authority,
        child_authority,
        actor,
        config,
    )
    target_bins = torch.abs(depths - target_depth[:, None]).argmin(dim=1)
    categorical = F.cross_entropy(energy, target_bins)
    probabilities = torch.softmax(energy, dim=1)
    expected_depth = torch.sum(probabilities * depths, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    anchor_evidential = -torch.sum(
        actor["authority_target_masses_t"]
        * torch.log_softmax(anchor_logits, dim=1),
        dim=1,
    ).mean()
    child_evidential = -torch.sum(
        actor["authority_child_target_masses_t"]
        * torch.log_softmax(child_logits, dim=1),
        dim=1,
    ).mean()
    loss = (
        categorical
        + float(config["categorical_depth_weight"]) * depth_l1
        + float(config["anchor_evidential_mass_weight"]) * anchor_evidential
        + float(config["child_evidential_mass_weight"]) * child_evidential
    )
    return {
        "loss": loss,
        "categorical_nll": categorical,
        "depth_l1": depth_l1,
        "anchor_evidential_cross_entropy": anchor_evidential,
        "child_evidential_cross_entropy": child_evidential,
        "target_probability": probabilities.gather(1, target_bins[:, None]).mean(),
    }


def _train(
    reference_anchor: EvidentialGaussianAuthority,
    reference_child: EvidentialGaussianAuthority,
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
        "anchor_evidential_cross_entropy",
        "child_evidential_cross_entropy",
        "target_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [
                _losses(
                    reference_anchor,
                    reference_child,
                    anchor_authority,
                    child_authority,
                    actors[index],
                    config,
                )
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
        print(json.dumps({"stage": "m41_conserved_measure_train", **row}), flush=True)
    return history


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {"actor_count": len(selected), "ray_count": rays}
        for name in ("baseline", "m39", "m41"):
            early = sum(int(row[f"{name}_early_count"]) for row in selected)
            hit = sum(int(row[f"{name}_hit_count"]) for row in selected)
            output[f"{name}_early_rate"] = early / rays
            output[f"{name}_hit_rate"] = hit / rays
        for reference in ("baseline", "m39"):
            output[f"m41_vs_{reference}_early_delta"] = (
                output["m41_early_rate"] - output[f"{reference}_early_rate"]
            )
            output[f"m41_vs_{reference}_hit_delta"] = (
                output["m41_hit_rate"] - output[f"{reference}_hit_rate"]
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
    m37_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M41 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        actors = m40_runner._load_actors(config, device)
        anchor_checkpoint = torch.load(
            Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        child_checkpoint = torch.load(
            Path(config["m38_run"]) / "CHILD_MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        reference_anchor = m38_runner._load_authority(anchor_checkpoint, device)
        reference_anchor.eval().requires_grad_(False)
        reference_child = m38_runner._load_authority(child_checkpoint, device)
        reference_child.eval().requires_grad_(False)
        anchor_authority = m38_runner._load_authority(anchor_checkpoint, device)
        child_authority = m38_runner._load_authority(child_checkpoint, device)

        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            list(anchor_authority.parameters()) + list(child_authority.parameters()),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        m37_runner._write_json(
            run_dir / "status.json", {"status": "running", "phase": "training"}
        )
        anchor_authority.train()
        child_authority.train()
        history = _train(
            reference_anchor,
            reference_child,
            anchor_authority,
            child_authority,
            train_actors,
            config["model"],
            optimizer,
        )
        anchor_authority.eval()
        child_authority.eval()
        torch.save(
            {
                "anchor_state_dict": anchor_authority.state_dict(),
                "anchor_hidden_dim": int(anchor_checkpoint["hidden_dim"]),
                "anchor_input_dim": int(anchor_checkpoint["input_dim"]),
                "child_state_dict": child_authority.state_dict(),
                "child_hidden_dim": int(child_checkpoint["hidden_dim"]),
                "child_input_dim": int(child_checkpoint["input_dim"]),
                "reference_anchor": str(Path(config["m35_run"]) / "MODEL.pt"),
                "reference_child": str(Path(config["m38_run"]) / "CHILD_MODEL.pt"),
                "surface_measure": "per-actor separate anchor/child reference-total conservation",
                "seed": int(config["model"]["seed"]),
                "states": ["FREE", "OCCUPIED", "UNKNOWN"],
            },
            run_dir / "MODEL.pt",
        )

        rows: list[dict[str, Any]] = []
        correlations = {
            "anchor_predicted": [],
            "anchor_target": [],
            "child_predicted": [],
            "child_target": [],
        }
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = m20_runner._energy_partition(
                    actor,
                    config["evaluation"],
                    float(config["model"]["anchor_scale_m"]),
                    device,
                )
                reference_anchor_logits = reference_anchor(
                    actor["authority_anchor_features_t"]
                )
                reference_child_logits = reference_child(
                    actor["authority_child_features_t"]
                )
                reference_occupied = torch.cat(
                    [
                        occupied_masses(reference_anchor_logits),
                        occupied_masses(reference_child_logits),
                    ]
                )
                m39_result = m39_runner._categorical_partition(
                    actor, reference_occupied, config["evaluation"]
                )
                anchor_logits = anchor_authority(actor["authority_anchor_features_t"])
                child_logits = child_authority(actor["authority_child_features_t"])
                anchor_measure, anchor_probabilities = _conserved_measure(
                    anchor_logits, reference_anchor_logits
                )
                child_measure, child_probabilities = _conserved_measure(
                    child_logits, reference_child_logits
                )
                m41_result = m39_runner._categorical_partition(
                    actor,
                    torch.cat([anchor_measure, child_measure]),
                    config["evaluation"],
                )
                correlations["anchor_predicted"].append(anchor_probabilities.cpu().numpy())
                correlations["anchor_target"].append(
                    actor["authority_target_masses_t"][:, 1].cpu().numpy()
                )
                correlations["child_predicted"].append(child_probabilities.cpu().numpy())
                correlations["child_target"].append(
                    actor["authority_child_target_masses_t"][:, 1].cpu().numpy()
                )
                rows.append(
                    {
                        "scene_name": m34_runner._scalar_text(actor["scene_name"]),
                        "track_id": m34_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "m39_early_count": int(np.count_nonzero(m39_result["early"])),
                        "m39_hit_count": int(np.count_nonzero(m39_result["hit"])),
                        "m41_early_count": int(np.count_nonzero(m41_result["early"])),
                        "m41_hit_count": int(np.count_nonzero(m41_result["hit"])),
                        "reference_anchor_total": float(occupied_masses(reference_anchor_logits).sum()),
                        "m41_anchor_total": float(anchor_measure.sum()),
                        "reference_child_total": float(occupied_masses(reference_child_logits).sum()),
                        "m41_child_total": float(child_measure.sum()),
                        "mean_anchor_probabilities": torch.softmax(anchor_logits, dim=1).mean(dim=0).cpu().tolist(),
                        "mean_child_probabilities": torch.softmax(child_logits, dim=1).mean(dim=0).cpu().tolist(),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m41_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )

        metrics = _summarize(rows)
        anchor_correlation = float(
            np.corrcoef(
                np.concatenate(correlations["anchor_predicted"]),
                np.concatenate(correlations["anchor_target"]),
            )[0, 1]
        )
        child_correlation = float(
            np.corrcoef(
                np.concatenate(correlations["child_predicted"]),
                np.concatenate(correlations["child_target"]),
            )[0, 1]
        )
        worst_stratum_delta = max(
            float(metrics["hazard"]["m41_vs_baseline_early_delta"]),
            float(metrics["clear"]["m41_vs_baseline_early_delta"]),
        )
        decisions = {
            "all_early_nonincrease": float(
                metrics["all"]["m41_vs_baseline_early_delta"]
            ) <= float(config["decision"]["maximum_all_early_delta"]),
            "hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "all_hit_retained": float(metrics["all"]["m41_vs_baseline_hit_delta"])
            >= float(config["decision"]["minimum_all_hit_delta"]),
            "anchor_authority_identifiable": anchor_correlation
            >= float(config["decision"]["minimum_anchor_occupied_correlation"]),
            "child_authority_identifiable": child_correlation
            >= float(config["decision"]["minimum_child_occupied_correlation"]),
        }
        total_residual = max(
            max(abs(row["reference_anchor_total"] - row["m41_anchor_total"]) for row in rows),
            max(abs(row["reference_child_total"] - row["m41_child_total"]) for row in rows),
        )
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "m41_development_passed" if all(decisions.values())
                else "m41_development_rejected"
            ),
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "training_history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "holdout_anchor_occupied_correlation": anchor_correlation,
            "holdout_child_occupied_correlation": child_correlation,
            "maximum_family_total_residual": total_residual,
            "mean_holdout_anchor_probabilities": np.mean(
                [row["mean_anchor_probabilities"] for row in rows], axis=0
            ).tolist(),
            "mean_holdout_child_probabilities": np.mean(
                [row["mean_child_probabilities"] for row in rows], axis=0
            ).tolist(),
            "family_surface_measure_conserved": True,
            "geometry_centers_and_scales_frozen": True,
            "train_deploy_categorical_distribution_identical": True,
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
        m37_runner._write_jsonl(run_dir / "HOLDOUT_CONSERVED_MEASURE_ROWS.jsonl", rows)
        m37_runner._write_json(run_dir / "summary.json", summary)
        m37_runner._write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        m37_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m41", "error": f"{type(error).__name__}: {error}"},
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

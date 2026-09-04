"""Refine completion-child authority with native pre-hit free-space survival."""

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
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
import run_worldsim_v71_m35_transmittance_anchor_authority as m35_runner
import run_worldsim_v71_m37_supervised_child_transmittance as m37_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
)


def _child_only_prehit_survival_loss(
    child_logits: torch.Tensor,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    child_occupied = occupied_masses(child_logits)
    child_only = torch.cat(
        [torch.zeros(len(actor["anchors_t"]), device=child_occupied.device), child_occupied]
    )
    midpoints, target_depth, _, _, optical_thickness = m35_runner._ray_distribution(
        actor,
        child_only,
        int(config["maximum_training_rays"]),
        int(config["categorical_train_segments"]),
        float(config["training_cuboid_padding_m"]),
    )
    observed_free = midpoints < (
        target_depth[:, None] - float(config["prehit_free_space_margin_m"])
    )
    return torch.sum(optical_thickness * observed_free, dim=1).mean()


def _losses(
    anchor_authority: EvidentialGaussianAuthority,
    child_authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    occupied, child_logits = m37_runner._occupied(
        anchor_authority, child_authority, actor
    )
    midpoints, target_depth, log_probabilities, no_return, _ = (
        m35_runner._ray_distribution(
            actor,
            occupied,
            int(config["maximum_training_rays"]),
            int(config["categorical_train_segments"]),
            float(config["training_cuboid_padding_m"]),
        )
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
    prehit_free = _child_only_prehit_survival_loss(child_logits, actor, config)
    loss = (
        categorical
        + float(config["categorical_depth_weight"]) * depth_l1
        + float(config["evidential_mass_weight"]) * evidential
        + float(config["prehit_free_space_weight"]) * prehit_free
    )
    return {
        "loss": loss,
        "categorical_nll": categorical,
        "depth_l1": depth_l1,
        "evidential_cross_entropy": evidential,
        "prehit_child_survival_nll": prehit_free,
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
        "prehit_child_survival_nll",
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
        print(json.dumps({"stage": "m38_prehit_survival_train", **row}), flush=True)
    return history


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {"actor_count": len(selected), "ray_count": rays}
        for name in ("baseline", "m37", "prehit"):
            early = sum(int(row[f"{name}_early_count"]) for row in selected)
            hit = sum(int(row[f"{name}_hit_count"]) for row in selected)
            output[f"{name}_early_rate"] = early / rays
            output[f"{name}_hit_rate"] = hit / rays
        output["prehit_vs_baseline_early_delta"] = (
            output["prehit_early_rate"] - output["baseline_early_rate"]
        )
        output["prehit_vs_baseline_hit_delta"] = (
            output["prehit_hit_rate"] - output["baseline_hit_rate"]
        )
        output["prehit_vs_m37_early_delta"] = (
            output["prehit_early_rate"] - output["m37_early_rate"]
        )
        output["prehit_vs_m37_hit_delta"] = (
            output["prehit_hit_rate"] - output["m37_hit_rate"]
        )
        return output

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def _diagnostic_prehit_mass(
    child_authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    child_logits = child_authority(actor["authority_child_features_t"])
    diagnostic_config = dict(config)
    diagnostic_config["maximum_training_rays"] = int(config["diagnostic_rays"])
    diagnostic_config["categorical_train_segments"] = int(config["diagnostic_segments"])
    return float(_child_only_prehit_survival_loss(child_logits, actor, diagnostic_config))


def _load_authority(
    checkpoint: Mapping[str, Any], device: torch.device
) -> EvidentialGaussianAuthority:
    model = EvidentialGaussianAuthority(
        hidden_dim=int(checkpoint["hidden_dim"]), input_dim=int(checkpoint["input_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


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
        raise RuntimeError("M38 requires CUDA")
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
            actor["authority_child_features_t"] = m37_runner._child_features(actor)
            masses, stats = m37_runner._child_evidence_targets(actor, config["model"])
            actor["authority_child_target_masses_t"] = masses
            actor["authority_child_target_stats"] = stats

        anchor_checkpoint = torch.load(
            Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        anchor_authority = _load_authority(anchor_checkpoint, device)
        anchor_authority.eval().requires_grad_(False)
        child_checkpoint = torch.load(
            Path(config["m37_run"]) / "CHILD_MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        m37_reference = _load_authority(child_checkpoint, device)
        m37_reference.eval().requires_grad_(False)
        child_authority = _load_authority(child_checkpoint, device)

        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            child_authority.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        m37_runner._write_json(
            run_dir / "status.json", {"status": "running", "phase": "training"}
        )
        child_authority.train()
        history = _train(
            anchor_authority, child_authority, train_actors, config["model"], optimizer
        )
        child_authority.eval()
        torch.save(
            {
                "state_dict": child_authority.state_dict(),
                "hidden_dim": int(child_checkpoint["hidden_dim"]),
                "input_dim": int(child_checkpoint["input_dim"]),
                "initialized_from": str(Path(config["m37_run"]) / "CHILD_MODEL.pt"),
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
                m37_occupied, m37_logits = m37_runner._occupied(
                    anchor_authority, m37_reference, actor
                )
                m37_result = m35_runner._transmittance_partition(
                    actor, m37_occupied, config["evaluation"]
                )
                prehit_occupied, prehit_logits = m37_runner._occupied(
                    anchor_authority, child_authority, actor
                )
                prehit_result = m35_runner._transmittance_partition(
                    actor, prehit_occupied, config["evaluation"]
                )
                child_masses = torch.softmax(prehit_logits, dim=1)
                target_masses = actor["authority_child_target_masses_t"]
                predicted_occupied_parts.append(child_masses[:, 1].cpu().numpy())
                target_occupied_parts.append(target_masses[:, 1].cpu().numpy())
                rows.append(
                    {
                        "scene_name": m34_runner._scalar_text(actor["scene_name"]),
                        "track_id": m34_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "m37_early_count": int(np.count_nonzero(m37_result["early"])),
                        "m37_hit_count": int(np.count_nonzero(m37_result["hit"])),
                        "prehit_early_count": int(np.count_nonzero(prehit_result["early"])),
                        "prehit_hit_count": int(np.count_nonzero(prehit_result["hit"])),
                        "m37_no_return_probability": m37_result["mean_no_return_probability"],
                        "prehit_no_return_probability": prehit_result["mean_no_return_probability"],
                        "m37_prehit_child_survival_nll": _diagnostic_prehit_mass(
                            m37_reference, actor, config["model"]
                        ),
                        "prehit_child_survival_nll": _diagnostic_prehit_mass(
                            child_authority, actor, config["model"]
                        ),
                        "mean_child_masses": child_masses.mean(dim=0).cpu().tolist(),
                        "target_mass_cross_entropy": float(
                            -torch.sum(
                                target_masses * torch.log_softmax(prehit_logits, dim=1),
                                dim=1,
                            ).mean()
                        ),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m38_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
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
            float(metrics["hazard"]["prehit_vs_baseline_early_delta"]),
            float(metrics["clear"]["prehit_vs_baseline_early_delta"]),
        )
        decisions = {
            "all_early_nonincrease": float(
                metrics["all"]["prehit_vs_baseline_early_delta"]
            ) <= float(config["decision"]["maximum_all_early_delta"]),
            "hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "all_hit_retained": float(metrics["all"]["prehit_vs_baseline_hit_delta"])
            >= float(config["decision"]["minimum_all_hit_delta"]),
            "child_authority_identifiable": occupied_correlation
            >= float(config["decision"]["minimum_child_occupied_correlation"]),
        }
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "m38_development_passed" if all(decisions.values())
                else "m38_development_rejected"
            ),
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
            "m37_mean_prehit_child_survival_nll": float(
                np.mean([row["m37_prehit_child_survival_nll"] for row in rows])
            ),
            "prehit_mean_child_survival_nll": float(
                np.mean([row["prehit_child_survival_nll"] for row in rows])
            ),
            "m37_mean_no_return_probability": float(
                np.mean([row["m37_no_return_probability"] for row in rows])
            ),
            "prehit_mean_no_return_probability": float(
                np.mean([row["prehit_no_return_probability"] for row in rows])
            ),
            "geometry_centers_and_scales_frozen": True,
            "anchor_authority_frozen": True,
            "initialized_from_m37": True,
            "prehit_supervision_source": "native_lidar_observed_free_interval",
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
        m37_runner._write_jsonl(run_dir / "HOLDOUT_PREHIT_ROWS.jsonl", rows)
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
            {"status": "failed", "phase": "m38", "error": f"{type(error).__name__}: {error}"},
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

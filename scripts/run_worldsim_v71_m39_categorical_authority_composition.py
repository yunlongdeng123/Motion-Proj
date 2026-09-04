"""Audit learned Gaussian authority under direct categorical return composition."""

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
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
import run_worldsim_v71_m37_supervised_child_transmittance as m37_runner
import run_worldsim_v71_m38_prehit_free_space_survival as m38_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    occupied_masses,
    weighted_gaussian_energy,
)


def _categorical_partition(
    actor: Mapping[str, Any],
    occupied: torch.Tensor,
    evaluation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    device = actor["features"].device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = m34_runner.field_runner._ray_box_intervals(
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
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "observable": valid_box.cpu().numpy().astype(bool),
        "early": (valid_box & (first_depth < target_depth - tolerance)).cpu().numpy(),
        "hit": (
            valid_box & (torch.abs(first_depth - target_depth) <= tolerance)
        ).cpu().numpy(),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {"actor_count": len(selected), "ray_count": rays}
        for name in ("baseline", "m34", "m37", "m38"):
            early = sum(int(row[f"{name}_early_count"]) for row in selected)
            hit = sum(int(row[f"{name}_hit_count"]) for row in selected)
            output[f"{name}_early_rate"] = early / rays
            output[f"{name}_hit_rate"] = hit / rays
        for name in ("m34", "m37", "m38"):
            output[f"{name}_vs_baseline_early_delta"] = (
                output[f"{name}_early_rate"] - output["baseline_early_rate"]
            )
            output[f"{name}_vs_baseline_hit_delta"] = (
                output[f"{name}_hit_rate"] - output["baseline_hit_rate"]
            )
        output["m38_vs_m37_early_delta"] = (
            output["m38_early_rate"] - output["m37_early_rate"]
        )
        output["m38_vs_m37_hit_delta"] = (
            output["m38_hit_rate"] - output["m37_hit_rate"]
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
        raise RuntimeError("M39 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = m22_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_training_actors"])
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
        stride = int(config["holdout_stride"])
        holdout_actors = []
        for index, actor in enumerate(actors):
            if index % stride != 0:
                continue
            actor["m8_residuals_t"] = actor["m8_residuals_t"].clone()
            m34_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["anchor_scale_m"]),
                config["features"],
                device,
            )
            actor["authority_child_features_t"] = m37_runner._child_features(actor)
            target_masses, _ = m37_runner._child_evidence_targets(actor, config["evidence"])
            actor["authority_child_target_masses_t"] = target_masses
            holdout_actors.append(actor)

        anchor_checkpoint = torch.load(
            Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        anchor_authority = m38_runner._load_authority(anchor_checkpoint, device)
        anchor_authority.eval().requires_grad_(False)
        m37_checkpoint = torch.load(
            Path(config["m37_run"]) / "CHILD_MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        m37_authority = m38_runner._load_authority(m37_checkpoint, device)
        m37_authority.eval().requires_grad_(False)
        m38_checkpoint = torch.load(
            Path(config["m38_run"]) / "CHILD_MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        m38_authority = m38_runner._load_authority(m38_checkpoint, device)
        m38_authority.eval().requires_grad_(False)

        rows: list[dict[str, Any]] = []
        correlations = {"m37": [[], []], "m38": [[], []]}
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = m20_runner._energy_partition(
                    actor, config["evaluation"], float(config["anchor_scale_m"]), device
                )
                anchor_occupied = occupied_masses(
                    anchor_authority(actor["authority_anchor_features_t"])
                )
                unit_children = torch.ones(
                    len(actor["m8_children_t"]), dtype=torch.float32, device=device
                )
                m34_result = _categorical_partition(
                    actor, torch.cat([anchor_occupied, unit_children]), config["evaluation"]
                )
                target_occupied = actor["authority_child_target_masses_t"][:, 1]
                partitions = {}
                child_masses = {}
                for name, model in (("m37", m37_authority), ("m38", m38_authority)):
                    logits = model(actor["authority_child_features_t"])
                    masses = torch.softmax(logits, dim=1)
                    partitions[name] = _categorical_partition(
                        actor,
                        torch.cat([anchor_occupied, masses[:, 1]]),
                        config["evaluation"],
                    )
                    child_masses[name] = masses
                    correlations[name][0].append(masses[:, 1].cpu().numpy())
                    correlations[name][1].append(target_occupied.cpu().numpy())
                rows.append(
                    {
                        "scene_name": m34_runner._scalar_text(actor["scene_name"]),
                        "track_id": m34_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "m34_early_count": int(np.count_nonzero(m34_result["early"])),
                        "m34_hit_count": int(np.count_nonzero(m34_result["hit"])),
                        "m37_early_count": int(np.count_nonzero(partitions["m37"]["early"])),
                        "m37_hit_count": int(np.count_nonzero(partitions["m37"]["hit"])),
                        "m38_early_count": int(np.count_nonzero(partitions["m38"]["early"])),
                        "m38_hit_count": int(np.count_nonzero(partitions["m38"]["hit"])),
                        "m37_mean_child_masses": child_masses["m37"].mean(dim=0).cpu().tolist(),
                        "m38_mean_child_masses": child_masses["m38"].mean(dim=0).cpu().tolist(),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m39_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )

        metrics = _summarize(rows)
        occupied_correlations = {}
        for name, (predicted_parts, target_parts) in correlations.items():
            occupied_correlations[name] = float(
                np.corrcoef(
                    np.concatenate(predicted_parts), np.concatenate(target_parts)
                )[0, 1]
            )
        worst_stratum_delta = max(
            float(metrics["hazard"]["m38_vs_baseline_early_delta"]),
            float(metrics["clear"]["m38_vs_baseline_early_delta"]),
        )
        decisions = {
            "m38_all_early_nonincrease": float(
                metrics["all"]["m38_vs_baseline_early_delta"]
            ) <= float(config["decision"]["maximum_all_early_delta"]),
            "m38_hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "m38_all_hit_retained": float(
                metrics["all"]["m38_vs_baseline_hit_delta"]
            ) >= float(config["decision"]["minimum_all_hit_delta"]),
        }
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "m39_categorical_composition_supported" if all(decisions.values())
                else "m39_categorical_composition_rejected"
            ),
            "holdout_actor_count": len(holdout_actors),
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "child_occupied_correlations": occupied_correlations,
            "training": False,
            "primary_candidate": "m38_authority_under_categorical_composition",
            "m37_arm_descriptive_only": True,
            "geometry_centers_and_scales_frozen": True,
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
        m37_runner._write_jsonl(run_dir / "HOLDOUT_COMPOSITION_ROWS.jsonl", rows)
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
            {"status": "failed", "phase": "m39", "error": f"{type(error).__name__}: {error}"},
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

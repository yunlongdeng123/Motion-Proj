"""Compose frozen M39 authority with GT-supervised M11 oriented kernels."""

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
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m8_runner
import run_worldsim_v71_m11_exact_support_supervision as oriented_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as energy_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as loader_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as anchor_runner
import run_worldsim_v71_m37_supervised_child_transmittance as child_runner
import run_worldsim_v71_m38_prehit_free_space_survival as authority_runner
import run_worldsim_v71_m39_categorical_authority_composition as composition_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import occupied_masses
from motion_proj.worldsim_v71.gaussian_anchor_relocation import OrientedGaussianSeedExpansionMLP


def _oriented_energy(
    queries: torch.Tensor,
    actor: Mapping[str, Any],
    anchor_occupied: torch.Tensor,
    child_occupied: torch.Tensor,
    child_normals: torch.Tensor,
    child_thickness: torch.Tensor,
) -> torch.Tensor:
    anchor_count = len(actor["anchors_t"])
    anchor_scales = actor["authority_scales_t"][:anchor_count]
    anchor_distance = torch.cdist(queries, actor["anchors_t"])
    anchor_components = (
        -0.5 * (anchor_distance / anchor_scales.reshape(1, -1)).square()
        + torch.log(anchor_occupied.clamp_min(1.0e-8)).reshape(1, -1)
    )
    displacement = queries[:, None, :] - actor["m8_children_t"][None, :, :]
    normal_coordinate = torch.sum(displacement * child_normals[None, :, :], dim=-1)
    tangent_sq = (
        torch.sum(displacement.square(), dim=-1) - normal_coordinate.square()
    ).clamp_min(0.0)
    child_components = -0.5 * (
        tangent_sq / actor["m8_scales_t"].square().reshape(1, -1)
        + normal_coordinate.square() / child_thickness.square().reshape(1, -1)
    ) + torch.log(child_occupied.clamp_min(1.0e-8)).reshape(1, -1)
    return torch.logsumexp(
        torch.cat([anchor_components, child_components], dim=1), dim=1
    )


def _partition(
    actor: Mapping[str, Any],
    anchor_occupied: torch.Tensor,
    child_occupied: torch.Tensor,
    child_normals: torch.Tensor,
    child_thickness: torch.Tensor,
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
    entry, exit_depth, valid_box = anchor_runner.field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    fractions = torch.linspace(
        0.0,
        1.0,
        int(evaluation["field_sample_count"]),
        dtype=torch.float32,
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
        energy = _oriented_energy(
            queries.reshape(-1, 3),
            actor,
            anchor_occupied,
            child_occupied,
            child_normals,
            child_thickness,
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
        for name in ("baseline", "m39", "oriented"):
            early = sum(int(row[f"{name}_early_count"]) for row in selected)
            hit = sum(int(row[f"{name}_hit_count"]) for row in selected)
            output[f"{name}_early_rate"] = early / rays
            output[f"{name}_hit_rate"] = hit / rays
        output["oriented_vs_m39_early_delta"] = (
            output["oriented_early_rate"] - output["m39_early_rate"]
        )
        output["oriented_vs_m39_hit_delta"] = (
            output["oriented_hit_rate"] - output["m39_hit_rate"]
        )
        output["oriented_vs_baseline_early_delta"] = (
            output["oriented_early_rate"] - output["baseline_early_rate"]
        )
        output["oriented_vs_baseline_hit_delta"] = (
            output["oriented_hit_rate"] - output["baseline_hit_rate"]
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
    child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M45 requires CUDA")
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
        holdout = []
        for index, actor in enumerate(actors):
            if index % int(config["holdout_stride"]) != 0:
                continue
            anchor_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["anchor_scale_m"]),
                config["features"],
                device,
            )
            actor["authority_child_features_t"] = child_runner._child_features(actor)
            holdout.append(actor)

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
        oriented_checkpoint = torch.load(
            Path(config["m11_run"]) / "MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        oriented_model = OrientedGaussianSeedExpansionMLP(
            int(oriented_checkpoint["input_dim"]),
            int(oriented_checkpoint["hidden_dim"]),
            int(oriented_checkpoint["branch_factor"]),
            int(oriented_checkpoint["slot_dim"]),
        ).to(device)
        oriented_model.load_state_dict(oriented_checkpoint["state_dict"])
        oriented_model.eval().requires_grad_(False)
        oriented_config = yaml.safe_load(
            (Path(config["m11_run"]) / "resolved.yaml").read_text(encoding="utf-8")
        )["model"]

        rows = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout):
                baseline = energy_runner._energy_partition(
                    actor, config["evaluation"], float(config["anchor_scale_m"]), device
                )
                anchor_occupied = occupied_masses(
                    anchor_authority(actor["authority_anchor_features_t"])
                )
                child_occupied = occupied_masses(
                    child_authority(actor["authority_child_features_t"])
                )
                isotropic = composition_runner._categorical_partition(
                    actor,
                    torch.cat([anchor_occupied, child_occupied]),
                    config["evaluation"],
                )
                _, _, normals, thickness = oriented_runner._predict_support(
                    oriented_model, actor, oriented_config
                )
                oriented = _partition(
                    actor,
                    anchor_occupied,
                    child_occupied,
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
                if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                    print(
                        json.dumps(
                            {
                                "stage": "m45_oriented_categorical",
                                "progress": f"{index + 1}/{len(holdout)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        worst_delta = max(
            float(metrics["hazard"]["oriented_vs_m39_early_delta"]),
            float(metrics["clear"]["oriented_vs_m39_early_delta"]),
        )
        decisions = {
            "oriented_all_early_nonincrease": float(
                metrics["all"]["oriented_vs_m39_early_delta"]
            )
            <= float(config["decision"]["maximum_all_early_delta"]),
            "oriented_hazard_and_clear_early_nonincrease": worst_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "oriented_all_hit_retained": float(
                metrics["all"]["oriented_vs_m39_hit_delta"]
            )
            >= float(config["decision"]["minimum_all_hit_delta"]),
        }
        mean_thickness = float(np.mean([row["mean_normal_thickness_m"] for row in rows]))
        mean_anisotropy = float(np.mean([row["mean_anisotropy_ratio"] for row in rows]))
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m45_oriented_categorical_supported"
            if all(decisions.values())
            else "m45_oriented_categorical_rejected",
            "holdout_actor_count": len(holdout),
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_delta,
            "mean_normal_thickness_m": mean_thickness,
            "mean_anisotropy_ratio": mean_anisotropy,
            "training": False,
            "geometry_centers_and_tangent_scales_frozen": True,
            "posthoc_filter": False,
            "model_selection": False,
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
        child_runner._write_jsonl(run_dir / "ORIENTED_COMPOSITION_ROWS.jsonl", rows)
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
            {"status": "failed", "phase": "m45", "error": f"{type(error).__name__}: {error}"},
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

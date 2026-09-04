"""Audit the exact CDF sign boundary for child visibility attenuation."""

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
import run_worldsim_v71_m22_se3_dynamic_static_composition as loader_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as anchor_runner
import run_worldsim_v71_m37_supervised_child_transmittance as child_runner
import run_worldsim_v71_m38_prehit_free_space_survival as authority_runner
import run_worldsim_v71_m48_supervised_child_visibility_measure as m48_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import occupied_masses
from motion_proj.worldsim_v71.gaussian_anchor_relocation import OrientedGaussianSeedExpansionMLP


def _boundary_weights(depths: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    ray_count, bin_count = depths.shape
    right_raw = torch.searchsorted(depths.contiguous(), boundary[:, None]).squeeze(1)
    after = right_raw >= bin_count
    right = right_raw.clamp(0, bin_count - 1)
    previous = (right - 1).clamp_min(0)
    right_depth = depths.gather(1, right[:, None]).squeeze(1)
    previous_depth = depths.gather(1, previous[:, None]).squeeze(1)
    left_depth = torch.where(right > 0, previous_depth, depths[:, 0])
    ratio = ((boundary - left_depth) / (right_depth - left_depth).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    indices = torch.arange(bin_count, device=depths.device).reshape(1, -1)
    weights = (indices < right[:, None]).to(depths.dtype)
    weights.scatter_add_(1, right[:, None], ratio[:, None])
    weights = torch.where(after[:, None], torch.ones_like(weights), weights)
    weights = torch.where((boundary <= depths[:, 0])[:, None], torch.zeros_like(weights), weights)
    return weights


def _actor_audit(
    visibility_model: m48_runner.ChildVisibilityMLP,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    device = actor["features"].device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(actor["target_sensor_origins"], dtype=torch.float32, device=device)
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid = anchor_runner.field_runner._ray_box_intervals(origins, directions, bounds)
    fractions = torch.linspace(
        0.0, 1.0, int(evaluation["field_sample_count"]), dtype=torch.float32, device=device
    )
    threshold = float(evaluation["categorical_median_threshold"])
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    chunk_size = int(evaluation["ray_chunk_size"])
    totals: dict[str, float | int] = {
        "ray_count": len(targets),
        "valid_ray_count": 0,
        "m45_early_count": 0,
        "m48_early_count": 0,
        "added_early_count": 0,
        "removed_early_count": 0,
        "cdf_increase_count": 0,
        "cdf_decrease_count": 0,
        "uniform_child_attenuation_safe_count": 0,
        "uniform_child_attenuation_adverse_count": 0,
        "adverse_pressure_dominant_count": 0,
        "linear_exact_sign_agreement_count": 0,
        "boundary_cdf_delta_sum": 0.0,
        "linear_delta_sum": 0.0,
        "favorable_pressure_sum": 0.0,
        "adverse_pressure_sum": 0.0,
        "maximum_linearization_error": 0.0,
        "visibility_sum": 0.0,
    }
    for start in range(0, len(targets), chunk_size):
        local_origins = origins[start : start + chunk_size]
        local_directions = directions[start : start + chunk_size]
        local_entry = entry[start : start + chunk_size]
        local_exit = exit_depth[start : start + chunk_size]
        local_valid = valid[start : start + chunk_size]
        depths = local_entry[:, None] + (local_exit - local_entry)[:, None] * fractions[None, :]
        queries = local_origins[:, None, :] + depths[:, :, None] * local_directions[:, None, :]
        ray_count, bin_count = depths.shape
        anchor_log, child_log = m48_runner._component_energy(
            queries.reshape(-1, 3),
            actor,
            actor["fixed_anchor_occupied_t"],
            actor["fixed_child_occupied_t"],
            actor["fixed_child_normals_t"],
            actor["fixed_child_thickness_t"],
        )
        anchor_log = anchor_log.reshape(ray_count, bin_count, -1)
        child_log = child_log.reshape(ray_count, bin_count, -1)
        component_log = torch.cat([anchor_log, child_log], dim=2)
        stabilizer = torch.amax(component_log, dim=(1, 2), keepdim=True)
        component_mass = torch.exp(component_log - stabilizer)
        visibility = m48_runner._visibility(
            visibility_model,
            actor,
            local_origins,
            local_directions,
            actor["fixed_child_normals_t"],
            actor["fixed_child_thickness_t"],
            actor["fixed_child_occupied_t"],
        )
        boundary = target_depth[start : start + chunk_size] - tolerance
        boundary_weight = _boundary_weights(depths, boundary)
        component_total = component_mass.sum(dim=1)
        component_pre = (component_mass * boundary_weight[:, :, None]).sum(dim=1)
        total = component_total.sum(dim=1)
        base_cdf = component_pre.sum(dim=1) / total.clamp_min(1.0e-12)
        component_cdf = component_pre / component_total.clamp_min(1.0e-12)
        responsibility = component_total / total[:, None].clamp_min(1.0e-12)
        derivative = responsibility * (component_cdf - base_cdf[:, None])
        child_derivative = derivative[:, len(actor["anchors_t"]) :]
        log_visibility = torch.log(visibility.clamp_min(1.0e-8))
        linear_delta = torch.sum(child_derivative * log_visibility, dim=1)
        attenuation = -log_visibility
        favorable_pressure = torch.sum(attenuation * child_derivative.clamp_min(0.0), dim=1)
        adverse_pressure = torch.sum(attenuation * (-child_derivative).clamp_min(0.0), dim=1)
        uniform_derivative = child_derivative.sum(dim=1)

        new_component_mass = component_mass.clone()
        new_component_mass[:, :, len(actor["anchors_t"]) :] *= visibility[:, None, :]
        new_total = new_component_mass.sum(dim=(1, 2))
        new_pre = (new_component_mass * boundary_weight[:, :, None]).sum(dim=(1, 2))
        new_cdf = new_pre / new_total.clamp_min(1.0e-12)
        exact_delta = new_cdf - base_cdf
        base_early = local_valid & (base_cdf > threshold)
        new_early = local_valid & (new_cdf > threshold)
        selected = local_valid
        totals["valid_ray_count"] += int(torch.count_nonzero(selected))
        totals["m45_early_count"] += int(torch.count_nonzero(base_early))
        totals["m48_early_count"] += int(torch.count_nonzero(new_early))
        totals["added_early_count"] += int(torch.count_nonzero(new_early & ~base_early))
        totals["removed_early_count"] += int(torch.count_nonzero(base_early & ~new_early))
        totals["cdf_increase_count"] += int(torch.count_nonzero(selected & (exact_delta > 0.0)))
        totals["cdf_decrease_count"] += int(torch.count_nonzero(selected & (exact_delta < 0.0)))
        totals["uniform_child_attenuation_safe_count"] += int(
            torch.count_nonzero(selected & (uniform_derivative > 0.0))
        )
        totals["uniform_child_attenuation_adverse_count"] += int(
            torch.count_nonzero(selected & (uniform_derivative < 0.0))
        )
        totals["adverse_pressure_dominant_count"] += int(
            torch.count_nonzero(selected & (adverse_pressure > favorable_pressure))
        )
        totals["linear_exact_sign_agreement_count"] += int(
            torch.count_nonzero(selected & (linear_delta * exact_delta >= 0.0))
        )
        totals["boundary_cdf_delta_sum"] += float(exact_delta[selected].sum())
        totals["linear_delta_sum"] += float(linear_delta[selected].sum())
        totals["favorable_pressure_sum"] += float(favorable_pressure[selected].sum())
        totals["adverse_pressure_sum"] += float(adverse_pressure[selected].sum())
        totals["maximum_linearization_error"] = max(
            float(totals["maximum_linearization_error"]),
            float(torch.max(torch.abs(exact_delta[selected] - linear_delta[selected]))),
        )
        totals["visibility_sum"] += float(visibility.mean(dim=1)[selected].sum())
    return {
        "scene_name": anchor_runner._scalar_text(actor["scene_name"]),
        "track_id": anchor_runner._scalar_text(actor["track_id"]),
        "hazardous": bool(actor["hazardous"]),
        **totals,
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        integers = (
            "ray_count", "valid_ray_count", "m45_early_count", "m48_early_count",
            "added_early_count", "removed_early_count", "cdf_increase_count", "cdf_decrease_count",
            "uniform_child_attenuation_safe_count", "uniform_child_attenuation_adverse_count",
            "adverse_pressure_dominant_count", "linear_exact_sign_agreement_count",
        )
        sums = {name: sum(int(row[name]) for row in selected) for name in integers}
        valid = sums["valid_ray_count"]
        float_sums = {
            name: sum(float(row[name]) for row in selected)
            for name in (
                "boundary_cdf_delta_sum", "linear_delta_sum", "favorable_pressure_sum",
                "adverse_pressure_sum", "visibility_sum",
            )
        }
        return {
            "actor_count": len(selected),
            **sums,
            "m48_vs_m45_early_delta": (sums["m48_early_count"] - sums["m45_early_count"]) / sums["ray_count"],
            "added_early_rate": sums["added_early_count"] / sums["ray_count"],
            "removed_early_rate": sums["removed_early_count"] / sums["ray_count"],
            "cdf_increase_fraction": sums["cdf_increase_count"] / valid,
            "cdf_decrease_fraction": sums["cdf_decrease_count"] / valid,
            "uniform_child_attenuation_safe_fraction": sums["uniform_child_attenuation_safe_count"] / valid,
            "uniform_child_attenuation_adverse_fraction": sums["uniform_child_attenuation_adverse_count"] / valid,
            "adverse_pressure_dominant_fraction": sums["adverse_pressure_dominant_count"] / valid,
            "linear_exact_sign_agreement": sums["linear_exact_sign_agreement_count"] / valid,
            "mean_boundary_cdf_delta": float_sums["boundary_cdf_delta_sum"] / valid,
            "mean_linear_delta": float_sums["linear_delta_sum"] / valid,
            "mean_favorable_pressure": float_sums["favorable_pressure_sum"] / valid,
            "mean_adverse_pressure": float_sums["adverse_pressure_sum"] / valid,
            "mean_visibility": float_sums["visibility_sum"] / valid,
            "maximum_linearization_error": max(float(row["maximum_linearization_error"]) for row in selected),
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
    child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M49 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = loader_runner._load_m8(config, device)
        actors = [
            actor
            for path in m0_runner._paths(Path(config["cache_root"]), int(config["maximum_training_actors"]))
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, actor["m5_centers_t"] = m5_runner._move(base, actor, base_config)
                actor["m8_children_t"], actor["m8_residuals_t"], actor["m8_scales_t"] = m8_runner._predict(
                    surface, actor, surface_config
                )
        holdout = []
        for index, actor in enumerate(actors):
            if index % int(config["holdout_stride"]) != 0:
                continue
            anchor_runner._attach_frozen_authority_state(
                actor, Path(config["sidecar_root"]), float(config["anchor_scale_m"]), config["features"], device
            )
            actor["authority_child_features_t"] = child_runner._child_features(actor)
            holdout.append(actor)
        anchor_checkpoint = torch.load(Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        anchor_authority = authority_runner._load_authority(anchor_checkpoint, device)
        anchor_authority.eval().requires_grad_(False)
        child_checkpoint = torch.load(Path(config["m38_run"]) / "CHILD_MODEL.pt", map_location=device, weights_only=False)
        child_authority = authority_runner._load_authority(child_checkpoint, device)
        child_authority.eval().requires_grad_(False)
        support_checkpoint = torch.load(Path(config["m11_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        support_model = OrientedGaussianSeedExpansionMLP(
            int(support_checkpoint["input_dim"]), int(support_checkpoint["hidden_dim"]),
            int(support_checkpoint["branch_factor"]), int(support_checkpoint["slot_dim"]),
        ).to(device)
        support_model.load_state_dict(support_checkpoint["state_dict"])
        support_model.eval().requires_grad_(False)
        support_config = yaml.safe_load((Path(config["m11_run"]) / "resolved.yaml").read_text(encoding="utf-8"))["model"]
        visibility_checkpoint = torch.load(Path(config["m48_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        visibility_model = m48_runner.ChildVisibilityMLP(int(visibility_checkpoint["hidden_dim"]), 0.95).to(device)
        visibility_model.load_state_dict(visibility_checkpoint["state_dict"])
        visibility_model.eval().requires_grad_(False)
        with torch.inference_mode():
            for actor in holdout:
                actor["fixed_anchor_occupied_t"] = occupied_masses(anchor_authority(actor["authority_anchor_features_t"]))
                actor["fixed_child_occupied_t"] = occupied_masses(child_authority(actor["authority_child_features_t"]))
                _, _, actor["fixed_child_normals_t"], actor["fixed_child_thickness_t"] = oriented_runner._predict_support(
                    support_model, actor, support_config
                )
            rows = []
            for index, actor in enumerate(holdout):
                rows.append(_actor_audit(visibility_model, actor, config["evaluation"]))
                if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                    print(json.dumps({"stage": "m49_sign_boundary", "progress": f"{index + 1}/{len(holdout)}"}), flush=True)
        metrics = _summarize(rows)
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "visibility_attenuation_sign_boundary_confirmed",
            "holdout_actor_count": len(holdout),
            "identity": "dC/dlog(w_j)=r_j*(C_j-C); attenuation lowers pre-boundary CDF iff C_j>C",
            "metrics": metrics,
            "training": False,
            "decision_gate": False,
            "model_selection": False,
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
        child_runner._write_jsonl(run_dir / "VISIBILITY_SIGN_ROWS.jsonl", rows)
        child_runner._write_json(run_dir / "summary.json", summary)
        child_runner._write_json(
            run_dir / "status.json", {"status": "done", "phase": "sign_boundary", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        )
        return summary
    except Exception as error:
        child_runner._write_json(
            run_dir / "status.json", {"status": "failed", "phase": "m49", "error": f"{type(error).__name__}: {error}"}
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

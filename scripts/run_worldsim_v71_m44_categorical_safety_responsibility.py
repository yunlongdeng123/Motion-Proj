"""Explain M39 categorical safety by exact family responsibility accounting."""

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
import run_worldsim_v71_m22_se3_dynamic_static_composition as loader_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as anchor_runner
import run_worldsim_v71_m37_supervised_child_transmittance as child_runner
import run_worldsim_v71_m38_prehit_free_space_survival as authority_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import occupied_masses


def _interpolate_cdf(
    depths: torch.Tensor,
    sample_mass: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    cdf = sample_mass.cumsum(dim=1)
    sample_count = depths.shape[1]
    right = torch.searchsorted(depths.contiguous(), boundary[:, None]).squeeze(1)
    after = right >= sample_count
    right = right.clamp(0, sample_count - 1)
    previous = (right - 1).clamp_min(0)
    right_depth = depths.gather(1, right[:, None]).squeeze(1)
    left_depth_gathered = depths.gather(1, previous[:, None]).squeeze(1)
    left_depth = torch.where(right > 0, left_depth_gathered, depths[:, 0])
    left_cdf_gathered = cdf.gather(1, previous[:, None]).squeeze(1)
    left_cdf = torch.where(right > 0, left_cdf_gathered, torch.zeros_like(boundary))
    right_mass = sample_mass.gather(1, right[:, None]).squeeze(1)
    ratio = (boundary - left_depth) / (right_depth - left_depth).clamp_min(1.0e-6)
    interpolated = left_cdf + ratio.clamp(0.0, 1.0) * right_mass
    interpolated = torch.where(boundary <= depths[:, 0], torch.zeros_like(interpolated), interpolated)
    return torch.where(after, cdf[:, -1], interpolated)


def _measure(
    actor: Mapping[str, Any],
    occupied: torch.Tensor,
    evaluation: Mapping[str, Any],
) -> dict[str, np.ndarray | float]:
    device = occupied.device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid = anchor_runner.field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    fractions = torch.linspace(
        0.0,
        1.0,
        int(evaluation["field_sample_count"]),
        dtype=torch.float32,
        device=device,
    )
    anchor_count = len(actor["anchors_t"])
    threshold = float(evaluation["categorical_median_threshold"])
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    chunk = int(evaluation["ray_chunk_size"])
    outputs: dict[str, list[torch.Tensor]] = {
        "valid": [],
        "early": [],
        "boundary_early": [],
        "boundary_cdf": [],
        "anchor_pre": [],
        "child_pre": [],
    }
    maximum_residual = 0.0
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
        normalized = (
            queries[:, :, None, :] - actor["authority_centers_t"][None, None, :, :]
        ) / actor["authority_scales_t"][None, None, :, None]
        log_components = -0.5 * normalized.square().sum(dim=-1) + torch.log(
            occupied.clamp_min(1.0e-8)
        )[None, None, :]
        log_normalizer = torch.logsumexp(
            log_components.reshape(len(depths), -1), dim=1
        )
        joint = torch.exp(log_components - log_normalizer[:, None, None])
        sample_mass = joint.sum(dim=2)
        anchor_sample_mass = joint[:, :, :anchor_count].sum(dim=2)
        child_sample_mass = joint[:, :, anchor_count:].sum(dim=2)
        cdf = sample_mass.cumsum(dim=1)
        median_index = (cdf >= threshold).to(torch.int64).argmax(dim=1)
        previous = (median_index - 1).clamp_min(0)
        right_cdf = cdf.gather(1, median_index[:, None]).squeeze(1)
        left_cdf_gathered = cdf.gather(1, previous[:, None]).squeeze(1)
        left_cdf = torch.where(
            median_index > 0, left_cdf_gathered, torch.zeros_like(right_cdf)
        )
        right_depth = depths.gather(1, median_index[:, None]).squeeze(1)
        left_depth_gathered = depths.gather(1, previous[:, None]).squeeze(1)
        left_depth = torch.where(median_index > 0, left_depth_gathered, local_entry)
        ratio = (threshold - left_cdf) / (right_cdf - left_cdf).clamp_min(1.0e-6)
        first_depth = left_depth + ratio.clamp(0.0, 1.0) * (right_depth - left_depth)
        boundary = target_depth[start : start + chunk] - tolerance
        boundary_cdf = _interpolate_cdf(depths, sample_mass, boundary)
        anchor_pre = _interpolate_cdf(depths, anchor_sample_mass, boundary)
        child_pre = _interpolate_cdf(depths, child_sample_mass, boundary)
        maximum_residual = max(
            maximum_residual,
            float(torch.max(torch.abs(boundary_cdf - anchor_pre - child_pre))),
        )
        local_valid = valid[start : start + chunk]
        outputs["valid"].append(local_valid)
        outputs["early"].append(local_valid & (first_depth < boundary))
        outputs["boundary_early"].append(local_valid & (boundary_cdf > threshold))
        outputs["boundary_cdf"].append(boundary_cdf)
        outputs["anchor_pre"].append(anchor_pre)
        outputs["child_pre"].append(child_pre)
    return {
        name: torch.cat(values).cpu().numpy() for name, values in outputs.items()
    } | {"maximum_decomposition_residual": maximum_residual}


def _actor_row(
    actor: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    valid = np.asarray(candidate["valid"], dtype=bool)
    baseline_early = np.asarray(baseline["early"], dtype=bool)
    candidate_early = np.asarray(candidate["early"], dtype=bool)
    baseline_boundary = np.asarray(baseline["boundary_cdf"], dtype=np.float64)
    candidate_boundary = np.asarray(candidate["boundary_cdf"], dtype=np.float64)
    candidate_safe = valid & ~candidate_early
    return {
        "scene_name": anchor_runner._scalar_text(actor["scene_name"]),
        "track_id": anchor_runner._scalar_text(actor["track_id"]),
        "hazardous": bool(actor["hazardous"]),
        "ray_count": int(len(valid)),
        "valid_ray_count": int(np.count_nonzero(valid)),
        "baseline_early_count": int(np.count_nonzero(baseline_early)),
        "m39_early_count": int(np.count_nonzero(candidate_early)),
        "baseline_boundary_agreement_count": int(
            np.count_nonzero(
                baseline_early[valid]
                == np.asarray(baseline["boundary_early"], dtype=bool)[valid]
            )
        ),
        "m39_boundary_agreement_count": int(
            np.count_nonzero(
                candidate_early[valid]
                == np.asarray(candidate["boundary_early"], dtype=bool)[valid]
            )
        ),
        "baseline_boundary_cdf_sum": float(baseline_boundary[valid].sum()),
        "m39_boundary_cdf_sum": float(candidate_boundary[valid].sum()),
        "baseline_anchor_pre_sum": float(
            np.asarray(baseline["anchor_pre"], dtype=np.float64)[valid].sum()
        ),
        "baseline_child_pre_sum": float(
            np.asarray(baseline["child_pre"], dtype=np.float64)[valid].sum()
        ),
        "m39_anchor_pre_sum": float(
            np.asarray(candidate["anchor_pre"], dtype=np.float64)[valid].sum()
        ),
        "m39_child_pre_sum": float(
            np.asarray(candidate["child_pre"], dtype=np.float64)[valid].sum()
        ),
        "m39_early_boundary_cdf_sum": float(candidate_boundary[candidate_early].sum()),
        "m39_safe_boundary_cdf_sum": float(candidate_boundary[candidate_safe].sum()),
        "m39_safe_count": int(np.count_nonzero(candidate_safe)),
        "baseline_maximum_decomposition_residual": float(
            baseline["maximum_decomposition_residual"]
        ),
        "m39_maximum_decomposition_residual": float(
            candidate["maximum_decomposition_residual"]
        ),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        valid = sum(int(row["valid_ray_count"]) for row in selected)
        baseline_early = sum(int(row["baseline_early_count"]) for row in selected)
        m39_early = sum(int(row["m39_early_count"]) for row in selected)
        m39_safe = sum(int(row["m39_safe_count"]) for row in selected)
        baseline_anchor = sum(float(row["baseline_anchor_pre_sum"]) for row in selected)
        baseline_child = sum(float(row["baseline_child_pre_sum"]) for row in selected)
        m39_anchor = sum(float(row["m39_anchor_pre_sum"]) for row in selected)
        m39_child = sum(float(row["m39_child_pre_sum"]) for row in selected)
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "valid_ray_count": valid,
            "baseline_early_rate": baseline_early / rays,
            "m39_early_rate": m39_early / rays,
            "m39_vs_baseline_early_delta": (m39_early - baseline_early) / rays,
            "baseline_boundary_cdf_mean": sum(
                float(row["baseline_boundary_cdf_sum"]) for row in selected
            )
            / valid,
            "m39_boundary_cdf_mean": sum(
                float(row["m39_boundary_cdf_sum"]) for row in selected
            )
            / valid,
            "boundary_cdf_delta": sum(
                float(row["m39_boundary_cdf_sum"])
                - float(row["baseline_boundary_cdf_sum"])
                for row in selected
            )
            / valid,
            "baseline_anchor_pre_mean": baseline_anchor / valid,
            "baseline_child_pre_mean": baseline_child / valid,
            "m39_anchor_pre_mean": m39_anchor / valid,
            "m39_child_pre_mean": m39_child / valid,
            "anchor_pre_delta": (m39_anchor - baseline_anchor) / valid,
            "child_pre_delta": (m39_child - baseline_child) / valid,
            "m39_early_boundary_cdf_mean": sum(
                float(row["m39_early_boundary_cdf_sum"]) for row in selected
            )
            / max(m39_early, 1),
            "m39_safe_boundary_cdf_mean": sum(
                float(row["m39_safe_boundary_cdf_sum"]) for row in selected
            )
            / max(m39_safe, 1),
            "baseline_boundary_equivalence_rate": sum(
                int(row["baseline_boundary_agreement_count"]) for row in selected
            )
            / valid,
            "m39_boundary_equivalence_rate": sum(
                int(row["m39_boundary_agreement_count"]) for row in selected
            )
            / valid,
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
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M44 requires CUDA")
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

        rows = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout):
                anchor_occupied = occupied_masses(
                    anchor_authority(actor["authority_anchor_features_t"])
                )
                child_occupied = occupied_masses(
                    child_authority(actor["authority_child_features_t"])
                )
                candidate_occupied = torch.cat([anchor_occupied, child_occupied])
                baseline = _measure(
                    actor, torch.ones_like(candidate_occupied), config["evaluation"]
                )
                candidate = _measure(actor, candidate_occupied, config["evaluation"])
                rows.append(_actor_row(actor, baseline, candidate))
                if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                    print(
                        json.dumps(
                            {
                                "stage": "m44_safety_responsibility",
                                "progress": f"{index + 1}/{len(holdout)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        maximum_residual = max(
            max(float(row["baseline_maximum_decomposition_residual"]) for row in rows),
            max(float(row["m39_maximum_decomposition_residual"]) for row in rows),
        )
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m44_exact_categorical_safety_responsibility_completed",
            "holdout_actor_count": len(holdout),
            "metrics": metrics,
            "maximum_family_decomposition_residual": maximum_residual,
            "identity": "median is early iff interpolated CDF at d_gt-tolerance exceeds 0.5",
            "family_measure": "p(depth_bin, primitive) normalized over all bins and primitives",
            "training": False,
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
        child_runner._write_jsonl(run_dir / "SAFETY_RESPONSIBILITY_ROWS.jsonl", rows)
        child_runner._write_json(run_dir / "summary.json", summary)
        child_runner._write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "holdout_explanation",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        child_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m44", "error": f"{type(error).__name__}: {error}"},
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

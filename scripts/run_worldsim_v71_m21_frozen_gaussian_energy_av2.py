"""Evaluate frozen M8 Gaussian energy on the untouched V7.1 AV2 cohort."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m5_av2_zero_shot as m5_external_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_av2_zero_shot as m8_external_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as energy_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.dataset_av2 import compile_av2_log_v71, load_frozen_av2_cohort
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _evaluate_bundle(
    bundle: Mapping[str, Any],
    base: RaySurfaceRelocationMLP,
    model: GaussianSeedExpansionMLP,
    standardizer: FeatureStandardizer,
    base_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    anchor_scale_m: float,
    device: torch.device,
) -> dict[str, Any] | None:
    actor = m8_external_runner._prepare_bundle_actor(bundle, standardizer, device)
    if actor is None:
        return None
    diagnostics = bundle["diagnostics"]
    with torch.inference_mode():
        _, base_centers = m5_runner._move(base, actor, base_config)
        actor["m5_centers_t"] = base_centers
        children, residuals, scales = m7_runner._predict(model, actor, model_config)
        actor["m8_children_t"] = children
        actor["m8_scales_t"] = scales
        output = torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy()
    target = np.asarray(diagnostics["target"], dtype=np.float32)
    origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32)
    output = _voxel_unique(output, float(evaluation["output_voxel_size_m"]))
    row = evaluate_actor_surface(
        np.asarray(diagnostics["compiled"], dtype=np.float32),
        output,
        target,
        origins,
        hazardous=bool(bundle["row"]["hazardous"]),
        device=device,
        lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
        depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
        distance_chunk_size=int(evaluation["distance_chunk_size"]),
    )
    actor["target"] = target
    actor["target_sensor_origins"] = origins
    with torch.inference_mode():
        partition = energy_runner._energy_partition(
            actor, evaluation, anchor_scale_m, device
        )
    row.update(
        {
            "scene_name": bundle["scene_name"],
            "track_id": str(bundle["row"]["track_id"]),
            "category": str(bundle["row"]["category"]),
            "candidate_count": len(actor["candidates"]),
            "generated_child_count": int(len(children)),
            "mean_child_residual_m": float(
                torch.linalg.vector_norm(residuals, dim=1).mean()
            ),
            "mean_gaussian_scale_m": float(scales.mean()),
            "energy_early_count": int(np.count_nonzero(partition["early"])),
            "energy_hit_count": int(np.count_nonzero(partition["hit"])),
            "energy_observable_count": int(
                np.count_nonzero(partition["observable"])
            ),
        }
    )
    return row


def _energy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[dict[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["target_ray_count"]) for row in selected)
        baseline_early = sum(int(row["baseline_early_count"]) for row in selected)
        energy_early = sum(int(row["energy_early_count"]) for row in selected)
        baseline_hit = sum(int(row["baseline_hit_count"]) for row in selected)
        energy_hit = sum(int(row["energy_hit_count"]) for row in selected)
        baseline_early_rate = baseline_early / rays if rays else None
        energy_early_rate = energy_early / rays if rays else None
        baseline_hit_rate = baseline_hit / rays if rays else None
        energy_hit_rate = energy_hit / rays if rays else None
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "baseline_early_rate": baseline_early_rate,
            "energy_early_rate": energy_early_rate,
            "relative_early_reduction": (
                (baseline_early_rate - energy_early_rate) / baseline_early_rate
                if baseline_early_rate is not None and baseline_early_rate > 0.0
                else None
            ),
            "baseline_hit_recall": baseline_hit_rate,
            "energy_hit_recall": energy_hit_rate,
            "hit_recall_delta": (
                energy_hit_rate - baseline_hit_rate
                if baseline_hit_rate is not None
                else None
            ),
            "energy_observable_rate": (
                sum(int(row["energy_observable_count"]) for row in selected) / rays
                if rays
                else None
            ),
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = load_frozen_av2_cohort(repo_root / config["cohort_config"])
    if len(cohort["logs"]) != int(config["expected_log_count"]):
        raise RuntimeError("frozen V7.1 AV2 cohort count changed")
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    energy_runner.m0_runner._deep_update(compiler, config["compiler_overrides"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    m5_external_runner._write_json(
        run_dir / "status.json",
        {"status": "running", "phase": "waiting_fresh_av2", "completed_logs": 0},
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M21 AV2 zero-shot requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        model_run = Path(config["model_run"])
        checkpoint = torch.load(model_run / "MODEL.pt", map_location=device, weights_only=False)
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model_config = yaml.safe_load(
            (model_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        m5_run = Path(checkpoint["m5_run"])
        m5_checkpoint = torch.load(
            m5_run / "MODEL.pt", map_location=device, weights_only=False
        )
        m5_config = yaml.safe_load(
            (m5_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        base = RaySurfaceRelocationMLP(
            int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
        ).to(device)
        base.load_state_dict(m5_checkpoint["state_dict"])
        base.eval()
        state = Path(config["download_state"])
        rows: list[dict[str, Any]] = []
        processed_logs: list[str] = []
        for position, cohort_row in enumerate(cohort["logs"]):
            log_id = str(cohort_row["log_id"])
            marker = state / f"{log_id}.complete"
            wait_started = time.monotonic()
            while not marker.is_file():
                if time.monotonic() - wait_started > int(config["wait_timeout_seconds"]):
                    raise TimeoutError(f"download timeout for {log_id}")
                m5_external_runner._write_json(
                    run_dir / "status.json",
                    {
                        "status": "running",
                        "phase": "waiting_fresh_av2",
                        "completed_logs": len(processed_logs),
                        "current_log": log_id,
                    },
                )
                time.sleep(int(config["poll_seconds"]))
            bundles = compile_av2_log_v71(
                Path(compiler["dataset_root"]) / log_id, compiler, device
            )
            log_rows = 0
            for bundle in bundles:
                bundle["scene_name"] = log_id
                row = _evaluate_bundle(
                    bundle,
                    base,
                    model,
                    standardizer,
                    m5_config["model"],
                    model_config["model"],
                    config["evaluation"],
                    float(config["anchor_scale_m"]),
                    device,
                )
                if row is None:
                    continue
                row["log_id"] = log_id
                row["external_role"] = str(cohort_row["role"])
                rows.append(row)
                log_rows += 1
            processed_logs.append(log_id)
            m5_external_runner._write_jsonl(
                run_dir / "EXTERNAL_ACTORS.partial.jsonl", rows
            )
            m5_external_runner._write_json(
                run_dir / "status.json",
                {
                    "status": "running",
                    "phase": "fresh_av2_evaluation",
                    "completed_logs": len(processed_logs),
                    "current_log": log_id,
                    "actor_rows": len(rows),
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "m21_fresh_av2",
                        "progress": f"{position + 1}/{len(cohort['logs'])}",
                        "log_id": log_id,
                        "log_actors": log_rows,
                        "total_actors": len(rows),
                    }
                ),
                flush=True,
            )
        point_metrics = summarize_surface_rows(rows)
        energy_metrics = _energy_summary(rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["log_id"])].append(row)
        per_log = {
            log_id: {
                "m8_point_surface": summarize_surface_rows(log_rows),
                "frozen_gaussian_energy": _energy_summary(log_rows),
            }
            for log_id, log_rows in grouped.items()
        }
        decisions = {
            "energy_hazard_early_reduction": float(
                energy_metrics["hazard"]["relative_early_reduction"]
            )
            >= float(config["decision"]["minimum_energy_hazard_relative_reduction"]),
            "energy_all_hit_retained": float(energy_metrics["all"]["hit_recall_delta"])
            >= float(config["decision"]["minimum_energy_all_hit_recall_delta"]),
            "point_chamfer_non_degradation": float(point_metrics["chamfer_delta_m"])
            <= float(config["decision"]["maximum_point_chamfer_delta_m"]),
            "actor_state_retention": float(point_metrics["minimum_actor_state_retention"])
            >= float(config["decision"]["required_actor_state_retention"]),
            "hazard_state_retention": float(point_metrics["minimum_hazard_state_retention"])
            >= float(config["decision"]["required_hazard_state_retention"]),
        }
        passed = all(decisions.values())
        m5_external_runner._write_jsonl(run_dir / "EXTERNAL_ACTORS.jsonl", rows)
        summary = {
            "schema_version": config["summary_schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": config["supported_verdict"] if passed else config["rejected_verdict"],
            "cohort_config": config["cohort_config"],
            "log_count": len(processed_logs),
            "actor_count": len(rows),
            "processed_logs": processed_logs,
            "m8_point_surface": point_metrics,
            "frozen_gaussian_energy": energy_metrics,
            "per_log": per_log,
            "decisions": decisions,
            "model_checkpoint": str(model_run),
            "anchor_scale_m": float(config["anchor_scale_m"]),
            "fine_tuning": False,
            "calibration": False,
            "threshold_selection": False,
            "failed_log_deletion": False,
            "partial_metric_read": False,
            "selection_read": False,
            "source_final_read": False,
            "external_read": True,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        m5_external_runner._write_json(run_dir / "summary.json", summary)
        m5_external_runner._write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "fresh_av2_evaluation",
                "completed_logs": len(processed_logs),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        m5_external_runner._write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "fresh_av2_evaluation",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


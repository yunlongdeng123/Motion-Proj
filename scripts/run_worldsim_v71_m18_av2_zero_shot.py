"""Evaluate frozen M18 categorical first returns on the fresh AV2 cohort."""

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
import run_worldsim_v71_m13_local_signed_field as query_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.dataset_av2 import compile_av2_log_v71, load_frozen_av2_cohort
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.local_signed_field import RayTerminationLogitField
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _evaluate_bundle(
    bundle: Mapping[str, Any],
    base: RaySurfaceRelocationMLP,
    m8_model: GaussianSeedExpansionMLP,
    field_model: RayTerminationLogitField,
    standardizer: FeatureStandardizer,
    base_config: Mapping[str, Any],
    m8_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    actor = m8_external_runner._prepare_bundle_actor(bundle, standardizer, device)
    if actor is None:
        return None
    diagnostics = bundle["diagnostics"]
    with torch.inference_mode():
        _, base_centers = m5_runner._move(base, actor, base_config)
        actor["m5_centers_t"] = base_centers
        children, residuals, scales = m7_runner._predict(
            m8_model, actor, m8_config
        )
        actor["m8_children_t"] = children
        actor["m8_scales_t"] = scales
        output = torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy()
    target = np.asarray(diagnostics["target"], dtype=np.float32)
    target_origins = np.asarray(
        diagnostics["target_sensor_origins"], dtype=np.float32
    )
    output = _voxel_unique(output, float(evaluation["output_voxel_size_m"]))
    row = evaluate_actor_surface(
        np.asarray(diagnostics["compiled"], dtype=np.float32),
        output,
        target,
        target_origins,
        hazardous=bool(bundle["row"]["hazardous"]),
        device=device,
        lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
        depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
        distance_chunk_size=int(evaluation["distance_chunk_size"]),
    )
    actor["target"] = target
    actor["target_sensor_origins"] = target_origins
    partition = query_runner._categorical_first_return_partition(
        field_model, actor, evaluation, device
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
            "field_early_count": int(np.count_nonzero(partition["early"])),
            "field_hit_count": int(np.count_nonzero(partition["hit"])),
            "field_observable_count": int(
                np.count_nonzero(partition["observable"])
            ),
        }
    )
    return row


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = load_frozen_av2_cohort(repo_root / config["cohort_config"])
    if len(cohort["logs"]) != int(config["expected_log_count"]):
        raise RuntimeError("frozen V7.1 AV2 cohort count changed")
    compiler = yaml.safe_load(
        (repo_root / config["p2_config"]).read_text(encoding="utf-8")
    )
    query_runner.m0_runner._deep_update(compiler, config["compiler_overrides"])
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
        raise RuntimeError("M18 AV2 zero-shot requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        model_run = Path(config["model_run"])
        checkpoint = torch.load(
            model_run / "MODEL.pt", map_location=device, weights_only=False
        )
        if str(checkpoint["field"]) != "categorical_first_return":
            raise RuntimeError("M18 checkpoint field type changed")
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        field_model = RayTerminationLogitField(
            int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            branch_factor=int(checkpoint["branch_factor"]),
            slot_dim=int(checkpoint["slot_dim"]),
            latent_dim=int(checkpoint["latent_dim"]),
            neighbor_count=int(checkpoint["neighbor_count"]),
            maximum_residual_fraction=float(checkpoint["maximum_residual_fraction"]),
        ).to(device)
        field_model.load_state_dict(checkpoint["state_dict"])
        field_model.eval()
        m8_run = Path(checkpoint["m8_run"])
        m8_checkpoint = torch.load(
            m8_run / "MODEL.pt", map_location=device, weights_only=False
        )
        m8_config = yaml.safe_load(
            (m8_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        m8_model = GaussianSeedExpansionMLP(
            int(m8_checkpoint["input_dim"]),
            int(m8_checkpoint["hidden_dim"]),
            int(m8_checkpoint["branch_factor"]),
            int(m8_checkpoint["slot_dim"]),
        ).to(device)
        m8_model.load_state_dict(m8_checkpoint["state_dict"])
        m8_model.eval()
        m5_run = Path(m8_checkpoint["m5_run"])
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
                    m8_model,
                    field_model,
                    standardizer,
                    m5_config["model"],
                    m8_config["model"],
                    config["evaluation"],
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
                        "stage": "m18_fresh_av2",
                        "progress": f"{position + 1}/{len(cohort['logs'])}",
                        "log_id": log_id,
                        "log_actors": log_rows,
                        "total_actors": len(rows),
                    }
                ),
                flush=True,
            )
        point_metrics = summarize_surface_rows(rows)
        categorical_metrics = query_runner._field_summary(rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["log_id"])].append(row)
        per_log = {
            log_id: {
                "m8_point_surface": summarize_surface_rows(log_rows),
                "categorical_first_return": query_runner._field_summary(log_rows),
            }
            for log_id, log_rows in grouped.items()
        }
        decisions = m5_external_runner._external_decisions(
            point_metrics, config["decision"]
        )
        hazard_reduction = categorical_metrics["hazard"][
            "relative_early_reduction"
        ]
        decisions["hazard_categorical_relative_early_reduction"] = (
            hazard_reduction is not None
            and float(hazard_reduction)
            >= float(config["decision"]["minimum_field_hazard_relative_reduction"])
        )
        hit_delta = categorical_metrics["all"]["hit_recall_delta"]
        decisions["categorical_hit_recall_retained"] = (
            hit_delta is not None
            and float(hit_delta)
            >= float(config["decision"]["minimum_field_hit_recall_delta"])
        )
        passed = all(decisions.values())
        m5_external_runner._write_jsonl(run_dir / "EXTERNAL_ACTORS.jsonl", rows)
        summary = {
            "schema_version": config["summary_schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": config["supported_verdict"]
            if passed
            else config["rejected_verdict"],
            "cohort_config": config["cohort_config"],
            "log_count": len(processed_logs),
            "actor_count": len(rows),
            "processed_logs": processed_logs,
            "m8_point_surface": point_metrics,
            "categorical_first_return": categorical_metrics,
            "per_log": per_log,
            "decisions": decisions,
            "model_checkpoint": str(model_run),
            "m8_checkpoint": str(m8_run),
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
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
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

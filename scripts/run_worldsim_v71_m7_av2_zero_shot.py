"""Evaluate frozen supervision-native M7 as fresh nuScenes-to-AV2 zero-shot."""

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

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_av2_zero_shot as m5_external_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.dataset_av2 import compile_av2_log_v71, load_frozen_av2_cohort
from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _prepare_bundle_actor(
    bundle: Mapping[str, Any],
    standardizer: FeatureStandardizer,
    device: torch.device,
) -> dict[str, Any] | None:
    diagnostics = bundle["diagnostics"]
    candidates = np.asarray(
        diagnostics["completion_candidates"], dtype=np.float32
    ).reshape(-1, 3)
    if len(candidates) == 0:
        return None
    anchors = np.concatenate(
        [
            np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
            np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
        ],
        axis=0,
    )
    build_points = np.concatenate(
        [np.asarray(points, dtype=np.float32) for points in diagnostics["build_frame_points"]],
        axis=0,
    )
    build_origins = np.concatenate(
        [
            np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(points), axis=0)
            for points, origin in zip(
                diagnostics["build_frame_points"], diagnostics["build_sensor_origins"]
            )
        ],
        axis=0,
    )
    if len(build_points) > 16384:
        selected = np.linspace(0, len(build_points) - 1, num=16384, dtype=np.int64)
        build_points = build_points[selected]
        build_origins = build_origins[selected]
    evidence = build_evidential_queries(
        candidates,
        build_origins,
        build_points,
        beam_radius_m=0.20,
        endpoint_radius_m=0.12,
        device=device,
        query_chunk_size=256,
    ).masses
    payload = {
        "base_features": np.asarray(diagnostics["completion_features"], dtype=np.float32),
        "candidates": candidates,
        "size_lwh_m": np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32),
        "evidence_masses": evidence,
        "query_sensor_origin": np.asarray(diagnostics["query_sensor_origin"], dtype=np.float32),
    }
    features, rays, normals = m0_runner._raw_features(payload, device)
    return {
        "features": torch.as_tensor(
            standardizer.transform(features), dtype=torch.float32, device=device
        ),
        "candidates_t": torch.as_tensor(candidates, dtype=torch.float32, device=device),
        "anchors_t": torch.as_tensor(anchors, dtype=torch.float32, device=device),
        "ray_directions_t": torch.as_tensor(rays, dtype=torch.float32, device=device),
        "normals_t": torch.as_tensor(normals, dtype=torch.float32, device=device),
        "size_t": torch.as_tensor(payload["size_lwh_m"], dtype=torch.float32, device=device),
        "candidates": candidates,
        "anchors": anchors,
    }


def _evaluate_bundle(
    bundle: Mapping[str, Any],
    base: RaySurfaceRelocationMLP,
    model: GaussianSeedExpansionMLP,
    standardizer: FeatureStandardizer,
    base_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    actor = _prepare_bundle_actor(bundle, standardizer, device)
    if actor is None:
        return None
    diagnostics = bundle["diagnostics"]
    with torch.inference_mode():
        _, base_centers = m5_runner._move(base, actor, base_config)
        actor["m5_centers_t"] = base_centers
        children, residuals, scales = m7_runner._predict(model, actor, model_config)
        output = torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy()
    output = _voxel_unique(output, float(evaluation["output_voxel_size_m"]))
    row = evaluate_actor_surface(
        np.asarray(diagnostics["compiled"], dtype=np.float32),
        output,
        np.asarray(diagnostics["target"], dtype=np.float32),
        np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32),
        hazardous=bool(bundle["row"]["hazardous"]),
        device=device,
        lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
        depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
        distance_chunk_size=int(evaluation["distance_chunk_size"]),
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
        }
    )
    return row


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = load_frozen_av2_cohort(repo_root / config["cohort_config"])
    if len(cohort["logs"]) != int(config["expected_log_count"]):
        raise RuntimeError("frozen V7.1 AV2 cohort count changed")
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    m0_runner._deep_update(compiler, config["compiler_overrides"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    m5_external_runner._write_json(
        run_dir / "status.json",
        {"status": "running", "phase": "waiting_fresh_av2", "completed_logs": 0},
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M7 AV2 zero-shot requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        m7_run = Path(config["m7_run"])
        checkpoint = torch.load(m7_run / "MODEL.pt", map_location=device, weights_only=False)
        m7_config = yaml.safe_load((m7_run / "resolved.yaml").read_text(encoding="utf-8"))
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = GaussianSeedExpansionMLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["branch_factor"]),
            int(checkpoint["slot_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        m5_run = Path(checkpoint["m5_run"])
        m5_checkpoint = torch.load(m5_run / "MODEL.pt", map_location=device, weights_only=False)
        m5_config = yaml.safe_load((m5_run / "resolved.yaml").read_text(encoding="utf-8"))
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
            bundles = compile_av2_log_v71(Path(compiler["dataset_root"]) / log_id, compiler, device)
            log_rows = 0
            for bundle in bundles:
                bundle["scene_name"] = log_id
                row = _evaluate_bundle(
                    bundle,
                    base,
                    model,
                    standardizer,
                    m5_config["model"],
                    m7_config["model"],
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
            m5_external_runner._write_jsonl(run_dir / "EXTERNAL_ACTORS.partial.jsonl", rows)
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
                        "stage": "m7_fresh_av2",
                        "progress": f"{position + 1}/{len(cohort['logs'])}",
                        "log_id": log_id,
                        "log_actors": log_rows,
                        "total_actors": len(rows),
                    }
                ),
                flush=True,
            )
        metrics = summarize_surface_rows(rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["log_id"])].append(row)
        per_log = {
            log_id: summarize_surface_rows(log_rows)
            for log_id, log_rows in grouped.items()
        }
        decisions = m5_external_runner._external_decisions(metrics, config["decision"])
        passed = all(decisions.values())
        m5_external_runner._write_jsonl(run_dir / "EXTERNAL_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m7_fresh_av2_zero_shot.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "zero_shot_supervision_native_transfer_supported"
            if passed
            else "source_only_supervision_native_geometry_cross_sensor_transfer_rejected",
            "cohort_config": config["cohort_config"],
            "log_count": len(processed_logs),
            "actor_count": len(rows),
            "processed_logs": processed_logs,
            "external": metrics,
            "per_log": per_log,
            "decisions": decisions,
            "m7_checkpoint": str(m7_run),
            "m5_base_checkpoint": str(m5_run),
            "fine_tuning": False,
            "calibration": False,
            "threshold_selection": False,
            "failed_log_deletion": False,
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
            {"status": "failed", "phase": "fresh_av2_evaluation", "error": f"{type(error).__name__}: {error}"},
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

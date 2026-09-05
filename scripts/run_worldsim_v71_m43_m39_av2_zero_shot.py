"""Evaluate frozen M39 evidential categorical surface measure on AV2."""

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

import run_worldsim_v71_m5_av2_zero_shot as external_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_av2_zero_shot as m8_external_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m8_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as energy_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as anchor_runner
import run_worldsim_v71_m37_supervised_child_transmittance as child_runner
import run_worldsim_v71_m38_prehit_free_space_survival as authority_runner
import run_worldsim_v71_m39_categorical_authority_composition as composition_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.dataset_av2 import compile_av2_log_v71, load_frozen_av2_cohort
from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries
from motion_proj.worldsim_v71.evidential_gaussian_authority import occupied_masses
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _limit(values: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= maximum:
        indices = np.arange(len(values), dtype=np.int64)
    else:
        indices = np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64)
    return values[indices], indices


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _anchor_input_sidecar(
    bundle: Mapping[str, Any],
    actor: Mapping[str, Any],
    evidence_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Rebuild the source-domain M33 input contract from AV2 build sweeps only."""
    diagnostics = bundle["diagnostics"]
    kept = np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3)
    projected = np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3)
    anchors = np.concatenate([kept, projected], axis=0)
    if anchors.shape != np.asarray(actor["anchors"]).shape or not np.allclose(
        anchors, np.asarray(actor["anchors"], dtype=np.float32)
    ):
        raise RuntimeError("AV2 anchor order changed before authority adaptation")

    kept_query_indices = np.asarray(diagnostics["kept_query_indices"], dtype=np.int64)
    projected_query_indices = np.asarray(
        diagnostics["projected_query_indices"], dtype=np.int64
    )
    source_query_indices = np.concatenate(
        [kept_query_indices, projected_query_indices]
    )
    canonical_indices = np.concatenate(
        [
            np.asarray(diagnostics["kept_surface_indices"], dtype=np.int64),
            np.asarray(diagnostics["projected_surface_indices"], dtype=np.int64),
        ]
    )
    provenance = np.concatenate(
        [
            np.zeros(len(kept), dtype=np.int8),
            np.ones(len(projected), dtype=np.int8),
        ]
    )

    build_parts = [
        np.asarray(points, dtype=np.float32).reshape(-1, 3)
        for points in diagnostics["build_frame_points"]
    ]
    build_origins_parts = [
        np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(points), axis=0)
        for points, origin in zip(build_parts, diagnostics["build_sensor_origins"])
    ]
    build_points = np.concatenate(build_parts, axis=0)
    build_origins = np.concatenate(build_origins_parts, axis=0)
    build_points, selected = _limit(
        build_points, int(evidence_config["maximum_build_evidence_points"])
    )
    build_origins = build_origins[selected]
    build = build_evidential_queries(
        anchors,
        build_origins,
        build_points,
        beam_radius_m=float(evidence_config["build_beam_radius_m"]),
        endpoint_radius_m=float(evidence_config["build_endpoint_radius_m"]),
        device=device,
        query_chunk_size=int(evidence_config["query_chunk_size"]),
    )

    query = np.asarray(diagnostics["query"], dtype=np.float32).reshape(-1, 3)
    source_points = query[source_query_indices]
    source_origin = np.asarray(
        diagnostics["query_sensor_origin"], dtype=np.float32
    ).reshape(3)
    source_vectors = source_points - source_origin[None, :]
    source_ranges = np.linalg.norm(source_vectors, axis=1).astype(np.float32)
    source_directions = source_vectors / np.maximum(source_ranges[:, None], 1.0e-6)
    canonical_origins = np.asarray(
        diagnostics["canonical_sensor_origins"], dtype=np.float32
    ).reshape(-1, 3)
    return {
        "input_provenance": provenance,
        "input_source_ray_directions": source_directions.astype(np.float32),
        "input_source_ranges_m": source_ranges,
        "input_projection_displacement_xyz_m": (anchors - source_points).astype(
            np.float32
        ),
        "input_canonical_hit_count": np.asarray(
            diagnostics["canonical_hit_count"], dtype=np.int32
        )[canonical_indices],
        "input_canonical_temporal_support": np.asarray(
            diagnostics["canonical_temporal_support"], dtype=np.int16
        )[canonical_indices],
        "input_canonical_view_support": np.asarray(
            diagnostics["canonical_view_support"], dtype=np.int8
        )[canonical_indices],
        "input_canonical_sensor_origins": canonical_origins[canonical_indices],
        "input_build_frame_count": np.asarray(len(build_parts), dtype=np.int16),
        "input_build_evidence_masses": build.masses,
        "input_build_evidence_opportunities": build.opportunity_count,
    }


def _prepare_authority_actor(
    bundle: Mapping[str, Any],
    base: RaySurfaceRelocationMLP,
    surface: GaussianSeedExpansionMLP,
    standardizer: FeatureStandardizer,
    base_config: Mapping[str, Any],
    surface_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    evidence_config: Mapping[str, Any],
    anchor_scale_m: float,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor] | None:
    actor = m8_external_runner._prepare_bundle_actor(bundle, standardizer, device)
    if actor is None:
        return None
    with torch.inference_mode():
        _, base_centers = m5_runner._move(base, actor, base_config)
        actor["m5_centers_t"] = base_centers
        children, residuals, scales = m8_runner._predict(
            surface, actor, surface_config
        )
    actor["m8_children_t"] = children
    actor["m8_residuals_t"] = residuals
    actor["m8_scales_t"] = scales
    actor["size_lwh_m"] = actor["size_t"].detach().cpu().numpy()
    sidecar = _anchor_input_sidecar(bundle, actor, evidence_config, device)
    actor["authority_anchor_features_t"] = anchor_runner._anchor_features(
        actor, sidecar, feature_config, device
    )
    actor["authority_child_features_t"] = child_runner._child_features(actor)
    actor["authority_centers_t"] = torch.cat([actor["anchors_t"], children], dim=0)
    actor["authority_scales_t"] = torch.cat(
        [
            torch.full(
                (len(actor["anchors_t"]),),
                float(anchor_scale_m),
                dtype=torch.float32,
                device=device,
            ),
            scales.reshape(-1),
        ]
    ).clamp_min(1.0e-4)
    return actor, residuals, scales


def _evaluate_bundle(
    bundle: Mapping[str, Any],
    base: RaySurfaceRelocationMLP,
    surface: GaussianSeedExpansionMLP,
    anchor_authority: torch.nn.Module,
    child_authority: torch.nn.Module,
    standardizer: FeatureStandardizer,
    base_config: Mapping[str, Any],
    surface_config: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    prepared = _prepare_authority_actor(
        bundle,
        base,
        surface,
        standardizer,
        base_config,
        surface_config,
        config["features"],
        config["evidence"],
        float(config["anchor_scale_m"]),
        device,
    )
    if prepared is None:
        return None
    actor, residuals, scales = prepared
    diagnostics = bundle["diagnostics"]
    target = np.asarray(diagnostics["target"], dtype=np.float32)
    origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32)
    actor["target"] = target
    actor["target_sensor_origins"] = origins

    with torch.inference_mode():
        baseline = energy_runner._energy_partition(
            actor, config["evaluation"], float(config["anchor_scale_m"]), device
        )
        anchor_masses = torch.softmax(
            anchor_authority(actor["authority_anchor_features_t"]), dim=1
        )
        child_masses = torch.softmax(
            child_authority(actor["authority_child_features_t"]), dim=1
        )
        categorical = composition_runner._categorical_partition(
            actor,
            torch.cat([anchor_masses[:, 1], child_masses[:, 1]], dim=0),
            config["evaluation"],
        )

    output = _voxel_unique(
        torch.cat([actor["anchors_t"], actor["m8_children_t"]], dim=0)
        .cpu()
        .numpy(),
        float(config["evaluation"]["output_voxel_size_m"]),
    )
    row = evaluate_actor_surface(
        np.asarray(diagnostics["compiled"], dtype=np.float32),
        output,
        target,
        origins,
        hazardous=bool(bundle["row"]["hazardous"]),
        device=device,
        lateral_tolerance_m=float(config["evaluation"]["literal_lateral_tolerance_m"]),
        depth_tolerance_m=float(config["evaluation"]["literal_depth_tolerance_m"]),
        distance_chunk_size=int(config["evaluation"]["distance_chunk_size"]),
    )
    row.update(
        {
            "scene_name": str(bundle["scene_name"]),
            "track_id": str(bundle["row"]["track_id"]),
            "category": str(bundle["row"]["category"]),
            "ray_count": int(len(target)),
            "candidate_count": int(len(actor["candidates"])),
            "generated_child_count": int(len(actor["m8_children_t"])),
            "mean_child_residual_m": float(
                torch.linalg.vector_norm(residuals, dim=1).mean()
            ),
            "mean_gaussian_scale_m": float(scales.mean()),
            "baseline_early_count": int(np.count_nonzero(baseline["early"])),
            "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
            "m39_early_count": int(np.count_nonzero(categorical["early"])),
            "m39_hit_count": int(np.count_nonzero(categorical["hit"])),
            "m39_observable_count": int(np.count_nonzero(categorical["observable"])),
            "mean_anchor_fou": anchor_masses.mean(dim=0).cpu().tolist(),
            "mean_child_fou": child_masses.mean(dim=0).cpu().tolist(),
        }
    )
    return row


def _surface_return_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        baseline_early = sum(int(row["baseline_early_count"]) for row in selected)
        baseline_hit = sum(int(row["baseline_hit_count"]) for row in selected)
        m39_early = sum(int(row["m39_early_count"]) for row in selected)
        m39_hit = sum(int(row["m39_hit_count"]) for row in selected)
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "baseline_early_rate": baseline_early / rays if rays else None,
            "baseline_hit_rate": baseline_hit / rays if rays else None,
            "m39_early_rate": m39_early / rays if rays else None,
            "m39_hit_rate": m39_hit / rays if rays else None,
            "m39_vs_baseline_early_delta": (m39_early - baseline_early) / rays
            if rays
            else None,
            "m39_vs_baseline_hit_delta": (m39_hit - baseline_hit) / rays
            if rays
            else None,
            "m39_observable_rate": sum(
                int(row["m39_observable_count"]) for row in selected
            )
            / rays
            if rays
            else None,
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(
    config_path: Path, repo_root: Path, run_id: str, *, resume: bool = False
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = load_frozen_av2_cohort(repo_root / config["cohort_config"])
    if len(cohort["logs"]) != int(config["expected_log_count"]):
        raise RuntimeError("frozen V7.1 AV2 cohort count changed")
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    energy_runner.m0_runner._deep_update(compiler, config["compiler_overrides"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    rows: list[dict[str, Any]] = []
    processed_logs: list[str] = []
    resume_count = 0
    if resume:
        if not run_dir.is_dir():
            raise RuntimeError(f"cannot resume missing run directory: {run_dir}")
        if (run_dir / "summary.json").is_file():
            raise RuntimeError("cannot resume completed M43 run")
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            raise RuntimeError("cannot resume M43 run without status.json")
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        resume_count = int(prior_status.get("completed_logs", 0))
        if not 0 <= resume_count <= len(cohort["logs"]):
            raise RuntimeError("M43 resume completed-log count is outside the cohort")
        partial_path = run_dir / "EXTERNAL_ACTORS.partial.jsonl"
        if resume_count and not partial_path.is_file():
            raise RuntimeError("cannot resume processed M43 logs without partial rows")
        rows = _read_jsonl(partial_path) if partial_path.is_file() else []
        processed_logs = [
            str(row["log_id"]) for row in cohort["logs"][:resume_count]
        ]
        processed_set = set(processed_logs)
        if any(str(row.get("log_id")) not in processed_set for row in rows):
            raise RuntimeError("M43 partial rows are outside the completed cohort prefix")
        external_runner._write_json(
            status_path,
            {
                "status": "running",
                "phase": "resuming_fresh_av2",
                "completed_logs": resume_count,
                "actor_rows": len(rows),
            },
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        external_runner._write_json(
            run_dir / "status.json",
            {"status": "running", "phase": "waiting_fresh_av2", "completed_logs": 0},
        )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M43 frozen AV2 evaluation requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface_run = Path(config["m8_run"])
        surface_checkpoint = torch.load(
            surface_run / "MODEL.pt", map_location=device, weights_only=False
        )
        standardizer = FeatureStandardizer.from_payload(
            surface_checkpoint["standardizer"]
        )
        surface_config = yaml.safe_load(
            (surface_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        surface = GaussianSeedExpansionMLP(
            int(surface_checkpoint["input_dim"]),
            int(surface_checkpoint["hidden_dim"]),
            int(surface_checkpoint["branch_factor"]),
            int(surface_checkpoint["slot_dim"]),
        ).to(device)
        surface.load_state_dict(surface_checkpoint["state_dict"])
        surface.eval().requires_grad_(False)
        base_run = Path(surface_checkpoint["m5_run"])
        base_checkpoint = torch.load(
            base_run / "MODEL.pt", map_location=device, weights_only=False
        )
        base_config = yaml.safe_load(
            (base_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        base = RaySurfaceRelocationMLP(
            int(base_checkpoint["input_dim"]), int(base_checkpoint["hidden_dim"])
        ).to(device)
        base.load_state_dict(base_checkpoint["state_dict"])
        base.eval().requires_grad_(False)
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

        state = Path(config["download_state"])
        for position, cohort_row in enumerate(
            cohort["logs"][resume_count:], start=resume_count
        ):
            log_id = str(cohort_row["log_id"])
            marker = state / f"{log_id}.complete"
            wait_started = time.monotonic()
            while not marker.is_file():
                if time.monotonic() - wait_started > int(config["wait_timeout_seconds"]):
                    raise TimeoutError(f"download timeout for {log_id}")
                external_runner._write_json(
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
                    surface,
                    anchor_authority,
                    child_authority,
                    standardizer,
                    base_config["model"],
                    surface_config["model"],
                    config,
                    device,
                )
                if row is None:
                    continue
                row["log_id"] = log_id
                row["external_role"] = str(cohort_row["role"])
                rows.append(row)
                log_rows += 1
            processed_logs.append(log_id)
            external_runner._write_jsonl(run_dir / "EXTERNAL_ACTORS.partial.jsonl", rows)
            external_runner._write_json(
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
                        "stage": "m43_m39_av2",
                        "progress": f"{position + 1}/{len(cohort['logs'])}",
                        "log_id": log_id,
                        "log_actors": log_rows,
                        "total_actors": len(rows),
                    }
                ),
                flush=True,
            )

        if not rows:
            raise RuntimeError("complete AV2 cohort produced no eligible actors")
        metrics = _surface_return_summary(rows)
        point_metrics = summarize_surface_rows(rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["log_id"])].append(row)
        per_log = {
            log_id: {
                "m8_point_surface": summarize_surface_rows(log_rows),
                "m39_surface_return": _surface_return_summary(log_rows),
            }
            for log_id, log_rows in grouped.items()
        }
        worst_stratum_delta = max(
            float(metrics["hazard"]["m39_vs_baseline_early_delta"]),
            float(metrics["clear"]["m39_vs_baseline_early_delta"]),
        )
        decisions = {
            "m39_all_early_nonincrease": float(
                metrics["all"]["m39_vs_baseline_early_delta"]
            )
            <= float(config["decision"]["maximum_all_early_delta"]),
            "m39_hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "m39_all_hit_retained": float(
                metrics["all"]["m39_vs_baseline_hit_delta"]
            )
            >= float(config["decision"]["minimum_all_hit_delta"]),
        }
        passed = all(decisions.values())
        external_runner._write_jsonl(run_dir / "EXTERNAL_ACTORS.jsonl", rows)
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
            "m39_surface_return": metrics,
            "per_log": per_log,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "m8_checkpoint": str(surface_run),
            "anchor_authority_checkpoint": str(config["m35_run"]),
            "child_authority_checkpoint": str(config["m38_run"]),
            "input_contract": "AV2 build/query only; target reserved for final evaluation",
            "fine_tuning": False,
            "calibration": False,
            "threshold_selection": False,
            "failed_log_deletion": False,
            "partial_metric_read": False,
            "source_final_read": False,
            "external_read": True,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        external_runner._write_json(run_dir / "summary.json", summary)
        external_runner._write_json(
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
        external_runner._write_json(
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.repo_root.resolve(),
                args.run_id,
                resume=args.resume,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

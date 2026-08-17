#!/usr/bin/env python3
"""Run the frozen V5.1 Stage A screening on the two S scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v5.ownership_renderer import rasterize_ownership_probability
from motion_proj.worldsim_v51.evidence.abstention import (
    build_semantic_unknown_state,
    finalize_selective_semantic_metrics,
    merge_selective_semantic_statistics,
    selective_semantic_statistics,
)
from motion_proj.worldsim_v51.protocol import (
    V51_BRANCH,
    load_yaml,
    sha256_file,
    verify_canonical_run,
)
from scripts import run_worldsim_v5_m1_unary_diagnostic as v5_runner
from scripts import run_worldsim_v51_unary_visibility as a1_runner
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V51-M1-A-UNARY-OBSERVABILITY-01"
SCHEMA_VERSION = "worldsim_v51_stage_a_screening_v1"
RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v51")
METRICS = a1_runner.METRICS


class ScreeningRunError(RuntimeError):
    """A frozen Stage A screening contract failed."""


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _prepare_run(run_dir: Path) -> str:
    resolved = run_dir.resolve()
    task_root = (RUN_ROOT / TASK_ID).resolve()
    if resolved.exists():
        raise ScreeningRunError(f"run directory already exists: {resolved}")
    if task_root not in resolved.parents:
        raise ScreeningRunError(f"run must be under {task_root}")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise ScreeningRunError(f"screening must execute on {V51_BRANCH}")
    if _git("status", "--short"):
        raise ScreeningRunError("formal screening requires a clean worktree")
    source_commit = _git("rev-parse", "HEAD")
    resolved.mkdir(parents=True)
    return source_commit


def evaluate_screening_gate(
    scene_summaries: list[Mapping[str, Any]],
    gate: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the numeric gates frozen before the S quality read."""
    if len(scene_summaries) != int(gate["required_scene_count"]):
        raise ScreeningRunError("screening scene denominator drift")
    deltas = [row["evaluation_delta_a1_minus_b3"] for row in scene_summaries]
    mean_delta = {
        name: float(np.mean([float(row[name]) for row in deltas]))
        for name in METRICS
    }
    nonnegative_count = sum(float(row["boundary_f1"]) >= 0.0 for row in deltas)
    clear_threshold = float(gate["boundary_f1_clear_delta_minimum_inclusive"])
    clearly_positive_count = sum(
        float(row["boundary_f1"]) >= clear_threshold for row in deltas
    )
    conditional_checks = {
        "boundary_f1_nonnegative_scene_count": nonnegative_count
        >= int(gate["boundary_f1_nonnegative_scene_count_minimum"]),
        "boundary_f1_clearly_positive_scene_count": clearly_positive_count
        >= int(gate["boundary_f1_clearly_positive_scene_count_minimum"]),
        "mean_boundary_f1_positive": mean_delta["boundary_f1"]
        > float(gate["mean_boundary_f1_delta_minimum_exclusive"]),
        "mean_false_negative_semantic_mass_bounded": mean_delta[
            "false_negative_semantic_mass"
        ]
        <= float(gate["mean_false_negative_semantic_mass_delta_maximum"]),
        "mean_brier_delta_bounded": mean_delta["brier"]
        <= float(gate["calibration"]["maximum_mean_brier_delta"]),
        "mean_ece_delta_bounded": mean_delta["ece"]
        <= float(gate["calibration"]["maximum_mean_ece_delta"]),
    }
    conditional_passed = all(conditional_checks.values())

    selective_rows = [row["a2_selective_metrics"] for row in scene_summaries]
    mean_coverage = float(np.mean([float(row["coverage"]) for row in selective_rows]))
    accepted = [row["accepted_subset_error"] for row in selective_rows]
    abstained = [row["abstained_subset_error"] for row in selective_rows]
    complete = all(value is not None for value in accepted + abstained)
    mean_accepted = (
        float(np.mean([float(value) for value in accepted])) if complete else None
    )
    mean_abstained = (
        float(np.mean([float(value) for value in abstained])) if complete else None
    )
    selective_checks = {
        "mean_coverage_minimum": mean_coverage
        >= float(gate["unknown_candidate"]["mean_coverage_minimum"]),
        "all_scenes_have_accepted_and_abstained_pixels": complete,
        "abstained_subset_error_exceeds_accepted": bool(
            complete and mean_abstained > mean_accepted
        ),
    }
    selective_passed = all(selective_checks.values())

    if conditional_passed and selective_passed:
        survivor = "A2"
    elif conditional_passed:
        survivor = "A1"
    else:
        survivor = "U2_B3"
    if int(selection_policy["maximum_survivors_after_s"]) != 1:
        raise ScreeningRunError("selection policy survivor cardinality drift")
    return {
        "passed_conditional_gate": conditional_passed,
        "passed_a2_selective_gate": selective_passed,
        "selected_survivor": survivor,
        "conditional_checks": conditional_checks,
        "selective_checks": selective_checks,
        "boundary_f1_nonnegative_scene_count": nonnegative_count,
        "boundary_f1_clearly_positive_scene_count": clearly_positive_count,
        "scene_balanced_mean_delta_a1_minus_b3": mean_delta,
        "scene_balanced_mean_coverage": mean_coverage,
        "scene_balanced_mean_accepted_subset_error": mean_accepted,
        "scene_balanced_mean_abstained_subset_error": mean_abstained,
        "scene_balanced_error_separation": (
            None if not complete else float(mean_abstained - mean_accepted)
        ),
    }


def _run_scene(
    scene: str,
    spec: Mapping[str, Any],
    source_config_path: Path,
    config: Mapping[str, Any],
    a1_config: Mapping[str, Any],
    a2_config: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = verify_canonical_run(scene, spec)
    source_config = load_yaml(source_config_path)
    outputs, observation_diagnostics, base_model = a1_runner._build_posteriors(
        source_config,
        spec,
        minimum_visibility=float(a1_config["visibility"]["minimum_visibility"]),
    )
    thresholds = a2_config["unknown"]["frozen_thresholds"]
    a2_state = build_semantic_unknown_state(
        conditional_actor_probability=outputs["A1"]["unary_posterior"],
        effective_observation_count=outputs["A1"]["effective_evidence_count"],
        cross_view_disagreement=outputs["A1"]["multi_view_disagreement"],
        effective_count_maximum=float(
            thresholds["effective_observation_count_maximum_inclusive"]
        ),
        entropy_minimum=float(thresholds["posterior_entropy_minimum_inclusive"]),
        disagreement_minimum=float(
            thresholds["cross_view_disagreement_minimum_inclusive"]
        ),
    )
    posterior_path = run_dir / "artifacts/posteriors" / f"{scene}.npz"
    atomic_save_npz(
        posterior_path,
        {
            **{
                f"{arm}_{field}": outputs[arm][field]
                for arm in ("B3", "A1")
                for field in (
                    "unary_posterior",
                    "unary_uncertainty",
                    "effective_evidence_count",
                    "multi_view_disagreement",
                    "boundary_ambiguity",
                    "depth_support",
                )
            },
            **{f"A2_{name}": value for name, value in a2_state.items()},
        },
    )

    source_run = Path(spec["path"])
    source_summary = json.loads(
        (source_run / "summary.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(source_config["inputs"]["formal_checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != spec["checkpoint_sha256"]:
        raise ScreeningRunError(f"checkpoint SHA drift: {scene}")
    sam_manifest_path = Path(source_config["inputs"]["sam_mask_manifest"]["path"])
    sam_manifest = json.loads(sam_manifest_path.read_text(encoding="utf-8"))
    evaluation_rows = [
        row
        for row in sam_manifest["views"]
        if row["split"] == "evaluation"
        and bool(row["sam_probability_available"])
        and bool(row["mask_quality_accepted"])
    ]
    if len(evaluation_rows) != int(spec["accepted_evaluation_view_count"]):
        raise ScreeningRunError(f"accepted evaluation denominator drift: {scene}")

    _, dataset, trainer = v5_runner._build_runtime(source_config, device)
    get_view_data, _, release_trainer_render_info = v5_runner._runtime_helpers()
    a1_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    selective_statistics: list[dict[str, Any]] = []
    evaluation_config = source_config["evaluation"]
    abstain_threshold = float(a2_config["unknown"]["image_abstain_threshold"])
    try:
        for row in evaluation_rows:
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
            mask_path = sam_manifest_path.parents[1] / row["mask"]["path"]
            if sha256_file(mask_path) != row["mask"]["sha256"]:
                raise ScreeningRunError(f"SAM mask SHA drift: {mask_path}")
            target = _load_npz(mask_path)["binary"].astype(bool)
            image_infos, camera_infos, *_ = get_view_data(
                dataset, frame, camera_id, device
            )
            try:
                with torch.inference_mode():
                    processed_camera, gaussians = v5_runner._collect_gaussians(
                        trainer, image_infos, camera_infos
                    )
                    render_kwargs = {
                        "means": gaussians.means,
                        "quats": gaussians.quats,
                        "scales": gaussians.scales,
                        "base_opacities": gaussians.opacities,
                        "viewmats": torch.linalg.inv(processed_camera.camtoworlds)[
                            None, ...
                        ],
                        "intrinsics": processed_camera.Ks[None, ...],
                        "width": int(processed_camera.W),
                        "height": int(processed_camera.H),
                        "near_plane": float(trainer.render_cfg.near_plane),
                        "far_plane": float(trainer.render_cfg.far_plane),
                        "packed": bool(trainer.render_cfg.packed),
                        "radius_clip": float(
                            trainer.render_cfg.get("radius_clip", 0.0)
                        ),
                        "antialiased": bool(trainer.render_cfg.antialiased),
                    }
                    actor_alpha, _ = rasterize_ownership_probability(
                        probability=outputs["A1"]["unary_posterior"],
                        **render_kwargs,
                    )
                    unknown_alpha, _ = rasterize_ownership_probability(
                        probability=a2_state["unknown_probability"],
                        **render_kwargs,
                    )
                    probability = actor_alpha.detach().cpu().numpy().astype(np.float32)
                    unknown_probability = (
                        unknown_alpha.detach().cpu().numpy().astype(np.float32)
                    )
                if probability.shape != target.shape or unknown_probability.shape != target.shape:
                    raise ScreeningRunError("screening render/SAM shape drift")
                accepted = unknown_probability < abstain_threshold
                output = (
                    run_dir
                    / "artifacts/evaluation"
                    / scene
                    / "A1_A2"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                atomic_save_npz(
                    output,
                    {
                        "conditional_actor_probability": probability.astype(np.float16),
                        "unknown_probability": unknown_probability.astype(np.float16),
                        "accepted": accepted.astype(np.int8),
                        "target": target.astype(np.int8),
                    },
                )
                conditional = a1_runner._render_metrics(
                    probability,
                    target.astype(np.float32),
                    threshold=float(evaluation_config["probability_threshold"]),
                    boundary_tolerance_px=int(
                        evaluation_config["boundary_tolerance_px"]
                    ),
                    ece_bins=int(evaluation_config["ece_bins"]),
                )
                statistics = selective_semantic_statistics(
                    probability,
                    target.astype(np.float32),
                    unknown_probability,
                    probability_threshold=float(
                        evaluation_config["probability_threshold"]
                    ),
                    abstain_threshold=abstain_threshold,
                )
                selective = finalize_selective_semantic_metrics(statistics)
                a1_rows.append(
                    {
                        "frame": frame,
                        "camera_id": camera_id,
                        **conditional,
                        "path": str(output.relative_to(run_dir)),
                        "sha256": sha256_file(output),
                    }
                )
                selective_rows.append(
                    {"frame": frame, "camera_id": camera_id, **selective}
                )
                selective_statistics.append(statistics)
            finally:
                release_trainer_render_info(trainer)
            print(
                f"Stage A screening {scene} frame={frame} camera={camera_id}",
                flush=True,
            )
    finally:
        del trainer
        del dataset
        torch.cuda.empty_cache()

    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise ScreeningRunError(f"checkpoint mutation: {scene}")
    a1_aggregate = a1_runner._aggregate_metrics(a1_rows)
    canonical_b3 = source_summary["arm_evaluation_aggregate"]["B3"]
    delta = {
        name: float(a1_aggregate[name] - canonical_b3[name]) for name in METRICS
    }
    selective_aggregate = finalize_selective_semantic_metrics(
        merge_selective_semantic_statistics(selective_statistics)
    )
    unknown_count = int(np.count_nonzero(a2_state["unknown_probability"]))
    gaussian_count = int(a2_state["unknown_probability"].size)
    scene_summary = {
        **canonical,
        "accepted_evaluation_view_count": len(evaluation_rows),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "b3_posterior_replay_exact": True,
        "b3_evaluation_reused_read_only": True,
        "posterior_path": str(posterior_path.relative_to(run_dir)),
        "posterior_sha256": sha256_file(posterior_path),
        "gaussian_count": gaussian_count,
        "unknown_gaussian_count": unknown_count,
        "unknown_gaussian_ratio": float(unknown_count / gaussian_count),
        "evaluation_aggregate": {"B3": canonical_b3, "A1": a1_aggregate},
        "evaluation_delta_a1_minus_b3": delta,
        "a2_conditional_equals_a1_by_construction": True,
        "a2_selective_metrics": selective_aggregate,
        "observation_diagnostics": {
            key: value
            for key, value in observation_diagnostics.items()
            if key != "chunk_reports"
        },
    }
    scene_diagnostics = {
        "scene": scene,
        "observation_diagnostics": observation_diagnostics,
        "conditional_rows": a1_rows,
        "selective_rows": selective_rows,
        "selective_statistics": selective_statistics,
    }
    return scene_summary, scene_diagnostics


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ScreeningRunError("screening config schema drift")
    if config.get("task_id") != TASK_ID or config.get("phase") != "stage_a_screening":
        raise ScreeningRunError("screening task/phase drift")
    if config.get("status") != "frozen_before_candidate_s_quality_read":
        raise ScreeningRunError("screening was not frozen before candidate S quality")
    if list(config["scenes"]) != ["scene-0998", "scene-0359"]:
        raise ScreeningRunError("screening scene set or ordering drift")

    freeze_binding = config["inputs"]["screening_freeze"]
    freeze_path = PROJECT / freeze_binding["path"]
    verify_file(freeze_path, freeze_binding["sha256"])
    freeze = load_yaml(freeze_path)
    a1_binding = config["inputs"]["a1_config"]
    a1_path = PROJECT / a1_binding["path"]
    verify_file(a1_path, a1_binding["sha256"])
    a1_config = load_yaml(a1_path)
    a2_binding = config["inputs"]["a2_config"]
    a2_path = PROJECT / a2_binding["path"]
    verify_file(a2_path, a2_binding["sha256"])
    a2_config = load_yaml(a2_path)
    if freeze["screening_gate"] != config["screening_gate"]:
        raise ScreeningRunError("screening gate drift from freeze")
    if freeze["selection_policy"] != config["selection_policy"]:
        raise ScreeningRunError("selection policy drift from freeze")

    if not torch.cuda.is_available():
        raise ScreeningRunError("screening requires CUDA")
    device = torch.device(config["resources"]["device"])
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    maximum_start = int(config["resources"]["maximum_gpu_allocated_at_start_mib"])
    if torch.cuda.memory_allocated(device) > maximum_start * 1024**2:
        raise ScreeningRunError("screening GPU preflight is not idle")
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    scene_summaries = []
    scene_diagnostics = []
    for scene in config["scenes"]:
        binding = config["inputs"]["canonical_s_runs"][scene]
        source_config_binding = binding["source_config"]
        source_config_path = PROJECT / source_config_binding["path"]
        verify_file(source_config_path, source_config_binding["sha256"])
        summary, diagnostics = _run_scene(
            scene,
            binding,
            source_config_path,
            config,
            a1_config,
            a2_config,
            run_dir,
            device,
        )
        scene_summaries.append(summary)
        scene_diagnostics.append(diagnostics)
    gate = evaluate_screening_gate(
        scene_summaries, config["screening_gate"], config["selection_policy"]
    )
    diagnostics_path = run_dir / "artifacts/diagnostics.json"
    atomic_json(
        diagnostics_path,
        {
            "schema_version": "worldsim_v51_stage_a_screening_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene_diagnostics": scene_diagnostics,
        },
    )
    survivor = gate["selected_survivor"]
    return {
        "schema_version": "worldsim_v51_stage_a_screening_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": config["phase"],
        "conclusion": f"stage_a_screening_selected_{survivor.lower()}",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "seed": int(config["seed"]),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": int(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "scene_summaries": scene_summaries,
        "stage_a_s_gate": gate,
        "selected_survivor": survivor,
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "method_inference_started": True,
        "graph_inference_started": False,
        "parameter_search_performed": False,
        "evaluation_quality_used_for_thresholds": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": "pending_stage_a_closeout",
    }


def _write_terminal(
    run_dir: Path,
    *,
    status: str,
    source_commit: str | None,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None = None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v51_stage_a_screening_status_v1",
            "task_id": TASK_ID,
            "status": status,
            "source_commit": source_commit,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_a_screening_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_commit = _prepare_run(run_dir)
    config = load_yaml(args.config.resolve())
    resolved_config = write_resolved_config(run_dir, config)
    events = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_commit}
    ]
    write_events(run_dir, events)
    try:
        summary = run(args.config.resolve(), run_dir)
        source_snapshot = copy_source_snapshot(
            run_dir,
            [
                args.config.resolve(),
                PROJECT / "configs/worldsim_v51/stage_a_screening_freeze_v1.yaml",
                PROJECT / "configs/worldsim_v51/m1_unary_visibility_v1.yaml",
                PROJECT / "configs/worldsim_v51/m1_unary_unknown_v1.yaml",
                PROJECT / "motion_proj/worldsim_v51/evidence/abstention.py",
                PROJECT / "motion_proj/worldsim_v51/evidence/visibility.py",
                PROJECT / "scripts/run_worldsim_v51_stage_a_screening.py",
                PROJECT / "scripts/run_worldsim_v51_unary_visibility.py",
                PROJECT / "scripts/run_worldsim_v5_m1_unary_diagnostic.py",
                PROJECT / "tests/test_run_worldsim_v51_stage_a_screening.py",
            ],
            PROJECT,
        )
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v51_stage_a_screening_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_commit,
                "source_branch": V51_BRANCH,
                "worktree_clean": True,
                "resolved_config": resolved_config,
                "inputs": config["inputs"],
                "runtime": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(
                        torch.device(config["resources"]["device"])
                    ),
                },
                "source_snapshot": source_snapshot,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        manifest_path = run_dir / "manifest.json"
        atomic_json(
            manifest_path,
            {
                "schema_version": "worldsim_v51_stage_a_screening_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": inventory_files(
                    run_dir, {"manifest.json", "status.json"}
                ),
            },
        )
        _write_terminal(
            run_dir,
            status="done",
            source_commit=source_commit,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        write_events(run_dir, events)
        _write_terminal(
            run_dir,
            status="blocked",
            source_commit=source_commit,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()

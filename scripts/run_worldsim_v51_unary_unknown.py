#!/usr/bin/env python3
"""Execute V5.1 Stage A2 semantic UNKNOWN matched ablation on H scenes."""

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
    posterior_entropy,
    selective_semantic_statistics,
)
from motion_proj.worldsim_v51.protocol import (
    V51_BRANCH,
    load_yaml,
    sha256_file,
    verify_canonical_run,
)
from scripts import run_worldsim_v51_unary_visibility as a1_runner
from scripts import run_worldsim_v5_m1_unary_diagnostic as v5_runner
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
SCHEMA_VERSION = "worldsim_v51_m1_unary_unknown_v1"
RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v51")
CONDITIONAL_METRICS = a1_runner.METRICS


class UnknownRunError(RuntimeError):
    """A2 input, replay, rendering, or gate contract failed."""


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _verify_frozen_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute frozen thresholds from A1 evidence arrays only."""

    counts = []
    entropies = []
    disagreements = []
    scene_rows = []
    for scene in config["scenes"]:
        binding = config["inputs"]["a1_posteriors"][scene]
        path = Path(binding["path"])
        verify_file(path, binding["sha256"])
        payload = _load_npz(path)
        count = np.asarray(
            payload["A1_effective_evidence_count"], dtype=np.float64
        )
        observed = count > 0.0
        if not observed.any():
            raise UnknownRunError(f"A2 calibration population empty: {scene}")
        entropy = posterior_entropy(payload["A1_unary_posterior"])
        disagreement = np.asarray(
            payload["A1_multi_view_disagreement"], dtype=np.float64
        )
        counts.append(count[observed])
        entropies.append(entropy[observed])
        disagreements.append(disagreement[observed])
        scene_rows.append(
            {
                "scene": scene,
                "gaussian_count": int(count.size),
                "observed_gaussian_count": int(observed.sum()),
                "unobserved_gaussian_count": int((~observed).sum()),
                "unobserved_gaussian_ratio": float((~observed).mean()),
                "posterior_path": str(path),
                "posterior_sha256": binding["sha256"],
            }
        )
    count_values = np.concatenate(counts)
    entropy_values = np.concatenate(entropies)
    disagreement_values = np.concatenate(disagreements)
    quantiles = config["unknown"]["quantiles"]
    computed = {
        "effective_observation_count_maximum_inclusive": float(
            np.quantile(count_values, float(quantiles["low_effective_count"]))
        ),
        "posterior_entropy_minimum_inclusive": float(
            np.quantile(
                entropy_values, float(quantiles["high_posterior_entropy"])
            )
        ),
        "cross_view_disagreement_minimum_inclusive": float(
            np.quantile(
                disagreement_values,
                float(quantiles["high_cross_view_disagreement"]),
            )
        ),
    }
    frozen = config["unknown"]["frozen_thresholds"]
    if computed != {name: float(frozen[name]) for name in computed}:
        raise UnknownRunError(
            f"A2 frozen evidence thresholds do not reproduce: {computed}"
        )
    return {
        "calibration_quality_read": False,
        "calibration_population": config["unknown"]["calibration_population"],
        "pooled_observed_gaussian_count": int(count_values.size),
        "computed_thresholds": computed,
        "frozen_thresholds_exact": True,
        "scenes": scene_rows,
    }


def _build_a2_state(
    source_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    scene: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    a1_config = load_yaml(PROJECT / config["inputs"]["a1_config"]["path"])
    outputs, observation_diagnostics, base_model = a1_runner._build_posteriors(
        source_config,
        spec,
        minimum_visibility=float(a1_config["visibility"]["minimum_visibility"]),
    )
    binding = config["inputs"]["a1_posteriors"][scene]
    frozen = _load_npz(Path(binding["path"]))
    for field in (
        "unary_posterior",
        "unary_uncertainty",
        "effective_evidence_count",
        "multi_view_disagreement",
        "boundary_ambiguity",
        "depth_support",
    ):
        if not np.array_equal(outputs["A1"][field], frozen[f"A1_{field}"]):
            raise UnknownRunError(f"A2 parent A1 replay drift: {scene}/{field}")
    thresholds = config["unknown"]["frozen_thresholds"]
    state = build_semantic_unknown_state(
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
    return state, observation_diagnostics, base_model


def _unknown_gate(
    scene_summaries: list[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    if len(scene_summaries) != 3:
        raise UnknownRunError("A2 H gate must retain exactly three historical scenes")
    standard_gate = a1_runner._stage_a_gate(
        [row["evaluation_delta_a2_minus_b3"] for row in scene_summaries],
        {
            "boundary_f1_positive_scene_count_minimum": gate[
                "boundary_f1_positive_scene_count_minimum_vs_b3"
            ],
            "mean_boundary_f1_delta_minimum_exclusive": gate[
                "mean_boundary_f1_delta_minimum_exclusive_vs_b3"
            ],
            "mean_iou_delta_minimum": gate[
                "mean_iou_delta_minimum_vs_b3"
            ],
            "mean_fn_semantic_mass_delta_maximum": gate[
                "mean_fn_semantic_mass_delta_maximum_vs_b3"
            ],
            "brier_or_ece_must_improve": gate[
                "brier_or_ece_must_improve_vs_b3"
            ],
        },
    )
    selective = [row["selective_metrics"] for row in scene_summaries]
    coverage = float(np.mean([float(row["coverage"]) for row in selective]))
    accepted = [row["accepted_subset_error"] for row in selective]
    abstained = [row["abstained_subset_error"] for row in selective]
    complete = all(value is not None for value in accepted + abstained)
    mean_accepted = (
        float(np.mean([float(value) for value in accepted])) if complete else None
    )
    mean_abstained = (
        float(np.mean([float(value) for value in abstained])) if complete else None
    )
    selective_checks = {
        "mean_coverage_minimum": coverage
        >= float(gate["mean_coverage_minimum"]),
        "all_scenes_have_accepted_and_abstained_pixels": complete,
        "abstained_subset_error_exceeds_accepted": bool(
            complete and mean_abstained > mean_accepted
        ),
    }
    return {
        "passed": standard_gate["passed"] and all(selective_checks.values()),
        "conditional_stage_a_gate_vs_b3": standard_gate,
        "selective_checks": selective_checks,
        "scene_balanced_mean_coverage": coverage,
        "scene_balanced_mean_accepted_subset_error": mean_accepted,
        "scene_balanced_mean_abstained_subset_error": mean_abstained,
        "scene_balanced_error_separation": (
            None
            if not complete
            else float(mean_abstained - mean_accepted)
        ),
    }


def _prepare_run(run_dir: Path) -> str:
    resolved = run_dir.resolve()
    task_root = (RUN_ROOT / TASK_ID).resolve()
    if resolved.exists():
        raise UnknownRunError(f"run directory already exists: {resolved}")
    if task_root not in resolved.parents:
        raise UnknownRunError(f"run must be under {task_root}")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise UnknownRunError(f"A2 must execute on {V51_BRANCH}")
    if _git("status", "--short"):
        raise UnknownRunError("A2 formal run requires a clean worktree")
    source_commit = _git("rev-parse", "HEAD")
    resolved.mkdir(parents=True)
    return source_commit


def _run_scene(
    scene: str,
    spec: Mapping[str, Any],
    source_config_path: Path,
    config: Mapping[str, Any],
    a1_source_summary: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = verify_canonical_run(scene, spec)
    source_config = load_yaml(source_config_path)
    state, observation_diagnostics, base_model = _build_a2_state(
        source_config, spec, config, scene
    )
    unknown_count = int(state["unknown_probability"].sum())
    gaussian_count = int(state["unknown_probability"].size)
    posterior_path = run_dir / "artifacts/posteriors" / f"{scene}.npz"
    atomic_save_npz(posterior_path, state)

    checkpoint = Path(source_config["inputs"]["formal_checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != spec["checkpoint_sha256"]:
        raise UnknownRunError(f"A2 checkpoint SHA drift: {scene}")
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
        raise UnknownRunError(f"A2 accepted evaluation denominator drift: {scene}")

    _, dataset, trainer = v5_runner._build_runtime(source_config, device)
    get_view_data, _, release_trainer_render_info = v5_runner._runtime_helpers()
    conditional_rows = []
    selective_rows = []
    selective_statistics = []
    a1_byte_exact_count = 0
    evaluation_config = source_config["evaluation"]
    a1_run = Path(config["inputs"]["a1_run"]["path"])
    try:
        for row in evaluation_rows:
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
            mask_path = sam_manifest_path.parents[1] / row["mask"]["path"]
            if sha256_file(mask_path) != row["mask"]["sha256"]:
                raise UnknownRunError(f"A2 SAM mask SHA drift: {mask_path}")
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
                        probability=state["conditional_actor_probability"],
                        **render_kwargs,
                    )
                    unknown_alpha, _ = rasterize_ownership_probability(
                        probability=state["unknown_probability"],
                        **render_kwargs,
                    )
                    probability = actor_alpha.detach().cpu().numpy().astype(np.float32)
                    unknown_probability = (
                        unknown_alpha.detach().cpu().numpy().astype(np.float32)
                    )
                if probability.shape != target.shape or unknown_probability.shape != target.shape:
                    raise UnknownRunError("A2 render/SAM shape drift")
                a1_output = (
                    run_dir
                    / "artifacts/evaluation"
                    / scene
                    / "A1"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                atomic_save_npz(
                    a1_output,
                    {
                        "probability": probability.astype(np.float16),
                        "target": target.astype(np.int8),
                    },
                )
                canonical_a1_path = (
                    a1_run
                    / "artifacts/evaluation"
                    / scene
                    / "A1"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                if sha256_file(a1_output) != sha256_file(canonical_a1_path):
                    raise UnknownRunError(
                        f"A2 conditional A1 render not byte exact: {scene}/{frame}/{camera_id}"
                    )
                a1_byte_exact_count += 1
                accepted = unknown_probability < float(
                    config["unknown"]["image_abstain_threshold"]
                )
                a2_output = (
                    run_dir
                    / "artifacts/evaluation"
                    / scene
                    / "A2"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                atomic_save_npz(
                    a2_output,
                    {
                        "conditional_actor_probability": probability.astype(
                            np.float16
                        ),
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
                    abstain_threshold=float(
                        config["unknown"]["image_abstain_threshold"]
                    ),
                )
                selective = finalize_selective_semantic_metrics(statistics)
                conditional_rows.append(
                    {
                        "frame": frame,
                        "camera_id": camera_id,
                        **conditional,
                        "path": str(a2_output.relative_to(run_dir)),
                        "sha256": sha256_file(a2_output),
                    }
                )
                selective_rows.append(
                    {"frame": frame, "camera_id": camera_id, **selective}
                )
                selective_statistics.append(statistics)
            finally:
                release_trainer_render_info(trainer)
            print(
                f"A2 {scene} evaluation frame={frame} camera={camera_id}",
                flush=True,
            )
    finally:
        del trainer
        del dataset
        torch.cuda.empty_cache()

    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise UnknownRunError(f"A2 checkpoint mutation: {scene}")
    conditional_aggregate = a1_runner._aggregate_metrics(conditional_rows)
    source_scene = next(
        row for row in a1_source_summary["scene_summaries"] if row["scene"] == scene
    )
    canonical_a1 = source_scene["evaluation_aggregate"]["A1"]
    a1_delta = {
        name: float(conditional_aggregate[name] - canonical_a1[name])
        for name in CONDITIONAL_METRICS
    }
    if any(value != 0.0 for value in a1_delta.values()):
        raise UnknownRunError(f"A2 conditional metrics drift from A1: {scene}")
    b3 = source_scene["evaluation_aggregate"]["B3"]
    delta_b3 = {
        name: float(conditional_aggregate[name] - b3[name])
        for name in CONDITIONAL_METRICS
    }
    selective_aggregate = finalize_selective_semantic_metrics(
        merge_selective_semantic_statistics(selective_statistics)
    )
    scene_summary = {
        **canonical,
        "accepted_evaluation_view_count": len(evaluation_rows),
        "a1_conditional_rerender_byte_exact_count": a1_byte_exact_count,
        "a1_conditional_metric_delta": a1_delta,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "posterior_path": str(posterior_path.relative_to(run_dir)),
        "posterior_sha256": sha256_file(posterior_path),
        "gaussian_count": gaussian_count,
        "unknown_gaussian_count": unknown_count,
        "unknown_gaussian_ratio": float(unknown_count / gaussian_count),
        "conditional_evaluation_aggregate": conditional_aggregate,
        "evaluation_delta_a2_minus_b3": delta_b3,
        "selective_metrics": selective_aggregate,
        "observation_diagnostics": {
            key: value
            for key, value in observation_diagnostics.items()
            if key != "chunk_reports"
        },
    }
    scene_diagnostics = {
        "scene": scene,
        "observation_diagnostics": observation_diagnostics,
        "conditional_rows": conditional_rows,
        "selective_rows": selective_rows,
        "selective_statistics": selective_statistics,
    }
    return scene_summary, scene_diagnostics


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise UnknownRunError("A2 config schema drift")
    if config.get("task_id") != TASK_ID or config.get("phase") != "a2_semantic_unknown":
        raise UnknownRunError("A2 task/phase drift")
    if config.get("status") != "frozen_before_quality_read":
        raise UnknownRunError("A2 thresholds were not frozen before quality read")
    a1_binding = config["inputs"]["a1_config"]
    a1_config_path = PROJECT / a1_binding["path"]
    verify_file(a1_config_path, a1_binding["sha256"])
    a1_config = load_yaml(a1_config_path)
    baseline_binding = a1_config["inputs"]["a0_baseline_config"]
    baseline_path = PROJECT / baseline_binding["path"]
    verify_file(baseline_path, baseline_binding["sha256"])
    baseline = load_yaml(baseline_path)
    if list(config["scenes"]) != list(baseline["canonical_runs"]):
        raise UnknownRunError("A2 H scene set or ordering drift")
    a1_run_binding = config["inputs"]["a1_run"]
    a1_run = Path(a1_run_binding["path"])
    verify_file(a1_run / "summary.json", a1_run_binding["summary_sha256"])
    verify_file(a1_run / "manifest.json", a1_run_binding["manifest_sha256"])
    a1_summary = json.loads((a1_run / "summary.json").read_text(encoding="utf-8"))
    calibration = _verify_frozen_thresholds(config)
    if not torch.cuda.is_available():
        raise UnknownRunError("A2 requires CUDA")
    device = torch.device(config["resources"]["device"])
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    maximum_start = int(config["resources"]["maximum_gpu_allocated_at_start_mib"])
    if torch.cuda.memory_allocated(device) > maximum_start * 1024**2:
        raise UnknownRunError("A2 GPU preflight is not idle")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    scene_summaries = []
    scene_diagnostics = []
    for scene in config["scenes"]:
        summary, diagnostics = _run_scene(
            scene,
            baseline["canonical_runs"][scene],
            PROJECT / baseline["source_configs"][scene],
            config,
            a1_summary,
            run_dir,
            device,
        )
        scene_summaries.append(summary)
        scene_diagnostics.append(diagnostics)
    gate = _unknown_gate(
        scene_summaries, config["evaluation"]["scene_balanced_gate"]
    )
    diagnostics_path = run_dir / "artifacts/diagnostics.json"
    atomic_json(
        diagnostics_path,
        {
            "schema_version": "worldsim_v51_m1_a2_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "calibration": calibration,
            "scene_diagnostics": scene_diagnostics,
        },
    )
    return {
        "schema_version": "worldsim_v51_m1_a2_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": config["phase"],
        "conclusion": (
            "a2_unknown_passed_h_gate_candidate_for_stage_a"
            if gate["passed"]
            else "a2_unknown_rejected_h_gate"
        ),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "seed": int(config["seed"]),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": int(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "unknown_policy": config["unknown"],
        "calibration": calibration,
        "scene_summaries": scene_summaries,
        "stage_a_h_gate": gate,
        "a1_conditional_rerender_byte_exact_view_count": sum(
            row["a1_conditional_rerender_byte_exact_count"]
            for row in scene_summaries
        ),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "method_inference_started": True,
        "graph_inference_started": False,
        "parameter_search_performed": False,
        "evaluation_quality_used_for_thresholds": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": (
            "none" if gate["passed"] else "pending_rejection_entry"
        ),
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
            "schema_version": "worldsim_v51_m1_a2_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/m1_unary_unknown_v1.yaml",
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
                PROJECT / "configs/worldsim_v51/m1_unary_visibility_v1.yaml",
                PROJECT / "configs/worldsim_v51/m1_unary_baselines_v1.yaml",
                PROJECT / "motion_proj/worldsim_v51/evidence/abstention.py",
                PROJECT / "motion_proj/worldsim_v51/evidence/visibility.py",
                PROJECT / "scripts/run_worldsim_v51_unary_unknown.py",
                PROJECT / "scripts/run_worldsim_v51_unary_visibility.py",
                PROJECT / "scripts/run_worldsim_v5_m1_unary_diagnostic.py",
                PROJECT / "tests/test_worldsim_v51_abstention.py",
                PROJECT / "tests/test_run_worldsim_v51_unary_unknown.py",
            ],
            PROJECT,
        )
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v51_m1_a2_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_commit,
                "source_branch": V51_BRANCH,
                "worktree_clean": True,
                "resolved_config": resolved_config,
                "a1_inputs": config["inputs"],
                "source_snapshot": source_snapshot,
                "runtime": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(
                        torch.device(config["resources"]["device"])
                    ),
                },
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        manifest_path = run_dir / "manifest.json"
        atomic_json(
            manifest_path,
            {
                "schema_version": "worldsim_v51_m1_a2_manifest_v1",
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

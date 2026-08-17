#!/usr/bin/env python3
"""执行 V5.1 Stage A1 visibility-masked B3 matched ablation。"""

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
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v4.evidence_metrics import probability_metrics
from motion_proj.worldsim_v5.bayesian_unary import (
    accumulate_effective_count_statistics,
    empty_effective_count_statistics,
    finalize_effective_count_unary,
)
from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v5.ownership_renderer import (
    rasterize_ownership_probability,
)
from motion_proj.worldsim_v51.evidence.visibility import (
    accumulate_visibility_masked_b3_statistics,
)
from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
    verify_canonical_run,
)
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
SCHEMA_VERSION = "worldsim_v51_m1_unary_visibility_v1"
RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v51")
ARMS = ("B3", "A1")
METRICS = (
    "iou_at_frozen_threshold",
    "boundary_f1",
    "false_positive_semantic_mass",
    "false_negative_semantic_mass",
    "brier",
    "ece",
    "nll",
)


class VisibilityRunError(RuntimeError):
    """A1 输入、重放、渲染或门禁合同失败。"""


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _aggregate_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise VisibilityRunError("没有可聚合的 evaluation rows")
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in METRICS
    }


def _render_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    boundary_tolerance_px: int,
    ece_bins: int,
) -> dict[str, float]:
    metrics = probability_metrics(probability, target, bins=ece_bins)
    metrics.update(
        iou_at_frozen_threshold=v5_runner._binary_iou(
            probability >= threshold, target
        ),
        boundary_f1=v5_runner._boundary_f1(
            probability >= threshold,
            target,
            tolerance=boundary_tolerance_px,
        ),
        nll=v5_runner._negative_log_likelihood(probability, target),
    )
    return metrics


def _gaussian_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    metrics = probability_metrics(probability, target, bins=ece_bins)
    metrics.update(
        iou_at_frozen_threshold=v5_runner._binary_iou(
            probability >= threshold, target
        ),
        nll=v5_runner._negative_log_likelihood(probability, target),
    )
    return metrics


def _stage_a_gate(
    scene_deltas: list[Mapping[str, float]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    if len(scene_deltas) != 3:
        raise VisibilityRunError("A1 H gate 必须保留 3 个 historical scenes")
    mean_delta = {
        name: float(np.mean([float(row[name]) for row in scene_deltas]))
        for name in METRICS
    }
    positive = sum(float(row["boundary_f1"]) > 0.0 for row in scene_deltas)
    checks = {
        "boundary_f1_positive_scene_count": positive
        >= int(gate["boundary_f1_positive_scene_count_minimum"]),
        "mean_boundary_f1_positive": mean_delta["boundary_f1"]
        > float(gate["mean_boundary_f1_delta_minimum_exclusive"]),
        "mean_iou_nonnegative": mean_delta["iou_at_frozen_threshold"]
        >= float(gate["mean_iou_delta_minimum"]),
        "mean_fn_delta_bounded": mean_delta["false_negative_semantic_mass"]
        <= float(gate["mean_fn_semantic_mass_delta_maximum"]),
        "brier_or_ece_improves": (
            mean_delta["brier"] < 0.0 or mean_delta["ece"] < 0.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_boundary_f1_scene_count": positive,
        "mean_delta_a1_minus_b3": mean_delta,
    }


def _prepare_run(run_dir: Path) -> str:
    resolved = run_dir.resolve()
    task_root = (RUN_ROOT / TASK_ID).resolve()
    if resolved.exists():
        raise VisibilityRunError(f"run 目录已存在，禁止覆盖: {resolved}")
    if task_root not in resolved.parents:
        raise VisibilityRunError(f"run 必须位于 {task_root} 下")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise VisibilityRunError(f"A1 必须在 {V51_BRANCH} 执行")
    if _git("status", "--short"):
        raise VisibilityRunError("A1 formal run 要求 clean worktree")
    source_commit = _git("rev-parse", "HEAD")
    resolved.mkdir(parents=True)
    return source_commit


def _build_posteriors(
    source_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    minimum_visibility: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], np.ndarray]:
    run_dir = Path(spec["path"])
    canonical_b3 = _load_npz(run_dir / "artifacts/gaussians/B3.npz")
    base_model = canonical_b3["base_model"]
    gaussian_count = int(base_model.size)
    unary = source_config["unary"]
    unassigned = float(unary["unassigned_probability"])
    prior = np.full(gaussian_count, unassigned, dtype=np.float64)
    prior[base_model == "RigidNodes"] = 1.0 - unassigned
    b3_statistics = empty_effective_count_statistics(gaussian_count)
    a1_statistics = empty_effective_count_statistics(gaussian_count)
    chunk_reports = []
    visible_values = []
    observation_paths = sorted((run_dir / "artifacts/observations").glob("*.npz"))
    if len(observation_paths) != int(spec["evidence_view_count"]):
        raise VisibilityRunError("A1 evidence observation 分母漂移")
    for path in observation_paths:
        observations = _load_npz(path)
        accumulate_effective_count_statistics(
            b3_statistics,
            observations=observations,
            gaussian_count=gaussian_count,
            sam_confidence_floor=float(unary["sam_confidence_floor"]),
            boundary_distance_scale_px=float(unary["boundary_distance_scale_px"]),
            depth_residual_scale_m=float(unary["depth_residual_scale_m"]),
        )
        report = accumulate_visibility_masked_b3_statistics(
            a1_statistics,
            observations=observations,
            gaussian_count=gaussian_count,
            minimum_visibility=minimum_visibility,
            sam_confidence_floor=float(unary["sam_confidence_floor"]),
            boundary_distance_scale_px=float(unary["boundary_distance_scale_px"]),
            depth_residual_scale_m=float(unary["depth_residual_scale_m"]),
        )
        semantic_available = (
            np.asarray(observations["mask_quality_accepted"], dtype=bool)
            & np.asarray(observations["sam_probability_available"], dtype=bool)
        )
        visible_values.append(
            np.asarray(observations["visibility"], dtype=np.float32)[
                semantic_available
            ]
        )
        chunk_reports.append({"path": str(path), **report})
    outputs = {
        "B3": finalize_effective_count_unary(
            prior_probability=prior,
            prior_strength=float(unary["prior_strength"]),
            statistics=b3_statistics,
        ),
        "A1": finalize_effective_count_unary(
            prior_probability=prior,
            prior_strength=float(unary["prior_strength"]),
            statistics=a1_statistics,
        ),
    }
    for field in (
        "unary_posterior",
        "unary_uncertainty",
        "effective_evidence_count",
        "multi_view_disagreement",
        "boundary_ambiguity",
        "depth_support",
    ):
        if not np.array_equal(outputs["B3"][field], canonical_b3[field]):
            raise VisibilityRunError(f"A1 输入 B3 posterior 漂移: {field}")
    semantic_count = sum(row["semantic_available_count"] for row in chunk_reports)
    qualified_count = sum(row["visibility_qualified_count"] for row in chunk_reports)
    values = np.concatenate(visible_values) if semantic_count else np.empty(0)
    diagnostics = {
        "observation_file_count": len(observation_paths),
        "observation_count": sum(row["observation_count"] for row in chunk_reports),
        "semantic_available_count": semantic_count,
        "visibility_qualified_count": qualified_count,
        "visibility_rejected_count": sum(
            row["visibility_rejected_count"] for row in chunk_reports
        ),
        "semantic_unavailable_count": sum(
            row["semantic_unavailable_count"] for row in chunk_reports
        ),
        "valid_observation_ratio": float(qualified_count / max(semantic_count, 1)),
        "configured_minimum_visibility": float(minimum_visibility),
        "applied_minimum_visibility": chunk_reports[0][
            "applied_minimum_visibility"
        ],
        "visibility_quantiles": {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "chunk_reports": chunk_reports,
    }
    return outputs, diagnostics, base_model


def _run_scene(
    scene: str,
    spec: Mapping[str, Any],
    source_config_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = verify_canonical_run(scene, spec)
    source_config = load_yaml(source_config_path)
    outputs, observation_diagnostics, base_model = _build_posteriors(
        source_config,
        spec,
        minimum_visibility=float(config["visibility"]["minimum_visibility"]),
    )
    source_run = Path(spec["path"])
    source_summary = json.loads(
        (source_run / "summary.json").read_text(encoding="utf-8")
    )
    gaussian_target = (base_model == "RigidNodes").astype(np.float32)
    evaluation_config = source_config["evaluation"]
    gaussian_metrics = {
        arm: _gaussian_metrics(
            outputs[arm]["unary_posterior"],
            gaussian_target,
            threshold=float(evaluation_config["probability_threshold"]),
            ece_bins=int(evaluation_config["ece_bins"]),
        )
        for arm in ARMS
    }
    posterior_path = run_dir / "artifacts/posteriors" / f"{scene}.npz"
    atomic_save_npz(
        posterior_path,
        {
            f"{arm}_{field}": outputs[arm][field]
            for arm in ARMS
            for field in (
                "unary_posterior",
                "unary_uncertainty",
                "effective_evidence_count",
                "multi_view_disagreement",
                "boundary_ambiguity",
                "depth_support",
            )
        },
    )
    checkpoint = Path(source_config["inputs"]["formal_checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != spec["checkpoint_sha256"]:
        raise VisibilityRunError(f"A1 checkpoint SHA 漂移: {scene}")
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
        raise VisibilityRunError(f"A1 accepted evaluation 分母漂移: {scene}")
    _, dataset, trainer = v5_runner._build_runtime(source_config, device)
    get_view_data, _, release_trainer_render_info = v5_runner._runtime_helpers()
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    b3_byte_exact_count = 0
    try:
        for row in evaluation_rows:
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
            mask_path = sam_manifest_path.parents[1] / row["mask"]["path"]
            if sha256_file(mask_path) != row["mask"]["sha256"]:
                raise VisibilityRunError(f"A1 SAM mask SHA 漂移: {mask_path}")
            target = _load_npz(mask_path)["binary"].astype(bool)
            image_infos, camera_infos, *_ = get_view_data(
                dataset, frame, camera_id, device
            )
            try:
                with torch.inference_mode():
                    processed_camera, gaussians = v5_runner._collect_gaussians(
                        trainer, image_infos, camera_infos
                    )
                    for arm in ARMS:
                        alpha, _ = rasterize_ownership_probability(
                            means=gaussians.means,
                            quats=gaussians.quats,
                            scales=gaussians.scales,
                            base_opacities=gaussians.opacities,
                            probability=outputs[arm]["unary_posterior"],
                            viewmats=torch.linalg.inv(
                                processed_camera.camtoworlds
                            )[None, ...],
                            intrinsics=processed_camera.Ks[None, ...],
                            width=int(processed_camera.W),
                            height=int(processed_camera.H),
                            near_plane=float(trainer.render_cfg.near_plane),
                            far_plane=float(trainer.render_cfg.far_plane),
                            packed=bool(trainer.render_cfg.packed),
                            radius_clip=float(
                                trainer.render_cfg.get("radius_clip", 0.0)
                            ),
                            antialiased=bool(trainer.render_cfg.antialiased),
                        )
                        probability = alpha.detach().cpu().numpy().astype(np.float32)
                        if probability.shape != target.shape:
                            raise VisibilityRunError("A1 render/SAM shape 漂移")
                        output = (
                            run_dir
                            / "artifacts/evaluation"
                            / scene
                            / arm
                            / f"f{frame:03d}_c{camera_id}.npz"
                        )
                        atomic_save_npz(
                            output,
                            {
                                "probability": probability.astype(np.float16),
                                "target": target.astype(np.int8),
                            },
                        )
                        if arm == "B3":
                            canonical_path = (
                                source_run
                                / "artifacts/evaluation/B3"
                                / f"f{frame:03d}_c{camera_id}.npz"
                            )
                            if sha256_file(output) != sha256_file(canonical_path):
                                raise VisibilityRunError(
                                    f"A1 B3 GPU rerender 不 byte exact: {scene}/{frame}/{camera_id}"
                                )
                            b3_byte_exact_count += 1
                        metrics = _render_metrics(
                            probability,
                            target.astype(np.float32),
                            threshold=float(
                                evaluation_config["probability_threshold"]
                            ),
                            boundary_tolerance_px=int(
                                evaluation_config["boundary_tolerance_px"]
                            ),
                            ece_bins=int(evaluation_config["ece_bins"]),
                        )
                        rows_by_arm[arm].append(
                            {
                                "frame": frame,
                                "camera_id": camera_id,
                                **metrics,
                                "path": str(output.relative_to(run_dir)),
                                "sha256": sha256_file(output),
                            }
                        )
            finally:
                release_trainer_render_info(trainer)
            print(
                f"A1 {scene} evaluation frame={frame} camera={camera_id}",
                flush=True,
            )
    finally:
        del trainer
        del dataset
        torch.cuda.empty_cache()
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise VisibilityRunError(f"A1 checkpoint mutation: {scene}")
    aggregate = {arm: _aggregate_metrics(rows_by_arm[arm]) for arm in ARMS}
    canonical_b3 = source_summary["arm_evaluation_aggregate"]["B3"]
    b3_metric_delta = {
        name: float(aggregate["B3"][name] - canonical_b3[name])
        for name in METRICS
    }
    if any(value != 0.0 for value in b3_metric_delta.values()):
        raise VisibilityRunError(f"A1 B3 GPU metric replay 不 exact: {scene}")
    delta = {
        name: float(aggregate["A1"][name] - aggregate["B3"][name])
        for name in METRICS
    }
    scene_summary = {
        **canonical,
        "accepted_evaluation_view_count": len(evaluation_rows),
        "b3_gpu_rerender_byte_exact_count": b3_byte_exact_count,
        "b3_gpu_metric_delta_vs_canonical": b3_metric_delta,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "posterior_path": str(posterior_path.relative_to(run_dir)),
        "posterior_sha256": sha256_file(posterior_path),
        "gaussian_metrics": gaussian_metrics,
        "evaluation_aggregate": aggregate,
        "evaluation_delta_a1_minus_b3": delta,
        "observation_diagnostics": {
            key: value
            for key, value in observation_diagnostics.items()
            if key != "chunk_reports"
        },
    }
    scene_diagnostics = {
        "scene": scene,
        "observation_diagnostics": observation_diagnostics,
        "evaluation_rows": rows_by_arm,
    }
    return scene_summary, scene_diagnostics


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise VisibilityRunError("A1 config schema 漂移")
    if config.get("task_id") != TASK_ID or config.get("phase") != "a1_visibility_masked_b3":
        raise VisibilityRunError("A1 task/phase 漂移")
    baseline_binding = config["inputs"]["a0_baseline_config"]
    baseline_path = PROJECT / baseline_binding["path"]
    verify_file(baseline_path, baseline_binding["sha256"])
    baseline = load_yaml(baseline_path)
    if list(config["scenes"]) != list(baseline["canonical_runs"]):
        raise VisibilityRunError("A1 H scene 集合或顺序漂移")
    a0_binding = config["inputs"]["a0_run"]
    verify_file(Path(a0_binding["path"]) / "summary.json", a0_binding["summary_sha256"])
    verify_file(Path(a0_binding["path"]) / "manifest.json", a0_binding["manifest_sha256"])
    if not torch.cuda.is_available():
        raise VisibilityRunError("A1 需要 CUDA")
    device = torch.device(config["resources"]["device"])
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    maximum_start = int(config["resources"]["maximum_gpu_allocated_at_start_mib"])
    if torch.cuda.memory_allocated(device) > maximum_start * 1024**2:
        raise VisibilityRunError("A1 GPU preflight 非空闲")
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
            run_dir,
            device,
        )
        scene_summaries.append(summary)
        scene_diagnostics.append(diagnostics)
    gate = _stage_a_gate(
        [row["evaluation_delta_a1_minus_b3"] for row in scene_summaries],
        config["evaluation"]["scene_balanced_gate"],
    )
    diagnostics_path = run_dir / "artifacts/diagnostics.json"
    atomic_json(
        diagnostics_path,
        {
            "schema_version": "worldsim_v51_m1_a1_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene_diagnostics": scene_diagnostics,
        },
    )
    return {
        "schema_version": "worldsim_v51_m1_a1_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": config["phase"],
        "conclusion": (
            "a1_visibility_passed_h_gate_candidate_for_stage_a"
            if gate["passed"]
            else "a1_visibility_rejected_h_gate"
        ),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "seed": int(config["seed"]),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": int(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "visibility_threshold": config["visibility"],
        "scene_summaries": scene_summaries,
        "stage_a_h_gate": gate,
        "b3_gpu_rerender_byte_exact_view_count": sum(
            row["b3_gpu_rerender_byte_exact_count"] for row in scene_summaries
        ),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "method_inference_started": True,
        "graph_inference_started": False,
        "parameter_search_performed": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": "none" if gate["passed"] else "pending_rejection_entry",
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
            "schema_version": "worldsim_v51_m1_a1_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/m1_unary_visibility_v1.yaml",
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
                PROJECT / "configs/worldsim_v51/m1_unary_baselines_v1.yaml",
                PROJECT / "motion_proj/worldsim_v51/evidence/visibility.py",
                PROJECT / "scripts/run_worldsim_v51_unary_visibility.py",
                PROJECT / "scripts/replay_worldsim_v51_v5_unary.py",
                PROJECT / "scripts/run_worldsim_v5_m1_unary_diagnostic.py",
                PROJECT / "tests/test_worldsim_v51_visibility.py",
                PROJECT / "tests/test_run_worldsim_v51_unary_visibility.py",
            ],
            PROJECT,
        )
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v51_m1_a1_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_commit,
                "source_branch": V51_BRANCH,
                "worktree_clean": True,
                "resolved_config": resolved_config,
                "a0_inputs": config["inputs"],
                "source_snapshot": source_snapshot,
                "runtime": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(torch.device(config["resources"]["device"])),
                },
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        manifest_path = run_dir / "manifest.json"
        atomic_json(
            manifest_path,
            {
                "schema_version": "worldsim_v51_m1_a1_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
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

"""WorldSim V6 R111: immutable 2x2 actor-composition interaction certificate."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R111-SCENE0255-TWO-ACTOR-FACTORIAL-INTERACTION-CERTIFICATE-01"


class R111ExperimentError(RuntimeError):
    """The preregistered R111 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _verified_member(run: Path, manifest: dict[str, Any], relative: str) -> Path:
    if relative not in manifest["files"]:
        raise R111ExperimentError(f"R111 missing source manifest member: {relative}")
    path = run / relative
    _verify(path, manifest["files"][relative]["sha256"])
    return path


def _source_payload(
    run: Path,
    manifest: dict[str, Any],
    frame_indices: list[int],
    repeat_count: int,
) -> dict[str, Any]:
    sensor_metrics_path = _verified_member(run, manifest, "sensor_worker/FRAME_METRICS.jsonl")
    perception_outputs_path = _verified_member(run, manifest, "perception/PERCEPTION_OUTPUTS.jsonl")
    sensor_rows = _load_jsonl(sensor_metrics_path)
    if [int(row["frame_index"]) for row in sensor_rows] != frame_indices:
        raise R111ExperimentError("R111 source sensor denominator/order drift")
    sensor_paths: dict[int, Path] = {}
    for row in sensor_rows:
        relative = f"sensor_worker/{row['sensor_path']}"
        sensor_paths[int(row["frame_index"])] = _verified_member(run, manifest, relative)

    perception_rows = _load_jsonl(perception_outputs_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in perception_rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    expected_cases = {
        f"frame{frame:03d}_{variant}" for frame in frame_indices for variant in ("logged", "edited")
    }
    if set(grouped) != expected_cases:
        raise R111ExperimentError("R111 source perception case denominator drift")
    labels: dict[tuple[int, str], Path] = {}
    repeat_exact = True
    for case_id, items in grouped.items():
        repeat_exact = repeat_exact and len(items) == repeat_count and len(
            {str(item["label_array_sha256"]) for item in items}
        ) == 1
        first = sorted(items, key=lambda item: int(item["repeat_index"]))[0]
        relative = f"perception/{first['label_path']}"
        label_path = _verified_member(run, manifest, relative)
        if _sha256(label_path) != first["label_file_sha256"]:
            raise R111ExperimentError("R111 perception metadata/content hash drift")
        frame_text, variant = case_id.split("_", 1)
        labels[(int(frame_text.removeprefix("frame")), variant)] = label_path
    return {
        "sensor_paths": sensor_paths,
        "labels": labels,
        "repeat_exact": repeat_exact,
        "sensor_rows": sensor_rows,
        "perception_row_count": len(perception_rows),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R111ExperimentError("formal R111 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R111ExperimentError("R111 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    frame_indices = list(
        range(int(runtime["frame_start"]), int(runtime["frame_stop_exclusive"]), int(runtime["frame_stride"]))
    )
    if len(frame_indices) != int(runtime["expected_frame_count"]):
        raise R111ExperimentError("R111 frame denominator drift")
    repeat_count = int(runtime["repeat_count"])

    source_specs = {
        "actor34_only": {
            "run": _resolve_runs_uri(sources["r102_run"]),
            "gate_name": "R102_GATE.json",
            "prefix": "r102",
        },
        "actor24_only": {
            "run": _resolve_runs_uri(sources["r110_run"]),
            "gate_name": "R110_GATE.json",
            "prefix": "r110",
        },
        "joint": {
            "run": _resolve_runs_uri(sources["r109_run"]),
            "gate_name": "R109_GATE.json",
            "prefix": "r109",
        },
    }
    frozen_files: dict[Path, str] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    source_gates: dict[str, dict[str, Any]] = {}
    source_transfers: dict[str, dict[str, Any]] = {}
    for name, spec in source_specs.items():
        run = spec["run"]
        prefix = spec["prefix"]
        frozen_files.update(
            {
                run / "MANIFEST.json": sources[f"{prefix}_manifest_sha256"],
                run / spec["gate_name"]: sources[f"{prefix}_gate_sha256"],
                run / "SUMMARY.json": sources[f"{prefix}_summary_sha256"],
                run / "SELECTOR_TRANSFER.json": sources[f"{prefix}_selector_transfer_sha256"],
            }
        )
    for path, expected in frozen_files.items():
        _verify(path, expected)
    for name, spec in source_specs.items():
        run = spec["run"]
        source_manifests[name] = json.loads((run / "MANIFEST.json").read_text(encoding="utf-8"))
        source_gates[name] = json.loads((run / spec["gate_name"]).read_text(encoding="utf-8"))
        source_transfers[name] = json.loads((run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
    if not all(gate["checks"]["passed"] for gate in source_gates.values()):
        raise R111ExperimentError("R111 source authority rejected")

    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R111ExperimentError("R111 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scene0255-two-actor-factorial-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)

    payloads = {
        name: _source_payload(spec["run"], source_manifests[name], frame_indices, repeat_count)
        for name, spec in source_specs.items()
    }
    frame_rows: list[dict[str, Any]] = []
    actor34_targets: list[bool] = []
    actor24_targets: list[bool] = []
    joint_targets: list[bool] = []
    actor34_selector: list[bool] = []
    actor24_selector: list[bool] = []
    pixel_union_tp = pixel_union_fp = pixel_union_fn = pixel_union_tn = 0
    interaction_abs_sum = 0.0
    interaction_value_count = 0
    interaction_abs_max = 0.0
    interaction_pixels_gt_tolerance = 0
    all_logged_sensor_exact = True
    all_logged_label_exact = True
    interaction_tolerance = float(thresholds["sensor_interaction_absolute_tolerance"])
    frozen_threshold = int(runtime["frozen_policy_threshold_pixels"])

    for frame in frame_indices:
        sensor_arrays: dict[str, dict[str, np.ndarray]] = {}
        for name in source_specs:
            with np.load(payloads[name]["sensor_paths"][frame], allow_pickle=False) as archive:
                sensor_arrays[name] = {
                    "logged": np.asarray(archive["logged_rgb"]),
                    "edited": np.asarray(archive["compiled_rgb"]),
                }
        logged = sensor_arrays["actor34_only"]["logged"]
        all_logged_sensor_exact = all_logged_sensor_exact and np.array_equal(
            logged, sensor_arrays["actor24_only"]["logged"]
        ) and np.array_equal(logged, sensor_arrays["joint"]["logged"])
        actor34_rgb = sensor_arrays["actor34_only"]["edited"]
        actor24_rgb = sensor_arrays["actor24_only"]["edited"]
        joint_rgb = sensor_arrays["joint"]["edited"]
        actor34_changed = int(np.any(actor34_rgb != logged, axis=-1).sum())
        actor24_changed = int(np.any(actor24_rgb != logged, axis=-1).sum())
        joint_changed = int(np.any(joint_rgb != logged, axis=-1).sum())
        interaction = (
            joint_rgb.astype(np.float32)
            - actor34_rgb.astype(np.float32)
            - actor24_rgb.astype(np.float32)
            + logged.astype(np.float32)
        )
        abs_interaction = np.abs(interaction)
        interaction_abs_sum += float(abs_interaction.sum(dtype=np.float64))
        interaction_value_count += int(abs_interaction.size)
        interaction_abs_max = max(interaction_abs_max, float(abs_interaction.max()))
        interaction_pixel_count = int(np.any(abs_interaction > interaction_tolerance, axis=-1).sum())
        interaction_pixels_gt_tolerance += interaction_pixel_count

        label_arrays: dict[str, dict[str, np.ndarray]] = {}
        for name in source_specs:
            label_arrays[name] = {
                variant: np.load(payloads[name]["labels"][(frame, variant)], allow_pickle=False)
                for variant in ("logged", "edited")
            }
        base_label = label_arrays["actor34_only"]["logged"]
        all_logged_label_exact = all_logged_label_exact and np.array_equal(
            base_label, label_arrays["actor24_only"]["logged"]
        ) and np.array_equal(base_label, label_arrays["joint"]["logged"])
        actor34_label = label_arrays["actor34_only"]["edited"]
        actor24_label = label_arrays["actor24_only"]["edited"]
        joint_label = label_arrays["joint"]["edited"]
        actor34_mask = actor34_label != base_label
        actor24_mask = actor24_label != base_label
        joint_mask = joint_label != base_label
        union_mask = actor34_mask | actor24_mask
        tp = int((joint_mask & union_mask).sum())
        fp = int((joint_mask & ~union_mask).sum())
        fn = int((~joint_mask & union_mask).sum())
        tn = int((~joint_mask & ~union_mask).sum())
        pixel_union_tp += tp
        pixel_union_fp += fp
        pixel_union_fn += fn
        pixel_union_tn += tn
        actor34_label_pixels = int(actor34_mask.sum())
        actor24_label_pixels = int(actor24_mask.sum())
        joint_label_pixels = int(joint_mask.sum())
        actor34_marginal_pixels = int((joint_label != actor24_label).sum())
        actor24_marginal_pixels = int((joint_label != actor34_label).sum())
        actor34_targets.append(actor34_label_pixels > 0)
        actor24_targets.append(actor24_label_pixels > 0)
        joint_targets.append(joint_label_pixels > 0)
        actor34_selector.append(actor34_changed >= frozen_threshold)
        actor24_selector.append(actor24_changed >= frozen_threshold)
        frame_rows.append(
            {
                "frame_index": frame,
                "actor34_sensor_changed_pixels": actor34_changed,
                "actor24_sensor_changed_pixels": actor24_changed,
                "joint_sensor_changed_pixels": joint_changed,
                "sensor_factorial_interaction_pixels_gt_tolerance": interaction_pixel_count,
                "actor34_label_changed_pixels": actor34_label_pixels,
                "actor24_label_changed_pixels": actor24_label_pixels,
                "joint_label_changed_pixels": joint_label_pixels,
                "single_cell_union_label_pixels": int(union_mask.sum()),
                "joint_vs_single_union_true_positive_pixels": tp,
                "joint_vs_single_union_false_positive_pixels": fp,
                "joint_vs_single_union_false_negative_pixels": fn,
                "actor34_marginal_label_pixels_given_actor24": actor34_marginal_pixels,
                "actor24_marginal_label_pixels_given_actor34": actor24_marginal_pixels,
                "actor34_single_target": actor34_label_pixels > 0,
                "actor24_single_target": actor24_label_pixels > 0,
                "joint_target": joint_label_pixels > 0,
                "or_of_single_selectors": actor34_changed >= frozen_threshold
                or actor24_changed >= frozen_threshold,
            }
        )

    _write_jsonl(run_dir / "FACTORIAL_FRAME_METRICS.jsonl", frame_rows)
    actor34_target_array = np.asarray(actor34_targets, dtype=bool)
    actor24_target_array = np.asarray(actor24_targets, dtype=bool)
    joint_target_array = np.asarray(joint_targets, dtype=bool)
    single_truth_union = actor34_target_array | actor24_target_array
    selector_or = np.asarray(actor34_selector, dtype=bool) | np.asarray(actor24_selector, dtype=bool)
    frame_union_metrics = _metrics(joint_target_array, single_truth_union)
    selector_or_metrics = _metrics(selector_or, joint_target_array)
    pixel_precision = float(pixel_union_tp / (pixel_union_tp + pixel_union_fp)) if pixel_union_tp + pixel_union_fp else 0.0
    pixel_recall = float(pixel_union_tp / (pixel_union_tp + pixel_union_fn)) if pixel_union_tp + pixel_union_fn else 0.0
    pixel_f1 = (
        float(2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall))
        if pixel_precision + pixel_recall
        else 0.0
    )
    pixel_jaccard = (
        float(pixel_union_tp / (pixel_union_tp + pixel_union_fp + pixel_union_fn))
        if pixel_union_tp + pixel_union_fp + pixel_union_fn
        else 0.0
    )
    certificate = {
        "schema_version": "worldsim_v6.r111_factorial_certificate.v1",
        "scene": runtime["scene"],
        "factorial_cells": {
            "00": "logged",
            "10": "actor34_only",
            "01": "actor24_only",
            "11": "actor34_actor24_joint",
        },
        "frame_count": len(frame_indices),
        "actor34_single_positive_frames": int(actor34_target_array.sum()),
        "actor24_single_positive_frames": int(actor24_target_array.sum()),
        "single_truth_union_positive_frames": int(single_truth_union.sum()),
        "joint_positive_frames": int(joint_target_array.sum()),
        "frame_joint_vs_single_truth_union_metrics": frame_union_metrics,
        "or_of_single_selectors_vs_joint_target_metrics": selector_or_metrics,
        "actor34_marginal_frame_count_given_actor24": sum(
            row["actor34_marginal_label_pixels_given_actor24"] > 0 for row in frame_rows
        ),
        "actor24_marginal_frame_count_given_actor34": sum(
            row["actor24_marginal_label_pixels_given_actor34"] > 0 for row in frame_rows
        ),
        "pixel_joint_vs_single_union": {
            "true_positive": pixel_union_tp,
            "false_positive": pixel_union_fp,
            "false_negative": pixel_union_fn,
            "true_negative": pixel_union_tn,
            "precision": pixel_precision,
            "recall": pixel_recall,
            "f1": pixel_f1,
            "jaccard": pixel_jaccard,
        },
        "sensor_factorial_interaction": {
            "formula": "rgb11-rgb10-rgb01+rgb00",
            "absolute_tolerance": interaction_tolerance,
            "mean_absolute_value": float(interaction_abs_sum / interaction_value_count),
            "maximum_absolute_value": interaction_abs_max,
            "pixels_above_tolerance": interaction_pixels_gt_tolerance,
        },
        "semantic_correctness_local_causality_contact_dynamics_physics_planning_safety": "ABSTAIN",
    }
    _write_json(run_dir / "FACTORIAL_CERTIFICATE.json", certificate)

    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "r102_r109_r110_authorities_accepted": all(
            gate["checks"]["passed"] for gate in source_gates.values()
        ),
        "source_manifests_and_consumed_arrays_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "three_source_frame_denominators_exact": all(
            len(payload["sensor_rows"]) == len(frame_indices) for payload in payloads.values()
        ),
        "three_source_perception_denominators_exact": all(
            payload["perception_row_count"] == len(frame_indices) * 2 * repeat_count
            for payload in payloads.values()
        ),
        "source_perception_repeat_exact": all(payload["repeat_exact"] for payload in payloads.values()),
        "logged_sensor_cell_exact_across_sources": all_logged_sensor_exact,
        "logged_perception_cell_exact_across_sources": all_logged_label_exact,
        "actor34_marginal_support_nontrivial": certificate[
            "actor34_marginal_frame_count_given_actor24"
        ] >= int(thresholds["minimum_actor34_marginal_frames"]),
        "actor24_marginal_support_nontrivial": certificate[
            "actor24_marginal_frame_count_given_actor34"
        ] >= int(thresholds["minimum_actor24_marginal_frames"]),
        "joint_frame_target_matches_single_truth_union": frame_union_metrics["f1"]
        >= float(thresholds["minimum_frame_union_f1"]),
        "joint_pixel_target_matches_single_pixel_union": pixel_f1
        >= float(thresholds["minimum_pixel_union_f1"]),
        "or_of_single_selectors_safe_for_joint_target": selector_or_metrics["f1"]
        >= float(thresholds["minimum_selector_or_joint_f1"]),
        "sensor_factorial_interaction_detected": interaction_pixels_gt_tolerance
        >= int(thresholds["minimum_sensor_interaction_pixels"]),
        "semantic_correctness_local_causality_contact_dynamics_physics_planning_safety_abstain": True,
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R111_GATE.json",
        {
            "schema_version": "worldsim_v6.r111_gate.v1",
            "checks": checks,
            "decision": "accept_bounded_two_actor_factorial_interaction_certificate"
            if checks["passed"]
            else "reject_two_actor_factorial_interaction_certificate",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r111_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r111_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_bounded_two_actor_factorial_interaction"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "frame_count": len(frame_indices),
        "actor34_single_positive_frames": certificate["actor34_single_positive_frames"],
        "actor24_single_positive_frames": certificate["actor24_single_positive_frames"],
        "single_truth_union_positive_frames": certificate["single_truth_union_positive_frames"],
        "joint_positive_frames": certificate["joint_positive_frames"],
        "frame_union_f1": frame_union_metrics["f1"],
        "pixel_union_f1": pixel_f1,
        "pixel_union_jaccard": pixel_jaccard,
        "selector_or_joint_f1": selector_or_metrics["f1"],
        "actor34_marginal_frames": certificate["actor34_marginal_frame_count_given_actor24"],
        "actor24_marginal_frames": certificate["actor24_marginal_frame_count_given_actor34"],
        "sensor_interaction_pixels": interaction_pixels_gt_tolerance,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R111_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
        "FACTORIAL_CERTIFICATE.json",
        "FACTORIAL_FRAME_METRICS.jsonl",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r111_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        },
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {
            "schema_version": "worldsim_v6.terminal.v1",
            "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        },
    )
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/worldsim_v6/r111_scene0255_two_actor_factorial_interaction_certificate_v1.yaml"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WorldSim V6 R133: execute threshold-13 selective perception on real GPU inputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


TASK_ID = "WS-V6-R133-SELECTIVE-PERCEPTION-EXECUTION-01"


class R133ExperimentError(RuntimeError):
    """The preregistered R133 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
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


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R133ExperimentError("formal R133 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R133ExperimentError("R133 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r130_run = _resolve_runs_uri(sources["r130_run"])
    r131_run = _resolve_runs_uri(sources["r131_run"])
    r132_run = _resolve_runs_uri(sources["r132_run"])
    model_root = Path(sources["semantic_model_root"])
    frozen_files: dict[Path, str] = {
        r130_run / "MANIFEST.json": sources["r130_manifest_sha256"],
        r130_run / "R130_GATE.json": sources["r130_gate_sha256"],
        r130_run / "SUMMARY.json": sources["r130_summary_sha256"],
        r130_run / "SELECTOR_TRANSFER.json": sources["r130_selector_transfer_sha256"],
        r130_run / "RESOURCE_AUDIT.json": sources["r130_resource_audit_sha256"],
        r131_run / "MANIFEST.json": sources["r131_manifest_sha256"],
        r131_run / "R131_GATE.json": sources["r131_gate_sha256"],
        r131_run / "SUMMARY.json": sources["r131_summary_sha256"],
        r131_run / "SELECTOR_TRANSFER.json": sources["r131_selector_transfer_sha256"],
        r131_run / "RESOURCE_AUDIT.json": sources["r131_resource_audit_sha256"],
        r132_run / "MANIFEST.json": sources["r132_manifest_sha256"],
        r132_run / "R132_GATE.json": sources["r132_gate_sha256"],
        r132_run / "SUMMARY.json": sources["r132_summary_sha256"],
        r132_run / "VALIDATION_CERTIFICATE.json": sources["r132_certificate_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)

    r130_gate = json.loads((r130_run / "R130_GATE.json").read_text(encoding="utf-8"))
    r131_gate = json.loads((r131_run / "R131_GATE.json").read_text(encoding="utf-8"))
    r132_gate = json.loads((r132_run / "R132_GATE.json").read_text(encoding="utf-8"))
    certificate = json.loads(
        (r132_run / "VALIDATION_CERTIFICATE.json").read_text(encoding="utf-8")
    )
    threshold = int(certificate["threshold_pixels"])
    if threshold != int(runtime["frozen_threshold_pixels"]):
        raise R133ExperimentError("R132 certificate threshold drift")

    condition_specs = (
        ("r130_scene0230_antithetic", r130_run),
        ("r131_scene0048_orthogonal", r131_run),
    )
    frame_records: list[dict[str, Any]] = []
    input_index: list[dict[str, Any]] = []
    output_reference: dict[tuple[str, int, str, int], str] = {}
    input_sensor_hashes: dict[Path, str] = {}
    reference_index_hashes: dict[Path, str] = {}
    condition_trigger_counts: dict[str, int] = {}
    for condition, run in condition_specs:
        manifest = json.loads((run / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest_files = manifest["files"]
        transfer = json.loads((run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
        if int(transfer["frame_count"]) != int(runtime["frames_per_condition"]):
            raise R133ExperimentError(f"{condition} frame denominator drift")
        reference_path = run / "perception/PERCEPTION_OUTPUTS.jsonl"
        reference_record = manifest_files["perception/PERCEPTION_OUTPUTS.jsonl"]
        _verify(reference_path, reference_record["sha256"])
        reference_index_hashes[reference_path] = reference_record["sha256"]
        for row in _load_jsonl(reference_path):
            output_reference[
                (condition, int(row["frame_index"]), row["variant"], int(row["repeat_index"]))
            ] = row["label_array_sha256"]

        condition_triggers = 0
        for frame in range(int(runtime["frames_per_condition"])):
            key = str(frame)
            feature = int(transfer["sensor_changed_pixels_by_frame"][key])
            target_count = int(transfer["changed_label_pixels_by_frame"][key])
            trigger = feature >= threshold
            condition_triggers += int(trigger)
            record = {
                "condition": condition,
                "frame_index": frame,
                "changed_rgb_pixels": feature,
                "full_reference_changed_label_pixels": target_count,
                "trigger_expensive_perception": trigger,
            }
            frame_records.append(record)
            if not trigger:
                continue
            relative_sensor = f"sensor_worker/sensors/frame{frame:03d}.npz"
            sensor_record = manifest_files[relative_sensor]
            sensor_path = run / relative_sensor
            _verify(sensor_path, sensor_record["sha256"])
            input_sensor_hashes[sensor_path] = sensor_record["sha256"]
            for variant, rgb_key in (("logged", "logged_rgb"), ("edited", "compiled_rgb")):
                for repeat in range(int(evaluation["repeat_count"])):
                    case_id = f"{condition}__frame{frame:03d}_{variant}"
                    input_index.append(
                        {
                            "case_id": case_id,
                            "condition": condition,
                            "frame_index": frame,
                            "variant": variant,
                            "rgb_key": rgb_key,
                            "repeat_index": repeat,
                            "render_path": str(sensor_path),
                        }
                    )
        condition_trigger_counts[condition] = condition_triggers

    trigger_count = sum(condition_trigger_counts.values())
    frame_count = len(frame_records)
    full_invocations = frame_count * 2 * int(evaluation["repeat_count"])
    selective_invocations = len(input_index)
    avoided_invocations = full_invocations - selective_invocations
    invocation_reduction_fraction = float(avoided_invocations / full_invocations)
    if free_gib := shutil.disk_usage(run_root).free / (1024**3):
        if free_gib < float(resources["minimum_disk_free_gib"]):
            raise R133ExperimentError("R133 disk resource insufficient")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__selective-perception-execution-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    index_path = run_dir / "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl"
    _write_jsonl(index_path, input_index)
    output_dir = run_dir / "selective_perception"
    command = [
        sys.executable,
        str(repo_root / "scripts/worldsim_v6/r87_full_episode_perception_worker.py"),
        "--index",
        str(index_path),
        "--model-root",
        str(model_root),
        "--output-dir",
        str(output_dir),
    ]
    worker_env = os.environ.copy()
    worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with (run_dir / "selective_perception.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            cwd=repo_root,
            env=worker_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=float(resources["maximum_worker_seconds"]),
        )
    worker = json.loads((output_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
    output_rows = _load_jsonl(output_dir / "PERCEPTION_OUTPUTS.jsonl")
    index_by_case = {row["case_id"]: row for row in input_index}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    bit_exact_to_reference = True
    for row in output_rows:
        item = index_by_case[row["case_id"]]
        key = (item["condition"], int(row["frame_index"]), row["variant"])
        grouped.setdefault(key, []).append(row)
        reference_key = (
            item["condition"],
            int(row["frame_index"]),
            row["variant"],
            int(row["repeat_index"]),
        )
        bit_exact_to_reference &= row["label_array_sha256"] == output_reference[reference_key]
    repeat_exact = all(
        len(items) == int(evaluation["repeat_count"])
        and len({item["label_array_sha256"] for item in items}) == 1
        for items in grouped.values()
    )

    observed_counts: dict[tuple[str, int], int] = {}
    for record in frame_records:
        if not record["trigger_expensive_perception"]:
            continue
        condition = record["condition"]
        frame = int(record["frame_index"])
        arrays: dict[str, np.ndarray] = {}
        for variant in ("logged", "edited"):
            first = sorted(grouped[(condition, frame, variant)], key=lambda row: row["repeat_index"])[0]
            arrays[variant] = np.load(output_dir / first["label_path"], allow_pickle=False)
        observed_counts[(condition, frame)] = int((arrays["edited"] != arrays["logged"]).sum())

    decision_rows: list[dict[str, Any]] = []
    predictions: list[bool] = []
    targets: list[bool] = []
    triggered_counts_match = True
    for record in frame_records:
        key = (record["condition"], int(record["frame_index"]))
        observed = observed_counts.get(key, 0)
        target_count = int(record["full_reference_changed_label_pixels"])
        if record["trigger_expensive_perception"]:
            triggered_counts_match &= observed == target_count
        reconstructed = observed > 0
        target = target_count > 0
        predictions.append(reconstructed)
        targets.append(target)
        decision_rows.append(
            {
                **record,
                "selective_observed_changed_label_pixels": observed
                if record["trigger_expensive_perception"]
                else None,
                "reconstructed_any_changed_label": reconstructed,
                "full_reference_any_changed_label": target,
            }
        )
    reconstruction_metrics = _metrics(
        np.asarray(predictions, dtype=bool), np.asarray(targets, dtype=bool)
    )
    _write_jsonl(run_dir / "RUNTIME_DECISIONS.jsonl", decision_rows)

    source_full_seconds = sum(
        float(json.loads((run / "RESOURCE_AUDIT.json").read_text(encoding="utf-8"))[
            "perception_worker_seconds"
        ])
        for _, run in condition_specs
    )
    elapsed_ratio = float(worker["elapsed_seconds"] / source_full_seconds)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    sources_immutable = all(_sha256(path) == expected for path, expected in frozen_files.items())
    sources_immutable &= all(
        _sha256(path) == expected for path, expected in input_sensor_hashes.items()
    )
    sources_immutable &= all(
        _sha256(path) == expected for path, expected in reference_index_hashes.items()
    )
    checks = {
        "r130_r131_r132_authorities_accepted": bool(
            r130_gate["checks"]["passed"]
            and r131_gate["checks"]["passed"]
            and r132_gate["checks"]["passed"]
        ),
        "source_sensor_model_and_reference_files_immutable": sources_immutable,
        "threshold13_bound_from_r132_certificate": threshold == 13
        and certificate["validation_scope"] == "r130_and_r131_prospective_conditions_only",
        "frame_trigger_skip_denominators_exact": frame_count == int(runtime["expected_frame_count"])
        and trigger_count == int(runtime["expected_trigger_count"])
        and frame_count - trigger_count == int(runtime["expected_skip_count"]),
        "per_condition_trigger_counts_exact": condition_trigger_counts
        == runtime["expected_condition_trigger_counts"],
        "selective_invocation_denominator_exact": selective_invocations
        == len(output_rows)
        == int(runtime["expected_selective_invocations"]),
        "invocation_reduction_exact": full_invocations == int(runtime["expected_full_invocations"])
        and avoided_invocations == int(runtime["expected_avoided_invocations"])
        and abs(invocation_reduction_fraction - float(runtime["expected_reduction_fraction"]))
        <= float(evaluation["metric_tolerance"]),
        "selective_outputs_repeat_exact": repeat_exact,
        "selective_outputs_bit_exact_to_full_reference": bit_exact_to_reference,
        "triggered_changed_label_counts_exact_to_full_reference": triggered_counts_match,
        "reconstructed392_binary_decisions_zero_error": reconstruction_metrics["false_positive"] == 0
        and reconstruction_metrics["false_negative"] == 0
        and reconstruction_metrics["f1"] == 1.0,
        "measured_worker_elapsed_reduced": elapsed_ratio
        <= float(evaluation["maximum_elapsed_ratio_vs_full_reference"]),
        "gpu_within_budget": float(worker["peak_gpu_memory_mib"])
        <= float(resources["maximum_peak_gpu_memory_mib"]),
        "worker_and_wall_within_budget": float(worker["elapsed_seconds"])
        <= float(resources["maximum_worker_seconds"])
        and wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
        "training_not_started": True,
        "confirmation_not_read": True,
        "sensor_render_semantics_causality_physics_planning_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    result = {
        "schema_version": "worldsim_v6.r133_selective_execution_result.v1",
        "threshold_pixels": threshold,
        "frame_count": frame_count,
        "trigger_count": trigger_count,
        "skip_count": frame_count - trigger_count,
        "full_reference_invocations": full_invocations,
        "selective_invocations": selective_invocations,
        "avoided_invocations": avoided_invocations,
        "invocation_reduction_fraction": invocation_reduction_fraction,
        "condition_trigger_counts": condition_trigger_counts,
        "reconstruction_metrics": reconstruction_metrics,
        "source_full_reference_worker_seconds": source_full_seconds,
        "selective_worker_seconds": float(worker["elapsed_seconds"]),
        "elapsed_ratio_vs_full_reference": elapsed_ratio,
        "peak_gpu_memory_mib": float(worker["peak_gpu_memory_mib"]),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SELECTIVE_EXECUTION_RESULT.json", result)
    _write_json(
        run_dir / "R133_GATE.json",
        {
            "schema_version": "worldsim_v6.r133_gate.v1",
            "checks": checks,
            "decision": "accept_real_selective_perception_execution"
            if checks["passed"]
            else "reject_or_repair_selective_perception_execution",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r133_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "perception_worker_seconds": float(worker["elapsed_seconds"]),
            "source_full_reference_worker_seconds": source_full_seconds,
            "elapsed_ratio_vs_full_reference": elapsed_ratio,
            "peak_gpu_memory_mib": float(worker["peak_gpu_memory_mib"]),
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r133_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_real_selective_perception_execution"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "threshold_pixels": threshold,
        "frame_count": frame_count,
        "trigger_count": trigger_count,
        "skip_count": frame_count - trigger_count,
        "selective_invocations": selective_invocations,
        "avoided_invocations": avoided_invocations,
        "invocation_reduction_fraction": invocation_reduction_fraction,
        "false_positive": reconstruction_metrics["false_positive"],
        "false_negative": reconstruction_metrics["false_negative"],
        "f1": reconstruction_metrics["f1"],
        "selective_worker_seconds": float(worker["elapsed_seconds"]),
        "source_full_reference_worker_seconds": source_full_seconds,
        "elapsed_ratio_vs_full_reference": elapsed_ratio,
        "peak_gpu_memory_mib": float(worker["peak_gpu_memory_mib"]),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl",
        "RUNTIME_DECISIONS.jsonl",
        "SELECTIVE_EXECUTION_RESULT.json",
        "selective_perception.log",
        "selective_perception/PERCEPTION_OUTPUTS.jsonl",
        "selective_perception/WORKER_RESULT.json",
        "R133_GATE.json",
        "RESOURCE_AUDIT.json",
        "SUMMARY.json",
    ]
    tracked.extend(
        str(path.relative_to(run_dir)) for path in sorted(output_dir.glob("*.npy"))
    )
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r133_manifest.v1",
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
        default=Path("configs/worldsim_v6/r133_selective_perception_execution_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

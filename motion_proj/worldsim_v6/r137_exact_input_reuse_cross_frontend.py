"""WorldSim V6 R137: exact-input cache reuse with real cross-frontend execution."""

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


TASK_ID = "WS-V6-R137-EXACT-INPUT-REUSE-CROSS-FRONTEND-01"


class R137ExperimentError(RuntimeError):
    """The preregistered R137 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R137ExperimentError("formal R137 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R137ExperimentError("R137 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r133_run = _resolve_runs_uri(sources["r133_run"])
    r134_run = _resolve_runs_uri(sources["r134_run"])
    model_root = Path(sources["semantic_model_root"])
    frozen_files: dict[Path, str] = {
        r133_run / "MANIFEST.json": sources["r133_manifest_sha256"],
        r133_run / "R133_GATE.json": sources["r133_gate_sha256"],
        r133_run / "SUMMARY.json": sources["r133_summary_sha256"],
        r133_run / "SELECTIVE_EXECUTION_RESULT.json": sources["r133_result_sha256"],
        r133_run / "RESOURCE_AUDIT.json": sources["r133_resource_audit_sha256"],
        r134_run / "MANIFEST.json": sources["r134_manifest_sha256"],
        r134_run / "R134_GATE.json": sources["r134_gate_sha256"],
        r134_run / "SUMMARY.json": sources["r134_summary_sha256"],
        r134_run / "CROSS_FRONTEND_TRANSFER.json": sources["r134_transfer_sha256"],
        r134_run / "RESOURCE_AUDIT.json": sources["r134_resource_audit_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    r133_gate = json.loads((r133_run / "R133_GATE.json").read_text(encoding="utf-8"))
    r133_result = json.loads(
        (r133_run / "SELECTIVE_EXECUTION_RESULT.json").read_text(encoding="utf-8")
    )
    r134_gate = json.loads((r134_run / "R134_GATE.json").read_text(encoding="utf-8"))
    r134_manifest = json.loads((r134_run / "MANIFEST.json").read_text(encoding="utf-8"))
    r134_manifest_files = r134_manifest["files"]

    sensor_metrics_path = r134_run / "sensor_worker/FRAME_METRICS.jsonl"
    reference_index_path = r134_run / "perception/PERCEPTION_OUTPUTS.jsonl"
    for path, relative in (
        (sensor_metrics_path, "sensor_worker/FRAME_METRICS.jsonl"),
        (reference_index_path, "perception/PERCEPTION_OUTPUTS.jsonl"),
    ):
        _verify(path, r134_manifest_files[relative]["sha256"])
    sensor_rows = _load_jsonl(sensor_metrics_path)
    frames = [int(row["frame_index"]) for row in sensor_rows]
    if len(frames) != int(runtime["expected_frame_count"]) or len(set(frames)) != len(frames):
        raise R137ExperimentError("R137 AD-GS frame denominator drift")

    reference: dict[tuple[int, str, int], str] = {}
    for row in _load_jsonl(reference_index_path):
        reference[(int(row["frame_index"]), row["variant"], int(row["repeat_index"]))] = row[
            "label_array_sha256"
        ]
    expected_full = len(frames) * 2 * int(evaluation["repeat_count"])
    if len(reference) != expected_full:
        raise R137ExperimentError("R137 full perception reference denominator drift")

    decisions: list[dict[str, Any]] = []
    input_index: list[dict[str, Any]] = []
    sensor_hashes: dict[Path, str] = {}
    for row in sensor_rows:
        frame = int(row["frame_index"])
        relative = row["sensor_path"]
        sensor_path = r134_run / "sensor_worker" / relative
        manifest_relative = f"sensor_worker/{relative}"
        _verify(sensor_path, r134_manifest_files[manifest_relative]["sha256"])
        sensor_hashes[sensor_path] = r134_manifest_files[manifest_relative]["sha256"]
        with np.load(sensor_path, allow_pickle=False) as arrays:
            logged = arrays["logged_rgb"]
            compiled = arrays["compiled_rgb"]
            identical = bool(np.array_equal(logged, compiled))
            changed_pixels = int(np.any(logged != compiled, axis=2).sum())
        if changed_pixels != int(row["edited_vs_logged_rgb_changed_pixels"]):
            raise R137ExperimentError(f"R137 sensor feature drift at frame {frame}")
        execute = not identical
        decisions.append(
            {
                "frame_index": frame,
                "logged_compiled_rgb_byte_identical": identical,
                "changed_rgb_pixels": changed_pixels,
                "decision": "reuse_cached_logged_perception" if identical else "execute_perception",
            }
        )
        if execute:
            for variant, rgb_key in (("logged", "logged_rgb"), ("edited", "compiled_rgb")):
                for repeat in range(int(evaluation["repeat_count"])):
                    input_index.append(
                        {
                            "case_id": f"adgs_frame{frame:03d}_{variant}",
                            "frame_index": frame,
                            "variant": variant,
                            "rgb_key": rgb_key,
                            "repeat_index": repeat,
                            "render_path": str(sensor_path),
                        }
                    )

    trigger_count = sum(row["decision"] == "execute_perception" for row in decisions)
    skip_count = len(decisions) - trigger_count
    full_invocations = expected_full
    selective_invocations = len(input_index)
    avoided_invocations = full_invocations - selective_invocations
    reduction_fraction = float(avoided_invocations / full_invocations)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R137ExperimentError("R137 disk resource insufficient")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__exact-input-reuse-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    index_path = run_dir / "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl"
    _write_jsonl(index_path, input_index)
    output_dir = run_dir / "selective_perception"
    worker_env = os.environ.copy()
    worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
    actual: dict[tuple[int, str, int], str] = {
        (int(row["frame_index"]), row["variant"], int(row["repeat_index"])): row[
            "label_array_sha256"
        ]
        for row in output_rows
    }
    repeat_exact = all(
        len({actual[(frame, variant, repeat)] for repeat in range(int(evaluation["repeat_count"]))})
        == 1
        for frame in frames
        if any(row["frame_index"] == frame and row["decision"] == "execute_perception" for row in decisions)
        for variant in ("logged", "edited")
    )

    reconstruction_errors: list[dict[str, Any]] = []
    reuse_contract_errors: list[int] = []
    for decision in decisions:
        frame = int(decision["frame_index"])
        reuse = decision["decision"] == "reuse_cached_logged_perception"
        if reuse:
            all_reference_hashes = {
                reference[(frame, variant, repeat)]
                for variant in ("logged", "edited")
                for repeat in range(int(evaluation["repeat_count"]))
            }
            if not decision["logged_compiled_rgb_byte_identical"] or len(all_reference_hashes) != 1:
                reuse_contract_errors.append(frame)
            cached_hash = reference[(frame, "logged", 0)]
        for variant in ("logged", "edited"):
            for repeat in range(int(evaluation["repeat_count"])):
                key = (frame, variant, repeat)
                reconstructed = cached_hash if reuse else actual[key]
                if reconstructed != reference[key]:
                    reconstruction_errors.append(
                        {
                            "frame_index": frame,
                            "variant": variant,
                            "repeat_index": repeat,
                            "reconstructed_sha256": reconstructed,
                            "reference_sha256": reference[key],
                        }
                    )
    _write_jsonl(run_dir / "RUNTIME_DECISIONS.jsonl", decisions)

    r134_full_seconds = float(
        json.loads((r134_run / "RESOURCE_AUDIT.json").read_text(encoding="utf-8"))[
            "perception_worker_seconds"
        ]
    )
    elapsed_ratio = float(worker["elapsed_seconds"] / r134_full_seconds)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "r133_streetgs_real_execution_authority_accepted": bool(r133_gate["checks"]["passed"])
        and r133_result["reconstruction_metrics"]["false_positive"] == 0
        and r133_result["reconstruction_metrics"]["false_negative"] == 0,
        "r134_adgs_source_is_rejected_threshold_transfer_not_rehabilitated": not bool(
            r134_gate["checks"]["passed"]
        ),
        "source_sensor_model_and_reference_files_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        )
        and all(_sha256(path) == expected for path, expected in sensor_hashes.items()),
        "exact157_adgs_development_denominator": len(decisions)
        == int(runtime["expected_frame_count"]),
        "identity_guard_has_no_numeric_threshold": all(
            (row["decision"] == "reuse_cached_logged_perception")
            == row["logged_compiled_rgb_byte_identical"]
            == (row["changed_rgb_pixels"] == 0)
            for row in decisions
        ),
        "trigger_skip_denominators_exact": trigger_count == int(runtime["expected_trigger_count"])
        and skip_count == int(runtime["expected_skip_count"]),
        "selective_invocation_denominator_exact": selective_invocations
        == len(output_rows)
        == int(runtime["expected_selective_invocations"]),
        "invocation_reduction_gate": full_invocations == int(runtime["expected_full_invocations"])
        and avoided_invocations == int(runtime["expected_avoided_invocations"])
        and reduction_fraction >= float(evaluation["minimum_invocation_reduction_fraction"]),
        "executed_outputs_repeat_exact": repeat_exact,
        "zero_false_reuse_and_all628_reconstructed_hashes_exact": not reuse_contract_errors
        and not reconstruction_errors,
        "measured_worker_elapsed_reduced": elapsed_ratio
        <= float(evaluation["maximum_elapsed_ratio_vs_full_reference"]),
        "gpu_within_budget": float(worker["peak_gpu_memory_mib"])
        <= float(resources["maximum_peak_gpu_memory_mib"]),
        "worker_and_wall_within_budget": float(worker["elapsed_seconds"])
        <= float(resources["maximum_worker_seconds"])
        and wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
        "training_not_started_and_r136_heldout_not_read": True,
        "semantic_correctness_causality_contact_physics_planning_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    result = {
        "schema_version": "worldsim_v6.r137_exact_input_reuse_result.v1",
        "policy": "reuse_cached_logged_perception_iff_rgb_byte_identical_else_execute",
        "frontend": "ad_gs",
        "frame_count": len(decisions),
        "trigger_count": trigger_count,
        "skip_count": skip_count,
        "full_reference_invocations": full_invocations,
        "selective_invocations": selective_invocations,
        "avoided_invocations": avoided_invocations,
        "invocation_reduction_fraction": reduction_fraction,
        "reconstruction_error_count": len(reconstruction_errors),
        "false_reuse_count": len(reuse_contract_errors),
        "source_full_reference_worker_seconds": r134_full_seconds,
        "selective_worker_seconds": float(worker["elapsed_seconds"]),
        "elapsed_ratio_vs_full_reference": elapsed_ratio,
        "streetgs_bound_result": {
            "frame_count": r133_result["frame_count"],
            "reconstruction_metrics": r133_result["reconstruction_metrics"],
            "invocation_reduction_fraction": r133_result["invocation_reduction_fraction"],
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "EXACT_INPUT_REUSE_RESULT.json", result)
    _write_json(
        run_dir / "R137_GATE.json",
        {
            "schema_version": "worldsim_v6.r137_gate.v1",
            "checks": checks,
            "decision": "accept_exact_input_reuse_cross_frontend_development"
            if checks["passed"]
            else "reject_exact_input_reuse_cross_frontend_development",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r137_resource_audit.v1",
            "worker_seconds": float(worker["elapsed_seconds"]),
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_mib": float(worker["peak_gpu_memory_mib"]),
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "r136_heldout_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r137_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_cross_frontend_exact_input_reuse_development"
        if checks["passed"]
        else "rejected_cross_frontend_exact_input_reuse_development",
        "source_commit": source_commit,
        "adgs_frame_count": len(decisions),
        "adgs_trigger_count": trigger_count,
        "adgs_skip_count": skip_count,
        "adgs_avoided_invocations": avoided_invocations,
        "adgs_invocation_reduction_fraction": reduction_fraction,
        "adgs_reconstruction_error_count": len(reconstruction_errors),
        "adgs_false_reuse_count": len(reuse_contract_errors),
        "adgs_elapsed_ratio_vs_full_reference": elapsed_ratio,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl",
        "selective_perception.log",
        "selective_perception/PERCEPTION_OUTPUTS.jsonl",
        "selective_perception/WORKER_RESULT.json",
        "RUNTIME_DECISIONS.jsonl",
        "EXACT_INPUT_REUSE_RESULT.json",
        "R137_GATE.json",
        "RESOURCE_AUDIT.json",
        "SUMMARY.json",
    ]
    tracked.extend(
        str(path.relative_to(run_dir)) for path in sorted(output_dir.glob("*.npy"))
    )
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r137_manifest.v1",
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
        default=Path("configs/worldsim_v6/r137_exact_input_reuse_cross_frontend_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

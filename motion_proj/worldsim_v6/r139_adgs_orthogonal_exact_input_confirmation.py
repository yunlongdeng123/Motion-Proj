"""WorldSim V6 R139: exact-once orthogonal AD-GS exact-input reuse confirmation."""

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

from motion_proj.worldsim_v6.r3_experiment import (
    _bind_frozen_adgs_point_cloud,
    _materialize_inference_only_depth_placeholders,
)
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)
from motion_proj.worldsim_v6.r134_adgs_cross_frontend_threshold13 import _tree_sha256


TASK_ID = "WS-V6-R139-ADGS-ORTHOGONAL-EXACT-INPUT-CONFIRMATION-01"


class R139ExperimentError(RuntimeError):
    """The preregistered exact-once R139 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_perception(
    repo_root: Path,
    model_root: Path,
    index_path: Path,
    output_dir: Path,
    log_path: Path,
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    with log_path.open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            cwd=repo_root,
            env=worker_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    worker = json.loads((output_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(output_dir / "PERCEPTION_OUTPUTS.jsonl")
    return worker, rows


def _row_hashes(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], str]:
    return {
        (int(row["frame_index"]), row["variant"], int(row["repeat_index"])): row[
            "label_array_sha256"
        ]
        for row in rows
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R139ExperimentError("formal R139 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R139ExperimentError("R139 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r137_run = _resolve_runs_uri(sources["r137_run"])
    adgs_root = Path(sources["adgs_implementation_root"])
    adgs_model = Path(sources["adgs_model_root"])
    adgs_source = Path(sources["adgs_source_scene"])
    training_adapter = Path(sources["adgs_training_adapter"])
    model_root = Path(sources["semantic_model_root"])
    frozen_files: dict[Path, str] = {
        r137_run / "MANIFEST.json": sources["r137_manifest_sha256"],
        r137_run / "R137_GATE.json": sources["r137_gate_sha256"],
        r137_run / "SUMMARY.json": sources["r137_summary_sha256"],
        r137_run / "EXACT_INPUT_REUSE_RESULT.json": sources["r137_result_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
        training_adapter / "points3d.ply": sources["training_points3d_sha256"],
    }
    checkpoint_files: dict[Path, str] = {
        Path(record["path"]): record["sha256"]
        for record in sources["adgs_checkpoint_files"].values()
    }
    for path, expected in {**frozen_files, **checkpoint_files}.items():
        _verify(path, expected)
    if _git(adgs_root, "rev-parse", "HEAD") != sources["adgs_implementation_commit"]:
        raise R139ExperimentError("R139 AD-GS implementation commit drift")
    if not adgs_source.is_dir() or not training_adapter.is_dir():
        raise R139ExperimentError("R139 AD-GS adapter source preflight failed")
    r137_gate = json.loads((r137_run / "R137_GATE.json").read_text(encoding="utf-8"))
    r137_result = json.loads(
        (r137_run / "EXACT_INPUT_REUSE_RESULT.json").read_text(encoding="utf-8")
    )
    if not r137_gate["checks"]["passed"]:
        raise R139ExperimentError("R137 development authority is not accepted")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R139ExperimentError("R139 disk resource insufficient")

    task_root = run_root / TASK_ID
    if task_root.exists() and any(task_root.iterdir()):
        raise R139ExperimentError("R139 exact-once confirmation attempt already exists")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = task_root / f"{now}__adgs-orthogonal-exact-input-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema_version": "worldsim_v6.r139_confirmation_attempt.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "candidate_r137_gate_sha256": sources["r137_gate_sha256"],
        "policy": "reuse_cached_logged_perception_iff_rgb_byte_identical_else_execute",
        "translation_world_m": runtime["translation_world_m"],
        "attempt_index": 1,
        "attempt_limit": 1,
        "created_before_new_condition_heldout_quality_read": True,
        "confirmation_consumed": True,
    }
    _write_json(run_dir / "ATTEMPT.json", attempt)
    attempt_sha256 = _sha256(run_dir / "ATTEMPT.json")

    adapter = run_dir / "heldout_adapter"
    prepare_command = [
        sources["adgs_python"],
        str(repo_root / "scripts/prepare_worldsim_v4_adgs.py"),
        "--source",
        str(adgs_source),
        "--destination",
        str(adapter),
        "--partitions",
        "train",
        "heldout",
    ]
    with (run_dir / "adapter.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            prepare_command,
            cwd=repo_root,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=float(resources["maximum_adapter_seconds"]),
        )
    _materialize_inference_only_depth_placeholders(adapter)
    _bind_frozen_adgs_point_cloud(adapter, training_adapter)
    adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text(encoding="utf-8"))
    partition = json.loads((adapter / "partition.json").read_text(encoding="utf-8"))
    frames = sorted(
        {
            int(row["timestep"])
            for row in partition["rows"]
            if int(row["camera"]) == int(runtime["camera_id"])
            and row["partition"] == "heldout"
        }
    )
    if len(frames) != int(runtime["expected_heldout_frames"]):
        raise R139ExperimentError("R139 heldout frame denominator drift after attempt consumption")
    adapter_tree_sha256 = _tree_sha256(adapter)
    _write_json(
        run_dir / "HELDOUT_ADAPTER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r139_heldout_adapter_audit.v1",
            "tree_sha256": adapter_tree_sha256,
            "included_partitions": adapter_manifest["included_partitions"],
            "partition_image_counts": adapter_manifest["partition_image_counts"],
            "heldout_camera0_frames": frames,
            "heldout_frame_count": len(frames),
            "attempt_sha256": attempt_sha256,
        },
    )

    sensor_dir = run_dir / "sensor_worker"
    sensor_command = [
        sources["adgs_python"],
        str(repo_root / "scripts/worldsim_v6/r134_adgs_actor_translation_sensor_worker.py"),
        "--source-root",
        str(adgs_root),
        "--model-root",
        str(adgs_model),
        "--adapter",
        str(adapter),
        "--output",
        str(sensor_dir),
        "--frames",
        ",".join(str(frame) for frame in frames),
        "--translation-world="
        + ",".join(str(value) for value in runtime["translation_world_m"]),
        "--test-partition-name",
        "heldout",
        "--confirmation-content-read",
    ]
    with (run_dir / "sensor_worker.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            sensor_command,
            cwd=repo_root,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=float(resources["maximum_sensor_worker_seconds"]),
        )
    sensor_rows = _load_jsonl(sensor_dir / "FRAME_METRICS.jsonl")
    sensor_audit = json.loads((sensor_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
    if [int(row["frame_index"]) for row in sensor_rows] != frames:
        raise R139ExperimentError("R139 heldout sensor row order drift")

    decisions: list[dict[str, Any]] = []
    full_index: list[dict[str, Any]] = []
    selective_index: list[dict[str, Any]] = []
    for row in sensor_rows:
        frame = int(row["frame_index"])
        sensor_path = sensor_dir / row["sensor_path"]
        with np.load(sensor_path, allow_pickle=False) as arrays:
            logged = arrays["logged_rgb"]
            compiled = arrays["compiled_rgb"]
            identical = bool(np.array_equal(logged, compiled))
            changed_pixels = int(np.any(logged != compiled, axis=2).sum())
        if changed_pixels != int(row["edited_vs_logged_rgb_changed_pixels"]):
            raise R139ExperimentError(f"R139 sensor feature drift at frame {frame}")
        decisions.append(
            {
                "frame_index": frame,
                "logged_compiled_rgb_byte_identical": identical,
                "changed_rgb_pixels": changed_pixels,
                "decision": "reuse_cached_logged_perception" if identical else "execute_perception",
            }
        )
        for variant, rgb_key in (("logged", "logged_rgb"), ("edited", "compiled_rgb")):
            for repeat in range(int(evaluation["repeat_count"])):
                item = {
                    "case_id": f"heldout_frame{frame:03d}_{variant}",
                    "frame_index": frame,
                    "variant": variant,
                    "rgb_key": rgb_key,
                    "repeat_index": repeat,
                    "render_path": str(sensor_path),
                }
                full_index.append(item)
                if not identical:
                    selective_index.append(item)
    _write_jsonl(run_dir / "FULL_PERCEPTION_INPUT_INDEX.jsonl", full_index)
    _write_jsonl(run_dir / "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl", selective_index)
    _write_jsonl(run_dir / "RUNTIME_DECISIONS.jsonl", decisions)

    full_dir = run_dir / "full_perception"
    full_worker, full_rows = _run_perception(
        repo_root,
        model_root,
        run_dir / "FULL_PERCEPTION_INPUT_INDEX.jsonl",
        full_dir,
        run_dir / "full_perception.log",
        float(resources["maximum_perception_worker_seconds"]),
    )
    selective_dir = run_dir / "selective_perception"
    selective_worker, selective_rows = _run_perception(
        repo_root,
        model_root,
        run_dir / "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl",
        selective_dir,
        run_dir / "selective_perception.log",
        float(resources["maximum_perception_worker_seconds"]),
    )
    full_hashes = _row_hashes(full_rows)
    selective_hashes = _row_hashes(selective_rows)
    repeat_count = int(evaluation["repeat_count"])
    full_repeat_exact = all(
        len({full_hashes[(frame, variant, repeat)] for repeat in range(repeat_count)}) == 1
        for frame in frames
        for variant in ("logged", "edited")
    )
    selective_repeat_exact = all(
        len({selective_hashes[(frame, variant, repeat)] for repeat in range(repeat_count)}) == 1
        for frame in frames
        if any(d["frame_index"] == frame and d["decision"] == "execute_perception" for d in decisions)
        for variant in ("logged", "edited")
    )
    reconstruction_errors: list[dict[str, Any]] = []
    false_reuse_frames: list[int] = []
    for decision in decisions:
        frame = int(decision["frame_index"])
        reuse = decision["decision"] == "reuse_cached_logged_perception"
        if reuse:
            cached_hash = full_hashes[(frame, "logged", 0)]
            if not decision["logged_compiled_rgb_byte_identical"] or len(
                {
                    full_hashes[(frame, variant, repeat)]
                    for variant in ("logged", "edited")
                    for repeat in range(repeat_count)
                }
            ) != 1:
                false_reuse_frames.append(frame)
        for variant in ("logged", "edited"):
            for repeat in range(repeat_count):
                key = (frame, variant, repeat)
                reconstructed = cached_hash if reuse else selective_hashes[key]
                if reconstructed != full_hashes[key]:
                    reconstruction_errors.append(
                        {
                            "frame_index": frame,
                            "variant": variant,
                            "repeat_index": repeat,
                            "reconstructed_sha256": reconstructed,
                            "reference_sha256": full_hashes[key],
                        }
                    )

    trigger_count = sum(row["decision"] == "execute_perception" for row in decisions)
    skip_count = len(decisions) - trigger_count
    full_invocations = len(full_index)
    selective_invocations = len(selective_index)
    avoided_invocations = full_invocations - selective_invocations
    reduction_fraction = float(avoided_invocations / full_invocations)
    elapsed_ratio = float(selective_worker["elapsed_seconds"] / full_worker["elapsed_seconds"])
    adapter_tree_after = _tree_sha256(adapter)
    checkpoint_expected = {
        name: record["sha256"] for name, record in sources["adgs_checkpoint_files"].items()
    }
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "exact_once_attempt_precedes_new_condition_heldout_read": attempt["attempt_index"] == 1
        and attempt["attempt_limit"] == 1
        and attempt["created_before_new_condition_heldout_quality_read"]
        and attempt["confirmation_consumed"],
        "r137_exact_input_reuse_development_authority_accepted": bool(
            r137_gate["checks"]["passed"]
        )
        and r137_result["false_reuse_count"] == 0,
        "source_model_checkpoint_and_adapter_support_immutable": _git(
            adgs_root, "rev-parse", "HEAD"
        )
        == sources["adgs_implementation_commit"]
        and all(_sha256(path) == expected for path, expected in frozen_files.items())
        and all(_sha256(path) == expected for path, expected in checkpoint_files.items())
        and sensor_audit["checkpoint_sha256_before"]
        == sensor_audit["checkpoint_sha256_after"]
        == checkpoint_expected
        and adapter_tree_sha256 == adapter_tree_after,
        "adapter_contains_train_plus_heldout_only": adapter_manifest["included_partitions"]
        == runtime["adapter_included_partitions"]
        and adapter_manifest["partition_image_counts"] == runtime["expected_partition_image_counts"],
        "exact39_heldout_metric_denominator": len(frames)
        == len(sensor_rows)
        == len(decisions)
        == int(runtime["expected_heldout_frames"])
        and all(
            frame % int(runtime["partition_modulus"]) == int(runtime["heldout_remainder"])
            for frame in frames
        )
        and sensor_audit["partition_counts"] == {"train": 0, "heldout": 39},
        "orthogonal_edit_and_actor_state_restoration_exact": all(
            row["translation_world_m"] == runtime["translation_world_m"]
            and row["aggregate_actor_state_restored_exact"]
            for row in sensor_rows
        )
        and sensor_audit["all_actor_state_restored_exact"],
        "identity_guard_has_no_numeric_threshold": all(
            (row["decision"] == "reuse_cached_logged_perception")
            == row["logged_compiled_rgb_byte_identical"]
            == (row["changed_rgb_pixels"] == 0)
            for row in decisions
        ),
        "changed_and_identical_support_nontrivial": trigger_count
        >= int(evaluation["minimum_changed_input_frames"])
        and skip_count >= int(evaluation["minimum_identical_input_frames"]),
        "full156_reference_denominator_exact": full_invocations
        == len(full_rows)
        == int(runtime["expected_full_invocations"]),
        "selective_invocation_denominator_exact": selective_invocations
        == len(selective_rows)
        == trigger_count * 2 * repeat_count,
        "full_and_selective_outputs_repeat_exact": full_repeat_exact and selective_repeat_exact,
        "zero_false_reuse_and_all156_reconstructed_hashes_exact": not false_reuse_frames
        and not reconstruction_errors,
        "invocation_and_elapsed_reduction_gate": reduction_fraction
        >= float(evaluation["minimum_invocation_reduction_fraction"])
        and elapsed_ratio <= float(evaluation["maximum_elapsed_ratio_vs_full_reference"]),
        "gpu_within_budget": max(
            float(sensor_audit["peak_gpu_memory_mib"]),
            float(full_worker["peak_gpu_memory_mib"]),
            float(selective_worker["peak_gpu_memory_mib"]),
        )
        <= float(resources["maximum_peak_gpu_memory_mib"]),
        "workers_and_wall_within_budget": float(sensor_audit["wall_seconds"])
        <= float(resources["maximum_sensor_worker_seconds"])
        and float(full_worker["elapsed_seconds"])
        <= float(resources["maximum_perception_worker_seconds"])
        and float(selective_worker["elapsed_seconds"])
        <= float(resources["maximum_perception_worker_seconds"])
        and wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
        "training_not_started_and_confirmation_consumed": not sensor_audit["training_started"]
        and sensor_audit["confirmation_content_read"],
        "semantic_correctness_causality_contact_physics_planning_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    result = {
        "schema_version": "worldsim_v6.r139_adgs_orthogonal_exact_input_confirmation.v1",
        "policy": attempt["policy"],
        "translation_world_m": runtime["translation_world_m"],
        "frame_count": len(frames),
        "trigger_count": trigger_count,
        "skip_count": skip_count,
        "full_reference_invocations": full_invocations,
        "selective_invocations": selective_invocations,
        "avoided_invocations": avoided_invocations,
        "invocation_reduction_fraction": reduction_fraction,
        "full_worker_seconds": float(full_worker["elapsed_seconds"]),
        "selective_worker_seconds": float(selective_worker["elapsed_seconds"]),
        "elapsed_ratio_vs_full_reference": elapsed_ratio,
        "reconstruction_error_count": len(reconstruction_errors),
        "false_reuse_count": len(false_reuse_frames),
        "attempt_sha256": attempt_sha256,
        "confirmation_consumed": True,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "ORTHOGONAL_CONFIRMATION.json", result)
    _write_json(
        run_dir / "R139_GATE.json",
        {
            "schema_version": "worldsim_v6.r139_gate.v1",
            "checks": checks,
            "decision": "accept_adgs_orthogonal_exact_input_confirmation"
            if checks["passed"]
            else "reject_adgs_orthogonal_exact_input_candidate_consumed",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r139_resource_audit.v1",
            "sensor_worker_seconds": float(sensor_audit["wall_seconds"]),
            "full_worker_seconds": float(full_worker["elapsed_seconds"]),
            "selective_worker_seconds": float(selective_worker["elapsed_seconds"]),
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_mib": max(
                float(sensor_audit["peak_gpu_memory_mib"]),
                float(full_worker["peak_gpu_memory_mib"]),
                float(selective_worker["peak_gpu_memory_mib"]),
            ),
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_attempt_consumed": True,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r139_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_adgs_orthogonal_exact_input_confirmation"
        if checks["passed"]
        else "rejected_adgs_orthogonal_exact_input_candidate_consumed",
        "source_commit": source_commit,
        "attempt_sha256": attempt_sha256,
        "confirmation_consumed": True,
        "frame_count": len(frames),
        "trigger_count": trigger_count,
        "skip_count": skip_count,
        "selective_invocations": selective_invocations,
        "avoided_invocations": avoided_invocations,
        "invocation_reduction_fraction": reduction_fraction,
        "elapsed_ratio_vs_full_reference": elapsed_ratio,
        "reconstruction_error_count": len(reconstruction_errors),
        "false_reuse_count": len(false_reuse_frames),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "ATTEMPT.json",
        "HELDOUT_ADAPTER_AUDIT.json",
        "adapter.log",
        "heldout_adapter/adapter_manifest.json",
        "heldout_adapter/partition.json",
        "heldout_adapter/R3_ADGS_POINT_CLOUD_BINDING.json",
        "heldout_adapter/R3_INFERENCE_ONLY_DEPTH_PLACEHOLDERS.json",
        "sensor_worker.log",
        "sensor_worker/FRAME_METRICS.jsonl",
        "sensor_worker/WORKER_AUDIT.json",
        "FULL_PERCEPTION_INPUT_INDEX.jsonl",
        "SELECTIVE_PERCEPTION_INPUT_INDEX.jsonl",
        "RUNTIME_DECISIONS.jsonl",
        "full_perception.log",
        "full_perception/PERCEPTION_OUTPUTS.jsonl",
        "full_perception/WORKER_RESULT.json",
        "selective_perception.log",
        "selective_perception/PERCEPTION_OUTPUTS.jsonl",
        "selective_perception/WORKER_RESULT.json",
        "ORTHOGONAL_CONFIRMATION.json",
        "R139_GATE.json",
        "RESOURCE_AUDIT.json",
        "SUMMARY.json",
    ]
    tracked.extend(f"sensor_worker/{row['sensor_path']}" for row in sensor_rows)
    tracked.extend(str(path.relative_to(run_dir)) for path in sorted(full_dir.glob("*.npy")))
    tracked.extend(str(path.relative_to(run_dir)) for path in sorted(selective_dir.glob("*.npy")))
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r139_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
            "heldout_adapter_tree_sha256": adapter_tree_sha256,
        },
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {
            "schema_version": "worldsim_v6.terminal.v1",
            "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            "confirmation_consumed": True,
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
            "configs/worldsim_v6/r139_adgs_orthogonal_exact_input_confirmation_v1.yaml"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

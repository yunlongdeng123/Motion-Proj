"""WorldSim V6 R136: exact-once heldout confirmation of the AD-GS threshold-1 route."""

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


TASK_ID = "WS-V6-R136-ADGS-HELDOUT-POLICY-CONFIRMATION-01"


class R136ExperimentError(RuntimeError):
    """The preregistered exact-once R136 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
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
        "trigger_count": int(predicted.sum()),
        "skip_count": int((~predicted).sum()),
        "skip_fraction": float((~predicted).mean()),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R136ExperimentError("formal R136 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R136ExperimentError("R136 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r134_run = _resolve_runs_uri(sources["r134_run"])
    r135_run = _resolve_runs_uri(sources["r135_run"])
    r135_package = r135_run / "package_a"
    adgs_root = Path(sources["adgs_implementation_root"])
    adgs_model = Path(sources["adgs_model_root"])
    adgs_source = Path(sources["adgs_source_scene"])
    training_adapter = Path(sources["adgs_training_adapter"])
    model_root = Path(sources["semantic_model_root"])
    frozen_files: dict[Path, str] = {
        r134_run / "MANIFEST.json": sources["r134_manifest_sha256"],
        r134_run / "R134_GATE.json": sources["r134_gate_sha256"],
        r134_run / "SUMMARY.json": sources["r134_summary_sha256"],
        r135_run / "MANIFEST.json": sources["r135_manifest_sha256"],
        r135_run / "R135_GATE.json": sources["r135_gate_sha256"],
        r135_run / "SUMMARY.json": sources["r135_summary_sha256"],
        r135_package / "PACKAGE_MANIFEST.json": sources["r135_package_manifest_sha256"],
        r135_package / "POLICY.json": sources["r135_policy_sha256"],
        r135_package / "DEVELOPMENT_CERTIFICATE.json": sources[
            "r135_development_certificate_sha256"
        ],
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
        raise R136ExperimentError("AD-GS implementation commit drift")
    if not adgs_source.is_dir() or not training_adapter.is_dir():
        raise R136ExperimentError("AD-GS heldout adapter source preflight failed")
    r134_gate = json.loads((r134_run / "R134_GATE.json").read_text(encoding="utf-8"))
    r135_gate = json.loads((r135_run / "R135_GATE.json").read_text(encoding="utf-8"))
    policy = json.loads((r135_package / "POLICY.json").read_text(encoding="utf-8"))
    threshold = int(policy["routes"]["ad_gs"]["threshold_pixels"])
    if threshold != int(evaluation["frozen_adgs_threshold_pixels"]):
        raise R136ExperimentError("R135 AD-GS route threshold drift")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R136ExperimentError("R136 disk resource insufficient")

    task_root = run_root / TASK_ID
    if task_root.exists() and any(task_root.iterdir()):
        raise R136ExperimentError("R136 exact-once confirmation attempt already exists")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = task_root / f"{now}__adgs-heldout-confirmation-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema_version": "worldsim_v6.r136_confirmation_attempt.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "candidate_policy_sha256": sources["r135_policy_sha256"],
        "candidate_package_manifest_sha256": sources["r135_package_manifest_sha256"],
        "attempt_index": 1,
        "attempt_limit": 1,
        "created_before_heldout_quality_read": True,
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
        raise R136ExperimentError("R136 heldout frame denominator drift after attempt consumption")
    adapter_tree_sha256 = _tree_sha256(adapter)
    _write_json(
        run_dir / "HELDOUT_ADAPTER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r136_heldout_adapter_audit.v1",
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
        "--translation-world",
        ",".join(str(value) for value in runtime["translation_world_m"]),
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
        raise R136ExperimentError("R136 heldout sensor row order drift")

    index_rows = [
        {
            "case_id": f"heldout_frame{frame:03d}_{variant}",
            "frame_index": frame,
            "variant": variant,
            "rgb_key": rgb_key,
            "repeat_index": repeat,
            "render_path": str(sensor_dir / f"sensors/frame{frame:03d}.npz"),
        }
        for frame in frames
        for variant, rgb_key in (("logged", "logged_rgb"), ("edited", "compiled_rgb"))
        for repeat in range(int(evaluation["repeat_count"]))
    ]
    index_path = run_dir / "PERCEPTION_INPUT_INDEX.jsonl"
    _write_jsonl(index_path, index_rows)
    perception_dir = run_dir / "perception"
    worker_env = os.environ.copy()
    worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    perception_command = [
        sys.executable,
        str(repo_root / "scripts/worldsim_v6/r87_full_episode_perception_worker.py"),
        "--index",
        str(index_path),
        "--model-root",
        str(model_root),
        "--output-dir",
        str(perception_dir),
    ]
    with (run_dir / "perception.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            perception_command,
            cwd=repo_root,
            env=worker_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=float(resources["maximum_perception_worker_seconds"]),
        )
    perception_worker = json.loads(
        (perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8")
    )
    perception_rows = _load_jsonl(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in perception_rows:
        grouped.setdefault(row["case_id"], []).append(row)
    repeat_exact = all(
        len(items) == int(evaluation["repeat_count"])
        and len({item["label_array_sha256"] for item in items}) == 1
        for items in grouped.values()
    )
    changed_label_pixels: list[int] = []
    for frame in frames:
        arrays: dict[str, np.ndarray] = {}
        for variant in ("logged", "edited"):
            first = sorted(
                grouped[f"heldout_frame{frame:03d}_{variant}"],
                key=lambda row: row["repeat_index"],
            )[0]
            arrays[variant] = np.load(perception_dir / first["label_path"], allow_pickle=False)
        changed_label_pixels.append(int((arrays["edited"] != arrays["logged"]).sum()))
    features = np.asarray(
        [int(row["edited_vs_logged_rgb_changed_pixels"]) for row in sensor_rows], dtype=np.int64
    )
    label_counts = np.asarray(changed_label_pixels, dtype=np.int64)
    predicted = features >= threshold
    target = label_counts >= int(evaluation["minimum_changed_label_pixels"])
    metrics = _metrics(predicted, target)
    positives = features[target]
    negatives = features[~target]
    result = {
        "schema_version": "worldsim_v6.r136_adgs_heldout_confirmation.v1",
        "policy_id": policy["policy_id"],
        "frontend": "ad_gs",
        "threshold_pixels": threshold,
        "calibration_frames_in_heldout": 0,
        "heldout_frames": frames,
        "frame_count": len(frames),
        "positive_target_frame_count": int(target.sum()),
        "negative_target_frame_count": int((~target).sum()),
        "metrics": metrics,
        "maximum_negative_feature": int(negatives.max()) if negatives.size else None,
        "minimum_positive_feature": int(positives.min()) if positives.size else None,
        "sensor_changed_pixels_by_frame": dict(zip(map(str, frames), features.astype(int).tolist())),
        "changed_label_pixels_by_frame": dict(zip(map(str, frames), label_counts.astype(int).tolist())),
        "attempt_sha256": attempt_sha256,
        "confirmation_consumed": True,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "HELDOUT_CONFIRMATION.json", result)

    adapter_tree_after = _tree_sha256(adapter)
    checkpoint_expected = {
        name: record["sha256"] for name, record in sources["adgs_checkpoint_files"].items()
    }
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "exact_once_attempt_precedes_heldout_read": attempt["attempt_index"] == 1
        and attempt["attempt_limit"] == 1
        and attempt["created_before_heldout_quality_read"]
        and attempt["confirmation_consumed"],
        "r134_rejected_and_r135_policy_accepted": not r134_gate["checks"]["passed"]
        and r135_gate["checks"]["passed"],
        "frozen_policy_routes_adgs_to_threshold1": policy["routes"]["ad_gs"][
            "threshold_pixels"
        ]
        == 1
        and policy["routes"]["ad_gs"]["heldout_status"] == "PENDING_EXACT_ONCE",
        "source_model_checkpoint_and_adapter_support_immutable": _git(
            adgs_root, "rev-parse", "HEAD"
        )
        == sources["adgs_implementation_commit"]
        and all(_sha256(path) == expected for path, expected in frozen_files.items())
        and all(_sha256(path) == expected for path, expected in checkpoint_files.items())
        and sensor_audit["checkpoint_sha256_before"]
        == sensor_audit["checkpoint_sha256_after"]
        == checkpoint_expected
        and _sha256(training_adapter / "points3d.ply") == sources["training_points3d_sha256"]
        and adapter_tree_sha256 == adapter_tree_after,
        "adapter_contains_train_plus_heldout_only": adapter_manifest["included_partitions"]
        == runtime["adapter_included_partitions"]
        and adapter_manifest["partition_image_counts"] == runtime["expected_partition_image_counts"],
        "exact39_heldout_metric_denominator": len(frames) == len(sensor_rows)
        == int(runtime["expected_heldout_frames"])
        and all(frame % int(runtime["partition_modulus"]) == int(runtime["heldout_remainder"]) for frame in frames)
        and sensor_audit["partition_counts"] == {"train": 0, "heldout": 39}
        and all(row["adapter_partition"] == "heldout" for row in sensor_rows),
        "aggregate_actor_edit_state_restored_all_heldout_frames": bool(
            sensor_audit["all_actor_state_restored_exact"]
        )
        and all(row["aggregate_actor_state_restored_exact"] for row in sensor_rows),
        "full156_perception_denominator_exact": len(index_rows) == len(perception_rows)
        == int(runtime["expected_heldout_frames"]) * 2 * int(evaluation["repeat_count"]),
        "perception_repeat_exact_every_heldout_frame_variant": repeat_exact,
        "heldout_positive_negative_support_nontrivial": int(target.sum())
        >= int(evaluation["minimum_positive_target_frames"])
        and int((~target).sum()) >= int(evaluation["minimum_negative_target_frames"]),
        "zero_heldout_calibration": result["calibration_frames_in_heldout"] == 0,
        "threshold1_zero_false_positive_and_false_negative": metrics["false_positive"] == 0
        and metrics["false_negative"] == 0,
        "threshold1_precision_recall_f1_one": metrics["precision"] == 1.0
        and metrics["recall"] == 1.0
        and metrics["f1"] == 1.0,
        "heldout_skip_fraction_gate": metrics["skip_fraction"]
        >= float(evaluation["minimum_skip_fraction"]),
        "gpu_within_budget": max(
            float(sensor_audit["peak_gpu_memory_mib"]),
            float(perception_worker["peak_gpu_memory_mib"]),
        )
        <= float(resources["maximum_peak_gpu_memory_mib"]),
        "workers_and_wall_within_budget": float(sensor_audit["wall_seconds"])
        <= float(resources["maximum_sensor_worker_seconds"])
        and float(perception_worker["elapsed_seconds"])
        <= float(resources["maximum_perception_worker_seconds"])
        and wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
        "training_not_started_and_confirmation_consumed": not sensor_audit["training_started"]
        and sensor_audit["confirmation_content_read"],
        "identity_semantics_causality_contact_physics_planning_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R136_GATE.json",
        {
            "schema_version": "worldsim_v6.r136_gate.v1",
            "checks": checks,
            "decision": "accept_adgs_threshold1_exact_once_heldout_confirmation"
            if checks["passed"]
            else "reject_adgs_threshold1_confirmation_candidate_consumed",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r136_resource_audit.v1",
            "sensor_worker_seconds": float(sensor_audit["wall_seconds"]),
            "perception_worker_seconds": float(perception_worker["elapsed_seconds"]),
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_mib": max(
                float(sensor_audit["peak_gpu_memory_mib"]),
                float(perception_worker["peak_gpu_memory_mib"]),
            ),
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": True,
            "confirmation_attempt_consumed": True,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r136_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_adgs_threshold1_exact_once_heldout_confirmation"
        if checks["passed"]
        else "rejected_confirmation_candidate_consumed",
        "source_commit": source_commit,
        "attempt_sha256": attempt_sha256,
        "confirmation_attempt_index": 1,
        "confirmation_attempt_limit": 1,
        "confirmation_consumed": True,
        "frame_count": len(frames),
        "threshold_pixels": threshold,
        "positive_target_frame_count": int(target.sum()),
        "negative_target_frame_count": int((~target).sum()),
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "trigger_count": metrics["trigger_count"],
        "skip_count": metrics["skip_count"],
        "skip_fraction": metrics["skip_fraction"],
        "maximum_negative_feature": result["maximum_negative_feature"],
        "minimum_positive_feature": result["minimum_positive_feature"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "ATTEMPT.json",
        "HELDOUT_ADAPTER_AUDIT.json",
        "HELDOUT_CONFIRMATION.json",
        "R136_GATE.json",
        "RESOURCE_AUDIT.json",
        "SUMMARY.json",
        "adapter.log",
        "heldout_adapter/adapter_manifest.json",
        "heldout_adapter/partition.json",
        "heldout_adapter/R3_ADGS_POINT_CLOUD_BINDING.json",
        "heldout_adapter/R3_INFERENCE_ONLY_DEPTH_PLACEHOLDERS.json",
        "sensor_worker.log",
        "sensor_worker/FRAME_METRICS.jsonl",
        "sensor_worker/WORKER_AUDIT.json",
        "PERCEPTION_INPUT_INDEX.jsonl",
        "perception.log",
        "perception/PERCEPTION_OUTPUTS.jsonl",
        "perception/WORKER_RESULT.json",
    ]
    tracked.extend(f"sensor_worker/{row['sensor_path']}" for row in sensor_rows)
    tracked.extend(
        str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy"))
    )
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r136_manifest.v1",
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
        default=Path("configs/worldsim_v6/r136_adgs_heldout_policy_confirmation_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

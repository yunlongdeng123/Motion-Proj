"""WorldSim V6 R134: test threshold-13 transfer on full-episode AD-GS edits."""

from __future__ import annotations

import hashlib
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


TASK_ID = "WS-V6-R134-ADGS-CROSS-FRONTEND-THRESHOLD13-01"


class R134ExperimentError(RuntimeError):
    """The preregistered R134 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tree_sha256(root: Path) -> str:
    payload = "".join(
        f"{_sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        raise R134ExperimentError("formal R134 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R134ExperimentError("R134 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r132_run = _resolve_runs_uri(sources["r132_run"])
    r133_run = _resolve_runs_uri(sources["r133_run"])
    adgs_root = Path(sources["adgs_implementation_root"])
    adgs_model = Path(sources["adgs_model_root"])
    adapter = Path(sources["adgs_adapter_root"])
    model_root = Path(sources["semantic_model_root"])
    frozen_files: dict[Path, str] = {
        r132_run / "MANIFEST.json": sources["r132_manifest_sha256"],
        r132_run / "R132_GATE.json": sources["r132_gate_sha256"],
        r132_run / "SUMMARY.json": sources["r132_summary_sha256"],
        r132_run / "VALIDATION_CERTIFICATE.json": sources["r132_certificate_sha256"],
        r133_run / "MANIFEST.json": sources["r133_manifest_sha256"],
        r133_run / "R133_GATE.json": sources["r133_gate_sha256"],
        r133_run / "SUMMARY.json": sources["r133_summary_sha256"],
        r133_run / "SELECTIVE_EXECUTION_RESULT.json": sources[
            "r133_selective_execution_result_sha256"
        ],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
        adapter / "adapter_manifest.json": sources["adapter_manifest_sha256"],
        adapter / "R3_ADGS_POINT_CLOUD_BINDING.json": sources["adapter_binding_sha256"],
        adapter / "R3_INFERENCE_ONLY_DEPTH_PLACEHOLDERS.json": sources[
            "adapter_depth_placeholders_sha256"
        ],
        adapter / "points3d.ply": sources["adapter_points3d_sha256"],
    }
    checkpoint_files: dict[Path, str] = {
        Path(record["path"]): record["sha256"]
        for record in sources["adgs_checkpoint_files"].values()
    }
    for path, expected in {**frozen_files, **checkpoint_files}.items():
        _verify(path, expected)
    if _git(adgs_root, "rev-parse", "HEAD") != sources["adgs_implementation_commit"]:
        raise R134ExperimentError("AD-GS implementation commit drift")
    adapter_tree_before = _tree_sha256(adapter)
    if adapter_tree_before != sources["adgs_adapter_tree_sha256"]:
        raise R134ExperimentError("AD-GS adapter tree drift")
    r132_gate = json.loads((r132_run / "R132_GATE.json").read_text(encoding="utf-8"))
    r133_gate = json.loads((r133_run / "R133_GATE.json").read_text(encoding="utf-8"))
    certificate = json.loads(
        (r132_run / "VALIDATION_CERTIFICATE.json").read_text(encoding="utf-8")
    )
    threshold = int(certificate["threshold_pixels"])
    if threshold != int(evaluation["frozen_threshold_pixels"]):
        raise R134ExperimentError("frozen threshold drift")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R134ExperimentError("R134 disk resource insufficient")

    frames = list(range(int(runtime["frame_start"]), int(runtime["frame_stop_exclusive"])))
    if len(frames) != int(runtime["expected_frame_count"]):
        raise R134ExperimentError("R134 frame denominator drift")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__adgs-cross-frontend-threshold13-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    sensor_dir = run_dir / "sensor_worker"
    frames_text = ",".join(str(frame) for frame in frames)
    translation_text = ",".join(str(value) for value in runtime["translation_world_m"])
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
        frames_text,
        "--translation-world",
        translation_text,
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
        raise R134ExperimentError("AD-GS sensor row order drift")

    index_rows = [
        {
            "case_id": f"frame{frame:03d}_{variant}",
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
    worker_env = os.environ.copy()
    worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
                grouped[f"frame{frame:03d}_{variant}"], key=lambda row: row["repeat_index"]
            )[0]
            arrays[variant] = np.load(perception_dir / first["label_path"], allow_pickle=False)
        changed_label_pixels.append(int((arrays["edited"] != arrays["logged"]).sum()))
    features = np.asarray(
        [int(row["edited_vs_logged_rgb_changed_pixels"]) for row in sensor_rows],
        dtype=np.int64,
    )
    label_counts = np.asarray(changed_label_pixels, dtype=np.int64)
    predicted = features >= threshold
    target = label_counts >= int(evaluation["minimum_changed_label_pixels"])
    metrics = _metrics(predicted, target)
    positives = features[target]
    negatives = features[~target]
    transfer = {
        "schema_version": "worldsim_v6.r134_adgs_cross_frontend_transfer.v1",
        "source_policy_scope": certificate["validation_scope"],
        "source_frontend": "streetgs",
        "target_frontend": "ad_gs",
        "target_scene": runtime["scene"],
        "edit": "aggregate_dynamic_gaussians_translate_world_x_plus0p5m",
        "translation_world_m": runtime["translation_world_m"],
        "calibration_frames_in_target_frontend": 0,
        "frozen_threshold_pixels": threshold,
        "frame_count": len(frames),
        "positive_target_frame_count": int(target.sum()),
        "negative_target_frame_count": int((~target).sum()),
        "metrics": metrics,
        "maximum_negative_feature": int(negatives.max()) if negatives.size else None,
        "minimum_positive_feature": int(positives.min()) if positives.size else None,
        "sensor_changed_pixels_by_frame": dict(zip(map(str, frames), features.astype(int).tolist())),
        "changed_label_pixels_by_frame": dict(zip(map(str, frames), label_counts.astype(int).tolist())),
        "semantic_correctness": "ABSTAIN",
        "identity_binding": "ABSTAIN_AGGREGATE_DYNAMIC_SET",
        "contact_road_dynamics_physics_planning_safety": "ABSTAIN",
    }
    _write_json(run_dir / "CROSS_FRONTEND_TRANSFER.json", transfer)

    adapter_tree_after = _tree_sha256(adapter)
    sensor_output_bytes = sum(path.stat().st_size for path in sensor_dir.rglob("*") if path.is_file())
    perception_output_bytes = sum(
        path.stat().st_size for path in perception_dir.rglob("*") if path.is_file()
    )
    wall_seconds = time.monotonic() - started
    checkpoint_expected = {
        name: record["sha256"] for name, record in sources["adgs_checkpoint_files"].items()
    }
    checks = {
        "r132_and_r133_streetgs_policy_authorities_accepted": bool(
            r132_gate["checks"]["passed"] and r133_gate["checks"]["passed"]
        ),
        "adgs_implementation_checkpoint_adapter_and_model_immutable": _git(
            adgs_root, "rev-parse", "HEAD"
        )
        == sources["adgs_implementation_commit"]
        and sensor_audit["checkpoint_sha256_before"]
        == sensor_audit["checkpoint_sha256_after"]
        == checkpoint_expected
        and adapter_tree_before == adapter_tree_after == sources["adgs_adapter_tree_sha256"]
        and all(_sha256(path) == expected for path, expected in frozen_files.items())
        and all(_sha256(path) == expected for path, expected in checkpoint_files.items()),
        "full196_adgs_sensor_denominator_exact": len(sensor_rows)
        == int(runtime["expected_frame_count"]),
        "aggregate_actor_edit_state_restored_all_frames": bool(
            sensor_audit["all_actor_state_restored_exact"]
        )
        and all(row["aggregate_actor_state_restored_exact"] for row in sensor_rows),
        "world_x_plus0p5_translation_exact_all_frames": all(
            row["translation_world_m"] == runtime["translation_world_m"] for row in sensor_rows
        ),
        "full784_perception_denominator_exact": len(index_rows) == len(perception_rows)
        == int(runtime["expected_frame_count"]) * 2 * int(evaluation["repeat_count"]),
        "perception_repeat_exact_every_frame_and_variant": repeat_exact,
        "target_positive_and_negative_support_nontrivial": int(target.sum())
        >= int(evaluation["minimum_positive_target_frames"])
        and int((~target).sum()) >= int(evaluation["minimum_negative_target_frames"]),
        "threshold13_bound_without_adgs_calibration": threshold == 13
        and transfer["calibration_frames_in_target_frontend"] == 0,
        "cross_frontend_zero_false_positive_and_false_negative": metrics["false_positive"] == 0
        and metrics["false_negative"] == 0,
        "cross_frontend_precision_recall_f1_one": metrics["precision"] == 1.0
        and metrics["recall"] == 1.0
        and metrics["f1"] == 1.0,
        "cross_frontend_skip_fraction_gate": metrics["skip_fraction"]
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
        "outputs_within_budget": sensor_output_bytes + perception_output_bytes
        <= int(resources["maximum_output_bytes"]),
        "training_not_started": not sensor_audit["training_started"],
        "confirmation_not_read": not sensor_audit["confirmation_content_read"],
        "identity_semantics_causality_contact_physics_planning_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R134_GATE.json",
        {
            "schema_version": "worldsim_v6.r134_gate.v1",
            "checks": checks,
            "decision": "accept_adgs_cross_frontend_threshold13_transfer"
            if checks["passed"]
            else "reject_adgs_cross_frontend_threshold13_transfer",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r134_resource_audit.v1",
            "sensor_worker_seconds": float(sensor_audit["wall_seconds"]),
            "perception_worker_seconds": float(perception_worker["elapsed_seconds"]),
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_mib": max(
                float(sensor_audit["peak_gpu_memory_mib"]),
                float(perception_worker["peak_gpu_memory_mib"]),
            ),
            "sensor_output_bytes": sensor_output_bytes,
            "perception_output_bytes": perception_output_bytes,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r134_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_adgs_cross_frontend_threshold13_transfer"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "source_frontend": "streetgs",
        "target_frontend": "ad_gs",
        "target_scene": runtime["scene"],
        "edit": transfer["edit"],
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
        "maximum_negative_feature": transfer["maximum_negative_feature"],
        "minimum_positive_feature": transfer["minimum_positive_feature"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "CROSS_FRONTEND_TRANSFER.json",
        "R134_GATE.json",
        "RESOURCE_AUDIT.json",
        "SUMMARY.json",
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
            "schema_version": "worldsim_v6.r134_manifest.v1",
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
        default=Path("configs/worldsim_v6/r134_adgs_cross_frontend_threshold13_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the preregistered fresh three-scene F0j materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError
from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import _git_at
from scripts.run_worldsim_v51_f0b_three_view_association_parity import (
    _load_yaml,
    _verify,
    repository_source_identity,
)
from scripts.run_worldsim_v51_f0c_upstream_batch_association_repeatability import _nvidia_total_mib
from scripts.run_worldsim_v51_f0g_target_tensor_allocator_instrumentation import _run_trace_attempt
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMA = "worldsim_v51_stage_f_f0j_fresh_45_view_empty_cache_materialization_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _record_chain(records: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_config(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0j config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0j status or seed drift")

    auth = config["authorization"]["f0i_freeze"]
    freeze = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "F0i freeze", int(auth["bytes"])))
    if (
        freeze.get("status") != auth["required_status"]
        or freeze["interpretation"].get("failure") != auth["required_failure"]
        or freeze["governance"].get("next_phase") != auth["required_next_phase"]
    ):
        raise ProtocolError("F0j authorization drift")

    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if (
            _git_at(root, "rev-parse", "HEAD") != spec["commit"]
            or _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]
            or _git_at(root, "status", "--porcelain")
        ):
            raise ProtocolError(f"F0j source drift: {name}")
    traced = config["sources"]["traced_file"]
    _verify(Path(traced["path"]), traced["sha256"], "traced source", int(traced["bytes"]))
    for name, spec in config["assets"].items():
        _verify(Path(spec["path"]), spec["sha256"], name, int(spec["bytes"]))

    manifest_spec = config["input_manifest"]
    manifest_path = _verify(
        Path(manifest_spec["path"]), manifest_spec["sha256"], "train-only image manifest", int(manifest_spec["bytes"])
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("schema_version", "record_count", "total_bytes", "record_chain_sha256", "order", "image_pixels_decoded"):
        if manifest.get(key) != manifest_spec[key]:
            raise ProtocolError(f"F0j input manifest drift: {key}")
    records = [dict(row) for row in manifest.get("records", [])]
    if len(records) != 45 or sum(int(row["bytes"]) for row in records) != int(manifest_spec["total_bytes"]):
        raise ProtocolError("F0j input manifest denominator drift")
    for row in records:
        _verify(Path(row["path"]), row["sha256"], row["path"], int(row["bytes"]))
        row["staging_filename"] = Path(row["path"]).name

    expected_order = []
    for scene in config["scene_contracts"]:
        expected_order.extend(
            (scene["scene"], int(scene["scene_index"]), int(frame), int(camera))
            for frame in scene["frames"]
            for camera in scene["cameras"]
        )
    observed_order = [
        (row["scene"], int(row["scene_index"]), int(row["frame"]), int(row["camera"])) for row in records
    ]
    if observed_order != expected_order or len({(row["scene"], row["staging_filename"]) for row in records}) != 45:
        raise ProtocolError("F0j scene-local denominator/order drift")

    attempts = config["execution"]["attempts"]
    expected_groups = [scene["scene"] for scene in config["scene_contracts"]]
    if [row["input_group"] for row in attempts] != expected_groups:
        raise ProtocolError("F0j attempt order drift")
    if any(
        int(row["sam_num_points_per_side"]) != 32 or int(row["sam_num_points_per_batch"]) != 64
        for row in attempts
    ):
        raise ProtocolError("F0j method drift")
    if config["execution"].get("pre_matmul_empty_cache") is not True:
        raise ProtocolError("F0j recovery intervention drift")
    decision = config["decision"]
    if (
        decision.get("full_materialization_execution_authorized") is not True
        or decision.get("quality_read") is not False
        or decision.get("actor_identity_alignment_read") is not False
        or decision.get("identity_training_authorized") is not False
    ):
        raise ProtocolError("F0j decision lock drift")
    if config["environment"].get("CUDA_LAUNCH_BLOCKING") != "1":
        raise ProtocolError("F0j launch blocking drift")
    return config, records


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    config, records = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})

    input_groups: dict[str, dict[str, Any]] = {}
    input_dirs: dict[str, Path] = {}
    for scene in config["scene_contracts"]:
        scene_name = scene["scene"]
        scene_records = [row for row in records if row["scene"] == scene_name]
        input_groups[scene_name] = {"inputs": scene_records}
        input_dir = run_dir / "artifacts/inputs" / scene_name
        input_dir.mkdir(parents=True)
        for row in scene_records:
            (input_dir / row["staging_filename"]).symlink_to(Path(row["path"]))
        input_dirs[scene_name] = input_dir
    runtime_config = dict(config)
    runtime_config["input_groups"] = input_groups

    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    started = time.perf_counter()
    monitor.start()
    try:
        total = _nvidia_total_mib()
        start = _nvidia_used_mib()
        if total != int(config["resources"]["required_nvidia_total_mib"]):
            raise ProtocolError("F0j GPU total drift")
        if start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
            raise ProtocolError("F0j GPU start-use drift")

        attempts = []
        total_empty_cache_calls = 0
        materialized_records = []
        metadata_records = []
        for attempt_spec in config["execution"]["attempts"]:
            attempt = _run_trace_attempt(
                runtime_config, attempt_spec, input_dirs[attempt_spec["input_group"]], run_dir
            )
            if attempt["classification"] != "success" or len(attempt["masks"]) != 15:
                raise ProtocolError(f"F0j scene execution failed: {attempt_spec['input_group']}")
            pre_events = [row for row in attempt["trace"]["payload"]["events"] if row.get("event") == "pre_matmul"]
            if not pre_events or not all(
                "empty_cache" in row
                and int(row["empty_cache"]["after"]["free_bytes"])
                >= int(row["empty_cache"]["before"]["free_bytes"])
                for row in pre_events
            ):
                raise ProtocolError(f"F0j intervention evidence drift: {attempt_spec['input_group']}")
            total_empty_cache_calls += len(pre_events)
            attempts.append(attempt)
            for mask in attempt["masks"]:
                materialized_records.append({"scene": attempt_spec["input_group"], **mask})
            metadata_records.append({"scene": attempt_spec["input_group"], **attempt["metadata"]})

        if len(materialized_records) != 45 or len(metadata_records) != 3:
            raise ProtocolError("F0j output denominator drift")
        materialization_manifest = {
            "schema_version": "worldsim_v51_f0j_empty_cache_materialization_manifest_v1",
            "task_id": TASK_ID,
            "split": "train_only",
            "record_count": 45,
            "metadata_count": 3,
            "order": "scene_then_frame_then_camera",
            "input_record_chain_sha256": config["input_manifest"]["record_chain_sha256"],
            "output_record_chain_sha256": _record_chain([*materialized_records, *metadata_records]),
            "quality_read": False,
            "actor_identity_alignment_read": False,
            "records": materialized_records,
            "metadata": metadata_records,
        }
        _write_json(run_dir / "artifacts/materialization_manifest.json", materialization_manifest)

        for name in ("gaussian_grouping", "grounded_segment_anything"):
            if _git_at(Path(config["sources"][name]["path"]), "status", "--porcelain"):
                raise ProtocolError(f"F0j source mutated: {name}")
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0j resource monitor produced no valid sample")
        resources = {
            "nvidia_total_mib": total,
            "nvidia_start_mib": start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid),
            "cgroup_memory_peak_bytes": max(int(row["cgroup_memory_current_bytes"]) for row in valid),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid),
            "wall_seconds": time.perf_counter() - started,
            "disk_free_after_bytes": shutil.disk_usage(run_dir).free,
        }
        resources["nvidia_headroom_mib"] = total - resources["nvidia_peak_mib"]
        _write_json(run_dir / "artifacts/resources.json", resources)
        limits = config["resources"]
        checks = {
            "nvidia_total": resources["nvidia_total_mib"] == int(limits["required_nvidia_total_mib"]),
            "nvidia_peak": resources["nvidia_peak_mib"] <= int(limits["maximum_nvidia_peak_mib"]),
            "nvidia_headroom": resources["nvidia_headroom_mib"] >= int(limits["required_nvidia_headroom_mib"]),
            "cgroup": resources["cgroup_memory_peak_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]),
            "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(checks.values()):
            raise ProtocolError(f"F0j resource gate failed: {checks}")

        summary = {
            "schema_version": "worldsim_v51_f0j_materialization_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": config["decision"]["expected_conclusion"],
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "attempts": attempts,
            "materialization_manifest": materialization_manifest,
            "empty_cache_call_count": total_empty_cache_calls,
            "resources": resources,
            "resource_checks": checks,
            "input_image_pixels_decoded_count": 45,
            "output_schema_reads_count": 45,
            "full_materialization_execution": True,
            "quality_read": False,
            "actor_identity_alignment_read": False,
            "identity_training_authorized": False,
            "next_action": config["decision"]["next_action"],
            "validation_quality_read": False,
            "test_quality_read": False,
            "kitti_method_tuning": False,
            "f1_execution": False,
            "f2_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(run_dir / "manifest.json", {"task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
        _write_json(
            run_dir / "status.json",
            {
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": summary["conclusion"],
                "source_commit": identity["commit"],
            },
        )
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append({"event": "run_blocked", "at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {"task_id": TASK_ID, "status": "blocked", "error": f"{type(error).__name__}: {error}", "source_commit": identity["commit"]},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0j_fresh_45_view_empty_cache_materialization_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

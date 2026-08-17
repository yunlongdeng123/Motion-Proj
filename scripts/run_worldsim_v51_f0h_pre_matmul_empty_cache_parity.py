#!/usr/bin/env python3
"""执行 F0h pre-matmul empty-cache execution recovery parity。"""

from __future__ import annotations

import argparse
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
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
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


SCHEMA = "worldsim_v51_stage_f_f0h_pre_matmul_empty_cache_parity_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
ATTEMPT_NAMES = ["control_cache_1", "target_cache_1", "control_cache_2", "target_cache_2"]


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0h config identity drift")
    auth = config["authorization"]["f0g_freeze"]
    freeze_path = _verify(PROJECT / auth["path"], auth["sha256"], "F0g freeze", int(auth["bytes"]))
    freeze = _load_yaml(freeze_path)
    if (
        freeze.get("status") != auth["required_status"]
        or freeze["mechanism"].get("failure") != auth["required_failure"]
        or freeze["governance"].get("next_phase") != auth["required_next_phase"]
    ):
        raise ProtocolError("F0h authorization drift")
    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if _git_at(root, "rev-parse", "HEAD") != spec["commit"] or _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise ProtocolError(f"F0h source identity drift: {name}")
        if _git_at(root, "status", "--porcelain"):
            raise ProtocolError(f"F0h source dirty: {name}")
    for name, spec in config["assets"].items():
        _verify(Path(spec["path"]), spec["sha256"], name, int(spec["bytes"]))
    traced = config["sources"]["traced_file"]
    _verify(Path(traced["path"]), traced["sha256"], "traced source", int(traced["bytes"]))
    expected_orders = {
        "control": [("scene-0471", frame, 0) for frame in (0, 40, 80)],
        "target": [("scene-1087", 0, camera) for camera in (0, 1, 2)],
    }
    for group_name, group in config["input_groups"].items():
        observed = [(row["scene"], int(row["frame"]), int(row["camera"])) for row in group["inputs"]]
        if observed != expected_orders[group_name]:
            raise ProtocolError(f"F0h input order drift: {group_name}")
        for row in group["inputs"]:
            _verify(Path(row["path"]), row["sha256"], row["staging_filename"], int(row["bytes"]))
    attempts = config["execution"]["attempts"]
    if [row["name"] for row in attempts] != ATTEMPT_NAMES:
        raise ProtocolError("F0h attempt order drift")
    if [row["input_group"] for row in attempts] != ["control", "target"] * 2:
        raise ProtocolError("F0h ABAB binding drift")
    if any(int(row["sam_num_points_per_side"]) != 32 or int(row["sam_num_points_per_batch"]) != 64 for row in attempts):
        raise ProtocolError("F0h method drift")
    if config["execution"].get("pre_matmul_empty_cache") is not True:
        raise ProtocolError("F0h empty-cache intervention drift")
    recovery = config["execution"]["execution_recovery"]
    if any(recovery.get(key) is not False for key in ("tensor_content_change", "operator_change", "upstream_source_change")):
        raise ProtocolError("F0h recovery boundary drift")
    for key in ("full_materialization_authorized", "quality_read", "identity_training_authorized"):
        if config["decision"].get(key) is not False:
            raise ProtocolError(f"F0h decision lock drift: {key}")
    return config


def _trace_empty_cache_valid(attempt: Mapping[str, Any]) -> bool:
    trace = attempt["trace"]["payload"]
    if trace.get("pre_matmul_empty_cache") is not True:
        return False
    pre_events = [row for row in trace.get("events", []) if row.get("event") == "pre_matmul"]
    if len(pre_events) != 2:
        return False
    return all(
        "empty_cache" in row
        and int(row["empty_cache"]["after"]["free_bytes"])
        >= int(row["empty_cache"]["before"]["free_bytes"])
        for row in pre_events
    )


def _outcome(config: Mapping[str, Any], attempts: list[Mapping[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    classes = [row["classification"] for row in attempts]
    empty_cache_checks = [_trace_empty_cache_valid(row) for row in attempts]
    if classes != ["success"] * 4 or not all(empty_cache_checks):
        return (
            "recovery_failed",
            config["decision"]["failure_conclusion"],
            config["decision"]["next_action_on_failure"],
            {"classes": classes, "empty_cache_checks": empty_cache_checks},
        )
    reference_checks = []
    for attempt in attempts:
        reference = config["input_groups"][attempt["input_group"]]["reference"]
        reference_checks.append(
            [row["sha256"] for row in attempt["masks"]] == reference["mask_sha256"]
            and attempt["metadata"]["sha256"] == reference["metadata_sha256"]
        )
    pair_checks = {
        "control_pair_exact": [row["sha256"] for row in attempts[0]["masks"]]
        == [row["sha256"] for row in attempts[2]["masks"]]
        and attempts[0]["metadata"]["sha256"] == attempts[2]["metadata"]["sha256"],
        "target_pair_exact": [row["sha256"] for row in attempts[1]["masks"]]
        == [row["sha256"] for row in attempts[3]["masks"]]
        and attempts[1]["metadata"]["sha256"] == attempts[3]["metadata"]["sha256"],
    }
    checks = {
        "classes": classes,
        "empty_cache_checks": empty_cache_checks,
        "reference_checks": reference_checks,
        **pair_checks,
    }
    if not all(reference_checks) or not all(pair_checks.values()):
        return (
            "recovery_nonexact",
            config["decision"]["nonexact_conclusion"],
            config["decision"]["next_action_on_failure"],
            checks,
        )
    return (
        "recovery_pass",
        config["decision"]["pass_conclusion"],
        config["decision"]["next_action_on_pass"],
        checks,
    )


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    config = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"schema_version": "worldsim_v51_f0h_status_v1", "task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})
    inputs = {}
    for group_name, group in config["input_groups"].items():
        input_dir = run_dir / "artifacts/inputs" / group_name
        input_dir.mkdir(parents=True)
        for row in group["inputs"]:
            (input_dir / row["staging_filename"]).symlink_to(Path(row["path"]))
        inputs[group_name] = input_dir
    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    started = time.perf_counter()
    monitor.start()
    try:
        total = _nvidia_total_mib()
        if total != int(config["resources"]["required_nvidia_total_mib"]):
            raise ProtocolError("F0h GPU total drift")
        nvidia_start = _nvidia_used_mib()
        if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
            raise ProtocolError("F0h GPU start-use drift")
        attempts = [
            _run_trace_attempt(config, attempt, inputs[attempt["input_group"]], run_dir)
            for attempt in config["execution"]["attempts"]
        ]
        outcome, conclusion, next_action, parity_checks = _outcome(config, attempts)
        for name in ("gaussian_grouping", "grounded_segment_anything"):
            if _git_at(Path(config["sources"][name]["path"]), "status", "--porcelain"):
                raise ProtocolError(f"F0h source mutated: {name}")
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        resources = {
            "nvidia_total_mib": total,
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid),
            "cgroup_memory_peak_bytes": max(int(row["cgroup_memory_current_bytes"]) for row in valid),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid),
            "wall_seconds": time.perf_counter() - started,
            "disk_free_after_bytes": shutil.disk_usage(run_dir).free,
        }
        resources["nvidia_headroom_mib"] = total - resources["nvidia_peak_mib"]
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_total": total == int(ceilings["required_nvidia_total_mib"]),
            "nvidia_peak": resources["nvidia_peak_mib"] <= int(ceilings["maximum_nvidia_peak_mib"]),
            "nvidia_headroom": resources["nvidia_headroom_mib"] >= int(ceilings["required_nvidia_headroom_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"] <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "disk_free_after": resources["disk_free_after_bytes"] >= int(ceilings["minimum_disk_free_bytes_after"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        if not all(resource_checks.values()):
            raise ProtocolError(f"F0h resource gate failed: {resource_checks}")
        summary = {
            "schema_version": "worldsim_v51_f0h_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "outcome": outcome,
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "attempts": attempts,
            "parity_checks": parity_checks,
            "resources": resources,
            "resource_checks": resource_checks,
            "pre_matmul_empty_cache": True,
            "upstream_source_mutation": False,
            "operator_change": False,
            "tensor_content_change": False,
            "quality_read": False,
            "full_materialization": False,
            "identity_training_authorized": False,
            "next_action": next_action,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": "done"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v51_f0h_manifest_v1", "task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
        _write_json(run_dir / "status.json", {"schema_version": "worldsim_v51_f0h_status_v1", "task_id": TASK_ID, "status": "done", "outcome": outcome, "conclusion": conclusion, "source_commit": identity["commit"]})
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append({"event": "run_blocked", "at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(run_dir / "status.json", {"schema_version": "worldsim_v51_f0h_status_v1", "task_id": TASK_ID, "status": "blocked", "error": f"{type(error).__name__}: {error}", "source_commit": identity["commit"]})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

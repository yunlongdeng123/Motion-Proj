#!/usr/bin/env python3
"""执行 F0g source-neutral control/target tensor allocator instrumentation。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import _git_at
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _arm_command, _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_f0c_upstream_batch_association_repeatability import _nvidia_total_mib
from scripts.run_worldsim_v51_f0e_scene1087_cuda_fault_localization import _schema_record
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMA = "worldsim_v51_stage_f_f0g_target_tensor_allocator_instrumentation_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
ATTEMPT_NAMES = ["control_trace", "target_trace"]


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0g config identity drift")
    auth = config["authorization"]["f0f_freeze"]
    freeze_path = _verify(PROJECT / auth["path"], auth["sha256"], "F0f freeze", int(auth["bytes"]))
    freeze = _load_yaml(freeze_path)
    if (
        freeze.get("status") != auth["required_status"]
        or freeze["canonical_run"].get("outcome") != auth["required_outcome"]
        or freeze["interpretation"].get("failure") != auth["required_failure"]
        or freeze["governance"].get("next_phase") != auth["required_next_phase"]
    ):
        raise ProtocolError("F0g authorization drift")
    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if _git_at(root, "rev-parse", "HEAD") != spec["commit"] or _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise ProtocolError(f"F0g source identity drift: {name}")
        if _git_at(root, "status", "--porcelain"):
            raise ProtocolError(f"F0g source dirty: {name}")
    for name, spec in config["assets"].items():
        _verify(Path(spec["path"]), spec["sha256"], name, int(spec["bytes"]))
    traced = config["sources"]["traced_file"]
    _verify(Path(traced["path"]), traced["sha256"], "traced source", int(traced["bytes"]))
    if int(traced["pre_matmul_line"]) != 58 or int(traced["post_matmul_line"]) != 59:
        raise ProtocolError("F0g traced line contract drift")
    expected_orders = {
        "control": [("scene-0471", frame, 0) for frame in (0, 40, 80)],
        "target": [("scene-1087", 0, camera) for camera in (0, 1, 2)],
    }
    for group_name, group in config["input_groups"].items():
        observed = [(row["scene"], int(row["frame"]), int(row["camera"])) for row in group["inputs"]]
        if observed != expected_orders[group_name]:
            raise ProtocolError(f"F0g input order drift: {group_name}")
        for row in group["inputs"]:
            _verify(Path(row["path"]), row["sha256"], row["staging_filename"], int(row["bytes"]))
    attempts = config["execution"]["attempts"]
    if [row["name"] for row in attempts] != ATTEMPT_NAMES or [row["input_group"] for row in attempts] != ["control", "target"]:
        raise ProtocolError("F0g attempt order drift")
    if any(int(row["sam_num_points_per_side"]) != 32 or int(row["sam_num_points_per_batch"]) != 64 for row in attempts):
        raise ProtocolError("F0g method drift")
    contract = config["execution"]["source_neutral_contract"]
    if any(contract.get(key) is not False for key in ("upstream_files_modified", "operator_monkeypatch", "tensor_content_read")):
        raise ProtocolError("F0g source-neutral contract drift")
    for key in ("full_materialization_authorized", "quality_read", "actor_identity_alignment_read", "identity_training_authorized"):
        if config["decision"].get(key) is not False:
            raise ProtocolError(f"F0g decision lock drift: {key}")
    if config["environment"].get("CUDA_LAUNCH_BLOCKING") != "1":
        raise ProtocolError("F0g launch blocking drift")
    return config


def _trace_command(
    config: Mapping[str, Any], attempt: Mapping[str, Any], input_dir: Path, output_dir: Path, trace_path: Path
) -> list[str]:
    official = _arm_command(config, attempt, input_dir, output_dir)
    target_script = Path(config["sources"]["deva"]["path"]) / official[1]
    command = [
        official[0],
        str(PROJECT / config["execution"]["trace_launcher"]),
        "--trace-output",
        str(trace_path),
        "--target-script",
        str(target_script),
    ]
    if config["execution"].get("pre_matmul_empty_cache") is True:
        command.append("--pre-matmul-empty-cache")
    return [*command, "--", *official[2:]]


def _run_trace_attempt(
    config: Mapping[str, Any], attempt: Mapping[str, Any], input_dir: Path, run_dir: Path
) -> dict[str, Any]:
    name = attempt["name"]
    attempt_dir = run_dir / "artifacts/attempts" / name
    output_dir = attempt_dir / "output"
    output_dir.mkdir(parents=True)
    trace_path = attempt_dir / "trace.json"
    command = _trace_command(config, attempt, input_dir, output_dir, trace_path)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_HOME": config["environment"]["TORCH_HOME"],
            "PYTORCH_CUDA_ALLOC_CONF": config["environment"]["PYTORCH_CUDA_ALLOC_CONF"],
            "CUDA_LAUNCH_BLOCKING": config["environment"]["CUDA_LAUNCH_BLOCKING"],
        }
    )
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=Path(config["sources"]["deva"]["path"]),
            stdout=stdout,
            stderr=stderr,
            env=environment,
            check=False,
        )
    if not trace_path.is_file():
        raise ProtocolError(f"F0g trace missing: {name}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if (
        trace.get("tensor_content_read") is not False
        or trace.get("operator_monkeypatch") is not False
        or trace.get("trace_source", {}).get("sha256") != config["sources"]["traced_file"]["sha256"]
        or not any(row.get("event") == "pre_matmul" for row in trace.get("events", []))
    ):
        raise ProtocolError(f"F0g trace contract drift: {name}")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    mask_dir = output_dir / "Annotations"
    masks = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
    pred_path = output_dir / "pred.json"
    common = {
        "name": name,
        "input_group": attempt["input_group"],
        "command": command,
        "returncode": int(completed.returncode),
        "wall_seconds": time.perf_counter() - started,
        "stdout": {"bytes": stdout_path.stat().st_size, "sha256": sha256_file(stdout_path)},
        "stderr": {"bytes": stderr_path.stat().st_size, "sha256": sha256_file(stderr_path)},
        "trace": {"bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path), "payload": trace},
    }
    if completed.returncode == 0:
        expected_names = [
            f"{Path(row['staging_filename']).stem}.png"
            for row in config["input_groups"][attempt["input_group"]]["inputs"]
        ]
        if [path.name for path in masks] != expected_names or not pred_path.is_file():
            raise ProtocolError(f"F0g success output denominator drift: {name}")
        common.update(
            {
                "classification": "success",
                "masks": [_schema_record(path) for path in masks],
                "metadata": {"bytes": pred_path.stat().st_size, "sha256": sha256_file(pred_path)},
            }
        )
        return common
    if all(marker in stderr_text for marker in config["execution"]["expected_failure_markers"]):
        if masks or pred_path.exists():
            raise ProtocolError(f"F0g failure published partial output: {name}")
        common.update(
            {
                "classification": "expected_cublas_internal_failure",
                "explicit_pytorch_oom": "CUDA out of memory" in stderr_text,
                "mask_count": 0,
                "pred_json": False,
            }
        )
        return common
    raise ProtocolError(f"F0g unexpected subprocess failure: {name}")


def _outcome(attempts: list[Mapping[str, Any]]) -> tuple[str, str]:
    classes = [row["classification"] for row in attempts]
    if classes == ["success", "expected_cublas_internal_failure"]:
        return (
            "control_success_target_failure",
            "control_trace_succeeded_target_cublas_failure_tensor_allocator_snapshots_captured",
        )
    if classes == ["success", "success"]:
        return ("both_success", "control_and_target_trace_succeeded_tensor_allocator_snapshots_captured")
    if classes[0] == "expected_cublas_internal_failure":
        return ("control_failure", "control_trace_failed_instrumentation_boundary_not_interpretable")
    raise ProtocolError(f"F0g outcome drift: {classes}")


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
    _write_json(run_dir / "status.json", {"schema_version": "worldsim_v51_f0g_status_v1", "task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})
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
            raise ProtocolError("F0g GPU total drift")
        nvidia_start = _nvidia_used_mib()
        if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
            raise ProtocolError("F0g GPU start-use drift")
        attempts = [
            _run_trace_attempt(config, attempt, inputs[attempt["input_group"]], run_dir)
            for attempt in config["execution"]["attempts"]
        ]
        outcome, conclusion = _outcome(attempts)
        for name in ("gaussian_grouping", "grounded_segment_anything"):
            root = Path(config["sources"][name]["path"])
            if _git_at(root, "status", "--porcelain"):
                raise ProtocolError(f"F0g source mutated during trace: {name}")
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
            raise ProtocolError(f"F0g resource gate failed: {resource_checks}")
        summary = {
            "schema_version": "worldsim_v51_f0g_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "outcome": outcome,
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "attempts": attempts,
            "resources": resources,
            "resource_checks": resource_checks,
            "upstream_source_mutation": False,
            "operator_monkeypatch": False,
            "tensor_content_read": False,
            "quality_read": False,
            "full_materialization": False,
            "identity_training_authorized": False,
            "next_action": config["decision"]["next_action"],
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": "done"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v51_f0g_manifest_v1", "task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
        _write_json(run_dir / "status.json", {"schema_version": "worldsim_v51_f0g_status_v1", "task_id": TASK_ID, "status": "done", "outcome": outcome, "conclusion": conclusion, "source_commit": identity["commit"]})
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append({"event": "run_blocked", "at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(run_dir / "status.json", {"schema_version": "worldsim_v51_f0g_status_v1", "task_id": TASK_ID, "status": "blocked", "error": f"{type(error).__name__}: {error}", "source_commit": identity["commit"]})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""执行 F0f CUDA runtime control-target reproducibility gate。"""

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
from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import (
    _environment_import_report,
    _git_at,
    _solver_smokes,
)
from scripts.run_worldsim_v51_f0b_three_view_association_parity import (
    _load_yaml,
    _verify,
    repository_source_identity,
)
from scripts.run_worldsim_v51_f0c_upstream_batch_association_repeatability import _nvidia_total_mib
from scripts.run_worldsim_v51_f0e_scene1087_cuda_fault_localization import _run_attempt
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMA = "worldsim_v51_stage_f_f0f_cuda_runtime_health_reproducibility_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
GROUP_NAMES = [
    "control_scene0471_same_camera_temporal",
    "target_scene1087_same_frame_cross_camera",
]
ATTEMPT_NAMES = ["control_replay_1", "target_replay_1", "control_replay_2", "target_replay_2"]


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0f config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0f status or seed drift")
    auth = config["authorization"]["f0e_freeze"]
    freeze_path = _verify(
        PROJECT / auth["path"], auth["sha256"], "F0e freeze", int(auth["bytes"])
    )
    freeze = _load_yaml(freeze_path)
    if freeze.get("status") != auth["required_status"]:
        raise ProtocolError("F0f F0e freeze status drift")
    if freeze["canonical_run"].get("outcome") != auth["required_outcome"]:
        raise ProtocolError("F0f authorization outcome drift")
    if freeze["interpretation"].get("failure") != auth["required_failure"]:
        raise ProtocolError("F0f authorization failure drift")
    if freeze["governance"].get("next_phase") != auth["required_next_phase"]:
        raise ProtocolError("F0f authorization next phase drift")

    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if _git_at(root, "rev-parse", "HEAD") != spec["commit"]:
            raise ProtocolError(f"F0f source commit drift: {name}")
        if _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise ProtocolError(f"F0f source tree drift: {name}")
        if _git_at(root, "status", "--porcelain"):
            raise ProtocolError(f"F0f source checkout not clean: {name}")
    deva = Path(config["sources"]["deva"]["path"]).resolve()
    grouping = Path(config["sources"]["gaussian_grouping"]["path"]).resolve()
    if not deva.is_dir() or not deva.is_relative_to(grouping):
        raise ProtocolError("F0f DEVA source boundary drift")
    for name, spec in config["assets"].items():
        _verify(Path(spec["path"]), spec["sha256"], name, int(spec["bytes"]))

    provenance = config["input_provenance"]
    manifest_path = _verify(
        Path(provenance["path"]), provenance["sha256"], "train-only image manifest", int(provenance["bytes"])
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        int(manifest.get("record_count", -1)) != 45
        or manifest.get("record_chain_sha256") != provenance["record_chain_sha256"]
        or manifest.get("image_pixels_decoded") is not False
    ):
        raise ProtocolError("F0f input provenance drift")
    groups = config["input_groups"]
    if list(groups) != GROUP_NAMES:
        raise ProtocolError("F0f input group order drift")
    expected_orders = {
        GROUP_NAMES[0]: [("scene-0471", 382, frame, 0) for frame in (0, 40, 80)],
        GROUP_NAMES[1]: [("scene-1087", 827, 0, camera) for camera in (0, 1, 2)],
    }
    manifest_records = {
        (row["path"], row["sha256"], int(row["bytes"])) for row in manifest["records"]
    }
    for group_name, group in groups.items():
        inputs = group["inputs"]
        observed = [
            (row["scene"], int(row["scene_index"]), int(row["frame"]), int(row["camera"]))
            for row in inputs
        ]
        if observed != expected_orders[group_name]:
            raise ProtocolError(f"F0f input group order drift: {group_name}")
        for row in inputs:
            _verify(Path(row["path"]), row["sha256"], row["staging_filename"], int(row["bytes"]))
            if (row["path"], row["sha256"], int(row["bytes"])) not in manifest_records:
                raise ProtocolError(f"F0f input outside r026 manifest: {group_name}")

    attempts = config["execution"]["attempts"]
    if [row["name"] for row in attempts] != ATTEMPT_NAMES:
        raise ProtocolError("F0f attempt order drift")
    if [row["input_group"] for row in attempts] != [GROUP_NAMES[0], GROUP_NAMES[1]] * 2:
        raise ProtocolError("F0f ABAB order drift")
    if any(int(row["sam_num_points_per_side"]) != 32 for row in attempts):
        raise ProtocolError("F0f grid drift")
    if any(int(row["sam_num_points_per_batch"]) != 64 for row in attempts):
        raise ProtocolError("F0f upstream batch drift")
    if config["environment"].get("CUDA_LAUNCH_BLOCKING") != "1":
        raise ProtocolError("F0f CUDA launch blocking drift")
    decision = config["decision"]
    for key in (
        "full_materialization_authorized",
        "quality_read",
        "actor_identity_alignment_read",
        "identity_training_authorized",
    ):
        if decision.get(key) is not False:
            raise ProtocolError(f"F0f decision lock drift: {key}")
    resources = config["resources"]
    if (
        int(resources["required_nvidia_total_mib"]) != 24576
        or int(resources["required_nvidia_headroom_mib"]) != 256
        or int(resources["maximum_nvidia_peak_mib"]) != 24320
    ):
        raise ProtocolError("F0f resource contract drift")
    locks = config["locks"]
    for name, value in locks.items():
        if name in {"input_image_pixels_decoded_count", "output_schema_reads_maximum"}:
            continue
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"F0f milestone lock drift: {name}")
        elif value is not False:
            raise ProtocolError(f"F0f research lock drift: {name}")
    return config


def _pair_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["classification"] != "success" or right["classification"] != "success":
        return False
    return (
        [row["sha256"] for row in left["masks"]]
        == [row["sha256"] for row in right["masks"]]
        and left["metadata"]["sha256"] == right["metadata"]["sha256"]
    )


def _outcome(attempts: list[Mapping[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    classes = [row["classification"] for row in attempts]
    if any(name == "unexpected_failure" for name in classes):
        raise ProtocolError(f"F0f unexpected attempt outcome: {classes}")
    control = [attempts[0], attempts[2]]
    target = [attempts[1], attempts[3]]
    pair_checks = {
        "control_both_success": all(row["classification"] == "success" for row in control),
        "control_exact": _pair_exact(*control),
        "target_both_success": all(row["classification"] == "success" for row in target),
        "target_exact": _pair_exact(*target),
    }
    if not pair_checks["control_both_success"]:
        return (
            "control_failure",
            "known_good_control_exhibited_cublas_instability_runtime_not_healthy",
            "suspend_gaussian_grouping_gpu_route_and_record_runtime_failure",
            pair_checks,
        )
    if not pair_checks["control_exact"]:
        return (
            "success_nonexact",
            "successful_replays_not_bit_exact_identity_input_nonrepeatable",
            "close_faithful_identity_input_as_nonrepeatable",
            pair_checks,
        )
    if not pair_checks["target_both_success"]:
        return (
            "control_stable_target_failure",
            "control_two_replays_succeeded_exact_but_target_retained_cublas_instability",
            "preregister_source_neutral_target_tensor_allocator_instrumentation",
            pair_checks,
        )
    if not pair_checks["target_exact"]:
        return (
            "success_nonexact",
            "successful_replays_not_bit_exact_identity_input_nonrepeatable",
            "close_faithful_identity_input_as_nonrepeatable",
            pair_checks,
        )
    return (
        "all_success_exact",
        "control_and_target_two_replays_each_succeeded_exact_under_cuda_launch_blocking",
        "preregister_scene1087_15_view_blocking_recovery",
        pair_checks,
    )


def _capture_health(run_dir: Path) -> dict[str, Any]:
    health_dir = run_dir / "artifacts/health"
    health_dir.mkdir(parents=True)
    commands = {
        "identity": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        "ecc_page_row": ["nvidia-smi", "-q", "-d", "ECC,PAGE_RETIREMENT,ROW_REMAPPER"],
        "dmesg_error_warning": ["dmesg", "--level=err,warn"],
    }
    records = {}
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_path = health_dir / f"{name}.stdout.log"
        stderr_path = health_dir / f"{name}.stderr.log"
        _write_text(stdout_path, completed.stdout)
        _write_text(stderr_path, completed.stderr)
        records[name] = {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": {"bytes": stdout_path.stat().st_size, "sha256": sha256_file(stdout_path)},
            "stderr": {"bytes": stderr_path.stat().st_size, "sha256": sha256_file(stderr_path)},
        }
    return records


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
    _write_json(
        run_dir / "status.json",
        {"schema_version": "worldsim_v51_f0f_status_v1", "task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]},
    )
    staged_dirs = {}
    for group_name, group in config["input_groups"].items():
        input_dir = run_dir / "artifacts/inputs" / group_name
        input_dir.mkdir(parents=True)
        for row in group["inputs"]:
            (input_dir / row["staging_filename"]).symlink_to(Path(row["path"]))
        staged_dirs[group_name] = input_dir

    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    started = time.perf_counter()
    monitor.start()
    try:
        total = _nvidia_total_mib()
        if total != int(config["resources"]["required_nvidia_total_mib"]):
            raise ProtocolError(f"F0f GPU total drift: {total}")
        nvidia_start = _nvidia_used_mib()
        if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
            raise ProtocolError(f"F0f unexpected GPU use at start: {nvidia_start}")
        packages = list(config["environment"]["packages"])
        imports = _environment_import_report(Path(config["environment"]["runtime"]), packages)
        if imports != {row["import_name"]: row["version"] for row in packages}:
            raise ProtocolError("F0f isolated environment import drift")
        solvers = _solver_smokes(Path(config["environment"]["runtime"]))
        health = _capture_health(run_dir)
        attempts = []
        for attempt in config["execution"]["attempts"]:
            group_name = attempt["input_group"]
            attempt_config = dict(config)
            attempt_config["inputs"] = config["input_groups"][group_name]["inputs"]
            report = _run_attempt(attempt_config, attempt, staged_dirs[group_name], run_dir)
            report["input_group"] = group_name
            attempts.append(report)
        outcome, conclusion, next_action, pair_checks = _outcome(attempts)
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0f resource monitor produced no valid sample")
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
            raise ProtocolError(f"F0f resource gate failed: {resource_checks}")
        summary = {
            "schema_version": "worldsim_v51_f0f_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "outcome": outcome,
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "environment_imports": imports,
            "solver_smokes": solvers,
            "health_probes": health,
            "attempts": attempts,
            "pair_checks": pair_checks,
            "resources": resources,
            "resource_checks": resource_checks,
            "cuda_launch_blocking": True,
            "method_parameter_change": False,
            "input_image_pixels_decoded_count": 12,
            "quality_read": False,
            "actor_identity_alignment_read": False,
            "full_materialization": False,
            "gpu_reset": False,
            "driver_mutation": False,
            "smaller_batch_retry": False,
            "identity_training_authorized": False,
            "next_action": next_action,
            "f1_execution": False,
            "f2_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": "done"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "manifest.json",
            {"schema_version": "worldsim_v51_f0f_manifest_v1", "task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)},
        )
        _write_json(
            run_dir / "status.json",
            {"schema_version": "worldsim_v51_f0f_status_v1", "task_id": TASK_ID, "status": "done", "outcome": outcome, "conclusion": conclusion, "source_commit": identity["commit"]},
        )
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append({"event": "run_blocked", "at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {"schema_version": "worldsim_v51_f0f_status_v1", "task_id": TASK_ID, "status": "blocked", "error": f"{type(error).__name__}: {error}", "source_commit": identity["commit"]},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

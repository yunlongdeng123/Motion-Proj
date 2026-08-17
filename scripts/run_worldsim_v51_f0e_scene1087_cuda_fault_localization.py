#!/usr/bin/env python3
"""执行 F0e scene-1087 三视图 CUDA/CUBLAS 故障定位。"""

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

import numpy as np
from PIL import Image


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
    _arm_command,
    _load_yaml,
    _verify,
    repository_source_identity,
)
from scripts.run_worldsim_v51_f0c_upstream_batch_association_repeatability import (
    _nvidia_total_mib,
)
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMA = "worldsim_v51_stage_f_f0e_scene1087_cuda_fault_localization_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
ATTEMPT_NAMES = ["cuda_launch_blocking_replay_1", "cuda_launch_blocking_replay_2"]


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0e config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0e status or seed drift")
    auth = config["authorization"]["f0d_closeout"]
    closeout_path = _verify(
        PROJECT / auth["path"], auth["sha256"], "F0d closeout", int(auth["bytes"])
    )
    closeout = _load_yaml(closeout_path)
    if closeout.get("status") != auth["required_status"]:
        raise ProtocolError("F0e F0d closeout status drift")
    if closeout["failure"].get("id") != auth["required_failure"]:
        raise ProtocolError("F0e authorization failure identity drift")
    if closeout["governance"].get("next_phase") != auth["required_next_phase"]:
        raise ProtocolError("F0e authorization next phase drift")

    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if _git_at(root, "rev-parse", "HEAD") != spec["commit"]:
            raise ProtocolError(f"F0e source commit drift: {name}")
        if _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise ProtocolError(f"F0e source tree drift: {name}")
        if _git_at(root, "status", "--porcelain"):
            raise ProtocolError(f"F0e source checkout not clean: {name}")
    deva_path = Path(config["sources"]["deva"]["path"]).resolve()
    grouping_path = Path(config["sources"]["gaussian_grouping"]["path"]).resolve()
    if not deva_path.is_dir() or not deva_path.is_relative_to(grouping_path):
        raise ProtocolError("F0e DEVA source boundary drift")
    for name, spec in config["assets"].items():
        _verify(Path(spec["path"]), spec["sha256"], name, int(spec["bytes"]))

    provenance = config["input_provenance"]
    manifest_path = _verify(
        Path(provenance["path"]),
        provenance["sha256"],
        "train-only image manifest",
        int(provenance["bytes"]),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        int(manifest.get("record_count", -1)) != 45
        or manifest.get("record_chain_sha256") != provenance["record_chain_sha256"]
        or manifest.get("image_pixels_decoded") is not False
    ):
        raise ProtocolError("F0e input provenance drift")
    inputs = config["inputs"]
    expected_order = [("scene-1087", 827, 0, camera) for camera in (0, 1, 2)]
    observed_order = [
        (row["scene"], int(row["scene_index"]), int(row["frame"]), int(row["camera"]))
        for row in inputs
    ]
    if observed_order != expected_order:
        raise ProtocolError("F0e exact cross-camera input order drift")
    manifest_records = {
        (row["path"], row["sha256"], int(row["bytes"]))
        for row in manifest["records"]
    }
    for row in inputs:
        _verify(Path(row["path"]), row["sha256"], row["staging_filename"], int(row["bytes"]))
        if (row["path"], row["sha256"], int(row["bytes"])) not in manifest_records:
            raise ProtocolError("F0e input is not an exact r026 manifest record")

    attempts = config["execution"]["attempts"]
    if [row["name"] for row in attempts] != ATTEMPT_NAMES:
        raise ProtocolError("F0e attempt order drift")
    if any(int(row["sam_num_points_per_side"]) != 32 for row in attempts):
        raise ProtocolError("F0e grid drift")
    if any(int(row["sam_num_points_per_batch"]) != 64 for row in attempts):
        raise ProtocolError("F0e upstream batch drift")
    if config["environment"].get("CUDA_LAUNCH_BLOCKING") != "1":
        raise ProtocolError("F0e CUDA launch blocking drift")
    resources = config["resources"]
    if (
        int(resources["required_nvidia_total_mib"]) != 24576
        or int(resources["required_nvidia_headroom_mib"]) != 256
        or int(resources["maximum_nvidia_peak_mib"]) != 24320
    ):
        raise ProtocolError("F0e resource contract drift")
    decision = config["decision"]
    for key in (
        "full_materialization_authorized",
        "quality_read",
        "actor_identity_alignment_read",
        "identity_training_authorized",
    ):
        if decision.get(key) is not False:
            raise ProtocolError(f"F0e research decision lock drift: {key}")
    locks = config["locks"]
    for name, value in locks.items():
        if name in {"input_image_pixels_decoded_count", "output_schema_reads_maximum"}:
            continue
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"F0e milestone lock drift: {name}")
        elif value is not False:
            raise ProtocolError(f"F0e research lock drift: {name}")
    return config


def _schema_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        array = np.asarray(image)
    if list(array.shape) != [900, 1600] or str(array.dtype) != "uint8":
        raise ProtocolError(f"F0e success output schema drift: {path}")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _run_attempt(
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    input_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    name = str(attempt["name"])
    attempt_dir = run_dir / "artifacts/attempts" / name
    output_dir = attempt_dir / "output"
    output_dir.mkdir(parents=True)
    command = _arm_command(config, attempt, input_dir, output_dir)
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
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=Path(config["sources"]["deva"]["path"]),
            stdout=stdout,
            stderr=stderr,
            check=False,
            env=environment,
        )
    wall_seconds = time.perf_counter() - started
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    if "Downloading:" in stderr_text:
        raise ProtocolError(f"F0e hidden download drift: {name}")
    mask_dir = output_dir / "Annotations"
    mask_paths = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
    pred_path = output_dir / "pred.json"
    expected_names = [row["staging_filename"] for row in config["inputs"]]
    expected_mask_names = [f"{Path(name).stem}.png" for name in expected_names]
    common = {
        "name": name,
        "command": command,
        "returncode": int(completed.returncode),
        "wall_seconds": wall_seconds,
        "stdout": {"bytes": stdout_path.stat().st_size, "sha256": sha256_file(stdout_path)},
        "stderr": {"bytes": stderr_path.stat().st_size, "sha256": sha256_file(stderr_path)},
        "cuda_launch_blocking": environment["CUDA_LAUNCH_BLOCKING"],
    }
    if completed.returncode == 0:
        if [path.name for path in mask_paths] != expected_mask_names or not pred_path.is_file():
            raise ProtocolError(f"F0e successful output denominator drift: {name}")
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        if [row.get("file_name") for row in pred.get("annotations", [])] != expected_names:
            raise ProtocolError(f"F0e successful metadata order drift: {name}")
        common.update(
            {
                "classification": "success",
                "masks": [_schema_record(path) for path in mask_paths],
                "metadata": {
                    "bytes": pred_path.stat().st_size,
                    "sha256": sha256_file(pred_path),
                    "annotation_count": len(pred["annotations"]),
                },
            }
        )
        return common
    markers = list(config["execution"]["expected_failure_markers"])
    if all(marker in stderr_text for marker in markers):
        if mask_paths or pred_path.exists():
            raise ProtocolError(f"F0e expected failure published partial canonical output: {name}")
        common.update(
            {
                "classification": "expected_cublas_internal_failure",
                "mask_count": 0,
                "pred_json": False,
                "explicit_pytorch_oom": "CUDA out of memory" in stderr_text,
            }
        )
        return common
    common.update(
        {
            "classification": "unexpected_failure",
            "mask_count": len(mask_paths),
            "pred_json": pred_path.exists(),
        }
    )
    return common


def _outcome(attempts: list[Mapping[str, Any]]) -> tuple[str, str, str]:
    classes = [row["classification"] for row in attempts]
    if classes == ["expected_cublas_internal_failure"] * 2:
        return (
            "both_expected_failure",
            "deterministic_cublas_internal_failure_reproduced_twice_under_cuda_launch_blocking",
            "preregister_source_neutral_tensor_shape_and_allocator_instrumentation",
        )
    if classes == ["success", "success"]:
        exact_masks = [row["sha256"] for row in attempts[0]["masks"]] == [
            row["sha256"] for row in attempts[1]["masks"]
        ]
        exact_metadata = attempts[0]["metadata"]["sha256"] == attempts[1]["metadata"]["sha256"]
        if exact_masks and exact_metadata:
            return (
                "both_success_exact",
                "cuda_launch_blocking_two_replays_succeeded_exact_r035_fault_not_reproduced",
                "preregister_scene1087_15_view_blocking_recovery_before_fresh_full_materialization",
            )
        return (
            "success_nonexact",
            "cuda_launch_blocking_two_replays_succeeded_but_outputs_not_exact",
            "close_faithful_identity_input_as_nonrepeatable",
        )
    if set(classes) == {"success", "expected_cublas_internal_failure"}:
        return (
            "mixed",
            "cuda_launch_blocking_replays_mixed_r035_fault_is_not_deterministic_under_current_probe",
            "preregister_cuda_runtime_health_and_reproducibility_gate",
        )
    raise ProtocolError(f"F0e unexpected attempt outcome: {classes}")


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
        {
            "schema_version": "worldsim_v51_f0e_cuda_fault_status_v1",
            "task_id": TASK_ID,
            "status": "running",
            "source_commit": identity["commit"],
        },
    )
    input_dir = run_dir / "artifacts/input"
    input_dir.mkdir(parents=True)
    for row in config["inputs"]:
        (input_dir / row["staging_filename"]).symlink_to(Path(row["path"]))

    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    started = time.perf_counter()
    monitor.start()
    try:
        total = _nvidia_total_mib()
        if total != int(config["resources"]["required_nvidia_total_mib"]):
            raise ProtocolError(f"F0e GPU total drift: {total}")
        nvidia_start = _nvidia_used_mib()
        if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
            raise ProtocolError(f"F0e unexpected GPU use at start: {nvidia_start}")
        packages = list(config["environment"]["packages"])
        imports = _environment_import_report(Path(config["environment"]["runtime"]), packages)
        if imports != {row["import_name"]: row["version"] for row in packages}:
            raise ProtocolError("F0e isolated environment import drift")
        solvers = _solver_smokes(Path(config["environment"]["runtime"]))
        runtime_probe = json.loads(
            subprocess.check_output(
                [
                    config["environment"]["runtime"],
                    "-c",
                    (
                        "import json,torch; print(json.dumps({"
                        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
                        "'cudnn':torch.backends.cudnn.version()}))"
                    ),
                ],
                text=True,
            )
        )
        nvidia_probe = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        attempts = [
            _run_attempt(config, attempt, input_dir, run_dir)
            for attempt in config["execution"]["attempts"]
        ]
        outcome, conclusion, next_action = _outcome(attempts)
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0e resource monitor produced no valid sample")
        resources = {
            "nvidia_total_mib": total,
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid),
            "cgroup_memory_peak_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid
            ),
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
            raise ProtocolError(f"F0e resource gate failed: {resource_checks}")
        summary = {
            "schema_version": "worldsim_v51_f0e_cuda_fault_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "outcome": outcome,
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "environment_imports": imports,
            "solver_smokes": solvers,
            "runtime_probe": runtime_probe,
            "nvidia_probe": nvidia_probe,
            "attempts": attempts,
            "resources": resources,
            "resource_checks": resource_checks,
            "cuda_launch_blocking": True,
            "method_parameter_change": False,
            "input_image_pixels_decoded_count": 6,
            "quality_read": False,
            "actor_identity_alignment_read": False,
            "full_materialization": False,
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
            {
                "schema_version": "worldsim_v51_f0e_cuda_fault_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0e_cuda_fault_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "outcome": outcome,
                "conclusion": conclusion,
                "source_commit": identity["commit"],
            },
        )
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append(
            {"event": "run_blocked", "at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"}
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0e_cuda_fault_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "source_commit": identity["commit"],
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

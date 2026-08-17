#!/usr/bin/env python3
"""执行 Stage F F0c upstream batch64 三视图关联与重复性资源门。"""

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
    _run_arm,
    _verify,
    repository_source_identity,
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


SCHEMA = "worldsim_v51_stage_f_f0c_upstream_batch_association_repeatability_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _nvidia_total_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if len(values) != 1:
        raise ProtocolError("F0c expects exactly one GPU")
    return values[0]


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0c config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0c status or seed drift")
    authorization = config["authorization"]["f0b_closeout"]
    closeout_path = _verify(
        PROJECT / authorization["path"],
        authorization["sha256"],
        "F0b closeout",
        int(authorization["bytes"]),
    )
    closeout = _load_yaml(closeout_path)
    if closeout.get("status") != authorization["required_status"]:
        raise ProtocolError("F0b closeout status drift")
    if closeout.get("conclusion") != authorization["required_conclusion"]:
        raise ProtocolError("F0b closeout conclusion drift")
    if closeout["governance"].get("next_phase") != authorization["required_next_phase"]:
        raise ProtocolError("F0b closeout did not authorize F0c")

    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if _git_at(root, "rev-parse", "HEAD") != spec["commit"]:
            raise ProtocolError(f"source commit drift: {name}")
        if _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise ProtocolError(f"source tree drift: {name}")
        if _git_at(root, "status", "--porcelain"):
            raise ProtocolError(f"source checkout not clean: {name}")
    deva_path = Path(config["sources"]["deva"]["path"]).resolve()
    gaussian_grouping_path = Path(config["sources"]["gaussian_grouping"]["path"]).resolve()
    if not deva_path.is_dir() or not deva_path.is_relative_to(gaussian_grouping_path):
        raise ProtocolError("DEVA source must remain in the frozen Gaussian Grouping tree")
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
        raise ProtocolError("F0c train-only image manifest drift")
    inputs = config["inputs"]
    if len(inputs) != 3 or [int(row["frame"]) for row in inputs] != [0, 40, 80]:
        raise ProtocolError("F0c input denominator or order drift")
    if len({int(row["camera"]) for row in inputs}) != 1:
        raise ProtocolError("F0c inputs must remain same-camera")
    selected = set()
    for row in inputs:
        _verify(Path(row["path"]), row["sha256"], row["staging_filename"], int(row["bytes"]))
        selected.add((row["path"], row["sha256"], int(row["bytes"])))
    manifest_records = {
        (row["path"], row["sha256"], int(row["bytes"]))
        for row in manifest.get("records", [])
    }
    if not selected <= manifest_records:
        raise ProtocolError("F0c inputs are not an exact manifest subset")

    arms = config["execution"]["arms"]
    if [row["name"] for row in arms] != ["primary_batch64", "repeat_batch64"]:
        raise ProtocolError("F0c arm order drift")
    if any(int(row["sam_num_points_per_side"]) != 32 for row in arms):
        raise ProtocolError("F0c prompt grid drift")
    if any(int(row["sam_num_points_per_batch"]) != 64 for row in arms):
        raise ProtocolError("F0c must restore upstream batch64")
    resources = config["resources"]
    if (
        int(resources["required_nvidia_total_mib"]) != 24576
        or int(resources["required_nvidia_headroom_mib"]) != 256
        or int(resources["maximum_nvidia_peak_mib"]) != 24320
        or int(resources["required_nvidia_total_mib"])
        - int(resources["required_nvidia_headroom_mib"])
        != int(resources["maximum_nvidia_peak_mib"])
        or resources.get("r033_old_ceiling_not_retroactively_changed") is not True
    ):
        raise ProtocolError("F0c physical GPU headroom contract drift")
    decision = config["decision"]
    if decision.get("materialization_authorized") is not False:
        raise ProtocolError("F0c must not pre-authorize materialization")
    if decision.get("identity_training_authorized") is not False:
        raise ProtocolError("F0c must not pre-authorize identity training")
    if int(decision.get("minimum_nonzero_masks", -1)) != 1:
        raise ProtocolError("F0c non-empty gate drift")
    if int(decision.get("minimum_stable_short_id_frames", -1)) != 2:
        raise ProtocolError("F0c stable-ID gate drift")
    locks = config["locks"]
    if int(locks.get("input_image_pixels_decoded_count", -1)) != 6:
        raise ProtocolError("F0c input decode denominator drift")
    if int(locks.get("output_mask_pixels_read_count", -1)) != 6:
        raise ProtocolError("F0c output read denominator drift")
    for name, value in locks.items():
        if name in {"input_image_pixels_decoded_count", "output_mask_pixels_read_count"}:
            continue
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"{name} must remain pending")
        elif value is not False:
            raise ProtocolError(f"F0c research lock drift: {name}")
    return config


def _repeatability_report(
    arms: list[Mapping[str, Any]], decision: Mapping[str, Any]
) -> dict[str, Any]:
    primary, repeat = arms
    mask_exact = [
        left["sha256"] == right["sha256"]
        for left, right in zip(primary["masks"], repeat["masks"])
    ]
    metadata_exact = primary["metadata_sha256"] == repeat["metadata_sha256"]
    association_nonempty = int(primary["nonzero_mask_count"]) >= int(
        decision["minimum_nonzero_masks"]
    )
    stable_short_id = any(
        int(count) >= int(decision["minimum_stable_short_id_frames"])
        for count in primary["positive_short_id_presence"].values()
    )
    return {
        "mask_exact_by_frame": mask_exact,
        "all_masks_exact": all(mask_exact),
        "metadata_exact": metadata_exact,
        "association_nonempty": association_nonempty,
        "stable_short_id_across_at_least_two_frames": stable_short_id,
        "all_required_before_resource": all(
            (all(mask_exact), metadata_exact, association_nonempty, stable_short_id)
        ),
    }


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
            "schema_version": "worldsim_v51_f0c_upstream_batch_status_v1",
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
        nvidia_total = _nvidia_total_mib()
        if nvidia_total != int(config["resources"]["required_nvidia_total_mib"]):
            raise ProtocolError(f"GPU total drift: {nvidia_total} MiB")
        nvidia_start = _nvidia_used_mib()
        if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
            raise ProtocolError(f"unexpected GPU use at start: {nvidia_start} MiB")
        environment = config["environment"]
        packages = list(environment["packages"])
        imports = _environment_import_report(Path(environment["runtime"]), packages)
        if imports != {row["import_name"]: row["version"] for row in packages}:
            raise ProtocolError("F0c isolated environment import drift")
        solvers = _solver_smokes(Path(environment["runtime"]))
        arm_reports = []
        for arm in config["execution"]["arms"]:
            report = _run_arm(config, arm, run_dir, input_dir)
            arm_reports.append(report)
            _write_json(run_dir / "artifacts" / arm["name"] / "report.json", report)
        repeatability = _repeatability_report(arm_reports, config["decision"])
        _write_json(run_dir / "artifacts/repeatability_report.json", repeatability)
        if not repeatability["all_required_before_resource"]:
            raise ProtocolError(f"F0c association/repeatability gate failed: {repeatability}")

        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0c resource monitor produced no valid sample")
        resources = {
            "nvidia_total_mib": nvidia_total,
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
        resources["nvidia_headroom_mib"] = (
            resources["nvidia_total_mib"] - resources["nvidia_peak_mib"]
        )
        _write_json(run_dir / "artifacts/resources.json", resources)
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_total": resources["nvidia_total_mib"]
            == int(ceilings["required_nvidia_total_mib"]),
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "nvidia_headroom": resources["nvidia_headroom_mib"]
            >= int(ceilings["required_nvidia_headroom_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "disk_free_after": resources["disk_free_after_bytes"]
            >= int(ceilings["minimum_disk_free_bytes_after"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(resource_checks.values()):
            raise ProtocolError(f"F0c resource gate failed: {resource_checks}")
        conclusion = config["decision"]["expected_conclusion"]
        summary = {
            "schema_version": "worldsim_v51_f0c_upstream_batch_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "environment_imports": imports,
            "solver_smokes": solvers,
            "arms": arm_reports,
            "repeatability": repeatability,
            "resources": resources,
            "resource_checks": resource_checks,
            "upstream_batch64_restored": True,
            "input_image_pixels_decoded_count": 6,
            "output_mask_pixels_read_count": 6,
            "quality_read": False,
            "parameter_search": False,
            "smaller_batch_retry": False,
            "materialization_authorized": False,
            "identity_training_authorized": False,
            "next_action": config["decision"]["next_action"],
            "h_quality_read": False,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
            "kitti_method_tuning": False,
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
                "schema_version": "worldsim_v51_f0c_upstream_batch_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0c_upstream_batch_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": conclusion,
                "source_commit": identity["commit"],
            },
        )
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0c_upstream_batch_status_v1",
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
        default=PROJECT
        / "configs/worldsim_v51/stage_f_f0c_upstream_batch_association_repeatability_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

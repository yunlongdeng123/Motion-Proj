#!/usr/bin/env python3
"""独立审计 Stage F F0c upstream batch64 三视图 run。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.audit_worldsim_v51_f0b_three_view_association_parity import (
    _assert_report_mask,
    _load_json,
    _load_jsonl,
    _load_yaml,
    _mask,
)
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
ARM_NAMES = ["primary_batch64", "repeat_batch64"]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _manifest_inventory(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    ]


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("task_id") != TASK_ID:
        raise ProtocolError("F0c config task drift")
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if status.get("task_id") != TASK_ID or status.get("status") != "done":
        raise ProtocolError("r034 must remain done")
    if status.get("conclusion") != config["decision"]["expected_conclusion"]:
        raise ProtocolError("r034 conclusion drift")
    source_commit = str(status["source_commit"])
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    committed_config = subprocess.check_output(
        [
            "git",
            "-C",
            str(PROJECT),
            "show",
            f"{source_commit}:configs/worldsim_v51/{config_path.name}",
        ]
    )
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.read_bytes() != committed_config:
        raise ProtocolError("r034 resolved config differs from source commit")

    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r034 event terminal drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if summary.get("status") != "done" or summary.get("source_commit") != source_commit:
        raise ProtocolError("r034 summary identity drift")
    if summary.get("source_tree") != source_tree:
        raise ProtocolError("r034 source tree drift")

    inputs = {}
    input_dir = run_dir / "artifacts/input"
    for source in config["inputs"]:
        staged = input_dir / source["staging_filename"]
        if not staged.is_symlink() or staged.resolve() != Path(source["path"]).resolve():
            raise ProtocolError(f"r034 input staging drift: {staged}")
        if staged.stat().st_size != int(source["bytes"]) or sha256_file(staged) != source["sha256"]:
            raise ProtocolError(f"r034 input identity drift: {staged}")
        inputs[source["staging_filename"]] = {
            "bytes": staged.stat().st_size,
            "sha256": sha256_file(staged),
            "target": str(staged.resolve()),
        }

    reports = []
    mask_sha_by_arm: dict[str, list[str]] = {}
    for expected_arm, arm_config in zip(ARM_NAMES, config["execution"]["arms"]):
        if arm_config["name"] != expected_arm:
            raise ProtocolError("r034 arm order drift")
        arm_dir = run_dir / "artifacts" / expected_arm
        report_path = arm_dir / "report.json"
        report = _load_json(report_path)
        if report.get("name") != expected_arm or int(report.get("annotation_count", -1)) != 3:
            raise ProtocolError(f"r034 arm report identity drift: {expected_arm}")
        if int(report["sam_num_points_per_side"]) != 32:
            raise ProtocolError(f"r034 grid drift: {expected_arm}")
        if int(report["sam_num_points_per_batch"]) != 64:
            raise ProtocolError(f"r034 upstream batch drift: {expected_arm}")
        computed_masks = []
        for source, report_mask in zip(config["inputs"], report["masks"]):
            filename = f"{Path(source['staging_filename']).stem}.png"
            _, computed = _mask(arm_dir / "output/Annotations" / filename)
            _assert_report_mask(report_mask, computed, f"{expected_arm}/{filename}")
            computed_masks.append(computed)
        metadata_path = arm_dir / "output/pred.json"
        metadata = _load_json(metadata_path)
        if len(metadata.get("annotations", [])) != 3:
            raise ProtocolError(f"r034 metadata denominator drift: {expected_arm}")
        if (
            report["metadata_sha256"] != sha256_file(metadata_path)
            or int(report["metadata_bytes"]) != metadata_path.stat().st_size
        ):
            raise ProtocolError(f"r034 metadata identity drift: {expected_arm}")
        if "Downloading:" in (arm_dir / "stderr.log").read_text(encoding="utf-8"):
            raise ProtocolError(f"r034 hidden download drift: {expected_arm}")
        reports.append(report)
        mask_sha_by_arm[expected_arm] = [row["sha256"] for row in computed_masks]
    if summary.get("arms") != reports:
        raise ProtocolError("r034 summary arm reports drift")

    repeatability_path = run_dir / "artifacts/repeatability_report.json"
    repeatability = _load_json(repeatability_path)
    primary, repeat = reports
    replayed_repeatability = {
        "mask_exact_by_frame": [
            left["sha256"] == right["sha256"]
            for left, right in zip(primary["masks"], repeat["masks"])
        ],
        "metadata_exact": primary["metadata_sha256"] == repeat["metadata_sha256"],
        "association_nonempty": int(primary["nonzero_mask_count"]) >= 1,
        "stable_short_id_across_at_least_two_frames": any(
            int(count) >= 2 for count in primary["positive_short_id_presence"].values()
        ),
    }
    replayed_repeatability["all_masks_exact"] = all(
        replayed_repeatability["mask_exact_by_frame"]
    )
    replayed_repeatability["all_required_before_resource"] = all(
        (
            replayed_repeatability["all_masks_exact"],
            replayed_repeatability["metadata_exact"],
            replayed_repeatability["association_nonempty"],
            replayed_repeatability["stable_short_id_across_at_least_two_frames"],
        )
    )
    if repeatability != replayed_repeatability or not repeatability["all_required_before_resource"]:
        raise ProtocolError("r034 repeatability report does not replay")
    if summary.get("repeatability") != repeatability:
        raise ProtocolError("r034 summary repeatability drift")

    resources_path = run_dir / "artifacts/resources.json"
    resources = _load_json(resources_path)
    samples_path = run_dir / "artifacts/resource_samples.jsonl"
    samples = _load_jsonl(samples_path)
    valid = [row for row in samples if "monitor_error" not in row]
    if len(valid) != len(samples):
        raise ProtocolError("r034 resource monitor error")
    if (
        int(resources["sample_count"]) != len(samples)
        or int(resources["monitor_error_count"]) != 0
        or int(resources["nvidia_peak_mib"])
        != max(int(row["gpu_used_mib"]) for row in valid)
        or int(resources["cgroup_memory_peak_bytes"])
        != max(int(row["cgroup_memory_current_bytes"]) for row in valid)
        or int(resources["nvidia_headroom_mib"])
        != int(resources["nvidia_total_mib"]) - int(resources["nvidia_peak_mib"])
    ):
        raise ProtocolError("r034 resource samples do not replay")
    ceilings = config["resources"]
    resource_checks = {
        "nvidia_total": int(resources["nvidia_total_mib"])
        == int(ceilings["required_nvidia_total_mib"]),
        "nvidia_peak": int(resources["nvidia_peak_mib"])
        <= int(ceilings["maximum_nvidia_peak_mib"]),
        "nvidia_headroom": int(resources["nvidia_headroom_mib"])
        >= int(ceilings["required_nvidia_headroom_mib"]),
        "cgroup_memory_peak": int(resources["cgroup_memory_peak_bytes"])
        <= int(ceilings["maximum_cgroup_memory_bytes"]),
        "wall": float(resources["wall_seconds"]) <= float(ceilings["maximum_wall_seconds"]),
        "disk_free_after": int(resources["disk_free_after_bytes"])
        >= int(ceilings["minimum_disk_free_bytes_after"]),
        "monitor": int(resources["monitor_error_count"]) == 0,
    }
    if not all(resource_checks.values()) or summary.get("resource_checks") != resource_checks:
        raise ProtocolError("r034 resource gate does not replay")
    if summary.get("resources") != resources:
        raise ProtocolError("r034 summary resources drift")

    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r034 manifest inventory does not replay")
    locked_false = (
        "quality_read",
        "parameter_search",
        "smaller_batch_retry",
        "materialization_authorized",
        "identity_training_authorized",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
        "f1_execution",
        "f2_execution",
    )
    if any(summary.get(name) is not False for name in locked_false):
        raise ProtocolError("r034 research lock drift")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise ProtocolError("r034 milestone lock drift")
    if summary.get("upstream_batch64_restored") is not True:
        raise ProtocolError("r034 upstream batch restoration drift")

    input_logical_bytes = sum(int(row["bytes"]) for row in inputs.values())
    return {
        "schema_version": "worldsim_v51_stage_f_f0c_r034_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "done",
        "audited_conclusion": status["conclusion"],
        "run_dir": str(run_dir),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "resolved_config": {
            "bytes": resolved_path.stat().st_size,
            "sha256": sha256_file(resolved_path),
        },
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "entry_count": len(inventory),
            "logical_bytes": sum(int(row["bytes"]) for row in inventory),
            "regular_bytes_excluding_input_symlink_targets": sum(
                int(row["bytes"]) for row in inventory
            )
            - input_logical_bytes,
        },
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "inputs": inputs,
        "mask_sha256": mask_sha_by_arm,
        "metadata_sha256": {
            name: report["metadata_sha256"] for name, report in zip(ARM_NAMES, reports)
        },
        "nonzero_pixels": [int(row["nonzero_pixels"]) for row in primary["masks"]],
        "stable_short_ids": primary["stable_short_ids"],
        "repeatability": repeatability,
        "resources": resources,
        "resource_checks": resource_checks,
        "quality_read": False,
        "materialization_authorized": False,
        "identity_training_authorized": False,
        "f1_execution": False,
        "f2_execution": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_delta": "V51-F60_resolved_for_f0c_by_upstream_batch64_V51-F61_resolved_by_r034_headroom",
        "next_action": "preregister_train_only_full_identity_mask_materialization",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT
        / "configs/worldsim_v51/stage_f_f0c_upstream_batch_association_repeatability_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"refusing to overwrite audit: {output}")
    result = audit(args.config.resolve(), args.run_dir.resolve())
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

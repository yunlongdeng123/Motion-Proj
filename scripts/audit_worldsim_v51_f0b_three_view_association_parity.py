#!/usr/bin/env python3
"""独立审计 Stage F F0b 三视图关联与 batch parity run。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
ARM_NAMES = ["primary_batch32", "parity_batch16", "repeat_batch32"]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON root must be a mapping: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProtocolError(f"JSONL rows missing or invalid: {path}")
    return rows


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        array = np.asarray(image)
    if list(array.shape) != [900, 1600] or str(array.dtype) != "uint8":
        raise ProtocolError(f"mask schema drift: {path}")
    labels, counts = np.unique(array, return_counts=True)
    if int(labels.min()) < 0 or int(labels.max()) > 199:
        raise ProtocolError(f"short-ID range drift: {path}")
    record = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "unique_label_histogram": {
            str(int(label)): int(count) for label, count in zip(labels, counts)
        },
        "positive_short_ids": [int(label) for label in labels if int(label) > 0],
        "nonzero_pixels": int((array > 0).sum()),
    }
    return array, record


def _assert_report_mask(
    report_mask: Mapping[str, Any], computed: Mapping[str, Any], label: str
) -> None:
    for key in (
        "bytes",
        "sha256",
        "shape",
        "dtype",
        "unique_label_histogram",
        "positive_short_ids",
        "nonzero_pixels",
    ):
        if report_mask.get(key) != computed[key]:
            raise ProtocolError(f"{label} report mask drift: {key}")


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    exact_pixels = int((left == right).sum())
    total = int(left.size)
    left_foreground = left > 0
    right_foreground = right > 0
    intersection = int(np.logical_and(left_foreground, right_foreground).sum())
    union = int(np.logical_or(left_foreground, right_foreground).sum())
    return {
        "array_exact": bool(np.array_equal(left, right)),
        "exact_label_pixels": exact_pixels,
        "different_label_pixels": total - exact_pixels,
        "exact_label_fraction": exact_pixels / total,
        "foreground_intersection_pixels": intersection,
        "foreground_union_pixels": union,
        "foreground_iou": intersection / union if union else 1.0,
        "left_nonzero_pixels": int(left_foreground.sum()),
        "right_nonzero_pixels": int(right_foreground.sum()),
    }


def _inventory(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "is_symlink": path.is_symlink(),
            }
        )
    return {
        "entry_count": len(rows),
        "logical_bytes": sum(int(row["bytes"]) for row in rows),
        "records": rows,
    }


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("task_id") != TASK_ID:
        raise ProtocolError("F0b config task drift")
    status = _load_json(run_dir / "status.json")
    if status.get("task_id") != TASK_ID or status.get("status") != "blocked":
        raise ProtocolError("r033 must remain blocked")
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
        raise ProtocolError("r033 resolved config differs from source commit")
    if (run_dir / "summary.json").exists() or (run_dir / "manifest.json").exists():
        raise ProtocolError("blocked r033 must not publish done summary or manifest")

    events = _load_jsonl(run_dir / "events.jsonl")
    if [row.get("event") for row in events] != ["run_started", "run_blocked"]:
        raise ProtocolError("r033 event terminal drift")
    elapsed_seconds = (
        datetime.fromisoformat(str(events[-1]["at_utc"]))
        - datetime.fromisoformat(str(events[0]["at_utc"]))
    ).total_seconds()

    inputs = {}
    input_dir = run_dir / "artifacts/input"
    for source in config["inputs"]:
        staged = input_dir / source["staging_filename"]
        if not staged.is_symlink() or staged.resolve() != Path(source["path"]).resolve():
            raise ProtocolError(f"r033 input staging drift: {staged}")
        if staged.stat().st_size != int(source["bytes"]) or sha256_file(staged) != source["sha256"]:
            raise ProtocolError(f"r033 input identity drift: {staged}")
        inputs[source["staging_filename"]] = {
            "bytes": staged.stat().st_size,
            "sha256": sha256_file(staged),
            "target": str(staged.resolve()),
        }

    reports: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for expected_arm, arm_config in zip(ARM_NAMES, config["execution"]["arms"]):
        if arm_config["name"] != expected_arm:
            raise ProtocolError("r033 arm order drift")
        arm_dir = run_dir / "artifacts" / expected_arm
        report = _load_json(arm_dir / "report.json")
        if report.get("name") != expected_arm or int(report.get("annotation_count", -1)) != 3:
            raise ProtocolError(f"r033 arm report identity drift: {expected_arm}")
        if int(report["sam_num_points_per_side"]) != 32:
            raise ProtocolError(f"r033 grid drift: {expected_arm}")
        if int(report["sam_num_points_per_batch"]) != int(
            arm_config["sam_num_points_per_batch"]
        ):
            raise ProtocolError(f"r033 batch drift: {expected_arm}")
        arm_arrays: dict[str, np.ndarray] = {}
        computed_masks = []
        for source, report_mask in zip(config["inputs"], report["masks"]):
            filename = f"{Path(source['staging_filename']).stem}.png"
            path = arm_dir / "output/Annotations" / filename
            array, computed = _mask(path)
            _assert_report_mask(report_mask, computed, f"{expected_arm}/{filename}")
            arm_arrays[filename] = array
            computed_masks.append(computed)
        metadata_path = arm_dir / "output/pred.json"
        metadata = _load_json(metadata_path)
        if len(metadata.get("annotations", [])) != 3:
            raise ProtocolError(f"r033 metadata denominator drift: {expected_arm}")
        if (
            report["metadata_sha256"] != sha256_file(metadata_path)
            or int(report["metadata_bytes"]) != metadata_path.stat().st_size
        ):
            raise ProtocolError(f"r033 metadata identity drift: {expected_arm}")
        if "Downloading:" in (arm_dir / "stderr.log").read_text(encoding="utf-8"):
            raise ProtocolError(f"r033 hidden download drift: {expected_arm}")
        reports[expected_arm] = {
            "wall_seconds": float(report["wall_seconds"]),
            "metadata_bytes": metadata_path.stat().st_size,
            "metadata_sha256": sha256_file(metadata_path),
            "mask_records": computed_masks,
            "stable_short_ids": report["stable_short_ids"],
            "nonzero_mask_count": int(report["nonzero_mask_count"]),
            "total_nonzero_pixels": int(report["total_nonzero_pixels"]),
        }
        arrays[expected_arm] = arm_arrays

    primary_parity = {}
    primary_repeat = {}
    for filename in arrays["primary_batch32"]:
        primary_parity[filename] = _comparison(
            arrays["primary_batch32"][filename], arrays["parity_batch16"][filename]
        )
        primary_repeat[filename] = _comparison(
            arrays["primary_batch32"][filename], arrays["repeat_batch32"][filename]
        )
    parity = _load_json(run_dir / "artifacts/parity_report.json")
    expected_parity = {
        "batch_mask_exact": all(row["array_exact"] for row in primary_parity.values()),
        "batch_metadata_exact": reports["primary_batch32"]["metadata_sha256"]
        == reports["parity_batch16"]["metadata_sha256"],
        "repeat_mask_exact": all(row["array_exact"] for row in primary_repeat.values()),
        "repeat_metadata_exact": reports["primary_batch32"]["metadata_sha256"]
        == reports["repeat_batch32"]["metadata_sha256"],
        "association_nonempty": reports["primary_batch32"]["nonzero_mask_count"] >= 1,
        "stable_short_id_across_at_least_two_frames": len(
            reports["primary_batch32"]["stable_short_ids"]
        )
        >= 1,
    }
    expected_parity["all_required"] = all(expected_parity.values())
    if parity != expected_parity:
        raise ProtocolError("r033 parity report does not replay")
    required_failure_pattern = {
        "batch_mask_exact": False,
        "batch_metadata_exact": False,
        "repeat_mask_exact": True,
        "repeat_metadata_exact": True,
        "association_nonempty": True,
        "stable_short_id_across_at_least_two_frames": True,
        "all_required": False,
    }
    if parity != required_failure_pattern:
        raise ProtocolError("r033 failure pattern drift")

    samples = _load_jsonl(run_dir / "artifacts/resource_samples.jsonl")
    valid_samples = [row for row in samples if "monitor_error" not in row]
    if len(valid_samples) != len(samples):
        raise ProtocolError("r033 resource monitor error")
    resources = {
        "sample_count": len(samples),
        "monitor_error_count": 0,
        "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid_samples),
        "cgroup_memory_peak_bytes": max(
            int(row["cgroup_memory_current_bytes"]) for row in valid_samples
        ),
        "event_elapsed_seconds": elapsed_seconds,
    }
    resources["checks"] = {
        "nvidia_peak": resources["nvidia_peak_mib"]
        <= int(config["resources"]["maximum_nvidia_peak_mib"]),
        "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
        <= int(config["resources"]["maximum_cgroup_memory_bytes"]),
        "event_elapsed": elapsed_seconds
        <= float(config["resources"]["maximum_wall_seconds"]),
        "monitor": resources["monitor_error_count"] == 0,
    }
    resources["all_recorded_checks_pass"] = all(resources["checks"].values())

    return {
        "schema_version": "worldsim_v51_stage_f_f0b_r033_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "blocked",
        "audited_conclusion": "batch_size_is_output_affecting_primary_repeat_and_association_pass",
        "run_dir": str(run_dir),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "resolved_config": {
            "bytes": resolved_path.stat().st_size,
            "sha256": sha256_file(resolved_path),
        },
        "inputs": inputs,
        "reports": reports,
        "parity": parity,
        "primary_vs_batch16": primary_parity,
        "primary_vs_repeat": primary_repeat,
        "resources": resources,
        "inventory": _inventory(run_dir),
        "quality_read": False,
        "materialization_authorized": False,
        "identity_training_authorized": False,
        "f1_execution": False,
        "f2_execution": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_delta": "V51-F60_active_resource_delta_from_audit",
        "next_action": "preregister_grid32_upstream_batch64_three_view_association_repeatability_resource_smoke",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT
        / "configs/worldsim_v51/stage_f_f0b_three_view_association_parity_v1.yaml",
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

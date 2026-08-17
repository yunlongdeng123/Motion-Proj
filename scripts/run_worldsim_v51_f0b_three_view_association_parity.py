#!/usr/bin/env python3
"""执行 Stage F F0b 三视图关联、batch parity 与 repeatability smoke。"""

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
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import (
    _environment_import_report,
    _git_at,
    _solver_smokes,
)
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _git,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMA = "worldsim_v51_stage_f_f0b_three_view_association_parity_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML root must be a mapping: {path}")
    return payload


def _verify(path: Path, digest: str, label: str, expected_bytes: int | None = None) -> Path:
    if not path.is_file() or sha256_file(path) != digest:
        raise ProtocolError(f"identity drift: {label}: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ProtocolError(f"byte drift: {label}: {path}")
    return path


def repository_source_identity(project: Path = PROJECT) -> dict[str, str]:
    return {
        "commit": _git(project, "rev-parse", "HEAD"),
        "tree": _git(project, "rev-parse", "HEAD^{tree}"),
    }


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0b config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0b status or seed drift")
    authorization = config["authorization"]["f0a_environment_freeze"]
    freeze_path = _verify(
        PROJECT / authorization["path"],
        authorization["sha256"],
        "F0a environment freeze",
    )
    freeze = _load_yaml(freeze_path)
    if freeze.get("status") != authorization["required_status"]:
        raise ProtocolError("F0a environment freeze status drift")
    if freeze["canonical_run"].get("conclusion") != authorization["required_conclusion"]:
        raise ProtocolError("F0a environment freeze conclusion drift")
    if freeze["governance"].get("next_phase") != authorization["required_next_phase"]:
        raise ProtocolError("F0a environment freeze did not unlock F0b")

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
        raise ProtocolError("DEVA source must remain inside the frozen Gaussian Grouping tree")
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
        raise ProtocolError("F0b train-only image manifest contract drift")
    if len(config["inputs"]) != 3:
        raise ProtocolError("F0b input denominator must be exactly three")
    if [int(row["frame"]) for row in config["inputs"]] != [0, 40, 80]:
        raise ProtocolError("F0b temporal input order drift")
    if len({int(row["camera"]) for row in config["inputs"]}) != 1:
        raise ProtocolError("F0b inputs must remain same-camera")
    for row in config["inputs"]:
        _verify(Path(row["path"]), row["sha256"], row["staging_filename"], int(row["bytes"]))
    selected_records = {
        (row["path"], row["sha256"], int(row["bytes"])) for row in config["inputs"]
    }
    manifest_records = {
        (row["path"], row["sha256"], int(row["bytes"]))
        for row in manifest.get("records", [])
    }
    if not selected_records <= manifest_records:
        raise ProtocolError("F0b inputs are not an exact subset of the frozen manifest")

    arms = config["execution"]["arms"]
    if [row["name"] for row in arms] != ["primary_batch32", "parity_batch16", "repeat_batch32"]:
        raise ProtocolError("F0b arm order drift")
    if [int(row["sam_num_points_per_batch"]) for row in arms] != [32, 16, 32]:
        raise ProtocolError("F0b batch parity contract drift")
    if any(int(row["sam_num_points_per_side"]) != 32 for row in arms):
        raise ProtocolError("F0b prompt grid drift")
    if config["decision"].get("materialization_authorized") is not False:
        raise ProtocolError("F0b must not pre-authorize materialization")
    if config["decision"].get("identity_training_authorized") is not False:
        raise ProtocolError("F0b must not pre-authorize identity training")
    if int(config["decision"].get("minimum_nonzero_masks", -1)) != 1:
        raise ProtocolError("F0b non-empty association gate drift")
    if int(config["decision"].get("minimum_stable_short_id_frames", -1)) != 2:
        raise ProtocolError("F0b stable short-ID gate drift")
    locks = config["locks"]
    if int(locks.get("input_image_pixels_decoded_count", -1)) != 9:
        raise ProtocolError("F0b input decode denominator drift")
    if int(locks.get("output_mask_pixels_read_count", -1)) != 9:
        raise ProtocolError("F0b output read denominator drift")
    for name, value in locks.items():
        if name in {"input_image_pixels_decoded_count", "output_mask_pixels_read_count"}:
            continue
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"{name} must remain pending")
        elif value is not False:
            raise ProtocolError(f"F0b research lock drift: {name}")
    return config


def _arm_command(
    config: Mapping[str, Any], arm: Mapping[str, Any], input_dir: Path, output_dir: Path
) -> list[str]:
    args = config["execution"]["common_arguments"]
    return [
        config["environment"]["runtime"],
        config["execution"]["upstream_command"],
        "--model",
        config["assets"]["deva"]["path"],
        "--SAM_CHECKPOINT_PATH",
        config["assets"]["sam_vit_h"]["path"],
        "--chunk_size",
        str(args["chunk_size"]),
        "--img_path",
        str(input_dir),
        "--amp",
        "--temporal_setting",
        args["temporal_setting"],
        "--size",
        str(args["size"]),
        "--output",
        str(output_dir),
        "--use_short_id",
        "--suppress_small_objects",
        "--SAM_PRED_IOU_THRESHOLD",
        str(args["SAM_PRED_IOU_THRESHOLD"]),
        "--SAM_NUM_POINTS_PER_SIDE",
        str(arm["sam_num_points_per_side"]),
        "--SAM_NUM_POINTS_PER_BATCH",
        str(arm["sam_num_points_per_batch"]),
    ]


def _mask_record(mask_path: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if list(mask.shape) != [900, 1600] or str(mask.dtype) != "uint8":
        raise ProtocolError(f"F0b output mask schema drift: {mask_path}")
    labels, counts = np.unique(mask, return_counts=True)
    if int(labels.min()) < 0 or int(labels.max()) > 199:
        raise ProtocolError(f"F0b short-ID range drift: {mask_path}")
    return {
        "frame": int(source["frame"]),
        "filename": mask_path.name,
        "path": str(mask_path),
        "bytes": mask_path.stat().st_size,
        "sha256": sha256_file(mask_path),
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "unique_label_histogram": {
            str(int(label)): int(count) for label, count in zip(labels, counts)
        },
        "positive_short_ids": [int(label) for label in labels if int(label) > 0],
        "nonzero_pixels": int((mask > 0).sum()),
    }


def _run_arm(
    config: Mapping[str, Any], arm: Mapping[str, Any], run_dir: Path, input_dir: Path
) -> dict[str, Any]:
    arm_dir = run_dir / "artifacts" / arm["name"]
    output_dir = arm_dir / "output"
    output_dir.mkdir(parents=True)
    command = _arm_command(config, arm, input_dir, output_dir)
    subprocess_environment = os.environ.copy()
    subprocess_environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_HOME": config["environment"]["TORCH_HOME"],
            "PYTORCH_CUDA_ALLOC_CONF": config["environment"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ],
        }
    )
    stdout_path = arm_dir / "stdout.log"
    stderr_path = arm_dir / "stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            command,
            cwd=Path(config["sources"]["deva"]["path"]),
            stdout=stdout,
            stderr=stderr,
            check=True,
            env=subprocess_environment,
        )
    if "Downloading:" in stderr_path.read_text(encoding="utf-8"):
        raise ProtocolError(f"F0b arm attempted hidden network download: {arm['name']}")
    mask_dir = output_dir / "Annotations"
    records = []
    for source in config["inputs"]:
        mask_path = mask_dir / f"{Path(source['staging_filename']).stem}.png"
        if not mask_path.is_file():
            raise ProtocolError(f"F0b output mask missing: {arm['name']}: {mask_path.name}")
        records.append(_mask_record(mask_path, source))
    expected_masks = {f"{Path(row['staging_filename']).stem}.png" for row in config["inputs"]}
    observed_masks = {path.name for path in mask_dir.glob("*.png")}
    if observed_masks != expected_masks:
        raise ProtocolError(f"F0b mask denominator drift: {arm['name']}")
    metadata_path = output_dir / "pred.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 3:
        raise ProtocolError(f"F0b pred.json denominator drift: {arm['name']}")
    id_presence: dict[int, int] = {}
    for row in records:
        for short_id in row["positive_short_ids"]:
            id_presence[short_id] = id_presence.get(short_id, 0) + 1
    return {
        "name": arm["name"],
        "command": command,
        "sam_num_points_per_side": int(arm["sam_num_points_per_side"]),
        "sam_num_points_per_batch": int(arm["sam_num_points_per_batch"]),
        "wall_seconds": time.perf_counter() - started,
        "masks": records,
        "metadata_path": str(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": sha256_file(metadata_path),
        "annotation_count": len(annotations),
        "positive_short_id_presence": {
            str(short_id): count for short_id, count in sorted(id_presence.items())
        },
        "stable_short_ids": [
            short_id for short_id, count in sorted(id_presence.items()) if count >= 2
        ],
        "nonzero_mask_count": sum(row["nonzero_pixels"] > 0 for row in records),
        "total_nonzero_pixels": sum(int(row["nonzero_pixels"]) for row in records),
        "input_image_pixels_decoded_count": 3,
        "output_mask_pixels_read_count": 3,
        "quality_claim": False,
    }


def _parity_report(
    arms: list[Mapping[str, Any]], decision: Mapping[str, Any]
) -> dict[str, Any]:
    table = {row["name"]: row for row in arms}
    primary = table["primary_batch32"]
    parity = table["parity_batch16"]
    repeat = table["repeat_batch32"]

    def mask_shas(row: Mapping[str, Any]) -> list[str]:
        return [str(mask["sha256"]) for mask in row["masks"]]

    batch_mask_exact = mask_shas(primary) == mask_shas(parity)
    batch_metadata_exact = primary["metadata_sha256"] == parity["metadata_sha256"]
    repeat_mask_exact = mask_shas(primary) == mask_shas(repeat)
    repeat_metadata_exact = primary["metadata_sha256"] == repeat["metadata_sha256"]
    association_nonempty = int(primary["nonzero_mask_count"]) >= int(
        decision["minimum_nonzero_masks"]
    )
    stable_short_id = any(
        int(count) >= int(decision["minimum_stable_short_id_frames"])
        for count in primary["positive_short_id_presence"].values()
    )
    return {
        "batch_mask_exact": batch_mask_exact,
        "batch_metadata_exact": batch_metadata_exact,
        "repeat_mask_exact": repeat_mask_exact,
        "repeat_metadata_exact": repeat_metadata_exact,
        "association_nonempty": association_nonempty,
        "stable_short_id_across_at_least_two_frames": stable_short_id,
        "all_required": all(
            (
                batch_mask_exact,
                batch_metadata_exact,
                repeat_mask_exact,
                repeat_metadata_exact,
                association_nonempty,
                stable_short_id,
            )
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
            "schema_version": "worldsim_v51_f0b_association_parity_status_v1",
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
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(f"unexpected GPU use at start: {nvidia_start} MiB")
    started = time.perf_counter()
    monitor.start()
    try:
        environment = config["environment"]
        packages = list(environment["packages"])
        imports = _environment_import_report(Path(environment["runtime"]), packages)
        if imports != {row["import_name"]: row["version"] for row in packages}:
            raise ProtocolError("F0b isolated environment import drift")
        solvers = _solver_smokes(Path(environment["runtime"]))
        arm_reports = []
        for arm in config["execution"]["arms"]:
            report = _run_arm(config, arm, run_dir, input_dir)
            arm_reports.append(report)
            _write_json(run_dir / "artifacts" / arm["name"] / "report.json", report)
        parity = _parity_report(arm_reports, config["decision"])
        _write_json(run_dir / "artifacts/parity_report.json", parity)
        if not parity["all_required"]:
            raise ProtocolError(f"F0b parity/association gate failed: {parity}")

        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0b resource monitor produced no valid sample")
        resources = {
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
        _write_json(run_dir / "artifacts/resources.json", resources)
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "disk_free_after": resources["disk_free_after_bytes"]
            >= int(ceilings["minimum_disk_free_bytes_after"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(resource_checks.values()):
            raise ProtocolError(f"F0b resource gate failed: {resource_checks}")
        conclusion = config["decision"]["expected_conclusion"]
        summary = {
            "schema_version": "worldsim_v51_f0b_association_parity_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "environment_imports": imports,
            "solver_smokes": solvers,
            "arms": arm_reports,
            "parity": parity,
            "resources": resources,
            "resource_checks": resource_checks,
            "input_image_pixels_decoded_count": 9,
            "output_mask_pixels_read_count": 9,
            "quality_read": False,
            "parameter_search": False,
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
                "schema_version": "worldsim_v51_f0b_association_parity_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0b_association_parity_status_v1",
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
                "schema_version": "worldsim_v51_f0b_association_parity_status_v1",
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
        / "configs/worldsim_v51/stage_f_f0b_three_view_association_parity_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

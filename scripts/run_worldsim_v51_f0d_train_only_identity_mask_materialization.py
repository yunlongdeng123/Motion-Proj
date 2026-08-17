#!/usr/bin/env python3
"""执行 Stage F F0d 45-view train-only identity mask materialization。"""

from __future__ import annotations

import argparse
import hashlib
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
    _arm_command,
    _load_yaml,
    _mask_record,
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


SCHEMA = "worldsim_v51_stage_f_f0d_train_only_identity_mask_materialization_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _record_chain(records: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_config(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0d config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0d status or seed drift")
    authorization = config["authorization"]["f0c_freeze"]
    freeze_path = _verify(
        PROJECT / authorization["path"],
        authorization["sha256"],
        "F0c freeze",
        int(authorization["bytes"]),
    )
    freeze = _load_yaml(freeze_path)
    if freeze.get("status") != authorization["required_status"]:
        raise ProtocolError("F0c freeze status drift")
    if freeze["canonical_run"].get("conclusion") != authorization["required_conclusion"]:
        raise ProtocolError("F0c freeze conclusion drift")
    if freeze["governance"].get("next_phase") != authorization["required_next_phase"]:
        raise ProtocolError("F0c freeze did not authorize F0d preregistration")

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

    manifest_spec = config["input_manifest"]
    manifest_path = _verify(
        Path(manifest_spec["path"]),
        manifest_spec["sha256"],
        "train-only image manifest",
        int(manifest_spec["bytes"]),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "record_count",
        "total_bytes",
        "record_chain_sha256",
        "order",
        "image_pixels_decoded",
    ):
        if manifest.get(key) != manifest_spec[key]:
            raise ProtocolError(f"F0d input manifest drift: {key}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 45:
        raise ProtocolError("F0d input manifest denominator drift")
    if sum(int(row["bytes"]) for row in records) != int(manifest_spec["total_bytes"]):
        raise ProtocolError("F0d input manifest total bytes drift")
    for row in records:
        _verify(Path(row["path"]), row["sha256"], row["path"], int(row["bytes"]))

    expected_order = []
    for scene in config["scene_contracts"]:
        pairs = [(frame, camera) for frame in scene["frames"] for camera in scene["cameras"]]
        if len(pairs) != int(scene["record_count"]):
            raise ProtocolError(f"F0d scene denominator drift: {scene['scene']}")
        expected_order.extend(
            (scene["scene"], int(scene["scene_index"]), int(frame), int(camera))
            for frame, camera in pairs
        )
    observed_order = [
        (row["scene"], int(row["scene_index"]), int(row["frame"]), int(row["camera"]))
        for row in records
    ]
    if observed_order != expected_order:
        raise ProtocolError("F0d scene/frame/camera order drift")
    if len({(row["scene"], Path(row["path"]).name) for row in records}) != 45:
        raise ProtocolError("F0d scene-local staging filename collision")
    if len({Path(row["path"]).name for row in records}) == 45:
        raise ProtocolError("F0d expected cross-scene filename duplication is absent")

    arm = config["execution"]["arm"]
    if int(arm["sam_num_points_per_side"]) != 32:
        raise ProtocolError("F0d grid drift")
    if int(arm["sam_num_points_per_batch"]) != 64:
        raise ProtocolError("F0d upstream batch drift")
    if int(config["execution"]["expected_mask_count"]) != 45:
        raise ProtocolError("F0d output mask denominator drift")
    if int(config["execution"]["expected_pred_json_count"]) != 3:
        raise ProtocolError("F0d metadata denominator drift")
    decision = config["decision"]
    if decision.get("full_materialization_execution_authorized") is not True:
        raise ProtocolError("F0d materialization execution authorization drift")
    if decision.get("identity_training_authorized") is not False:
        raise ProtocolError("F0d must not authorize identity training")
    if decision.get("quality_gate_authorized") is not False:
        raise ProtocolError("F0d must not pre-authorize a quality gate")
    resources = config["resources"]
    if (
        int(resources["required_nvidia_total_mib"]) != 24576
        or int(resources["required_nvidia_headroom_mib"]) != 256
        or int(resources["maximum_nvidia_peak_mib"]) != 24320
    ):
        raise ProtocolError("F0d GPU resource contract drift")
    locks = config["locks"]
    if int(locks.get("input_image_pixels_decoded_count", -1)) != 45:
        raise ProtocolError("F0d input decode denominator drift")
    if int(locks.get("output_mask_pixels_read_count", -1)) != 45:
        raise ProtocolError("F0d output read denominator drift")
    if locks.get("full_materialization_execution") is not True:
        raise ProtocolError("F0d materialization lock drift")
    for name, value in locks.items():
        if name in {
            "input_image_pixels_decoded_count",
            "output_mask_pixels_read_count",
            "full_materialization_execution",
        }:
            continue
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"{name} must remain pending")
        elif value is not False:
            raise ProtocolError(f"F0d research lock drift: {name}")
    return config, records


def _scene_command(
    config: Mapping[str, Any], input_dir: Path, output_dir: Path
) -> list[str]:
    arm = dict(config["execution"]["arm"])
    arm["name"] = "materialize"
    return _arm_command(config, arm, input_dir, output_dir)


def _run_scene(
    config: Mapping[str, Any], records: list[Mapping[str, Any]], run_dir: Path
) -> dict[str, Any]:
    scene = str(records[0]["scene"])
    scene_dir = run_dir / "artifacts/scenes" / scene
    input_dir = scene_dir / "input"
    output_dir = scene_dir / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    for row in records:
        (input_dir / Path(row["path"]).name).symlink_to(Path(row["path"]))
    command = _scene_command(config, input_dir, output_dir)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_HOME": config["environment"]["TORCH_HOME"],
            "PYTORCH_CUDA_ALLOC_CONF": config["environment"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ],
        }
    )
    stdout_path = scene_dir / "stdout.log"
    stderr_path = scene_dir / "stderr.log"
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
            env=environment,
        )
    if "Downloading:" in stderr_path.read_text(encoding="utf-8"):
        raise ProtocolError(f"F0d scene attempted hidden download: {scene}")

    expected_names = [Path(row["path"]).name for row in records]
    expected_masks = {f"{Path(name).stem}.png" for name in expected_names}
    mask_dir = output_dir / "Annotations"
    if {path.name for path in mask_dir.glob("*.png")} != expected_masks:
        raise ProtocolError(f"F0d mask denominator drift: {scene}")
    mask_records = []
    id_presence: dict[int, int] = {}
    for source in records:
        filename = f"{Path(source['path']).stem}.png"
        record = _mask_record(mask_dir / filename, source)
        record.update(
            {
                "scene": scene,
                "scene_index": int(source["scene_index"]),
                "camera": int(source["camera"]),
                "source_path": source["path"],
                "source_sha256": source["sha256"],
            }
        )
        for short_id in record["positive_short_ids"]:
            id_presence[short_id] = id_presence.get(short_id, 0) + 1
        mask_records.append(record)
    metadata_path = output_dir / "pred.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 15:
        raise ProtocolError(f"F0d pred.json denominator drift: {scene}")
    if [row.get("file_name") for row in annotations] != expected_names:
        raise ProtocolError(f"F0d pred.json order drift: {scene}")
    stable_ids = [
        short_id
        for short_id, count in sorted(id_presence.items())
        if count >= int(config["decision"]["minimum_stable_short_id_views"])
    ]
    report = {
        "scene": scene,
        "scene_index": int(records[0]["scene_index"]),
        "command": command,
        "record_count": len(records),
        "wall_seconds": time.perf_counter() - started,
        "masks": mask_records,
        "mask_count": len(mask_records),
        "nonzero_mask_count": sum(row["nonzero_pixels"] > 0 for row in mask_records),
        "zero_mask_count": sum(row["nonzero_pixels"] == 0 for row in mask_records),
        "total_nonzero_pixels": sum(int(row["nonzero_pixels"]) for row in mask_records),
        "positive_short_id_presence": {
            str(short_id): count for short_id, count in sorted(id_presence.items())
        },
        "stable_short_ids": stable_ids,
        "metadata_path": str(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": sha256_file(metadata_path),
        "annotation_count": len(annotations),
        "input_image_pixels_decoded_count": len(records),
        "output_mask_pixels_read_count": len(mask_records),
        "quality_claim": False,
    }
    if report["nonzero_mask_count"] < int(
        config["decision"]["minimum_nonzero_masks_per_scene"]
    ):
        raise ProtocolError(f"F0d non-empty scene gate failed: {scene}")
    if not stable_ids:
        raise ProtocolError(f"F0d stable short-ID scene gate failed: {scene}")
    _write_json(scene_dir / "report.json", report)
    return report


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
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v51_f0d_materialization_status_v1",
            "task_id": TASK_ID,
            "status": "running",
            "source_commit": identity["commit"],
        },
    )
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
        packages = list(config["environment"]["packages"])
        imports = _environment_import_report(
            Path(config["environment"]["runtime"]), packages
        )
        if imports != {row["import_name"]: row["version"] for row in packages}:
            raise ProtocolError("F0d isolated environment import drift")
        solvers = _solver_smokes(Path(config["environment"]["runtime"]))

        reports = []
        for scene in config["scene_contracts"]:
            scene_records = [row for row in records if row["scene"] == scene["scene"]]
            reports.append(_run_scene(config, scene_records, run_dir))
        materialized_records = []
        for report in reports:
            for mask in report["masks"]:
                materialized_records.append(
                    {
                        "scene": mask["scene"],
                        "scene_index": mask["scene_index"],
                        "frame": mask["frame"],
                        "camera": mask["camera"],
                        "source_path": mask["source_path"],
                        "source_sha256": mask["source_sha256"],
                        "mask_path": mask["path"],
                        "mask_bytes": mask["bytes"],
                        "mask_sha256": mask["sha256"],
                        "nonzero_pixels": mask["nonzero_pixels"],
                        "positive_short_ids": mask["positive_short_ids"],
                    }
                )
        if len(materialized_records) != 45:
            raise ProtocolError("F0d materialized record denominator drift")
        materialization_manifest = {
            "schema_version": "worldsim_v51_f0d_identity_mask_materialization_manifest_v1",
            "task_id": TASK_ID,
            "split": "train_only",
            "record_count": len(materialized_records),
            "order": "scene_then_frame_then_camera",
            "input_record_chain_sha256": config["input_manifest"][
                "record_chain_sha256"
            ],
            "output_record_chain_sha256": _record_chain(materialized_records),
            "quality_read": False,
            "records": materialized_records,
        }
        _write_json(run_dir / "artifacts/materialization_manifest.json", materialization_manifest)

        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0d resource monitor produced no valid sample")
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
            raise ProtocolError(f"F0d resource gate failed: {resource_checks}")
        conclusion = config["decision"]["expected_conclusion"]
        summary = {
            "schema_version": "worldsim_v51_f0d_materialization_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": conclusion,
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "environment_imports": imports,
            "solver_smokes": solvers,
            "scene_reports": reports,
            "materialization_manifest": materialization_manifest,
            "resources": resources,
            "resource_checks": resource_checks,
            "input_image_pixels_decoded_count": 45,
            "output_mask_pixels_read_count": 45,
            "full_materialization_execution": True,
            "quality_read": False,
            "parameter_search": False,
            "smaller_batch_retry": False,
            "quality_gate_authorized": False,
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
                "schema_version": "worldsim_v51_f0d_run_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0d_materialization_status_v1",
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
                "schema_version": "worldsim_v51_f0d_materialization_status_v1",
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
        / "configs/worldsim_v51/stage_f_f0d_train_only_identity_mask_materialization_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

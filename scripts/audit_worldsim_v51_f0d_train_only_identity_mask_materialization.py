#!/usr/bin/env python3
"""独立审计 Stage F F0d r035 被 CUDA/CUBLAS 中断的 materialization。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


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
EXPECTED_SCENES = ["scene-0471", "scene-1087", "scene-0379"]
EXPECTED_FAILURE_MARKERS = (
    "consensus_associated.py\", line 58, in spatial_alignment",
    "memory_readout = value @ affinity",
    "CUBLAS_STATUS_INTERNAL_ERROR",
    "cublasGemmStridedBatchedExFix",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _duration_seconds(start: str, end: str) -> float:
    return (
        datetime.fromisoformat(end.replace("Z", "+00:00"))
        - datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds()


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
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
    return rows


def _scene_records(
    records: list[dict[str, Any]], scene: str
) -> list[dict[str, Any]]:
    selected = [row for row in records if row["scene"] == scene]
    if len(selected) != 15:
        raise ProtocolError(f"r035 input manifest scene denominator drift: {scene}")
    return selected


def _audit_staging(
    run_dir: Path, records: list[dict[str, Any]], scene: str
) -> list[dict[str, Any]]:
    scene_dir = run_dir / "artifacts/scenes" / scene
    input_dir = scene_dir / "input"
    expected = _scene_records(records, scene)
    expected_names = [Path(row["path"]).name for row in expected]
    observed_names = sorted(path.name for path in input_dir.iterdir())
    if observed_names != expected_names:
        raise ProtocolError(f"r035 staged input denominator/order drift: {scene}")
    staged = []
    for source in expected:
        path = input_dir / Path(source["path"]).name
        target = Path(source["path"]).resolve()
        if not path.is_symlink() or path.resolve() != target:
            raise ProtocolError(f"r035 staged input target drift: {path}")
        if path.stat().st_size != int(source["bytes"]):
            raise ProtocolError(f"r035 staged input byte drift: {path}")
        if sha256_file(path) != source["sha256"]:
            raise ProtocolError(f"r035 staged input SHA drift: {path}")
        staged.append(
            {
                "filename": path.name,
                "target": str(target),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return staged


def _audit_completed_scene(
    config: Mapping[str, Any],
    run_dir: Path,
    records: list[dict[str, Any]],
    scene: str,
) -> dict[str, Any]:
    expected = _scene_records(records, scene)
    scene_dir = run_dir / "artifacts/scenes" / scene
    report_path = scene_dir / "report.json"
    report = _load_json(report_path)
    if report.get("scene") != scene or int(report.get("record_count", -1)) != 15:
        raise ProtocolError(f"r035 completed report identity drift: {scene}")
    if int(report.get("scene_index", -1)) != int(expected[0]["scene_index"]):
        raise ProtocolError(f"r035 completed report scene index drift: {scene}")
    if int(report.get("mask_count", -1)) != 15:
        raise ProtocolError(f"r035 completed mask denominator drift: {scene}")
    if int(report.get("annotation_count", -1)) != 15:
        raise ProtocolError(f"r035 completed metadata denominator drift: {scene}")
    command = report.get("command")
    if not isinstance(command, list):
        raise ProtocolError(f"r035 completed command missing: {scene}")
    for flag, value in (
        ("--SAM_NUM_POINTS_PER_SIDE", "32"),
        ("--SAM_NUM_POINTS_PER_BATCH", "64"),
        ("--size", "480"),
    ):
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ProtocolError(f"r035 completed command drift: {scene}/{flag}")

    mask_dir = scene_dir / "output/Annotations"
    expected_mask_names = [f"{Path(row['path']).stem}.png" for row in expected]
    if sorted(path.name for path in mask_dir.glob("*.png")) != expected_mask_names:
        raise ProtocolError(f"r035 completed output denominator drift: {scene}")
    recomputed_masks = []
    id_presence: dict[int, int] = {}
    for source, report_mask in zip(expected, report["masks"]):
        filename = f"{Path(source['path']).stem}.png"
        _, computed = _mask(mask_dir / filename)
        _assert_report_mask(report_mask, computed, f"{scene}/{filename}")
        for key, expected_value in (
            ("scene", scene),
            ("scene_index", int(source["scene_index"])),
            ("frame", int(source["frame"])),
            ("camera", int(source["camera"])),
            ("source_path", source["path"]),
            ("source_sha256", source["sha256"]),
        ):
            if report_mask.get(key) != expected_value:
                raise ProtocolError(f"r035 completed report mask drift: {scene}/{filename}/{key}")
        for short_id in computed["positive_short_ids"]:
            id_presence[short_id] = id_presence.get(short_id, 0) + 1
        recomputed_masks.append(computed)

    pred_path = scene_dir / "output/pred.json"
    pred = _load_json(pred_path)
    if [row.get("file_name") for row in pred.get("annotations", [])] != [
        Path(row["path"]).name for row in expected
    ]:
        raise ProtocolError(f"r035 completed metadata order drift: {scene}")
    if (
        report.get("metadata_sha256") != sha256_file(pred_path)
        or int(report.get("metadata_bytes", -1)) != pred_path.stat().st_size
    ):
        raise ProtocolError(f"r035 completed metadata identity drift: {scene}")

    stable_ids = [
        short_id
        for short_id, count in sorted(id_presence.items())
        if count >= int(config["decision"]["minimum_stable_short_id_views"])
    ]
    nonzero_count = sum(int(row["nonzero_pixels"]) > 0 for row in recomputed_masks)
    total_nonzero = sum(int(row["nonzero_pixels"]) for row in recomputed_masks)
    if (
        int(report.get("nonzero_mask_count", -1)) != nonzero_count
        or int(report.get("zero_mask_count", -1)) != 15 - nonzero_count
        or int(report.get("total_nonzero_pixels", -1)) != total_nonzero
        or report.get("stable_short_ids") != stable_ids
        or report.get("quality_claim") is not False
    ):
        raise ProtocolError(f"r035 completed scene aggregate drift: {scene}")
    if nonzero_count < 1 or not stable_ids:
        raise ProtocolError(f"r035 completed scene association gate drift: {scene}")
    stderr = (scene_dir / "stderr.log").read_text(encoding="utf-8")
    if "Downloading:" in stderr:
        raise ProtocolError(f"r035 completed scene hidden download: {scene}")
    return {
        "scene": scene,
        "report": {"bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        "mask_count": 15,
        "nonzero_mask_count": nonzero_count,
        "zero_mask_count": 15 - nonzero_count,
        "total_nonzero_pixels": total_nonzero,
        "stable_short_ids": stable_ids,
        "mask_sha256": [row["sha256"] for row in recomputed_masks],
        "metadata": {"bytes": pred_path.stat().st_size, "sha256": sha256_file(pred_path)},
        "quality_read": False,
    }


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("task_id") != TASK_ID:
        raise ProtocolError("F0d config task drift")
    if [row["scene"] for row in config["scene_contracts"]] != EXPECTED_SCENES:
        raise ProtocolError("F0d scene order drift")

    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if status.get("task_id") != TASK_ID or status.get("status") != "blocked":
        raise ProtocolError("r035 must remain blocked")
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
        raise ProtocolError("r035 resolved config differs from source commit")
    forbidden_done_artifacts = (
        "summary.json",
        "manifest.json",
        "artifacts/materialization_manifest.json",
        "artifacts/resources.json",
    )
    if any((run_dir / name).exists() for name in forbidden_done_artifacts):
        raise ProtocolError("blocked r035 published a done-only artifact")

    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_blocked"]:
        raise ProtocolError("r035 event terminal drift")
    if events[-1].get("error") != status.get("error"):
        raise ProtocolError("r035 status/event error drift")
    if not str(status.get("error", "")).startswith("CalledProcessError:"):
        raise ProtocolError("r035 terminal error class drift")

    manifest = _load_json(Path(config["input_manifest"]["path"]))
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 45:
        raise ProtocolError("r035 source input manifest denominator drift")
    staged = {
        scene: _audit_staging(run_dir, records, scene)
        for scene in EXPECTED_SCENES[:2]
    }
    if (run_dir / "artifacts/scenes/scene-0379").exists():
        raise ProtocolError("r035 scene-0379 must remain not started")

    completed = _audit_completed_scene(config, run_dir, records, "scene-0471")
    failed_dir = run_dir / "artifacts/scenes/scene-1087"
    if (failed_dir / "report.json").exists():
        raise ProtocolError("r035 failed scene must not publish a report")
    partial_masks = sorted((failed_dir / "output/Annotations").glob("*.png"))
    partial_pred = failed_dir / "output/pred.json"
    if partial_masks or partial_pred.exists():
        raise ProtocolError("r035 failed scene unexpectedly published partial canonical outputs")
    stderr_path = failed_dir / "stderr.log"
    stderr = stderr_path.read_text(encoding="utf-8")
    if any(marker not in stderr for marker in EXPECTED_FAILURE_MARKERS):
        raise ProtocolError("r035 CUDA/CUBLAS failure signature drift")
    stdout_path = failed_dir / "stdout.log"
    stdout = stdout_path.read_text(encoding="utf-8")
    for marker in (
        "'amp': True",
        "'size': 480",
        "'SAM_NUM_POINTS_PER_SIDE': 32",
        "'SAM_NUM_POINTS_PER_BATCH': 64",
        "'temporal_setting': 'semionline'",
        "'num_voting_frames': 3",
    ):
        if marker not in stdout:
            raise ProtocolError(f"r035 failed-scene configuration marker missing: {marker}")
    if "Downloading:" in stderr:
        raise ProtocolError("r035 failed scene attempted a hidden download")

    samples_path = run_dir / "artifacts/resource_samples.jsonl"
    samples = _load_jsonl(samples_path)
    valid = [row for row in samples if "monitor_error" not in row]
    if not valid or len(valid) != len(samples):
        raise ProtocolError("r035 resource monitor sample drift")
    nvidia_peak = max(int(row["gpu_used_mib"]) for row in valid)
    cgroup_peak = max(int(row["cgroup_memory_current_bytes"]) for row in valid)
    total = int(config["resources"]["required_nvidia_total_mib"])
    duration = _duration_seconds(events[0]["at_utc"], events[-1]["at_utc"])
    inventory = _inventory(run_dir)

    return {
        "schema_version": "worldsim_v51_stage_f_f0d_r035_blocked_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "blocked",
        "audited_conclusion": (
            "scene0471_materialized_scene1087_cublas_internal_blocked_"
            "scene0379_not_started"
        ),
        "run_dir": str(run_dir),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "resolved_config": {
            "bytes": resolved_path.stat().st_size,
            "sha256": sha256_file(resolved_path),
        },
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {
            "bytes": events_path.stat().st_size,
            "sha256": sha256_file(events_path),
            "wall_seconds": duration,
        },
        "input_staging": {
            "scene-0471": {"count": 15, "records": staged["scene-0471"]},
            "scene-1087": {"count": 15, "records": staged["scene-1087"]},
            "scene-0379": {"count": 0, "not_started": True},
        },
        "completed_scenes": [completed],
        "failed_scene": {
            "scene": "scene-1087",
            "progress_before_failure": "2/15",
            "failure_class": "CUDA_CUBLAS_STATUS_INTERNAL_ERROR",
            "failure_site": "consensus_associated.py:58 spatial_alignment value_at_affinity",
            "explicit_pytorch_oom": False,
            "root_cause_confirmed": False,
            "partial_mask_count": 0,
            "partial_pred_json": False,
            "stderr": {"bytes": stderr_path.stat().st_size, "sha256": sha256_file(stderr_path)},
            "stdout": {"bytes": stdout_path.stat().st_size, "sha256": sha256_file(stdout_path)},
        },
        "materialization": {
            "complete_scene_count": 1,
            "attempted_scene_count": 2,
            "not_started_scene_count": 1,
            "canonical_mask_count": 15,
            "required_mask_count": 45,
            "canonical_pred_json_count": 1,
            "required_pred_json_count": 3,
            "complete": False,
        },
        "resources": {
            "nvidia_total_mib": total,
            "nvidia_peak_mib": nvidia_peak,
            "nvidia_headroom_mib": total - nvidia_peak,
            "cgroup_memory_peak_bytes": cgroup_peak,
            "sample_count": len(samples),
            "monitor_error_count": 0,
            "event_wall_seconds": duration,
        },
        "inventory": {
            "entry_count": len(inventory),
            "logical_bytes": sum(int(row["bytes"]) for row in inventory),
            "records": inventory,
        },
        "quality_read": False,
        "actor_identity_alignment_read": False,
        "identity_training_authorized": False,
        "smaller_batch_retry": False,
        "f1_execution": False,
        "f2_execution": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_delta": "V51-F62_active",
        "next_action": (
            "preregister_scene1087_three_view_cuda_fault_localization_without_"
            "smaller_batch_or_quality_read"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT
        / "configs/worldsim_v51/stage_f_f0d_train_only_identity_mask_materialization_v1.yaml",
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

#!/usr/bin/env python3
"""审计 KITTI Tracking 0000/0001 的真实 adapter smoke，不读取方法质量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from motion_proj.worldsim_v5.datasets.kitti import (
    freeze_sensor_frame_policy,
    parse_tracking_calibration,
    parse_tracking_labels,
    parse_tracking_oxts,
    project_camera_points,
    transform_lidar_to_rectified_camera,
    world_from_rectified_camera,
)


TASK_ID = "WS-V5-D1-KITTI-ADAPTER-01"
SCHEMA_VERSION = "worldsim_v5_kitti_adapter_smoke_v1"
PROJECT_ROOT = PROJECT_IMPORT_ROOT
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v5/kitti_adapter_smoke_v1.yaml",
    "docs/KITTI_TRACKING_ARCHIVE_METADATA_V5.json",
    "motion_proj/worldsim_v5/datasets/kitti.py",
    "scripts/extract_worldsim_v5_kitti_smoke.py",
    "scripts/audit_worldsim_v5_kitti_adapter_smoke.py",
    "tests/test_worldsim_v5_kitti_adapter.py",
    "tests/test_worldsim_v5_kitti_smoke_extraction.py",
    "tests/test_audit_worldsim_v5_kitti_adapter_smoke.py",
)


class KittiAdapterSmokeError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(payload))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise KittiAdapterSmokeError(
            process.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return process.stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KittiAdapterSmokeError(f"配置根节点必须为 mapping：{path}")
    return payload


def validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("task_id") != TASK_ID
        or config.get("status") != "running"
        or config.get("phase") != "adapter_smoke"
    ):
        raise KittiAdapterSmokeError("KITTI adapter config schema/task/status/phase 漂移")
    if config.get("sequences") != {
        "0000": {
            "image_02": 154,
            "image_03": 154,
            "velodyne": 154,
            "full_denominator": 154,
            "lidar_missing_abstain_frames": [],
        },
        "0001": {
            "image_02": 447,
            "image_03": 447,
            "velodyne": 443,
            "full_denominator": 447,
            "lidar_missing_abstain_frames": [177, 178, 179, 180],
        },
    }:
        raise KittiAdapterSmokeError("KITTI 0000/0001 identity 或缺帧合同漂移")
    restrictions = config.get("restrictions", {})
    required_false = (
        "method_parameter_search",
        "method_training",
        "method_inference",
        "quality_read",
        "cross_domain_claim",
    )
    if any(restrictions.get(name) is not False for name in required_false):
        raise KittiAdapterSmokeError("KITTI adapter restriction 漂移")
    if config_path.resolve() != (
        PROJECT_ROOT / "configs/worldsim_v5/kitti_adapter_smoke_v1.yaml"
    ).resolve():
        raise KittiAdapterSmokeError("正式 adapter config 路径漂移")
    audit = config.get("archive_audit", {})
    audit_path = Path(str(audit.get("path", "")))
    if not audit_path.is_file() or sha256_file(audit_path) != audit.get(
        "file_sha256"
    ):
        raise KittiAdapterSmokeError("KITTI archive audit binding 漂移")


def verify_raw_manifest(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["raw_manifest"]
    path = Path(str(binding["path"]))
    if not path.is_file() or sha256_file(path) != binding.get("file_sha256"):
        raise KittiAdapterSmokeError("KITTI smoke raw manifest 漂移")
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = dict(payload)
    frozen_content_sha = content.pop("manifest_sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    if (
        payload.get("schema_version") != "worldsim_v5_kitti_tracking_smoke_raw_v1"
        or payload.get("task_id") != TASK_ID
        or payload.get("status") != "done"
        or payload.get("sequences") != ["0000", "0001"]
        or payload.get("file_count") != 1805
        or payload.get("uncompressed_bytes") != 2104258586
        or payload.get("complete") is not True
        or payload.get("sensor_payload_decoded_for_quality") is not False
        or payload.get("method_parameter_search") is not False
        or hashlib.sha256(canonical).hexdigest() != frozen_content_sha
        or frozen_content_sha != binding.get("content_sha256")
        or payload.get("archive_audit_sha256")
        != config["archive_audit"]["file_sha256"]
    ):
        raise KittiAdapterSmokeError("KITTI smoke raw manifest 合同漂移")
    root = Path(str(payload["output"]))
    if root.resolve() != Path(str(config["dataset_root"])).resolve():
        raise KittiAdapterSmokeError("KITTI smoke dataset root 漂移")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 1805:
        raise KittiAdapterSmokeError("KITTI smoke raw file inventory 漂移")
    observed_paths = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    expected_paths = sorted(str(row["path"]) for row in files)
    if observed_paths != expected_paths:
        raise KittiAdapterSmokeError("KITTI smoke extracted file set 漂移")
    total_bytes = 0
    for row in files:
        target = root.joinpath(*Path(str(row["path"])).parts)
        if (
            not target.is_file()
            or target.stat().st_size != int(row["bytes"])
            or sha256_file(target) != row["sha256"]
        ):
            raise KittiAdapterSmokeError(f"KITTI extracted payload 漂移：{row['path']}")
        total_bytes += int(row["bytes"])
    return payload, {
        "path": str(path),
        "file_sha256": binding["file_sha256"],
        "content_sha256": frozen_content_sha,
        "file_count": len(files),
        "bytes_rehashed": total_bytes,
    }


def frame_indices(path: Path, suffix: str) -> list[int]:
    files = sorted(path.glob(f"*{suffix}"))
    try:
        indices = [int(file.stem) for file in files]
    except ValueError as error:
        raise KittiAdapterSmokeError(f"非数字 frame 名：{path}") from error
    if len(indices) != len(set(indices)):
        raise KittiAdapterSmokeError(f"重复 frame identity：{path}")
    return indices


def stereo_baseline_m(calibration: Mapping[str, np.ndarray]) -> float:
    p2, p3 = calibration["P2"], calibration["P3"]
    tx2 = float(p2[0, 3] / p2[0, 0])
    tx3 = float(p3[0, 3] / p3[0, 0])
    baseline = abs(tx3 - tx2)
    if not np.isfinite(baseline) or not 0.1 <= baseline <= 1.5:
        raise KittiAdapterSmokeError(f"stereo baseline 非法：{baseline}")
    return baseline


def image_probe(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
        if width <= 0 or height <= 0 or image_format != "PNG":
            raise KittiAdapterSmokeError(f"KITTI image decode/header 非法：{path}")
        rows.append(
            {
                "path": str(path),
                "width": width,
                "height": height,
                "format": image_format,
            }
        )
    return rows


def lidar_probe(
    paths: list[Path], calibration: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        values = np.fromfile(path, dtype=np.float32)
        if values.size == 0 or values.size % 4 or not np.isfinite(values).all():
            raise KittiAdapterSmokeError(f"KITTI LiDAR N×4/finite gate 失败：{path}")
        points = values.reshape(-1, 4)
        camera = transform_lidar_to_rectified_camera(points, dict(calibration))
        pixels_2, valid_2 = project_camera_points(camera, calibration["P2"])
        pixels_3, valid_3 = project_camera_points(camera, calibration["P3"])
        if not valid_2.any() or not valid_3.any():
            raise KittiAdapterSmokeError(f"KITTI LiDAR projection 无前向点：{path}")
        rows.append(
            {
                "path": str(path),
                "point_count": int(len(points)),
                "reflectance_min": float(points[:, 3].min()),
                "reflectance_max": float(points[:, 3].max()),
                "front_projectable_p2": int(valid_2.sum()),
                "front_projectable_p3": int(valid_3.sum()),
                "finite_projected_pixels": bool(
                    np.isfinite(pixels_2[valid_2]).all()
                    and np.isfinite(pixels_3[valid_3]).all()
                ),
            }
        )
    return rows


def _sample_positions(length: int) -> list[int]:
    return sorted({0, length // 2, length - 1})


def analyze_sequence(
    root: Path, sequence: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    base = root / "training"
    image_02_paths = sorted((base / "image_02" / sequence).glob("*.png"))
    image_03_paths = sorted((base / "image_03" / sequence).glob("*.png"))
    lidar_paths = sorted((base / "velodyne" / sequence).glob("*.bin"))
    image_02 = frame_indices(base / "image_02" / sequence, ".png")
    image_03 = frame_indices(base / "image_03" / sequence, ".png")
    lidar = frame_indices(base / "velodyne" / sequence, ".bin")
    policy = freeze_sensor_frame_policy(image_02, image_03, lidar)
    if (
        len(image_02) != expected["image_02"]
        or len(image_03) != expected["image_03"]
        or len(lidar) != expected["velodyne"]
        or policy["full_denominator_count"] != expected["full_denominator"]
        or policy["lidar_missing_abstain_frames"]
        != expected["lidar_missing_abstain_frames"]
        or policy["stereo_unpaired_abstain_frames"]
        or policy["stereo_missing_abstain_frames"]
    ):
        raise KittiAdapterSmokeError(f"{sequence} sensor denominator 合同漂移")
    if any(path.stat().st_size <= 0 or path.stat().st_size % 16 for path in lidar_paths):
        raise KittiAdapterSmokeError(f"{sequence} LiDAR byte layout 不是 N×4 float32")
    calibration_path = base / "calib" / f"{sequence}.txt"
    oxts_path = base / "oxts" / f"{sequence}.txt"
    label_path = base / "label_02" / f"{sequence}.txt"
    calibration = parse_tracking_calibration(calibration_path)
    poses = parse_tracking_oxts(oxts_path)
    labels = parse_tracking_labels(label_path)
    if len(poses) != expected["full_denominator"]:
        raise KittiAdapterSmokeError(f"{sequence} OXTS denominator 漂移")
    denominator = set(image_02) | set(image_03) | set(lidar)
    if any(int(row["frame"]) not in denominator for row in labels):
        raise KittiAdapterSmokeError(f"{sequence} label frame 越界")
    pose_probes = []
    for index in _sample_positions(len(poses)):
        camera_pose = world_from_rectified_camera(poses[index], calibration)
        determinant = float(np.linalg.det(camera_pose[:3, :3]))
        if not np.isfinite(camera_pose).all() or abs(determinant - 1.0) > 5e-3:
            raise KittiAdapterSmokeError(f"{sequence} camera pose gate 失败：{index}")
        pose_probes.append(
            {
                "frame": index,
                "translation_norm_m": float(np.linalg.norm(camera_pose[:3, 3])),
                "rotation_determinant": determinant,
            }
        )
    image_samples = []
    for paths in (image_02_paths, image_03_paths):
        image_samples.extend(image_probe([paths[index] for index in _sample_positions(len(paths))]))
    paired_dimensions = {
        (row["width"], row["height"]) for row in image_samples
    }
    if len(paired_dimensions) != 1:
        raise KittiAdapterSmokeError(f"{sequence} sampled stereo dimensions 漂移")
    lidar_sample_paths = [lidar_paths[index] for index in _sample_positions(len(lidar_paths))]
    if sequence == "0001":
        by_frame = {int(path.stem): path for path in lidar_paths}
        lidar_sample_paths.extend(by_frame[frame] for frame in (176, 181))
    lidar_samples = lidar_probe(sorted(set(lidar_sample_paths)), calibration)
    nonnegative_tracks = {
        int(row["track_id"]) for row in labels if int(row["track_id"]) >= 0
    }
    expected_label_rows = {"0000": 1089, "0001": 4271}[sequence]
    expected_track_count = {"0000": 15, "0001": 98}[sequence]
    if len(labels) != expected_label_rows or len(nonnegative_tracks) != expected_track_count:
        raise KittiAdapterSmokeError(f"{sequence} label/track identity 合同漂移")
    gates = {
        "sensor_denominator_exact": True,
        "known_gap_explicitly_abstained": policy["lidar_missing_abstain_frames"]
        == expected["lidar_missing_abstain_frames"],
        "calibration_finite_and_right_handed": True,
        "stereo_baseline_plausible": True,
        "oxts_30_field_pose_chain_valid": True,
        "label_frames_within_denominator": True,
        "sampled_images_decode": True,
        "sampled_lidar_projects_to_both_cameras": True,
        "all_lidar_files_nx4_byte_aligned": True,
    }
    if not all(gates.values()):
        raise KittiAdapterSmokeError(f"{sequence} adapter gates 未全部通过")
    return {
        "schema_version": "worldsim_v5_kitti_adapter_sequence_v1",
        "task_id": TASK_ID,
        "status": "done",
        "sequence": sequence,
        "sensor_policy": policy,
        "stereo_baseline_m": stereo_baseline_m(calibration),
        "oxts_pose_count": len(poses),
        "label_row_count": len(labels),
        "nonnegative_track_id_count": len(nonnegative_tracks),
        "pose_probes": pose_probes,
        "image_probes": image_samples,
        "lidar_probes": lidar_samples,
        "gates": gates,
        "method_quality_read": False,
        "method_inference_started": False,
        "parameter_search": False,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config_path, run_dir = config_path.resolve(), run_dir.resolve()
    if run_dir.exists():
        raise KittiAdapterSmokeError(f"run 目录已存在，禁止复用：{run_dir}")
    config = load_yaml(config_path)
    validate_config(config, config_path)
    if git_output("status", "--porcelain"):
        raise KittiAdapterSmokeError("正式 KITTI adapter smoke 要求 clean worktree")
    for name in ("artifacts", "source_snapshot"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    events = run_dir / "events.jsonl"
    append_jsonl(
        events,
        {
            "at_utc": now_utc(),
            "event": "kitti_adapter_smoke_started",
            "sequences": ["0000", "0001"],
            "quality_read": False,
        },
    )
    raw_manifest, raw_verification = verify_raw_manifest(config)
    root = Path(str(config["dataset_root"]))
    results = []
    for sequence, expected in config["sequences"].items():
        result = analyze_sequence(root, sequence, expected)
        artifact = run_dir / "artifacts" / f"sequence-{sequence}.json"
        atomic_json(artifact, result)
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "sequence": sequence,
                "full_denominator_count": result["sensor_policy"][
                    "full_denominator_count"
                ],
                "evaluable_multimodal_count": result["sensor_policy"][
                    "evaluable_multimodal_count"
                ],
                "multimodal_coverage": result["sensor_policy"][
                    "multimodal_coverage"
                ],
                "lidar_missing_abstain_count": len(
                    result["sensor_policy"]["lidar_missing_abstain_frames"]
                ),
                "status": "done",
            },
        )
        results.append(
            {
                "sequence": sequence,
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "coverage": result["sensor_policy"]["multimodal_coverage"],
            }
        )
        append_jsonl(
            events,
            {
                "at_utc": now_utc(),
                "event": "kitti_adapter_sequence_complete",
                "sequence": sequence,
                "status": "done",
            },
        )
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = PROJECT_ROOT / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    archive_audit = Path(str(config["archive_audit"]["path"]))
    fingerprint = {
        "config_sha256": sha256_file(config_path),
        "raw_manifest": raw_verification,
        "raw_manifest_archive_audit_sha256": raw_manifest[
            "archive_audit_sha256"
        ],
        "archive_audit_file_sha256": sha256_file(archive_audit),
        "source_snapshots": snapshots,
        "project_git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "dirty": False,
        },
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    summary = {
        "schema_version": "worldsim_v5_kitti_adapter_smoke_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "conclusion": "two_sequence_adapter_smoke_passed_with_explicit_sensor_abstain",
        "finished_at_utc": now_utc(),
        "sequence_count": len(results),
        "sequences": results,
        "raw_payload_bytes_rehashed": raw_verification["bytes_rehashed"],
        "checkpoint": "N/A_dataset_adapter",
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "quality_read": False,
        "method_training_started": False,
        "method_inference_started": False,
        "parameter_search": False,
        "cross_domain_method_authorized": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "finished_at_utc": summary["finished_at_utc"],
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    append_jsonl(
        events,
        {
            "at_utc": now_utc(),
            "event": "kitti_adapter_smoke_complete",
            "status": "done",
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_kitti_adapter_smoke_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "artifacts": artifacts,
            "quality_read": False,
            "method_training_started": False,
            "method_inference_started": False,
            "parameter_search": False,
        },
    )
    return summary


def record_blocked(
    config_path: Path, run_dir: Path, error: BaseException
) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    event = {
        "at_utc": now_utc(),
        "event": "kitti_adapter_smoke_blocked",
        "status": "blocked",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    append_jsonl(run_dir / "events.jsonl", event)
    fingerprint = {
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "project_head": git_output("rev-parse", "HEAD"),
        "error": event,
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    summary = {
        "schema_version": "worldsim_v5_kitti_adapter_smoke_summary_v1",
        "task_id": TASK_ID,
        "status": "blocked",
        "reason": "dataset_adapter_gate_failed",
        "error": event,
        "checkpoint": "N/A_dataset_adapter",
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "quality_read": False,
        "method_training_started": False,
        "method_inference_started": False,
        "parameter_search": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "blocked",
            "finished_at_utc": now_utc(),
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_kitti_adapter_smoke_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "blocked",
            "artifacts": artifacts,
            "quality_read": False,
            "method_training_started": False,
            "method_inference_started": False,
            "parameter_search": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/worldsim_v5/kitti_adapter_smoke_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        summary = run(args.config, args.run_dir)
    except BaseException as error:
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

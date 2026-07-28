#!/usr/bin/env python3
"""审计 AD-GS nuScenes 六场景原始资产与 upstream 协议的一致性。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image
from nuscenes.nuscenes import NuScenes


SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
SENSORS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
EXPECTED_FRAMES = list(range(10, 70))
EXPECTED_VAL_REL_FRAMES = list(range(4, 60, 4))


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def finite_vector(value):
    if isinstance(value, (list, tuple)):
        return all(finite_vector(item) for item in value)
    return math.isfinite(float(value))


def add_failure(result, message):
    result["failures"].append(message)


def audit(args):
    manifest_path = Path(args.manifest)
    frame_tables_path = Path(args.frame_tables)
    raw_root = Path(args.raw_root)
    upstream_path = Path(args.upstream_script)
    manifest = json.loads(manifest_path.read_text())
    frame_tables = json.loads(frame_tables_path.read_text())
    upstream_source = upstream_path.read_text()
    result = {
        "schema_version": 1,
        "status": "running",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frame_tables": str(frame_tables_path),
        "frame_tables_sha256": sha256_file(frame_tables_path),
        "raw_root": str(raw_root),
        "upstream_script": str(upstream_path),
        "upstream_script_sha256": sha256_file(upstream_path),
        "protocol": {
            "scenes": SCENES,
            "sensors_in_upstream_order": SENSORS,
            "raw_frames": EXPECTED_FRAMES,
            "val_rel_frames": EXPECTED_VAL_REL_FRAMES,
            "train_rel_frames": [
                idx for idx in range(60) if idx not in EXPECTED_VAL_REL_FRAMES
            ],
        },
        "per_scene": {},
        "failures": [],
    }

    if not manifest.get("complete"):
        add_failure(result, "manifest.complete 不是 true")
    if manifest.get("n_required") != 1440:
        add_failure(result, "manifest.n_required 不是 1440")
    if manifest.get("protocol", {}).get("sensors") != SENSORS:
        add_failure(result, "manifest camera 顺序与 upstream 不一致")
    for snippet in [
        "get_val_frames(last_frame - first_frame + 1, 4)",
        "'CAM_FRONT',",
        "'CAM_FRONT_LEFT',",
        "'CAM_FRONT_RIGHT'",
        "default=10",
        "default=69",
    ]:
        if snippet not in upstream_source:
            add_failure(result, "upstream 协议源码缺少预期片段: {}".format(snippet))

    file_rows = manifest.get("files", [])
    rows_by_name = {row["filename"]: row for row in file_rows}
    if len(file_rows) != 1440 or len(rows_by_name) != 1440:
        add_failure(result, "manifest files 不是 1440 个唯一成员")

    bad_payload = []
    symlinks = []
    hash_mismatch = []
    total_bytes = 0
    for name, row in sorted(rows_by_name.items()):
        path = raw_root / name
        if path.is_symlink():
            symlinks.append(name)
        if not path.is_file() or path.stat().st_size <= 0:
            bad_payload.append(name)
            continue
        total_bytes += path.stat().st_size
        digest = sha256_file(path)
        if digest != row.get("sha256"):
            hash_mismatch.append(name)
    if bad_payload:
        add_failure(result, "缺失或空 payload: {}".format(bad_payload[:5]))
    if symlinks:
        add_failure(result, "required payload 包含 symlink: {}".format(symlinks[:5]))
    if hash_mismatch:
        add_failure(result, "payload SHA-256 不匹配: {}".format(hash_mismatch[:5]))
    result["payload"] = {
        "n_files": len(rows_by_name),
        "bytes": total_bytes,
        "n_missing_or_empty": len(bad_payload),
        "n_symlink": len(symlinks),
        "n_hash_mismatch": len(hash_mismatch),
    }

    map_records = json.loads(
        (raw_root / "v1.0-trainval" / "map.json").read_text()
    )
    expected_maps = sorted(record["filename"] for record in map_records)
    auxiliary_rows = {
        row["filename"]: row for row in manifest.get("auxiliary_files", [])
    }
    auxiliary_failures = []
    for name in expected_maps:
        path = raw_root / name
        row = auxiliary_rows.get(name)
        if (
            row is None
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size <= 0
            or sha256_file(path) != row.get("sha256")
        ):
            auxiliary_failures.append(name)
    if sorted(auxiliary_rows) != expected_maps:
        add_failure(result, "manifest auxiliary map 清单与 map.json 不一致")
    if auxiliary_failures:
        add_failure(result, "nuScenes map mask 缺失或哈希异常: {}".format(
            auxiliary_failures
        ))
    result["auxiliary"] = {
        "n_expected_map_masks": len(expected_maps),
        "n_manifest_map_masks": len(auxiliary_rows),
        "n_failures": len(auxiliary_failures),
    }

    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=str(raw_root),
        verbose=False,
    )
    lidar_names = {
        row["lidar_filename"]
        for rows in frame_tables.values()
        for row in rows
    }
    lidar_by_name = {
        row["filename"]: row
        for row in nusc.sample_data
        if row["filename"] in lidar_names
    }

    for scene in SCENES:
        rows = frame_tables.get(scene, [])
        scene_result = {
            "n_rows": len(rows),
            "n_rgb": 0,
            "n_lidar": 0,
            "n_images_bad_size": 0,
            "n_nonfinite_calibration_or_pose": 0,
            "n_bad_order": 0,
            "n_bad_timestamps": 0,
            "lidar_dt_ms": {},
        }
        result["per_scene"][scene] = scene_result
        if len(rows) != 180:
            add_failure(result, "{} frame table 行数不是 180".format(scene))
            continue

        by_frame = {}
        by_sensor = {sensor: [] for sensor in SENSORS}
        lidar_scene = set()
        lidar_dt = []
        for row in rows:
            by_frame.setdefault(row["frame_idx"], []).append(row)
            by_sensor.setdefault(row["camera"], []).append(row)
            lidar_scene.add(row["lidar_filename"])
            lidar_dt.append(float(row["lidar_dt_ms"]))

            image_path = raw_root / row["filename"]
            if row["filename"] not in rows_by_name:
                add_failure(result, "{} RGB 不在 manifest: {}".format(
                    scene, row["filename"]
                ))
            if row["lidar_filename"] not in rows_by_name:
                add_failure(result, "{} LiDAR 不在 manifest: {}".format(
                    scene, row["lidar_filename"]
                ))
            try:
                with Image.open(str(image_path)) as image:
                    if image.size != (1600, 900):
                        scene_result["n_images_bad_size"] += 1
                    image.verify()
            except Exception:
                scene_result["n_images_bad_size"] += 1

            sample_data = nusc.get("sample_data", row["sample_data_token"])
            calibrated = nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            finite = (
                finite_vector(calibrated["translation"])
                and finite_vector(calibrated["rotation"])
                and finite_vector(calibrated["camera_intrinsic"])
                and finite_vector(ego_pose["translation"])
                and finite_vector(ego_pose["rotation"])
            )
            if not finite:
                scene_result["n_nonfinite_calibration_or_pose"] += 1

        scene_result["n_rgb"] = len(rows)
        scene_result["n_lidar"] = len(lidar_scene)
        scene_result["lidar_dt_ms"] = {
            "min": min(lidar_dt),
            "mean": sum(lidar_dt) / len(lidar_dt),
            "max": max(lidar_dt),
        }

        if sorted(by_frame) != EXPECTED_FRAMES:
            add_failure(result, "{} raw frame index 不等于 10..69".format(scene))
        for frame_idx in EXPECTED_FRAMES:
            frame_rows = by_frame.get(frame_idx, [])
            sensors = [row["camera"] for row in frame_rows]
            expected_ids = [
                (frame_idx - 10) * 3 + offset for offset in range(3)
            ]
            image_ids = [row["image_id"] for row in frame_rows]
            if sensors != SENSORS or image_ids != expected_ids:
                scene_result["n_bad_order"] += 1

        for sensor in SENSORS:
            sensor_rows = by_sensor.get(sensor, [])
            timestamps = [row["timestamp"] for row in sensor_rows]
            if (
                len(sensor_rows) != 60
                or any(b <= a for a, b in zip(timestamps, timestamps[1:]))
            ):
                scene_result["n_bad_timestamps"] += 1

        for lidar_name in lidar_scene:
            lidar_path = raw_root / lidar_name
            lidar_row = lidar_by_name.get(lidar_name)
            if (
                lidar_row is None
                or not lidar_path.is_file()
                or lidar_path.stat().st_size <= 0
                or lidar_path.stat().st_size % 20 != 0
            ):
                add_failure(result, "{} LiDAR 不可解析: {}".format(scene, lidar_name))
                continue
            calibrated = nusc.get(
                "calibrated_sensor", lidar_row["calibrated_sensor_token"]
            )
            ego_pose = nusc.get("ego_pose", lidar_row["ego_pose_token"])
            if not (
                finite_vector(calibrated["translation"])
                and finite_vector(calibrated["rotation"])
                and finite_vector(ego_pose["translation"])
                and finite_vector(ego_pose["rotation"])
            ):
                scene_result["n_nonfinite_calibration_or_pose"] += 1

        if scene_result["n_lidar"] != 60:
            add_failure(result, "{} 唯一 LiDAR 文件数不是 60".format(scene))
        if scene_result["n_images_bad_size"]:
            add_failure(result, "{} 存在分辨率异常图像".format(scene))
        if scene_result["n_nonfinite_calibration_or_pose"]:
            add_failure(result, "{} 标定或 pose 含非有限值".format(scene))
        if scene_result["n_bad_order"]:
            add_failure(result, "{} camera/image_id 顺序不匹配".format(scene))
        if scene_result["n_bad_timestamps"]:
            add_failure(result, "{} camera 时间戳不严格递增".format(scene))

    result["status"] = "done" if not result["failures"] else "blocked"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/root/autodl-tmp/data/dynamic_recon/manifests/"
        "adgs_nuscenes_v1_manifest.json",
    )
    parser.add_argument(
        "--frame-tables",
        default="/root/autodl-tmp/data/dynamic_recon/manifests/"
        "adgs_nuscenes_v1_frame_tables.json",
    )
    parser.add_argument(
        "--raw-root",
        default="/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1",
    )
    parser.add_argument(
        "--upstream-script",
        default="/root/autodl-tmp/third_party/AD-GS/scripts/nuscene/nuscene.py",
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    result = audit(args)
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".partial")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(str(tmp), str(output))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "done" else 2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare a real Waymo v1.4.3 FRONT pilot for the official default loader.

Preserves DrivingEditor's official LiDAR conversion (both returns, ego motion
compensation, non-label-zone filtering). Adapts only the SDK return signature,
directory layout, PNG requirement, and explicit initialization PLY creation.
No masks or dynamic point clouds are fabricated. Run extract_sky.py afterwards.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import numpy as np
from PIL import Image
import tensorflow as tf
from waymo_open_dataset import dataset_pb2


def load_converter(repository):
    # Official converter only needs mkdir_or_exist and a parallel progress
    # helper from mmcv; the pilot processes frames serially and needs no mmcv.
    try:
        import mmcv
    except ImportError:
        mmcv = types.ModuleType("mmcv")
        mmcv.mkdir_or_exist = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
        mmcv.track_parallel_progress = lambda fn, items, workers: [fn(i) for i in items]
        sys.modules["mmcv"] = mmcv
    filename = repository / "scripts/waymo_converter.py"
    spec = importlib.util.spec_from_file_location("drivingeditor_official_waymo_converter", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_parse = module.parse_range_image_and_camera_projection

    def compatible_parse(frame):
        result = original_parse(frame)
        if len(result) == 3:
            return result
        if len(result) == 4:
            # New SDK additionally returns segmentation labels before TOP pose.
            return result[0], result[1], result[3]
        raise RuntimeError(f"Unknown Waymo parser contract: {len(result)} values")

    module.parse_range_image_and_camera_projection = compatible_parse
    return module, hashlib.sha256(filename.read_bytes()).hexdigest()


def save_calibration(frame, path):
    # Same pinhole camera convention as official Waymo2KITTI.save_calib,
    # explicitly ordered by Waymo enum instead of protobuf container order.
    camera_to_kitti = np.eye(4)
    camera_to_kitti[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    projections, extrinsics = [], []
    cameras = sorted(frame.context.camera_calibrations, key=lambda x: x.name)
    assert [x.name for x in cameras] == [1, 2, 3, 4, 5]
    for camera in cameras:
        intrinsic = camera.intrinsic
        projection = np.array([[intrinsic[0], 0, intrinsic[2], 0],
                               [0, intrinsic[1], intrinsic[3], 0], [0, 0, 1, 0]])
        projections.append(projection)
        extrinsics.append((camera_to_kitti @ np.linalg.inv(
            np.array(camera.extrinsic.transform).reshape(4, 4)))[:3])
    lines = []
    for name, value in [(f"P{i}", p) for i, p in enumerate(projections)] + [
            ("R0_rect", np.eye(3))] + [(f"Tr_velo_to_cam_{i}", p) for i, p in enumerate(extrinsics)]:
        lines.append(name + ": " + " ".join(f"{v:.12e}" for v in value.ravel()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_ply(path, xyz, timestamps):
    # Matches scene.scene_utils.storePly: world-space XYZ, normals store the
    # normalized/model time in every axis, plus an explicit time property.
    dtype = np.dtype([(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")] +
                     [(k, "u1") for k in ("red", "green", "blue")] + [("time", "<f4")])
    vertices = np.zeros(len(xyz), dtype=dtype)
    for axis, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, axis]
    for key in ("nx", "ny", "nz", "time"):
        vertices[key] = timestamps
    # The official loader discards PLY colors and initializes them as zeros.
    header = "ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(xyz)) + "\n"
    header += "".join("property " + ("uchar" if key in ("red", "green", "blue") else "float") +
                      " " + key + "\n" for key in dtype.names) + "end_header\n"
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--tfrecord", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--file-index", type=int, default=0, help="Author's sorted training-file index, e.g. 145")
    parser.add_argument("--num-points", type=int, default=600000)
    parser.add_argument("--model-frame-interval", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sky-mask-dir", type=Path)
    args = parser.parse_args()
    assert args.frames >= 2 and args.start >= 0 and args.num_points > 0
    assert args.model_frame_interval > 0
    assert args.tfrecord.is_file()
    module, converter_hash = load_converter(args.repository)
    args.output.mkdir(parents=True, exist_ok=False)
    for subdir in ("calib", "pose", "timestamp", "velodyne", "image/image_0", "sky/sky_0"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)

    class PilotConverter(module.Waymo2KITTI):
        def create_folder(self):
            pass

    converter = PilotConverter(str(args.tfrecord.parent), str(args.output), prefix="0", workers=1, test_mode=True)
    manifest = {"status": "extracting", "dataset_version_requested": "1.4.3", "raw_path": str(args.tfrecord.resolve()),
                "raw_size_bytes": args.tfrecord.stat().st_size, "start_frame": args.start,
                "author_file_index": args.file_index, "scene_id": args.output.name,
                "requested_frames": args.frames, "camera": "FRONT", "camera_index": 0,
                "converter_sha256": converter_hash, "model_frame_interval": args.model_frame_interval,
                "seed": args.seed, "num_points": args.num_points, "frames": [],
                "limitations": ["Static-branch pilot; dynamic foreground masks and point clouds are not prepared.",
                                "Pinhole calibration follows official converter, which ignores lens distortion.",
                                "Model time defaults to official config 0.02; raw timestamp interval is recorded separately.",
                                "Pilot subset is not the original paper's full experiment scene set."]}
    manifest_path = args.output / "preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    world_points, point_times = [], []
    center = args.model_frame_interval * (args.frames - 1) / 2
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    dataset = tf.data.TFRecordDataset(str(args.tfrecord), compression_type="").with_options(options)
    for original_index, value in enumerate(dataset):
        if original_index < args.start:
            continue
        if original_index >= args.start + args.frames:
            break
        local_index = original_index - args.start
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(value.numpy()))
        basename = f"0{args.file_index:03d}{original_index:03d}"
        front = next(image for image in frame.images if image.name == 1)
        image = Image.open(io.BytesIO(front.image)).convert("RGB")
        image.save(args.output / "image/image_0" / (basename + ".png"))
        save_calibration(frame, args.output / "calib" / (basename + ".txt"))
        converter.save_lidar(frame, args.file_index, original_index)
        converter.save_pose(frame, args.file_index, original_index)
        converter.save_timestamp(frame, args.file_index, original_index)
        cloud = np.fromfile(args.output / "velodyne" / (basename + ".bin"), dtype=np.float32).reshape(-1, 6)
        assert len(cloud) and np.isfinite(cloud).all()
        pose = np.array(frame.pose.transform).reshape(4, 4)
        xyz_world = (np.pad(cloud[:, :3], ((0, 0), (0, 1)), constant_values=1) @ pose.T)[:, :3]
        model_time = local_index * args.model_frame_interval - center
        world_points.append(xyz_world.astype(np.float32))
        point_times.append(np.full(len(cloud), model_time, dtype=np.float32))
        if args.sky_mask_dir:
            mask = Image.open(args.sky_mask_dir / (basename + ".png"))
            assert mask.size == image.size
            shutil.copy2(args.sky_mask_dir / (basename + ".png"), args.output / "sky/sky_0" / (basename + ".png"))
        manifest["frames"].append({"id": basename, "source_index": original_index,
                                   "timestamp_micros": frame.timestamp_micros, "model_time": model_time,
                                   "lidar_points": len(cloud), "image_size": list(image.size)})
        manifest["segment_context_name"] = frame.context.name
        print(json.dumps(manifest["frames"][-1]), flush=True)
    assert len(manifest["frames"]) == args.frames, "TFRecord ended before requested pilot completed"
    points = np.concatenate(world_points)
    times = np.concatenate(point_times)
    # Official initializer samples with replacement. Use deterministic legacy
    # RNG to match np.random.seed(seed)/np.random.choice behavior.
    selected = np.random.RandomState(args.seed).choice(len(points), args.num_points, replace=True)
    save_ply(args.output / "points3d.ply", points[selected], times[selected])
    raw_times = np.array([f["timestamp_micros"] for f in manifest["frames"]], dtype=np.int64)
    manifest["raw_interval_seconds_median"] = float(np.median(np.diff(raw_times)) / 1e6)
    manifest["total_lidar_points_before_sampling"] = len(points)
    manifest["sky_masks_present"] = len(list((args.output / "sky/sky_0").glob("*.png")))
    manifest["status"] = "static_inputs_prepared" if manifest["sky_masks_present"] == args.frames else "awaiting_sky_masks"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(args.output), "frames": args.frames}), flush=True)


if __name__ == "__main__":
    main()

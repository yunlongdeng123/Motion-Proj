"""红scene0029真实raw CPU准备；默认plan-only，未知49窗口或未收raw时拒绝。

不下载、不分割、不调用模型、不造actor/mask，不写旧input或正式training_ready。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from waymo_numpy_cpu import protobuf, records, lidar, calibration_text, front_rgb, save_ply, PB_ROOT, CRC_LIBRARY

BASE = Path("/root/autodl-tmp/drivingeditor_repro")
FOLLOWUP = BASE / "campaign/followup_scene_prep_20261013"
SEGMENT = "10526338824408452410_5714_660_5734_660"
RAW_FILENAME = "segment-" + SEGMENT + "_with_camera_labels.tfrecord"
EXPECTED_BYTES = 1160423074
EXPECTED_SHA256 = "8b49fad64c8ee8084f759f4e76fa780cc9227657f0bc25d00dda3ad00cf95fd3"
REVISION = "8a6fccd2b022077308c167a4d3c6f646d6e5e65b"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def check_spec(path, equivalence_path):
    if path is None:
        return None, ["SOURCE_MAPPING_AND_CONFIRMED49_WINDOW_NOT_PROVIDED"]
    spec = read(path)
    errors = []
    for key, value in (("global_training_index", 29), ("segment", SEGMENT), ("camera", "FRONT"), ("pinned_revision", REVISION), ("raw_expected_bytes", EXPECTED_BYTES),
                       ("training_frame_count", 49), ("model_frame_interval", .02), ("seed", 0), ("initial_real_lidar_samples", 600000), ("training_ready", False)):
        if spec.get(key) != value:
            errors.append("SOURCE_OR_POLICY_MISMATCH:" + key)
    window = spec.get("raw_training_range")
    if not isinstance(window, list) or len(window) != 2 or any(type(value) is not int for value in window) or window[0] < 0 or window[1] - window[0] != 48:
        errors.append("EXACT49_RAW_TRAINING_WINDOW_UNKNOWN_OR_INVALID")
    if spec.get("author_training_window_publicly_confirmed") is not False or not spec.get("training_window_provenance"):
        errors.append("DECLARED_LOCAL_WINDOW_PROVENANCE_REQUIRED")
    raw = Path(spec.get("raw_path") or "/RAW_NOT_PROVIDED")
    if raw.name != RAW_FILENAME or not raw.is_file() or raw.stat().st_size != EXPECTED_BYTES:
        errors.append("COMPLETE_REAL_RAW_NOT_RECEIVED_OR_SOURCE_SIZE_MAPPING_MISMATCH")
    download_receipt = spec.get("full_raw_receipt")
    if not download_receipt or not Path(download_receipt).is_file():
        errors.append("PINNED_FULL_RAW_DOWNLOAD_RECEIPT_MISSING")
    else:
        downloaded = read(download_receipt)
        required = {"global_training_index": 29, "segment": SEGMENT, "pinned_revision": REVISION, "local_raw_path": str(raw),
                    "status": "complete", "raw_size_bytes": EXPECTED_BYTES, "size_verified": True, "full_sha256_verified": True,
                    "raw_full_sha256": EXPECTED_SHA256}
        if any(downloaded.get(key) != value for key, value in required.items()):
            errors.append("PINNED_FULL_RAW_DOWNLOAD_RECEIPT_INVALID")
    if spec.get("identity_verified") is not True:
        errors.append("ROOT_IDENTITY_VISUAL_RECEIPT_PENDING")
    for key in ("frame_match_receipt", "identity_review_receipt"):
        receipt = spec.get(key)
        if not receipt or not Path(receipt).is_file():
            errors.append("MISSING:" + key)
            continue
        evidence = read(receipt)
        if evidence.get("global_training_index") != 29 or evidence.get("segment") != SEGMENT:
            errors.append("WRONG_IDENTITY_RECEIPT:" + key)
        if key == "frame_match_receipt":
            if evidence.get("pinned_revision") != REVISION or evidence.get("camera") != "FRONT" or evidence.get("all_record_crc32c_and_context") != "passed" or evidence.get("raw_path") != str(raw):
                errors.append("FRAME_MATCH_SOURCE_MAPPING_OR_COMPLETE_RAW_CRC_PENDING")
            visible = evidence.get("raw_frame_range_inclusive")
            best = evidence.get("best_raw_frames")
            if evidence.get("all_video_frames_matched") is not True or evidence.get("video_frame_count") != 21 or not isinstance(visible, list) or len(visible) != 2 or any(type(value) is not int for value in visible) or not isinstance(best, list) or len(best) != 21 or any(type(value) is not int for value in best) or len(set(best)) != 21 or best != sorted(best) or visible != [best[0], best[-1]]:
                errors.append("ALL21_AUTHOR_VIDEO_FRAME_MATCH_PENDING")
            elif spec.get("author_video_verified_raw_range") != visible:
                errors.append("AUTHOR_VISIBLE_WINDOW_RECEIPT_MISMATCH")
            elif "EXACT49_RAW_TRAINING_WINDOW_UNKNOWN_OR_INVALID" not in errors and not (window[0] <= visible[0] <= visible[1] <= window[1]):
                errors.append("AUTHOR_VISIBLE_WINDOW_NOT_CONTAINED_IN_LOCAL49_WINDOW")
            if evidence.get("full_raw_receipt") != download_receipt:
                errors.append("FRAME_MATCH_FULL_RAW_RECEIPT_MISMATCH")
            if type(evidence.get("raw_frame_count")) is not int or ("EXACT49_RAW_TRAINING_WINDOW_UNKNOWN_OR_INVALID" not in errors and evidence["raw_frame_count"] <= window[1]):
                errors.append("LOCAL49_WINDOW_OUTSIDE_FULL_RAW_COUNT")
        elif evidence.get("identity_verified") is not True:
            errors.append("ROOT_VISUAL_REVIEW_NOT_VERIFIED")
    if equivalence_path is None or not Path(equivalence_path).is_file():
        errors.append("REAL_CONTROL_EQUIVALENCE_RECEIPT_MISSING")
    else:
        control = read(equivalence_path)
        if control.get("status") != "PASS_CONTROL_FRAME_EQUIVALENCE" or control.get("point_count_and_order_and_intensity_elongation_rangeindex_exact") is not True or control.get("calibration_text_pose_and_RGB_exact") is not True or control.get("GPU_used") is not False or control.get("TF_imported") is not False or control.get("Torch_imported") is not False:
            errors.append("REAL_CONTROL_EQUIVALENCE_RECEIPT_INVALID")
    return spec, errors


def plan(spec_path, equivalence_path):
    spec, errors = check_spec(spec_path, equivalence_path)
    disk = shutil.disk_usage(BASE)
    if disk.free < 51 * 1024 ** 3:
        errors.append("INSUFFICIENT_SHARED_DISK_HEADROOM:30GiBfloor+20GiBmatrix+1GiBinputreserve")
    output = None
    if spec and "EXACT49_RAW_TRAINING_WINDOW_UNKNOWN_OR_INVALID" not in errors:
        window = spec.get("raw_training_range")
        if isinstance(window, list) and len(window) == 2 and all(type(value) is int for value in window):
            scene = "scene0029_%03d_%03d" % tuple(window)
            output = str(FOLLOWUP / "inputs" / (scene + "_red_cpu_v1"))
    return {"status": "READY_CPU_RAW_PREPARATION_PLAN" if not errors else "BLOCKED_CPU_PREPARATION_PLAN", "errors": errors,
            "spec": spec, "output": output, "equivalence_receipt": str(equivalence_path) if equivalence_path else None,
            "source_available": bool(spec and Path(spec.get("raw_path") or "/RAW_NOT_PROVIDED").is_file()),
            "disk_free_bytes": disk.free, "GPU_used": False, "training_started": False, "output_written": False,
            "pending_after_CPU": ["Real49 sky", "Real49 DEVA/SAM", "Reviewed physicalactor-to-instance mapping", "Canonical fullbody union and exactdynamicRGB", "Real actorLiDAR priors from selectedbody", "ActualCPU loader/SGCM/readiness", "GN11 gate and training runtime checks"]}


def raw_metadata(frame, front, image, index, crc, model_time):
    calibration = next(item for item in frame.context.camera_calibrations if item.name == 1)
    scalar_fields = {field.name: value for field, value in front.ListFields() if isinstance(value, (float, int, str, bool)) and field.name != "name"}
    return {"raw_frame_index": index, "context_name": frame.context.name, "frame_timestamp_micros": int(frame.timestamp_micros),
            "frame_pose_world_from_vehicle": list(frame.pose.transform), "front_camera_pose": list(front.pose.transform),
            "front_camera_scalar_fields": scalar_fields, "front_camera_intrinsic_including_distortion": list(calibration.intrinsic),
            "front_camera_extrinsic": list(calibration.extrinsic.transform), "image_size": [image.shape[1], image.shape[0]],
            "model_time": model_time, "source_record": crc,
            "official_loader_limit": "Originalpinhole camera convention; distortion and rolling-shutter metadata preserved but not applied"}


def track_audit(frame):
    """保存真实raw track/框仅做身份和运动审计，完全不用于training筛点/造mask。"""
    from google.protobuf.json_format import MessageToDict
    pose = np.asarray(frame.pose.transform).reshape(4, 4)
    laser_labels = []
    for label in frame.laser_labels:
        center = np.array([label.box.center_x, label.box.center_y, label.box.center_z, 1.])
        laser_labels.append({"raw_track_id": label.id, "world_center_xyz": (pose @ center)[:3].tolist(),
                             "source_label": MessageToDict(label, preserving_proto_field_name=True)})
    def camera_fields(collection):
        return [{"camera_enum": row.name, "labels": [MessageToDict(label, preserving_proto_field_name=True) for label in row.labels]} for row in collection if row.name == 1]
    return {"laser_labels": laser_labels, "camera_labels_FRONT": camera_fields(frame.camera_labels),
            "projected_lidar_labels_FRONT": camera_fields(frame.projected_lidar_labels),
            "selected_actor_track_ids": None, "selected_DEVA_instance_ids": None,
            "raw_labels_use": "Identity/worldmotion diagnostics only; never masks, actorpriors or point filters"}


def prepare(spec_path, equivalence_path, pb_root, crc_library):
    report = plan(spec_path, equivalence_path)
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]))
    spec = report["spec"]
    start, end = spec["raw_training_range"]
    output = Path(report["output"]).resolve()
    if (FOLLOWUP / "inputs").resolve() not in output.parents or output.exists():
        raise ValueError("No-clobber: output must be a newstrict followup input child")
    scene_id = "scene0029_%03d_%03d" % (start, end)
    metadata = FOLLOWUP / "metadata/red_preparation" / scene_id
    if metadata.exists():
        raise FileExistsError("Refuse to overwrite previousred preparation metadata")
    output.mkdir(parents=True)
    for name in ("image/image_0", "calib", "pose", "timestamp", "velodyne", "raw_front_metadata", "sky/sky_0"):
        (output / name).mkdir(parents=True)
    metadata.mkdir(parents=True)
    dataset_pb2 = protobuf(pb_root)
    started = time.monotonic()
    rows, world_points, point_times, audits = [], [], [], []
    for index, payload, crc in records(spec["raw_path"], end, crc_library):
        if index < start:
            continue
        frame = dataset_pb2.Frame.FromString(payload)
        if frame.context.name != SEGMENT or {item.name for item in frame.images} != {1, 2, 3, 4, 5}:
            raise ValueError("Actualraw context/camera set mismatches declaredsource")
        name, local = "0029%03d" % index, index - start
        image, front = front_rgb(frame)
        if image.shape != (1280, 1920, 3):
            raise ValueError("Red FRONT dimensions differ from1920x1280 contract")
        Image.fromarray(image).save(output / "image/image_0" / (name + ".png"))
        (output / "calib" / (name + ".txt")).write_text(calibration_text(frame), encoding="utf-8")
        pose = np.asarray(frame.pose.transform).reshape(4, 4)
        np.savetxt(output / "pose" / (name + ".txt"), pose)
        (output / "timestamp" / (name + ".txt")).write_text(str(frame.timestamp_micros), encoding="ascii")
        cloud, counts = lidar(frame, dataset_pb2)
        if len(cloud) == 0:
            raise ValueError("Emptyreal LiDAR; no placeholder fallback")
        cloud.tofile(output / "velodyne" / (name + ".bin"))
        model_time = local * .02 - .48
        world = (np.column_stack((cloud[:, :3], np.ones(len(cloud)))) @ pose.T)[:, :3].astype(np.float32)
        world_points.append(world)
        point_times.append(np.full(len(cloud), model_time, dtype=np.float32))
        raw = raw_metadata(frame, front, image, index, crc, model_time)
        write_new(output / "raw_front_metadata" / (name + ".json"), raw)
        audit = {"frame_id": name, "timestamp_micros": int(frame.timestamp_micros), **track_audit(frame)}
        write_new(metadata / "raw_actor_identity_frames" / (name + ".json"), audit)
        audits.append(audit)
        rows.append({"id": name, "source_index": index, "timestamp_micros": int(frame.timestamp_micros), "model_time": model_time,
                     "lidar_points": len(cloud), "image_size": [1920, 1280], "lidar_returns": counts})
        print(json.dumps({"stage": "CPU_RAW_FIELDS", "frame_id": name, "local_index": local, "real_lidar_points": len(cloud)}), flush=True)
    if len(rows) != 49:
        raise ValueError("Declared49 extraction not complete")
    if np.any(np.diff([row["timestamp_micros"] for row in rows]) <= 0):
        raise ValueError("Realraw physical timestamps must be strictly increasing")
    points, times = np.concatenate(world_points), np.concatenate(point_times)
    selected = np.random.RandomState(0).choice(len(points), 600000, replace=True)
    save_ply(output / "points3d.ply", points[selected], times[selected])
    np.savez_compressed(metadata / "static_seed_real_source_indices.npz", concatenated_real_point_indices=selected.astype(np.int32))
    tracks = {}
    for audit in audits:
        for label in audit["laser_labels"]:
            tracks.setdefault(label["raw_track_id"], []).append({"frame_id": audit["frame_id"], "timestamp_micros": audit["timestamp_micros"], "world_center_xyz": label["world_center_xyz"]})
    summaries = []
    for track, observations in tracks.items():
        centers = np.array([row["world_center_xyz"] for row in observations])
        physical_time = np.array([row["timestamp_micros"] for row in observations], dtype=np.float64) / 1e6
        speed = np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1) / np.diff(physical_time) if len(observations) > 1 else np.array([])
        summaries.append({"raw_track_id": track, "observed_frames": len(observations), "first_frame": observations[0]["frame_id"], "last_frame": observations[-1]["frame_id"],
                          "world_displacement_m": float(np.linalg.norm(centers[-1] - centers[0])), "median_world_XY_speed_mps": float(np.median(speed)) if len(speed) else None,
                          "role": "unassigned_identity_candidate; mustreview realRGB+DEVA before selection"})
    write_new(metadata / "raw_actor_identity_summary.json", {"tracks": summaries, "selected_actor_ids": None, "training_filter_from_GT": False})
    control = read(equivalence_path)
    manifest = {"status": "CPU_RAW_AND_REAL_STATIC_SEED_READY_SKY_BODY_PENDING", "created_utc": datetime.now(timezone.utc).isoformat(),
                "scene_id": scene_id, "raw_path": spec["raw_path"], "raw_size_bytes": EXPECTED_BYTES, "raw_pinned_sha256": EXPECTED_SHA256,
                "full_raw_receipt": spec["full_raw_receipt"], "author_file_index": 29,
                "start_frame": start, "requested_frames": 49, "frames": rows, "camera": "FRONT", "camera_index": 0,
                "seed": 0, "num_points": 600000, "model_frame_interval": .02, "model_time_range": [-.48, .48],
                "raw_training_range": [start, end], "author_video_verified_raw_range": spec["author_video_verified_raw_range"],
                "author_training_window_publicly_confirmed": False, "training_window_provenance": spec["training_window_provenance"],
                "raw_interval_seconds_median": float(np.median(np.diff([row["timestamp_micros"] for row in rows])) / 1e6),
                "total_lidar_points_before_sampling": len(points), "sky_masks_present": 0,
                "fixed_pose_anchor_for_later_training": "0029%03d" % (start + 10),
                "static_prior_role": "Realworld LiDAR mixture,600k seed0 withreplacement; actor exclusion waitsrealbodypriors and originalstrict0.1m loader rule",
                "initial_PLY_colors": "Zeros per originalinitialization; not measuredRGB", "control_equivalence": control,
                "CPU_port": "PurePB/NumPy originalequations; knownnonbitwise float32 XYZ differences recorded incontrolreceipt; no scientificimpact certification",
                "lidar_binary_columns": ["x_vehicle", "y_vehicle", "z_vehicle", "intensity", "elongation", "range_image_index_or_minus1"],
                "shadow_supervision": "absent", "m_detector_used": False, "GT_labels_used_to_filter_training_inputs": False,
                "dynamic_actor_priors_created": False, "canonical_body_masks_created": False,
                "GPU_used": False, "Torch_imported": any(name == "torch" for name in sys.modules), "TF_imported": any(name.startswith("tensorflow") for name in sys.modules),
                "training_started": False, "training_ready": False, "pending": report["pending_after_CPU"], "wall_seconds": time.monotonic() - started}
    if manifest["Torch_imported"] or manifest["TF_imported"] or list(output.glob("points3d_dynamic*.ply")) or (output / "dynamic_masks").exists():
        raise RuntimeError("CPU scope unexpectedly createdframework or actor/mask data")
    write_new(output / "preparation_manifest.json", manifest)
    write_new(metadata / "cpu_raw_readiness.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output), "readiness": str(metadata / "cpu_raw_readiness.json"), "training_ready": False}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-spec", type=Path)
    parser.add_argument("--equivalence-receipt", type=Path)
    parser.add_argument("--proto-pythonpath", type=Path, default=PB_ROOT)
    parser.add_argument("--crc-library", type=Path, default=CRC_LIBRARY)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.scene_spec, args.equivalence_receipt, args.proto_pythonpath, args.crc_library)
    else:
        result = plan(args.scene_spec, args.equivalence_receipt)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result["errors"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

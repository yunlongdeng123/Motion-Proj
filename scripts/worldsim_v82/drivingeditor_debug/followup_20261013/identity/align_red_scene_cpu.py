#!/usr/bin/env python3
"""作者红车视频与完整scene0029原始TFRecord逐帧对齐；默认只打印计划。

--run仅顺序读已完整校验的本地文件，不发送网络请求。复用冻结搜索的
sharedlock/caps/账本；只写同搜索root的identity/index029/，不改旧工具与receipt。
数值匹配输出候选，root视觉复核前identity_verified始终false。
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

INDEX = 29
SEGMENT = "10526338824408452410_5714_660_5734_660"
FILENAME = "segment-" + SEGMENT + "_with_camera_labels.tfrecord"
ALIGNMENT_OWNER = "red_identity_alignment_cpu_20261013_v1"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--output-root", type=Path, required=True, help="现有red_search_cpu root，使用同一锁与账本")
    p.add_argument("--search-tools-dir", type=Path, required=True, help="冻结v1与temporal脚本同目录")
    p.add_argument("--reference-dir", type=Path, required=True, help="21个videoNNN.png与reference_manifest.json")
    p.add_argument("--raw-tfrecord", type=Path)
    p.add_argument("--raw-receipt", type=Path)
    p.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro"))
    p.add_argument("--raw-root", type=Path, default=Path("/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files"))
    p.add_argument("--source-index", type=Path, default=Path("/root/autodl-tmp/dggt/.preparation/hf_waymo_training_tree.json"))
    p.add_argument("--opencv-pythonpath", type=Path, default=Path("/root/autodl-tmp/envs/motionproj/lib/python3.10/site-packages"))
    p.add_argument("--max-raw-frames", type=int, default=200)
    args = p.parse_args()
    if not 1 <= args.max_raw_frames <= 200:
        p.error("the finite raw-frame/header cap cannot exceed200")
    args.output_root = args.output_root.resolve()
    args.reference_dir = args.reference_dir.resolve()
    if args.raw_receipt is None:
        args.raw_receipt = args.output_root.parent / "inputs/raw_scene0029/download_receipt.json"
    return args


def load_tools(args):
    sys.path.insert(0, str(args.search_tools_dir.resolve()))
    import search_red_scene_cpu as base
    import search_red_scene_temporal_cpu as temporal
    return base, temporal


def read_reference(args, base):
    manifest = json.loads((args.reference_dir / "reference_manifest.json").read_text(encoding="utf-8"))
    expected = {"official_filename": "20241214-005533.mp4", "video_frame_count": 21,
        "video_fps_rational": "8/1", "matching_wh": [480, 320], "excluded_top_matching_pixels": 28,
        "frame_indices": list(range(21))}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise base.IntegrityError("reference manifest differs from the21-frame official red-car video")
    files = [args.reference_dir / ("video%03d.png" % i) for i in range(21)]
    if any(not p.is_file() for p in files):
        raise base.IntegrityError("not all21 decoded author-video panes are present")
    images = []
    for path in files:
        with base.Image.open(path) as im:
            if im.size != (480, 320):
                raise base.IntegrityError("reference pane has incorrect480x320 dimensions")
            images.append(base.np.array(im.convert("RGB")))
    return manifest, files, base.np.stack(images)


def validate_raw(args, row, base, require_complete):
    receipt = None
    if args.raw_receipt.is_file():
        receipt = json.loads(args.raw_receipt.read_text(encoding="utf-8"))
    if args.raw_tfrecord is None:
        args.raw_tfrecord = Path(receipt["local_raw_path"]) if receipt and receipt.get("local_raw_path") else args.raw_receipt.parent / FILENAME
    path = args.raw_tfrecord.resolve()
    if not require_complete:
        return path, receipt
    if receipt is None or receipt.get("status") != "complete":
        raise base.IntegrityError("complete local raw download receipt is required before CPU scanning")
    expected = {"global_training_index": INDEX, "segment": SEGMENT, "public_repository": base.REPO,
        "pinned_revision": base.REV, "public_source_path": row["path"],
        "raw_expected_size_bytes": row["size"], "raw_size_bytes": row["size"],
        "raw_expected_full_sha256": row["lfs"]["oid"], "raw_full_sha256": row["lfs"]["oid"],
        "size_verified": True, "full_sha256_verified": True}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise base.IntegrityError("full-raw receipt does not attest the pinned publicscene29 file")
    if path.name != FILENAME or Path(receipt["local_raw_path"]).resolve() != path or not path.is_file() or path.stat().st_size != row["size"]:
        raise base.IntegrityError("complete local raw path/name/stat does not match its attested receipt")
    return path, receipt


def scalar(proto, name):
    value = getattr(proto, name)
    try:
        present = proto.HasField(name)
    except (ValueError, AttributeError):
        present = None
    return {"value": float(value), "field_present": present}


def frame_metadata(frame, front, index, offset, length, image_wh):
    calibration = next(c for c in frame.context.camera_calibrations if c.name == 1)
    return {"raw_frame": index, "converter_frame_id": "0029%03d" % index,
        "segment": frame.context.name, "header_offset": offset, "payload_length": length,
        "next_record_offset": offset + length + 16, "length_crc32c": "passed", "payload_crc32c": "passed",
        "protobuf_context": "passed", "timestamp_micros": int(frame.timestamp_micros),
        "frame_pose_transform_row_major": list(frame.pose.transform),
        "front": {"camera_enum": 1, "camera_name": "FRONT", "image_wh": list(image_wh),
            "pose_transform_row_major": list(front.pose.transform), "pose_timestamp": scalar(front, "pose_timestamp"),
            "camera_trigger_time": scalar(front, "camera_trigger_time"),
            "camera_readout_done_time": scalar(front, "camera_readout_done_time"),
            "intrinsic": list(calibration.intrinsic), "extrinsic_transform_row_major": list(calibration.extrinsic.transform)},
        "time_semantics": "Original SDK fields; frame timestamp, frame pose and camera exposure times are preserved separately. No guessed time offset or velocity.",
        "raw_labels_used": False}


def score_sequences(full, variation, raw_frames, np):
    combined = 0.5 * full + 0.5 * variation
    order = np.argsort(combined, axis=1)
    independent = [int(raw_frames[i]) for i in order[:, 0]]
    margins = [float(combined[i, order[i, 1]] - combined[i, order[i, 0]]) for i in range(21)]
    lookup = {frame: i for i, frame in enumerate(raw_frames)}
    sequences = []
    for stride in (1, 2):
        for start in raw_frames:
            ids = [start + stride * i for i in range(21)]
            if not all(i in lookup for i in ids):
                continue
            cols = np.array([lookup[i] for i in ids])
            video = np.arange(21)
            sequences.append({"raw_start": start, "raw_end_inclusive": ids[-1], "stride": stride,
                "raw_frames": ids, "mean_combined_mae": float(combined[video, cols].mean()),
                "mean_rgb_mae": float(full[video, cols].mean()), "max_rgb_mae": float(full[video, cols].max()),
                "mean_temporal_variation_region_mae": float(variation[video, cols].mean()),
                "per_frame_rgb_mae": full[video, cols].tolist(), "per_frame_combined_mae": combined[video, cols].tolist()})
    if not sequences:
        raise ValueError("fewer than21 usable raw frames; cannot propose a complete video alignment")
    sequences.sort(key=lambda row: row["mean_combined_mae"])
    best = sequences[0]
    consistency = independent == best["raw_frames"] and min(margins) > 1e-6
    # 阈值只给出数值候选条件；即使通过也不会填人工identity verdict。
    low_error = best["mean_rgb_mae"] < 0.03 and best["max_rgb_mae"] < 0.05
    return combined, independent, margins, sequences[:10], consistency, low_error


def job_class(base):
    class AlignmentJob(base.Search):
        def __init__(self, inherited, options, reference, references):
            super().__init__(inherited)
            self.options, self.reference, self.references = options, reference, references
            self.identity_root = self.root / "identity/index029"
            marker = self.identity_root / "owned_identity.json"
            if self.identity_root.exists() and not marker.exists() and any(self.identity_root.iterdir()):
                raise base.IntegrityError("identity output is not owned by this alignment helper")
            self.identity_root.mkdir(parents=True, exist_ok=True)
            if marker.exists():
                if json.loads(marker.read_text()).get("owner") != ALIGNMENT_OWNER:
                    raise base.IntegrityError("identity output owner differs")
            else:
                base.atomic_json(marker, {"owner": ALIGNMENT_OWNER, "created_utc": base.utc(),
                    "single_segment": SEGMENT, "maximum_raw_frames": 200, "network_requests": 0})
            self.state["other_segment_search_paused_for_identity"] = True
            self.state["identity_focus_index"] = INDEX
            self.state["status"] = "scene29_local_crc_pose_video_alignment"
            self.save()

        def scan_local(self, raw, row):
            from waymo_open_dataset import dataset_pb2
            if "torch" in sys.modules or "tensorflow" in sys.modules:
                raise base.IntegrityError("CPU alignment unexpectedly imported a GPU/training framework")
            self.setup_crc()
            start_stat = raw.stat()
            destinations = self.identity_root / "raw_front"
            destinations.mkdir(exist_ok=True)
            targets = self.references[:, 28:].astype(base.np.int16)
            median = base.np.median(targets, axis=0)
            variation_mask = base.np.max(base.np.abs(targets - median), axis=0).mean(axis=2) > 8
            if variation_mask.sum() < 100:
                variation_mask[:] = True
            base.Image.fromarray((variation_mask.astype(base.np.uint8) * 255)).save(self.identity_root / "author_temporal_variation_region.png")
            full_columns, variation_columns, metadata = [], [], []
            with raw.open("rb") as stream:
                while stream.tell() < row["size"]:
                    self.check_limits()
                    index = len(metadata)
                    if index >= self.options.max_raw_frames:
                        raise base.StopSearch("complete source exceeds the finite200-frame/header scope")
                    offset = stream.tell()
                    header = stream.read(12)
                    if len(header) != 12 or self.masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
                        raise base.IntegrityError("local source TFRecord length CRC32C failed")
                    length = struct.unpack("<Q", header[:8])[0]
                    if length <= 0 or length + 16 > self.contract["max_single_record_bytes"] or offset + length + 16 > row["size"]:
                        raise base.IntegrityError("local source TFRecord record exceeds the capped size or attested file end")
                    rest = stream.read(length + 4)
                    if len(rest) != length + 4 or self.masked_crc(rest[:-4]) != struct.unpack("<I", rest[-4:])[0]:
                        raise base.IntegrityError("local source TFRecord payload CRC32C failed")
                    frame = dataset_pb2.Frame.FromString(rest[:-4])
                    if frame.context.name != SEGMENT or {image.name for image in frame.images} != set(base.CAM):
                        raise base.IntegrityError("local frame context/five-camera set differs from publicscene29")
                    front = next(image for image in frame.images if image.name == 1)
                    with base.Image.open(base.io.BytesIO(front.image)) as image:
                        image_wh = image.size
                        preview = image.convert("RGB").resize((480, 320), base.Image.Resampling.BOX)
                        preview.save(destinations / ("raw%03d.png" % index))
                        array = base.np.array(preview)[28:].astype(base.np.int16)
                    pose = list(frame.pose.transform)
                    if len(pose) != 16 or not base.np.isfinite(pose).all():
                        raise base.IntegrityError("original SDK Frame.pose is absent or nonfinite")
                    info = frame_metadata(frame, front, index, offset, length, image_wh)
                    if metadata and info["timestamp_micros"] <= metadata[-1]["timestamp_micros"]:
                        raise base.IntegrityError("original raw frame timestamps are not strictly increasing")
                    metadata.append(info)
                    diff = base.np.abs(targets - array)
                    full_columns.append(diff.mean(axis=(1, 2, 3)) / 255)
                    variation_columns.append(diff[:, variation_mask, :].mean(axis=(1, 2)) / 255)
                    if (index + 1) % 10 == 0:
                        self.state["identity_cpu_progress"] = {"raw_frames": index + 1, "file_bytes_read": stream.tell(),
                            "file_bytes_total": row["size"], "network_requests": 0}
                        self.save()
                        print(json.dumps({"phase": "local_raw_alignment", "raw_frames": index + 1,
                            "bytes_read": stream.tell(), "network_requests": 0}), flush=True)
            end_stat = raw.stat()
            if start_stat.st_size != end_stat.st_size or start_stat.st_mtime_ns != end_stat.st_mtime_ns:
                raise base.IntegrityError("attested raw file changed during the sequential CPU scan")
            if len(metadata) < 21:
                raise base.IntegrityError("source has fewer than21 frames")
            base.atomic_json(self.identity_root / "raw_frame_metadata.json", {"global_training_index": INDEX,
                "segment": SEGMENT, "pinned_revision": base.REV, "raw_frame_count": len(metadata),
                "all_record_length_crc32c": "passed", "all_record_payload_crc32c": "passed",
                "all_contexts": "passed", "frames": metadata, "raw_labels_used": False})
            return base.np.stack(full_columns, axis=1), base.np.stack(variation_columns, axis=1), metadata, int(variation_mask.sum())

        def publish(self, raw, receipt, full, variation, metadata, mask_pixels):
            raw_frames = [r["raw_frame"] for r in metadata]
            combined, independent, margins, sequences, consistent, low_error = score_sequences(full, variation, raw_frames, base.np)
            best = sequences[0]
            base.np.save(self.identity_root / "all_video_to_raw_rgb_mae.npy", full)
            base.np.save(self.identity_root / "all_video_to_raw_combined_mae.npy", combined)
            records, comparisons = [], []
            directory = self.identity_root / "comparisons"
            directory.mkdir(exist_ok=True)
            for i, raw_index in enumerate(best["raw_frames"]):
                with base.Image.open(self.identity_root / "raw_front" / ("raw%03d.png" % raw_index)) as im:
                    raw_image = im.convert("RGB")
                panel = base.Image.new("RGB", (960, 348), (24, 27, 32))
                panel.paste(base.Image.fromarray(self.references[i]), (0, 28))
                panel.paste(raw_image, (480, 28))
                draw = base.ImageDraw.Draw(panel)
                draw.text((8, 6), "Author video %03d" % i, fill="white")
                draw.text((488, 6), "Raw FRONT %03d | MAE %.5f" % (raw_index, full[i, raw_index]), fill="white")
                path = directory / ("video%03d_raw%03d.png" % (i, raw_index))
                panel.save(path)
                comparisons.append(str(path))
                records.append({"video_frame": i, "raw_frame": raw_index, "raw_timestamp_micros": metadata[raw_index]["timestamp_micros"],
                    "rgb_mae": float(full[i, raw_index]), "temporal_variation_region_mae": float(variation[i, raw_index]),
                    "independent_best_raw_frame": independent[i], "independent_best_margin": margins[i],
                    "raw_metadata_index": raw_index, "comparison": str(path)})
            for page in range(7):
                sheet = base.Image.new("RGB", (960, 1044), (24, 27, 32))
                for slot in range(3):
                    with base.Image.open(comparisons[page * 3 + slot]) as image:
                        sheet.paste(image, (0, slot * 348))
                sheet.save(directory / ("all21_page%02d.png" % page))
            start, end = best["raw_start"], best["raw_end_inclusive"]
            report = {"schema_version": 1, "status": "computed_requires_root_visual_review",
                "global_training_index": INDEX, "segment": SEGMENT, "relative_path": self.files[INDEX]["path"],
                "pinned_revision": base.REV, "camera": "FRONT", "raw_path": str(raw),
                "raw_size_bytes": raw.stat().st_size, "full_raw_receipt": str(self.options.raw_receipt.resolve()),
                "full_raw_sha256": receipt["raw_full_sha256"], "checksum_verified_by_download_receipt_not_rehashed_here": True,
                "raw_frame_count": len(metadata), "all_record_crc32c_and_context": "passed",
                "raw_pose_and_front_exposure_metadata": str(self.identity_root / "raw_frame_metadata.json"),
                "official_filename": self.reference["official_filename"], "official_asset_url": self.reference["official_asset_url"],
                "video_frame_count": 21, "video_fps_rational": "8/1", "video_duration_seconds": 2.625,
                "matching_resolution_wh": [480, 320], "excluded_top_pixels": 28,
                "matching_method": "RGB MAE and a fixed50/50 RGB/author-temporal-variation-region score; original unwarped FRONT pixels. Exhaustive all21 video frames against every raw frame; candidate stride1/2 sequences.",
                "temporal_variation_region_threshold_8bit": 8, "temporal_variation_region_pixels": mask_pixels,
                "independent_best_raw_frames": independent, "unique_minimum_all21": min(margins) > 1e-6,
                "minimum_runner_up_margin": min(margins), "candidate_sequences_top10": sequences,
                "best_raw_frames": best["raw_frames"], "stride": best["stride"],
                "all_video_frames_matched": consistent and low_error, "numerical_temporal_consistency": consistent,
                "numerical_low_error_condition": low_error,
                "raw_frame_range_inclusive": [start, end] if consistent and low_error else None,
                "candidate_raw_frame_range_inclusive": [start, end],
                "raw_clip_span_seconds_candidate": (metadata[end]["timestamp_micros"] - metadata[start]["timestamp_micros"]) / 1e6,
                "mean_rgb_mae": best["mean_rgb_mae"], "max_rgb_mae": best["max_rgb_mae"],
                "matches": records, "first_mid_last_comparisons": [comparisons[i] for i in (0, 10, 20)],
                "all21_comparisons": comparisons, "visual_review_required": True, "identity_verified": False,
                "author_training_window": None, "network_requests": 0,
                "boundary": "Segment location/actor sequence and exact source frames are separate claims. Numbers propose a window; root must inspect first/mid/last and all21 before recording identity acceptance. Video playback8fps does not establish raw acquisition fps or author training window."}
            base.atomic_json(self.identity_root / "frame_match.json", report)
            self.state.update(status="scene29_alignment_computed_requires_visual_review",
                identity_alignment_receipt=str(self.identity_root / "frame_match.json"),
                identity_candidate_raw_range=[start, end], identity_numeric_temporal_consistency=consistent,
                other_segment_search_paused_for_identity=True)
            self.save()
            return report
    return AlignmentJob


def main():
    args = arguments()
    job, base = None, None
    try:
        base, temporal = load_tools(args)
        manifest, files, references = read_reference(args, base)
        row = base.load_index(args.source_index)[INDEX]
        if row["segment"] != SEGMENT or Path(row["path"]).name != FILENAME:
            raise base.IntegrityError("pinned training index29 differs from the reviewed scene candidate")
        raw, receipt = validate_raw(args, row, base, args.run)
        if not args.run:
            print(json.dumps({"status": "PLAN_ONLY_NO_NETWORK_NO_SOURCE_SCAN", "global_training_index": INDEX,
                "segment": SEGMENT, "reference_frames": 21, "reference_fps": "8/1",
                "raw_path": str(raw), "raw_received": raw.is_file(),
                "raw_receipt_status": receipt.get("status") if receipt else None,
                "max_raw_frames": args.max_raw_frames, "output_root": str(args.output_root / "identity/index029"),
                "scope": "Same existing ledger/lock,40GiB floor512MiB outputs; zero new network; one complete source only.",
                "identity_verified": False}), flush=True)
            return 0
        inherited = temporal.inherited_arguments(SimpleNamespace(**dict(vars(args), cached_review_complete=True, new_review_complete=False)))
        inherited.mode = "identity_local_alignment"
        job = job_class(base)(inherited, args, manifest, references)
        before = job.state["network_reserved_bytes"]
        full, variation, metadata, pixels = job.scan_local(raw, row)
        if job.state["network_reserved_bytes"] != before:
            raise base.IntegrityError("pureCPU local alignment unexpectedly changed the network ledger")
        report = job.publish(raw, receipt, full, variation, metadata, pixels)
        print(json.dumps({"status": report["status"], "receipt": str(job.identity_root / "frame_match.json"),
            "candidate_raw_range": report["candidate_raw_frame_range_inclusive"], "stride": report["stride"],
            "numeric_temporal_consistency": report["numerical_temporal_consistency"], "mean_rgb_mae": report["mean_rgb_mae"],
            "network_requests": 0, "identity_verified": False}), flush=True)
        return 0
    except Exception as error:
        summary = base.safe_error(error) if base is not None else {"type": type(error).__name__}
        if isinstance(error, BlockingIOError):
            summary["reason"] = "shared search.lock busy; no state change or raw scan"
        if job is not None:
            job.state.update(status="identity_alignment_stopped", error=summary)
            job.save()
        print(json.dumps({"status": "identity_alignment_stopped", "error": summary, "network_requests": 0}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

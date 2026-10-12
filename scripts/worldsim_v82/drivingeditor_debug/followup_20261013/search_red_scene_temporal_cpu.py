#!/usr/bin/env python3
"""在冻结v1的同一5GiB账本内，对显式少量候选做TFRecord时间抽样。

只读每帧12-byte header，按length+16跳过未选payload；默认只读取
raw0/25/50/75/100/125/150/175的body。最多累计3个候选、每段200个header、
8个选定body；只写temporal/子目录与共用状态，不修改v1或首record receipts。
"""
import argparse
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import search_red_scene_cpu as base


def bounded_csv(value, upper, maximum):
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("use an explicit comma-separated integer list")
    if not result or len(result) > maximum or len(set(result)) != len(result) or any(i < 0 or i >= upper for i in result):
        raise argparse.ArgumentTypeError("indices are duplicated or outside the finite allowed scope")
    return result


def inherited_arguments(options):
    # 只读现有状态。随后Search先取同一flock，忙时不写状态也不发请求。
    root = options.output_root.resolve()
    marker = json.loads((root / base.OWNER_FILE).read_text())
    state = json.loads((root / "status.json").read_text())
    if marker.get("owner") != base.VERSION or state.get("owner") != base.VERSION:
        raise base.IntegrityError("temporal extension requires the existing frozen-v1 owned search root")
    cap = state["contract"]
    if cap["network_cap_bytes"] != 5 * base.GIB or cap["disk_free_floor_bytes"] < 40 * base.GIB or cap["max_output_bytes"] > 512 * base.MIB or cap["max_new_segments_total"] != 435 or cap["workers"] != 1:
        raise base.IntegrityError("existing shared contract differs from the authorized 5GiB/40GiB/512MiB/435/1 scope")
    # 所有caps原值继承，不升级旧账、不把header或body请求分到另一预算。
    return SimpleNamespace(mode="temporal", project_root=options.project_root, raw_root=options.raw_root,
        source_index=options.source_index, output_root=options.output_root,
        opencv_pythonpath=options.opencv_pythonpath,
        network_cap_gib=cap["network_cap_bytes"] / base.GIB,
        disk_free_floor_gib=cap["disk_free_floor_bytes"] / base.GIB,
        max_output_mib=cap["max_output_bytes"] // base.MIB,
        max_new_segments=cap["max_new_segments_total"], max_new_this_invocation=100,
        max_record_mib=cap["max_single_record_bytes"] // base.MIB,
        max_seconds=cap["wall_time_limit_seconds_per_invocation"],
        request_timeout_seconds=cap["request_timeout_seconds"], attempts_per_range=cap["attempts_per_range"],
        inter_record_delay_seconds=0.25, cached_review_complete=options.cached_review_complete,
        new_review_complete=options.new_review_complete)


class TemporalSearch(base.Search):
    def __init__(self, args, options):
        super().__init__(args)
        self.options = options
        self.temporal_root = self.root / "temporal"
        self.temporal_root.mkdir(exist_ok=True)
        for name in ("walk", "records", "previews"):
            (self.temporal_root / name).mkdir(exist_ok=True)

    def write_temporal_ranking(self, rows, index):
        # v1的排序函数只写self.root；临时选用temporal/，共用state_path保持原值。
        shared_root = self.root
        try:
            self.root = self.temporal_root
            return base.Search.write_ranking(self, rows, f"index{index:03d}")
        finally:
            self.root = shared_root

    def validate_header(self, row, index, offset, header):
        if len(header) != 12 or self.masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
            raise base.IntegrityError("temporal length header CRC32C mismatch")
        length = struct.unpack("<Q", header[:8])[0]
        next_offset = offset + length + 16
        if length <= 0 or length + 16 > self.contract["max_single_record_bytes"] or next_offset > row["size"]:
            raise base.IntegrityError("temporal record exceeds allowed size or pinned full-file bounds")
        return {"raw_frame": index, "header_offset": offset, "payload_length": length,
            "next_record_offset": next_offset, "length_crc32c": "passed"}

    def selected_body(self, row, record):
        index, offset, length = record["raw_frame"], record["header_offset"], record["payload_length"]
        receipt_path = self.temporal_root / "records" / row["segment"] / f"raw{index:03d}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            expected = {"status": "crc_verified_temporal_selected_record", "segment": row["segment"],
                "raw_frame": index, "header_offset": offset, "payload_length": length,
                "next_record_offset": record["next_record_offset"], "relative_path": row["path"],
                "global_training_index": row["global_training_index"],
                "pinned_revision": base.REV, "length_crc32c": "passed", "payload_crc32c": "passed", "protobuf_context": "passed"}
            if any(receipt.get(k) != v for k, v in expected.items()) or len(receipt.get("camera_previews", [])) != 5:
                raise base.IntegrityError("saved temporal selected-body receipt differs from the bounded header walk")
            for camera in receipt["camera_previews"]:
                path = Path(camera["path"]).resolve()
                if not path.is_relative_to(self.temporal_root.resolve()) or not path.is_file():
                    raise base.IntegrityError("saved temporal preview is missing or outside the owned subdirectory")
            return receipt
        from waymo_open_dataset import dataset_pb2
        # 仅此处读取选定record的payload+footer；未选帧不发payload请求。
        rest, http = self.range_request(row, offset + 12, offset + length + 15)
        if self.masked_crc(rest[:-4]) != struct.unpack("<I", rest[-4:])[0]:
            raise base.IntegrityError("temporal selected-body payload CRC32C mismatch")
        frame = dataset_pb2.Frame.FromString(rest[:-4])
        if frame.context.name != row["segment"] or {image.name for image in frame.images} != set(base.CAM):
            raise base.IntegrityError("temporal selected-body protobuf context or camera set mismatch")
        preview = self.temporal_root / "previews" / row["segment"] / f"raw{index:03d}"
        preview.mkdir(parents=True, exist_ok=True)
        cameras = []
        for camera in sorted(frame.images, key=lambda x: x.name):
            path = preview / f"camera{camera.name}.jpg"
            with base.Image.open(base.io.BytesIO(camera.image)) as image:
                image.draft("RGB", (240, 160))
                image.convert("RGB").resize((240, 160), base.Image.Resampling.BOX).save(path, quality=90)
            cameras.append({"path": str(path), "segment": row["segment"],
                "global_training_index": row["global_training_index"], "camera_enum": camera.name,
                "raw_frame": index, "source": "temporal_crc_verified_selected_record"})
        receipt = {"status": "crc_verified_temporal_selected_record", "segment": row["segment"],
            "global_training_index": row["global_training_index"], "relative_path": row["path"],
            "pinned_revision": base.REV, **record, "payload_crc32c": "passed", "protobuf_context": "passed",
            "timestamp_micros": frame.timestamp_micros, "location": frame.context.stats.location,
            "weather": frame.context.stats.weather, "time_of_day": frame.context.stats.time_of_day,
            "http_body_range": http, "camera_previews": cameras, "full_raw_downloaded": False,
            "identity_verified": False}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        base.atomic_json(receipt_path, receipt)
        return receipt

    def walk_segment(self, row):
        path = self.temporal_root / "walk" / f"{row['segment']}.json"
        if path.is_file():
            walk = json.loads(path.read_text())
            if walk.get("segment") != row["segment"] or walk.get("pinned_revision") != base.REV or walk.get("sample_frames") != self.options.sample_frames or walk.get("max_header_frames") != self.options.max_frames:
                raise base.IntegrityError("saved temporal walk differs from the declared finite plan")
        else:
            walk = {"segment": row["segment"], "global_training_index": row["global_training_index"],
                "pinned_revision": base.REV, "sample_frames": self.options.sample_frames,
                "max_header_frames": self.options.max_frames, "headers": [], "body_frames_completed": [],
                "status": "walking_headers", "identity_verified": False}
        headers = walk["headers"]
        if len(headers) > self.options.max_frames:
            raise base.IntegrityError("saved temporal walk already exceeds its header limit")
        walk["status"] = "walking_headers"
        rows, offset = [], 0
        last_selected = max(self.options.sample_frames)
        for index in range(min(self.options.max_frames, last_selected + 1)):
            self.check_limits()
            if offset == row["size"]:
                walk["status"] = "source_eof_before_remaining_selected_frames"
                walk["header_inferred_frame_count"] = index
                break
            if offset + 12 > row["size"]:
                raise base.IntegrityError("temporal next header crosses pinned file end")
            if index < len(headers):
                record = headers[index]
                if record.get("raw_frame") != index or record.get("header_offset") != offset or record.get("length_crc32c") != "passed":
                    raise base.IntegrityError("saved temporal header chain has a gap or invalid boundary")
                if record["next_record_offset"] != offset + record["payload_length"] + 16 or record["payload_length"] <= 0 or record["payload_length"] + 16 > self.contract["max_single_record_bytes"] or record["next_record_offset"] > row["size"]:
                    raise base.IntegrityError("saved temporal header chain exceeds record or file bounds")
            else:
                # 12-byte请求和任何重试均由v1先写同一network reservation。
                header, http = self.range_request(row, offset, offset + 11)
                record = {**self.validate_header(row, index, offset, header), "http_header_range": http}
                headers.append(record)
                base.atomic_json(path, walk)
            if index in self.options.sample_frames:
                receipt = self.selected_body(row, record)
                stamps = walk.setdefault("sampled_timestamps_micros", {})
                previous = [int(value) for key, value in stamps.items() if int(key) < index]
                if previous and receipt["timestamp_micros"] <= max(previous):
                    raise base.IntegrityError("temporal sampled frame timestamps are not strictly increasing")
                stamps[str(index)] = receipt["timestamp_micros"]
                for camera in receipt["camera_previews"]:
                    rows.extend(self.compare(camera))
                if index not in walk["body_frames_completed"]:
                    walk["body_frames_completed"].append(index)
                base.atomic_json(path, walk)
                self.write_temporal_ranking(rows, row["global_training_index"])
                acknowledged = {(x["segment"], x["raw_frame"]) for x in self.state.get("temporal_review_command_acknowledgements", [])}
                signals = sum(x["review_signal"] and (x["segment"], x["raw_frame"]) not in acknowledged for x in rows)
                print(json.dumps({"phase": "temporal_selected_body", "index": row["global_training_index"],
                    "raw_frame": index, "headers_walked": len(headers), "selected_bodies": len(walk["body_frames_completed"]),
                    "network_reserved_bytes": self.state["network_reserved_bytes"], "review_signal_comparisons": signals}), flush=True)
                if signals:
                    walk["status"] = "candidate_signal_stop_requires_visual_review"
                    self.state.update(status=walk["status"], new_candidate_review_pending=True,
                        temporal_candidate_review_pending=True,
                        temporal_candidate={"index": row["global_training_index"], "raw_frame": index, "segment": row["segment"]})
                    base.atomic_json(path, walk)
                    self.save()
                    return False
            if (index + 1) % 20 == 0:
                self.state["temporal_progress"] = {"index": row["global_training_index"],
                    "headers_walked": len(headers), "last_raw_frame": index, "selected_bodies": len(walk["body_frames_completed"])}
                self.save()
            # 跳过未选record的整个payload+footer；不read、不缓存wholefile。
            offset = record["next_record_offset"]
        if walk["status"] == "walking_headers":
            walk["status"] = "finite_temporal_samples_complete_requires_visual_review"
        walk["limitation"] = "Skipped records have length-header CRC only; sampled bodies have payload CRC and context. No full-file verification, exact identity or author training-window claim."
        base.atomic_json(path, walk)
        self.write_temporal_ranking(rows, row["global_training_index"])
        return True

    def temporal_search(self):
        if not self.state.get("cached_pass_completed") or not self.args.cached_review_complete:
            raise base.StopSearch("cached ranking must have completed and been reviewed before temporal requests")
        if not self.state.get("new_control_passed"):
            raise base.StopSearch("the frozen-v1 new mirror positive control must pass first")
        if (self.state.get("new_candidate_review_pending") or self.state.get("temporal_candidate_review_pending")) and not self.args.new_review_complete:
            raise base.StopSearch("a pending candidate must be visually reviewed before any more network requests")
        if self.args.new_review_complete:
            candidate = self.state.get("temporal_candidate")
            if candidate:
                self.state.setdefault("temporal_review_command_acknowledgements", []).append({**candidate,
                    "acknowledged_utc": base.utc(), "semantics": "Root command acknowledges review; no identity verdict."})
            self.state.update(new_candidate_review_pending=False, temporal_candidate_review_pending=False)
        plan = {"max_segments_total": self.options.max_segments, "max_header_frames_per_segment": self.options.max_frames,
            "sample_frames": self.options.sample_frames, "max_selected_bodies_per_segment": 8}
        previous_plan = self.state.get("temporal_contract")
        if previous_plan is not None and previous_plan != plan:
            raise base.IntegrityError("do not silently extend or reset the saved temporal scope")
        seen = set(self.state.get("temporal_indices_started", []))
        if len(seen | set(self.options.indices)) > self.options.max_segments:
            raise base.StopSearch("cumulative temporal candidate-segment limit reached")
        self.state["temporal_contract"] = plan
        self.state["temporal_indices_started"] = sorted(seen | set(self.options.indices))
        reasons = self.state.setdefault("temporal_selection_reasons", {})
        for index in self.options.indices:
            reasons[str(index)] = self.options.reason
        self.state["status"] = "walking_explicit_bounded_temporal_candidates"
        self.save()
        self.setup_features()
        self.setup_crc()
        if "torch" in sys.modules or "tensorflow" in sys.modules:
            raise base.IntegrityError("GPU/training framework unexpectedly imported")
        for index in self.options.indices:
            if not self.walk_segment(self.files[index]):
                return
        self.state["status"] = "finite_temporal_search_complete_requires_visual_review"
        self.save()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro"))
    p.add_argument("--raw-root", type=Path, default=Path("/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files"))
    p.add_argument("--source-index", type=Path, default=Path("/root/autodl-tmp/dggt/.preparation/hf_waymo_training_tree.json"))
    p.add_argument("--opencv-pythonpath", type=Path, default=Path("/root/autodl-tmp/envs/motionproj/lib/python3.10/site-packages"))
    p.add_argument("--indices", type=lambda v: bounded_csv(v, 798, 3), required=True)
    p.add_argument("--sample-frames", type=lambda v: sorted(bounded_csv(v, 200, 8)), default=[0, 25, 50, 75, 100, 125, 150, 175])
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--max-segments", type=int, default=3)
    p.add_argument("--reason", required=True)
    p.add_argument("--cached-review-complete", action="store_true")
    p.add_argument("--new-review-complete", action="store_true")
    options = p.parse_args()
    if not (1 <= options.max_frames <= 200 and 1 <= options.max_segments <= 3 and len(options.indices) <= options.max_segments):
        p.error("temporal scope exceeds the finite 200-frame/3-segment cap")
    if 0 not in options.sample_frames or max(options.sample_frames) >= options.max_frames:
        p.error("sample frames must include raw0 and remain below max-frames")
    if not options.reason.strip() or len(options.reason) > 1000:
        p.error("record a concise visual reason for the explicitly chosen candidates")
    return options


def main():
    options = arguments()
    job = None
    try:
        job = TemporalSearch(inherited_arguments(options), options)
        job.temporal_search()
        print(json.dumps({"status": job.state["status"], "indices": options.indices,
            "shared_network_reserved_bytes": job.state["network_reserved_bytes"],
            "temporal_output_root": str(job.temporal_root), "identity_verified": False}), flush=True)
        return 0
    except Exception as error:
        summary = base.safe_error(error)
        if isinstance(error, BlockingIOError):
            summary["reason"] = "shared search.lock is busy; no state changes or network requests"
        if job is not None:
            job.state.update(status="temporal_stopped_bounded", error=summary)
            job.save()
        print(json.dumps({"status": "temporal_stopped_bounded", "error": summary}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""复用红车场景缓存，再进行独立、有限的 Waymo 首 record CPU 搜索。

三个入口：inventory 只整理清单，cached 重排已有图片，preview 请求新首 record。
只写 --output-root 所属的新目录；不写旧搜索账本、原始数据或训练目录。
所有候选都需要视觉及全视频时间对应验证，本脚本不宣告原始场景身份。
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import ctypes
import datetime
import io
import json
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VERSION = "red_scene_cpu_20261013_v1"
REPO = "AnnaZhang/waymo_open_dataset_v_1_4_3"
REV = "8a6fccd2b022077308c167a4d3c6f646d6e5e65b"
CLIP = "20241214-005533"
MIRROR = "https://hf-mirror.com"
CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
GIB = 1024 ** 3
MIB = 1024 ** 2
OWNER_FILE = "owned_search.json"


class StopSearch(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def segment_name(path):
    return Path(path).name.removeprefix("segment-").removesuffix("_with_camera_labels.tfrecord")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_error(error):
    # 不记录异常 repr、redirect URL、请求 headers 或认证内容。
    row = {"type": type(error).__name__}
    if isinstance(error, urllib.error.HTTPError):
        row["http_status"] = error.code
    elif isinstance(error, (StopSearch, IntegrityError)):
        row["reason"] = str(error)
    return row


def load_index(path):
    files = sorted((r for r in json.loads(path.read_text()) if r["path"].endswith(".tfrecord")), key=lambda r: r["path"])
    if len(files) != 798:
        raise IntegrityError("pinned training index does not contain 798 files")
    for i, row in enumerate(files):
        if not row["path"].startswith("individual_files/training/segment-"):
            raise IntegrityError("unexpected public training path")
        if not re.fullmatch(r"[0-9a-f]{64}", row["lfs"]["oid"]) or row["size"] != row["lfs"]["size"]:
            raise IntegrityError("pinned index LFS metadata invalid")
        row["global_training_index"] = i
        row["segment"] = segment_name(row["path"])
    return files


def collect_inventory(project, raw, files):
    old_mirror = project / "campaign/scene_search_mirror"
    old_scan = project / "campaign/scene_search"
    by_name = {row["segment"]: row for row in files}
    cached_names, candidates, seen = set(), [], set()
    receipt_count = 0
    for p in sorted((old_mirror / "records").glob("*.json")):
        item = json.loads(p.read_text())
        expected = by_name.get(item.get("segment"))
        if expected is None or item.get("pinned_revision") != REV:
            raise IntegrityError("legacy first-record provenance mismatch")
        if item.get("raw_expected_full_sha256") != expected["lfs"]["oid"]:
            raise IntegrityError("legacy first-record expected identity mismatch")
        if item.get("status") != "crc_verified_first_record_preview" or any(item.get(k) != "passed" for k in ("length_crc32c", "payload_crc32c", "protobuf_context")):
            raise IntegrityError("legacy first-record receipt not validated")
        if len(item.get("camera_previews", [])) != 5:
            raise IntegrityError("legacy first-record does not have five camera previews")
        cached_names.add(item["segment"])
        receipt_count += 1
        for camera in item["camera_previews"]:
            image = (old_mirror / camera["thumbnail"]).resolve()
            if not image.is_relative_to(old_mirror.resolve()) or not image.is_file():
                raise IntegrityError("legacy thumbnail missing or outside its cache")
            key = (item["segment"], camera["camera_enum"], 0)
            seen.add(key)
            candidates.append({"path": str(image), "segment": item["segment"], "global_training_index": expected["global_training_index"], "camera_enum": camera["camera_enum"], "raw_frame": 0, "source": "legacy_verified_first_record"})
    fullscan_count = 0
    for p in sorted((old_scan / "candidates").glob("*/*.jpg")):
        match = re.fullmatch(r"cam([1-5])_frame([0-9]+)\.jpg", p.name)
        if match is None:
            continue
        fullscan_count += 1
        name = p.parent.name
        camera, frame = map(int, match.groups())
        key = (name, camera, frame)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"path": str(p.resolve()), "segment": name, "global_training_index": by_name.get(name, {}).get("global_training_index"), "camera_enum": camera, "raw_frame": frame, "source": "legacy_fullscan_retrieved_candidate"})
    raw_paths = sorted(raw.glob("*/*.tfrecord"))
    raw_names = {segment_name(p) for p in raw_paths}
    remaining = [r for r in files if r["segment"] not in raw_names | cached_names]
    report = {"created_utc": utc(), "revision": REV, "public_repository": REPO, "public_mirror_domain": urllib.parse.urlparse(MIRROR).hostname, "training_segment_count": len(files), "local_raw_count_all_splits": len(raw_paths), "local_raw_training_count": len(list((raw / "training").glob("*.tfrecord"))), "local_raw_validation_count": len(list((raw / "validation").glob("*.tfrecord"))), "legacy_validated_first_record_receipts": receipt_count, "legacy_first_record_images": receipt_count * 5, "legacy_fullscan_candidate_images": fullscan_count, "unique_cached_images_for_retrieval": len(candidates), "cached_and_local_segment_overlap": len(raw_names & cached_names), "remaining_unpreviewed_not_local_count": len(remaining), "remaining": [{k: r[k] for k in ("global_training_index", "segment", "path", "size")} for r in remaining], "limitations": ["Legacy fullscan previews are retrieved candidate frames, not every image of its 42 complete TFRecords.", "First-frame weak matches cannot exclude later frames of any segment.", "Pinned mirror metadata is third-party provenance; raw-scene identity needs temporal and visual verification."]}
    return report, candidates, remaining


class Search:
    def __init__(self, args):
        self.args = args
        self.project, self.raw, self.root = args.project_root.resolve(), args.raw_root.resolve(), args.output_root.resolve()
        self.started = time.monotonic()
        protected = [self.raw, self.project / "campaign/scene_search", self.project / "campaign/scene_search_mirror", self.project / "data", self.project / "runs", self.project / "DrivingEditor", self.project / "campaign/next_scene30k_20261012", self.project / "campaign/hybrid30k_20261012"]
        campaign = self.project / "campaign"
        if self.root == campaign or not self.root.is_relative_to(campaign):
            raise IntegrityError("output-root must be a new descendant of project campaign")
        if any(self.root == p or self.root.is_relative_to(p) or p.is_relative_to(self.root) for p in protected):
            raise IntegrityError("output-root overlaps preserved data, caches or training")
        owner = self.root / OWNER_FILE
        if self.root.exists() and not owner.exists() and any(self.root.iterdir()):
            raise IntegrityError("refuse an existing nonempty output-root without this script's owner marker")
        self.root.mkdir(parents=True, exist_ok=True)
        if owner.exists():
            if json.loads(owner.read_text()).get("owner") != VERSION:
                raise IntegrityError("output-root has another owner")
        else:
            atomic_json(owner, {"owner": VERSION, "created_utc": utc(), "scope": "New authorized capped search; legacy 4GiB ledger remains untouched."})
        import fcntl
        self.lock = (self.root / "search.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.root / "status.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"owner": VERSION, "created_utc": utc(), "network_reserved_bytes": 0, "network_received_bytes": 0, "completed_new_segments": []}
        if self.state.get("owner") != VERSION:
            raise IntegrityError("state owner mismatch")
        self.previous_status = self.state.get("status")
        self.contract = {"network_cap_bytes": int(args.network_cap_gib * GIB), "disk_free_floor_bytes": int(args.disk_free_floor_gib * GIB), "max_output_bytes": args.max_output_mib * MIB, "max_new_segments_total": args.max_new_segments, "max_single_record_bytes": args.max_record_mib * MIB, "request_timeout_seconds": args.request_timeout_seconds, "attempts_per_range": args.attempts_per_range, "workers": 1, "wall_time_limit_seconds_per_invocation": args.max_seconds}
        if "contract" in self.state and self.state["contract"] != self.contract:
            raise IntegrityError("saved search caps differ; do not reset or silently extend this ledger")
        self.state["contract"] = self.contract
        self.state.update(pid=os.getpid(), mode=args.mode, status="starting", gpu_used=False, cuda_visible_devices="-1")
        self.save()
        self.files = load_index(args.source_index)
        self.inventory, self.cached, self.remaining = collect_inventory(self.project, self.raw, self.files)
        atomic_json(self.root / "inventory.json", self.inventory)
        self.cv2 = None
        self.targets = None

    def save(self):
        self.state["updated_utc"] = utc()
        atomic_json(self.state_path, self.state)

    def check_limits(self):
        if time.monotonic() - self.started > self.args.max_seconds:
            raise StopSearch("wall time limit reached")
        if shutil.disk_usage(self.root).free < self.contract["disk_free_floor_bytes"]:
            raise StopSearch("free disk fell below configured floor")
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.state["output_bytes"] = size
        if size + self.contract["max_single_record_bytes"] > self.contract["max_output_bytes"]:
            raise StopSearch("output budget reserve reached")

    def setup_features(self):
        if self.cv2 is not None:
            return
        if self.args.opencv_pythonpath:
            # 追加现有只读包路径，保留当前CPU环境的 NumPy、PIL、protobuf 优先级。
            sys.path.append(str(self.args.opencv_pythonpath))
        import cv2
        cv2.setNumThreads(1)
        self.cv2 = cv2
        self.sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.targets = []
        for index in (0, 10, 20):
            path = self.project / f"campaign/scene_search/targets/{CLIP}_{index:03d}.png"
            array = self.array(path)
            kp, desc = self.features(array)
            self.targets.append({"video_frame": index, "array": array, "kp": kp, "desc": desc})
        silver = self.project / "campaign/scene_search/targets/20241214-010221_000.png"
        control_name = self.files[17]["segment"]
        control_image = self.project / f"campaign/scene_search_mirror/previews/{control_name}/camera1.jpg"
        mae = float(np.abs(self.array(silver).astype(np.int16) - self.array(control_image).astype(np.int16)).mean() / 255)
        if mae >= 0.025:
            raise IntegrityError("existing cached silver positive control failed")
        self.state["cached_control_rgb_mae"] = mae
        self.state["cv2_version"] = cv2.__version__
        self.save()

    @staticmethod
    def array(path):
        with Image.open(path) as image:
            return np.array(image.convert("RGB").resize((240, 160), Image.Resampling.BOX))[14:]

    def features(self, array):
        return self.sift.detectAndCompute(self.cv2.cvtColor(array, self.cv2.COLOR_RGB2GRAY), None)

    def compare(self, item):
        array = self.array(item["path"])
        kp, desc = self.features(array)
        result = []
        for target in self.targets:
            mae = float(np.abs(array.astype(np.int16) - target["array"].astype(np.int16)).mean() / 255)
            metrics = {"sift_mutual_good": 0, "homography_inliers": 0, "target_spread": 0.0, "candidate_spread": 0.0}
            if desc is not None and target["desc"] is not None and len(desc) >= 2 and len(target["desc"]) >= 2:
                forward = self.matcher.knnMatch(target["desc"], desc, k=2)
                reverse = self.matcher.knnMatch(desc, target["desc"], k=2)
                rev = {(x.queryIdx, x.trainIdx) for pair in reverse if len(pair) == 2 for x, y in [pair] if x.distance < 0.7 * y.distance}
                good = [x for pair in forward if len(pair) == 2 for x, y in [pair] if x.distance < 0.7 * y.distance and (x.trainIdx, x.queryIdx) in rev]
                metrics["sift_mutual_good"] = len(good)
                if len(good) >= 4:
                    src = np.float32([target["kp"][x.queryIdx].pt for x in good])
                    dst = np.float32([kp[x.trainIdx].pt for x in good])
                    _, mask = self.cv2.findHomography(src, dst, self.cv2.RANSAC, 3.0)
                    if mask is not None:
                        selected = mask.reshape(-1).astype(bool)
                        metrics["homography_inliers"] = int(selected.sum())
                        if selected.sum() >= 4:
                            for points, name in ((src[selected], "target_spread"), (dst[selected], "candidate_spread")):
                                span = points.max(axis=0) - points.min(axis=0)
                                metrics[name] = float(span[0] * span[1] / (240 * 146))
            score = metrics["homography_inliers"] * min(metrics["target_spread"], metrics["candidate_spread"]) ** 0.5
            review = metrics["homography_inliers"] >= 25 and min(metrics["target_spread"], metrics["candidate_spread"]) >= 0.10 and (mae < 0.10 or metrics["homography_inliers"] >= 60)
            result.append({**item, "camera": CAM[item["camera_enum"]], "target_video_frame": target["video_frame"], "rgb_mae": mae, **metrics, "feature_retrieval_score": score, "review_signal": review, "identity_verified": False})
        return result

    def write_ranking(self, rows, prefix):
        def unique_top(key, reverse=False):
            seen, chosen = set(), []
            for row in sorted(rows, key=key, reverse=reverse):
                identity = (row["segment"], row["camera_enum"], row["raw_frame"])
                if identity not in seen:
                    seen.add(identity)
                    chosen.append(row)
                if len(chosen) >= 40:
                    break
            return chosen
        rgb = unique_top(lambda x: x["rgb_mae"])
        feature = unique_top(lambda x: (x["feature_retrieval_score"], x["homography_inliers"], -x["rgb_mae"]), reverse=True)
        signals = [r for r in rows if r["review_signal"]]
        atomic_json(self.root / f"{prefix}_ranking.json", {"status": "retrieval_requires_visual_review", "comparisons": len(rows), "rgb_top40": rgb, "feature_top40": feature, "review_signals": signals, "identity_verified": False, "limitation": "Scores and stop signals retrieve candidates; they do not identify a scene or its temporal window."})
        for name, ranked in (("rgb", rgb), ("feature", feature)):
            sheet = Image.new("RGB", (720, max(1, len(ranked)) * 190), (24, 27, 32))
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(ranked):
                y = i * 190
                reference = self.project / f"campaign/scene_search/targets/{CLIP}_{row['target_video_frame']:03d}.png"
                with Image.open(reference) as im:
                    sheet.paste(im.convert("RGB").resize((240, 160)), (0, y + 24))
                with Image.open(row["path"]) as im:
                    sheet.paste(im.convert("RGB").resize((240, 160)), (242, y + 24))
                draw.text((4, y + 4), f"index {row['global_training_index']} | {row['camera']} | raw {row['raw_frame']} | video {row['target_video_frame']}", fill="white")
                draw.text((488, y + 30), f"MAE {row['rgb_mae']:.4f}", fill="white")
                draw.text((488, y + 50), f"mutual {row['sift_mutual_good']}", fill="white")
                draw.text((488, y + 70), f"inliers {row['homography_inliers']}", fill="white")
                draw.text((488, y + 90), f"spread {row['target_spread']:.2f}/{row['candidate_spread']:.2f}", fill="white")
                draw.text((488, y + 110), row["segment"][:30], fill="white")
            # 不拼接超过40行的大册；这两张图只用于候选复核。
            sheet.save(self.root / f"{prefix}_{name}_top40.jpg", quality=90)
        return len(signals)

    def cached_search(self):
        self.setup_features()
        self.state.update(status="ranking_all_cached_images", cached_pass_completed=False)
        self.save()
        rows = []
        with (self.root / "cached_comparisons.jsonl").open("w", encoding="utf-8") as stream:
            for i, item in enumerate(self.cached):
                self.check_limits()
                comparison = self.compare(item)
                rows.extend(comparison)
                for row in comparison:
                    stream.write(json.dumps(row) + "\n")
                if (i + 1) % 100 == 0:
                    self.state["cached_images_ranked"] = i + 1
                    self.save()
                    print(json.dumps({"phase": "cached", "images_ranked": i + 1, "total": len(self.cached)}), flush=True)
        signals = self.write_ranking(rows, "cached")
        self.state.update(status="cached_complete_requires_visual_review", cached_pass_completed=True, cached_images_ranked=len(self.cached), cached_review_signal_comparisons=signals)
        self.save()

    def setup_crc(self):
        library = self.project / "campaign/scene_search_mirror/libpreview_crc32c.so"
        self.crc_lib = ctypes.CDLL(str(library))
        self.crc_lib.preview_crc32c.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.crc_lib.preview_crc32c.restype = ctypes.c_uint32
        if self.crc_lib.preview_crc32c(b"123456789", 9) != 0xE3069283:
            raise IntegrityError("existing CRC32C library known-vector failure")

    def masked_crc(self, data):
        value = int(self.crc_lib.preview_crc32c(data, len(data)))
        return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    def range_request(self, row, start, end):
        size = end - start + 1
        for attempt in range(self.args.attempts_per_range):
            self.check_limits()
            if self.state["network_reserved_bytes"] + size > self.contract["network_cap_bytes"]:
                raise StopSearch("new search cumulative network reservation cap reached")
            # 每次尝试先持久化 reservation；失败或断线不退还、不重置。
            self.state["network_reserved_bytes"] += size
            self.save()
            url = f"{MIRROR}/datasets/{REPO}/resolve/{REV}/{row['path']}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "DrivingEditor-red-scene-bounded-cpu/1.0"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(request, timeout=self.args.request_timeout_seconds) as response:
                    expected = f"bytes {start}-{end}/{row['size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise IntegrityError("refuse body: HTTP status or exact Content-Range mismatch")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise IntegrityError("unexpected HTTP content encoding")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != size:
                        raise IntegrityError("HTTP Content-Length mismatch")
                    payload = response.read(size + 1)
                    self.state["network_received_bytes"] += len(payload)
                    self.save()
                    if len(payload) != size:
                        raise IntegrityError("short or excess HTTP range body")
                    metadata = {"http_status": 206, "content_range": expected, "final_host": urllib.parse.urlparse(response.geturl()).hostname}
                return payload, metadata
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                self.state["last_network_error"] = {**safe_error(error), "segment": row["segment"], "range": [start, end], "attempt": attempt + 1}
                self.save()
                if attempt + 1 == self.args.attempts_per_range:
                    raise
                time.sleep(attempt + 1)

    def fetch_first(self, row, positive=False):
        from waymo_open_dataset import dataset_pb2
        header, hm = self.range_request(row, 0, 11)
        length = struct.unpack("<Q", header[:8])[0]
        if self.masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
            raise IntegrityError("TFRecord length CRC32C mismatch")
        if length + 16 > self.contract["max_single_record_bytes"] or length + 16 > row["size"]:
            raise IntegrityError("first record exceeds configured cap or file size")
        rest, bm = self.range_request(row, 12, length + 15)
        payload = rest[:-4]
        if self.masked_crc(payload) != struct.unpack("<I", rest[-4:])[0]:
            raise IntegrityError("TFRecord payload CRC32C mismatch")
        frame = dataset_pb2.Frame.FromString(payload)
        if frame.context.name != row["segment"] or {im.name for im in frame.images} != set(CAM):
            raise IntegrityError("protobuf context or five-camera set mismatch")
        if positive:
            reference = self.raw / row["path"].removeprefix("individual_files/")
            with reference.open("rb") as stream:
                if stream.read(length + 16) != header + rest:
                    raise IntegrityError("new mirror control differs from locally verified raw first record")
        target = self.root / "previews" / row["segment"]
        target.mkdir(parents=True, exist_ok=True)
        cameras = []
        for camera in sorted(frame.images, key=lambda x: x.name):
            path = target / f"camera{camera.name}.jpg"
            with Image.open(io.BytesIO(camera.image)) as image:
                image.draft("RGB", (240, 160))
                image.convert("RGB").resize((240, 160), Image.Resampling.BOX).save(path, quality=90)
            cameras.append({"path": str(path), "segment": row["segment"], "global_training_index": row["global_training_index"], "camera_enum": camera.name, "raw_frame": 0, "source": "new_crc_verified_first_record"})
        if positive:
            silver = self.project / "campaign/scene_search/targets/20241214-010221_000.png"
            mae = float(np.abs(self.array(silver).astype(np.int16) - self.array(cameras[0]["path"]).astype(np.int16)).mean() / 255)
            if mae >= 0.025:
                raise IntegrityError("new mirror silver control RGB mismatch")
            self.state["new_control_rgb_mae"] = mae
        result = {"status": "crc_verified_first_record_preview", "positive_control": positive, "segment": row["segment"], "global_training_index": row["global_training_index"], "relative_path": row["path"], "pinned_revision": REV, "first_record_bytes": length + 16, "length_crc32c": "passed", "payload_crc32c": "passed", "protobuf_context": "passed", "timestamp_micros": frame.timestamp_micros, "location": frame.context.stats.location, "weather": frame.context.stats.weather, "time_of_day": frame.context.stats.time_of_day, "http_ranges": [hm, bm], "camera_previews": cameras, "full_raw_downloaded": False, "identity_verified": False}
        directory = self.root / "records"
        directory.mkdir(exist_ok=True)
        atomic_json(directory / f"{row['segment']}.json", result)
        return result

    def preview_search(self):
        if not self.state.get("cached_pass_completed") or not self.args.cached_review_complete:
            raise StopSearch("run cached mode, inspect its ranking and pass --cached-review-complete before new requests")
        if self.state.get("new_candidate_review_pending") and not self.args.new_review_complete:
            raise StopSearch("new candidate needs visual review; pass --new-review-complete only after review")
        if self.args.new_review_complete:
            self.state["new_candidate_review_pending"] = False
            self.save()
        self.setup_features()
        self.setup_crc()
        if "torch" in sys.modules or "tensorflow" in sys.modules:
            raise IntegrityError("GPU/training framework unexpectedly imported")
        # 新目录中已完成的合法 receipt 可恢复，但只读旧 ledger。
        rows, completed = [], set(self.state.get("completed_new_segments", []))
        for path in sorted((self.root / "records").glob("*.json")):
            item = json.loads(path.read_text())
            if item.get("pinned_revision") != REV or item.get("status") != "crc_verified_first_record_preview":
                raise IntegrityError("new search receipt invalid on resume")
            if item.get("positive_control"):
                continue
            completed.add(item["segment"])
            for camera in item["camera_previews"]:
                rows.extend(self.compare(camera))
        self.state["completed_new_segments"] = sorted(completed)
        if len(completed) >= self.args.max_new_segments:
            self.state["status"] = "finite_segment_limit_already_complete"
            self.save()
            return
        if not self.state.get("new_control_passed"):
            self.fetch_first(self.files[17], positive=True)
            self.state["new_control_passed"] = True
            self.save()
        self.state["status"] = "previewing_bounded_remaining_first_records"
        self.save()
        added_this_invocation = 0
        for row in self.remaining:
            if row["segment"] in completed:
                continue
            if len(completed) >= self.args.max_new_segments:
                break
            if added_this_invocation >= self.args.max_new_this_invocation:
                break
            self.check_limits()
            receipt = self.fetch_first(row)
            signals = 0
            for camera in receipt["camera_previews"]:
                comparison = self.compare(camera)
                rows.extend(comparison)
                signals += sum(x["review_signal"] for x in comparison)
            completed.add(row["segment"])
            added_this_invocation += 1
            self.state["completed_new_segments"] = sorted(completed)
            self.state["new_segments_this_invocation"] = added_this_invocation
            self.state["last_completed_segment"] = row["segment"]
            self.save()
            print(json.dumps({"phase": "preview", "completed_new_segments": len(completed), "index": row["global_training_index"], "network_reserved_bytes": self.state["network_reserved_bytes"], "record_bytes": receipt["first_record_bytes"], "review_signals": signals}), flush=True)
            if signals:
                self.state["status"] = "candidate_signal_stop_requires_visual_review"
                self.state["new_candidate_review_pending"] = True
                self.write_ranking(rows, "new")
                self.save()
                return
            if len(completed) % 20 == 0:
                self.write_ranking(rows, "new")
            time.sleep(self.args.inter_record_delay_seconds)
        self.write_ranking(rows, "new")
        if added_this_invocation >= self.args.max_new_this_invocation and len(completed) < min(self.args.max_new_segments, len(self.remaining)):
            self.state["status"] = "bounded_batch_complete_requires_visual_review"
        else:
            self.state["status"] = "finite_first_record_search_complete_requires_visual_review"
        self.save()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("inventory", "cached", "preview"), default="cached")
    p.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro"))
    p.add_argument("--raw-root", type=Path, default=Path("/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files"))
    p.add_argument("--source-index", type=Path, default=Path("/root/autodl-tmp/dggt/.preparation/hf_waymo_training_tree.json"))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--opencv-pythonpath", type=Path)
    p.add_argument("--network-cap-gib", type=float, default=5.0)
    p.add_argument("--disk-free-floor-gib", type=float, default=40.0)
    p.add_argument("--max-output-mib", type=int, default=512)
    p.add_argument("--max-new-segments", type=int, default=435)
    p.add_argument("--max-new-this-invocation", type=int, default=100)
    p.add_argument("--max-record-mib", type=int, default=16)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--request-timeout-seconds", type=int, default=60)
    p.add_argument("--attempts-per-range", type=int, default=2)
    p.add_argument("--inter-record-delay-seconds", type=float, default=0.25)
    p.add_argument("--cached-review-complete", action="store_true")
    p.add_argument("--new-review-complete", action="store_true")
    a = p.parse_args()
    if not (0 < a.network_cap_gib <= 10 and 30 <= a.disk_free_floor_gib <= 100 and 64 <= a.max_output_mib <= 1024 and 1 <= a.max_new_segments <= 798 and 1 <= a.max_new_this_invocation <= 798 and 1 <= a.max_record_mib <= 16 and 1 <= a.max_seconds <= 14400 and 1 <= a.request_timeout_seconds <= 75 and 1 <= a.attempts_per_range <= 3 and 0 <= a.inter_record_delay_seconds <= 10):
        p.error("one or more search caps are outside their bounded range")
    return a


def main():
    args = arguments()
    job = None
    try:
        job = Search(args)
        if args.mode == "inventory":
            job.state["status"] = "inventory_complete_no_network"
            job.save()
        elif args.mode == "cached":
            job.cached_search()
        else:
            job.preview_search()
        print(json.dumps({"status": job.state["status"], "output_root": str(job.root), "network_reserved_bytes": job.state["network_reserved_bytes"], "remaining_initial_count": len(job.remaining)}), flush=True)
        return 0
    except Exception as error:
        summary = safe_error(error)
        if job is not None:
            job.state.update(status="stopped_bounded", error=summary)
            job.save()
        print(json.dumps({"status": "stopped_bounded", "error": summary}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

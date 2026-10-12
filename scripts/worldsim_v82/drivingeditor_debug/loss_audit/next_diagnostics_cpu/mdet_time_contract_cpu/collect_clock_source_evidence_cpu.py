"""Bounded read-only source/clock audit. No detector imports, construction or scans."""
from pathlib import Path
import hashlib
import io
import json
import statistics
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
WORK = Path.cwd() / "work/vgpu_bugfix_20261012"
INIT = WORK / "loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu"
SRC = WORK / "preprocessing_gap_audit/mdetector_source"
REV = "0645dabe307a99f3b59a5a3d3007acfdcad1a39f"
ASSET_URL = "https://github.com/hku-mars/M-detector/releases/download/FAST_LIO/FAST_LIO.zip"
ASSET_SHA = "b8a81e567b0949447ff0ea73ae519fe5fd494340fd83c0f00e49dcb6f9f30954"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def get(url, bound):
    req = urllib.request.Request(url, headers={"User-Agent": "Codex-CPU-readonly-audit"})
    with urllib.request.urlopen(req, timeout=25) as response:
        data = response.read(bound + 1)
    if len(data) > bound:
        raise RuntimeError("source request exceeded its bounded size")
    return data

def pin(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha(data)}

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    canonical = {
        "raw49_labels_layout.json": "fa0448b99abc28c22f40fc7722ecb1d68dbcc68dbe88ae44b5aed0357b894806",
        "raw_field_availability_followup_cpu.json": "5eae46e7023aafc578517642c830c87543fdc256d75cfef4912e2301ef38da2b",
        "input_lineage_contract.json": "47dc9d6627b8a2630f65ece165d2a3b9d50e0a1b03f715f3b3048137a0be2f14",
    }
    for name, expected in canonical.items():
        if sha((INIT / name).read_bytes()) != expected:
            raise RuntimeError("frozen audit reference differs: " + name)
    layout = json.loads((INIT / "raw49_labels_layout.json").read_text())
    raw_fields = json.loads((INIT / "raw_field_availability_followup_cpu.json").read_text())
    times = [f["timestamp_micros"] for f in layout["frames"]]
    counts = [sum(c["points"] for c in f["chunks"]) for f in layout["frames"]]
    diffs = [b-a for a,b in zip(times, times[1:])]
    assert len(times) == 49 and all(d > 0 for d in diffs)

    releases_url = "https://api.github.com/repos/hku-mars/M-detector/releases"
    releases = json.loads(get(releases_url, 131072))
    release = next(r for r in releases if any(a["browser_download_url"] == ASSET_URL for a in r["assets"]))
    zip_bytes = get(ASSET_URL, 524288)
    assert len(zip_bytes) == 238366 and sha(zip_bytes) == ASSET_SHA
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    selected = ["FAST_LIO/config/waymo.yaml", "FAST_LIO/launch/mapping_waymo.launch",
                "FAST_LIO/src/preprocess.cpp", "FAST_LIO/src/laserMapping.cpp"]
    released_pins = []
    for name in selected:
        data = z.read(name)
        target = HERE / "released_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        released_pins.append({"archive_member": name, "bytes": len(data), "sha256": sha(data)})
    waymo_cfg = z.read(selected[0]).decode()
    mapping = z.read(selected[3]).decode()
    assert "imu_en: false" in waymo_cfg
    assert "lidar_end_time = meas.lidar_beg_time;" in mapping

    sdk_commit_url = "https://api.github.com/repos/waymo-research/waymo-open-dataset/commits?path=src/waymo_open_dataset/dataset.proto&per_page=1"
    sdk_revision = json.loads(get(sdk_commit_url, 131072))[0]["sha"]
    sdk_url = "https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/" + sdk_revision + "/src/waymo_open_dataset/dataset.proto"
    sdk_source = get(sdk_url, 65536)
    (HERE / "dataset.proto").write_bytes(sdk_source)
    assert b"Frame start time" in sdk_source and b"roughly (but not exactly)" in sdk_source
    tree_path = WORK / "preprocessing_gap_audit/mdetector_source_tree.json"
    tree = json.loads(tree_path.read_text())
    assert tree["sha"] == REV and tree["truncated"] is False

    result = {
        "status": "SOURCE_TIMING_MECHANISM_VERIFIED_AUTHOR_WAYMO_PACKET_MAPPING_UNRESOLVED",
        "GPU_or_training_or_detector_construction_or_scan_started": False,
        "author_equivalence": False,
        "author_exact_scan_end_known": False,
        "allow_training": False,
        "allow_detector_execution": False,
        "reference_clock_candidate": {
            "name": "WAYMO_FRAME_START_REFERENCE_CLOCK_SECONDS_V1",
            "formula": "float64(timestamp_micros - first_timestamp_micros) / 1000000",
            "subtract_integer_microseconds_before_float_conversion": True,
            "raw_timestamp_micros_kept_as_identity": True,
            "api_scan_end_time_argument_means_documented_frame_reference_clock_here": True,
            "claims_physical_scan_end_or_pose_at_frame_start": False,
            "uniform_time_assigned_to_all_frame_points": True,
            "model_normalized_time_or_column6_used": False,
            "guessed_offset_or_point_timestamps_or_pose_interpolation": False,
            "body_to_world": "unchanged original worldXYZ = Frame.pose @ homogeneous original .bin bodyXYZ",
            "TOP_conversion": "official SDK uses per-pixel vehicle-to-world pose, then inverse Frame.pose; original converter stores the resulting Frame-reference body coordinates",
            "other_lasers": "existing virtual-range-image fixed body/reference coordinates; preserve original conversion",
            "constant_clock_translation_note": "Core time decisions use differences of frame/map times; a common translation cancels in real arithmetic. This is not a bitwise native execution or author-equivalence proof.",
        },
        "currentsegment": {
            "frames": 49, "all_converted_rows": sum(counts),
            "max_rows_per_frame": max(counts), "min_rows_per_frame": min(counts),
            "first_timestamp_micros": times[0], "last_timestamp_micros": times[-1],
            "frame_interval_microseconds_min_median_max": [min(diffs), statistics.median(diffs), max(diffs)],
            "relative_duration_seconds": (times[-1]-times[0])/1e6,
            "all_frames_chronological": True,
            "exact_Frame_pose_timestamp_provided": False,
            "range_image_flow_present_frames": raw_fields["frames_with_point_flow_fields"],
            "per_frame_clock_and_count": [{"source_index": i, "timestamp_micros": t, "reference_clock_seconds": (t-times[0])/1e6, "rows": n} for i,(t,n) in enumerate(zip(times,counts))],
        },
        "bounded_public_source_search": {
            "mdetector_revision": REV, "full_git_tree_nontruncated": True,
            "source_tree_pin": pin(tree_path),
            "standalone_raw_Waymo_to_ROS_bag_producer_found": False,
            "release_metadata_url": releases_url,
            "all_release_tags": [r["tag_name"] for r in releases],
            "release_tag": release["tag_name"], "release_published_at": release["published_at"],
            "asset_url": ASSET_URL, "asset_bytes": len(zip_bytes), "asset_sha256": ASSET_SHA,
            "archive_members": z.namelist(),
            "selected_released_source_pins": released_pins,
            "no_raw_TFRecord_to_ROS_bag_producer_found_in_examined_release": True,
            "limit": "Only pinned public M-detector tree, sole released FAST_LIO source, and actual DrivingEditor converter examined. Private/unreleased producers and downloaded bags are not audited.",
        },
        "source_facts": [
            {"source": "dataset.proto", "lines": [366,375], "fact": "Frame timestamp marks start; Frame pose roughly midpoint and explicitly does not correspond to that timestamp."},
            {"source": "dataset.proto", "lines": [198,213], "fact": "TOP range-image poses encode per-pixel geometry, not an explicit point-time array; both returns share this pose."},
            {"source": "dataset.proto", "lines": [298,335], "fact": "CameraImage.pose_timestamp belongs to CameraImage.pose; trigger/readout timing is separate."},
            {"source": "DrivingEditor-run/scripts/waymo_converter.py", "lines": [372,396], "fact": "Converter saves original Frame.pose and Frame.timestamp independently, without a scan-end offset."},
            {"source": "DrivingEditor-run/scripts/waymo_converter.py", "lines": [249,253], "fact": "Sixth .bin column is mask_indices; the old timestamp comment is not executable time data."},
            {"source": "src/dynfilter_with_odom.cpp", "lines": [63,113], "fact": "Odom stamp becomes cur_time; cloud stamp ignored; FIFO association has no stamp identity check."},
            {"source": "src/DynObjFilter.cpp", "lines": [263,270], "fact": "Original geometry transformed by supplied R/t; all frame points get the same supplied scan_end_time."},
            {"source": "src/DynObjFilter.cpp", "lines": [582,598], "fact": "Buffer/map timing depends on physical time differences, buffer_delay and frame_dur."},
            {"source": "FAST_LIO/config/waymo.yaml", "lines": [8,28], "fact": "Released Waymo config selects lidar_type4 and imu_en=false."},
            {"source": "FAST_LIO/src/laserMapping.cpp", "lines": [511,521], "fact": "With imu_en=false the published lidar_end_time equals the input lidar_beg_time; no end offset is added."},
            {"source": "FAST_LIO/src/laserMapping.cpp", "lines": [929,1012], "fact": "Published body cloud and odometry share lidar_end_time."},
            {"source": "FAST_LIO/src/preprocess.cpp", "lines": [188,209], "fact": "L515-style path preserves XYZ/I, filters rows, and sets curvature0; it is not the raw Waymo producer."},
        ],
        "input_reference_pins": {name: pin(INIT/name) for name in canonical},
        "mdetector_examined_source_pins": [pin(SRC/p) for p in ["src/dynfilter_with_odom.cpp", "src/DynObjFilter.cpp", "include/m-detector/DynObjFilter.h", "config/waymo/waymo0.yaml", "config/waymo/waymo1.yaml"]],
        "sdk_public_source": {"revision": sdk_revision, "url": sdk_url, "bytes": len(sdk_source), "sha256": sha(sdk_source), "installed_pb2_sha256": raw_fields["source_pb2_SHA256"]},
        "minimum_next_validation": [
            "Root review explicit custom frame-clock receipt; author-equivalence remains false.",
            "Packet-only read-only preflight all49 original .bin/pose/rawID/chronology hashes with no native construction/filter; report direct shape/type/count/geometry/RAM estimates.",
            "Native ABI/type sizeof and allocation census before construction, reconcile configured capacity with actual max frame rows, review bounded RAM/cgroup limit and measured RSS gate.",
            "Only after separate root execution authorization: fresh-process bounded original-core construction and chronological CPU scans, log RSS/exit/results keyed to rawIDs; no training-input rewrite.",
            "Before I-only fromzero, resolve detector output domain versus singleactor/body-only Md; other movers/shadows are separate unresolved supervision inputs.",
        ],
        "canonical_references_unchanged_after": all(sha((INIT/name).read_bytes()) == expected for name,expected in canonical.items()),
    }
    out = HERE / "clock_source_audit_cpu.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(out), "sha256": sha(out.read_bytes()), "frames": 49, "max_frame_rows": max(counts), "clock_duration_seconds": result["currentsegment"]["relative_duration_seconds"], "author_equivalence": False, "allow_detector_execution": False}))

if __name__ == "__main__":
    main()

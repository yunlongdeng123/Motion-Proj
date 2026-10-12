"""只读 CPU 输入审计；不导入训练模块，不写远端文件，不启动 GPU。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from plyfile import PlyData

BASE = Path("/root/autodl-tmp/drivingeditor_repro")
SCENE = BASE / "data/waymo_paper_scenes/scene0008_023_071_fresh30k_v2_groundonly"
TASK = BASE / "campaign/next_scene30k_20261012"
SELECTION = TASK / "candidate_v2/dual_actor_selection.reviewed.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def mask(path):
    with Image.open(path) as image:
        return np.array(image)


def file_inventory(root):
    files = [path for path in Path(root).rglob("*") if path.is_file()]
    return {"file_count": len(files), "bytes": sum(path.stat().st_size for path in files)}


def resource_snapshot():
    disk = shutil.disk_usage(BASE)
    resources = {"disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
                 "disk_free_GiB": disk.free / 1024 ** 3, "audit_threads": 1}
    for name in ("cpu.max", "memory.max", "memory.current", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            resources[name] = path.read_text().strip()
    if "cpu.max" in resources:
        quota, period = resources["cpu.max"].split()
        resources["effective_cpu_quota"] = None if quota == "max" else int(quota) / int(period)
    return resources


def audit_scene(scene=SCENE, selection_path=SELECTION):
    """逐帧检查 canonical 两 body 并集、RGB、时间、真实点来源和前车保留。"""
    scene, selection_path = Path(scene), Path(selection_path)
    prep = read_json(scene / "preparation_manifest.json")
    chosen = read_json(selection_path)
    rows = prep["frames"]
    expected_ids = ["0008%03d" % index for index in range(23, 72)]
    assert [row["id"] for row in rows] == expected_ids
    assert chosen["preserved_raw_track_id"] == "vBOasuXpjI7h7d-X_VEQdA"
    assert len(chosen["actors"]) == 2
    required = {"RGB": "image/image_0", "sky": "sky/sky_0", "body_union": "dynamic_masks/dynamic_mask_0",
                "dynamic_RGB": "dynamic_image/dynamic_image_0", "DEVA": "mask/mask_0/Annotations",
                "pose": "pose", "calibration": "calib", "LiDAR": "velodyne",
                "timestamp": "timestamp", "raw_metadata": "raw_front_metadata"}
    counts = {}
    for name, directory in required.items():
        extension = ".txt" if name in ("pose", "calibration", "timestamp") else ".bin" if name == "LiDAR" else ".json" if name == "raw_metadata" else ".png"
        names = sorted(path.stem for path in (scene / directory).glob("*" + extension))
        assert names == expected_ids, (name, len(names))
        counts[name] = len(names)
    assets = [PlyData.read(str(scene / ("points3d_dynamic_%d.ply" % actor)))["vertex"].data for actor in range(2)]
    source_indices = [np.load(scene / ("dynamic_actor_assets/actor_%d/source_lidar_indices.npz" % actor), allow_pickle=False) for actor in range(2)]
    offsets = [0, 0]
    pixels, actor_nonempty, black_body, preserved_pixels = [], [[], []], 0, []
    sky_body_overlap, calibration_pose_checked = 0, 0
    try:
        for frame_index, row in enumerate(rows):
            frame = row["id"]
            original = rgb(scene / "image/image_0" / (frame + ".png"))
            categorical = mask(scene / "mask/mask_0/Annotations" / (frame + ".png"))
            assert original.shape == (1280, 1920, 3) and categorical.shape == (1280, 1920)
            union = np.zeros(categorical.shape, dtype=bool)
            raw = np.fromfile(scene / "velodyne" / (frame + ".bin"), dtype=np.float32).reshape(-1, 6)
            assert len(raw) == row["lidar_points"] and np.isfinite(raw).all()
            pose = np.loadtxt(scene / "pose" / (frame + ".txt"))
            metadata = read_json(scene / "raw_front_metadata" / (frame + ".json"))
            assert pose.shape == (4, 4) and np.isfinite(pose).all()
            assert np.array_equal(pose, np.array(metadata["frame_pose_world_from_vehicle"]).reshape(4, 4))
            assert int((scene / "timestamp" / (frame + ".txt")).read_text()) == row["timestamp_micros"] == metadata["frame_timestamp_micros"]
            calibration = {}
            for line in (scene / "calib" / (frame + ".txt")).read_text().splitlines():
                key, values = line.split(":", 1)
                assert key not in calibration
                calibration[key] = np.fromstring(values, sep=" ")
            projection = calibration["P0"].reshape(3, 4)
            extrinsic = calibration["Tr_velo_to_cam_0"].reshape(3, 4)
            assert np.isfinite(projection).all() and np.isfinite(extrinsic).all()
            assert projection[0, 0] > 0 and projection[1, 1] > 0
            assert np.allclose(extrinsic[:, :3] @ extrinsic[:, :3].T, np.eye(3), atol=1e-8)
            calibration_pose_checked += 1
            for actor in range(2):
                actor_row = chosen["actors"][actor]["frames"][frame_index]
                assert actor_row["frame_id"] == frame and actor_row["reviewed"]
                body = np.isin(categorical, actor_row["instance_ids"])
                actual = mask(scene / ("dynamic_actor_assets/actor_%d/body_masks" % actor) / (frame + ".png"))
                assert actual.shape == union.shape and set(np.unique(actual)).issubset({0, 255})
                assert np.array_equal(actual > 0, body) and not np.any(union & body)
                union |= body
                if body.any():
                    actor_nonempty[actor].append(frame)
                indices = source_indices[actor][frame]
                assert indices.dtype == np.int32 and np.all(indices >= 0) and np.all(indices < len(raw))
                assert len(np.unique(indices)) == len(indices)
                world = (np.column_stack((raw[indices, :3].astype(np.float64), np.ones(len(indices)))) @ pose.T)[:, :3].astype(np.float32)
                vertices = assets[actor][offsets[actor]:offsets[actor] + len(indices)]
                assert np.array_equal(np.column_stack([vertices[key] for key in ("x", "y", "z")]), world)
                for key in ("nx", "ny", "nz", "time"):
                    assert np.all(vertices[key] == np.float32(row["model_time"]))
                offsets[actor] += len(indices)
            binary = mask(scene / "dynamic_masks/dynamic_mask_0" / (frame + ".png"))
            assert binary.shape == union.shape and set(np.unique(binary)).issubset({0, 255})
            assert np.array_equal(binary > 0, union)
            dynamic = rgb(scene / "dynamic_image/dynamic_image_0" / (frame + ".png"))
            assert np.array_equal(dynamic, original * union[..., None])
            sky = mask(scene / "sky/sky_0" / (frame + ".png"))
            assert sky.shape == union.shape and set(np.unique(sky)).issubset({0, 255})
            sky_body_overlap += int(np.count_nonzero((sky > 0) & union))
            black_body += int(np.count_nonzero(union & (original[..., 0] == 0)))
            preserved = categorical == 153
            assert not np.any(preserved & union)
            preserved_pixels.append(int(preserved.sum()))
            pixels.append(int(union.sum()))
        assert offsets == [len(vertices) for vertices in assets] and all(offsets)
    finally:
        for indices in source_indices:
            indices.close()
    static_vertices = PlyData.read(str(scene / "points3d.ply"))["vertex"].data
    assert len(static_vertices) == 600000
    assert all(np.isfinite(static_vertices[key]).all() for key in ("x", "y", "z", "nx", "ny", "nz", "time"))
    for key in ("nx", "ny", "nz"):
        assert np.array_equal(static_vertices[key], static_vertices["time"])
    raw_path = Path(prep["raw_path"])
    markers = ["transform_matrix.txt", "sgcm_transform_matrix.txt", "full_loader_validation.json", "sgcm_input_validation.json", "formal_training_ready.json"]
    return {
        "status": "PASS_READONLY_INPUT_CONSISTENCY", "training_ready": False,
        "source_path": str(scene), "selection_path": str(selection_path), "inventory": file_inventory(scene),
        "frame_ids": expected_ids, "frame_count": 49, "camera": "FRONT", "raw_training_range": [23, 71],
        "author_video_verified_range": [23, 51], "author_training_window_publicly_confirmed": False,
        "original_WH": [1920, 1280], "render_WH_scale4": [480, 320], "field_counts": counts,
        "model_time_range": [rows[0]["model_time"], rows[-1]["model_time"]], "model_interval": 0.02,
        "physical_raw_interval_median_s": float(np.median(np.diff([row["timestamp_micros"] for row in rows])) / 1e6),
        "fixed_pose_anchor": "0008033", "raw_cache": {"path": str(raw_path), "exists": raw_path.is_file(), "bytes": raw_path.stat().st_size if raw_path.is_file() else None},
        "static_PLY_vertices": len(static_vertices), "static_PLY_role": "World-space real LiDAR seed mixture; actual loader removes only points within strict0.1m of selected actor priors.",
        "actor_PLY_vertices": offsets, "actor_body_nonempty_frames": actor_nonempty,
        "actor_instance_ids": [77, 82], "real_sensor_subset_and_temporal_properties_exact": True,
        "union_and_dynamic_RGB_exact": True, "canonical_full_mask_available": True,
        "body_pixels_min_max": [min(pixels), max(pixels)], "red_channel_zero_body_pixels": black_body,
        "sky_body_overlap_pixels": sky_body_overlap, "calibration_pose_frames_checked": calibration_pose_checked,
        "preserved_actor": {"raw_track_id": chosen["preserved_raw_track_id"], "DEVA_instance_id": 153,
                            "union_overlap_pixels": 0, "pixels_min_max": [min(preserved_pixels), max(preserved_pixels)]},
        "shadow_supervision": "absent", "m_detector_used": False,
        "preprocessing_caveat": "v2 removes pickup/raw29 ground-only20-point cluster; visible body mask remains. No GT label filtering is introduced.",
        "present_readiness_markers": [name for name in markers if (scene / name).is_file()],
        "missing_readiness_markers": [name for name in markers if not (scene / name).is_file()],
        "scope": "Read-only file/CPU consistency audit. No actual dataset reader, GaussianModel, checkpoint, render, train, GPU or source mutation."
    }


def red_inventory():
    task = read_json(BASE / "campaign/scene_search_mirror/task_handoff.json")
    review = read_json(BASE / "campaign/scene_search_mirror/batch5_visual_review.json")
    return {"status": "IDENTITY_UNKNOWN_NO_FORMAL_INPUT", "scene_key": review["scene_key"], "clip": review["clip"],
            "red_car_identity_verified": review["red_car_identity_verified"], "identity": review["red_car_identity"],
            "historical_search_status": task["status"], "full_temporal_identity_verified": review["full_temporal_identity_verified"],
            "formal_source_path": None, "raw_tfrecord_match": None, "preview_only": True,
            "limitation": "Existing search examines first-record thumbnails; it cannot exclude later windows or identify a whole segment.",
            "next_step": "Use confirmed screenshot identity or existing local caches before creating49-frame inputs; no new batch/download is started by this audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    args = parser.parse_args()
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": "wm-vgpu-1011",
              "gpu_used": False, "remote_written": False, "resources": resource_snapshot(),
              "scene0008": audit_scene(args.scene, args.selection), "red_candidate": red_inventory()}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

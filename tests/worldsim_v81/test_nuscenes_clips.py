"""nuScenes P0 连续链与原始分片推断的关键约束。"""
from scripts.worldsim_v81.prepare_nuscenes_clips import (
    build_candidate,
    infer_archive,
    normalize_archive,
    source_prefix,
    source_stamp,
)


def camera_chain():
    rows = {}
    scene_by_sample = {}
    for index in range(25):
        token = f"camera-{index}"
        sample = f"sample-{index}"
        rows[token] = {
            "token": token,
            "sample_token": sample,
            "timestamp": 1_000_000 + index * 83_333,
            "filename": f"sweeps/CAM_FRONT/log__CAM_FRONT__{index:016d}.jpg",
            "prev": f"camera-{index - 1}" if index else "",
            "next": f"camera-{index + 1}" if index < 24 else "",
            "is_key_frame": index % 6 == 0,
            "width": 1600,
            "height": 900,
        }
        scene_by_sample[sample] = "official-scene"
    return rows, scene_by_sample


def test_native_next_chain_keeps_order_and_scene():
    rows, scene_by_sample = camera_chain()
    clip = build_candidate(rows["camera-0"], rows, scene_by_sample)
    assert clip is not None
    assert [frame["token"] for frame in clip["frames"]] == [f"camera-{i}" for i in range(25)]
    assert clip["median_dt_us"] == 83_333

    scene_by_sample["sample-12"] = "different-scene"
    assert build_candidate(rows["camera-0"], rows, scene_by_sample) is None


def test_rejects_gap_or_broken_backlink():
    rows, scene_by_sample = camera_chain()
    rows["camera-9"]["prev"] = "camera-7"
    assert build_candidate(rows["camera-0"], rows, scene_by_sample) is None

    rows, scene_by_sample = camera_chain()
    rows["camera-9"]["timestamp"] += 200_000
    assert build_candidate(rows["camera-0"], rows, scene_by_sample) is None


def test_lidar_log_prefix_can_identify_camera_archive():
    lidar = "sweeps/LIDAR_TOP/n008-log__LIDAR_TOP__1535730433047413.pcd.bin"
    image = "sweeps/CAM_FRONT/n008-log__CAM_FRONT__1535730507612404.jpg"
    assert source_prefix(lidar) == source_prefix(image) == "n008-log"
    assert source_stamp(lidar) == 1535730433047413
    assert normalize_archive("06") == "v1.0-trainval06_blobs.tgz"
    assert infer_archive(image, {}, {"n008-log": [(1535730433047413, "v1.0-trainval06_blobs.tgz")]}) == "v1.0-trainval06_blobs.tgz"

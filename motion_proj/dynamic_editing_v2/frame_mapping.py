"""nuScenes raw/sample_data 与 AD-GS 处理/渲染帧的双重校验映射。"""
from __future__ import annotations

from typing import Any


CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
TEST_FRAMES = tuple(range(4, 60, 4))
TRAIN_FRAMES = tuple(frame for frame in range(60) if frame not in TEST_FRAMES)


def render_split_index(processed_frame: int, view: int) -> tuple[str, int]:
    if processed_frame < 0 or processed_frame >= 60:
        raise ValueError(f"processed_frame 越界: {processed_frame}")
    if view < 0 or view >= len(CAMERAS):
        raise ValueError(f"view 越界: {view}")
    if processed_frame in TEST_FRAMES:
        return "test", TEST_FRAMES.index(processed_frame) * 3 + view
    return "train", TRAIN_FRAMES.index(processed_frame) * 3 + view


def validate_frame_table(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 180:
        raise ValueError(f"frame table 必须为 180 行: {len(rows)}")
    seen_tokens: set[str] = set()
    for frame_idx in range(10, 70):
        frame_rows = [row for row in rows if int(row["frame_idx"]) == frame_idx]
        if [row["camera"] for row in frame_rows] != list(CAMERAS):
            raise ValueError(f"camera 顺序错误: frame={frame_idx}")
        for view, row in enumerate(frame_rows):
            if int(row["image_id"]) != (frame_idx - 10) * 3 + view:
                raise ValueError(f"image_id 错误: {row}")
            token = row["sample_data_token"]
            if token in seen_tokens:
                raise ValueError(f"sample_data token 重复: {token}")
            seen_tokens.add(token)


def nearest_camera_frame(
    timestamp_us: int,
    sample_token: str,
    camera: str,
    rows: list[dict[str, Any]],
    sample_data_by_token: dict[str, dict[str, Any]],
    max_delta_us: int,
) -> dict[str, Any]:
    if camera not in CAMERAS:
        raise ValueError(f"非冻结相机: {camera}")
    candidates = [row for row in rows if row["camera"] == camera]
    if not candidates:
        raise ValueError(f"相机无 frame: {camera}")
    exact = [
        row
        for row in candidates
        if sample_data_by_token.get(row["sample_data_token"], {}).get("sample_token")
        == sample_token
    ]
    if not exact:
        raise ValueError(f"sample_token 无精确 camera mapping: {sample_token} {camera}")
    selected = min(
        exact,
        key=lambda row: (abs(int(row["timestamp"]) - int(timestamp_us)), int(row["frame_idx"])),
    )
    delta = abs(int(selected["timestamp"]) - int(timestamp_us))
    if delta > max_delta_us:
        raise ValueError(f"timestamp 无法对齐: {camera} delta_us={delta}")
    sd = sample_data_by_token.get(selected["sample_data_token"])
    if sd is None:
        raise ValueError(f"sample_data token 不在冻结 metadata: {selected['sample_data_token']}")
    if int(sd["timestamp"]) != int(selected["timestamp"]):
        raise ValueError("frame table/sample_data timestamp 不一致")
    processed = int(selected["frame_idx"]) - 10
    view = CAMERAS.index(camera)
    split, render_index = render_split_index(processed, view)
    return {
        "camera": camera,
        "sample_token": sample_token,
        "sample_data_token": selected["sample_data_token"],
        "sample_data_sample_token": sd["sample_token"],
        "sample_token_match": sd["sample_token"] == sample_token,
        "mapping_basis": "timestamp_plus_exact_sample_token",
        "annotation_timestamp_us": int(timestamp_us),
        "camera_timestamp_us": int(selected["timestamp"]),
        "timestamp_delta_us": delta,
        "adgs_raw_frame": int(selected["frame_idx"]),
        "adgs_processed_frame": processed,
        "adgs_image_id": int(selected["image_id"]),
        "adgs_render_split": split,
        "adgs_render_index": render_index,
        "drivestudio_processed_frame": None,
        "drivestudio_mapping_status": "assets_missing_until_m3",
        "filename": selected["filename"],
        "calibrated_sensor_token": sd["calibrated_sensor_token"],
        "ego_pose_token": sd["ego_pose_token"],
    }

"""历史离线工具；当前 v3 构建/训练不再要求或加载本证据。

同49帧 LiDAR/3D框/已有道路几何构建选择性背景时间证据；纯CPU。

不读取新 RGB、旧渲染或 checkpoint，不改五身份初始化。所有非固定类别
的真实3D框只用于稳定证据否决，不新建Actor、不删除背景高斯。
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np


def transform(xyz, matrix):
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def object_box(label, pose, model, scale, padding):
    box = label["box"]
    angle = float(box["heading"])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    center_ego = np.array([[box["center_x"], box["center_y"], box["center_z"]]])
    world = transform(center_ego, pose)
    return {
        "center": transform(world, model)[0],
        "axes": model[:3, :3] @ pose[:3, :3] @ rotation / scale,
        "half": (np.array([box["length"], box["width"], box["height"]]) / 2 + padding) * scale,
    }


def inside_boxes(xyz, boxes):
    mask = np.zeros(len(xyz), dtype=bool)
    for box in boxes:
        extent = np.abs(box["axes"]) @ box["half"]
        candidates = np.flatnonzero((np.abs(xyz - box["center"]) <= extent).all(1))
        local = (xyz[candidates] - box["center"]) @ box["axes"]
        mask[candidates] |= (np.abs(local) <= box["half"]).all(1)
    return mask


def box_voxel_centers(box, cell_size):
    extent = np.abs(box["axes"]) @ box["half"]
    lower = np.floor((box["center"] - extent) / cell_size).astype(np.int64)
    upper = np.floor((box["center"] + extent) / cell_size).astype(np.int64)
    grid = np.stack(np.meshgrid(*[
        np.arange(lo, hi + 1) for lo, hi in zip(lower, upper)
    ], indexing="ij"), axis=-1).reshape(-1, 3)
    centers = (grid + 0.5) * cell_size
    # 扩半个体素对角线，防止体素边缘的目标几何被稳定标签穿透。
    local = (centers - box["center"]) @ box["axes"]
    return centers[(np.abs(local) <= box["half"] + np.sqrt(3) * cell_size / 2).all(1)]


def prepare(source, bundle_path, output):
    from utils.v3_temporal import (
        TEMPORAL_POLICY, TEMPORAL_VERSION, repeated_surface_cells, voxel_keys_numpy,
    )
    if output.exists():
        raise FileExistsError("preserve existing temporal evidence; use a fresh output")
    meta = json.loads(bundle_path.read_text(encoding="utf-8"))
    if meta.get("status") != "CPU_INPUT_BUNDLE_VERIFIED":
        raise ValueError("same verified v2 input bundle required")
    policy = TEMPORAL_POLICY
    model = np.asarray(meta["world_to_model"], dtype=np.float64)
    scale = float(meta["meter_to_model"])
    plane = np.asarray(meta["ground_plane_model"], dtype=np.float64)
    cell_size = scale * policy["voxel_m"]
    keys_per_frame, timestamps, veto_keys, rows = [], [], [], []
    expected = [f"0008{raw:03d}" for raw in range(89, 138)]
    if sorted(meta["frames"]) != expected:
        raise ValueError("only the paired raw89-137 input window is allowed")
    for index, frame_id in enumerate(expected):
        pose = np.loadtxt(source / "pose" / (frame_id + ".txt"))
        annotations = json.loads((source / "annotations" / (frame_id + ".json")).read_text())
        raw = np.fromfile(source / "velodyne" / (frame_id + ".bin"), np.float32).reshape(-1, 6)
        xyz = transform(transform(raw[:, :3].astype(np.float64), pose), model)
        boxes = [object_box(label, pose, model, scale, policy["object_box_padding_m"])
                 for label in annotations["laser_labels"]
                 if label["type"] in {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}
                 and min(label["box"][key] for key in ("length", "width", "height")) > 0]
        in_object = inside_boxes(xyz, boxes)
        finite = np.isfinite(xyz).all(1)
        keys = np.unique(voxel_keys_numpy(xyz[finite & ~in_object], cell_size))
        keys_per_frame.append(keys)
        timestamps.append(-0.48 + index * 0.02)
        for box in boxes:
            centers = box_voxel_centers(box, cell_size)
            # 道路薄层独立保护；不能把车辆上方/内部的体素否决扩到整条射线。
            above_road = (centers @ plane[:3] + plane[3]) > (
                policy["road_protection_height_m"] * scale
            )
            veto_keys.append(voxel_keys_numpy(centers[above_road], cell_size))
        rows.append({"frame": frame_id, "lidar_points": len(xyz),
                     "object_points_excluded_from_stability_evidence": int(in_object.sum()),
                     "independent_surface_cells": len(keys), "object_boxes": len(boxes)})
        print(frame_id, len(keys), flush=True)

    stable, counts, span = repeated_surface_cells(
        keys_per_frame, timestamps, policy["minimum_observation_frames"],
        policy["minimum_time_span_model"],
    )
    veto = np.unique(np.concatenate(veto_keys)) if veto_keys else np.empty(0, np.int64)
    keep = ~np.isin(stable, veto)
    stable, counts, span = stable[keep], counts[keep], span[keep]

    # 使用已冻结的路面几何支撑，不使用 RGB/provenance 来改变四臂的标签。
    with np.load(bundle_path.parent / "query_evidence.npz", allow_pickle=False) as query:
        road = transform(query["world"][query["surface_supported"].astype(bool)], model)
    road = road[np.abs(road @ plane[:3] + plane[3]) <= 0.2 * scale]
    road_keys = np.unique(voxel_keys_numpy(road, cell_size))
    road_keys = road_keys[~np.isin(road_keys, veto)]
    keys = np.union1d(stable, road_keys)
    confidence = np.full(len(keys), policy["road_geometry_confidence"], np.float32)
    confidence[np.searchsorted(keys, stable)] = 1.0
    output.mkdir(parents=True)
    np.savez_compressed(output / "field.npz", keys=keys, confidence=confidence,
                        repeated_keys=stable, observation_frames=counts, time_span=span,
                        road_geometry_keys=road_keys, object_veto_keys=veto)
    receipt = {
        "version": TEMPORAL_VERSION, "policy": policy, "status": "CPU_GEOMETRY_EVIDENCE_READY",
        "field": "field.npz", "source": str(source), "input_bundle": str(bundle_path),
        "world_to_model": model.tolist(), "meter_to_model": scale,
        "frames": expected, "time_range_model": [-0.48, 0.48],
        "repeated_surface_cells": len(stable), "road_geometry_cells": len(road_keys),
        "object_veto_cells": len(veto), "total_confident_cells": len(keys), "per_frame": rows,
        "classification": "three-dimensional repeated support plus supported road geometry; object-volume veto",
        "two_dimensional_mask_used_for_classification": False,
        "point_or_appearance_initialization_changed": False,
        "input_role": "same-window training geometry; not held-out truth or new editable actors",
        "road_geometry_role": "common weaker stable-surface prior in all four arms, independent of road RGB/depth switches",
        "GPU_used": False, "training_updates": 0,
    }
    (output / "metadata.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "per_frame"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.engine))
    prepare(args.source, args.bundle, args.output)


if __name__ == "__main__":
    main()

"""Build one GT-guided image-space target/neighbor geometry record per audit clip."""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import cv2
import ijson
import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def matrix(t, q):
    out = np.eye(4)
    out[:3, :3] = Quaternion(q).rotation_matrix
    out[:3, 3] = t
    return out


def interpolate(a, b, fraction):
    fraction = min(1.0, max(0.0, fraction))
    return {
        "translation": ((1 - fraction) * np.array(a["translation"]) + fraction * np.array(b["translation"])).tolist(),
        "size": ((1 - fraction) * np.array(a["size"]) + fraction * np.array(b["size"])).tolist(),
        "rotation": Quaternion.slerp(Quaternion(a["rotation"]), Quaternion(b["rotation"]), amount=fraction).elements.tolist(),
        "instance_token": a["instance_token"],
    }


def target_at(t, rows):
    if not rows:
        return None, "missing_track"
    times = [r["timestamp"] for r in rows]
    idx = bisect.bisect_left(times, t)
    if idx == 0:
        return (rows[0], "nearest_first") if abs(t - times[0]) <= 250_000 else (None, "outside_track")
    if idx == len(times):
        return (rows[-1], "nearest_last") if abs(t - times[-1]) <= 250_000 else (None, "outside_track")
    before, after = rows[idx - 1], rows[idx]
    if after["timestamp"] - before["timestamp"] > 1_100_000:
        return None, "annotation_gap_gt1.1s"
    alpha = (t - before["timestamp"]) / (after["timestamp"] - before["timestamp"])
    return interpolate(before, after, alpha), "interpolated"


def polygon(annotation, camera, wh=(1024, 576)):
    box = Box(annotation["translation"], annotation["size"], Quaternion(annotation["rotation"]))
    cam = camera["w2c"][:3, :3] @ box.corners() + camera["w2c"][:3, 3:4]
    if np.min(cam[2]) <= 0.2:
        return None
    xy = camera["k"] @ cam
    xy = xy[:2] / xy[2:]
    hull = cv2.convexHull(np.rint(xy.T).astype(np.int32)).reshape(-1, 2)
    x0, y0 = np.min(xy, axis=1)
    x1, y1 = np.max(xy, axis=1)
    w, h = wh
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    bounds = [float(max(0, x0)), float(max(0, y0)), float(min(w, x1)), float(min(h, y1))]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    return {"hull": hull.tolist(), "box_xyxy": [round(x, 2) for x in bounds]}


def choose_instance_prompt(frames):
    """全局优先选目标清楚、未截边且少覆盖邻车的提示帧。

    70个片段保持冻结；这里只使用实际RGB曝光时刻的GT投影，不看模型输出。
    """
    areas = [0 if f["target"] is None else
             (f["target"]["box_xyxy"][2] - f["target"]["box_xyxy"][0]) *
             (f["target"]["box_xyxy"][3] - f["target"]["box_xyxy"][1]) for f in frames]
    assert max(areas) > 0
    largest_frame = int(np.argmax(areas))
    eligible = [i for i, area in enumerate(areas) if area >= 0.5 * max(areas)]
    unclipped = []
    for i in eligible:
        x0, y0, x1, y1 = frames[i]["target"]["box_xyxy"]
        if min(x0, y0, 1024 - x1, 576 - y1) >= 8:
            unclipped.append(i)
    pool = unclipped or eligible
    scores = []
    for i in pool:
        frame = frames[i]
        x0, y0, x1, y1 = frame["target"]["box_xyxy"]
        bounds = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
                  min(1024, int(np.ceil(x1))), min(576, int(np.ceil(y1))))
        other = np.zeros((576, 1024), dtype=np.uint8)
        for neighbor in frame["neighbors"]:
            cv2.fillConvexPoly(other, np.asarray(neighbor["hull"], dtype=np.int32), 1)
        bx0, by0, bx1, by1 = bounds
        neighbor_fraction = float(other[by0:by1, bx0:bx1].mean())
        scores.append((neighbor_fraction, -areas[i], i))
    fraction, _, chosen = min(scores)
    return chosen, {"rule": "area_at_least_half_max_then_unclipped_8px_then_min_other_GT_hull_fraction_in_prompt_box",
                    "max_area_frame": largest_frame,
                    "chosen_other_GT_fraction_in_box": round(fraction, 5),
                    "candidate_count": len(pool),
                    "unclipped_candidate_available": bool(unclipped)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sel = read(args.selection)
    expected = read(args.expected)
    assert len(sel["clips"]) == len(expected["clips"]) == 70
    by_id = {x["clip_id"]: x for x in sel["clips"]}
    samples = {r["token"]: r for r in read(args.metadata / "sample.json")}
    categories = {r["token"]: r["name"] for r in read(args.metadata / "category.json")}
    instances = {r["token"]: categories[r["category_token"]] for r in read(args.metadata / "instance.json")}
    needed_keyframes = {t for clip in expected["clips"] for t in clip["keyframe_sample_tokens"]}
    annotations = defaultdict(list)
    with (args.metadata / "sample_annotation.json").open("rb") as handle:
        for row in ijson.items(handle, "item", use_float=True):
            if row["sample_token"] in needed_keyframes and instances.get(row["instance_token"], "").startswith("vehicle."):
                annotations[row["sample_token"]].append({
                    "instance_token": row["instance_token"], "category": instances[row["instance_token"]],
                    "timestamp": samples[row["sample_token"]]["timestamp"],
                    "translation": row["translation"], "size": row["size"],
                    "rotation": row["rotation"], "visibility": int(row["visibility_token"]),
                })
    selected_camera_tokens = {f["sample_data_token"] for clip in expected["clips"] for f in clip["frames"]}
    sd = {}
    with (args.metadata / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item", use_float=True):
            if row["token"] in selected_camera_tokens:
                sd[row["token"]] = row
    assert len(sd) == len(selected_camera_tokens)
    ego_tokens = {r["ego_pose_token"] for r in sd.values()}
    ego = {}
    with (args.metadata / "ego_pose.json").open("rb") as handle:
        for row in ijson.items(handle, "item", use_float=True):
            if row["token"] in ego_tokens:
                ego[row["token"]] = row
    assert len(ego) == len(ego_tokens)
    cal = {r["token"]: r for r in read(args.metadata / "calibrated_sensor.json")}
    results = []
    for clip in expected["clips"]:
        spec = by_id[clip["clip_id"]]
        target = spec["instance_token"]
        keyframes = clip["keyframe_sample_tokens"]
        target_rows = sorted((r for token in keyframes for r in annotations[token] if r["instance_token"] == target),
                             key=lambda r: r["timestamp"])
        assert target_rows
        frames = []
        for frame in clip["frames"]:
            datum = sd[frame["sample_data_token"]]
            calrow = cal[datum["calibrated_sensor_token"]]
            erow = ego[datum["ego_pose_token"]]
            c2w = matrix(erow["translation"], erow["rotation"]) @ matrix(calrow["translation"], calrow["rotation"])
            k = np.array(calrow["camera_intrinsic"], dtype=float)
            k[0] *= 1024 / datum["width"]
            k[1] *= 576 / datum["height"]
            view = {"w2c": np.linalg.inv(c2w), "k": k}
            t = frame["source_timestamp_us"]
            actor, interpolation = target_at(t, target_rows)
            target_shape = polygon(actor, view) if actor is not None else None
            nearest = min(keyframes, key=lambda token: abs(samples[token]["timestamp"] - t))
            neighbors = []
            for other in annotations[nearest]:
                if other["instance_token"] == target:
                    continue
                shape = polygon(other, view)
                if shape:
                    neighbors.append({"instance_token": other["instance_token"], **shape})
            frames.append({"frame": frame["frame"], "filename": frame["filename"],
                           "source_timestamp_us": t, "target_interpolation": interpolation,
                           "target": target_shape, "neighbors": neighbors})
        prompt, prompt_policy = choose_instance_prompt(frames)
        assert frames[prompt]["target"] is not None
        results.append({"clip_id": clip["clip_id"], "scene": clip["scene"],
                        "camera": clip["camera"], "instance_token": target,
                        "prompt_frame": prompt, "prompt_policy": prompt_policy,
                        "frames": frames,
                        "frames_with_GT_target": sum(f["target"] is not None for f in frames),
                        "human_verdict": None})
        if len(results) % 10 == 0:
            print("GEOMETRY", len(results), flush=True)
    write(args.out, {"task_id": sel["task_id"], "run_id": sel["run_id"],
                     "source": "official nuScenes annotations interpolated in world pose to actual camera exposures; neighbor protection from nearest keyframe boxes",
                     "clip_count": len(results), "clips": results, "human_verdict": None})
    print("GEOMETRY_DONE", len(results), flush=True)


if __name__ == "__main__":
    main()

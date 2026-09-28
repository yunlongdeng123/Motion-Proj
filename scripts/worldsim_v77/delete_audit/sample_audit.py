"""Freeze a nuScenes val DELETE audit before inspecting model outputs.

The selection uses official metadata and 3-D boxes only. The nine exposed
v77 development scenes are excluded by their original nuScenes scene names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import val
from pyquaternion import Quaternion


EXPOSED = {
    "scene-0230", "scene-0255", "scene-0061", "scene-0436", "scene-0875",
    "scene-0242", "scene-0535", "scene-0471", "scene-0998",
}
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT", "CAM_BACK",
]
VEHICLE_PREFIX = "vehicle."
SEED = "v77-delete-audit-20260928-r1"


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def rank(key: str) -> str:
    return hashlib.blake2b((SEED + ":" + key).encode(), digest_size=12).hexdigest()


def stream(path: Path):
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item", use_float=True)


def pose(translation, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def project(annotation, view):
    box = Box(annotation["translation"], annotation["size"], Quaternion(annotation["rotation"]))
    corners = box.corners()
    cam = view["w2c"][:3, :3] @ corners + view["w2c"][:3, 3:4]
    if np.min(cam[2]) < 0.2:
        return None
    xy = view["k"] @ cam
    xy = xy[:2] / xy[2:]
    w, h = view["wh"]
    left, top = np.min(xy, axis=1)
    right, bottom = np.max(xy, axis=1)
    if right <= 0 or bottom <= 0 or left >= w or top >= h:
        return None
    left, right = max(0, left), min(w, right)
    top, bottom = max(0, top), min(h, bottom)
    if right - left < 12 or bottom - top < 9:
        return None
    center = view["w2c"][:3, :3] @ np.array(annotation["translation"]) + view["w2c"][:3, 3]
    return [float(left), float(top), float(right), float(bottom), float(center[2])]


def intersect(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def select_scene_rows(metadata: Path):
    scenes = read(metadata / "scene.json")
    names = set(val)
    assert len(names) == 150
    eligible = [dict(row, official_index=i) for i, row in enumerate(scenes)
                if row["name"] in names and row["name"] not in EXPOSED]
    assert len(eligible) == 150, "Expected nine exposed DEV scenes to be outside official val"
    samples = read(metadata / "sample.json")
    samples_by_token = {x["token"]: x for x in samples}
    scene_by_token = {x["token"]: x for x in eligible}
    ordered = {}
    for scene in eligible:
        rows = []
        token = scene["first_sample_token"]
        while token:
            row = samples_by_token[token]
            assert row["scene_token"] == scene["token"]
            rows.append(row)
            token = row["next"]
        assert len(rows) == scene["nbr_samples"]
        ordered[scene["token"]] = rows
    return eligible, ordered, samples_by_token, scene_by_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta, out = args.metadata, args.out
    assert not (out / "selection.json").exists(), "Frozen selection already exists"
    eligible, ordered, samples, scene_by_token = select_scene_rows(meta)
    print("SCENES", len(eligible), flush=True)

    categories = {x["token"]: x["name"] for x in read(meta / "category.json")}
    instances = {x["token"]: categories[x["category_token"]] for x in read(meta / "instance.json")}
    wanted_samples = {x["token"] for rows in ordered.values() for x in rows}
    annotations = defaultdict(list)
    for row in stream(meta / "sample_annotation.json"):
        if row["sample_token"] in wanted_samples and instances.get(row["instance_token"], "").startswith(VEHICLE_PREFIX):
            annotations[row["sample_token"]].append({
                "instance_token": row["instance_token"],
                "category": instances[row["instance_token"]],
                "visibility": int(row["visibility_token"]),
                "translation": row["translation"], "size": row["size"],
                "rotation": row["rotation"], "num_lidar_pts": row["num_lidar_pts"],
            })
    print("ANNOTATED", sum(map(len, annotations.values())), flush=True)

    calibrated = {x["token"]: x for x in read(meta / "calibrated_sensor.json")}
    channels = {x["token"]: x["channel"] for x in read(meta / "sensor.json")}
    sample_data = defaultdict(dict)
    camera_frames = defaultdict(lambda: defaultdict(list))
    ego_tokens = set()
    for row in stream(meta / "sample_data.json"):
        sample = samples.get(row["sample_token"])
        if sample is None or sample["scene_token"] not in scene_by_token:
            continue
        channel = channels[calibrated[row["calibrated_sensor_token"]]["sensor_token"]]
        if channel not in CAMERAS:
            continue
        scene_token = sample["scene_token"]
        brief = {k: row[k] for k in ["token", "sample_token", "timestamp", "filename", "width", "height", "ego_pose_token", "calibrated_sensor_token", "is_key_frame"]}
        camera_frames[scene_token][channel].append(brief)
        if row["is_key_frame"]:
            sample_data[row["sample_token"]][channel] = brief
            ego_tokens.add(row["ego_pose_token"])
    ego = {}
    for row in stream(meta / "ego_pose.json"):
        if row["token"] in ego_tokens:
            ego[row["token"]] = (row["translation"], row["rotation"])
    assert len(ego) == len(ego_tokens)
    for by_cam in camera_frames.values():
        for rows in by_cam.values():
            rows.sort(key=lambda x: x["timestamp"])
    print("CAMERA_ROWS", sum(len(rows) for cams in camera_frames.values() for rows in cams.values()), flush=True)

    candidates = defaultdict(list)
    for scene in eligible:
        scene_token = scene["token"]
        rows = ordered[scene_token]
        # Every third 2-Hz keyframe is a start candidate. No result-dependent start shift.
        for start in range(0, len(rows) - 5, 3):
            clip = rows[start:start + 6]
            anchor = clip[2]
            objects = annotations[anchor["token"]]
            if not objects:
                continue
            present = defaultdict(list)
            for sample in clip:
                for obj in annotations[sample["token"]]:
                    present[obj["instance_token"]].append(obj["visibility"])
            views = {}
            for channel in CAMERAS:
                datum = sample_data[anchor["token"]].get(channel)
                if datum is None:
                    continue
                cal = calibrated[datum["calibrated_sensor_token"]]
                et, er = ego[datum["ego_pose_token"]]
                c2w = pose(et, er) @ pose(cal["translation"], cal["rotation"])
                views[channel] = {"w2c": np.linalg.inv(c2w), "k": np.array(cal["camera_intrinsic"]),
                                  "wh": (datum["width"], datum["height"])}
            for channel, view in views.items():
                projected = {obj["instance_token"]: project(obj, view) for obj in objects}
                for obj in objects:
                    track = obj["instance_token"]
                    visibility = present[track]
                    if len(visibility) < 5 or sum(v >= 2 for v in visibility) < 4:
                        continue
                    box = projected[track]
                    if box is None:
                        continue
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    area_frac = area / (view["wh"][0] * view["wh"][1])
                    if not (0.00035 <= area_frac <= 0.35):
                        continue
                    behind, near = 0, 0
                    for other in objects:
                        if other["instance_token"] == track:
                            continue
                        b2 = projected[other["instance_token"]]
                        if b2 is None:
                            continue
                        overlap = intersect(box, b2) / area
                        if overlap >= 0.12 and b2[4] > box[4] + 1:
                            behind += 1
                        if overlap >= 0.05 and b2[4] < box[4] + 1:
                            near += 1
                    candidates[scene_token].append({
                        "scene": scene["name"], "scene_index": scene["official_index"],
                        "scene_token": scene_token, "instance_token": track,
                        "category": obj["category"], "camera": channel, "camera_index": CAMERAS.index(channel),
                        "start_keyframe": start, "anchor_keyframe": start + 2,
                        "anchor_sample_token": anchor["token"],
                        "area_fraction": round(area_frac, 6), "anchor_box_xyxy": [round(x, 2) for x in box[:4]],
                        "visibility_6": visibility, "occlusion_proxy": "partial" if min(visibility) <= 3 else "clear",
                        "behind_vehicle_proxy": behind > 0, "near_vehicle_proxy": near > 0,
                        "vehicle_count": len(objects),
                        "night_proxy": any(x in scene["description"].lower() for x in ["night", "dark", "low light"]),
                        "scene_description": scene["description"],
                    })
        print("CANDIDATES", scene["name"], len(candidates[scene_token]), flush=True)

    # Serialize the complete model-blind candidate pool so sampling decisions are inspectable.
    write(out / "candidate_pool.json", {scene_by_token[k]["name"]: v for k, v in candidates.items()})
    summary = {
        "official_val_scenes": len(eligible), "scenes_with_candidate": sum(bool(v) for v in candidates.values()),
        "candidate_count": sum(map(len, candidates.values())),
        "night_scenes_with_candidate": sum(bool(candidates[s["token"]]) and any(c["night_proxy"] for c in candidates[s["token"]]) for s in eligible),
        "candidate_cameras": dict(Counter(c["camera"] for v in candidates.values() for c in v)),
        "candidate_visibility": dict(Counter(c["occlusion_proxy"] for v in candidates.values() for c in v)),
        "candidate_area_quantiles": np.quantile([c["area_fraction"] for v in candidates.values() for c in v], [0.0, .25, .5, .75, 1.0]).round(6).tolist(),
    }
    write(out / "pool_summary.json", summary)
    print("POOL_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

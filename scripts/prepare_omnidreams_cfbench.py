"""为冻结24-case生成nuScenes到官方Ludus输入；仅CPU，不读取生成结果选case。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_proj.cfbench.geometry import shifted_pose

PROMPT = "A dash-camera video of a road scene. Preserve the scene appearance and follow the supplied road layout and object motion."
SCENE_NAMES = {str(i): row['name'] for i, row in enumerate(json.loads(
    Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/map_metadata/scene.json').read_text()))}


def pose_at(poses, frame):
    lo, hi = math.floor(frame + 1e-8), math.ceil(frame - 1e-8)
    if lo not in poses or hi not in poses:
        return None
    if lo == hi:
        return poses[lo].copy()
    alpha = frame-lo
    result = np.eye(4)
    result[:3, 3] = poses[lo][:3, 3]*(1-alpha) + poses[hi][:3, 3]*alpha
    result[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([poses[lo][:3, :3], poses[hi][:3, :3]])))([alpha]).as_matrix()[0]
    return result


def changed_pose(case, poses, frame):
    event = case["anchor"]["event_frame"]
    if frame < event:
        return pose_at(poses, frame)
    family = case["intervention"]["family"]
    spec = case["intervention"]["counterfactual"]
    query = event + (frame-event)*spec["speed_scale"] if family == "actor_speed_change" else frame
    pose = pose_at(poses, query)
    if pose is None:
        return None
    if family == "actor_lateral_relocation":
        if case['intervention'].get('coordinate_convention')=='vehicle_forward_left_v2':
            duration_frames=case['intervention']['transition_duration_s']*10
            progress=np.clip((frame-event)/duration_frames,0,1)
            signed_left=spec['lateral_offset_m']*progress**2*(3-2*progress)
            # DriveStudio ego lidar轴为+Y前进/+X向右；actor位姿为+X前进/+Y向左。
            offset=[-signed_left,0,0] if case['target']['role']=='ego' else [0,signed_left,0]
            pose=shifted_pose(pose,offset)
        else:
            # 历史冻结配置使用旧坐标/整段平滑，不改写旧证据。
            progress = np.clip((frame-event)/(case["anchor"]["rollout_frames"]-1), 0, 1)
            pose = shifted_pose(pose, [0, spec["lateral_offset_m"]*progress**2*(3-2*progress), 0])
    if family == "actor_insertion":
        pose = shifted_pose(pose, case["target"]["proposal_offset_actor_frame_m"])
    return pose


def map_subset(data, origin, radius=220):
    nodes = {x["token"]: np.array([x["x"], x["y"], 0.0]) for x in data["node"]}
    lines = {x["token"]: x for x in data["line"]}
    polygons = {x["token"]: x for x in data["polygon"]}
    rows, crossings = [], []
    def points(tokens):
        return np.stack([nodes[token] for token in tokens])
    def near(xyz):
        return bool(np.any(np.linalg.norm(xyz[:, :2]-origin[None, :2], axis=1) < radius))
    for layer in ["lane_divider", "road_divider"]:
        for row in data[layer]:
            line = lines[row["line_token"]]
            xyz = points(line["node_tokens"])
            if not near(xyz):
                continue
            segments = row.get("lane_divider_segments", row.get("road_divider_segments", []))
            marks = {x["node_token"]: x["segment_type"] for x in segments}
            for i in range(len(xyz)-1):
                mark = marks.get(line["node_tokens"][i], "ROAD_BOUNDARY" if layer == "road_divider" else "SINGLE_SOLID_WHITE")
                if mark == "NIL":
                    continue
                rows.append({"kind": "road_boundary" if mark == "ROAD_BOUNDARY" else "lane", "mark": mark, "xyz": (xyz[i:i+2]-origin).tolist()})
    for row in data["ped_crossing"]:
        xyz = points(polygons[row["polygon_token"]]["exterior_node_tokens"])
        if near(xyz):
            crossings.append((xyz-origin).tolist())
    drivable = []
    for row in data["drivable_area"]:
        for token in row["polygon_tokens"]:
            poly = polygons[token]
            xyz = points(poly["exterior_node_tokens"])
            if near(xyz):
                drivable.append({"outer": (xyz[:, :2]-origin[None, :2]).tolist(),
                                 "holes": [(points(h["node_tokens"])[:, :2]-origin[None, :2]).tolist() for h in poly.get("holes", [])]})
    return rows, crossings, drivable


def category(value):
    if value.startswith("human"):
        return "PEDESTRIAN"
    if value in ["vehicle.bicycle", "vehicle.motorcycle"]:
        return "BICYCLIST"
    if "truck" in value or "bus" in value or "trailer" in value or "construction" in value:
        return "TRUCK"
    if value.startswith("vehicle"):
        return "REGULAR_VEHICLE"
    return "OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-frames", type=int, default=77)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    qualification = json.loads(args.qualification.read_text())
    # 兼容资格文件顶层列表命名；每个case仍显式按ID关联。
    qrows = qualification.get("cases", qualification.get("results", []))
    qindex = {x["case_id"]: x for x in qrows}
    maps = json.loads((args.map_root / "scene-map-index.json").read_text())
    map_data = {name: json.loads((args.map_root / f"{name}.json").read_text()) for name in {x["location"] for x in maps.values()}}
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in manifest["cases"]:
        case=item.get('case',item)
        if 'case' in item:
            assert item['approval']['inference_allowed'] and item['approval']['status']=='approved', \
                '新版case尚未人工批准，禁止创建推理输入'
        cid = case["case_id"]
        out = args.output / cid
        if out.exists():
            raise FileExistsError(out)
        out.mkdir()
        scene_root = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
        q = qindex[cid]
        assert q["automatic_status"] in ["geometry_pass_manual_pending", "long_window_input_checked_manual_pending"]
        views = q["gates"]["target_visibility"]["best_views"]
        camera = int(item['camera_index']) if 'case' in item else (0 if case["target"]["role"] == "ego" else int(views[0]["camera"]))
        start = case["anchor"]["event_frame"]-case["anchor"]["pre_frames"]
        stop = case["anchor"]["event_frame"]+case["anchor"]["rollout_frames"]-1
        # 原生分块为首块5帧，其余每块8帧；真实窗口与尾部填充分别登记。
        count = args.output_frames
        assert count >= 5 and (count-5) % 8 == 0
        long_review = case['anchor'].get('video_duration_s')==10.0
        score_frames = 300 if long_review else (stop-start)*3+1
        assert count >= score_frames, "不能截断请求的真实时长"
        if long_review:
            assert count==301, '10秒原生301帧，展示前300帧；不插帧或冻结尾部'
            query_frames=start+np.arange(count)/3
            assert query_frames[-1]<=stop+1
        else:
            query_frames = np.minimum(start+np.arange(count)/3, stop)
        times = 1000000 + np.rint(np.arange(count)*1e6/30).astype(np.int64)
        ego = {int(p.stem): np.loadtxt(p) for p in (scene_root / "lidar_pose").glob("*.txt")}
        cameras = {f: np.loadtxt(scene_root / f"extrinsics/{f:03d}_{camera}.txt") for f in range(start, stop+1+int(long_review))}
        origin = ego[start][:3, 3].copy()
        location = maps[SCENE_NAMES[case["dataset"]["scene_id"]]]["location"]
        lines, crossings, drivable = map_subset(map_data[location], origin)
        K = np.loadtxt(scene_root / f"intrinsics/{camera}.txt")[:4]*np.array([1280/1600, 704/900, 1280/1600, 704/900])
        image = Image.open(scene_root / f"images/{start:03d}_{camera}.jpg").convert("RGB")
        image.resize((1280, 704), Image.Resampling.BOX).save(out / "initial_rgb.png")
        (out / "prompt.txt").write_text(PROMPT+"\n")
        info = json.loads((scene_root / "instances/instances_info.json").read_text())
        target = str(case["target"].get("actor_key", case["target"].get("source_asset_actor_key", "ego")))
        family = case["intervention"]["family"]
        poses_by_branch, branch_scenes = {}, {}
        for branch in ["factual", "counterfactual"]:
            camera_poses = []
            for frame in query_frames:
                cam = pose_at(cameras, frame)
                if branch == "counterfactual" and case["target"]["role"] == "ego":
                    cam = changed_pose(case, ego, frame) @ np.linalg.inv(pose_at(ego, frame)) @ cam
                cam[:3, 3] -= origin
                camera_poses.append(cam)
            poses_by_branch[branch] = np.stack(camera_poses)
            tracks = []
            keys = list(info)
            if family == "actor_insertion" and branch == "counterfactual":
                keys.append("inserted:"+target)
            for key in keys:
                inserted = key.startswith("inserted:")
                source_key = key.split(":", 1)[1] if inserted else key
                row = info[source_key]
                ann = row["frame_annotations"]
                obj_poses = {int(f): np.asarray(p) for f,p in zip(ann["frame_idx"], ann["obj_to_world"])}
                size = np.median(np.asarray(ann["box_size"]), axis=0)
                frames, centers, quats = [], [], []
                for i, frame in enumerate(query_frames):
                    after = frame >= case["anchor"]["event_frame"]
                    if inserted and not after:
                        continue
                    edited = branch == "counterfactual" and case["target"]["role"] != "ego" and (inserted or (source_key == target and family != "actor_insertion"))
                    if edited and family == "actor_removal" and after:
                        continue
                    pose = changed_pose(case, obj_poses, frame) if edited else pose_at(obj_poses, frame)
                    if pose is None:
                        continue
                    frames.append(i)
                    centers.append((pose[:3, 3]-origin).tolist())
                    quats.append(Rotation.from_matrix(pose[:3, :3]).as_quat().tolist())
                if len(frames) >= 2:
                    tracks.append({"actor_key": key, "frames": frames, "centers": centers, "quaternions": quats,
                                   "dimensions": size.tolist(), "category": category(row["class_name"])})
            branch_scenes[branch] = {"lines": lines, "crossings": crossings, "tracks": tracks}
            (out / f"{branch}-scene.json").write_text(json.dumps(branch_scenes[branch])+"\n")
        np.savez(out / "trajectory.npz", timestamps_us=times, source_frames=query_frames, K=K, **poses_by_branch)
        (out / "drivable.json").write_text(json.dumps(drivable)+"\n")
        record = {"case_id": cid, "case": case, "camera_index": camera, "scene_name": SCENE_NAMES[case["dataset"]["scene_id"]],
                  "location": location, "origin_world": origin.tolist(), "target_key": target, "frames": count, "score_frames": score_frames,
                  "task_id": manifest.get("task_id", "WS-V75-DOWNSTREAM-FULL-01"), "run_id": manifest.get("run_id", "20260923-r1"),
                  "score_source_frame_range": [start, stop], "event_output_frame": case["anchor"]["pre_frames"]*3, "fps": 30,
                  "model_view_name_is_not_source_calibration": True,
                  "conditioning_variant": "nuScenes_GT_map_and_track_adapter_not_reconstructed_state",
                  "map_height_assumption": "nuScenes 2D map z=0 in global coordinates",
                  "map_marking_adapter": "double lines rendered with corresponding single-line primitive; NIL omitted",
                  "human_verdict": None, "input_dir": str(out.resolve())}
        (out / "input.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
        records.append(record)
        print(json.dumps({"prepared": cid, "camera": camera, "tracks": {b: len(x["tracks"]) for b,x in branch_scenes.items()}}), flush=True)
    (args.output / "index.json").write_text(json.dumps({"task_id": manifest.get("task_id", "WS-V75-DOWNSTREAM-FULL-01"), "model_id": "omnidreams", "cases": records}, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()

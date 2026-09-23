"""为同三个pilot场景准备HUGSIM原生布局和StreetGS/DriveStudio重建计划。"""
import argparse
import json
from pathlib import Path
import itertools

import numpy as np
from PIL import Image
import open3d as o3d

CAMS = {0: "CAM_FRONT", 1: "CAM_FRONT_LEFT", 2: "CAM_FRONT_RIGHT", 3: "CAM_BACK_LEFT", 4: "CAM_BACK_RIGHT", 5: "CAM_BACK"}
SCENES = {"179": "scene-0230", "191": "scene-0242", "204": "scene-0255"}
Q = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def native_vertices(dim):
    v = np.zeros((8, 3))
    v[:4, 0] = dim[0]/2
    v[4:, 0] = -dim[0]/2
    v[[0, 1, 4, 5], 1] = dim[1]/2
    v[[2, 3, 6, 7], 1] = -dim[1]/2
    v[[0, 2, 5, 7], 2] = dim[2]/2
    v[[1, 3, 4, 6], 2] = -dim[2]/2
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases = json.loads(args.manifest.read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for sid, scene_name in SCENES.items():
        relevant = [x for x in cases if x["dataset"]["scene_id"] == sid]
        source = Path(relevant[0]["dataset"]["root"]) / sid
        out = args.output / "hugsim" / scene_name
        out.mkdir(parents=True)
        rgb_files = sorted((source / "images").glob("*_0.jpg"))
        frame_ids = [int(x.stem.split("_")[0]) for x in rgb_files]
        base = np.loadtxt(source / f"extrinsics/{frame_ids[0]:03d}_0.txt")
        inv = np.linalg.inv(base)
        info = json.loads((source / "instances/instances_info.json").read_text())
        target_ids = {str(c["target"].get("actor_key", c["target"].get("source_asset_actor_key"))) for c in relevant if c["target"]["role"] != "ego"}
        objects, verts = {}, {}
        for key, row in info.items():
            ann = row["frame_annotations"]
            poses = np.array(ann["obj_to_world"])
            # 默认HUGSIM动态对象之外显式保留case目标，停放目标也必须可编辑。
            moving = np.linalg.norm(poses[-1, :3, 3]-poses[0, :3, 3]) > 2
            if not moving and key not in target_ids:
                continue
            if len(poses) < 2:
                continue
            dim = np.median(np.array(ann["box_size"]), axis=0)
            verts[key] = native_vertices(dim[[1, 0, 2]]).tolist()
            objects[key] = {int(f): (inv @ np.asarray(pose) @ Q).tolist() for f, pose in zip(ann["frame_idx"], ann["obj_to_world"])}
            # 坐标置换前后的八角点集合必须相同。
            native = np.asarray(verts[key]) @ Q[:3, :3].T
            reference = np.array(list(itertools.product([-.5, .5], repeat=3)))*dim
            assert max(min(np.linalg.norm(native-x, axis=1)) for x in reference) < 1e-6
        assert target_ids <= set(objects), target_ids-set(objects)
        meta = {"camera_model": "OPENCV", "inv_pose": inv.tolist(), "verts": verts, "frames": []}
        for f in frame_ids:
            dynamics = {key: values[f] for key, values in objects.items() if f in values}
            for cam in [0, 1, 2, 5, 3, 4]:
                name = CAMS[cam]
                folder = out / "images" / name
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{f:05d}.jpg"
                Image.open(source / f"images/{f:03d}_{cam}.jpg").resize((800, 450), Image.Resampling.BOX).save(path, quality=95)
                fx, fy, cx, cy = np.loadtxt(source / f"intrinsics/{cam}.txt")[:4]/2
                K = np.eye(4)
                K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
                pose = inv @ np.loadtxt(source / f"extrinsics/{f:03d}_{cam}.txt")
                meta["frames"].append({"rgb_path": f"./images/{name}/{f:05d}.jpg", "camtoworld": pose.tolist(), "intrinsics": K.tolist(),
                    "width": 800, "height": 450, "timestamp": (f-frame_ids[0])*.1, "dynamics": dynamics})
        save(out / "meta_data.json", meta)
        local = np.fromfile(source / f"lidar/{frame_ids[0]:03d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
        l2w = np.loadtxt(source / f"lidar_pose/{frame_ids[0]:03d}.txt")
        world = local @ l2w[:3, :3].T+l2w[:3, 3]
        nearby = world[np.linalg.norm(world[:, :2]-base[None, :2, 3], axis=1) < 12]
        nearby = nearby[nearby[:, 2] <= np.quantile(nearby[:, 2], .5)]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(nearby)
        o3d.utility.random.seed(42)
        plane, inliers = pc.segment_plane(distance_threshold=.03, ransac_n=3, num_iterations=2000)
        plane = np.array(plane)
        if plane[2] < 0:
            plane *= -1
        assert plane[2] > .95 and len(inliers) > 100, (plane, len(inliers))
        height = float(np.dot(plane[:3], base[:3, 3])+plane[3])
        assert 1 < height < 3, height
        save(out / "front_info.json", {"height": height, "rect_mat": np.eye(3).tolist(), "source": "local LiDAR ground plane; rect_mat not used by upstream ground merge"})
        record = {"scene_id": sid, "scene_name": scene_name, "source": str(source), "hugsim_input": str(out),
                  "frame_count": len(frame_ids), "cameras": 6, "fps": 10, "target_ids": sorted(target_ids),
                  "actor_count": len(objects), "ground_plane": plane.tolist(), "ground_inliers": len(inliers), "camera_height_m": height,
                  "status": "rgb_pose_track_prepared_semantics_depth_reconstruction_pending",
                  "variant": "official HUGSIM reconstruction with DriveStudio10Hz input adapter; all case targets retained as editable actors",
                  "hugsim_split": "native idx%30>=24 = source frame%5==4 holdout; factual report separates train/holdout",
                  "human_verdict": None}
        save(out / "adapter-contract.json", record)
        records.append(record)
        print(json.dumps(record), flush=True)
    save(args.output / "index.json", {"task_id": "WS-V75-DOWNSTREAM-FULL-01", "scenes": records, "generation_calls": 0, "training_started": False})


if __name__ == "__main__":
    main()

"""将现存 DriveStudio 10 Hz 场景导入 DriverQ SQLite 查询表（CPU）。

这是针对已预处理 train 场景的轻量适配，不冒充 DriverQ 官方 nuScenes
原始元数据 exporter：只填 scenes/poses/trajectories/几何可见性四类表。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_proj.cfbench.geometry import box_corners_world

SCENES_META = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/map_metadata/scene.json')
LOGS_META = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/map_metadata/log.json')
ROOTS = [
    Path('/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval'),
    Path('/root/autodl-tmp/data/worldsim_v5/drivestudio_processed_10Hz/trainval'),
]
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']


def quat(pose):
    x, y, z, w = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return (float(w), float(x), float(y), float(z))


def kinematics(frames, poses):
    speeds = [0.0]
    for a, b in zip(poses, poses[1:]):
        speeds.append(float(np.linalg.norm(b[:2, 3] - a[:2, 3]) * 10 / (frames[len(speeds)] - frames[len(speeds)-1])))
    accel = [0.0]
    for i in range(1, len(frames)):
        accel.append((speeds[i] - speeds[i-1]) * 10 / (frames[i] - frames[i-1]))
    return speeds, accel


def box_in_image(world_to_cam, intrinsics, pose, size):
    xyz = world_to_cam @ box_corners_world(pose, size)
    if np.any(xyz[2] <= .1):
        return None
    fx, fy, cx, cy = intrinsics[:4]
    u = fx * xyz[0] / xyz[2] + cx
    v = fy * xyz[1] / xyz[2] + cy
    left, top = max(0., float(u.min())), max(0., float(v.min()))
    right, bottom = min(1600., float(u.max())), min(900., float(v.max()))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def scene_rows(conn, root, scene, log_by_token, *, step=5):
    scene_dir = root / f"{scene['index']:03d}"
    if not (scene_dir / 'images').is_dir():
        return None
    pose_paths = sorted((scene_dir / 'lidar_pose').glob('*.txt'))
    frames = [int(p.stem) for p in pose_paths]
    if len(frames) < 100:
        return None
    token = scene['token']
    location = log_by_token[scene['log_token']]['location']
    conn.execute('INSERT INTO scenes VALUES (?,?,?,?,?,?)',
                 (token, scene['name'], location, len(frames), 'DriveStudio-processed10Hz', root.parent.name))
    ego = {f: np.loadtxt(path) for f, path in zip(frames, pose_paths)}
    speeds, accelerations = kinematics(frames, list(ego.values()))
    conn.executemany('INSERT INTO ego_poses (scene_token,frame_idx,timestamp,ego_x,ego_y,ego_z,ego_qw,ego_qx,ego_qy,ego_qz,ego_speed,ego_accel) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        [(token, f, f*100000, *map(float, ego[f][:3,3]), *quat(ego[f]), speeds[i], accelerations[i])
         for i,f in enumerate(frames)])
    info = json.loads((scene_dir / 'instances/instances_info.json').read_text())
    positions = {}
    objects = 0
    for key, item in info.items():
        ann = item['frame_annotations']
        ff = [int(f) for f in ann['frame_idx']]
        poses = [np.asarray(p) for p in ann['obj_to_world']]
        sizes = [np.asarray(s) for s in ann['box_size']]
        if not ff:
            continue
        track_id = item['id']
        category = item['class_name']
        vs, acc = kinematics(ff, poses)
        for i, f in enumerate(ff):
            if f not in ego:
                continue
            p, size = poses[i], sizes[i]
            x, y, z = map(float, p[:3,3])
            length, width, height = map(float, size)
            conn.execute('INSERT INTO object_poses (scene_token,frame_idx,instance_token,category,x,y,z,qw,qx,qy,qz,width,length,height,speed,accel) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (token,f,track_id,category,x,y,z,*quat(p),width,length,height,vs[i],acc[i]))
            if f % step == 0:
                positions.setdefault(f, []).append((track_id, p, size))
        trajectory = [[int(f),float(p[0,3]),float(p[1,3]),float(p[2,3])] for f,p in zip(ff,poses)]
        conn.execute('INSERT INTO object_trajectories VALUES (?,?,?,?,?,?)',
                     (token,track_id,category,min(ff),max(ff),json.dumps(trajectory,separators=(',',':'))))
        objects += 1
    vis = 0
    intrinsics = [np.loadtxt(scene_dir/'intrinsics'/f'{c}.txt') for c in range(6)]
    for f, actors in positions.items():
        for cam in range(6):
            if not (scene_dir/'images'/f'{f:03d}_{cam}.jpg').is_file():
                continue
            world_to_cam = np.linalg.inv(np.loadtxt(scene_dir/'extrinsics'/f'{f:03d}_{cam}.txt'))
            for track_id, p, size in actors:
                box = box_in_image(world_to_cam, intrinsics[cam], p, size)
                if box is None:
                    continue
                conn.execute('INSERT INTO visibility (scene_token,frame_idx,instance_token,camera,visibility_level,bbox_x1,bbox_y1,bbox_x2,bbox_y2) VALUES (?,?,?,?,?,?,?,?,?)',
                             (token,f,track_id,CAMERAS[cam],None,*box))
                vis += 1
    conn.commit()
    return {'scene':scene['name'],'scene_index':scene['index'],'root':str(root),
            'frames':len(frames),'objects':objects,'projected_visibility_rows':vis}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driverq',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Write to a new DB; preserve earlier mining evidence.'
    sys.path.insert(0,str(args.driverq/'exporter'))
    from schema import SCHEMA, INDEXES
    scenes = json.loads(SCENES_META.read_text())
    logs = {x['token']:x for x in json.loads(LOGS_META.read_text())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    conn = sqlite3.connect(args.output)
    conn.executescript(SCHEMA)
    summary = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for scene_dir in sorted(root.iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.isdigit():
                continue
            index = int(scene_dir.name)
            row = {**scenes[index], 'index':index}
            record = scene_rows(conn,root,row,logs)
            if record:
                summary.append(record)
                print(json.dumps(record,ensure_ascii=False),flush=True)
    conn.executescript(INDEXES)
    conn.execute('ANALYZE')
    conn.close()
    (args.output.parent/'adapter-provenance.json').write_text(json.dumps({
        'driverq_repo':str(args.driverq),'source_type':'DriveStudio processed 10Hz',
        'official_nuscenes_exporter_used':False,'visibility_level':None,
        'visibility_semantics':'3D box projected into camera, not verified unoccluded pixel area',
        'timestamp_semantics':'frame_idx times 0.1 s relative, not original sensor timestamp',
        'map_and_event_tables':'empty; requires raw nuScenes map exporter',
        'scenes':summary},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'scene_count':len(summary),'db':str(args.output),'model_calls':0}),flush=True)


if __name__=='__main__':
    main()

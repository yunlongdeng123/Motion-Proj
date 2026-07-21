#!/usr/bin/env python
"""O0 scene-local occupancy from LiDAR + 3D boxes (no Occ3D download, no learned predictor).

Builds per-frame occupancy grids for OccGS D0 scenes:
  occupied / free / unknown  (+ optional vehicle instance id overlay)

Grid convention (ego-centric at frame 0 / first lidar pose of the window):
  x forward, y left, z up; range matching Occ3D-nuScenes for optional later compare:
  [-40,40] x [-40,40] x [-1, 5.4] m, voxel 0.4 m → (200, 200, 16).

Unknown is preserved: voxels never hit by a LiDAR ray stay unknown (not free).
Free voxels = ray-cleared empty space (coarse: between sensor and first hit along ray,
approximated by carving a cone of free cells up to each return).
Occupied = LiDAR returns + vehicle box voxels (dynamic panoptic).

Outputs under /root/autodl-tmp/data/occgs/occupancy/{scene:03d}/:
  meta.json
  frame_{t:03d}.npz  with keys semantics(u8), mask_lidar(u8), instance_id(i32)
    semantics: 0=unknown, 1=free, 2=static_occ, 3=dynamic_vehicle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path("/root/autodl-tmp/data/occgs/processed_10Hz/mini")
OUT = Path("/root/autodl-tmp/data/occgs/occupancy")

X_MIN, Y_MIN, Z_MIN = -40.0, -40.0, -1.0
X_MAX, Y_MAX, Z_MAX = 40.0, 40.0, 5.4
VOX = 0.4
NX = int(round((X_MAX - X_MIN) / VOX))  # 200
NY = int(round((Y_MAX - Y_MIN) / VOX))  # 200
NZ = int(round((Z_MAX - Z_MIN) / VOX))  # 16

UNK, FREE, STATIC, DYNAMIC = 0, 1, 2, 3


def world_to_voxel(xyz: np.ndarray) -> np.ndarray:
    """xyz (N,3) world → integer voxel indices (N,3) in [ix,iy,iz]. Invalid → -1."""
    ix = np.floor((xyz[:, 0] - X_MIN) / VOX).astype(np.int32)
    iy = np.floor((xyz[:, 1] - Y_MIN) / VOX).astype(np.int32)
    iz = np.floor((xyz[:, 2] - Z_MIN) / VOX).astype(np.int32)
    valid = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
    out = np.stack([ix, iy, iz], axis=1)
    out[~valid] = -1
    return out


def load_lidar_bin(path: Path) -> np.ndarray:
    # DriveStudio nuScenes lidar: float32 x,y,z,intensity (possibly more)
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 4 == 0:
        pts = arr.reshape(-1, 4)[:, :3]
    elif arr.size % 5 == 0:
        pts = arr.reshape(-1, 5)[:, :3]
    else:
        pts = arr.reshape(-1, 3)
    return pts


def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(4, 4)


def box_corners(center, size, yaw):
    l, w, h = size
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    z = np.array([h, h, h, h, -h, -h, -h, -h]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ np.stack([x, y, z])) .T + center


def rasterize_box(sem, inst, center, size, yaw, instance_id, origin_pose_inv):
    """Fill voxels overlapping the oriented box (coarse AABB of corners in grid)."""
    corners = box_corners(np.asarray(center), np.asarray(size), float(yaw))
    # to grid frame (first-frame ego)
    ones = np.ones((8, 1))
    corners_h = np.concatenate([corners, ones], axis=1)
    local = (origin_pose_inv @ corners_h.T).T[:, :3]
    vox = world_to_voxel(local)
    valid = vox[:, 0] >= 0
    if not valid.any():
        return
    v = vox[valid]
    x0, x1 = v[:, 0].min(), v[:, 0].max()
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    sem[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = DYNAMIC
    inst[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = instance_id


def carve_free_and_occupy(sem, mask_lidar, pts_local, sensor_local, max_free_steps=200):
    """Mark lidar hits as STATIC (if not already DYNAMIC) and carve FREE along rays."""
    if len(pts_local) == 0:
        return
    vox = world_to_voxel(pts_local)
    valid = vox[:, 0] >= 0
    vox = vox[valid]
    pts = pts_local[valid]
    # occupy
    for ix, iy, iz in vox:
        if sem[ix, iy, iz] != DYNAMIC:
            sem[ix, iy, iz] = STATIC
        mask_lidar[ix, iy, iz] = 1
    # free carving (coarse DDA-lite): subsample points for speed
    step = max(1, len(pts) // 4000)
    origin = sensor_local
    for p in pts[::step]:
        d = p - origin
        dist = np.linalg.norm(d)
        if dist < 1e-3:
            continue
        direction = d / dist
        n_steps = int(min(max_free_steps, dist / VOX))
        for s in range(1, n_steps):
            q = origin + direction * (s * VOX)
            vv = world_to_voxel(q[None])[0]
            if vv[0] < 0:
                break
            if sem[vv[0], vv[1], vv[2]] in (STATIC, DYNAMIC):
                break
            if sem[vv[0], vv[1], vv[2]] == UNK:
                sem[vv[0], vv[1], vv[2]] = FREE
            mask_lidar[vv[0], vv[1], vv[2]] = 1


def quat_to_yaw(qw, qx, qy, qz):
    # yaw from quaternion (nuScenes / SciPy style w,x,y,z)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def process_scene(sid: int, start: int, end: int):
    d = ROOT / f"{sid:03d}"
    out = OUT / f"{sid:03d}"
    out.mkdir(parents=True, exist_ok=True)
    with open(d / "instances" / "instances_info.json") as f:
        inst_info = json.load(f)
    meta = dict(
        scene_idx=sid,
        start=start,
        end=end,
        frame="ego_centric_per_frame",
        grid=dict(xmin=X_MIN, ymin=Y_MIN, zmin=Z_MIN, xmax=X_MAX, ymax=Y_MAX, zmax=Z_MAX, voxel=VOX, shape=[NX, NY, NZ]),
        semantics={0: "unknown", 1: "free", 2: "static_occ", 3: "dynamic_vehicle"},
        note="per-frame ego-centric; unknown preserved; free only along lidar rays; dynamic from 3D boxes",
    )
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    per_frame = {}
    for iid_str, info in inst_info.items():
        if not info.get("class_name", "").startswith("vehicle"):
            continue
        fa = info["frame_annotations"]
        for i, fi in enumerate(fa["frame_idx"]):
            if fi < start or fi > end:
                continue
            T = np.array(fa["obj_to_world"][i])
            size = fa["box_size"][i]
            center = T[:3, 3]
            yaw = float(np.arctan2(T[1, 0], T[0, 0]))
            per_frame.setdefault(fi, []).append((int(iid_str) + 1, center, size, yaw))

    summary = []
    for t in tqdm(range(start, end + 1), desc=f"occ[{sid:03d}]"):
        sem = np.zeros((NX, NY, NZ), dtype=np.uint8)
        mask = np.zeros((NX, NY, NZ), dtype=np.uint8)
        inst = np.zeros((NX, NY, NZ), dtype=np.int32)
        pose = load_pose(d / "lidar_pose" / f"{t:03d}.txt")
        pose_inv = np.linalg.inv(pose)
        for iid, center, size, yaw in per_frame.get(t, []):
            rasterize_box(sem, inst, center, size, yaw, iid, pose_inv)
        pts = load_lidar_bin(d / "lidar" / f"{t:03d}.bin")
        # lidar bin is already in lidar/sensor frame for DriveStudio nuScenes
        # Confirm: save_lidar typically stores sensor-frame points. Transform via pose to world then back is identity if bin is sensor-local.
        # Empirically DriveStudio stores points already transformed? Check both: if mean radius huge, treat as world.
        pts_check = pts
        radii = np.linalg.norm(pts_check, axis=1)
        if float(np.median(radii)) > 200:  # likely world coords
            pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
            pts_local = (pose_inv @ pts_h.T).T[:, :3]
        else:
            pts_local = pts
        sensor_local = np.zeros(3)
        carve_free_and_occupy(sem, mask, pts_local, sensor_local)
        np.savez_compressed(out / f"frame_{t:03d}.npz", semantics=sem, mask_lidar=mask, instance_id=inst)
        counts = {k: int((sem == v).sum()) for k, v in [("unk", UNK), ("free", FREE), ("static", STATIC), ("dyn", DYNAMIC)]}
        summary.append(dict(frame=t, **counts, n_boxes=len(per_frame.get(t, []))))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_ids", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=79)
    args = ap.parse_args()
    for sid in args.scene_ids:
        process_scene(sid, args.start, args.end)


if __name__ == "__main__":
    main()

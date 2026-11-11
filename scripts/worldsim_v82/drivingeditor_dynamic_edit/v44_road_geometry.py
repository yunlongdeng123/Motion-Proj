"""测距道路几何准备；只使用相机/轨迹/掩码/道路标签/测距，不读取颜色或外观ROI。

初始测量面由显式seed_reference提供；自动扩展至全部目标地面射线的有限范围，
并用道路测距重新拟合局部面。该面只是后续8cm局部Surface的初始化，不是稠密GT。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from .v44_road import project
from .v44_handoff_policy import target_mask_from_labels


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def query_geometry(bank, observation, frame_id, anchor, plane, wh):
    """与原生相机相同的 pinhole 查询光线；不采样RGB。"""
    width, height = wh
    lines = (bank / 'calib' / (frame_id + '.txt')).read_text().splitlines()
    cal = {s.split(':')[0]: np.fromstring(s.split(':')[1], sep=' ') for s in lines}
    native_width = Image.open(bank / observation['image_path']).width
    scale = width / native_width
    intrinsic = cal['P0'].reshape(3, 4)[:, :3] * scale
    intrinsic[2, 2] = 1
    camera = np.eye(4)
    camera[:3] = cal['Tr_velo_to_cam_0'].reshape(3, 4)
    camera = camera @ np.linalg.inv(np.array(observation['ego_world']).reshape(4, 4)) @ anchor
    inverse = np.linalg.inv(camera)
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack((xx + .5, yy + .5, np.ones_like(xx)), 2) @ np.linalg.inv(intrinsic).T
    rays = rays @ inverse[:3, :3].T
    origin = inverse[:3, 3]
    denominator = rays[:, :, 2] - rays[:, :, :2] @ plane[:2]
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (plane[2] + plane[:2] @ origin[:2] - origin[2]) / denominator
    hit = origin + depth[:, :, None] * rays
    return hit, depth, np.linalg.norm(hit - origin, axis=2)


def build(config, output):
    started = time.monotonic()
    cfg = read(config)
    if cfg.get('schema') != 'v44.road_geometry.v1':
        raise ValueError('Explicit road geometry config required')
    out = Path(output); out.mkdir(exist_ok=False)
    bank, masks, semantics = (Path(cfg[k]) for k in ('bank', 'masks', 'semantics'))
    target = cfg['actor_id']; track = cfg['track_id']; width, height = cfg['render_wh']
    receipt = read(bank / 'receipt.json')
    if receipt['holdout_raws']:
        raise ValueError('This run contract uses all frames; unexpected heldout bank')
    raws = receipt['build_raws']; ids = {r['raw']: r['id'] for r in receipt['frames']}
    observations = {r: read(bank / 'observations' / f'{r:03d}.json') for r in raws}
    boxes = [b for d in observations.values() for b in d['boxes'] if b['track'] == track]
    if not boxes: raise ValueError('No registered target trajectory')
    anchor = np.array(boxes[0]['world_from_actor']).reshape(4, 4)
    anchor[:3, 3] = np.median([np.array(b['world_from_actor']).reshape(4, 4)[:3, 3] for b in boxes], axis=0)
    inverse = np.linalg.inv(anchor)
    with np.load(cfg['seed_reference']) as seed:
        seed_xyz = seed['local_xyz'].copy()
    plane = np.linalg.lstsq(np.c_[seed_xyz[:, :2], np.ones(len(seed_xyz))], seed_xyz[:, 2], rcond=None)[0]
    if np.max(abs(np.c_[seed_xyz[:, :2], np.ones(len(seed_xyz))] @ plane - seed_xyz[:, 2])) > 1e-5:
        raise ValueError('Seed chart is not one measured tangent plane')
    cached = {}; requested = []
    for raw, d in observations.items():
        labels = np.asarray(Image.open(masks / f'{raw:03d}' / 'labels.png'))
        if labels.shape != (height, width): raise ValueError('Accepted label resolution mismatch')
        semantic = np.asarray(Image.open(semantics / f'{raw:03d}.png').resize((width, height), Image.Resampling.NEAREST))
        foreground = np.asarray(Image.open(masks / f'{raw:03d}' / 'dynamic_union.png')) > 0
        guard = binary_dilation(foreground | (labels > 0), iterations=cfg['source_guard_pixels'])
        hit, depth, distance = query_geometry(bank, d, ids[raw], anchor, plane, (width, height))
        finite = np.isfinite(hit).all(2) & (depth > 0) & (distance <= cfg['max_ground_distance_m'])
        target_mask = target_mask_from_labels(labels, [target])
        requested.append(hit[target_mask & finite, :2])
        cached[raw] = dict(labels=labels, semantic=semantic, foreground=foreground, guard=guard,
            hit=hit, depth=depth, finite=finite, target=target_mask)
    # 有限地面射线并集，避免近地平线产生无界图表。
    request_xy = np.concatenate(requested)
    cell = float(cfg['grid_m']); pad = cfg['chart_margin_m']
    low = request_xy.min(0) - pad; high = request_xy.max(0) + pad
    road_returns = []
    for raw, d in observations.items():
        cache = cached[raw]; cloud = np.fromfile(bank / d['lidar_path'], dtype='<f4').reshape(-1, 6)[:, :3]
        ego = np.array(d['ego_world']).reshape(4, 4); measured = cloud @ ego[:3, :3].T + ego[:3, 3]
        uv, z = project(measured, d); scale = width / Image.open(bank / d['image_path']).width
        pix = np.floor(uv * scale).astype(int)
        inside = (z > .1) & (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
        indices = np.flatnonzero(inside); p = pix[indices]
        nearest = np.full((height, width), np.inf); np.minimum.at(nearest, (p[:, 1], p[:, 0]), z[indices])
        good = (cache['semantic'][p[:, 1], p[:, 0]] == cfg['road_semantic_id']) & ~cache['guard'][p[:, 1], p[:, 0]]
        good &= z[indices] <= nearest[p[:, 1], p[:, 0]] + .3
        local = measured[indices[good]] @ inverse[:3, :3].T + inverse[:3, 3]
        within = (local[:, :2] >= low - 1).all(1) & (local[:, :2] <= high + 1).all(1)
        within &= abs(local[:, 2] - np.c_[local[:, :2], np.ones(len(local))] @ plane) < cfg['plane_initial_tolerance_m']
        local = local[within]
        road_returns.append(local)
    road_returns = np.concatenate(road_returns)
    design = np.c_[road_returns[:, :2], np.ones(len(road_returns))]
    plane = np.linalg.lstsq(design, road_returns[:, 2], rcond=None)[0]
    for _ in range(5):
        residual = design @ plane - road_returns[:, 2]
        weight = np.minimum(1, .05 / np.maximum(abs(residual), 1e-8))
        plane = np.linalg.lstsq(design * weight[:, None]**.5, road_returns[:, 2] * weight**.5, rcond=None)[0]
    residual = abs(design @ plane - road_returns[:, 2]); plane_inliers = residual < .12
    if plane_inliers.sum() < 100: raise ValueError('Insufficient measured ground plane inliers')
    requested = []
    for raw, cache in cached.items():
        hit, depth, distance = query_geometry(bank, observations[raw], ids[raw], anchor, plane, (width, height))
        finite = np.isfinite(hit).all(2) & (depth > 0) & (distance <= cfg['max_ground_distance_m'])
        cache.update(hit=hit, depth=depth, finite=finite)
        requested.append(hit[cache['target'] & finite, :2])
    request_xy = np.concatenate(requested); low = request_xy.min(0) - pad; high = request_xy.max(0) + pad
    axes = [np.arange(int(np.floor(low[i] / cell)), int(np.ceil(high[i] / cell)) + 1) for i in range(2)]
    kx, ky = np.meshgrid(*axes, indexing='ij'); keys = np.c_[kx.ravel(), ky.ravel()]
    shape = kx.shape; xy = (keys + .5) * cell; count = len(keys)
    if count > cfg['max_chart_cells']: raise ValueError(f'Road chart {count} exceeds explicit memory budget')
    xyz = np.c_[xy, np.c_[xy, np.ones(count)] @ plane]
    world = xyz @ anchor[:3, :3].T + anchor[:3, 3]
    print('GEOMETRY', count, low.tolist(), high.tolist(), 'plane', plane.tolist(), flush=True)
    distance_to_measure, _ = cKDTree(road_returns[plane_inliers, :2]).query(xy)
    transform = np.loadtxt(cfg['world_to_model'])
    np.savez_compressed(out/'geometry.npz', keys=keys, local_xyz=xyz, world_xyz=world, model_from_world=transform,
        measured_distance_m=distance_to_measure.astype(np.float32), plane=plane, coordinate_anchor=anchor)
    write(out/'receipt.json', dict(status='BUILT_MEASURED_QUERY_ROAD_GEOMETRY_NO_RGB', cells=count, grid_m=cell,
        plane=plane.tolist(), measured_points=len(road_returns), plane_inliers=int(plane_inliers.sum()),
        plane_mae_m=float(residual[plane_inliers].mean()), low_xy=low.tolist(), high_xy=high.tolist(),
        max_ground_distance_m=cfg['max_ground_distance_m'], source_rgb_or_prior_used=False,
        frames=raws, heldout=[], seconds=time.monotonic()-started, config=cfg))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();build(args.config,args.output)


if __name__=='__main__':main()

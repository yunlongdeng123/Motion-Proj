"""历史 R2 复现入口；当前 R5 使用 prepare_v41_revision.py 复核框空角。"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, map_coordinates
from v41_dense_geometry import (MASK_VERSION, MASK_POLICY, FixedSurface, transform, project,
    measured_patch_evidence, ray_box_blocked, finalize_masks)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def masks(frame):
    paths = frame['occluder_masks']['paths']
    result = np.zeros((320, 480), bool)
    for name in paths:
        if name.endswith('.npz'):
            with np.load(name, allow_pickle=False) as archive:
                value = archive['masks'].astype(bool).any(0)
            value = Image.fromarray(value)
        else:
            value = Image.open(name)
        array = np.asarray(value.resize((480, 320), Image.Resampling.NEAREST))
        if array.ndim == 3:
            array = array.any(2)
        result |= array > 0
    return binary_dilation(result, iterations=MASK_POLICY['boundary_margin_px'])


def cloud_and_boxes(source, fid, anchor_inverse, plane):
    pose = np.loadtxt(source/'pose'/f'{fid}.txt')
    cloud = np.fromfile(source/'velodyne'/f'{fid}.bin', np.float32).reshape(-1, 6)[:, :3].astype(float)
    world = transform(cloud, pose)
    boxes, occupied = [], np.zeros(len(world), bool)
    for annotation in read(source/'annotations'/f'{fid}.json')['laser_labels']:
        b = annotation['box']
        if min(b[k] for k in ('length', 'width', 'height')) <= 0:
            continue
        c, s = np.cos(b['heading']), np.sin(b['heading'])
        axes = pose[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        box = dict(center=transform([[b['center_x'], b['center_y'], b['center_z']]], pose)[0],
            half=np.array([b['length'], b['width'], b['height']])/2, axes=axes)
        boxes.append(box)
        occupied |= (np.abs((world-box['center']) @ axes) <= box['half']).all(1)
    local = transform(world, anchor_inverse)
    ground = ((np.abs(local @ plane[:3]+plane[3]) <= .15) & ~occupied
        & (local[:, 0] >= 3) & (local[:, 0] <= 90) & (np.abs(local[:, 1]) <= 8))
    return world, ground, boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('密集证据只写新目录，保留所有旧输入')
    meta = read(args.bundle)
    if len(meta['frame_order']) != 49:
        raise ValueError('必须使用原49帧')
    with np.load(args.bundle.parent/meta['units'], allow_pickle=False) as archive:
        units = {key: archive[key] for key in archive.files}
    surface = FixedSurface(meta, units)
    readiness = read(meta['original_readiness'])
    plane = np.asarray(readiness['SGCM_geometry']['ground_fit']['plane_anchor'])
    source = Path(meta['original_engineering']).parent/'inputs/scene0008_089_137'
    args.output.mkdir(parents=True)
    (args.output/'frames').mkdir(); (args.output/'overlays').mkdir()
    candidates, singles, depths, protections, records = [], [], [], [], []
    for i, fid in enumerate(meta['frame_order']):
        frame = meta['frames'][fid]
        K, w2c = np.asarray(frame['intrinsic']), np.asarray(frame['world_to_camera'])
        domain = surface.camera(K, w2c)
        current_occluders = masks(frame)
        world, ground, boxes = cloud_and_boxes(source, fid, surface.inverse, plane)
        query = np.flatnonzero(domain['supported'])
        normals = units['normal_world'][np.maximum(domain['unit_id'][query], 0)]
        evidence, measured_depth, triangle, conflict = measured_patch_evidence(world, ground, K, w2c,
            domain['world'][query], domain['pixels'][query], normals)
        blocked = ray_box_blocked(domain['world'][query], np.linalg.inv(w2c)[:3, 3], boxes)
        single = np.zeros(320*480, bool)
        single[query] = evidence & ~blocked & ~current_occluders.ravel()[query]
        reference = np.zeros(320*480, np.float32)
        reference[query] = np.where(single[query], measured_depth, 0)
        protection = ~domain['potential'] | current_occluders.ravel()
        protection[query] |= blocked | conflict
        single &= ~protection
        # 几何确认可使用整片受支持曲面；训练仍限制原 observed 单元，不升级 UNKNOWN/PRIOR。
        candidates.append(domain)
        singles.append(single.reshape(320, 480))
        depths.append(reference.reshape(320, 480))
        protections.append(protection.reshape(320, 480))
        provenance = np.full((320*480, 3), -1, np.int32)
        provenance[query] = np.where(single[query, None], triangle, -1)
        np.savez_compressed(args.output/'frames'/f'{fid}_evidence.npz',
            source_lidar_triangle_indices=provenance.reshape(320, 480, 3),
            measured_patch_depth_m=reference.reshape(320, 480),
            single_view_valid=single.reshape(320, 480))
        records.append(dict(frame=fid, old_reliable_centers=int((units['base_weight'][i, surface.observed] > 0).sum()),
            supported_candidate_pixels=int(domain['supported'].sum()),
            observed_candidate_pixels=int(domain['observed'].sum()),
            single_view_patch_pixels=int(single.sum())))
        print(json.dumps(dict(stage='current_frame_evidence', frame=fid, **{k:v for k,v in records[-1].items() if k != 'frame'})), flush=True)
    for i, fid in enumerate(meta['frame_order']):
        domain = candidates[i]
        selected = np.flatnonzero(singles[i].ravel() & domain['observed'])
        counts = np.zeros(320*480, np.uint8)
        counts[selected] = 1
        for j, other in enumerate(meta['frame_order']):
            if i == j:
                continue
            pending = selected[counts[selected] < 2]
            if not len(pending):
                break
            fm = meta['frames'][other]
            uv, z = project(domain['world'][pending], np.asarray(fm['intrinsic']), np.asarray(fm['world_to_camera']))
            coordinates = [uv[:, 1], uv[:, 0]]
            valid = map_coordinates(singles[j].astype(float), coordinates, order=1, mode='constant', cval=0) >= 1-1e-7
            d = map_coordinates(depths[j], coordinates, order=1, mode='constant', cval=0)
            counts[pending] += valid & (z > .1) & (np.abs(d-z) <= .3)
        single = singles[i] & domain['observed'].reshape(320, 480)
        visible, windows = finalize_masks(single, counts.reshape(320, 480), protections[i])
        np.savez_compressed(args.output/'frames'/f'{fid}.npz', visible=visible,
            protect=protections[i], ssim_valid=windows, candidate=domain['observed'].reshape(320, 480),
            supported=domain['supported'].reshape(320, 480), unit_id=domain['unit_id'].reshape(320, 480),
            source_count_capped_two=counts.reshape(320, 480), fixed_surface_depth_m=np.where(
                domain['supported'], domain['depth'], 0).reshape(320, 480).astype(np.float32))
        records[i].update(visible_pixels=int(visible.sum()), ssim_windows=int(windows.sum()),
            protect_pixels=int(protections[i].sum()), supervised_units=int(np.unique(domain['unit_id'][visible.ravel()]).size),
            unknown_or_prior_supervised=int((visible.ravel() & ~domain['observed']).sum()))
        image = np.asarray(Image.open(meta['frames'][fid]['original_rgb']).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS)).copy()
        overlay = image.astype(float)
        for mask, color, mix in ((protections[i], [220, 50, 50], .25),
                                 (domain['observed'].reshape(320, 480) & ~visible, [255, 180, 0], .5),
                                 (visible, [0, 255, 70], .6), (windows, [0, 220, 255], .7)):
            overlay[mask] = overlay[mask]*(1-mix)+np.asarray(color)*mix
        Image.fromarray(np.uint8(overlay.clip(0, 255))).save(args.output/'overlays'/f'{fid}.png')
        print(json.dumps(dict(stage='multiview_final', **records[i])), flush=True)
    result = dict(version=MASK_VERSION, policy=MASK_POLICY, source_bundle=str(args.bundle.resolve()),
        frame_order=meta['frame_order'], frames=records,
        totals={key: sum(r[key] for r in records) for key in ('old_reliable_centers', 'observed_candidate_pixels',
            'visible_pixels', 'ssim_windows', 'protect_pixels', 'unknown_or_prior_supervised')},
        input_roles=dict(RGB='same-frame native camera original_image at training; overlay is display only',
            LiDAR='measured small triangles + fixed-surface agreement; interpolated depth explicitly derived',
            masks='five selected + all vehicles + native proposals, dilated; all annotated 3D boxes checked',
            cross_view='at least two current-frame measured patches agree; no color-based rejection',
            road='geometric ground candidate, not dense semantic ground truth'),
        CPU_seconds=time.perf_counter()-started, GPU_used=False, optimizer_updates=0,
        quality_accepted=False, human_verdict=None)
    result['dense_domain_ready'] = all(r['visible_pixels'] > r['old_reliable_centers'] and r['ssim_windows'] > 0 for r in records)
    write(args.output/'manifest.json', result)
    print(json.dumps({key:result[key] for key in ('dense_domain_ready', 'totals', 'CPU_seconds')}), flush=True)


if __name__ == '__main__':
    main()

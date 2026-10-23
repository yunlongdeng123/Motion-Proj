"""R5：框空角复核及有双源真实观测的删车监督，CPU、固定道路几何。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from prepare_v41_dense import read, write, masks, cloud_and_boxes
from v41_dense_geometry import (MASK_POLICY, FixedSurface, measured_patch_evidence,
    ray_box_blocked, project, finalize_masks)

VERSION = 'V41_FIXED_DENSE_EVIDENCE_R5'
REVISION_POLICY = dict(
    box='candidate flag, never sufficient evidence for actual silhouette occlusion',
    override='current measured patch outside vehicle guard AND another box-clear guarded measured patch',
    witness='bilinear footprint entirely valid; native depth agrees within 0.3m; distinct source frame',
    deletion='all_dynamic, same as existing default evaluation; static remains untouched',
    deletion_target='supported R-observed surface under selected vehicle silhouette, two OTHER box-clear views',
    deletion_color='mean of first two geometric witnesses, only if their RGB MAE <= 0.05; no target-frame RGB',
    deletion_source_color_tolerance=.05, witness_selection='ascending original frame order; no best-color search',
    unknown='no source, no label; no upgrade of prior/unknown units',
    outside_reliable='full composite matches frozen current V1; vehicles separately normalized')


def witness_at(world, frame, valid, depth):
    uv, z = project(world, np.asarray(frame['intrinsic']), np.asarray(frame['world_to_camera']))
    coordinates = [uv[:, 1], uv[:, 0]]
    accepted = map_coordinates(valid.astype(float), coordinates, order=1, mode='constant', cval=0) >= 1-1e-7
    measured = map_coordinates(depth, coordinates, order=1, mode='constant', cval=0)
    accepted &= (z > .1) & (np.abs(measured-z) <= MASK_POLICY['measured_depth_tolerance_m'])
    return accepted, uv


def restored_visible(current, observed, witness_index, vehicle_guard):
    """框相交本身既不能批准，也不能否决；批准须有独立框外见证。"""
    return current & observed & (witness_index >= 0) & ~vehicle_guard


def sample_rgb(image, uv):
    return np.stack([map_coordinates(image[..., c], [uv[:, 1], uv[:, 0]], order=1,
        mode='constant', cval=0) for c in range(3)], 1)


def native_rgb(path):
    # 与 waymo_loader2 -> camera_utils 一致：先 float64/255，再 cv2 默认线性缩放，最后 float32。
    rgb = np.asarray(Image.open(path).convert('RGB'))/255.
    return cv2.resize(rgb, (480, 320)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'previous', 'output', 'engine'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args(); start = time.perf_counter()
    if args.output.exists(): raise FileExistsError('所有旧证据保留，只写新目录')
    sys.path.insert(0, str(args.engine))
    from utils.v41_vehicle import load_vehicle_domains, VEHICLE_POLICY, ACTORS
    meta = read(args.bundle); previous = read(args.previous/'manifest.json')
    if len(meta['frame_order']) != 49 or previous['frame_order'] != meta['frame_order']:
        raise ValueError('必须同49帧')
    with np.load(args.bundle.parent/meta['units'], allow_pickle=False) as z:
        units = {k:z[k] for k in z.files}
    surface = FixedSurface(meta, units)
    plane = np.asarray(read(meta['original_readiness'])['SGCM_geometry']['ground_fit']['plane_anchor'])
    source = Path(meta['original_engineering']).parent/'inputs/scene0008_089_137'
    args.output.mkdir(); (args.output/'frames').mkdir(); (args.output/'overlays').mkdir()
    domains, frames, rows = [], [], []
    for i, fid in enumerate(meta['frame_order']):
        fm = meta['frames'][fid]; K = np.asarray(fm['intrinsic']); w2c = np.asarray(fm['world_to_camera'])
        d = surface.camera(K, w2c); q = np.flatnonzero(d['supported'])
        labels, bodies, edges = load_vehicle_domains(fm)
        guard = bodies.any(0) | edges.any(0) | masks(fm)
        world, ground, boxes = cloud_and_boxes(source, fid, surface.inverse, plane)
        ok, depth, triangle, conflict = measured_patch_evidence(world, ground, K, w2c,
            d['world'][q], d['pixels'][q], units['normal_world'][d['unit_id'][q]])
        blocked = ray_box_blocked(d['world'][q], np.linalg.inv(w2c)[:3, 3], boxes)
        current = np.zeros(153600, bool); current[q] = ok & ~guard.ravel()[q]
        box = np.zeros(153600, bool); box[q] = blocked
        refs = np.zeros(153600, np.float32); refs[q] = np.where(current[q], depth, 0)
        provenance = np.full((153600, 3), -1, np.int32); provenance[q] = np.where(current[q, None], triangle, -1)
        datum = dict(current=current, box=box, source_clear=(current & ~box).reshape(320, 480),
            depth=refs.reshape(320, 480), guard=guard, labels=labels, bodies=bodies, edges=edges,
            rgb=native_rgb(fm['original_rgb']))
        domains.append(d); frames.append(datum)
        np.savez_compressed(args.output/'frames'/f'{fid}_evidence.npz',
            source_lidar_triangle_indices=provenance.reshape(320, 480, 3),
            measured_patch_depth_m=datum['depth'], single_view_valid=current.reshape(320, 480),
            box_intersection=box.reshape(320, 480), box_clear_witness=datum['source_clear'])
        rows.append(dict(frame=fid, old_reliable_centers=int((units['base_weight'][i, surface.observed] > 0).sum()),
            supported_candidate_pixels=int(d['supported'].sum()), observed_candidate_pixels=int(d['observed'].sum()),
            single_view_patch_pixels=int(current.sum()), box_recheck_candidates=int((current & box & d['observed']).sum())))
        print(json.dumps(dict(stage='current', **rows[-1])), flush=True)
    for i, fid in enumerate(meta['frame_order']):
        d, f = domains[i], frames[i]
        selected = np.flatnonzero(f['current'] & d['observed'])
        witness = np.full(153600, -1, np.int16); witness_uv = np.full((153600, 2), -1, np.float32)
        # 每个 target 用真实独立框外 source；新恢复像素不能互相循环担保。
        for j, other in enumerate(meta['frame_order']):
            if j == i: continue
            pending = selected[witness[selected] < 0]
            if not len(pending): break
            valid, uv = witness_at(d['world'][pending], meta['frames'][other], frames[j]['source_clear'], frames[j]['depth'])
            ids = pending[valid]; witness[ids] = j; witness_uv[ids] = uv[valid]
        visible = restored_visible(f['current'], d['observed'], witness, f['guard'].ravel()).reshape(320, 480)
        counts = np.where(visible, 2, 0).astype(np.uint8)
        visible, windows = finalize_masks(visible, counts, f['guard'])
        # 删车监督不使用本帧车辆 RGB。只在现有 R 单元下投影，寻找两个其它帧的独立观测。
        deletion_candidate = d['observed'] & (f['labels'].ravel() > 0)
        target_ids = np.flatnonzero(deletion_candidate)
        source_ids = np.full((153600, 2), -1, np.int16)
        source_uv = np.full((153600, 2, 2), -1, np.float32)
        colors = np.zeros((153600, 2, 3), np.float32)
        for j, other in enumerate(meta['frame_order']):
            if j == i: continue
            pending = target_ids[source_ids[target_ids, 1] < 0]
            if not len(pending): break
            valid, uv = witness_at(d['world'][pending], meta['frames'][other], frames[j]['source_clear'], frames[j]['depth'])
            ids = pending[valid]; slots = (source_ids[ids, 0] >= 0).astype(int)
            source_ids[ids, slots] = j; source_uv[ids, slots] = uv[valid]
            colors[ids, slots] = sample_rgb(frames[j]['rgb'], uv[valid])
        two_sources = source_ids[:, 1] >= 0
        consistent = np.abs(colors[:, 0]-colors[:, 1]).mean(1) <= REVISION_POLICY['deletion_source_color_tolerance']
        deletion_valid = deletion_candidate & two_sources & consistent
        target_rgb = np.where(deletion_valid[:, None], colors.mean(1), 0).reshape(320, 480, 3).transpose(2, 0, 1)
        deletion_domains = np.stack([(deletion_valid & (f['labels'].ravel() == k+1)).reshape(320, 480) for k in range(5)])
        with np.load(args.previous/'frames'/f'{fid}.npz', allow_pickle=False) as z: old = z['visible']
        box_restored = visible & f['box'].reshape(320, 480)
        added, removed = visible & ~old, old & ~visible
        # 紧凑保存 source provenance：每个候选都可追查拒绝原因，训练域完全固定。
        np.savez_compressed(args.output/'frames'/f'{fid}.npz', visible=visible, protect=~visible,
            ssim_valid=windows, candidate=d['observed'].reshape(320, 480), supported=d['supported'].reshape(320, 480),
            unit_id=d['unit_id'].reshape(320, 480), source_count_capped_two=counts,
            fixed_surface_depth_m=np.where(d['supported'], d['depth'], 0).reshape(320, 480).astype(np.float32),
            vehicle_body=f['bodies'], vehicle_edge=f['edges'], actor_labels=f['labels'],
            witness_frame_index=witness.reshape(320, 480), witness_uv=witness_uv.reshape(320, 480, 2),
            box_restored=box_restored, added=added, removed=removed,
            deletion_candidate=deletion_candidate.reshape(320, 480), deletion_valid=deletion_valid.reshape(320, 480),
            deletion_domains=deletion_domains, deletion_target_rgb=target_rgb,
            deletion_source_ids=source_ids.reshape(320, 480, 2), deletion_source_uv=source_uv.reshape(320, 480, 2, 2),
            deletion_two_sources=two_sources.reshape(320, 480), deletion_color_consistent=consistent.reshape(320, 480))
        rows[i].update(previous_visible_pixels=int(old.sum()), visible_pixels=int(visible.sum()),
            added_pixels=int(added.sum()), removed_pixels=int(removed.sum()), box_restored_pixels=int(box_restored.sum()),
            box_still_unconfirmed=int(rows[i]['box_recheck_candidates']-box_restored.sum()),
            ssim_windows=int(windows.sum()), protect_pixels=int((~visible).sum()),
            vehicle_road_target_overlap=int((visible & f['guard']).sum()),
            unknown_or_prior_supervised=int((visible.ravel() & ~d['observed']).sum()),
            deletion_candidates=int(deletion_candidate.sum()), deletion_two_sources=int((deletion_candidate & two_sources).sum()),
            deletion_verified_pixels=int(deletion_valid.sum()), deletion_by_actor=dict(zip(map(str, ACTORS), map(int, deletion_domains.sum((1, 2))))))
        overlay = f['rgb'].astype(float)*255
        # 只涂有语义的域；普通参考保护背景不再整片染红，青色是绿色道路域的子集。
        for mask, color, mix in ((d['observed'].reshape(320, 480) & ~visible & ~f['guard'], [255, 180, 0], .45),
                (visible, [0, 255, 70], .55), (windows, [0, 220, 255], .6),
                (f['bodies'].any(0), [255, 70, 180], .45), (f['edges'].any(0), [255, 80, 80], .65)):
            overlay[mask] = overlay[mask]*(1-mix)+np.asarray(color)*mix
        Image.fromarray(np.uint8(overlay.clip(0, 255))).save(args.output/'overlays'/f'{fid}.png')
        print(json.dumps(dict(stage='final', **rows[i])), flush=True)
    total_keys = [k for k,v in rows[0].items() if isinstance(v, int)]
    result = dict(version=VERSION, policy=dict(MASK_POLICY, vehicle=VEHICLE_POLICY, revision=REVISION_POLICY),
        source_bundle=str(args.bundle.resolve()), derived_from=str(args.previous.resolve()), frame_order=meta['frame_order'],
        frames=rows, totals={k:sum(row[k] for row in rows) for k in total_keys},
        input_roles=dict(RGB='same native float resize; deletion targets use ONLY two other visible views',
            LiDAR='same measured triangles and geometry thresholds', boxes='flag and independent witness gate',
            semantic_ground_truth=False, hidden_ground_truth=False),
        dense_domain_ready=all(r['visible_pixels'] > r['old_reliable_centers'] and r['ssim_windows'] > 0
            and r['vehicle_road_target_overlap'] == 0 and r['unknown_or_prior_supervised'] == 0 for r in rows),
        CPU_seconds=time.perf_counter()-start, GPU_used=False, optimizer_updates=0,
        quality_accepted=False, human_verdict=None)
    write(args.output/'manifest.json', result); print(json.dumps(result['totals']), flush=True)


if __name__ == '__main__': main()

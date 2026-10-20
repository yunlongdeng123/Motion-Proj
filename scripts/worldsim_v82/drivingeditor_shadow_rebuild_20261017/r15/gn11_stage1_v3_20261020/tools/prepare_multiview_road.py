"""同49帧重建世界道路查询与未融合来源表；只写新 output，纯 CPU。"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, laplace
from scipy.spatial import cKDTree


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def obb(label, pose, transform):
    value = label['box']
    cosine, sine = np.cos(value['heading']), np.sin(value['heading'])
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1.]])
    return dict(center=transform(np.array([[value['center_x'], value['center_y'], value['center_z']]]), pose)[0],
                half=np.array([value['length'], value['width'], value['height']])/2,
                axes=pose[:3, :3] @ rotation)


def prepare(engineering, output, engine):
    from utils.v3_road_contract import ROAD_VERSION, ROAD_POLICY
    from utils.v3_road_geometry import (transform, MeasuredRoadSurface, sparse_depth_maps,
                                       source_observation)
    if output.exists():
        raise FileExistsError('never overwrite evidence, trained output or old V2 bundle')
    source = engineering.parent/'inputs/scene0008_089_137'
    selected = engineering/'full5_pipeline/inputs/scene0008_089_137'
    all_vehicle = engineering/'all_vehicle_pipeline/inputs/scene0008_089_137'
    metadata = engineering/'full5_pipeline/metadata/scene0008_089_137'
    readiness, spec = read(metadata/'CPU_readiness.json'), read(metadata/'scene_spec.json')
    receipt = read(engineering/'matched5_input_receipt.json')
    actors = [5, 26, 18, 6, 37]  # 与真实 prepared label 的序号绑定，不按字母排序。
    if receipt['frames'] != 49 or len(receipt['track_ids']) != 5:
        raise RuntimeError('five-ID49 binding missing')
    tracks = dict(zip(actors, receipt['track_ids']))
    model = np.asarray(readiness['transform_matrix'])
    scale = float(readiness['actual_meter_to_model_scale'])
    if not np.allclose(np.linalg.svd(model[:3, :3])[1], scale):
        raise ValueError('nonuniform world/model metric scale')
    if spec['fixed_pose_anchor'] != readiness['fixed_pose_anchor']:
        raise ValueError('model/camera coordinate anchors differ')
    geometry = readiness['SGCM_geometry']
    anchor = np.loadtxt(geometry['anchor_pose'])
    anchor_inverse = np.linalg.inv(anchor)
    plane = np.asarray(geometry['ground_fit']['plane_anchor'])
    plane_world = anchor_inverse.T @ plane
    plane_model = np.linalg.inv(model).T @ plane_world
    plane_model /= np.linalg.norm(plane_model[:3])
    expected_plane = np.asarray(geometry['plane_model'], float)
    expected_plane /= np.linalg.norm(expected_plane[:3])
    if not np.allclose(plane_model, expected_plane, atol=1e-6):
        raise ValueError('measured road/model plane roundtrip failed')
    frames, measured, frame_meta = [], [], {}
    for raw in range(89, 138):
        fid = f'0008{raw:03d}'
        pose = np.loadtxt(source/'pose'/f'{fid}.txt')
        calibration = {line.split(':', 1)[0]: np.fromstring(line.split(':', 1)[1], sep=' ')
                       for line in (source/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in line}
        velo_to_camera = np.eye(4)
        velo_to_camera[:3] = calibration['Tr_velo_to_cam_0'].reshape(3, 4)
        w2c = velo_to_camera @ np.linalg.inv(pose)
        intrinsic = calibration['P0'].reshape(3, 4)[:, :3].copy()
        intrinsic[:2] *= .25
        rgb_path = source/'image/image_0'/f'{fid}.png'
        rgb = np.asarray(Image.open(rgb_path).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS))
        labels = np.asarray(Image.open(selected/'matched_training_labels'/f'{fid}.png'))
        if labels.shape != (320, 480) or not set(np.unique(labels)).issubset(set(range(6))):
            raise RuntimeError('label dimensions or classes changed')
        cloud = np.fromfile(source/'velodyne'/f'{fid}.bin', np.float32).reshape(-1, 6)[:, :3].astype(float)
        world = transform(cloud, pose)
        local = transform(world, anchor_inverse)
        boxes, chosen, occupied = [], [], np.zeros(len(world), bool)
        for label in read(source/'annotations'/f'{fid}.json')['laser_labels']:
            if min(label['box'][key] for key in ['length', 'width', 'height']) <= 0:
                continue
            box = obb(label, pose, transform)
            boxes.append(box)
            occupied |= (np.abs((world-box['center']) @ box['axes']) <= box['half']).all(1)
            if label['id'] in tracks.values():
                chosen.append(dict(center=transform(box['center'][None], model)[0].tolist(),
                                   half=(box['half']*scale).tolist(),
                                   axes=(model[:3, :3] @ box['axes']/scale).tolist()))
        ground = (np.abs(local @ plane[:3]+plane[3]) <= .15) & ~occupied
        ground &= (local[:, 0] >= 3) & (local[:, 0] <= 90) & (np.abs(local[:, 1]) <= 8)
        measured.append(local[ground])
        depth, valid, road = sparse_depth_maps(world, ground, intrinsic, w2c)
        other_mask = np.asarray(Image.open(all_vehicle/'dynamic_masks/dynamic_mask_0'/f'{fid}.png').resize((480, 320), Image.Resampling.NEAREST)) > 0
        frames.append(dict(fid=fid, raw=raw, K=intrinsic, w2c=w2c, rgb=rgb, labels=labels,
                           depth=depth, valid=valid, road=road, boxes=boxes,
                           occluder=(labels > 0) | other_mask,
                           sharpness=float(np.var(laplace(rgb.astype(float).mean(2)/255.)))))
        frame_meta[fid] = dict(world_to_camera=w2c.tolist(), intrinsic=intrinsic.tolist(),
                               selected_boxes_model=chosen, original_rgb=str(rgb_path),
                               label_path=str(selected/'matched_training_labels'/f'{fid}.png'))
    measured = np.concatenate(measured)
    _, indices = np.unique(np.floor(measured[:, :2]/.1).astype(int), axis=0, return_index=True)
    measured = measured[indices]
    surface = MeasuredRoadSurface(anchor, -plane[[0, 1, 3]]/plane[2], measured,
                                  ROAD_POLICY['surface_support_radius_m'])
    median_sharpness = max(np.median([frame['sharpness'] for frame in frames]), 1e-9)
    query_keys, query_records, rejected = [], [], {}
    cell = ROAD_POLICY['cell_m']
    for frame_index, frame in enumerate(frames):
        frame['quality'] = min(frame['sharpness']/median_sharpness, 1.)
        yy, xx = np.nonzero(frame['labels'] > 0)
        pixels = np.c_[xx, yy]
        local, valid, _ = surface.intersect(pixels, frame['K'], frame['w2c'])
        # 同一 LiDAR 道路走廊；远离可信路面范围的投射仅记录 UNKNOWN。
        valid &= (local[:, 0] >= 3) & (local[:, 0] <= 90) & (np.abs(local[:, 1]) <= 8)
        keys = np.floor(local[valid, :2]/cell).astype(np.int64)
        actor_values = np.asarray(actors)[frame['labels'][yy[valid], xx[valid]]-1]
        query_keys.append(keys)
        query_records.append((np.full(len(keys), frame_index), actor_values, pixels[valid], keys))
        rejected[frame['fid']] = int((~valid).sum())
    unique_query = np.unique(np.concatenate(query_keys), axis=0)
    halo = int(np.ceil(ROAD_POLICY['context_halo_m']/cell))
    lower, upper = unique_query.min(0)-halo, unique_query.max(0)+halo
    grid = np.zeros(tuple(upper-lower+1), bool)
    grid[tuple((unique_query-lower).T)] = True
    expanded = binary_dilation(grid, iterations=halo)
    keys = np.argwhere(expanded)+lower
    xy = (keys+.5)*cell
    height, support = surface.height(xy)
    local = np.c_[xy, height]
    world = transform(local, anchor)
    normals = surface.normals(xy)
    is_query = grid[tuple((keys-lower).T)]
    lookup = {tuple(key): index for index, key in enumerate(keys)}
    edges = [(index, lookup[tuple(key+offset)]) for index, key in enumerate(keys)
             for offset in (np.array([1, 0]), np.array([0, 1])) if tuple(key+offset) in lookup]
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    distances, _ = cKDTree(measured[:, :2]).query(xy)
    source_rgb, source_uv, source_depth, weights, gate_stack = [], [], [], [], []
    for frame in frames:
        color, uv, depth, weight, gates = source_observation(world, normals, support, frame, ROAD_POLICY)
        source_rgb.append(color); source_uv.append(uv); source_depth.append(depth*scale)
        weights.append(weight); gate_stack.append(gates)
    weights = np.asarray(weights)
    count = (weights > 0).sum(0)
    # 仅无任何硬通过来源时才能成为先验；单视角候选保留 UNKNOWN。
    provenance = np.where(support & (count >= ROAD_POLICY['minimum_views']), 1,
                          np.where(support & (count == 0), 2, 0)).astype(np.uint8)
    frame_indices, actor_indices, pixels, unit_ids = [], [], [], []
    for frame_index, actor, pixel, query in query_records:
        frame_indices.append(frame_index); actor_indices.append(actor); pixels.append(pixel)
        unit_ids.append(np.array([lookup[tuple(key)] for key in query], np.int64))
    output.mkdir(parents=True)
    arrays = dict(unit_ids=np.arange(len(keys)), keys=keys, xy_local=xy, world=world,
                  model_xyz=transform(world, model), normal_world=normals,
                  normal_model=normals @ model[:3, :3].T/scale, surface_supported=support,
                  surface_distance_m=distances, is_query=is_query, edges=edges, provenance=provenance,
                  source_count=count, source_rgb=np.asarray(source_rgb, np.float32),
                  source_uv=np.asarray(source_uv, np.float32), source_depth=np.asarray(source_depth, np.float32),
                  base_weight=weights.astype(np.float32), query_frame=np.concatenate(frame_indices),
                  query_actor=np.concatenate(actor_indices), query_pixel=np.concatenate(pixels),
                  query_unit_id=np.concatenate(unit_ids))
    for name in gate_stack[0]:
        arrays['gate_'+name] = np.asarray([entry[name] for entry in gate_stack])
    np.savez_compressed(output/'units.npz', **arrays)
    summary = {}
    for actor in actors:
        ids = np.unique(arrays['query_unit_id'][(arrays['query_frame'] == 0) & (arrays['query_actor'] == actor)])
        summary[str(actor)] = dict(raw89_world_units=len(ids), observed=int((provenance[ids] == 1).sum()),
                                   prior=int((provenance[ids] == 2).sum()), unknown=int((provenance[ids] == 0).sum()),
                                   single_view_unconfirmed=int((count[ids] == 1).sum()))
    bundle = dict(status='CPU_MULTIVIEW_ROAD_INPUT_VERIFIED', version=ROAD_VERSION, policy=ROAD_POLICY,
                  actors=actors, frames=frame_meta, frame_order=[frame['fid'] for frame in frames],
                  world_to_model=model.tolist(), meter_to_model=scale, ground_plane_model=plane_model.tolist(),
                  coordinate_anchor=anchor.tolist(), units='units.npz', unit_count=len(keys),
                  supported_units=int(support.sum()), query_units=int(is_query.sum()),
                  observed_units=int((provenance == 1).sum()), prior_units=int((provenance == 2).sum()),
                  unknown_units=int((provenance == 0).sum()), source_observations=int((weights > 0).sum()),
                  raw89_actor_summary=summary, query_geometry_rejected=rejected,
                  RGB_fused=False, late_frame_preference=False, manual_query_or_donor_polygon=False,
                  prior_is_ground_truth=False, source_bundle_002_loaded=False,
                  evidence_roles=dict(camera='dataset calibration', boxes='all annotated 3D occluders',
                                      depth='nearest measured LiDAR within1.5px; not dense GT',
                                      road='plane-close measured non-object returns; not semantic GT',
                                      masks='original five-ID and all-vehicle prepared masks'),
                  config=dict(training=dict(volume_exclusion_weight=.1, road_clearance_m=.2, volume_edge_m=.1), road={}),
                  original_engineering=str(engineering), original_readiness=str(metadata/'CPU_readiness.json'),
                  limitations=['No confirmed exposure means prior, not recovered hidden texture.',
                               'Unlabeled occluders and sparse depth gaps fail closed where evidence is absent.',
                               'No learned exposure correction; original sRGB and robust residual retained.',
                               'Thin 3D Gaussians are not a replacement 2DGS rasterizer.'],
                  GPU_used=False, training_updates=0, quality_accepted=False, human_verdict=None)
    (output/'bundle.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: bundle[key] for key in ['version', 'unit_count', 'supported_units', 'source_observations', 'raw89_actor_summary']}, indent=2))
    return bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engineering', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.engine))
    prepare(args.engineering, args.output, args.engine)

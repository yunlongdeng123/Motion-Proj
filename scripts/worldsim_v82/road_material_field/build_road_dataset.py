"""CPU构建固定测距道路atlas、可追溯RGB观测及按世界区域隔离的伪遮挡样本。

输入路径全部由配置提供。目标是参考曝光下的道路外观，不是真实PBR/albedo。
"""
from __future__ import annotations
import os
if __name__ == '__main__':
    for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_name, '1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, minimum_filter, gaussian_filter
from scipy.spatial import cKDTree
if __package__:
    from .world_feature_field import (transform_xyz, camera_from_observation,
        project_world_to_camera, camera_ground_rays, bilinear_sample)
else:
    from world_feature_field import (transform_xyz, camera_from_observation,
        project_world_to_camera, camera_ground_rays, bilinear_sample)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def save_image(path, values):
    values = np.asarray(values)
    if values.dtype == bool:
        values = values.astype(np.uint8)*255
    elif values.dtype != np.uint8:
        values = np.uint8(np.clip(values, 0, 1)*255+.5)
    Image.fromarray(values).save(path)


def rgb_linear(rgb):
    return np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)


def make_guard(mask_root, semantics, annotation_root, raw, wh, observation, road_id):
    w, h = wh
    semantic = np.asarray(Image.open(semantics/f'{raw:03d}.png').resize((w, h), Image.Resampling.NEAREST))
    labels = np.asarray(Image.open(mask_root/f'{raw:03d}/labels.png').resize((w, h), Image.Resampling.NEAREST))
    foreground = np.asarray(Image.open(mask_root/f'{raw:03d}/dynamic_union.png').resize((w, h), Image.Resampling.NEAREST)) > 0
    guard = binary_dilation(foreground | (labels > 0), iterations=2)
    annotation = read(annotation_root/f'{raw:03d}.json') if annotation_root else None
    if annotation is not None:
        if annotation['timestamp_micros'] != observation['timestamp_micros']:
            raise ValueError('Raw annotation timestamp mismatch')
        for box in annotation['front_camera_rectangles']:
            xyxy = np.asarray(box['xyxy'])*[w/observation['camera_size'][0], h/observation['camera_size'][1],
                                              w/observation['camera_size'][0], h/observation['camera_size'][1]]
            x0, y0 = np.maximum(np.floor(xyxy[:2]).astype(int)-2, 0)
            x1, y1 = np.minimum(np.ceil(xyxy[2:]).astype(int)+3, [w, h])
            guard[y0:y1, x0:x1] = True
    return (semantic == road_id) & ~guard, guard, annotation


def inside_boxes(points_world, annotation):
    blocked = np.zeros(len(points_world), bool)
    if annotation is None:
        return blocked
    for box in annotation['boxes']:
        center = np.asarray(box['center']); axes = np.asarray(box['axes']); half = np.asarray(box['half'])+.05
        # The world-axis bounding sphere first avoids multiplying the complete point cloud per box.
        radius = np.linalg.norm(half)
        nearby = np.flatnonzero(~blocked & (abs(points_world-center) <= radius).all(1))
        if len(nearby):
            blocked[nearby] |= (abs((points_world[nearby]-center) @ axes) <= half).all(1)
    return blocked


def surface_height(xy, tree, measured_xy, measured_z, plane, radius=.65):
    distance, indices = tree.query(xy, k=8, workers=4)
    weight = (distance < radius)/np.maximum(distance, .06)**2
    base = np.c_[xy, np.ones(len(xy))] @ plane
    residual = measured_z-np.c_[measured_xy, np.ones(len(measured_xy))] @ plane
    height = base+(weight*residual[indices]).sum(1)/np.maximum(weight.sum(1), 1e-12)
    return height, distance[:, 0]


def visibility_for_points(xyz, cache, wh, normal_world, max_distance):
    """道路/前景保护及测距深度检查；允许相邻像素自身的地面透视深度梯度。"""
    camera=cache['camera'];w,h=wh
    uv,depth,inside=project_world_to_camera(xyz,**{k:camera[k] for k in ('camera_K_native','world_to_camera','image_hw','distortion_Brown')})
    safe_uv=np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)
    safe_uv=np.clip(safe_uv,-1e6,1e6)
    pix=np.floor(safe_uv*[w/camera['image_hw'][1],h/camera['image_hw'][0]]).astype(int)
    p=np.clip(pix,[0,0],[w-1,h-1]);eligible=inside&cache['good'][p[:,1],p[:,0]]
    origin=np.linalg.inv(np.asarray(camera['world_to_camera']))[:3,3]
    view=origin-xyz;distance=np.linalg.norm(view,axis=1)
    cosine=np.maximum((view@normal_world)/np.maximum(distance,1e-6),0.)
    nearest=cache.get('nearest_3x3')
    if nearest is None:
        nearest=minimum_filter(cache['nearest'],size=3,mode='constant',cval=np.inf)
        cache['nearest_3x3']=nearest
    # 地面z梯度约为z²/(f*h_camera)，不是固定0.5m。保守宽容仅用于相邻像素，
    # 真实前景仍由独立accepted instance及原始二维注释guard排除。
    fy=abs(camera['camera_K_native'][1][1])*h/camera['image_hw'][0]
    tolerance=.5+1.5*np.minimum(depth,100.)**2/max(fy*2.,1.)
    if cache.get('use_lidar_visibility',True):
        eligible &= depth <= nearest[p[:,1],p[:,0]]+tolerance
    eligible &= (distance<max_distance)&(cosine>.025)
    weight=(cosine/(distance**2+4.)).astype(np.float32)
    return uv,depth,eligible,weight


def coherent_tile(ids, world, caches, wh, normal_world, cfg):
    """一个图块只用一台源相机曝光，避免逐texel切帧产生拼缝。"""
    best=None;xyz=world[ids.ravel()]
    size=ids.shape[0];lo=size//4;hi=size-lo
    for raw,c in caches.items():
        uv,depth,eligible,weight=visibility_for_points(xyz,c,wh,normal_world,cfg.get('max_distance_m',60.))
        eligible=eligible.reshape(ids.shape)
        fraction=float(eligible.mean());inner_fraction=float(eligible[lo:hi,lo:hi].mean())
        if fraction<cfg.get('min_tile_valid',.95) or inner_fraction<cfg.get('min_inner_valid',.999):
            continue
        # 先完整性，再角度/距离质量；整个tile来自同一raw，不做颜色逐格切换。
        score=fraction+min(float(np.mean(weight[eligible.ravel()])),.01)
        if best is None or score>best['score']:
            rgb=bilinear_sample(c['image'],uv,c['camera']['image_hw']).reshape(size,size,3).astype(np.float32)
            best=dict(score=score,raw=raw,rgb=rgb,valid=eligible,uv_native=uv.reshape(size,size,2),weight=weight.reshape(size,size),
                valid_fraction=fraction,inner_valid_fraction=inner_fraction)
    return best


def build(config_path, output):
    started = time.monotonic(); cfg = read(config_path)
    if cfg.get('schema') != 'road_material_build.v1':
        raise ValueError('Explicit road material build config required')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    for name in ('atlas', 'observations', 'sources', 'samples'):
        (out/name).mkdir()
    bank, masks, semantics = (Path(cfg[k]) for k in ('bank', 'masks', 'semantics'))
    annotation_root = Path(cfg['annotations']) if cfg.get('annotations') else None
    receipt = read(bank/'receipt.json'); frames = sorted(receipt['frames'], key=lambda r: r['raw'])
    raws = [r['raw'] for r in frames]; source_raws = raws[::cfg.get('source_stride', 4)]
    if raws[-1] not in source_raws:
        source_raws.append(raws[-1])
    frame_by_raw = {r['raw']: r for r in frames}
    observations = {raw: read(bank/'observations'/f'{raw:03d}.json') for raw in raws}
    with np.load(cfg['geometry_seed']) as seed:
        anchor = seed['coordinate_anchor'].copy(); plane = seed['plane'].copy()
        model_from_world = seed['model_from_world'].copy()
    local_from_world = np.linalg.inv(anchor)
    wh = tuple(cfg.get('source_wh', [480, 320])); w, h = wh
    caches = {}; measured = []; requested = []; point_stats = []
    for raw in source_raws:
        obs = observations[raw]; camera = camera_from_observation(obs)
        good, guard, annotation = make_guard(masks, semantics, annotation_root, raw, wh, obs, cfg.get('road_semantic_id', 0))
        cloud = np.fromfile(bank/obs['lidar_path'], dtype='<f4').reshape(-1, 6)[:, :3]
        world = transform_xyz(cloud, np.asarray(obs['ego_world']).reshape(4, 4))
        local = transform_xyz(world, local_from_world)
        uv, depth, inside = project_world_to_camera(world, **{k:camera[k] for k in ('camera_K_native','world_to_camera','image_hw','distortion_Brown')})
        pix = np.floor(np.clip(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1),-1e6,1e6)*[w/camera['image_hw'][1], h/camera['image_hw'][0]]).astype(int)
        ids = np.flatnonzero(inside); p = np.clip(pix[ids], [0, 0], [w-1, h-1])
        nearest = np.full((h, w), np.inf, np.float32)
        np.minimum.at(nearest, (p[:, 1], p[:, 0]), depth[ids])
        keep = good[p[:, 1], p[:, 0]] & (depth[ids] <= nearest[p[:, 1], p[:, 0]]+.35)
        keep &= abs(local[ids, 2]-np.c_[local[ids, :2], np.ones(len(ids))]@plane) < .8
        ids = ids[keep]
        ids = ids[~inside_boxes(world[ids], annotation)]
        measured.append(local[ids])
        hits, ray_depth = camera_ground_rays(camera, wh, local_from_world, plane)
        origin_local = transform_xyz(np.linalg.inv(np.asarray(camera['world_to_camera']))[:3, 3][None], local_from_world)[0]
        finite = good & np.isfinite(hits).all(2) & (ray_depth > .1)
        finite &= np.linalg.norm(hits-origin_local, axis=2) < cfg.get('max_distance_m', 60.)
        requested.append(hits[finite, :2])
        image = np.asarray(Image.open(bank/obs['image_path']).convert('RGB').resize(wh, Image.Resampling.LANCZOS), np.float32)/255
        caches[raw] = dict(camera=camera, good=good, guard=guard, nearest=nearest, image=image,
                           use_lidar_visibility=cfg.get('source_visibility')!='road_foreground_guard_planar')
        point_stats.append(dict(raw=raw, road_lidar_points=len(ids), road_pixels=int(good.sum())))
        if len(caches)%10 == 0:
            print('MEASURED', len(caches), len(source_raws), time.monotonic()-started, flush=True)
    measured = np.concatenate(measured); requested = np.concatenate(requested)
    # 同世界0.1m格去重，防止停止车辆片段的重复帧主导几何拟合。
    _, inverse = np.unique(np.floor(measured[:, :2]/.1).astype(np.int64), axis=0, return_inverse=True)
    den = np.bincount(inverse); dedup = np.column_stack([np.bincount(inverse, weights=measured[:, i])/den for i in range(3)])
    design = np.c_[dedup[:, :2], np.ones(len(dedup))]
    for _ in range(6):
        residual = design@plane-dedup[:, 2]; weights = np.minimum(1., .08/np.maximum(abs(residual), 1e-8))
        plane = np.linalg.lstsq(design*weights[:, None]**.5, dedup[:, 2]*weights**.5, rcond=None)[0]
    clean = abs(design@plane-dedup[:, 2]) < .22
    dedup = dedup[clean]; tree = cKDTree(dedup[:, :2])
    cell = cfg.get('cell_m', .04); tile_px = cfg.get('tile_px', 128)
    # 全部可见道路射线确定范围，明确不使用某一被遮挡车辆的ROI。
    low = np.floor((requested.min(0)-.2)/cell).astype(int)
    high = np.ceil((requested.max(0)+.2)/cell).astype(int)
    shape_xy = high-low+1; count = int(np.prod(shape_xy))
    if count > cfg.get('max_cells', 6000000):
        raise ValueError(f'Broad chart requires {count} cells; explicit max_cells exceeded')
    yy, xx = np.mgrid[:shape_xy[1], :shape_xy[0]]
    xy = np.c_[xx.ravel()+low[0]+.5, yy.ravel()+low[1]+.5]*cell
    xyz_local = np.empty((count, 3), np.float32); measured_distance = np.empty(count, np.float32)
    for begin in range(0, count, 250000):
        end = min(begin+250000, count)
        if cfg.get('geometry_mode')=='plane':
            height=np.c_[xy[begin:end],np.ones(end-begin)]@plane
            distance=tree.query(xy[begin:end],workers=4)[0]
        else:
            height, distance = surface_height(xy[begin:end], tree, dedup[:, :2], dedup[:, 2], plane)
        xyz_local[begin:end] = np.c_[xy[begin:end], height]; measured_distance[begin:end] = distance
    world = transform_xyz(xyz_local, anchor).astype(np.float32)
    geometry_valid = measured_distance <= cfg.get('measured_support_radius_m', .6)
    measured_geometry_valid=geometry_valid.copy()
    if cfg.get('source_visibility')=='road_foreground_guard_planar':
        # 可见道路材料学习使用固定测距拟合面近似；不要求每个材质texel有独立回波。
        geometry_valid=np.ones(count,bool)
    geometric_ids = np.flatnonzero(geometry_valid)
    normal_local = np.array([-plane[0], -plane[1], 1.]); normal_local /= np.linalg.norm(normal_local)
    normal_world = anchor[:3, :3]@normal_local
    best_weight = np.zeros(count, np.float32); best_raw = np.full(count, -1, np.int16)
    best_uv = np.zeros((count, 2), np.float32); target_rgb = np.full((count, 3), .5, np.float32)
    count_observations = np.zeros(count, np.uint16); obs_records = []
    print('BROAD_ATLAS', shape_xy.tolist(), count, 'supported', len(geometric_ids), flush=True)
    for raw in source_raws:
        c = caches[raw]; camera = c['camera']; xyz = world[geometric_ids]
        uv,depth,eligible,all_weight=visibility_for_points(xyz,c,wh,normal_world,cfg.get('max_distance_m',60.))
        selected = np.flatnonzero(eligible); ids = geometric_ids[selected]
        weight = all_weight[selected]
        sampled = bilinear_sample(c['image'], uv[selected], camera['image_hw']).astype(np.float32)
        count_observations[ids] += 1
        better = weight > best_weight[ids]; bid = ids[better]
        best_weight[bid] = weight[better]; target_rgb[bid] = sampled[better]
        best_raw[bid] = raw; best_uv[bid] = uv[selected[better]]
        observation_path = f'observations/{raw:03d}.npz'
        np.savez_compressed(out/observation_path, point_ids=ids.astype(np.int32), uv_native=uv[selected].astype(np.float32),
                            weight=weight, depth=depth[selected].astype(np.float32))
        obs_records.append(dict(raw=raw, observations=observation_path, observations_count=len(ids),
            image_path=str(bank/observations[raw]['image_path']), lidar_path=str(bank/observations[raw]['lidar_path']),
            observation_path=str(bank/'observations'/f'{raw:03d}.json'), **camera))
        print('SOURCE', raw, len(ids), time.monotonic()-started, flush=True)
    valid = (best_weight > 0) & geometry_valid
    valid_grid = valid.reshape(shape_xy[::-1]); rgb_grid = target_rgb.reshape(*shape_xy[::-1], 3)
    # 不重叠tile保证同世界位置去重；按纵向世界区域留出，非逐帧随机划分。
    tiles = [];coherent={}
    for ty in range(0, shape_xy[1]-tile_px+1, tile_px):
        for tx in range(0, shape_xy[0]-tile_px+1, tile_px):
            fraction = valid_grid[ty:ty+tile_px, tx:tx+tile_px].mean()
            if fraction >= cfg.get('min_tile_valid', .6):
                if cfg.get('target_mode')=='single_source_tile':
                    yy1,xx1=np.mgrid[ty:ty+tile_px,tx:tx+tile_px];tile_ids=yy1*shape_xy[0]+xx1
                    warped=coherent_tile(tile_ids,world,caches,wh,normal_world,cfg)
                    if warped is None:
                        continue
                    coherent[(tx,ty)]=warped;fraction=warped['valid_fraction']
                tile_rgb = rgb_grid[ty:ty+tile_px, tx:tx+tile_px]
                if (tx,ty) in coherent:
                    tile_rgb=coherent[(tx,ty)]['rgb']
                marked = ((tile_rgb.max(2) > .65) & valid_grid[ty:ty+tile_px, tx:tx+tile_px]).mean()
                tiles.append(dict(tx=tx, ty=ty, valid_fraction=float(fraction), marking_fraction=float(marked)))
    if len(tiles) < cfg.get('train_count', 32)+cfg.get('eval_count', 8):
        write(out/'insufficient_tiles.json', dict(tiles=tiles, shape_xy=shape_xy.tolist(), valid_cells=int(valid.sum())))
        if not cfg.get('allow_smaller',False) or len(tiles)<4:
            raise ValueError(f'Only {len(tiles)} disjoint sufficiently valid tiles available')
    tiles.sort(key=lambda d:(d['tx'], d['ty']))
    train_count, eval_count = cfg.get('train_count', 32), cfg.get('eval_count', 8)
    if cfg.get('allow_smaller',False) and len(tiles)<train_count+eval_count:
        eval_count=max(1,min(eval_count,len(tiles)//4));train_count=min(train_count,len(tiles)-eval_count)
    # 尾端至少eval_count个tile构成完整纵向留出区域；训练选自其前方。
    unique_tx = sorted({t['tx'] for t in tiles}); split_tx = unique_tx[-1]
    for tx in reversed(unique_tx):
        if sum(t['tx'] >= tx for t in tiles) >= eval_count and sum(t['tx'] < tx for t in tiles) >= train_count:
            split_tx = tx; break
    train_pool = [t for t in tiles if t['tx'] < split_tx]
    eval_pool = [t for t in tiles if t['tx'] >= split_tx]
    rng = np.random.default_rng(cfg.get('seed', 20261112))
    def choose(pool, count):
        # 标线样本优先保留，剩余由固定seed无放回抽取。
        marked = sorted([t for t in pool if t['marking_fraction'] > .01], key=lambda t:-t['marking_fraction'])[:max(1,count//4)]
        rest = [t for t in pool if t not in marked]
        order = rng.permutation(len(rest))
        return marked+[rest[i] for i in order[:count-len(marked)]]
    chosen = [(t,'train') for t in choose(train_pool,train_count)]+[(t,'eval') for t in choose(eval_pool,eval_count)]
    blocked_world = np.zeros(count, bool); eval_world = np.zeros(count, bool); samples = []
    for number, (tile, split) in enumerate(chosen):
        tx, ty = tile['tx'], tile['ty']; sample_id = f'{cfg["scene_id"]}_{split}_{number:03d}'
        sample_dir = out/'samples'/sample_id; sample_dir.mkdir()
        sy, sx = np.mgrid[ty:ty+tile_px, tx:tx+tile_px]; ids = sy*shape_xy[0]+sx
        local_valid = valid[ids]; target = target_rgb[ids].copy()
        tile_source=coherent.get((tx,ty))
        if tile_source is not None:
            local_valid=tile_source['valid'];target=tile_source['rgb'].copy()
            valid[ids]=local_valid;target_rgb[ids]=target
            best_raw[ids]=np.where(local_valid,tile_source['raw'],-1)
            best_uv[ids]=tile_source['uv_native']
        target[~local_valid] = .5
        yy0, xx0 = np.mgrid[:tile_px, :tile_px]
        # 大矩形/椭圆是合成伪遮挡，仅在有真实可见来源的像素中计分。
        cx, cy = tile_px*(.48+rng.uniform(-.06,.06)), tile_px*(.5+rng.uniform(-.06,.06))
        rx, ry = tile_px*rng.uniform(.2,.3), tile_px*rng.uniform(.22,.34)
        if number % 2:
            hole = ((xx0-cx)/rx)**2+((yy0-cy)/ry)**2 < 1
        else:
            hole = (abs(xx0-cx)<rx) & (abs(yy0-cy)<ry)
        hole &= local_valid
        if hole.sum() < 500:
            raise ValueError(f'Insufficient pseudo-hole target support for {sample_id}')
        blocked_world[ids[hole]] = True
        if split == 'eval':
            # 全eval图块从候选/feature源中剔除，ring上下文仍来自本sample.input。
            eval_world[ids.ravel()] = True
        image_input = target.copy(); image_input[hole] = .5
        save_image(sample_dir/'target.png', target); save_image(sample_dir/'input.png', image_input)
        save_image(sample_dir/'hole.png', hole); save_image(sample_dir/'valid.png', local_valid)
        np.save(sample_dir/'world_xyz.npy', world[ids]); np.save(sample_dir/'point_ids.npy', ids.astype(np.int32))
        transform = np.eye(4); transform[:3, 0] = anchor[:3, 0]*cell+anchor[:3, 2]*plane[0]*cell
        transform[:3, 1] = anchor[:3, 1]*cell+anchor[:3, 2]*plane[1]*cell
        local_origin = np.array([(tx+low[0])*cell,(ty+low[1])*cell,0.])
        local_origin[2] = plane[:2]@local_origin[:2]+plane[2]
        transform[:3, 3] = transform_xyz(local_origin[None],anchor)[0]
        np.savez_compressed(sample_dir/'tile.npz',target=target,input=image_input,hole=hole,valid=local_valid,
                            atlas_indices=ids.astype(np.int32),world_xyz=world[ids],world_from_uv=transform)
        prefix = f'samples/{sample_id}/'
        samples.append(dict(id=sample_id, split=split, scene_id=cfg['scene_id'],
            target=prefix+'target.png', input=prefix+'input.png', reference=prefix+'reference.png',
            hole_mask=prefix+'hole.png',valid_mask=prefix+'valid.png', hole=prefix+'hole.png',valid=prefix+'valid.png',
            world_xyz=prefix+'world_xyz.npy', point_ids=prefix+'point_ids.npy',
            tile_npz=prefix+'tile.npz',
            world_from_uv=transform.tolist(),
            world_bounds=[world[ids].min((0,1)).tolist(),world[ids].max((0,1)).tolist()],
            atlas_pixel_bounds=[tx,ty,tx+tile_px,ty+tile_px], atlas_local_xy_bounds=[local_origin[:2].tolist(),(local_origin[:2]+tile_px*cell).tolist()],
            valid_pixels=int(local_valid.sum()), hole_pixels=int(hole.sum()), **tile,
            provenance=dict(target='same_world_visible_observation_reference_exposure', input='target_with_synthetic_hole',
                geometry='fixed_measured_plane_plus_local_lidar_residuals', evaluation='visible_road_pseudo_occlusion_only',
                generated=False, true_hidden_ground_truth=False)))
        if tile_source is not None:
            samples[-1]['target_source_raw']=tile_source['raw']
            samples[-1]['target_source_image']=str(bank/observations[tile_source['raw']]['image_path'])
            samples[-1]['provenance']['geometry']='fixed_measured_plane_no_local_residual'
            samples[-1]['provenance']['target']='single_source_camera_whole_tile_warp_reference_exposure'
            samples[-1]['source_inner_valid_fraction']=tile_source['inner_valid_fraction']
            np.save(sample_dir/'source_uv_native.npy',tile_source['uv_native'])
            np.save(sample_dir/'source_weight.npy',tile_source['weight'])
            samples[-1]['source_uv_native']=prefix+'source_uv_native.npy'
            samples[-1]['source_weight']=prefix+'source_weight.npy'
    # 使用训练区域中的另一个不重叠tile作为异地材质参考，不将同世界重拍混称为先验。
    train_samples = [s for s in samples if s['split']=='train']
    for sample in samples:
        source = max(train_samples, key=lambda s:abs(s['tx']-sample['tx'])+abs(s['ty']-sample['ty']))
        ref_key='target' if cfg.get('clean_reference',False) else 'input'
        source_image = np.asarray(Image.open(out/source[ref_key]))
        save_image(out/sample['reference'],source_image)
        reference_valid_path=f'samples/{sample["id"]}/reference_valid.png'
        save_image(out/reference_valid_path,np.asarray(Image.open(out/source['valid_mask'])))
        sample['reference_valid_mask']=reference_valid_path
        sample['reference_sample_id'] = source['id']; sample['reference_world_bounds'] = source['world_bounds']
        sample['provenance']['reference'] = ('different_world_clean_single_source_visible_reference' if cfg.get('clean_reference',False)
                                             else 'different_world_material_reference_with_its_own_query_removed')
    # 统一屏蔽所有查询hole及完整eval世界区；保存合法观测供C/D使用。
    feature_excluded = blocked_world.copy()
    blocked_world |= eval_world
    np.save(out/'atlas/world_xyz.npy',world); np.save(out/'atlas/local_xyz.npy',xyz_local)
    np.save(out/'atlas/geometry_valid.npy',geometry_valid); np.save(out/'atlas/eval_world_mask.npy',eval_world)
    np.save(out/'atlas/measured_geometry_support.npy',measured_geometry_valid)
    np.save(out/'atlas/exclude_world_mask.npy',blocked_world); np.save(out/'atlas/observed_valid.npy',valid & ~blocked_world)
    np.save(out/'atlas/feature_exclude_world_mask.npy',feature_excluded)
    np.save(out/'atlas/supervision_rgb.npy',target_rgb); np.save(out/'atlas/supervision_valid.npy',valid)
    input_atlas = target_rgb.copy(); input_atlas[blocked_world | ~valid] = .5
    np.save(out/'atlas/input_rgb.npy',input_atlas); np.save(out/'atlas/best_raw.npy',best_raw)
    np.save(out/'atlas/best_uv_native.npy',best_uv); np.save(out/'atlas/observation_count.npy',count_observations)
    np.save(out/'atlas/measured_distance_m.npy',measured_distance)
    sparse_masked_count = 0
    for record in obs_records:
        path = out/record['observations']
        with np.load(path) as d:
            values = {k:d[k].copy() for k in d.files}
        allowed = ~feature_excluded[values['point_ids']]
        sparse_masked_count += int((~allowed).sum())
        np.savez_compressed(path,**{k:v[allowed] for k,v in values.items()})
        record['observations_count'] = int(allowed.sum())
        record['observation_role'] = 'feature_legal_world_cells_after_all_query_holes_exclusion; candidate_exclude_world_mask_is_additional'
    # 源RGB必须在encoder之前移除测试区；mask由固定几何反投影，覆盖全部199帧。
    source_by_raw = {r['raw']:r for r in obs_records}; mask_audit = []
    for raw in raws:
        obs = observations[raw]; camera = camera_from_observation(obs)
        hits, depth = camera_ground_rays(camera,wh,local_from_world,plane)
        key = np.floor(np.nan_to_num(hits[:,:,:2],nan=0,posinf=0,neginf=0)/cell).astype(np.int64)-low
        inside = np.isfinite(hits).all(2) & (depth>.1) & (key>=0).all(2) & (key<shape_xy).all(2)
        safe = np.clip(key,[0,0],shape_xy-1); ids=safe[:,:,1]*shape_xy[0]+safe[:,:,0]
        query_mask = inside & feature_excluded[ids]
        # 再正投影实际非平面query cell，避免仅平面反投影遗漏局部高程偏移。
        query_ids=np.flatnonzero(feature_excluded)
        projected, _, projected_inside=project_world_to_camera(world[query_ids],**{k:camera[k] for k in ('camera_K_native','world_to_camera','image_hw','distortion_Brown')})
        pp=np.floor(projected[projected_inside]*[w/camera['image_hw'][1],h/camera['image_hw'][0]]).astype(int)
        pp=np.clip(pp,[0,0],[w-1,h-1]);query_mask[pp[:,1],pp[:,0]]=True
        query_mask = binary_dilation(query_mask,iterations=4)
        mask_audit.append(dict(raw=raw,query_pixels=int(query_mask.sum())))
        if raw in source_by_raw:
            # 4px低分辨率扩边覆盖局部测距残差及下采样滤波支持；feature encoder只读此图。
            image = Image.open(bank/obs['image_path']).convert('RGB'); native_wh=image.size
            query_mask |= ~caches[raw]['good']
            native_mask = np.asarray(Image.fromarray(query_mask.astype(np.uint8)*255).resize(native_wh,Image.Resampling.NEAREST))>0
            array = np.asarray(image).copy(); array[native_mask]=128
            feature_input=f'sources/{raw:03d}_feature_input.png'; allowed_path=f'sources/{raw:03d}_feature_allowed.png'
            save_image(out/feature_input,array);save_image(out/allowed_path,~native_mask)
            source_by_raw[raw].update(feature_input=feature_input,feature_allowed_mask=allowed_path,
                mask_policy='all_sample_query_holes_and_nonroad_or_foreground_removed_before_encoder; 4 source-pixel dilation; eval_context_retained')
    atlas_transform=np.eye(4);atlas_transform[:3,0]=anchor[:3,0]*cell+anchor[:3,2]*plane[0]*cell
    atlas_transform[:3,1]=anchor[:3,1]*cell+anchor[:3,2]*plane[1]*cell
    base=np.array([low[0]*cell,low[1]*cell,plane[:2]@(low*cell)+plane[2]])
    atlas_transform[:3,3]=transform_xyz(base[None],anchor)[0]
    manifest=dict(schema='road_material_dataset.v1',seed=cfg.get('seed',20261112),scene_id=cfg['scene_id'],
        coordinate=dict(units='m',world='Waymo world coordinates',atlas_uv_axes='u=local+x, v=local+y',
            pixel_centers='world_from_uv @ [u+0.5,v+0.5,0,1]; exact nonplanar height in world_xyz.npy',
            local_surface=('fixed measured tangent plane; no local residuals; geometry never trained' if cfg.get('geometry_mode')=='plane'
                           else 'measured tangent plane plus fixed local lidar residuals; geometry never trained')),
        image_size=[tile_px,tile_px],color_space='8-bit sRGB, reference exposure appearance, not intrinsic albedo or PBR',
        mask_convention=dict(hole_mask='255/1=to_fill; 0=known_or_invalid',valid_mask='255/1=visible_source_supported_target_domain; 0=unknown',
            invalid_rgb='constant gray 128; loss and metrics must ignore valid=0',scoring_domain='hole & valid',
            full_valid_domain='valid denotes every supported target pixel, including known context and pseudo-hole'),
        train=[s['id'] for s in samples if s['split']=='train'],eval=[s['id'] for s in samples if s['split']=='eval'],samples=samples,
        sources='sources.json',atlas=dict(world_xyz='atlas/world_xyz.npy',local_xyz='atlas/local_xyz.npy',shape_hw=shape_xy[::-1].tolist(),
            cell_m=cell,world_from_uv=atlas_transform.tolist(),coordinate_anchor=anchor.tolist(),plane=plane.tolist(),
            model_from_world=model_from_world.tolist(),input_rgb='atlas/input_rgb.npy',observed_valid='atlas/observed_valid.npy',
            geometry_valid='atlas/geometry_valid.npy',exclude_world_mask='atlas/exclude_world_mask.npy',eval_world_mask='atlas/eval_world_mask.npy',
            feature_exclude_world_mask='atlas/feature_exclude_world_mask.npy',
            supervision_only_rgb='atlas/supervision_rgb.npy',supervision_only_valid='atlas/supervision_valid.npy'),
        split_policy=dict(kind='same_scene_disjoint_world_region_pilot',eval_local_x_start_m=float((split_tx+low[0])*cell),
            duplicate_world_cells_between_samples=0,cross_street_generalization=False,
            exclusions='all query holes excluded from every source feature input across all frames; all query holes plus full eval tiles excluded from candidates; legal eval context retained'),
        evidence_limits=['Pseudo-occlusion measures recovery of visible road appearance, never true hidden road truth.',
            'The source view with best cosine/distance weight fixes target exposure; illumination and view dependence remain.',
            'Measured-plane/local-height geometry and semantic/foreground/depth guards are fixed approximations, not dense ground truth.',
            'Same-world visible evidence, different-world material reference, generated prediction and unknown are separate provenance roles.'],
        source_config=cfg,all_input_frames=raws,source_frames=source_raws)
    write(out/'sources.json',dict(schema='road_material_sources.v1',sources=obs_records,world_exclusion=manifest['atlas']['exclude_world_mask']))
    write(out/'dataset.json',manifest);write(out/'train.json',manifest['train']);write(out/'eval.json',manifest['eval'])
    write(out/'build_receipt.json',dict(status='BUILT_CPU_WORLD_ATLAS_PSEUDO_OCCLUSION',seconds=time.monotonic()-started,
        atlas_cells=count,geometry_supported_cells=int(geometry_valid.sum()),visible_source_cells=int(valid.sum()),
        raw_lidar_rows=len(measured),deduplicated_lidar_cells=len(dedup),source_frames=len(source_raws),all_frames_mask_checked=len(raws),
        excluded_world_cells=int(blocked_world.sum()),sparse_observations_excluded=sparse_masked_count,
        train=len(manifest['train']),eval=len(manifest['eval']),point_stats=point_stats,all_frame_mask_audit=mask_audit,
        native_model_updates=0,geometry_updates_during_training=0,render_updates=0))
    print('DATASET_READY',str(out/'dataset.json'),time.monotonic()-started,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.config,a.output)


if __name__=='__main__':
    main()

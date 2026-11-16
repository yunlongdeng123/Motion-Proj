"""从原始Waymo观测构建CPU道路纹理数据，支持世界块和整街段留出。

仅读取原始传感器观测与缓存预训练语义模型；几何在数据准备后冻结。
历史编辑、渲染图、学习几何和待评模型预测均不进入数据构建。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, label
from scipy.spatial import cKDTree


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save(path, value):
    value = np.asarray(value)
    if value.dtype == bool:
        value = value.astype(np.uint8) * 255
    elif value.dtype != np.uint8:
        value = np.uint8(np.clip(value, 0, 1) * 255 + .5)
    Image.fromarray(value).save(path)


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def robust_plane(points, center=None):
    center = np.zeros(2) if center is None else np.asarray(center)
    design = np.c_[points[:, :2] - center, np.ones(len(points))]
    coef = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
    for _ in range(7):
        residual = design @ coef - points[:, 2]
        weight = np.minimum(1., .07 / np.maximum(abs(residual), 1e-8))
        coef = np.linalg.lstsq(design * weight[:, None] ** .5, points[:, 2] * weight ** .5, rcond=None)[0]
    residual = abs(design @ coef - points[:, 2])
    coef[2] -= coef[:2] @ center
    return coef, residual


def extract(args, scene):
    import waymo_numpy_cpu as decoder
    from fast_range_matrix import matrix
    decoder.matrix = matrix
    pb = decoder.protobuf()
    source = args.selected / scene
    info = read(source / 'manifest.json')
    count = info['frames']
    raws = sorted(set(np.linspace(0, count - 1, args.source_frames).round().astype(int).tolist()))
    out = args.output / 'observations' / scene
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(raws)
    front_rows = {}
    for raw, payload, record in decoder.records(source / 'front.tfrecord', count - 1):
        if raw not in wanted:
            continue
        fr = pb.Frame.FromString(payload)
        front = next(x for x in fr.images if x.name == 1)
        cal = next(x for x in fr.context.camera_calibrations if x.name == 1)
        image_path = out / f'{raw:03d}.jpg'
        image_path.write_bytes(front.image)
        rectangles = []
        for group in fr.camera_labels:
            if group.name != 1:
                continue
            for obj in group.labels:
                box = obj.box
                rectangles.append(dict(id=obj.id, type=int(obj.type), xyxy=[
                    box.center_x-box.length/2, box.center_y-box.width/2,
                    box.center_x+box.length/2, box.center_y+box.width/2]))
        boxes = []
        for obj in fr.laser_labels:
            b = obj.box
            boxes.append(dict(id=obj.id, type=int(obj.type), center=[b.center_x, b.center_y, b.center_z],
                              dimensions=[b.length, b.width, b.height], heading=b.heading))
        obs = dict(raw=raw, scene_id=scene, context_id=fr.context.name, timestamp_micros=fr.timestamp_micros,
            ego_world=list(fr.pose.transform), camera_pose_world=list(front.pose.transform),
            camera_extrinsic=list(cal.extrinsic.transform), camera_intrinsic_Brown=list(cal.intrinsic),
            camera_size=[cal.width, cal.height], image_path=str(image_path),
            lidar_path=str(out / f'{raw:03d}_lidar.bin'), front_camera_rectangles=rectangles,
            boxes_ego=boxes, original_front_record=record)
        write(out / f'{raw:03d}.json', obs)
        front_rows[raw] = obs
    for raw, payload, record in decoder.records(source / 'lidar.tfrecord', count - 1):
        if raw not in wanted:
            continue
        fr = pb.Frame.FromString(payload)
        assert fr.timestamp_micros == front_rows[raw]['timestamp_micros']
        cloud, sensors = decoder.lidar(fr, pb)
        cloud.astype('<f4').tofile(out / f'{raw:03d}_lidar.bin')
        front_rows[raw].update(lidar_rows=len(cloud), sensor_returns=sensors, original_lidar_record=record)
        write(out / f'{raw:03d}.json', front_rows[raw])
        event('DECODE', scene=scene, raw=raw, points=len(cloud))
    write(out / 'receipt.json', dict(scene_id=scene, context_id=info['context_id'], source=info,
        sampled_raws=raws, source_frame_count=count, sampling='uniform linspace before any image scoring',
        RGB='exact original FRONT JPEG bytes', LiDAR='five sensors, two returns, NLZ filtered; TOP pixel ego compensated',
        frames=[dict(raw=x, path=str(out / f'{x:03d}.json')) for x in raws], GPU_queries=0))
    return out


def semantics(args, scene):
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    torch.set_num_threads(2)
    model = SegformerForSemanticSegmentation.from_pretrained(str(args.semantic_model), local_files_only=True).eval()
    processor = SegformerImageProcessor.from_pretrained(str(args.semantic_model), local_files_only=True)
    bank = args.output / 'observations' / scene
    receipt = read(bank / 'receipt.json')
    rows = []
    start = time.monotonic()
    for item in receipt['frames']:
        obs = read(item['path'])
        path = bank / f'{obs["raw"]:03d}_semantic.png'
        if not path.exists():
            image = Image.open(obs['image_path']).convert('RGB')
            with torch.inference_mode():
                data = processor(images=image, return_tensors='pt')
                logits = model(**data).logits
                labels = torch.nn.functional.interpolate(logits, size=(640, 960), mode='bilinear', align_corners=False).argmax(1)[0]
            save(path, labels.numpy().astype(np.uint8))
        labels = np.asarray(Image.open(path))
        obs['semantic_path'] = str(path)
        write(item['path'], obs)
        rows.append(dict(raw=obs['raw'], road_pixels=int((labels == 0).sum())))
        event('SEMANTIC', scene=scene, raw=obs['raw'], elapsed=time.monotonic()-start)
    write(bank / 'semantic_receipt.json', dict(model=str(args.semantic_model), model_family='pretrained SegFormer-B0 Cityscapes',
        road_class=0, inference_device='CPU', GPU_queries=0, frames=rows, seconds=time.monotonic()-start,
        limitation='semantic predictions plus raw camera boxes are protection proxies, not manual road truth'))


def annotations(args, scene):
    import waymo_numpy_cpu as decoder
    pb = decoder.protobuf()
    bank = args.output / 'observations' / scene
    receipt = read(bank/'receipt.json')
    wanted = set(receipt['sampled_raws'])
    source = receipt['source']['source_path']
    rows = []
    for raw,payload,_ in decoder.records(source,receipt['source_frame_count']-1):
        if raw not in wanted:
            continue
        frame = pb.Frame.FromString(payload)
        path = bank/f'{raw:03d}.json'
        obs = read(path)
        assert obs['timestamp_micros']==frame.timestamp_micros
        rectangles=[]
        for name,groups in [('raw_camera_labels',frame.camera_labels),('raw_projected_lidar_labels',frame.projected_lidar_labels)]:
            for group in groups:
                if group.name!=1:
                    continue
                for obj in group.labels:
                    b=obj.box
                    rectangles.append(dict(id=obj.id,type=int(obj.type),source=name,xyxy=[
                        b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2]))
        obs['front_camera_rectangles']=rectangles
        obs['foreground_annotation_source']=source
        obs['foreground_annotation_policy']='all original FRONT camera and original FRONT projected-LiDAR 2D boxes, union'
        write(path,obs)
        rows.append(dict(raw=raw,camera_boxes=sum(b['source']=='raw_camera_labels' for b in rectangles),
                         projected_lidar_boxes=sum(b['source']=='raw_projected_lidar_labels' for b in rectangles)))
    write(bank/'annotation_receipt.json',dict(source=source,frames=rows,policy='union raw camera_labels and projected_lidar_labels',GPU_queries=0))
    event('ANNOTATIONS_READY',scene=scene,frames=len(rows),boxes=sum(x['camera_boxes']+x['projected_lidar_boxes'] for x in rows))


def camera_kwargs(camera):
    return {k: camera[k] for k in ('camera_K_native', 'world_to_camera', 'image_hw', 'distortion_Brown')}


def road_guard(obs):
    semantic = np.asarray(Image.open(obs['semantic_path']))
    h, w = semantic.shape
    guard = semantic >= 11
    scale = np.asarray([w/obs['camera_size'][0], h/obs['camera_size'][1]] * 2)
    for box in obs['front_camera_rectangles']:
        q = np.asarray(box['xyxy']) * scale
        x0, y0 = np.maximum(np.floor(q[:2]).astype(int)-3, 0)
        x1, y1 = np.minimum(np.ceil(q[2:]).astype(int)+4, [w, h])
        guard[y0:y1, x0:x1] = True
    guard = binary_dilation(guard, iterations=2)
    # 收缩语义/前景边界，保护原图双线性采样的四邻域。
    good = (semantic == 0) & ~guard
    good &= ~binary_dilation(~good, iterations=1)
    return good, guard


def marking_component(rgb, valid):
    luma=rgb@np.array([.2126,.7152,.0722])
    # 所有街段统一对比度门控，避免低曝光排除标线样本。
    background=float(np.quantile(luma[valid],.30))
    white = (luma > background+.075) & (rgb.min(2) > .28) & (np.ptp(rgb, axis=2) < .20)
    yellow = (rgb[:, :, 0] > .28) & (rgb[:, :, 1] > .22) & (rgb[:, :, 2] < .50) & (rgb[:, :, 0]-rgb[:, :, 2] > .095)
    mask = (white | yellow) & valid
    components, n = label(mask)
    best = None
    for index in range(1, n+1):
        y, x = np.where(components == index)
        if len(x) < 45:
            continue
        points = np.c_[x, y]
        center = points.mean(0)
        _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
        t = (points-center) @ axes[0]
        extent = np.quantile(t, .98)-np.quantile(t, .02)
        if extent < 65 or singular[0]/max(singular[1], 1e-6) < 2.:
            continue
        candidate = dict(center=center, tangent=axes[0], extent=extent, component=components == index)
        if best is None or extent > best['extent']:
            best = candidate
    return best


def make_hole(tile, number):
    yy, xx = np.mgrid[:128, :128]
    component = tile['component']
    if component is None:
        return ((abs(xx-64) < 26) & (abs(yy-64) < 28)) if number % 2 == 0 else (((xx-64)/27)**2+((yy-64)/29)**2 < 1)
    center, tangent = component['center'], component['tangent']
    delta = np.stack((xx-center[0], yy-center[1]), -1)
    along = delta @ tangent
    across = delta @ np.asarray([-tangent[1], tangent[0]])
    half = min(25., component['extent']*.25)
    hole = (abs(along) < half) & (abs(across) < 27.)
    line = component['component']
    assert (line & hole).sum() > 15
    assert (line & ~hole & (along < -half)).sum() > 10
    assert (line & ~hole & (along > half)).sum() > 10
    return hole


def build(args, scene):
    from world_feature_field import transform_xyz, camera_from_observation, project_world_to_camera, bilinear_sample
    start = time.monotonic()
    bank = args.output / 'observations' / scene
    receipt = read(bank / 'receipt.json')
    obs = [read(x['path']) for x in receipt['frames']]
    anchor = np.asarray(obs[0]['ego_world']).reshape(4, 4)
    # 首帧ego观测固定道路图表朝向，Z轴保持世界竖直。
    yaw = np.arctan2(anchor[1, 0], anchor[0, 0])
    anchor[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    inv = np.linalg.inv(anchor)
    caches = []
    road_points = []
    for o in obs:
        camera = camera_from_observation(o)
        good, guard = road_guard(o)
        cloud = np.fromfile(o['lidar_path'], dtype='<f4').reshape(-1, 6)[:, :3]
        world = transform_xyz(cloud, np.asarray(o['ego_world']).reshape(4, 4))
        uv, depth, inside = project_world_to_camera(world, **camera_kwargs(camera))
        px = np.floor(np.clip(np.nan_to_num(uv, nan=-1), -1e6, 1e6)*[960/camera['image_hw'][1], 640/camera['image_hw'][0]]).astype(int)
        px = np.clip(px, [0, 0], [959, 639])
        near = np.full((640, 960), np.inf)
        ids = np.flatnonzero(inside)
        np.minimum.at(near, (px[ids, 1], px[ids, 0]), depth[ids])
        keep = inside & good[px[:, 1], px[:, 0]] & (depth <= near[px[:, 1], px[:, 0]]+.25)
        keep &= (cloud[:, 0] > 2) & (cloud[:, 0] < 48) & (abs(cloud[:, 1]) < 24) & (cloud[:, 2] > -4) & (cloud[:, 2] < .25)
        road_points.append(transform_xyz(world[keep], inv))
        caches.append(dict(obs=o, camera=camera, good=good, image=np.asarray(Image.open(o['image_path']).convert('RGB'), np.float32)/255))
    points = np.concatenate(road_points)
    assert len(points) > 1000, (scene, len(points))
    keys, inverse = np.unique(np.floor(points/.1).astype(np.int32), axis=0, return_inverse=True)
    den = np.bincount(inverse)
    points = np.column_stack([np.bincount(inverse, weights=points[:, i])/den for i in range(3)])
    tree = cKDTree(points[:, :2])
    blocks = np.unique(np.floor(points[:, :2]/2.56).astype(int), axis=0)
    pool = []
    unit = np.c_[np.tile(np.arange(128)+.5, 128), np.repeat(np.arange(128)+.5, 128), np.zeros(128**2)]
    probe = np.array([[64.5, 64.5, 0], [80.5, 64.5, 0], [48.5, 64.5, 0], [64.5, 80.5, 0], [64.5, 48.5, 0]])
    reject = dict(plane_support=0, slope_or_residual=0, projection_or_road=0)
    for bx, by in blocks:
        center = (np.asarray([bx, by])+.5)*2.56
        ids = tree.query_ball_point(center, 4.5)
        if len(ids) < 50:
            reject['plane_support'] += 1
            continue
        plane, residual = robust_plane(points[ids], center)
        if np.linalg.norm(plane[:2]) > .25 or np.median(residual) > .10:
            reject['slope_or_residual'] += 1
            continue
        local_origin = np.array([bx*2.56, by*2.56, plane[:2]@np.asarray([bx, by])*2.56+plane[2]])
        matrix = np.eye(4)
        matrix[:3, 0] = anchor[:3, :3] @ np.array([.02, 0, plane[0]*.02])
        matrix[:3, 1] = anchor[:3, :3] @ np.array([0, .02, plane[1]*.02])
        matrix[:3, 3] = transform_xyz(local_origin[None], anchor)[0]
        quality = []
        for ci, c in enumerate(caches):
            uv, _, inside = project_world_to_camera(transform_xyz(probe, matrix), **camera_kwargs(c['camera']))
            if not inside.all():
                continue
            jac = np.column_stack(((uv[1]-uv[2])/32, (uv[3]-uv[4])/32))
            span = float(np.linalg.svd(jac, compute_uv=False).min()*128)
            if span >= args.min_span:
                quality.append((span, ci))
        quality.sort(reverse=True)
        world = transform_xyz(unit, matrix)
        selected = None
        for span, ci in quality:
            c = caches[ci]
            uv, _, inside = project_world_to_camera(world, **camera_kwargs(c['camera']))
            # 四个原图双线性邻点都须满足保守收缩后的道路/前景保护。
            native = np.floor(uv-.5).astype(int)
            valid = inside.copy()
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                px = np.clip(np.floor((native+[dx+.5, dy+.5])*[960/c['camera']['image_hw'][1], 640/c['camera']['image_hw'][0]]).astype(int), [0, 0], [959, 639])
                valid &= c['good'][px[:, 1], px[:, 0]]
            valid = valid.reshape(128, 128)
            if valid.mean() < .98 or not valid[28:100, 28:100].all():
                continue
            rgb = bilinear_sample(c['image'], uv, c['camera']['image_hw']).reshape(128, 128, 3)
            rgb[~valid] = 128/255
            component = marking_component(rgb, valid)
            # 横向道路标线/人行横道同样保留，自动连通分量仅用于分层。
            selected = dict(key=[int(bx), int(by)], matrix=matrix, world=world.reshape(128, 128, 3),
                rgb=rgb, valid=valid, uv=uv.reshape(128, 128, 2), raw=c['obs']['raw'], image=c['obs']['image_path'],
                span=span, component=component, marked=component is not None, plane=plane,
                plane_support=len(ids), plane_residual_median_m=float(np.median(residual)))
            break
        if selected is None:
            reject['projection_or_road'] += 1
        else:
            pool.append(selected)
    event('POOL', scene=scene, total=len(pool), marked=sum(t['marked'] for t in pool), rejects=reject, seconds=time.monotonic()-start)
    preview = sorted(pool, key=lambda t:-t['span'])[:24]
    contact = Image.new('RGB',(128*6,148*((len(preview)+5)//6)),(18,22,30))
    draw = ImageDraw.Draw(contact)
    for i,t in enumerate(preview):
        contact.paste(Image.fromarray(np.uint8(np.clip(t['rgb'],0,1)*255)),((i%6)*128,(i//6)*148))
        draw.text(((i%6)*128,(i//6)*148+128),str(t['key'])+' '+str(t['raw']),fill='white')
    contact.save(args.output/'data'/f'{scene}_pool_contact.jpg',quality=90)
    assert len(pool) >= args.train_count+args.eval_count+4 if scene not in args.unseen else len(pool) >= args.eval_count+4
    rng = np.random.default_rng(args.seed)
    marked = [i for i, t in enumerate(pool) if t['marked']]
    plain = [i for i, t in enumerate(pool) if not t['marked']]
    rng.shuffle(marked)
    rng.shuffle(plain)
    unseen = scene in args.unseen
    if unseen:
        eval_idx = marked[:min(2, args.eval_count)]+plain[:max(0, args.eval_count-min(2, len(marked)))]
        train_idx = []
    else:
        assert len(marked) >= 4 and len(plain) >= 12, (scene, len(marked), len(plain))
        eval_idx = marked[:2]+plain[:args.eval_count-2]
        train_idx = marked[2:5]+plain[args.eval_count-2:args.eval_count-2+args.train_count-3]
    selected_ids = set(train_idx+eval_idx)
    refs = [i for i in range(len(pool)) if i not in selected_ids]
    assert refs
    out = args.output / 'data' / scene
    out.mkdir(parents=True, exist_ok=False)
    (out / 'samples').mkdir()
    (out / 'candidates').mkdir()
    samples = []
    candidates = []
    for number, index in enumerate(refs):
        t = pool[index]
        cid = f'{scene}_candidate_{number:03d}'
        folder = out / 'candidates' / cid
        folder.mkdir()
        save(folder / 'input.png', t['rgb'])
        save(folder / 'valid.png', t['valid'])
        candidates.append(dict(id=cid, key=t['key'], input=str(folder / 'input.png'), valid_mask=str(folder / 'valid.png'),
            world_center=t['world'].mean((0, 1)).tolist(), target_source_raw=t['raw'], min_axis_span_native_px=t['span'],
            world_from_uv=t['matrix'].tolist(), marking_stratum=t['marked'], role='different_world_visible_reference_only'))
    for number, (index, split) in enumerate([(i, 'train') for i in train_idx]+[(i, 'eval') for i in eval_idx]):
        t = pool[index]
        sid = f'{scene}_{split}_{number:03d}'
        folder = out / 'samples' / sid
        folder.mkdir()
        hole = make_hole(t, number) & t['valid']
        assert hole.sum() > 800
        target = t['rgb']
        inp = target.copy()
        inp[hole] = 128/255
        center = t['world'].mean((0, 1))
        reference = max(candidates, key=lambda c: np.linalg.norm(np.asarray(c['world_center'])-center))
        for name, value in [('target', target), ('input', inp), ('hole', hole), ('valid', t['valid'])]:
            save(folder / f'{name}.png', value)
        Image.open(reference['input']).save(folder / 'reference.png')
        Image.open(reference['valid_mask']).save(folder / 'reference_valid.png')
        np.save(folder / 'world_xyz.npy', t['world'].astype(np.float64))
        np.save(folder / 'source_uv_native.npy', t['uv'].astype(np.float32))
        np.savez_compressed(folder / 'tile.npz', world_xyz=t['world'], world_from_uv=t['matrix'], valid=t['valid'], hole=hole,
            target=target, input=inp, uv_native=t['uv'], fixed_plane=t['plane'])
        s = dict(id=sid, scene_id=scene, split=split,
            role='unseen_scene_transfer_evaluation' if unseen else ('texture_lora_train' if split=='train' else 'same_scene_world_block_holdout'),
            **{k:str(folder / filename) for k,filename in [('input','input.png'), ('reference','reference.png'),
            ('target','target.png'), ('hole_mask','hole.png'), ('valid_mask','valid.png'), ('world_xyz','world_xyz.npy'),
            ('reference_valid_mask','reference_valid.png'), ('source_uv_native','source_uv_native.npy'), ('tile_npz','tile.npz')]},
            world_from_uv=t['matrix'].tolist(), world_block_key=t['key'], tile_size_m=2.56, tile_texel_m=.02,
            world_bounds=[t['world'].min((0,1)).tolist(), t['world'].max((0,1)).tolist()],
            target_source_raw=t['raw'], target_source_image=t['image'], min_axis_span_native_px=t['span'],
            valid_pixels=int(t['valid'].sum()), hole_pixels=int(hole.sum()), marking_stratum=t['marked'],
            marking_hole_both_ends_visible=t['marked'], reference_candidate_id=reference['id'],
            reference_selection='maximum Euclidean world-center distance; no hidden target appearance or prediction used',
            reference_world_block_key=reference['key'], reference_source_raw=reference['target_source_raw'],
            plane_support_points=t['plane_support'], plane_residual_median_m=t['plane_residual_median_m'],
            provenance=dict(geometry='fixed independently measured local road plane per disjoint block; never trained',
                target='original single-frame visible-road RGB projected through Brown camera',
                reference='different-world visible-road candidate; all query blocks excluded',
                semantics='cached pretrained SegFormer-B0 Cityscapes CPU',
                foreground_guard='original FRONT camera 2D boxes plus semantic foreground; conservative dilation',
                evaluation='visible-road pseudo-occlusion only; not true hidden road ground truth', generated=False))
        samples.append(s)
    manifest = dict(schema='road_material_dataset.v1', scene_id=scene, seed=args.seed, image_size=[128,128],
        samples=samples, train=[s['id'] for s in samples if s['split']=='train'], eval=[s['id'] for s in samples if s['split']=='eval'],
        candidates=candidates, source_frames=receipt['sampled_raws'], source_observation_bank=str(bank),
        geometry=dict(coordinate_anchor=anchor.tolist(), mode='fixed measured local plane chart, one plane per nonoverlapping world block',
            training_geometry_updates=0, topology='all blocks use common fixed world XY grid; local measured heights fixed before training'),
        split_policy=dict(kind='whole_scene_holdout' if unseen else 'fixed_seed_disjoint_world_blocks',
            unseen_scene=unseen, train_eval_duplicate_world_blocks=0, query_reference_duplicate_world_blocks=0,
            note='same-scene eval uses spatially disjoint blocks; separate entire scene measures transfer'),
        quality_gate=dict(native_min_axis_span_px=args.min_span, valid_fraction_min=.98, inner_valid_fraction=1.,
            all_reference_candidates_same_gate=True, pool_size=len(pool), pool_marking_tiles=len(marked)),
        proxy_boundary=['RGB is exposure-dependent appearance, not intrinsic albedo or PBR.',
            'Predicted road semantics plus raw 2D foreground boxes are protection proxies.',
            'Geometry derives only from measured returns; per-texel LiDAR returns are not required.',
            'Pseudo-hole placement may use labelled visible target to define marking strata; hidden RGB never enters model condition.',
            'No per-frame manual selection and no historical edited imagery.'],
        source_provenance=receipt, semantic_provenance=read(bank/'semantic_receipt.json'),
        annotation_provenance=read(bank/'annotation_receipt.json'))
    write(out / 'dataset.json', manifest)
    write(out / 'selection_receipt.json', dict(scene_id=scene, train=len(train_idx), eval=len(eval_idx), candidates=len(candidates),
        train_marking=sum(s['marking_stratum'] for s in samples if s['split']=='train'),
        eval_marking=sum(s['marking_stratum'] for s in samples if s['split']=='eval'),
        scanned_world_blocks=len(blocks), rejects=reject, seconds=time.monotonic()-start, GPU_queries=0,
        min_native_span_px=min(s['min_axis_span_native_px'] for s in samples), architecture='sensor RGB + calibration + LiDAR + road/foreground proxy -> fixed world blocks -> pseudo-hole/context + other-world reference -> texture-only train/eval'))
    subset = [s for s in samples if s['split']=='train'][:3]+[s for s in samples if s['split']=='eval']
    canvas = Image.new('RGB', (86+128*5, (128+22)*len(subset)), (18,22,30))
    draw = ImageDraw.Draw(canvas)
    for row,s in enumerate(subset):
        draw.text((2,row*150+6),s['split']+' '+s['id'][-3:],fill='white')
        for col,key in enumerate(['target','input','reference','hole_mask','valid_mask']):
            canvas.paste(Image.open(s[key]).convert('RGB'), (86+128*col,row*150))
        draw.text((86,row*150+129),'target | masked input | other-world reference | hole | valid',fill='white')
    canvas.save(out/'contact_qa.png')
    event('DATASET_READY', scene=scene, path=str(out/'dataset.json'), train=len(train_idx), eval=len(eval_idx), seconds=time.monotonic()-start)


def merge(args):
    samples = []
    manifests = []
    for path in [args.legacy]+[args.output/'data'/s/'dataset.json' for s in args.scenes]:
        m = read(path)
        manifests.append(str(path))
        for row in m['samples']:
            row = dict(row)
            for key in ['input','reference','target','hole_mask','valid_mask','world_xyz','point_ids','reference_valid_mask','source_uv_native','source_weight','tile_npz']:
                if row.get(key):
                    row[key] = str(path.parent / row[key])
            source_scene=m['scene_id']
            row['source_scene_key']=source_scene
            row['scene_id']='scene'+source_scene.rsplit('-',1)[-1] if source_scene.startswith('waymo-') else source_scene
            row['evaluation_role']=('old_scene_spatial' if row['scene_id']=='scene065' else
                'unseen_scene' if source_scene in args.unseen else 'new_train_scene_spatial')
            row.setdefault('role', 'legacy_texture_lora_train' if row['split']=='train' else 'legacy_same_scene_world_block_holdout')
            samples.append(row)
    counts = {}
    for s in samples:
        c = counts.setdefault(s['scene_id'],dict(train=0,eval=0,marking_train=0,marking_eval=0))
        c[s['split']] += 1
        c['marking_'+s['split']] += int(s.get('marking_stratum',False))
    m = dict(schema='road_material_dataset.v1',scene_id='multiscene',samples=samples,image_size=[128,128],seed=args.seed,
        train=[s['id'] for s in samples if s['split']=='train'],eval=[s['id'] for s in samples if s['split']=='eval'],
        component_manifests=manifests,scene_counts=counts,unseen_scenes=['scene'+s.rsplit('-',1)[-1] for s in args.unseen],
        geometry_policy='all component geometries fixed; legacy geometry/data untouched; texture LoRA only',
        evaluation_policy='same-scene disjoint-world-block holdout and separately reported whole unseen scene',
        reference_policy='fixed farthest-world-block candidates; no hidden-query appearance selection',
        coordinate=dict(units='m',pixel_centers='world_from_uv @ [u+0.5,v+0.5,0,1]',
            new_world_xyz='NPY array path, shape128x128x3 float64',geometry_updates=0))
    write(args.output/'data'/'dataset.json',m)
    event('MERGED_READY',path=str(args.output/'data'/'dataset.json'),scene_counts=counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['extract','semantics','annotations','build','merge'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selected',type=Path,default=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101/selected'))
    p.add_argument('--decoder-root',type=Path,default=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101'))
    p.add_argument('--geometry-library',type=Path,default=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/road_material_field'))
    p.add_argument('--semantic-model',type=Path,default=Path('/root/autodl-tmp/models/worldsim_v6/r20_semantic_segformer_cityscapes'))
    p.add_argument('--legacy',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_material_field_20261112_r1/data/scene065_v3_clear64/dataset_fineguard.json'))
    p.add_argument('--scenes',nargs='+',default=['waymo-val-067','waymo-val-079','waymo-val-081'])
    p.add_argument('--unseen',nargs='+',default=['waymo-val-081'])
    p.add_argument('--source-frames',type=int,default=17)
    p.add_argument('--train-count',type=int,default=12)
    p.add_argument('--eval-count',type=int,default=4)
    p.add_argument('--min-span',type=float,default=64.)
    p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args()
    sys.path.insert(0,str(args.decoder_root))
    sys.path.insert(0,str(args.geometry_library))
    if args.stage=='merge':
        merge(args)
    else:
        for scene in args.scenes:
            globals()[args.stage](args,scene)


if __name__=='__main__':
    main()

"""固定 V3 世界单元的 49/199 配对曝光审计；只读原始数据，不生成训练包。"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import zlib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import numpy as np
from PIL import Image
from scipy.ndimage import laplace

from prepare_multiview_road import obb, read
from road_source_audit import GATES


def packed_matrix(compressed, dataset_pb2):
    """已知 MatrixFloat 的 packed float32 零拷贝解码；形状仍由原 protobuf 解码。"""
    if not compressed:
        raise ValueError('missing measured range/pose matrix')
    fields = [(f.name, f.number, f.type, f.GetOptions().packed) for f in dataset_pb2.MatrixFloat.DESCRIPTOR.fields]
    if fields != [('data', 1, 2, True), ('shape', 2, 11, False)]:
        raise ValueError('unsupported MatrixFloat schema')
    value = zlib.decompress(compressed)
    cursor, chunks, shape = 0, [], None
    def varint():
        nonlocal cursor
        result = 0
        for shift in range(0, 70, 7):
            if cursor >= len(value):
                raise ValueError('truncated matrix field')
            byte = value[cursor]
            cursor += 1
            result |= (byte & 127) << shift
            if byte < 128:
                return result
        raise ValueError('oversized matrix varint')
    while cursor < len(value):
        tag, size = varint(), varint()
        end = cursor+size
        if end > len(value) or tag not in [10, 18]:
            raise ValueError('unsupported or truncated matrix wire data')
        if tag == 10:
            if size % 4:
                raise ValueError('invalid packed float length')
            chunks.append(np.frombuffer(value, '<f4', count=size//4, offset=cursor))
        else:
            if shape is not None:
                raise ValueError('duplicate matrix shape')
            shape = dataset_pb2.MatrixShape()
            shape.ParseFromString(value[cursor:end])
        cursor = end
    if not chunks or shape is None or not shape.dims or min(shape.dims) <= 0:
        raise ValueError('empty measured matrix')
    array = np.concatenate(chunks).reshape(tuple(shape.dims))
    if not np.isfinite(array).all():
        raise ValueError('nonfinite measured matrix')
    return array


def frame_inventory(path):
    """从真实文件边界确定帧数，不假定 199，也不创建新的校验和。"""
    offsets = []
    with path.open('rb') as stream:
        while stream.tell() < path.stat().st_size:
            offsets.append(stream.tell())
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError('truncated record header')
            size = struct.unpack('<Q', header[:8])[0]
            if not 0 < size < 96*1024**2:
                raise ValueError('invalid record size')
            stream.seek(size+4, 1)
        if stream.tell() != path.stat().st_size:
            raise ValueError('truncated record payload')
    return offsets


def rectangle(mask, bounds):
    x0, y0, x1, y1 = bounds
    if not np.isfinite(bounds).all():
        raise ValueError('nonfinite annotation rectangle')
    x0, y0 = np.maximum(np.floor([x0, y0]), 0).astype(int)
    x1, y1 = np.minimum(np.ceil([x1, y1])+1, [mask.shape[1], mask.shape[0]]).astype(int)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True


def annotation_envelope(frame, boxes, intrinsic, w2c, transform):
    """全类别原始相机框、投影标签框及三维框投影外包络；不是 SAM/语义 GT。"""
    mask = np.zeros((320, 480), bool)
    groups = list(frame.camera_labels)+list(frame.projected_lidar_labels)
    if not any(group.name == 1 for group in groups):
        return np.ones_like(mask)  # 缺失 FRONT 标注不能解释为已知无遮挡。
    for group in groups:
        if group.name != 1:
            continue
        for label in group.labels:
            b = label.box
            rectangle(mask, np.array([b.center_x-b.length/2, b.center_y-b.width/2,
                                      b.center_x+b.length/2, b.center_y+b.width/2])*.25)
    signs = np.array([[a, b, c] for a in [-1, 1] for b in [-1, 1] for c in [-1, 1]])
    for box in boxes:
        corners = box['center']+(signs*box['half']) @ box['axes'].T
        camera = transform(corners, w2c)
        if (camera[:, 2] <= .1).all():
            continue
        if (camera[:, 2] <= .1).any():
            # 先裁剪三维框的12条边；相机侧后方跨近面车框不能误遮整幅图。
            clipped = list(camera[camera[:, 2] > .1])
            for a in range(8):
                for b in range(a+1, 8):
                    if np.count_nonzero(signs[a] != signs[b]) != 1:
                        continue
                    first, second = camera[a], camera[b]
                    if (first[2] <= .1) != (second[2] <= .1):
                        clipped.append(first+(.1-first[2])/(second[2]-first[2])*(second-first))
            camera = np.asarray(clipped)
        uvw = camera @ intrinsic.T
        uv = uvw[:, :2]/uvw[:, 2:]
        rectangle(mask, [*uv.min(0), *uv.max(0)])
    return mask


def paired_summary(old, added, supported, ids, minimum=2):
    before, extra = old[:, ids].sum(0), added[:, ids].sum(0)
    after = before+extra
    support = supported[ids]
    def counts(value):
        return dict(confirmed=int((support & (value >= minimum)).sum()),
                    zero_source_supported=int((support & (value == 0)).sum()),
                    single_source_unconfirmed=int((support & (value == 1)).sum()),
                    unsupported=int((~support).sum()))
    return dict(world_units=len(ids), original49=counts(before), all199=counts(after),
                newly_confirmed=int((support & (before < minimum) & (after >= minimum)).sum()),
                zero_to_confirmed=int((support & (before == 0) & (after >= minimum)).sum()),
                single_to_confirmed=int((support & (before == 1) & (after >= minimum)).sum()),
                zero_to_single=int((support & (before == 0) & (after == 1)).sum()),
                already_confirmed_with_more_sources=int((support & (before >= minimum) & (extra > 0)).sum()),
                added_source_unit_pairs=int(extra.sum()))


def audit(bundle_path, output, raw_reader, engine):
    from utils.v3_road_geometry import transform, sparse_depth_maps, source_observation
    if output.exists():
        raise FileExistsError('preserve all previous audit outputs')
    meta = read(bundle_path)
    engineering = Path(meta['original_engineering'])
    source = engineering.parent/'inputs/scene0008_089_137'
    manifest = read(source/'preparation_manifest.json')
    raw = Path(manifest['raw_path'])
    offsets = frame_inventory(raw)
    if len(offsets) != 199 or meta['frame_order'] != [f'0008{i:03d}' for i in range(89, 138)]:
        raise ValueError('actual199/original49 frame contract changed')
    spec = importlib.util.spec_from_file_location('existing_waymo_numpy_cpu', raw_reader)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    pb = reader.protobuf()
    # 只替换相同数据的解码，不改变 LiDAR 数学、NLZ、returns 或 TOP pose。
    reader.matrix = packed_matrix
    with np.load(bundle_path.parent/meta['units'], allow_pickle=False) as archive:
        units = {k: archive[k] for k in archive.files}
    readiness = read(Path(meta['original_readiness']))
    anchor_inverse = np.linalg.inv(np.asarray(meta['coordinate_anchor']))
    plane = np.asarray(readiness['SGCM_geometry']['ground_fit']['plane_anchor'])
    sharpness = []
    for fid in meta['frame_order']:
        image = Image.open(meta['frames'][fid]['original_rgb']).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS)
        sharpness.append(float(np.var(laplace(np.asarray(image).astype(float).mean(2)/255.))))
    sharpness_reference = max(float(np.median(sharpness)), 1e-9)
    output.mkdir(parents=True)
    (output/'frames').mkdir()
    frame_rows, colors, uvs, depths, weights, bits, overlaps = [], [], [], [], [], [], []
    for index, payload, _ in reader.records(raw, len(offsets)-1):
        frame = pb.Frame()
        frame.ParseFromString(payload)
        if frame.context.name != manifest['segment_context_name']:
            raise ValueError('raw segment mismatch')
        pose = np.asarray(frame.pose.transform).reshape(4, 4)
        calibration = {line.split(':', 1)[0]: np.fromstring(line.split(':', 1)[1], sep=' ')
                       for line in reader.calibration_text(frame).splitlines()}
        camera = np.eye(4)
        camera[:3] = calibration['Tr_velo_to_cam_0'].reshape(3, 4)
        w2c = camera @ np.linalg.inv(pose)
        intrinsic = calibration['P0'].reshape(3, 4)[:, :3].copy()
        intrinsic[:2] *= .25
        full_rgb, _ = reader.front_rgb(frame)
        if full_rgb.shape[:2] != (1280, 1920):
            raise ValueError('camera resize contract changed')
        rgb = np.asarray(Image.fromarray(full_rgb).resize((480, 320), Image.Resampling.LANCZOS))
        cloud, _ = reader.lidar(frame, pb)
        if 89 <= index <= 137:
            fid = f'0008{index:03d}'
            original_cloud = np.fromfile(source/'velodyne'/f'{fid}.bin', np.float32).reshape(-1, 6)
            overlap = dict(raw=index, pose_equal=bool(np.array_equal(pose, np.loadtxt(source/'pose'/f'{fid}.txt'))),
                           image_equal=bool(np.array_equal(full_rgb, np.asarray(Image.open(source/'image/image_0'/f'{fid}.png').convert('RGB')))),
                           cloud_equal=bool(np.array_equal(cloud, original_cloud)),
                           calibration_equal=bool(np.allclose(w2c, meta['frames'][fid]['world_to_camera'], atol=1e-10, rtol=0)
                                                  and np.allclose(intrinsic, meta['frames'][fid]['intrinsic'], atol=1e-10, rtol=0)))
            if not all(value for key, value in overlap.items() if key != 'raw'):
                raise ValueError(f'raw/prepared mismatch: {overlap}')
            overlaps.append(overlap)
        world = transform(cloud[:, :3].astype(float), pose)
        local = transform(world, anchor_inverse)
        ground = (np.abs(local @ plane[:3]+plane[3]) <= .15)
        ground &= (local[:, 0] >= 3) & (local[:, 0] <= 90) & (np.abs(local[:, 1]) <= 8)
        ground_rows = np.flatnonzero(ground)
        ground_candidates = world[ground_rows]
        boxes, occupied = [], np.zeros(len(ground_rows), bool)
        for label in frame.laser_labels:
            b = label.box
            if min(b.length, b.width, b.height) <= 0:
                continue
            box = obb(dict(box={k: getattr(b, k) for k in ['center_x', 'center_y', 'center_z', 'length', 'width', 'height', 'heading']}), pose, transform)
            boxes.append(box)
            occupied |= (np.abs((ground_candidates-box['center']) @ box['axes']) <= box['half']).all(1)
        ground[ground_rows] &= ~occupied
        depth, valid, road = sparse_depth_maps(world, ground, intrinsic, w2c)
        quality = min(float(np.var(laplace(rgb.astype(float).mean(2)/255.)))/sharpness_reference, 1.)
        occluder = annotation_envelope(frame, boxes, intrinsic, w2c, transform)
        data = dict(K=intrinsic, w2c=w2c, rgb=rgb, boxes=boxes, depth=depth, valid=valid,
                    road=road, occluder=occluder, quality=quality)
        color, uv, measured, weight, gates = source_observation(units['world'], units['normal_world'],
                                                               units['surface_supported'], data, meta['policy'])
        gate_bits = sum(gates[g].astype(np.uint16) << i for i, g in enumerate(GATES))
        colors.append(color.astype(np.float32)); uvs.append(uv.astype(np.float32))
        depths.append(measured.astype(np.float32)); weights.append(weight.astype(np.float32)); bits.append(gate_bits)
        Image.fromarray(rgb).save(output/'frames'/f'{index}.jpg', quality=90)
        Image.fromarray(occluder.astype(np.uint8)*255).save(output/'frames'/f'{index}_occluder.png')
        frame_rows.append(dict(raw=index, timestamp_micros=int(frame.timestamp_micros),
                               original49=89 <= index <= 137, quality=quality, lidar_points=len(cloud),
                               annotated_boxes=len(boxes), camera_groups=len(frame.camera_labels),
                               projected_groups=len(frame.projected_lidar_labels),
                               front_annotation_present=any(g.name == 1 for g in list(frame.camera_labels)+list(frame.projected_lidar_labels)),
                               source_unit_pairs=int((weight > 0).sum()),
                               gate_rejections={g:int((~gates[g]).sum()) for g in GATES},
                               world_to_camera=w2c.tolist(), intrinsic=intrinsic.tolist()))
        (output/'progress.json').write_text(json.dumps(dict(status='CPU_AUDIT_RUNNING', completed_frames=index+1,
                                                           expected_frames=len(offsets), GPU_used=False, training_updates=0)), encoding='utf-8')
        print(json.dumps(dict(raw=index, pairs=frame_rows[-1]['source_unit_pairs'])), flush=True)
    weights = np.asarray(weights)
    accepted = weights > 0
    old_mask = np.array([row['original49'] for row in frame_rows])
    summaries = {}
    for name, ids in [('all_query', np.flatnonzero(units['is_query']))]+[
            (f'raw89_a{actor}', np.unique(units['query_unit_id'][(units['query_frame'] == 0) & (units['query_actor'] == actor)]))
            for actor in meta['actors']]:
        summaries[name] = paired_summary(accepted[old_mask], accepted[~old_mask], units['surface_supported'], ids,
                                         meta['policy']['minimum_views'])
        summaries[name]['native49_reference_confirmed'] = int((units['provenance'][ids] == 1).sum())
    if len(overlaps) != 49 or not np.array_equal(np.array([r['timestamp_micros'] for r in frame_rows])[89:138],
                                                [r['timestamp_micros'] for r in manifest['frames']]):
        raise ValueError('incomplete49 overlap/timestamp binding')
    np.savez_compressed(output/'exposure.npz', source_uv=np.asarray(uvs), source_rgb=np.asarray(colors),
                        measured_depth_m=np.asarray(depths), base_weight=weights, gate_bits=np.asarray(bits),
                        native49_weight=units['base_weight'], original49_mask=old_mask,
                        original49_count=accepted[old_mask].sum(0), added150_count=accepted[~old_mask].sum(0))
    report = dict(task='WS-V82-GN11-ROAD-EXPOSURE199-20261020', status='CPU_AUDIT_COMPLETE_NOT_TRAINING_INPUT',
                  input_bundle=str(bundle_path), raw_path=str(raw), raw_reader=str(raw_reader),
                  actual_frames=len(frame_rows), raw_range=[0, len(frame_rows)-1], original49_range=[89, 137],
                  added_ranges=[[0, 88], [138, 198]], fixed_world_unit_count=len(units['world']),
                  fixed_query_units=int(units['is_query'].sum()), fixed_surface=True, fixed_query_cohorts=True,
                  policy=meta['policy'], quality_reference_original49_median=sharpness_reference,
                  mask_policy='COMMON_ALL_RAW_CAMERA_PROJECTED_LABEL_AND_3D_BOX_ENVELOPE',
                  native49_mask_policy='five-ID + all-vehicle prepared masks; reference only, not paired numerator',
                  minimum_views=meta['policy']['minimum_views'], summaries=summaries, frames=frame_rows,
                  original49_overlap=overlaps, RGB_fused=False, training_input_promoted=False,
                  training_updates=0, GPU_used=False, human_verdict=None, quality_accepted=False,
                  limitations=['Confirmed means geometric audit gates, not dense semantic GT or texture quality acceptance.',
                               'Annotation envelopes replace unavailable outside49 SAM masks only in this separate audit; paired49 is recomputed identically.',
                               'Unlabeled occluders and sparse LiDAR can still limit evidence; zero source is not proof of permanent invisibility.',
                               'Surface, normals, units and original49 sharpness reference remain frozen; new geometry is not added.',
                               'No training reliability, gradients, causal benefit or extension experiment results exist.'],
                  failure_ledger_refs=['V82-F01', 'V82-F06'], failure_ledger_delta='none')
    (output/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output/'progress.json').write_text(json.dumps(dict(status=report['status'], completed_frames=len(frame_rows),
                                                       expected_frames=len(offsets), GPU_used=False, training_updates=0)), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-reader', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.engine))
    result = audit(args.bundle, args.output, args.raw_reader, args.engine)
    print(json.dumps(result['summaries'], indent=2))

"""Audit real first-return free segments against fixed canonical a6 volumes."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


READER = Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101')
CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
SCENE = '10094743350625019937_3420_000_3440_000'
RECORD = Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/training') / (
    f'segment-{SCENE}_with_camera_labels.tfrecord')
TRACK = '7MxwK9zTRZ9g5lc_ZC0z6A'
HELDOUT = (89, 95, 101, 105, 113, 121, 125, 130, 137)
PILOTS = (0, 105, 113, 130)
MARGIN_M = .15


def raw_ids(value):
    if ':' in value:
        start, stop = map(int, value.split(':'))
        return tuple(range(start, stop))
    return tuple(sorted(set(map(int, value.split(',')))))


def segments(frame, decoder, matrix, pb):
    """Yield world origins and endpoints with original first-return pixel lineage."""
    ego = np.asarray(frame.pose.transform).reshape(4, 4)
    lasers = {laser.name: laser for laser in frame.lasers}
    calibrations = sorted(frame.context.laser_calibrations, key=lambda item: item.name)
    assert [item.name for item in calibrations] == [1, 2, 3, 4, 5]
    top = matrix(lasers[1].ri_return1.range_image_pose_compressed, pb)
    for calibration in calibrations:
        image = matrix(lasers[calibration.name].ri_return1.range_image_compressed, pb)
        height, width, channels = image.shape
        assert channels >= 4
        distance = image[..., 0]
        valid = (distance > 0) & (image[..., 3] != np.float32(1.))
        y, x = np.nonzero(valid)
        ext = np.asarray(calibration.extrinsic.transform, dtype=np.float32).reshape(4, 4)
        if calibration.beam_inclinations:
            inclination = np.asarray(calibration.beam_inclinations, dtype=np.float32)[::-1]
        else:
            low, high = np.float32(calibration.beam_inclination_min), np.float32(calibration.beam_inclination_max)
            inclination = (((np.arange(height, dtype=np.float32) + .5) / height) * (high - low) + low)[::-1]
        assert len(inclination) == height
        ratio = (np.arange(width, 0, -1, dtype=np.float32) - .5) / width
        azimuth = (ratio * 2. - 1.) * np.float32(np.pi) - np.arctan2(ext[1, 0], ext[0, 0])
        ranges = distance[valid]
        sensor = np.column_stack((np.cos(azimuth[x]) * np.cos(inclination[y]) * ranges,
                                  np.sin(azimuth[x]) * np.cos(inclination[y]) * ranges,
                                  np.sin(inclination[y]) * ranges))
        endpoint_ego = sensor @ ext[:3, :3].T + ext[:3, 3]
        if calibration.name == 1:
            assert top.shape == (height, width, 6)
            poses = top[y, x]
            rotation = decoder.rotation(poses[:, 0], poses[:, 1], poses[:, 2]).astype(np.float64)
            origin = np.einsum('nij,j->ni', rotation, ext[:3, 3]) + poses[:, 3:]
            endpoint = np.einsum('nij,nj->ni', rotation, endpoint_ego.astype(np.float64)) + poses[:, 3:]
        else:
            origin = np.broadcast_to(ego[:3, :3] @ ext[:3, 3] + ego[:3, 3], endpoint_ego.shape)
            endpoint = endpoint_ego.astype(np.float64) @ ego[:3, :3].T + ego[:3, 3]
        yield calibration.name, origin, endpoint, ranges, np.column_stack((y, x)), dict(
            positive=int((distance > 0).sum()), nlz_excluded=int(((distance > 0) & ~valid).sum()))


def intersections(origin, unit, length, boxes):
    direction = unit[:, None, :]
    parallel = np.abs(direction) < 1e-12
    low, high = boxes[:, 0], boxes[:, 1]
    first = (low - origin[:, None, :]) / np.where(parallel, 1., direction)
    last = (high - origin[:, None, :]) / np.where(parallel, 1., direction)
    enter = np.maximum(np.where(parallel, -np.inf, np.minimum(first, last)).max(2), 0.)
    leave = np.minimum(np.where(parallel, np.inf, np.maximum(first, last)).min(2), length[:, None])
    outside = (parallel & ((origin[:, None, :] < low) | (origin[:, None, :] > high))).any(2)
    return (leave > enter) & ~outside & (high > low).all(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--raw-ids', type=raw_ids, default=tuple(range(199)))
    parser.add_argument('--runtime-limit-seconds', type=float, default=600.)
    args = parser.parse_args()
    assert args.raw_ids and min(args.raw_ids) >= 0 and max(args.raw_ids) <= 198
    assert args.output.is_dir() and not any(args.output.iterdir()), 'Output must be an existing empty run directory'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(READER))
    import waymo_numpy_cpu as decoder
    from fast_range_matrix import matrix
    decoder.matrix = matrix
    pb = decoder.protobuf()
    with np.load(args.campaign / 'scene008_a6_volume_carve_20261105/volume_accepted.npz', allow_pickle=False) as data:
        boxes = data['boxes'].astype(np.float64)
    with np.load(args.campaign / 'scene008_instances_20261105_r1/observations/a6.npz', allow_pickle=False) as data:
        half = data['box_lwh_m'].astype(np.float64) / 2
    assert boxes.shape == (36, 2, 3) and (boxes[:, 1] > boxes[:, 0]).all()
    zcut = float(boxes[:, 1, 2].max())
    narrow = boxes.copy()
    narrow[:, 0] = np.maximum(narrow[:, 0], -half)
    narrow[:, 1] = np.minimum(narrow[:, 1], [0., half[1], zcut])
    union = np.array([[boxes[:, 0].min(0), boxes[:, 1].max(0)]])
    witnesses = args.output / 'witnesses'
    witnesses.mkdir()
    rows, checks = [], []
    start = time.monotonic()
    status = 'PASS'
    for raw, payload, _ in decoder.records(RECORD, max(args.raw_ids)):
        if raw not in args.raw_ids:
            continue
        frame = pb.Frame.FromString(payload)
        assert frame.context.name == SCENE
        labels = [label for label in frame.laser_labels if label.id == TRACK]
        assert len(labels) == 1, f'raw{raw}: a6 annotated pose missing or duplicated'
        box = labels[0].box
        c, s = np.cos(box.heading), np.sin(box.heading)
        local_to_ego = np.eye(4)
        local_to_ego[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        local_to_ego[:3, 3] = [box.center_x, box.center_y, box.center_z]
        ego = np.asarray(frame.pose.transform).reshape(4, 4)
        local_from_world = np.linalg.inv(ego @ local_to_ego)
        world_to_ego = np.linalg.inv(ego)
        chunks, endpoints, counts = [], [], []
        range_error, total, nlz = 0., 0, 0
        for laser, origin, endpoint, ranges, pixel, info in segments(frame, decoder, matrix, pb):
            delta = endpoint - origin
            length = np.linalg.norm(delta, axis=1)
            range_error = max(range_error, float(np.max(np.abs(length - ranges), initial=0.)))
            assert range_error < .003, (raw, laser, range_error)
            if raw in PILOTS or not rows:
                endpoints.append(endpoint @ world_to_ego[:3, :3].T + world_to_ego[:3, 3])
            local_origin = origin @ local_from_world[:3, :3].T + local_from_world[:3, 3]
            unit = (delta / length[:, None]) @ local_from_world[:3, :3].T
            free_length = np.maximum(length - MARGIN_M, 0.)
            coarse = intersections(local_origin, unit, free_length, union)[:, 0]
            hit = intersections(local_origin[coarse], unit[coarse], free_length[coarse], boxes)
            clipped = intersections(local_origin[coarse], unit[coarse], free_length[coarse], narrow)
            retain = hit.any(1)
            counts.append((hit.sum(0), clipped.sum(0)))
            chunks.append(dict(origin_local=local_origin[coarse][retain], unit_dir=unit[coarse][retain],
                               max_free_distance=free_length[coarse][retain], first_return_range_m=ranges[coarse][retain],
                               raw=np.full(retain.sum(), raw, np.int16), laser=np.full(retain.sum(), laser, np.int8),
                               range_pixel=pixel[coarse][retain].astype(np.int32),
                               cell_hit=hit[retain], narrow_cell_hit=clipped[retain]))
            total += len(ranges)
            nlz += info['nlz_excluded']
        check = dict(raw=raw, range_maxabs_m=range_error)
        if endpoints:
            reference, details = decoder.lidar(frame, pb)
            first_count = sum(row['retained_points'] for row in details if row['return_index'] == 0)
            actual = np.concatenate(endpoints)
            assert actual.shape == (first_count, 3)
            error = float(np.max(np.abs(actual - reference[:first_count, :3])))
            assert error < .01, (raw, error)
            check['original_first_return_endpoint_maxabs_m'] = error
        checks.append(check)
        np.savez_compressed(witnesses / f'{raw:03d}.npz', **{
            key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]})
        full, clipped = np.array(counts).sum(0)
        rows.append(dict(raw=raw, split='heldout' if raw in HELDOUT else 'build', first_returns=total,
                         nlz_excluded=nlz, cell_rays=full.tolist(), narrow_cell_rays=clipped.tolist(),
                         witness_rays=sum(len(chunk['raw']) for chunk in chunks)))
        elapsed = time.monotonic() - start
        print(json.dumps(dict(raw=raw, processed=len(rows), seconds=elapsed, witness_rays=rows[-1]['witness_rays'])), flush=True)
        if len(rows) == 10 and elapsed / len(rows) * 199 > args.runtime_limit_seconds:
            status = 'SMOKE_RUNTIME_LIMIT'
            break
    pools = {}
    for split in ('all', 'build', 'heldout'):
        selected = [row for row in rows if split == 'all' or row['split'] == split]
        pools[split] = dict(raw_ids=[row['raw'] for row in selected])
        for key in ('cell_rays', 'narrow_cell_rays'):
            values = np.array([row[key] for row in selected], dtype=np.int64).reshape(-1, 36)
            pools[split][key] = dict(total_rays=values.sum(0).tolist(), distinct_raw_frames=(values > 0).sum(0).tolist())
    elapsed = time.monotonic() - start
    report = dict(status=status, requested_raw_ids=list(args.raw_ids), rows=rows, pooled=pools, validation=checks,
                  elapsed_seconds=elapsed, prospective_full199_seconds=elapsed / len(rows) * 199,
                  boxes_local_m=boxes.tolist(), narrow_boxes_local_m=narrow.tolist(), zcut_m=zcut,
                  heldout=list(HELDOUT), first_return_only=True, margin_m=MARGIN_M,
                  semantics='Positive-range first returns with NLZ != 1. Free segment ends 15 cm before the measured return.',
                  limits=['A witnessed line segment does not certify a whole cell empty.',
                          'TOP ego motion uses per-pixel pose. Actor pose is frame-level, so within-scan actor motion remains uncertain.',
                          'Side laser pose follows the frame ego pose used by the original decoder.',
                          'No-return and NLZ pixels supply no free-space evidence.',
                          'Evidence maps must use pooled.build and witness rows outside the heldout set.',
                          'Fixed volume cells and median box bounds were inherited from earlier full-data processing.'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

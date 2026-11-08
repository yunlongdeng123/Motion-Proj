"""CPU 验收真实观测、隔离留出帧，并发布原生加载时间表；不生成掩码。"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    args = ap.parse_args()
    bank = args.bank
    receipt, selection = read(bank/'receipt.json'), read(args.selection)
    assert receipt['status'] == 'PASS_CPU_OBSERVATIONS'
    assert receipt['context_id'] == selection['context_id']
    assert receipt['GPU_queries'] == receipt['training_updates'] == receipt['render_calls'] == 0
    expected, held = set(selection['build']), set(selection['holdout'])
    assert not expected & held
    assert expected | held == set(range(selection['source_frame_count']))
    assert expected == set(receipt['build_raws'])
    assert held == set(receipt['holdout_raws']) == set(receipt['skipped_holdout'])
    folders = [('calib', '*.txt', True), ('pose', '*.txt', True), ('timestamp', '*.txt', True),
               ('velodyne', '*.bin', True), ('image/image_0', '*.png', True),
               ('observations', '*.json', False), ('prompts', '*.json', False),
               ('rgb_preview', '*.jpg', False)]
    for folder, pattern, suffix in folders:
        actual = {int(p.stem[-3:] if suffix else p.stem) for p in (bank/folder).glob(pattern)}
        assert actual == expected, (folder, actual ^ expected)
    registry = read(bank/'track_registry.json')['actors']
    assert registry[selection['target']['track_id']]['alias'] == selection['target']['alias']
    ids = [int(a['alias'][1:])+1 for a in registry.values()]
    assert len(set(ids)) == len(ids) and max(ids) < 256
    rows = sorted(receipt['frames'], key=lambda r: r['raw'])
    assert [r['raw'] for r in rows] == sorted(expected)
    hashes = points = prompts = 0
    min_returns = None
    for row in rows:
        raw, name = row['raw'], row['id']
        assert name == receipt['native_file_prefix'] + f'{raw:03d}'
        for relative, identity in row['files'].items():
            p = bank/relative
            assert p.stat().st_size == identity['bytes'] and digest(p) == identity['sha256']
            hashes += 1
        cloud = np.fromfile(bank/'velodyne'/f'{name}.bin', dtype='<f4').reshape(-1, 6)
        assert len(cloud) == row['lidar_points'] > 0 and np.isfinite(cloud).all()
        points += len(cloud)
        min_returns = min(min_returns or row['sensor_return_count'], row['sensor_return_count'])
        obs = read(bank/'observations'/f'{raw:03d}.json')
        assert obs['raw'] == raw and obs['timestamp_micros'] == row['timestamp_micros']
        with Image.open(bank/'image/image_0'/f'{name}.png') as im:
            im.verify()
        with Image.open(bank/'image/image_0'/f'{name}.png') as im:
            assert list(im.size) == obs['camera_size'] == [1920, 1280]
        pose = np.loadtxt(bank/'pose'/f'{name}.txt')
        assert np.allclose(pose, np.array(obs['ego_world']).reshape(4, 4), atol=1e-9)
        assert np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-6)
        cal = {s.split(':')[0]: np.fromstring(s.split(':')[1], sep=' ')
               for s in (bank/'calib'/f'{name}.txt').read_text().splitlines()}
        k = cal['P0'].reshape(3, 4)
        fx, fy, cx, cy = obs['camera_intrinsic_Brown'][:4]
        assert np.allclose(k, [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], atol=1e-8)
        assert np.isfinite(cal['Tr_velo_to_cam_0']).all()
        suggestions = read(bank/'prompts'/f'{raw:03d}.json')
        assert len(suggestions) == row['prompt_count']
        assert len({p['track_id'] for p in suggestions}) == len(suggestions)
        for p in suggestions:
            assert registry[p['track_id']]['alias'] == p['alias']
            x0, y0, x1, y1 = p['bbox_xyxy_native']
            assert 0 <= x0 < x1 < 1920 and 0 <= y0 < y1 < 1280
        prompts += len(suggestions)
    times = np.array([r['model_time'] for r in rows])
    assert np.allclose(np.diff(times), np.diff([r['raw'] for r in rows])*.02)
    timeline = dict(schema='waymo.explicit_timeline.v1', context_id=receipt['context_id'],
                    model_frame_interval=.02, holdout_raws=sorted(held),
                    frames=[{k: row[k] for k in ('id', 'raw', 'timestamp_micros', 'model_time')} for row in rows])
    write(bank/'explicit_timeline.json', timeline)
    write(bank/'preparation_manifest.json', dict(status='awaiting_instance_and_sky_masks',
        segment_context_name=receipt['context_id'], raw_path=receipt['raw'], raw_size_bytes=receipt['raw_bytes'],
        camera='FRONT', camera_index=0, requested_frames=len(rows), source_frame_count=selection['source_frame_count'],
        model_frame_interval=.02, sky_masks_present=0, excluded_holdout_raws=sorted(held),
        frames=[dict(id=r['id'], source_index=r['raw'], timestamp_micros=r['timestamp_micros'],
                     model_time=r['model_time'], lidar_points=r['lidar_points'], image_size=[1920,1280]) for r in rows],
        limitations=receipt['limitations']))
    report = dict(status='PASS_CPU_BANK', context_id=receipt['context_id'], build_frames=len(expected),
        heldout_frames=len(held), measured_points=points, prompts=prompts, hashes_checked=hashes,
        min_sensor_return_groups=min_returns, finite_lidar=True, calib_K_matches_original=True,
        target=selection['target'], original_track_count=len(ids), model_time_gap_preserved=True,
        holdout_RGB_and_LiDAR_absent_from_bank=True, masks_generated=False, training_ready=False,
        GPU_queries=0, training_updates=0, render_calls=0, human_verdict=None)
    write(bank/'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

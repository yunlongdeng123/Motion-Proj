"""从已保存的C00 FP32评价导出离线查看素材；不导入torch或重渲染。"""
import argparse
import datetime
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image


def rgb(array):
    if array.shape != (3, 320, 480) or not np.isfinite(array).all():
        raise ValueError('RGB维数或finite合同失败')
    return np.rint(array.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def gray(array):
    array = array.squeeze()
    if array.shape != (320, 480) or not np.isfinite(array).all():
        raise ValueError('alpha/mask维数或finite合同失败')
    return np.rint(array.clip(0, 1) * 255).astype(np.uint8)


def bbox(mask):
    yy, xx = np.where(mask.squeeze())
    if not len(xx):
        return [0, 0, 480, 320]
    cx, cy = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
    width = min(480, max(140, int(xx.max() - xx.min() + 1) + 40))
    height = min(320, max(94, int(yy.max() - yy.min() + 1) + 40))
    x0 = max(0, min(480 - width, int(round(cx - width / 2))))
    y0 = max(0, min(320 - height, int(round(cy - height / 2))))
    return [x0, y0, x0 + width, y0 + height]


def dump(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def export(source, destination):
    event = json.loads((source / 'checkpoint_events/iteration_030000.json').read_text())
    if event['status'] != 'complete' or event['iteration'] != 30000 or len(event['frames']) != 49:
        raise ValueError('C00完整30k与49帧事件缺失')
    if destination.exists():
        raise FileExistsError('拒绝覆盖已有查看素材: ' + str(destination))
    destination.mkdir(parents=True)
    assets = destination / 'assets'
    observed, control = assets / 'observed', assets / 'static_time'
    observed.mkdir(parents=True)
    control.mkdir()
    frames = []
    rawroot = source / 'eval/train_30000_render/hybrid_raw'
    rows = sorted(event['frames'], key=lambda row: row['timestamp'])
    for index, row in enumerate(rows):
        name = row['frame_id'].replace('/', '_')
        metrics = json.loads((rawroot / (name + '.json')).read_text())
        with np.load(rawroot / (name + '.npz'), allow_pickle=False) as arrays:
            images = {}
            for key, raw_key in [('gt', 'gt'), ('composite', 'composite'), ('static', 'static'),
                                 ('dynamic', 'dynamic'), ('alpha_s', 'alpha_s'),
                                 ('alpha_d', 'alpha_d'), ('body_mask', 'actor_mask')]:
                filename = '%03d_%s.png' % (index, key)
                converted = rgb(arrays[raw_key]) if key in ('gt', 'composite', 'static', 'dynamic') else gray(arrays[raw_key])
                Image.fromarray(converted).save(observed / filename, compress_level=4)
                images[key] = 'assets/observed/' + filename
            rectangle = bbox(arrays['actor_mask'])
        frames.append({'id': row['frame_id'], 'raw_index': index, 'timestamp': row['timestamp'],
                       'images': images, 'bbox': rectangle, 'metrics': metrics['regions']})
    time_summary = json.loads((source / 'static_time_30000/summary.json').read_text())
    if len(time_summary['rows']) != 49 or time_summary['anchor_pose_frame'] != '0017010':
        raise ValueError('固定pose静态时间合同失败')
    static_frames = []
    for index, row in enumerate(time_summary['rows']):
        with np.load(source / 'static_time_30000' / ('%03d.npz' % index), allow_pickle=False) as arrays:
            images = {}
            for key, raw_key in [('static', 'static_rgb'), ('alpha_s', 'static_alpha')]:
                filename = '%03d_%s.png' % (index, key)
                Image.fromarray(rgb(arrays[raw_key]) if key == 'static' else gray(arrays[raw_key])).save(control / filename, compress_level=4)
                images[key] = 'assets/static_time/' + filename
        static_frames.append({'id': row['source_timestamp_frame'], 'raw_index': index,
                              'timestamp': row['timestamp_model'], 'images': images,
                              'bbox': frames[10]['bbox'], 'metrics': row})
    memory = json.loads((source / 'memory_summary.json').read_text())
    status = json.loads((source / 'status.json').read_text())
    def mean(region, key):
        return float(np.mean([frame['metrics'][region][key] for frame in frames]))
    summary = {'training_steps': 30000, 'wall_seconds': status['wall_seconds'],
               'wall_minutes': status['wall_seconds'] / 60,
               'device_peak_GiB': memory['nvidia_smi']['device_peak_used_bytes'] / 1024**3,
               'peak_vram_gib': memory['nvidia_smi']['device_peak_used_bytes'] / 1024**3,
               'reserved_peak_GiB': memory['pytorch']['peak_reserved_bytes'] / 1024**3,
               'static_points': memory['extra']['render_static_points'],
               'dynamic_points': memory['extra']['render_dynamic_points'],
               'full_composite_psnr_db': mean('full', 'composite_psnr_db'),
               'actor_composite_psnr_db': mean('actor', 'composite_psnr_db'),
               'background_static_psnr_db': mean('non_sky_outside_actor', 'static_psnr_db'),
               'outside_dynamic_alpha_mean': mean('non_sky_outside_actor', 'alpha_d_mean')}
    manifest = {'title': 'DrivingEditor · C00 / 30k', 'arm': 'C00', 'checkpoint': 30000,
                'scene': 'scene0017_000_048', 'frame_count': 49, 'width': 480, 'height': 320,
                'generated_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                'summary': summary, 'frames': frames, 'static_time_frames': static_frames,
                'static_time_summary': time_summary,
                'source': {'model_path': str(source), 'event': str(source / 'checkpoint_events/iteration_030000.json'),
                           'task_run': 'WS-V82-DRIVINGEDITOR-HYBRID30K-20261012/r1',
                           'RGB_display': 'stored FP32 clipped [0,1], rounded 8bit; no per-image normalization or body premask',
                           'alpha_display': 'raw unmasked alpha clipped [0,1], black0 white1; no per-image normalization',
                           'metrics': '49-frame equal means; clamped RGB errors from FP32, raw alpha statistics',
                           'static_time': 'fixed pose0017010, 49 model timestamps; counterfactual, no GT quality score',
                           'scope': 'all-observed development reconstruction, body-only mask without shadows',
                           'GPU_used_by_export': False, 'checkpoint_loaded': False}}
    dump(destination / 'manifest.json', manifest)
    dump(destination / 'summary.json', summary)
    with tarfile.open(destination.with_suffix('.tar.gz'), 'w:gz', compresslevel=4) as archive:
        for child in sorted(destination.iterdir()):
            archive.add(child, arcname=child.name)
    print(json.dumps({'status': 'EXPORTED_FROM_SAVED_ARRAYS_CPU_ONLY', 'destination': str(destination),
                      'archive': str(destination.with_suffix('.tar.gz')), 'frames': len(frames),
                      'static_time_frames': len(static_frames), 'images': 49 * 9, 'summary': summary}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == '__main__':
    main()

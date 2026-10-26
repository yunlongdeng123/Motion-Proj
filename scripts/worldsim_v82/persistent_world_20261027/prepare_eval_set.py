"""版本化场景清单：真实TFRecord逐帧CRC/标注盘点，先划场景，再生成案例。

只做评测数据准备，不运行/选择修复算法。数据与结果留在campaign，不进入Git。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import concurrent.futures as futures
import ctypes
import hashlib
import io
import json
import struct
import time
from pathlib import Path
import numpy as np
from PIL import Image
from waymo_open_dataset import dataset_pb2

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
DEST = ROOT / 'evaluation_set_v1'
KNOWN = {'10094743350625019937_3420_000_3440_000': 'previous_method_development',
         '10235335145367115211_5420_000_5440_000': 'previous_reconstruction_development',
         '10526338824408452410_5714_660_5734_660': 'previous_reconstruction_development'}
FAMILIES = ['noop_fidelity', 'delete_and_reveal', 'known_road_withdrawal',
            'move_and_insert', 'ego_view_change', 'repeated_world_edit']


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def masked_crc(crc):
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def frames(path):
    lib = ctypes.CDLL('/root/autodl-tmp/drivingeditor_repro/campaign/scene_search_mirror/libpreview_crc32c.so')
    crc = lib.preview_crc32c
    crc.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    crc.restype = ctypes.c_uint32
    with path.open('rb') as stream:
        index = 0
        while True:
            offset = stream.tell()
            header = stream.read(12)
            if not header:
                break
            assert len(header) == 12
            size, checksum = struct.unpack('<QI', header)
            assert size < 64 * 1024 * 1024 and masked_crc(crc(header[:8], 8)) == checksum
            payload = stream.read(size)
            footer = stream.read(4)
            assert len(payload) == size and len(footer) == 4
            assert masked_crc(crc(payload, size)) == struct.unpack('<I', footer)[0]
            yield index, offset, dataset_pb2.Frame.FromString(payload)
            index += 1


def scene_worker(path_string):
    path = Path(path_string)
    key = path.name[len('segment-'):-len('_with_camera_labels.tfrecord')]
    folder = DEST / 'scenes' / key
    output = folder / 'scene.json'
    if output.exists():
        data = json.loads(output.read_text())
        assert data['raw_size_bytes'] == path.stat().st_size and data['crc_all_records_verified']
        return data
    started = time.time()
    rows, tracks, offsets = [], {}, []
    context = None
    for index, offset, frame in frames(path):
        assert frame.context.name == key
        context = frame.context
        pose = np.array(frame.pose.transform).reshape(4, 4)
        front = next((x for x in frame.projected_lidar_labels if x.name == 1), None)
        laser = {x.id: x for x in frame.laser_labels}
        objects = []
        if front:
            for label in front.labels:
                if label.type != 1:
                    continue
                track = label.id[:-6] if label.id.endswith('_FRONT') else label.id
                box = label.box
                vehicle = laser.get(track)
                entry = dict(track=track, xywh=[box.center_x, box.center_y, box.length, box.width],
                             area=float(box.length * box.width))
                if vehicle is not None:
                    b = vehicle.box
                    center = pose @ np.array([b.center_x, b.center_y, b.center_z, 1.])
                    entry.update(center_world=center[:3].tolist(),
                                 size_m=[b.length, b.width, b.height],
                                 heading_vehicle=b.heading,
                                 speed_mps=float(np.hypot(vehicle.metadata.speed_x, vehicle.metadata.speed_y)))
                objects.append(entry)
                tracks.setdefault(track, []).append(dict(raw=index, **entry))
        rows.append(dict(raw=index, timestamp_micros=frame.timestamp_micros,
                         vehicle_position_world=pose[:3, 3].tolist(), front_vehicles=objects))
        offsets.append(offset)
    assert len(rows) >= 49 and all(b['timestamp_micros'] > a['timestamp_micros'] for a, b in zip(rows, rows[1:]))
    # 输入标注决定窗口；不看任何编辑模型的输出/误差，避免挑选成绩好的案例。
    candidates = [(np.median([r['area'] for r in sequence]) * min(len(sequence), 49), track)
                  for track, sequence in tracks.items() if len(sequence) >= 25]
    target = max(candidates)[1] if candidates else None
    peak = max(tracks[target], key=lambda x: x['area'])['raw'] if target else len(rows) // 2
    start = max(0, min(len(rows) - 49, peak - 24))
    window = list(range(start, start + 49))
    split = KNOWN.get(key)
    if split is None:
        split = 'heldout_reserved' if int(hashlib.sha256(('eval-v1:' + key).encode()).hexdigest()[:8], 16) % 3 else 'new_diagnostic_development'
    folder.mkdir(parents=True, exist_ok=True)
    # 原始RGB统一编码为PNG；独立保留GT，绝不把洞内GT自动喂给补全。
    with path.open('rb') as stream:
        for raw in window:
            stream.seek(offsets[raw])
            size = struct.unpack('<Q', stream.read(8))[0]
            stream.read(4)
            frame = dataset_pb2.Frame.FromString(stream.read(size))
            image = next(x for x in frame.images if x.name == 1)
            rgb = Image.open(io.BytesIO(image.image)).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS)
            (folder / 'gt_front').mkdir(exist_ok=True)
            rgb.save(folder / 'gt_front' / f'{raw:03d}.png')
    positions = np.array([r['vehicle_position_world'] for r in rows])
    distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    data = dict(schema_version=1, context=key, split=split, raw_path=str(path),
                raw_size_bytes=path.stat().st_size, raw_mtime_ns=path.stat().st_mtime_ns,
                frames=len(rows), crc_all_records_verified=True, front_png_frames=len(window),
                location=context.stats.location, time_of_day=context.stats.time_of_day,
                weather=context.stats.weather, ego_travel_m=distance,
                selected_track=target, selected_window=window,
                selection_policy='long-lived FRONT vehicle; median area times capped duration; window around largest annotation; no model outputs',
                physical_route_overlap_audited=False,
                heldout_status='reserved; not yet evaluated by editing algorithms' if split == 'heldout_reserved' else 'development',
                semantic_pixel_masks='pending; projected boxes are not pixel masks',
                road_surface_gt='pending; no inferred plane is called dense GT',
                raw_annotations=rows, seconds=time.time() - started)
    write(output, data)
    return data


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    raw = Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/training')
    paths = list(raw.glob('*.tfrecord'))
    extra = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/inputs/raw_scene0029')
    paths += list(extra.glob('*.tfrecord'))
    unique = {p.name: p for p in paths}
    write(DEST / 'status.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                                   phase='raw_crc_annotations_png_preparation', expected_scenes=len(unique)))
    scenes, errors = [], []
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(scene_worker, str(p)): p for p in unique.values()}
        for future in futures.as_completed(jobs):
            try:
                scene = future.result()
                scenes.append({k: v for k, v in scene.items() if k != 'raw_annotations'})
                print('SCENE', scene['context'], scene['split'], scene['frames'], flush=True)
            except Exception as error:
                errors.append(dict(raw=str(jobs[future]), error=repr(error)))
            write(DEST / 'status.json', dict(phase='preparing', completed=len(scenes), errors=errors, expected_scenes=len(unique)))
    assert not errors, errors
    scenes.sort(key=lambda x: x['context'])
    cases = []
    for scene in scenes:
        for family in FAMILIES:
            cases.append(dict(case_id=scene['context'] + '/' + family, scene=scene['context'], split=scene['split'], family=family,
                              frames=scene['selected_window'], target_track=scene['selected_track'],
                              status='specification_prepared_not_executed',
                              required_inputs=['calibrated_native_checkpoint', 'pixel_masks', 'geometry_and_reference_domains'],
                              heldout_locked=scene['split'] == 'heldout_reserved',
                              withdrawal_scope='all-source before initialization/encoding required; module-only results reported separately' if family == 'known_road_withdrawal' else None,
                              quality_accepted=False, human_verdict=None))
    manifest = dict(schema_version=1, name='4D editing evaluation v1', seed=0,
                    protocol='EVALUATION_PROTOCOL.md', scenes=scenes, cases=cases,
                    counts=dict(raw_contexts=len(scenes), prepared_pngs=sum(s['front_png_frames'] for s in scenes),
                                case_specifications=len(cases), executed_cases=0,
                                splits={split: sum(s['split'] == split for s in scenes) for split in sorted(set(s['split'] for s in scenes))}),
                    validity_limits=['unique raw contexts; physical-road overlap not yet audited',
                                     'RGB/annotation-ready is not Gaussian/edit-ready',
                                     'no source-matching MAE labeled independent road truth',
                                     'no video-only image promoted into 3D observations'])
    write(DEST / 'manifest.json', manifest)
    write(DEST / 'status.json', dict(phase='raw_metadata_and_png_prepared', **manifest['counts']))


if __name__ == '__main__':
    main()

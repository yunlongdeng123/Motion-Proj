"""固定r46查询窗口，扩展真实例的邻时刻/多相机证据；不读取生成输出选输入。"""
from common import *
from collections import defaultdict
import subprocess, glob
import numpy as np
from prepare import rows, annotation_view, camera_actors
from geometry_factory import transform


def main():
    if (O/'manifest.json').exists():
        print('ALREADY_PREPARED'); return
    O.mkdir(parents=True, exist_ok=True)
    parent = read(T/'r46/manifest.json')
    cases = parent['cases']
    oldreal = {c['eval_id']: c for c in read(T/'r21/real_input_plan.json')['cases']}
    expected = {c['clip_id']: c for c in read(A/'expected_sources.json')['clips']}
    meta = A/'metadata/v1.0-trainval'
    sample = {r['token']: r for r in read(meta/'sample.json')}
    sensor = {r['token']: r for r in read(meta/'sensor.json')}
    cal = {r['token']: r for r in read(meta/'calibrated_sensor.json')}
    scenes = {r['name']: r['token'] for r in read(meta/'scene.json')}
    primary, keyframes, needed = {}, {}, set()
    for c in cases:
        if c['kind'] != 'real': continue
        src = expected[oldreal[c['case_id']]['clip_id']]
        primary[c['case_id']] = {f['sample_data_token']: f['frame'] for f in src['frames']}
        times = [c['frames'][0]['timestamp']-1000000, c['frames'][0]['timestamp'],
                 c['frames'][-1]['timestamp'], c['frames'][-1]['timestamp']+1000000]
        pool = [s for s in sample.values() if s['scene_token']==scenes[c['scene']]]
        tokens = {min(pool, key=lambda s: abs(s['timestamp']-t))['token'] for t in times}
        keyframes[c['case_id']] = tokens
        needed |= tokens
    camera_tokens = set().union(*(set(v) for v in primary.values()))
    sd, scans = {}, []
    for d in rows(meta/'sample_data.json'):
        channel = sensor[cal[d['calibrated_sensor_token']]['sensor_token']]['channel']
        if d['token'] in camera_tokens or (d['sample_token'] in needed and d['is_key_frame'] and channel.startswith('CAM_')):
            sd[d['token']] = d | {'camera': channel}
        if d['sample_token'] in needed and d['is_key_frame'] and channel=='LIDAR_TOP': scans.append(d)
    needed |= {d['sample_token'] for d in sd.values()}
    needed |= {sample[t]['prev'] for t in list(needed) if sample[t]['prev']}
    categories = {r['token']: r['name'] for r in read(meta/'category.json')}
    instances = {r['token']: categories[r['category_token']] for r in read(meta/'instance.json')}
    ann = defaultdict(list)
    for a in rows(meta/'sample_annotation.json'):
        if a['sample_token'] in needed:
            ann[a['sample_token']].append(a | {'category': instances[a['instance_token']],
                'category_name': instances[a['instance_token']], 'timestamp': sample[a['sample_token']]['timestamp']})
    ego_needed = {d['ego_pose_token'] for d in list(sd.values())+scans}
    ego = {e['token']: e for e in rows(meta/'ego_pose.json') if e['token'] in ego_needed}
    sdk = annotation_view(sample, sd, ann)
    # 文件索引只保存位置；不展开公共数据集，不重复制已有RGB。
    inventory = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', '*.jpg', '-g', '*.bin',
        '/root/autodl-tmp/data', str(T), str(A)], capture_output=True, text=True, check=True).stdout.splitlines()
    byname = defaultdict(list)
    for p in inventory: byname[Path(p).name].append(p)
    byprefix = defaultdict(set)
    for pattern in ('/root/autodl-tmp/data/worldsim_v67/*member_shards*.json',
                    '/root/autodl-tmp/data/worldsim_v5/manifests/*member_shards*.json',
                    '/root/autodl-tmp/data/dynamic_editing_v2/manifests/*member_shards*.json'):
        for file in glob.glob(pattern):
            for name, shard in read(file).items():
                shard = str(shard)
                shard = f'{int(shard):02}' if shard.isdigit() else shard.split('trainval', 1)[1][:2]
                byprefix[Path(name).name.split('__', 1)[0]].add(shard)
    wanted = {}
    for c in cases:
        if c['kind']=='synthetic':
            c['references'] = [{'source_frame': i, 'camera': 'original_training_camera',
                'frame': c['frames'][i], 'path': None, 'mask_path': None, 'in_window': True,
                'source_kind': 'existing_masked_X', 'available': True} for i in [0,3,6,9]]
            continue
        refs = []
        for d in sd.values():
            if d['token'] not in primary[c['case_id']] and d['sample_token'] not in keyframes[c['case_id']]: continue
            cs, ep = cal[d['calibrated_sensor_token']], ego[d['ego_pose_token']]
            K = np.array(cs['camera_intrinsic']); K[0] *= 1024/d['width']; K[1] *= 576/d['height']
            frame = {'timestamp': d['timestamp'], 'sample_data_token': d['token'],
                'sample_token': d['sample_token'], 'is_key_frame': d['is_key_frame'],
                'camera_to_world': (transform(ep['translation'],ep['rotation'])@transform(cs['translation'],cs['rotation'])).tolist(),
                'ego_to_world': transform(ep['translation'],ep['rotation']).tolist(),
                'intrinsics_1024': K.tolist(), 'actors': camera_actors(sdk,d)}
            index = primary[c['case_id']].get(d['token'])
            if index is not None:
                rgb = sorted((Path(c['folder'])/'rgb').glob('*.jpg'))[index]
                hole = sorted((Path(c['folder'])/'model_mask').glob('*.png'))[index]
                path, mask_path, kind = str(rgb), str(hole), 'r21_full_SAM'
            else:
                places = byname.get(Path(d['filename']).name, [])
                path = places[0] if places else None
                mask_path, kind = None, 'GT_target_envelope_source_exclusion'
                if not path:
                    wanted[d['filename']] = {'filename': d['filename'], 'kind': 'RGB',
                        'shards': sorted(byprefix[Path(d['filename']).name.split('__',1)[0]])}
            refs.append({'source_frame': index, 'camera': d['camera'], 'frame': frame,
                'path': path, 'mask_path': mask_path, 'source_kind': kind, 'filename': d['filename'],
                'available': path is not None, 'target_excluded_before_encoding': True,
                'auxiliary_mask_requires_GPU_validation': index is None})
        c['references'] = sorted(refs, key=lambda r: (r['frame']['timestamp'],r['camera']))
        c['bev_scans'] = []
        for d in scans:
            if d['sample_token'] not in keyframes[c['case_id']]: continue
            places = byname.get(Path(d['filename']).name, [])
            path = places[0] if places else None
            if not path:
                wanted[d['filename']] = {'filename': d['filename'], 'kind': 'LiDAR',
                    'shards': sorted(byprefix[Path(d['filename']).name.split('__',1)[0]])}
            c['bev_scans'].append(d | {'path': path, 'ego_pose': ego[d['ego_pose_token']],
                'calibrated_sensor': cal[d['calibrated_sensor_token']], 'actors': ann[d['sample_token']]})
    if any(not w['shards'] for w in wanted.values()):
        raise ValueError('公共分片线索缺失；不能静默跳过新输入')
    plan = {'task_id': parent['task_id'], 'run_id': 'r47', 'parent_run': 'r46', 'cases': cases,
        'baseline': 'original DriveEditor checkpoint + r21 full SAM; no r7/r8/r14/r46 adapter weights',
        'architecture_change': 'new RGB reference cross-attention + BEV encoder and image-space geometry controls; original UNet9 channels unchanged',
        'training_backbone': 'original main/3D UNet and pretrained RGB VAE frozen; new branch only',
        'eval_seed': 42, 'eval_steps': 25, 'train_seed': 6201, 'train_steps': 320,
        'scope': 'existing 12 train / 4 synthetic DEV / 8 real DEV; no new final-test claims',
        'reference_policy': 'main observed target crop + up to6 legal masked references, selected by retained-actor visibility; real raw bank includes26 primary frames and6 cameras at4 key times',
        'geometry_auxiliary': 'official GT poses/tracks + measured LiDAR; boxes are envelopes, not silhouettes/free space',
        'text_policy': 'short deterministic role/pose description for audit; numeric parameters condition branch; no new free-language encoder',
        'arms': ['baseline', 'null_priors', 'RGB_only', 'geometry_only', 'RGB_and_geometry'],
        'failure_ledger_refs': ['V77-F02'], 'GPU_jobs': 0, 'human_verdict': None}
    dump(O/'manifest.json', plan)
    dump(O/'extra_input_plan.json', {'files': list(wanted.values()), 'readers': 1, 'bounded': True,
        'expected_small_output': 'only selected RGB and LiDAR, not whole archive'})
    print(json.dumps({'cases':len(cases),'real_bank':{c['case_id']:len(c['references']) for c in cases if c['kind']=='real'},
        'missing_files':len(wanted),'shards':sorted(set(s for w in wanted.values() for s in w['shards']))}, ensure_ascii=False),flush=True)


if __name__=='__main__': main()

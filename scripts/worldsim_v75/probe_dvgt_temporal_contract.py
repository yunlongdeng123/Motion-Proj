"""两个已有任务的有限2Hz上下文检查；不调参、不生成世界模型视频。"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from audit_dvgt_native_contract import ROOT, REPO, SOURCES, A, save

OUT = ROOT / 'WS-V75-DVGT-TEMPORAL-CONTRACT-01/20260920-r1'


def prepare():
    import pandas as pd
    from prepare_argoverse import ROOT as RAW, poses
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-DVGT-TEMPORAL-CONTRACT-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'sources': [str(s) for s in SOURCES], 'seed': 7501,
        'repository': str(REPO), 'revision': '51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236',
        'weights': '/root/autodl-tmp/models/worldsim_v81/dvgt1.pt',
        'offsets_s': [-1, -.5, 0], 'shape': [1, 3, 7, 3, 512, 512],
        'sampling': 'nearest front-center capture to each requested past anchor; other views latest <= actual anchor; all captures <= original cutoff; final frame identical to saved T1 input',
        'forward': 'official DVGT1, strict full checkpoint, BF16, frames_chunk_size=1, official crop loader',
        'budget': 'two calls maximum, one per existing task; stop queue on any failure or OOM, no retry/downsize',
        'input_roles': '21 real RGB captures only; known calibration and ego poses for evaluation, not forward; no labels or LiDAR fitting; history is extra information, not equal-budget comparison',
        'evaluation': 'last frame, paired common fixed 8px grid; both predictions finite and within 2..80m of cutoff ego, exclude 4px padding margin; report all 7 views, positive camera-depth fraction, median angular error and positive-only reprojection',
        'native_use_gate': 'every camera median angular error <=1 degree AND positive camera-depth fraction >=0.95; screening criterion only, not proof of depth/shape accuracy',
        'stop': 'no further context lengths, seeds, camera orders or source substitutions; do not infer closed-loop harm from projection inconsistency',
        'world_model_generation_calls': 0, 'human_verdict': None,
        'failure_ledger_refs': ['V74-H2-F20', 'V74-H2-F22'], 'failure_ledger_delta': 'none'}
    save(OUT/'protocol.json', protocol)
    sys.path.insert(0, str(REPO))
    from dvgt.utils.load_fn import load_and_preprocess_images
    cases = []
    for source in SOURCES:
        old = json.loads((source/'protocol.json').read_text())
        base = json.loads((Path(old['base_run'])/'input_manifest.json').read_text())
        raw = RAW/old['log_id']; ego = pd.read_feather(raw/'city_SE3_egovehicle.feather')
        folder = OUT/old['log_id']; folder.mkdir()
        front = sorted((raw/'sensors/cameras/ring_front_center').glob('*.jpg'))
        front = [f for f in front if int(f.stem) <= old['cutoff_ns']]
        frames = []
        for t, offset in enumerate(protocol['offsets_s']):
            requested = old['cutoff_ns'] + int(offset*1e9)
            anchor = int(min(front, key=lambda f: abs(int(f.stem)-requested)).stem)
            assert abs(anchor-requested) < 1_000_000
            views = []; inp = folder/f'native_input/frame_{t}'; inp.mkdir(parents=True)
            for i, v in enumerate(old['views']):
                image = max((f for f in (raw/'sensors/cameras'/v['camera']).glob('*.jpg')
                             if int(f.stem) <= anchor), key=lambda f: int(f.stem))
                stamp = int(image.stem); assert 0 <= anchor-stamp < 50_000_000
                E = poses(ego, [stamp])[0]; E[:3, 3] -= np.array(base['city_origin'])
                extrinsic = np.linalg.inv(np.array(v['ego_world'])) @ np.array(v['camera_world'])
                view = copy.deepcopy(v)
                view.update(image=str(image), timestamp_ns=stamp, delta_to_cutoff_ms=(stamp-old['cutoff_ns'])/1e6,
                            delay_to_anchor_ms=(anchor-stamp)/1e6, ego_world=E.tolist(), camera_world=(E@extrinsic).tolist())
                views.append(view); (inp/f'{i:02d}_{v["camera"]}.jpg').symlink_to(image)
            frames.append({'requested_offset_s': offset, 'anchor_timestamp_ns': anchor, 'views': views})
        # 标定元数据与最后一帧必须严格保留旧输入，消除浮点重构差异。
        assert all(v['image'] == q['image'] for v, q in zip(frames[-1]['views'], old['views']))
        frames[-1]['views'] = old['views']
        rgb = load_and_preprocess_images(str(folder/'native_input'), mode='crop')
        assert list(rgb.shape) == protocol['shape']
        last = (rgb[0, -1].permute(0, 2, 3, 1)*255).round().byte().numpy()
        assert np.array_equal(last, np.load(source/'network_rgb.npy'))
        case = {'log_id': old['log_id'], 'source': str(source), 'frames': frames,
                'last_frame_input_pixel_delta': 0, 'human_verdict': None}
        save(folder/'input_manifest.json', case)
        cases.append({'log_id': old['log_id'], 'anchor_offsets_s': [(f['anchor_timestamp_ns']-old['cutoff_ns'])/1e9 for f in frames],
                      'last_frame_input_pixel_delta': 0})
    save(OUT/'preflight.json', {'status': 'complete', 'cases': cases, 'cuda_initialized': torch.cuda.is_initialized()})
    print(json.dumps(cases), flush=True)


def infer(case_index):
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'preflight.json').read_text())['status'] == 'complete'
    source = Path(protocol['sources'][case_index]); old = json.loads((source/'protocol.json').read_text())
    # 第二次调用只能在第一例完整通过后开始；已有终态绝不覆盖。
    if case_index:
        first = json.loads((Path(protocol['sources'][0])/'protocol.json').read_text())['log_id']
        assert json.loads((OUT/first/'inference_result.json').read_text())['status'] == 'complete'
    folder = OUT/old['log_id']; path = folder/'inference_result.json'; assert not path.exists()
    result = {'status': 'started', 'model': 'official DVGT-1', 'human_verdict': None, 'world_model_generation_calls': 0}
    save(path, result); began = time.monotonic()
    try:
        torch.set_num_threads(4); torch.manual_seed(protocol['seed'])
        os.chdir(REPO); sys.path.insert(0, str(REPO))
        from dvgt.models.architectures.dvgt1 import DVGT1
        from dvgt.utils.load_fn import load_and_preprocess_images
        original = torch.hub.load
        def hub(*args, **kwargs):
            if args and 'dinov3' in str(args[0]):
                kwargs['pretrained'] = False; kwargs.pop('weights', None)
            return original(*args, **kwargs)
        torch.hub.load = hub
        try:
            model = DVGT1(dino_v3_weight_path=None, frames_chunk_size=1)
        finally:
            torch.hub.load = original
        state = torch.load(protocol['weights'], map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True); del state
        model = model.eval().cuda()
        inputs = load_and_preprocess_images(str(folder/'native_input'), mode='crop').cuda()
        assert list(inputs.shape) == protocol['shape']
        print(json.dumps({'stage': 'forward', 'log': old['log_id'], 'shape': list(inputs.shape)}), flush=True)
        torch.cuda.reset_peak_memory_stats(); forward_start = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(inputs)
        torch.cuda.synchronize(); forward_s = time.monotonic()-forward_start
        arrays = {k: pred[k].float().cpu().numpy() for k in ['points', 'points_conf', 'absolute_ego_pose_enc']}
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez(folder/'native_outputs.npz', **arrays)
        np.save(folder/'network_rgb.npy', (inputs[0].permute(0, 1, 3, 4, 2)*255).round().byte().cpu().numpy())
        result.update(status='complete', shape=list(inputs.shape), forward_s=forward_s,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      output_shapes={k: list(v.shape) for k, v in arrays.items()}, finite=True,
                      strict_weights=True, official_source_changed=False)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['wall_s'] = time.monotonic()-began; save(path, result); print(json.dumps(result), flush=True)


def evaluate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    protocol = json.loads((OUT/'protocol.json').read_text()); cases = []
    assert not (OUT/'result.json').exists()
    sys.path.insert(0, str(REPO))
    from dvgt.utils.pose_encoding import decode_pose
    yy, xx = np.mgrid[4:512:8, 4:512:8]; pixels = np.stack([xx, yy], -1).reshape(-1, 2)
    for source in SOURCES:
        p = json.loads((source/'protocol.json').read_text()); folder = OUT/p['log_id']
        inf = json.loads((folder/'inference_result.json').read_text()); assert inf['status'] == 'complete'
        m = json.loads((folder/'input_manifest.json').read_text()); z = np.load(folder/'native_outputs.npz')
        native = {'T1': np.load(source/'native_outputs.npz')['points'][0, -1], 'T3': z['points'][0, -1]}
        first = {'T1': np.array(p['views'][0]['ego_world']), 'T3': np.array(m['frames'][0]['views'][0]['ego_world'])}
        last = first['T1']; rows = []; display = {}
        for i, v in enumerate(p['views']):
            K = np.array(v['K_network']); cw = np.array(v['camera_world'])
            rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
            shared = (pixels[:, 1] >= v['pad_top']+4) & (pixels[:, 1] < 508-v['pad_top'])
            computed = {}
            for key in native:
                point = native[key][i, yy, xx].reshape(-1, 3).astype(float)/.1
                E = first[key]; world = point @ A.T @ E[:3, :3].T + E[:3, 3]
                norm = np.linalg.norm(world-last[:3, 3], axis=-1)
                shared &= np.isfinite(point).all(-1) & (norm > 2) & (norm < 80)
                camera = (world-cw[:3, 3]) @ cw[:3, :3]
                project = camera @ K.T; project = project[:, :2]/project[:, 2:]
                cosine = np.einsum('ij,ij->i', rays, camera)/(np.linalg.norm(rays, axis=-1)*np.linalg.norm(camera, axis=-1))
                computed[key] = (camera, project, np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            assert shared.sum() > 0
            row = {'camera': v['camera'], 'common_samples': int(shared.sum())}
            for key, (camera, project, angle) in computed.items():
                positive = shared & (camera[:, 2] > .2)
                err = np.linalg.norm(project-pixels, axis=-1)
                row[key] = {'positive_fraction': float(positive.sum()/shared.sum()),
                            'median_angular_error_deg': float(np.median(angle[shared])),
                            'median_positive_reprojection_px': float(np.median(err[positive])) if positive.any() else None,
                            'p90_positive_reprojection_px': float(np.percentile(err[positive], 90)) if positive.any() else None}
                display[f'{i}_{key}_angles'] = np.where(shared, angle, np.nan).reshape(64, 64)
            rows.append(row)
        pred_pose, _ = decode_pose(torch.from_numpy(z['absolute_ego_pose_enc'])); pred_pose = pred_pose.numpy()[0]
        affine = np.eye(4); affine[:3, :3] = A; origin = first['T3'] @ affine; pose_rows = []
        for t, frame in enumerate(m['frames']):
            truth = np.linalg.inv(origin) @ np.array(frame['views'][0]['ego_world']) @ affine
            pred = pred_pose[t].copy(); pred[:3, 3] /= .1
            pose_rows.append({'t': t, 'reference_translation_m': truth[:3, 3].tolist(), 'predicted_translation_m': pred[:3, 3].tolist(),
                             'translation_error_m': float(np.linalg.norm(pred[:3, 3]-truth[:3, 3])),
                             'rotation_error_deg': float(np.degrees(Rotation.from_matrix(pred[:3, :3] @ truth[:3, :3].T).magnitude()))})
        gates = {key: all(row[key]['median_angular_error_deg'] <= 1 and row[key]['positive_fraction'] >= .95 for row in rows) for key in native}
        np.savez(folder/'projection_display.npz', **display)
        case = {'log_id': p['log_id'], 'source': str(source), 'projection': rows, 'pose': pose_rows,
                'native_use_gate': gates, 'inference': inf, 'last_frame_input_pixel_delta': 0, 'human_verdict': None}
        save(folder/'evaluation.json', case); cases.append(case)
        print(json.dumps({'log': p['log_id'], 'gate': gates, 'projection': rows, 'pose': pose_rows}), flush=True)
    save(OUT/'result.json', {'status': 'complete', 'cases': cases, 'model_calls': 2, 'world_model_generation_calls': 0,
                           'failure_ledger_delta': 'none', 'human_verdict': None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'infer', 'evaluate'])
    parser.add_argument('--case-index', type=int, choices=[0, 1]); args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'infer':
        assert args.case_index is not None
        infer(args.case_index)
    else: evaluate()

"""Prepared read-only diagnostic. Default prints a plan; no Torch or GPU import.

Run only after a new human instruction that the GPU host is open. No training,
checkpoint writing, persistent point deletion, background box pruning or queue.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default=str(Path(__file__).with_name('experiment_plan.json')))
    p.add_argument('--cpu-attribution')
    p.add_argument('--output')
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--user-opened-card', action='store_true')
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    if not args.execute_gpu:
        print(json.dumps({'status': 'WAITING_FOR_USER_TO_OPEN_CARD', 'GPU_started': False,
                          'training_updates': 0, 'scheduled': False, 'plan': plan['gpu_diagnostics']}))
        return
    if not args.user_opened_card:
        raise SystemExit('STOP: explicit new human GPU-ready instruction is required; preparation is not permission to run')
    if not args.cpu_attribution:
        raise SystemExit('STOP: actual CPU attribution and FP32 replay validation required first')
    receipt_path = Path(args.cpu_attribution)
    cpu = json.loads(receipt_path.read_text(encoding='utf-8'))
    check = cpu['CPU_FP32_replay']
    if check['static_raw_RGB_MAE'] > check['MAE_limit'] or check['static_raw_RGB_max_abs'] > check['max_abs_limit']:
        raise SystemExit('STOP: CPU replay failed validation')
    root = Path(plan['remote_root'])
    case = plan['primary_case']
    if (cpu['scene'], cpu['frame_id']) != (case['scene_id'], case['frame_id']):
        raise SystemExit('STOP: wrong actual case')
    spec_path = root / case['spec_relative']
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    source = Path(spec['source_repository'])
    if hashlib.sha256((source/'scene/gaussian_model.py').read_bytes()).hexdigest() != cpu['source_model_sha256']:
        raise SystemExit('STOP: Gaussian model source changed since CPU attribution')
    checkpoint = root / case['checkpoint_relative']
    identity = cpu['checkpoint_identity']
    if (identity['path'], identity['bytes'], identity['mtime_ns']) != (str(checkpoint), checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns):
        raise SystemExit('STOP: actual checkpoint changed since CPU attribution')
    evaluation = root / case['evaluation_relative']
    manifest = json.loads((evaluation/'scene_evaluation_manifest.json').read_text(encoding='utf-8'))
    target = next(f for f in manifest['streams']['observed']['frames'] if f['frame_id'] == case['frame_id'])
    out = Path(args.output or root/'black_shadow_audit_20261016_v1'/'gpu_diagnostics')
    if out.exists():
        raise SystemExit('STOP: output exists; preserve partial assets and choose a new directory')
    sys.path.insert(0, str(root/'infra'))
    from priority import require_gpu_slot, inspect_priority
    lock, admission = require_gpu_slot(plan['gpu_lock'])
    try:
        import numpy as np
        import torch
        from PIL import Image
        from runtime import load_models
        from renderer import render
        cp_stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        out.mkdir(parents=True)
        restored = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if restored['step'] != 30000 or restored['spec'] != spec:
            raise ValueError('Actual checkpoint step/spec differs')
        config, scene, static, instances, optimizer, env, exclusion = load_models(spec, out, restore=restored)
        del restored
        if exclusion['removed_points'] != 0 or len(static.get_xyz) != case['expected_background_points']:
            raise ValueError('Complete-background topology contract differs')
        cameras = list(scene.getTrainCameras())
        if [Path(c.image_name).stem for c in cameras] != [f['frame_id'] for f in manifest['streams']['observed']['frames']]:
            raise ValueError('Actual camera sequence differs')
        camera = cameras[target['frame_index']]
        if not np.allclose(camera.world_view_transform.cpu().numpy(), target['camera']['world_view_transform'], atol=1e-7, rtol=0):
            raise ValueError('Actual camera pose differs')
        fields = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', '_t', '_scaling_t', '_velocity')
        versions = {name: getattr(static, name)._version for name in fields}
        ids = torch.as_tensor(cpu['selected_static_ids'], dtype=torch.long, device=static.get_xyz.device)
        if ids.numel() == 0 or ids.min() < 0 or ids.max() >= len(static.get_xyz) or len(ids.unique()) != len(ids):
            raise ValueError('Invalid actual contributor IDs')
        bg = static.get_xyz.new_zeros(3)
        files = []

        def call(cam, model=static, mask=None, other=None):
            if not inspect_priority()['allowed']:
                raise RuntimeError('Yield to StreetCrafter/other work; partial diagnostic preserved, no automatic retry')
            return render(cam, model, None, config, bg, env_map=env, mask=mask, other=other,
                          render_separate=False, max_features=10, is_training=False)

        def save(name, pkg):
            arrays = {'RGB': pkg['render'].cpu().numpy(), 'raw_RGB': pkg['raw_render'].cpu().numpy(),
                      'alpha': pkg['alpha'].cpu().numpy(),
                      'expected_center_depth_model': (pkg['depth']/pkg['alpha'].clamp_min(1e-8)).cpu().numpy()}
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Nonfinite actual diagnostic')
            np.savez_compressed(out/(name+'.npz'), **arrays)
            rgb = np.moveaxis(np.clip(arrays['RGB'], 0, 1), 0, -1)
            Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(out/(name+'.png'))
            files.append(name)

        with torch.no_grad():
            baseline = call(camera)
            with np.load(evaluation/target['npz'], allow_pickle=False) as saved:
                delta = np.abs(baseline['raw_render'].cpu().numpy()-saved['static_raw_rgb'])
            actual_check = {'raw_RGB_max_abs': float(delta.max()), 'raw_RGB_MAE': float(delta.mean())}
            if actual_check['raw_RGB_max_abs'] > 1e-5 or actual_check['raw_RGB_MAE'] > 1e-6:
                raise AssertionError('GPU baseline does not reproduce existing FP32; stop before ablations')
            save('target_original_Static', baseline)
            xy = np.asarray(case['probe_pixels_xy'], int)
            exact = []
            # True per-Gaussian alpha*T from feature channels; keep ALL geometry
            # and sorting in EVERY pass. Do not treat contrib counts as IDs.
            for first in range(0, len(ids), 7):
                group = ids[first:first+7]
                payload = static.get_xyz.new_zeros((len(static.get_xyz), len(group)))
                payload[group, torch.arange(len(group), device=group.device)] = 1
                pkg = call(camera, other=payload)
                if not torch.equal(pkg['render'], baseline['render']):
                    raise AssertionError('Feature tracing changed scene RGB')
                roi = case['ROI_xyxy_exclusive']
                feature = pkg['feature'].cpu().numpy()
                for j, identity in enumerate(group.tolist()):
                    exact.append({'static_id': identity,
                        'ROI_mean_alpha_T': float(feature[j, roi[1]:roi[3], roi[0]:roi[2]].mean()),
                        'probe_alpha_T': feature[j, xy[:, 1], xy[:, 0]].tolist()})
                del pkg, payload
            for count in (25, 50, 100):
                count = min(count, len(ids))
                keep = torch.ones(len(static.get_xyz), dtype=torch.bool, device=ids.device)
                keep[ids[:count]] = False
                save('target_counterfactual_hide_top%d_NOT_A_REPAIR'%count, call(camera, mask=keep))

            class FrozenTime:
                def __init__(self, base, t, motion, marginal):
                    self.base, self.t, self.motion, self.marginal = base, t, motion, marginal
                def __getattr__(self, name):
                    return getattr(self.base, name)
                def get_xyz_SHM(self, t):
                    return self.base.get_xyz_SHM(self.t if self.motion else t)
                def get_marginal_t(self, t):
                    return self.base.get_marginal_t(self.t if self.marginal else t)

            for frame_id in plan['gpu_diagnostics']['fixed_camera_time_frames']:
                row = next(f for f in manifest['streams']['observed']['frames'] if f['frame_id'] == frame_id)
                fixed = copy.copy(camera)
                fixed.timestamp = row['model_time']
                for name, model in [('normal', static),
                    ('freeze_position_only', FrozenTime(static, camera.timestamp, True, False)),
                    ('freeze_marginal_only', FrozenTime(static, camera.timestamp, False, True)),
                    ('freeze_both', FrozenTime(static, camera.timestamp, True, True))]:
                    pkg = call(fixed, model=model)
                    if name == 'freeze_both' and not torch.equal(pkg['render'], baseline['render']):
                        raise AssertionError('Fixed camera plus frozen position/marginal must reproduce anchor RGB')
                    save('fixed_'+frame_id+'_'+name, pkg)
            # Original GN11 uses a separate in-memory object and the same camera.
            from scene import GaussianModel
            original_cap, step = torch.load(spec['background_checkpoint'], map_location='cuda', weights_only=False)
            if step != 30000 or len(original_cap[1]) != len(static.get_xyz):
                raise ValueError('Original GN11 identity/topology differs')
            original = GaussianModel(config)
            original.restore(original_cap, config)
            save('target_original_GN11_Static', call(camera, model=original))
            for name in fields:
                if getattr(static, name)._version != versions[name]:
                    raise AssertionError('Read-only diagnosis modified model parameters')
            if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != cp_stat:
                raise AssertionError('Source checkpoint changed during diagnosis')
        receipt = {'status': 'DIAGNOSTIC_OUTPUTS_REQUIRE_ACTUAL_IMAGE_AND_CROSS_VIEW_REVIEW',
            'scene': cpu['scene'], 'frame_id': cpu['frame_id'], 'GPU_used': True,
            'training_updates': 0, 'background_points_permanently_removed': 0,
            'temporary_hide_is_causal_ablation_not_repair': True, 'checkpoint_written': False,
            'GPU_baseline_FP32_check': actual_check, 'admission': admission,
            'exact_GPU_selected_contributors': exact, 'exported_diagnostic_names': files,
            'selected_contributor_ROI_alpha_T_sum': sum(r['ROI_mean_alpha_T'] for r in exact),
            'freeze_both_fixed_camera_matches_anchor_bitwise': True,
            'ground_depth_is_expected_center_depth_not_surface_truth': True,
            'final_cause_classification': None, 'photographic_quality_accepted': False}
        (out/'GPU_diagnostic_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps({'status': receipt['status'], 'output': str(out), 'training_updates': 0}))
    finally:
        lock.close()


if __name__ == '__main__':
    main()

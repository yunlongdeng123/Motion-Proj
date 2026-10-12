"""Supplementary localisation of existing saved arrays, never a renderer or trainer.

Boundary width5 derives from the actual SSIM11 window radius; it is descriptive,
not an acceptance threshold or an outcome-selected training mask.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
C = BASE / 'campaign/vgpu_bugfix_20261012'
HERE = C / 'loss_audit/next_diagnostics_cpu/longV_domains_focus'
D = C / 'continuation_diagnostic'
DATA = BASE / 'data/waymo_paper_scenes/scene0017_000_048'
RADIUS = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def metrics(values, mask):
    count = int(mask.sum())
    if not count:
        return {'pixel_count': 0}
    gt = values['gt'][:, mask].astype(np.float64)
    result = {'pixel_count': count}
    for channel in ('composite', 'static', 'dynamic'):
        rgb = values[channel][:, mask].astype(np.float64)
        result[channel + '_clamped_MSE'] = float(np.mean((np.clip(rgb, 0, 1) - gt) ** 2))
        result[channel + '_clamped_MAE'] = float(np.mean(np.abs(np.clip(rgb, 0, 1) - gt)))
    a = values['alpha_d'][0, mask].astype(np.float64)
    rgb = values['dynamic'][:, mask].astype(np.float64)
    result['alpha_d_mean'] = float(a.mean())
    support = np.any(np.floor(np.clip(rgb, 0, 1) * 255 + .5) > 2, axis=0)
    result['dynamic_RGB_support_fraction'] = float(support.mean())
    result['dynamic_support_and_alpha_lt_0_1_fraction'] = float((support & (a < .1)).mean())
    result['dynamic_raw_RGB_mean'] = float(rgb.mean())
    result['dynamic_RGB_over_alpha_mean'] = float((rgb / np.maximum(a, 1e-5)[None]).mean())
    center_env_dynamic = rgb + (1 - a)[None] * values['environment_sample'][:, mask].astype(np.float64)
    result['center_env_filled_dynamic_clamped_MSE'] = float(np.mean((np.clip(center_env_dynamic, 0, 1) - gt) ** 2))
    return result


def main():
    finals = {arm: read(D / name / 'extra_eval/32000/diagnostic_eval.json') for arm, name in [('B1', 'original30k_to32000'), ('V', 'opacity_outside30k_to32000')]}
    frames = []
    for f1, fv in zip(finals['B1']['frames'], finals['V']['frames']):
        assert f1['frame_id'] == fv['frame_id']
        name = f1['frame_id'].split('/')[-1]
        with Image.open(DATA / 'dynamic_masks/dynamic_mask_0' / (name + '.png')) as image:
            actor = np.asarray(image.convert('L').resize((480, 320), Image.Resampling.NEAREST)) > 0
        assert int(actor.sum()) == f1['regions']['actor']['pixel_count'] == fv['regions']['actor']['pixel_count']
        inside, outside = distance_transform_edt(actor), distance_transform_edt(~actor)
        masks = {'actor': actor, 'actor_core': actor & (inside > RADIUS), 'actor_inner_boundary': actor & (inside <= RADIUS), 'outside_boundary': ~actor & (outside <= RADIUS), 'outside_far': ~actor & (outside > RADIUS)}
        row = {'frame_id': f1['frame_id'], 'actor_pixel_count': int(actor.sum()), 'actor_fraction_full_HW': float(actor.mean()), 'body_mask_matches_both_actual_eval_counts': True, 'regions': {}}
        for arm, frame in [('B1', f1), ('V', fv)]:
            with np.load(frame['float32_arrays'], allow_pickle=False) as arrays:
                values = {key: arrays[key] for key in ('gt', 'composite', 'static', 'dynamic', 'alpha_d', 'environment_sample')}
            for label, mask in masks.items():
                row['regions'].setdefault(label, {})[arm] = metrics(values, mask)
        frames.append(row)
    aggregate = {}
    for region in ('actor', 'actor_core', 'actor_inner_boundary', 'outside_boundary', 'outside_far'):
        result = {'nonempty_frame_count': sum(f['regions'][region]['B1']['pixel_count'] > 0 for f in frames)}
        for arm in ('B1', 'V'):
            rows = [f['regions'][region][arm] for f in frames if f['regions'][region][arm]['pixel_count'] > 0]
            total = sum(r['pixel_count'] for r in rows)
            result[arm] = {'pixel_count_summed': total}
            for key in rows[0]:
                if key != 'pixel_count':
                    result[arm][key + '_equal_frame'] = sum(r[key] for r in rows) / len(rows)
                    result[arm][key + '_pixel_weighted'] = sum(r[key] * r['pixel_count'] for r in rows) / total
            result[arm]['composite_pooled_PSNR'] = -10 * math.log10(result[arm]['composite_clamped_MSE_pixel_weighted'])
        aggregate[region] = result
    primary_actor_summary = {}
    for arm, name in [('B1', 'original30k_to32000'), ('B2', 'original30k_to32000_noise_reference'), ('V', 'opacity_outside30k_to32000')]:
        x = read(D / name / 'extra_eval/32000/diagnostic_eval.json')['frames']
        counts = [f['regions']['actor']['pixel_count'] for f in x]
        mses = [f['regions']['actor']['metrics']['composite']['clamped_rgb_mse_vs_gt'] for f in x]
        pooled = sum(c * m for c, m in zip(counts, mses)) / sum(counts)
        primary_actor_summary[arm] = {'actor_pixels_summed': sum(counts), 'actor_min_max_pixels': [min(counts), max(counts)], 'equal_frame_MSE': sum(mses) / len(mses), 'pooled_MSE': pooled, 'pooled_PSNR': -10 * math.log10(pooled)}
    result = {'status': 'CPU_SAVED_ARRAYS_ACTOR_LOCALISATION_COMPLETE', 'training_or_GPU_started': False, 'checkpoint_source_config_modified': False, 'supplementary_only_original_locked_endpoint_unchanged': True, 'SSIM_original_window': 11, 'derived_descriptive_boundary_radius': RADIUS, 'original_mask_input_and_eval_actor_counts_match_all49': True, 'environment_reconstruction_limit': 'Raw eval dynamic excludes environment. CPU center-env-filled dynamic reconstructs Eq6 with saved eval rays, not actual jittered training RGB or loss.', 'ratio_limit': 'RGB/alpha is a rendered conditional radiance descriptor, not proof of an illegal color range or individual Gaussian albedo.', 'original_actor_summaries': primary_actor_summary, 'aggregation': aggregate, 'frames': frames, 'saved_eval_source_SHA': {arm: sha(D / name / 'extra_eval/32000/diagnostic_eval.json') for arm, name in [('B1', 'original30k_to32000'), ('V', 'opacity_outside30k_to32000')]}, 'script_SHA': sha(__file__)}
    (HERE / 'actor_localisation_saved_arrays_cpu.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'status': result['status'], 'actor_primary': primary_actor_summary, 'core_boundary_pooled': {r: {arm: {k: value for k, value in aggregate[r][arm].items() if k in ('pixel_count_summed', 'composite_pooled_PSNR', 'alpha_d_mean_pixel_weighted', 'dynamic_RGB_support_fraction_pixel_weighted', 'center_env_filled_dynamic_clamped_MSE_pixel_weighted')} for arm in ('B1', 'V')} for r in aggregate}}, allow_nan=False))


if __name__ == '__main__':
    main()

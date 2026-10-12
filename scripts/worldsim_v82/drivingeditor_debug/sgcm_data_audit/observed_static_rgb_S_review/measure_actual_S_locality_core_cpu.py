"""Completed-S supplement using existing locality/core recipes; default prep only.

No Torch/renderer/training. Reuse cached B1/V core and B1/B2/V locality; read only
the completed S/B2 saved NPZ needed for the new supplemental comparisons.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S = ROOT / 'continuation_diagnostic/observed_static_rgb30k_to32000'
RUNS = {'B1': S.parent / 'original30k_to32000', 'B2': S.parent / 'original30k_to32000_noise_reference',
        'V': S.parent / 'opacity_outside30k_to32000', 'S': S}
FOCUS = ROOT / 'loss_audit/next_diagnostics_cpu/longV_domains_focus'
OLD_LOCAL = ROOT / 'sgcm_data_audit/long_pair_review/removal_locality_actual_cpu.json'
PINS = {FOCUS / 'localize_saved_actor_cpu.py': '22d18c0cc225ed1b0ec0effe14c85436eabddb333f7a2851b5698d40a9672069',
        FOCUS / 'actor_localisation_saved_arrays_cpu.json': '0c01fafefd3c5ce29a20fdf178b4213d281e3fd92c923072366d64f4fc017bc3',
        OLD_LOCAL: '463fa751241988eb49375d4de447a46000d7c506fbaa755f854d7d3ac8817b8d',
        HERE / 'final_review_S.json': 'ef4423f2a82adb830ef8366e0982607d662947330c98f871d8ffc93e743eeb89'}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(frames, domains, value_field):
    result = {}
    for region in domains:
        selected = [r for r in frames if r[value_field][region]['B1']['pixel_count'] > 0]
        result[region] = {'nonempty_frame_count': len(selected), 'empty_frame_ids': [r['frame_id'] for r in frames if r not in selected]}
        for arm in RUNS:
            rows = [r[value_field][region][arm] for r in selected]
            total = sum(r['pixel_count'] for r in rows)
            if not rows:
                result[region][arm] = {'pixel_count_summed': 0, 'metrics': {}}
                continue
            values = {key: {'equal_frame_mean': float(np.mean([r[key] for r in rows])),
                            'pixel_weighted_mean': float(np.average([r[key] for r in rows], weights=[r['pixel_count'] for r in rows]))}
                      for key in rows[0] if key != 'pixel_count'}
            if 'composite_clamped_MSE' in values:
                values['composite_pooled_PSNR'] = -10 * math.log10(values['composite_clamped_MSE']['pixel_weighted_mean'])
            result[region][arm] = {'pixel_count_summed': total, 'metrics': values}
    return result


def paired_supplement(frames, domains, fields):
    """Keep every nonempty fixed region and all signed effects; no acceptance rule."""
    result = {}
    for region in domains:
        result[region] = {}
        for key in fields:
            comparisons = {}
            for reference in ('B1', 'B2', 'V'):
                pairs = []
                for row in frames:
                    values = row['regions'][region]
                    if not values['S']['pixel_count']:
                        continue
                    delta = values['S'][key] - values[reference][key]
                    pairs.append({'frame_id': row['frame_id'], 'pixel_count': values['S']['pixel_count'],
                                  'S': values['S'][key], 'reference': values[reference][key],
                                  'S_minus_reference': delta, 'abs_S_minus_reference': abs(delta)})
                comparisons[reference] = {'interpretation': 'descriptive_only_different_added_objectives' if reference == 'V' else 'candidate_vs_original',
                                           'frames': pairs, 'lower_count': sum(p['S_minus_reference'] < 0 for p in pairs),
                                           'higher_count': sum(p['S_minus_reference'] > 0 for p in pairs),
                                           'signed_mean': float(np.mean([p['S_minus_reference'] for p in pairs])) if pairs else None,
                                           'absolute_mean': float(np.mean([p['abs_S_minus_reference'] for p in pairs])) if pairs else None}
            result[region][key] = comparisons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed', action='store_true')
    args = parser.parse_args()
    assert all(sha(path) == digest for path, digest in PINS.items())
    if not args.completed:
        print(json.dumps({'status': 'S_LOCALITY_CORE_RECIPE_READY_WAIT_TRUE_COMPLETED_EXPORT',
                          'S_result_or_placeholder_written': False, 'existing_boundary_radius': 5,
                          'radius_derivation': 'Existing actual SSIM11 radius, descriptive only', 'GPU_used': False}))
        return
    status = read(S / 'status.json')
    assert status['status'] == 'complete' and status['actual_exit_code'] == 0 and not status['gpu_lock_held']
    summary = read(S / 'receipts/full49_summary.json')
    assert summary['status'] == 'complete_long_paired_static_anchor_receipts_verified_CPU'
    assert summary['harness_exit_code'] == summary['wrapper_exit_code'] == 0 and summary['step_count'] == 2000
    assert summary['all_protected_SHA_unchanged'] and summary['all4_node_eval_RNG_preserved']
    common = read(S / 'receipts/common49_inputs_after_S_cpu.json')
    assert common['actual467_input_SHA_match_original'] and common['new_shadow_mask_absent']
    plan = read(S / 'diagnostic_plan.json')
    data_source = Path(plan['training_config']['source_path'])
    spec = importlib.util.spec_from_file_location('unchanged_CPU_core_metric_recipe', FOCUS / 'localize_saved_actor_cpu.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.RADIUS == 5
    cached_core = read(FOCUS / 'actor_localisation_saved_arrays_cpu.json')
    cached_locality = read(OLD_LOCAL)
    finals = {a: read(p / 'extra_eval/32000/diagnostic_eval.json') for a, p in RUNS.items()}
    summaries = {a: read(p / 'receipts/full49_summary.json') for a, p in RUNS.items()}
    tables = {a: {r['frame_id']: r for r in d['frames']} for a, d in finals.items()}
    previous_core = {r['frame_id']: r for r in cached_core['frames']}
    previous_locality = {r['frame_id']: r for r in cached_locality['frames']}
    for arm in ('B1', 'V'):
        assert sha(RUNS[arm] / 'extra_eval/32000/diagnostic_eval.json') == cached_core['saved_eval_source_SHA'][arm]
    core_rows, locality_rows, mask_pins = [], [], []
    for raw in range(49):
        name = f'0017{raw:03}'
        frame_id = 'image_0/' + name
        actor_rel = 'dynamic_masks/dynamic_mask_0/' + name + '.png'
        sky_rel = 'sky/sky_0/' + name + '.png'
        for relative in (actor_rel, sky_rel):
            assert sha(data_source / relative) == common['actual_input_SHA'][relative]
        with Image.open(data_source / actor_rel) as image:
            actor = np.asarray(image.convert('L').resize((480, 320), Image.Resampling.NEAREST)) > 0
        with Image.open(data_source / sky_rel) as image:
            sky = cv2.resize((np.asarray(image) > 0).astype(np.float32), (480, 320)) > 0
        domains = {'full': np.ones((320, 480), bool), 'sky': sky, 'non_sky': ~sky, 'actor': actor,
                   'outside_actor': ~actor, 'non_sky_outside_actor': ~sky & ~actor}
        counts = {r: int(m.sum()) for r, m in domains.items()}
        assert all(counts == {r: x['pixel_count'] for r, x in tables[a][frame_id]['regions'].items()} for a in RUNS)
        inside, outside = distance_transform_edt(actor), distance_transform_edt(~actor)
        core_domains = {'actor': actor, 'actor_core': actor & (inside > 5), 'actor_inner_boundary': actor & (inside <= 5),
                        'outside_boundary': ~actor & (outside <= 5), 'outside_far': ~actor & (outside > 5)}
        cr = {'frame_id': frame_id, 'regions': {r: {a: previous_core[frame_id]['regions'][r][a] for a in ('B1', 'V')} for r in core_domains}}
        lr = {'frame_id': frame_id, 'regions': {r: {a: {'pixel_count': counts[r], **previous_locality[frame_id]['arms'][a][r]} for a in ('B1', 'B2', 'V')} for r in domains}}
        for arm in ('B2', 'S'):
            record = tables[arm][frame_id]
            npz = Path(record['float32_arrays'])
            pin = next(r for r in summaries[arm]['artifact_pins'] if r.get('artifact') == 'raw_float32_arrays' and r['frame_id'] == frame_id)
            assert pin['path'] == str(npz) and npz.stat().st_size == pin['bytes']
            with np.load(npz, allow_pickle=False) as arrays:
                values = {k: arrays[k] for k in ('gt', 'composite', 'static', 'dynamic', 'alpha_d', 'environment_sample')}
            for region, mask in core_domains.items():
                assert int(mask.sum()) == cr['regions'][region]['B1']['pixel_count'] == cr['regions'][region]['V']['pixel_count']
                cr['regions'][region][arm] = helper.metrics(values, mask)
            if arm == 'S':
                composite, static = np.clip(values['composite'], 0, 1), np.clip(values['static'], 0, 1)
                error = np.abs(composite.astype(np.float64) - static.astype(np.float64))
                edited = np.any(np.abs(np.floor(composite * 255. + .5) - np.floor(static * 255. + .5)) > 2, axis=0)
                for region, mask in domains.items():
                    lr['regions'][region]['S'] = {'pixel_count': counts[region], 'mean_absolute_composite_static_difference': float(error[:, mask].mean()) if counts[region] else None,
                                                  'edited_support_gt2_uint8_pixel_fraction': float(edited[mask].mean()) if counts[region] else None}
        core_rows.append(cr)
        locality_rows.append(lr)
        mask_pins.append({'frame_id': frame_id, 'all_six_actual_eval_counts': counts,
                          'actor_bool_sha256': hashlib.sha256(actor.tobytes()).hexdigest(),
                          'sky_bool_sha256': hashlib.sha256(sky.tobytes()).hexdigest()})
    result = {'status': 'ACTUAL_COMPLETED_S_EXISTING_LOCALITY_CORE_CPU_SUPPLEMENT', 'frame_count': 49, 'endpoint': 32000,
              'primary_metrics_redefined_or_overwritten': False, 'GPU_or_training_started': False,
              'source_script_sha256': sha(Path(__file__)), 'existing_recipe_pins': {str(p): v for p, v in PINS.items()},
              'core_regions': aggregate(core_rows, core_domains, 'regions'), 'locality_regions': aggregate(locality_rows, domains, 'regions'),
              'S_core_supplement_vs_each_reference': paired_supplement(core_rows, core_domains, ('composite_clamped_MSE', 'composite_clamped_MAE', 'alpha_d_mean', 'dynamic_RGB_support_fraction')),
              'S_locality_supplement_vs_each_reference': paired_supplement(locality_rows, domains, ('mean_absolute_composite_static_difference', 'edited_support_gt2_uint8_pixel_fraction')),
              'core_frames': core_rows, 'locality_frames': locality_rows, 'mask_receipts': mask_pins,
              'existing_core_radius': 5, 'radius_reason': 'Prior actual SSIM11 radius; unchanged descriptive supplement, not acceptance threshold or training mask.',
              'cached_B1_V_core_and_B1_B2_V_locality_reused': True,
              'source_eval_SHA': {a: sha(p / 'extra_eval/32000/diagnostic_eval.json') for a, p in RUNS.items()},
              'limitations': ['Primary S-vs-B1/B2 uses the original11 and126 domains; core/boundary is supplementary.',
                              'S-vsV is descriptive comparison of two different added objectives, not a single-variable transition.',
                              'Actor regions have GT for composite fidelity; static GT error inside body is not edited-scene accuracy.',
                              'Outsidebody may include cast shadows and sky. Locality is composite-versus-static change, not dense edited-GT accuracy.',
                              cached_core['environment_reconstruction_limit'], cached_core['ratio_limit'],
                              'Actual masks are reconstructed with unchanged recipe and input hashes/counts, not saved runtime masks/actualScene pose snapshots.',
                              'NPZ paths/bytes and canonical exporter hashes bind actual arrays; large NPZ/checkpoint SHA not redundantly recomputed.']}
    path = HERE / 'actual_S_locality_core_cpu.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'sha256': sha(path), 'frame_count': 49}))


if __name__ == '__main__':
    main()

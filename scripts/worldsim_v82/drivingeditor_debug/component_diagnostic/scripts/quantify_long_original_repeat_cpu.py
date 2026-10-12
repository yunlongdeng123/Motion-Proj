"""Describe the completed B1/B2 repeat variation under the locked endpoint protocol.

Reads saved JSON and genuine PNGs only. Does not import torch, run a renderer,
alter a checkpoint, or claim a statistical noise bound.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
C = BASE / 'campaign/vgpu_bugfix_20261012'
P = C / 'loss_audit/long_pair_30k_32000/materialized_plan'
B1 = C / 'continuation_diagnostic/original30k_to32000'
B2 = C / 'continuation_diagnostic/original30k_to32000_noise_reference'
PROTOCOL_SHA = 'c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658'
REGIONS = ('full', 'sky', 'non_sky', 'actor', 'outside_actor', 'non_sky_outside_actor')


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def json_number(value):
    if value is None:
        return None
    if math.isfinite(value):
        return float(value)
    return {'nonfinite_number': '+infinity' if value > 0 else '-infinity'}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    finite = values[np.isfinite(values)]
    result = {'count': len(values), 'finite_count': len(finite),
              'infinite_count': int(np.isinf(values).sum()),
              'mean': json_number(float(values.mean()))}
    if len(finite):
        result['finite_values'] = {label: float(np.quantile(finite, q))
                                  for label, q in (('min', 0), ('p05', .05),
                                                   ('p50', .5), ('p95', .95), ('max', 1))}
    return result


def compare_metric(frames1, frames2, region, array, metric, primary=False):
    paired = []
    excluded = []
    for f1, f2 in zip(frames1, frames2):
        r1, r2 = f1['regions'][region], f2['regions'][region]
        assert f1['frame_id'] == f2['frame_id'] and r1['pixel_count'] == r2['pixel_count']
        if not r1['pixel_count']:
            excluded.append(f1['frame_id'])
            continue
        v1 = r1['metrics'][array][metric]
        v2 = r2['metrics'][array][metric]
        if metric == 'clamped_rgb_psnr_db':
            if v1 is None:
                assert r1['metrics'][array]['clamped_rgb_mse_vs_gt'] == 0
                v1 = math.inf
            if v2 is None:
                assert r2['metrics'][array]['clamped_rgb_mse_vs_gt'] == 0
                v2 = math.inf
        else:
            assert v1 is not None and v2 is not None and math.isfinite(v1) and math.isfinite(v2)
        delta = v2 - v1
        if math.isnan(delta):
            delta = None  # +infinity minus +infinity is not a numeric repeat difference.
        paired.append({'frame_id': f1['frame_id'], 'pixel_count': r1['pixel_count'],
                       'B1': json_number(v1), 'B2': json_number(v2),
                       'B2_minus_B1': json_number(delta),
                       'abs_B2_minus_B1': json_number(abs(delta)) if delta is not None else None,
                       '_B1': v1, '_B2': v2, '_delta': delta})
    assert paired, region
    mean1 = float(np.mean([r['_B1'] for r in paired], dtype=np.float64))
    mean2 = float(np.mean([r['_B2'] for r in paired], dtype=np.float64))
    mean_delta = mean2 - mean1
    if math.isnan(mean_delta):
        mean_delta = None
    defined = [r['_delta'] for r in paired if r['_delta'] is not None]
    result = {'region': region, 'array': array, 'metric': metric, 'preregistered_primary': primary,
              'equal_frame_mean_B1': json_number(mean1), 'equal_frame_mean_B2': json_number(mean2),
              'equal_frame_mean_B2_minus_B1': json_number(mean_delta),
              'N_abs_difference_of_equal_frame_means': json_number(abs(mean_delta)) if mean_delta is not None else None,
              'paired_signed_difference_distribution': distribution(defined),
              'paired_absolute_difference_distribution': distribution([abs(v) for v in defined]),
              'paired_difference_undefined_count': len(paired) - len(defined),
              'paired_exact_numeric_equal_count': sum(r['_B1'] == r['_B2'] for r in paired),
              'paired_B2_lower_count': sum(v < 0 for v in defined),
              'paired_B2_higher_count': sum(v > 0 for v in defined),
              'included_nonempty_frame_count': len(paired), 'empty_frames_excluded': excluded,
              'frames': [{k: v for k, v in r.items() if not k.startswith('_')} for r in paired]}
    return result


def main():
    protocol_path = P / 'evaluation_protocol_locked.json'
    assert pin(protocol_path)['sha256'] == PROTOCOL_SHA
    protocol = read(protocol_path)
    summaries = {name: read(out / 'receipts/full49_summary.json') for name, out in (('B1', B1), ('B2', B2))}
    for name, summary in summaries.items():
        assert summary['arm'] == name and summary['step_count'] == 2000
        assert summary['first_step_original10_exact'] and summary['all4_node_eval_RNG_preserved']
        assert summary['harness_exit_code'] == 0 and summary['wrapper_exit_code'] == 0
        assert summary['all_protected_SHA_unchanged'] and not summary['original_algorithm_changed']
    pair = summaries['B2']['paired_original_repeat_evidence']
    assert pair['B1_B2_complete_pair_validated'] and pair['all2000_camera_time_shift_LR_exact']
    assert pair['all49_GT_float32_dimensions_region_counts_exact']
    finals = [read(out / 'extra_eval/32000/diagnostic_eval.json') for out in (B1, B2)]
    frames1, frames2 = [f['frames'] for f in finals]
    ids = [f'image_0/0017{i:03}' for i in range(49)]
    assert [f['frame_id'] for f in frames1] == [f['frame_id'] for f in frames2] == ids
    for f1, f2 in zip(frames1, frames2):
        assert {k: v['pixel_count'] for k, v in f1['regions'].items()} == {k: v['pixel_count'] for k, v in f2['regions'].items()}
    primary = []
    for spec in protocol['primary_metrics']:
        for metric in spec['metrics']:
            primary.append(compare_metric(frames1, frames2, spec['domain'], spec['array'], metric, primary=True))
    context = []
    for region in REGIONS:
        for array, fields in (
            ('alpha_s', ('mean', 'fraction_gt_0_1', 'fraction_gt_0_5', 'fraction_gt_0_9')),
            ('alpha_d', ('mean', 'fraction_gt_0_1', 'fraction_gt_0_5', 'fraction_gt_0_9')),
            ('environment_sample', ('mean', 'fraction_lt0', 'fraction_gt1')),
            ('static_foreground_contribution', ('mean',)),
            ('static_env_contribution', ('mean',)),
            ('dynamic', ('mean', 'rgb_support_gt2_uint8_pixel_fraction')),
            ('static', ('clamped_rgb_mae_vs_gt', 'clamped_rgb_mse_vs_gt', 'clamped_rgb_psnr_db')),
            ('composite', ('clamped_rgb_mae_vs_gt', 'clamped_rgb_mse_vs_gt', 'clamped_rgb_psnr_db'))):
            for field in fields:
                context.append(compare_metric(frames1, frames2, region, array, field))
    step_rows = [[json.loads(s) for s in (out / 'step_losses.jsonl').read_text().splitlines()] for out in (B1, B2)]
    assert [r['iteration'] for r in step_rows[0]] == [r['iteration'] for r in step_rows[1]] == list(range(30001, 32001))
    assert all(all(r1[k] == r2[k] for k in ('iteration', 'camera_id', 'time_shift', 'learning_rates'))
               for r1, r2 in zip(*step_rows))
    scalar_keys = read(B1 / 'first_step_reference_comparison.json')['comparisons'].keys()
    train_variation = {}
    for key in scalar_keys:
        deltas = [r2['losses'][key] - r1['losses'][key] for r1, r2 in zip(*step_rows)]
        assert all(math.isfinite(v) for v in deltas)
        differing = [r2['iteration'] for r1, r2 in zip(*step_rows) if r1['losses'][key] != r2['losses'][key]]
        train_variation[key] = {'first_iteration_not_bitwise_equal': differing[0] if differing else None,
                                'exact_equal_step_count': 2000 - len(differing),
                                'signed_difference_distribution': distribution(deltas),
                                'absolute_difference_distribution': distribution([abs(v) for v in deltas]),
                                'last_step_B1': step_rows[0][-1]['losses'][key],
                                'last_step_B2': step_rows[1][-1]['losses'][key]}
    png_differences = []
    for f1, f2 in zip(frames1, frames2):
        channels = {}
        for channel in ('gt', 'composite', 'static', 'dynamic', 'alpha_s', 'alpha_d'):
            with Image.open(f1['outputs'][channel]) as im1, Image.open(f2['outputs'][channel]) as im2:
                a1, a2 = np.asarray(im1.convert('RGB')), np.asarray(im2.convert('RGB'))
                assert a1.shape == a2.shape == (320, 480, 3)
                diff = a2.astype(np.int16) - a1.astype(np.int16)
                exact = bool(np.array_equal(a1, a2))
                if channel == 'gt':
                    assert exact
                channels[channel] = {'exact_pixels_equal': exact,
                                     'mean_abs_uint8_channel_difference': float(np.abs(diff).mean()),
                                     'max_abs_uint8_channel_difference': int(np.abs(diff).max()),
                                     'fraction_pixels_any_channel_changed': float(np.any(diff != 0, axis=2).mean())}
        png_differences.append({'frame_id': f1['frame_id'], 'channels': channels})
    inputs = [protocol_path]
    for out in (B1, B2):
        inputs.extend(out / name for name in ('receipts/full49_summary.json', 'status.json',
                                             'continuation_diagnostic.json', 'step_losses.jsonl',
                                             'first_step_reference_comparison.json',
                                             'extra_eval/32000/diagnostic_eval.json',
                                             'terminal_32000/diagnostic_snapshot_manifest.json'))
    result = {'schema_version': 1, 'status': 'completed_B1_B2_observed_original_repeat_variation',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'CPU_only_saved_evidence_analysis': True, 'training_or_CUDA_started': False,
              'analysis_script': pin(Path(__file__)), 'input_pins': [pin(path) for path in inputs],
              'pairing_validity_receipt': pair, 'endpoint_iteration': 32000,
              'aggregation': 'Equal-frame mean over every nonempty frozen domain. Pixel-weighted values remain in each arm summary, separately labelled.',
              'noise_definition': 'N=abs(mean(q_B2)-mean(q_B1)); also all49 paired signed and absolute differences.',
              'primary_metrics': primary, 'context_metrics_all_six_domains': context,
              'train_original10_observed_variation': train_variation,
              'terminal49_genuine_PNG_pixel_differences': png_differences,
              'limitations': ['One observed original repeat difference, not a confidence interval, universal noise bound, or statistical significance.',
                              'The first10 exact gate does not establish a bitwise-identical 2000-step CUDA trajectory.',
                              'Reconstruction PSNR does not certify static/dynamic separation or normal algorithm effects.',
                              'V is absent from this analysis; no candidate outcome or long GPU authorization is inferred.'],
              'whole_debug_task_complete': False, 'normal_algorithm_success_claimed': False}
    destination = B2 / 'receipts/B1_B2_observed_noise.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'receipt': pin(destination),
                      'primary_equal_frame_noise': [{k: row[k] for k in ('region', 'array', 'metric',
                                                                         'equal_frame_mean_B1', 'equal_frame_mean_B2',
                                                                         'N_abs_difference_of_equal_frame_means')}
                                                   for row in primary]}, allow_nan=False))


if __name__ == '__main__':
    main()

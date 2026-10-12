"""Saved three-arm endpoint comparison, with both originals and observed repeat variation."""
import datetime
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
C = BASE / 'campaign/vgpu_bugfix_20261012'
D = C / 'component_diagnostic'
P = C / 'loss_audit/long_pair_30k_32000/materialized_plan'
V = C / 'continuation_diagnostic/opacity_outside30k_to32000'
B1 = V.parent / 'original30k_to32000'
B2 = V.parent / 'original30k_to32000_noise_reference'
NOISE_SHA = 'e4ae32c07b63ed7be0171a727f8a12ec153f66e63d6ac89f4594fa4985145f08'
HELPER_SHA = 'd2a5d00ff35f352a186b2f6fa5bfcc7d3036b92eecd0cf7a889122f88e51a705'


def main():
    helper_path = D / 'scripts/quantify_long_original_repeat_cpu.py'
    spec = importlib.util.spec_from_file_location('frozen_original_repeat_CPU_metric_math', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    read, pin = helper.read, helper.pin
    assert pin(helper_path)['sha256'] == HELPER_SHA
    noise_path = B2 / 'receipts/B1_B2_observed_noise.json'
    assert pin(noise_path)['sha256'] == NOISE_SHA
    noise = read(noise_path)
    protocol_path = P / 'evaluation_protocol_locked.json'
    assert pin(protocol_path)['sha256'] == helper.PROTOCOL_SHA
    protocol = read(protocol_path)
    summary = read(V / 'receipts/full49_summary.json')
    assert summary['status'] == 'complete_long_paired_variant_receipts_verified_CPU'
    assert summary['harness_exit_code'] == summary['wrapper_exit_code'] == 0
    assert summary['step_count'] == 2000 and summary['all4_node_eval_RNG_preserved']
    assert summary['all_protected_SHA_unchanged'] and not summary['formal_original_source_changed']
    assert all(row['all2000_camera_time_shift_LR_exact'] for row in summary['paired_originals'].values())
    finals = {arm: read(out / 'extra_eval/32000/diagnostic_eval.json')
              for arm, out in (('B1', B1), ('B2', B2), ('V', V))}
    frames = {arm: value['frames'] for arm, value in finals.items()}
    keys = [f'image_0/0017{i:03}' for i in range(49)]
    assert all([row['frame_id'] for row in rows] == keys for rows in frames.values())
    noise_metrics = noise['primary_metrics'] + noise['context_metrics_all_six_domains']
    noise_lookup = {(row['region'], row['array'], row['metric']): row for row in noise_metrics}

    def metric(region, array, field, primary=False):
        original = noise_lookup[(region, array, field)]
        comparisons = {}
        for arm in ('B1', 'B2'):
            raw = helper.compare_metric(frames[arm], frames['V'], region, array, field, primary=primary)
            comparisons[arm] = {
                'reference_arm': arm, 'candidate_arm': 'V',
                'equal_frame_mean_reference': raw['equal_frame_mean_B1'],
                'equal_frame_mean_V': raw['equal_frame_mean_B2'],
                'equal_frame_mean_V_minus_reference': raw['equal_frame_mean_B2_minus_B1'],
                'absolute_difference_of_equal_frame_means': raw['N_abs_difference_of_equal_frame_means'],
                'paired_signed_difference_distribution': raw['paired_signed_difference_distribution'],
                'paired_absolute_difference_distribution': raw['paired_absolute_difference_distribution'],
                'included_nonempty_frame_count': raw['included_nonempty_frame_count'],
                'empty_frames_excluded': raw['empty_frames_excluded'],
                'V_lower_frame_count': raw['paired_B2_lower_count'],
                'V_higher_frame_count': raw['paired_B2_higher_count'],
                'exact_equal_frame_count': raw['paired_exact_numeric_equal_count'],
                'frames': [{'frame_id': row['frame_id'], 'pixel_count': row['pixel_count'],
                            'reference_value': row['B1'], 'V_value': row['B2'],
                            'V_minus_reference': row['B2_minus_B1'],
                            'abs_V_minus_reference': row['abs_B2_minus_B1']} for row in raw['frames']]}
        mean_v = comparisons['B1']['equal_frame_mean_V']
        assert mean_v == comparisons['B2']['equal_frame_mean_V']
        mean1, mean2 = original['equal_frame_mean_B1'], original['equal_frame_mean_B2']
        delta_middle = mean_v - (mean1 + mean2) / 2 if all(isinstance(x, (int, float)) for x in (mean_v, mean1, mean2)) else None
        return {'region': region, 'array': array, 'metric': field, 'preregistered_primary': primary,
                'equal_frame_mean_B1': mean1, 'equal_frame_mean_B2': mean2, 'equal_frame_mean_V': mean_v,
                'mean_V_minus_mean_of_B1_B2': delta_middle,
                'N_observed_original_repeat_abs_difference_of_means': original['N_abs_difference_of_equal_frame_means'],
                'original_repeat_paired_signed_difference_distribution': original['paired_signed_difference_distribution'],
                'original_repeat_paired_absolute_difference_distribution': original['paired_absolute_difference_distribution'],
                'V_vs_each_original': comparisons}

    primary = [metric(spec['domain'], spec['array'], field, primary=True)
               for spec in protocol['primary_metrics'] for field in spec['metrics']]
    assert len(primary) == 11
    context = [metric(row['region'], row['array'], row['metric'])
               for row in noise['context_metrics_all_six_domains']]
    # Retain every scalar and camera in the genuine logs, plus the common recovery nodes.
    recovery = []
    for step in protocol['nodes']:
        per_arm = {}
        for arm, out in (('B1', B1), ('B2', B2), ('V', V)):
            node = read(out / f'extra_eval/{step}/diagnostic_eval.json')
            assert node['all_rng_unchanged']
            per_arm[arm] = {'iteration': step, 'camera_count': len(node['frames']),
                            'all_rng_unchanged': True, 'model_statistics': node['model_statistics'],
                            'actual_fixed5_frame_metrics': [row for row in node['frames'] if row['frame_id'] in
                                                            [f'image_0/0017{i:03}' for i in (0, 10, 20, 30, 48)]]}
        recovery.append({'iteration': step, 'three_arm_saved_metrics': per_arm})
    rows = [json.loads(line) for line in (V / 'step_losses.jsonl').read_text().splitlines()]
    assert len(rows) == 2000 and all(math.isfinite(value) for row in rows for value in row['losses'].values())
    blocks = []
    for first in range(30001, 32001, 100):
        block = [row for row in rows if first <= row['iteration'] <= first + 99]
        blocks.append({'first_iteration': first, 'last_iteration': first + 99, 'step_count': len(block),
                       'mean_actual_psnr': float(np.mean([row['losses']['psnr'] for row in block])),
                       'mean_original_objective_before_diagnostic': float(np.mean([row['losses']['loss_original_objective_before_diagnostic'] for row in block])),
                       'mean_added_background_opacity_raw': float(np.mean([row['losses']['loss_diagnostic_dynamic_background_opacity'] for row in block])),
                       'mean_new_total_loss': float(np.mean([row['losses']['loss'] for row in block]))})
    inputs = [protocol_path, noise_path, helper_path, V / 'receipts/common49_inputs_after_V_cpu.json']
    for out in (B1, B2, V):
        inputs += [out / name for name in ('receipts/full49_summary.json', 'continuation_diagnostic.json',
                                          'extra_eval/32000/diagnostic_eval.json', 'step_losses.jsonl',
                                          'terminal_32000/diagnostic_snapshot_manifest.json')]
    result = {'schema_version': 1, 'status': 'completed_three_arm_locked_endpoint_metric_export',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'CPU_only_saved_evidence_analysis': True, 'training_or_CUDA_started': False,
              'analysis_script': pin(Path(__file__)), 'input_pins': [pin(path) for path in inputs],
              'endpoint_iteration': 32000, 'parent_iteration': 30000, 'retained_horizon': 50000,
              'outside_hypothesis_weight': .05, 'preregistered_primary_metric_count': 11,
              'paired_validity': summary['paired_originals'], 'all4_node_RNG_preserved': True,
              'primary_metrics': primary, 'context_metrics_all_six_domains': context,
              'recovery_nodes_actual_fixed5': recovery, 'V_actual_training_100step_blocks': blocks,
              'aggregation': noise['aggregation'], 'noise_definition': noise['noise_definition'],
              'limitations': ['One original repeat difference is descriptive, not a universal noise bound, confidence interval or statistical significance.',
                              'All49 per-frame actor and composite errors are retained as reconstruction tradeoffs. No dense edited GT is available inside actor removal regions.',
                              'Static is the actual observed renderer branch including the shared environment; dynamic is the official evaluation output excluding environment.',
                              'Only recovery from a degenerated30k checkpoint is tested. The candidate is not author-certified Eq13, from-zero evidence, or a complete repair.'],
              'automatic_directional_pass_or_success_assigned': False,
              'whole_debug_task_complete': False, 'normal_algorithm_success_claimed': False}
    destination = V / 'receipts/three_arm_locked_metrics.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'receipt': pin(destination),
                      'primary_equal_frame_comparison': [{key: row[key] for key in
                                                          ('region', 'array', 'metric', 'equal_frame_mean_B1',
                                                           'equal_frame_mean_B2', 'equal_frame_mean_V',
                                                           'mean_V_minus_mean_of_B1_B2',
                                                           'N_observed_original_repeat_abs_difference_of_means')}
                                                         for row in primary]}, allow_nan=False))


if __name__ == '__main__':
    main()

"""Independent reader of the sole canonical completed three-arm metric export.

Default is a preparation check. --completed requires actual S completion/export
and produces only a numeric/evidence review, never another canonical metrics file
or a normal-effect PASS. No Torch, renderer, model restoration, or GPU API.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE_ROOT = '/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012'
LONG = ROOT / 'loss_audit/long_pair_30k_32000'
S = ROOT / 'continuation_diagnostic/observed_static_rgb30k_to32000'
ARMS = {'B1': S.parent / 'original30k_to32000', 'B2': S.parent / 'original30k_to32000_noise_reference', 'S': S}
PINS = {
    ROOT / 'component_diagnostic/scripts/quantify_long_static_anchor_pair_cpu.py': '3844ec995c2cc9e9f2564291d9c2f689a019ff2ca66b721be6a2df5b94910a7c',
    ROOT / 'component_diagnostic/scripts/quantify_long_original_repeat_cpu.py': 'd2a5d00ff35f352a186b2f6fa5bfcc7d3036b92eecd0cf7a889122f88e51a705',
    LONG / 'evaluation_protocol_locked.json': 'c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658',
    ARMS['B2'] / 'receipts/B1_B2_observed_noise.json': 'e4ae32c07b63ed7be0171a727f8a12ec153f66e63d6ac89f4594fa4985145f08',
    ROOT / 'sgcm_data_audit/long_pair_review/interpretation_preregistered.json': 'af9726931762fde14c00684ed1004ce2a6464a46299e832f969cdce3c916a75d',
    HERE / 'final_review_S.json': 'ef4423f2a82adb830ef8366e0982607d662947330c98f871d8ffc93e743eeb89',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def local(path):
    assert path.startswith(REMOTE_ROOT + '/')
    return ROOT / path[len(REMOTE_ROOT) + 1:]


def computed_close(a, b):
    """1e-12 CPU aggregation roundoff only; never a first-step/quality tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(computed_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(computed_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed', action='store_true')
    args = parser.parse_args()
    assert all(sha(path) == expected for path, expected in PINS.items())
    if not args.completed:
        print(json.dumps({'status': 'CPU_READER_READY_WAIT_ACTUAL_S_ENDPOINT_AND_CANONICAL_EXPORT', 'source_pin_count': len(PINS), 'S_results_or_placeholders_written': False, 'GPU_used': False}))
        return
    status = read(S / 'status.json')
    assert status['status'] == 'complete' and status['actual_exit_code'] == 0 and status['gpu_lock_held'] is False
    effects_path = S / 'receipts/S_vs_originals_locked_metrics.json'
    effect = read(effects_path)
    assert effect['schema_version'] == 1 and effect['status'] == 'completed_S_vs_both_originals_locked_endpoint_metric_export'
    assert effect['endpoint_iteration'] == 32000 and effect['retained_horizon'] == 50000 and effect['observed_static_RGB_coefficient'] == 1.0
    assert effect['analysis_script']['sha256'] == PINS[ROOT / 'component_diagnostic/scripts/quantify_long_static_anchor_pair_cpu.py']
    assert effect['automatic_directional_pass_or_success_assigned'] is False and effect['normal_algorithm_success_claimed'] is False
    assert all(sha(local(pin['path'])) == pin['sha256'] and local(pin['path']).stat().st_size == pin['bytes'] for pin in effect['input_pins'])
    checks = []

    def check(label, condition):
        checks.append({'check': label, 'PASS': bool(condition)})
        if not condition:
            raise AssertionError(label)

    check('Actual canonical exporter/input hashes bind real completed32000 evidence, no automatic science acceptance', True)
    frames = {}
    step_rows = {}
    summaries = {}
    for arm, root in ARMS.items():
        s = read(root / 'status.json')
        report = read(root / 'continuation_diagnostic.json')
        summary = read(root / 'receipts/full49_summary.json')
        snapshot = read(root / 'terminal_32000/diagnostic_snapshot_manifest.json')
        final = read(root / 'extra_eval/32000/diagnostic_eval.json')
        check(arm + ' actual2000/exit0/complete restore/RNG49/4state receipts', s['status'] == 'complete' and s['actual_exit_code'] == int((root / 'actual_exit_code.txt').read_text()) == 0 and s['gpu_lock_held'] is False and report['step_count'] == 2000 and report['genuine_full_loss_backward_Adam'] is True and report['restore']['status'] == 'actual_restore_pass' and report['all_node_eval_rng_preserved'] is True and report['terminal_full49_evaluation'] is True and summary['all_protected_SHA_unchanged'] is True and set(snapshot['checkpoints']) == {'static', 'dynamic', 'environment', 'training_state'} and all(row['cpu_load_iteration_schema_validated'] is True and summary['checkpoint_validation'][key]['sha256'] == row['sha256'] for key, row in snapshot['checkpoints'].items()) and [f['frame_id'] for f in final['frames']] == [f'image_0/0017{i:03}' for i in range(49)] and final['all_rng_unchanged'] is True)
        frames[arm] = final['frames']
        step_rows[arm] = [json.loads(s) for s in (root / 'step_losses.jsonl').read_text().splitlines()]
        summaries[arm] = summary
    fields = ('iteration', 'camera_id', 'time_shift', 'learning_rates')
    trajectories = {arm: [{k: row[k] for k in fields} for row in rows] for arm, rows in step_rows.items()}
    check('Actual all2000 sampling/time_shift/LR identical for all3 and finite real loss scalars', trajectories['B1'] == trajectories['B2'] == trajectories['S'] and [r['iteration'] for r in trajectories['S']] == list(range(30001, 32001)) and all(math.isfinite(value) for rows in step_rows.values() for row in rows for value in row['losses'].values()))
    first = read(S / 'first_backward_reference_comparison.json')
    check('S original first10 exact before actual backward, no tolerance; original capture not new total', first['phase'] == 'before_actual_first_Tensor_backward' and first['original_backward_invoked'] is True and first['original_backward_returned'] is True and first['tolerance_applied'] is False and len(first['comparisons']) == 10 and all(row['exact_equal'] is True and row['abs_delta'] == 0 for row in first['comparisons'].values()) and first['added_weight'] == 1.0 and first['total_decomposition_exact'] is True)
    compose = read(S / 'first_original_compose_reference_comparison.json')
    check('S actual original10 exact after unchanged compose and before S executes', compose['added_static_RGB_executed'] is False and compose['original_return_tensor_preserved'] is True and len(compose['comparisons']) == 10 and all(r['exact_equal'] is True and r['abs_delta'] == 0 for r in compose['comparisons'].values()))
    check('Actual same49 IDs/counts and CPU float32-GT pairing to both original arms', all({k: r['pixel_count'] for k, r in f1['regions'].items()} == {k: r['pixel_count'] for k, r in f2['regions'].items()} == {k: r['pixel_count'] for k, r in fv['regions'].items()} for f1, f2, fv in zip(frames['B1'], frames['B2'], frames['S'])) and all(p['all2000_camera_time_shift_LR_exact'] is True and p['final49_actual_GT_dimensions_regioncounts']['frame_count'] == 49 and all(r['GT_float32_exact_equal'] is True and r['region_pixel_counts_exact_equal'] is True for r in p['final49_actual_GT_dimensions_regioncounts']['rows']) for p in summaries['S']['paired_originals'].values()))
    common = read(S / 'receipts/common49_inputs_after_S_cpu.json')
    b2_common = read(ARMS['B2'] / 'receipts/common49_inputs_after_B2_cpu.json')
    check('S actual original467 inputs/shadow absence identical and runtime snapshot limitations preserved', common['actual467_input_SHA_match_original'] is True and common['actual_input_SHA'] == b2_common['actual_input_SHA'] and len(common['actual_input_SHA']) == 467 and common['new_shadow_mask_absent'] is True)
    helper_path = ROOT / 'component_diagnostic/scripts/quantify_long_original_repeat_cpu.py'
    spec = importlib.util.spec_from_file_location('pinned_CPU_comparison_math_for_independent_check', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    noise = read(ARMS['B2'] / 'receipts/B1_B2_observed_noise.json')
    noise_lookup = {(r['region'], r['array'], r['metric']): r for r in noise['primary_metrics'] + noise['context_metrics_all_six_domains']}
    protocol = read(LONG / 'evaluation_protocol_locked.json')
    primary_keys = [(d['domain'], d['array'], field) for d in protocol['primary_metrics'] for field in d['metrics']]
    check('Only locked11 primary and same126 unique six-domain context rows, no frame selection', [(r['region'], r['array'], r['metric']) for r in effect['primary_metrics']] == primary_keys and len(primary_keys) == 11 and len(effect['context_metrics_all_six_domains']) == 126 and {(r['region'], r['array'], r['metric']) for r in effect['context_metrics_all_six_domains']} == {(r['region'], r['array'], r['metric']) for r in noise['context_metrics_all_six_domains']})
    for row in effect['primary_metrics'] + effect['context_metrics_all_six_domains']:
        key = row['region'], row['array'], row['metric']
        original = noise_lookup[key]
        assert row['equal_frame_mean_B1'] == original['equal_frame_mean_B1'] and row['equal_frame_mean_B2'] == original['equal_frame_mean_B2']
        assert row['N_observed_original_repeat_abs_difference_of_means'] == original['N_abs_difference_of_equal_frame_means']
        assert row['original_repeat_paired_absolute_difference_distribution'] == original['paired_absolute_difference_distribution']
        for arm in ('B1', 'B2'):
            actual = row['S_vs_each_original'][arm]
            expected = helper.compare_metric(frames[arm], frames['S'], *key, primary=row['preregistered_primary'])
            assert actual['reference_arm'] == arm and actual['candidate_arm'] == 'S'
            mappings = {'equal_frame_mean_reference': 'equal_frame_mean_B1', 'equal_frame_mean_S': 'equal_frame_mean_B2', 'equal_frame_mean_S_minus_reference': 'equal_frame_mean_B2_minus_B1', 'absolute_difference_of_equal_frame_means': 'N_abs_difference_of_equal_frame_means', 'paired_signed_difference_distribution': 'paired_signed_difference_distribution', 'paired_absolute_difference_distribution': 'paired_absolute_difference_distribution', 'included_nonempty_frame_count': 'included_nonempty_frame_count', 'empty_frames_excluded': 'empty_frames_excluded', 'S_lower_frame_count': 'paired_B2_lower_count', 'S_higher_frame_count': 'paired_B2_higher_count', 'exact_equal_frame_count': 'paired_exact_numeric_equal_count'}
            assert all(computed_close(actual[k], expected[v]) for k, v in mappings.items())
            expected_frames = [{'frame_id': r['frame_id'], 'pixel_count': r['pixel_count'], 'reference_value': r['B1'], 'S_value': r['B2'], 'S_minus_reference': r['B2_minus_B1'], 'abs_S_minus_reference': r['abs_B2_minus_B1']} for r in expected['frames']]
            assert actual['frames'] == expected_frames
            assert computed_close(row['equal_frame_mean_S'], expected['equal_frame_mean_B2'])
        values = (row['equal_frame_mean_S'], row['equal_frame_mean_B1'], row['equal_frame_mean_B2'])
        middle = values[0] - (values[1] + values[2]) / 2 if all(isinstance(x, (int, float)) for x in values) else None
        assert computed_close(row['mean_S_minus_mean_of_B1_B2'], middle)
    check('Every137 row S-B1/S-B2 actual49 values/signs/counts/distributions/means and midpoint match pinned canonical math and raw statistics', True)
    allrows = {(r['region'], r['array'], r['metric']): r for r in effect['primary_metrics'] + effect['context_metrics_all_six_domains']}
    tradeoffs = {}
    for region, array in (('full', 'composite'), ('actor', 'composite'), ('non_sky_outside_actor', 'static')):
        row = allrows[(region, array, 'clamped_rgb_mse_vs_gt')]
        a, b = row['S_vs_each_original']['B1']['frames'], row['S_vs_each_original']['B2']['frames']
        pairs = [{'frame_id': x['frame_id'], 'S_minus_B1_MSE': x['S_minus_reference'], 'S_minus_B2_MSE': y['S_minus_reference']} for x, y in zip(a, b)]
        tradeoffs[region + '/' + array] = {'higher_error_vs_B1_count': sum(x['S_minus_B1_MSE'] > 0 for x in pairs), 'higher_error_vs_B2_count': sum(x['S_minus_B2_MSE'] > 0 for x in pairs), 'higher_error_vs_both_count': sum(x['S_minus_B1_MSE'] > 0 and x['S_minus_B2_MSE'] > 0 for x in pairs), 'all49_signed_error_differences': pairs, 'top5_adverse_frames_by_max_signed_MSE_increase': sorted([x for x in pairs if max(x['S_minus_B1_MSE'], x['S_minus_B2_MSE']) > 0], key=lambda x: max(x['S_minus_B1_MSE'], x['S_minus_B2_MSE']), reverse=True)[:5], 'selection_disclosure': 'Endpoint-selected adverse examples, in addition to the preselected real raw0/10/17/20/24/30/40/48. They cannot replace complete49 statistics.'}
    check('4 actual RNG-safe nodes retain same5 camera subset; endpoint49 and training blocks remain distinct', [r['iteration'] for r in effect['recovery_nodes_actual_fixed5']] == [30001, 30300, 31000, 32000] and all([f['frame_id'] for f in node['three_arm_saved_metrics'][arm]['actual_fixed5_frame_metrics']] == [f'image_0/0017{i:03}' for i in (0, 10, 20, 30, 48)] for node in effect['recovery_nodes_actual_fixed5'] for arm in ARMS) and len(effect['S_actual_training_100step_blocks']) == 20)
    result = {'status': 'PASS_INDEPENDENT_ACTUAL_S_NUMERIC_EVIDENCE_REVIEW', 'scope': 'Numeric/input/runtime receipt verification only, not an automatic direction/quality PASS or complete paper reproduction.', 'checks': checks, 'reviewer_source_sha256': sha(Path(__file__)), 'canonical_three_arm_metrics_sha256': sha(effects_path), 'canonical_exporter_sha256': effect['analysis_script']['sha256'], 'canonical_original_repeat_noise_sha256': PINS[ARMS['B2'] / 'receipts/B1_B2_observed_noise.json'], 'locked_protocol_sha256': PINS[LONG / 'evaluation_protocol_locked.json'], 'frame_adverse_error_screen': tradeoffs, 'CPU_aggregation_roundoff_comparison': '1e-12 absolute for independently recomputed CPU means/distributions only; first10/source/input/frame values/counts remain exact. No performance margin introduced.', 'canonical_metrics_file_overwritten_or_duplicate_created': False, 'large_checkpoint_NPZ_input_hash_repeated_by_reader': False, 'actual_runtime_masks_or_Scene_pose_snapshots_available': False, 'real_image_qualitative_review_done_by_this_script': False, 'GPU_used': False, 'normal_algorithm_or_complete_reproduction_claimed': False}
    result['actor_area_and_existing_pooled_context'] = {'per_frame_pixel_counts': [{'frame_id': f['frame_id'], 'pixel_count': f['regions']['actor']['pixel_count']} for f in frames['S']], 'canonical_summary_actor_by_arm': {arm: summaries[arm]['final49_regions']['actor'] for arm in ARMS}, 'is_additional_primary_metric': False}
    path = HERE / 'independent_S_numeric_review.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'review': str(path), 'sha256': sha(path), 'checks': len(checks), 'tradeoffs': {k: {f: v[f] for f in ('higher_error_vs_B1_count', 'higher_error_vs_B2_count', 'higher_error_vs_both_count', 'top5_adverse_frames_by_max_signed_MSE_increase')} for k, v in tradeoffs.items()}}))


if __name__ == '__main__':
    main()

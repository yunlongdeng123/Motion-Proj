"""Verify completed long V and export true saved PNG/JSON; never start CUDA/training."""
import importlib.util
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
C = BASE / 'campaign/vgpu_bugfix_20261012'
D = C / 'component_diagnostic'
P = C / 'loss_audit/long_pair_30k_32000/materialized_plan'
V = C / 'continuation_diagnostic/opacity_outside30k_to32000'
ORIGINALS = {'B1': V.parent / 'original30k_to32000',
             'B2': V.parent / 'original30k_to32000_noise_reference'}
NODES = (30001, 30300, 31000, 32000)


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    helper = module_from_file('unchanged_CPU_truePNG_receipts', D / 'scripts/prepare_continuation_receipts_cpu.py')
    read, pin, sha, write = helper.read, helper.pin, helper.sha, helper.write
    status = read(V / 'status.json')
    launcher = read(P / 'opacity_outside_launcher_status.json')
    assert status['status'] == 'complete' and status['actual_exit_code'] == 0 and not status['gpu_lock_held']
    assert launcher['actual_exit_code'] == int((V / 'actual_exit_code.txt').read_text()) == 0
    report = read(V / 'continuation_diagnostic.json')
    assert report['step_count'] == 2000 and report['first_iteration'] == 30001 and report['stop_iteration'] == 32000
    assert report['horizon'] == 50000 and report['seed'] == 0 and report['all_node_eval_rng_preserved']
    assert report['terminal_full49_evaluation'] and report['all2000_camera_time_shift_LR_exactly_paired']
    assert report['original_algorithm_changed'] and not report['formal_original_source_changed'] and not report['formal50k_resumed']
    first = read(V / 'first_backward_reference_comparison.json')
    assert first['original_backward_invoked'] and first['original_backward_returned'] and not first['tolerance_applied']
    assert len(first['comparisons']) == 10 and all(row['exact_equal'] and row['abs_delta'] == 0 for row in first['comparisons'].values())
    assert first['added_weight'] == .05
    rows = [json.loads(line) for line in (V / 'step_losses.jsonl').read_text().splitlines()]
    assert [row['iteration'] for row in rows] == list(range(30001, 32001))
    assert all(math.isfinite(value) for row in rows for value in row['losses'].values())
    plan = read(V / 'diagnostic_plan.json')
    assert plan['training_config']['diagnostic_background_domain_complement'] is True
    assert plan['training_config']['diagnostic_lambda_dynamic_background_opacity'] == .05
    protected = {}
    for path, expected in plan['original_file_sha256'].items():
        assert sha(path) == expected, path
        protected[path] = {'sha256': expected, 'matches_before_training': True}
    snapshot = read(V / 'terminal_32000/diagnostic_snapshot_manifest.json')
    assert snapshot['iteration'] == 32000 and snapshot['not_a_formal_paper_checkpoint_event'] and len(snapshot['checkpoints']) == 4
    checkpoints = {}
    for branch, saved in snapshot['checkpoints'].items():
        actual = pin(saved['path'])
        assert actual['bytes'] == saved['bytes'] and actual['sha256'] == saved['sha256'] and saved['cpu_load_iteration_schema_validated'], branch
        checkpoints[branch] = actual | {'matches_saved_manifest': True, 'CPU_load_iteration_schema_validated_by_harness': True}
    final = read(V / 'extra_eval/32000/diagnostic_eval.json')
    assert [frame['frame_id'] for frame in final['frames']] == [f'image_0/0017{i:03}' for i in range(49)]
    assert final['all_rng_unchanged']
    paired = {}
    # The source helper is imported only for its pure CPU array verification function.
    variant = module_from_file('approved_long_variant_CPU_pair_function', P / 'continuation_variant_harness.py')
    for arm, out in ORIGINALS.items():
        reference = [json.loads(line) for line in (out / 'step_losses.jsonl').read_text().splitlines()]
        assert len(reference) == 2000
        assert all(actual['paired_original_camera_time_shift_LR_exact'] and
                   all(actual[key] == expected[key] for key in ('iteration', 'camera_id', 'time_shift', 'learning_rates'))
                   for actual, expected in zip(rows, reference)), arm
        paired[arm] = {'all2000_camera_time_shift_LR_exact': True,
                       'final49_actual_GT_dimensions_regioncounts': variant.verify_paired_evaluation(final, read(out / 'extra_eval/32000/diagnostic_eval.json'), np)}
    artifacts = []
    for frame in final['frames']:
        for channel in helper.CHANNELS:
            path = Path(frame['outputs'][channel])
            with Image.open(path) as im:
                assert im.size == (480, 320)
            artifacts.append(pin(path) | {'frame_id': frame['frame_id'], 'channel': channel})
        artifacts.append(pin(frame['float32_arrays']) | {'frame_id': frame['frame_id'], 'artifact': 'raw_float32_arrays'})
    # Recheck the whole original dataset without rewriting the frozen B1/B2 receipt.
    common_reference = read(ORIGINALS['B2'] / 'receipts/common49_inputs_after_B2_cpu.json')
    assert sha(ORIGINALS['B2'] / 'receipts/common49_inputs_after_B2_cpu.json') == 'dfd33893200805b4aebf147121b2be72270163a2ec61674e7704d2053de7f381'
    source = Path(plan['training_config']['source_path'])
    actual_inputs = {relative: sha(source / relative) for relative in common_reference['actual_input_SHA']}
    assert actual_inputs == common_reference['actual_input_SHA'] and len(actual_inputs) == 467
    assert not (source / 'dynamic_masks/shadow_mask_0').exists()
    receipts = V / 'receipts'
    receipts.mkdir(exist_ok=True)
    write(receipts / 'common49_inputs_after_V_cpu.json', {
        'status': 'PASS_common49_original_input_SHA_after_V', 'CPU_only': True,
        'B2_common_input_reference_sha256': sha(ORIGINALS['B2'] / 'receipts/common49_inputs_after_B2_cpu.json'),
        'actual467_input_SHA_match_original': True, 'actual_input_SHA': actual_inputs,
        'new_shadow_mask_absent': True,
        'runtime_GT_float32_and_regioncounts_pairing_to_both': paired,
        'evidence_limit': 'Same hashed original mask inputs and pinned frozen loader; actual GT arrays and region counts paired. Runtime masks and an independent actualScene pose snapshot are not stored in NPZ.'})
    lookup = {frame['frame_id']: frame for frame in final['frames']}
    b1_lookup = {frame['frame_id']: frame for frame in read(ORIGINALS['B1'] / 'extra_eval/32000/diagnostic_eval.json')['frames']}
    b2_lookup = {frame['frame_id']: frame for frame in read(ORIGINALS['B2'] / 'extra_eval/32000/diagnostic_eval.json')['frames']}
    contacts = [helper.sheet([(arm + '_32000', table['image_0/0017010']['outputs'])
                              for arm, table in (('B1', b1_lookup), ('B2', b2_lookup), ('V', lookup))], receipts / 'paired_raw10_32000.png'),
                helper.sheet([(frame_id, lookup[frame_id]['outputs']) for frame_id in helper.SELECTED], receipts / 'outside32000_fixed5.png')]
    recovery, raw10 = [], []
    for iteration in NODES:
        node = read(V / f'extra_eval/{iteration}/diagnostic_eval.json')
        fixed = [frame for frame in node['frames'] if frame['frame_id'] in helper.SELECTED]
        assert node['all_rng_unchanged'] and len(fixed) == 5
        raw10.append((f'V_{iteration}', next(frame['outputs'] for frame in fixed if frame['frame_id'] == 'image_0/0017010')))
        recovery.append({'iteration': iteration, 'all_rng_unchanged': True, 'actual_camera_count': len(node['frames']),
                         'fixed5_full': helper.aggregate(fixed, 'full'),
                         'fixed5_non_sky_outside_actor': helper.aggregate(fixed, 'non_sky_outside_actor'),
                         'model_statistics': node['model_statistics']})
    contacts.append(helper.sheet(raw10, receipts / 'raw10_long_recovery.png'))
    summary = {'schema_version': 1, 'status': 'complete_long_paired_variant_receipts_verified_CPU',
               'training_or_CUDA_started_by_this_script': False, 'parent_iteration': 30000,
               'first_iteration': 30001, 'stop_iteration': 32000, 'step_count': 2000, 'horizon': 50000,
               'hypothesis_weight': .05, 'scientific_status': plan['scientific_status'],
               'actual_first10_original_scalars_exact_before_backward': first,
               'paired_originals': paired, 'all4_node_eval_RNG_preserved': True,
               'all_protected_SHA_unchanged': True, 'protected_original_files': protected,
               'checkpoint_validation': checkpoints,
               'final49_regions': {region: helper.aggregate(final['frames'], region) for region in
                                   ('full', 'sky', 'non_sky', 'actor', 'outside_actor', 'non_sky_outside_actor')},
               'recovery_nodes_same_fixed5': recovery, 'final_model_statistics': final['model_statistics'],
               'contact_sheets': contacts, 'terminal49_png_count': 294, 'artifact_pins': artifacts,
               'GPU_peak_allocated_bytes': report['cuda_peak_allocated_bytes'],
               'GPU_peak_reserved_bytes': report['cuda_peak_reserved_bytes'], 'elapsed_seconds': report['elapsed_seconds'],
               'harness_exit_code': 0, 'wrapper_exit_code': 0, 'gpu_lock_released_by_harness': True,
               'formal_original_source_changed': False, 'normal_algorithm_success_claimed': False,
               'whole_debug_task_complete': False}
    write(receipts / 'full49_summary.json', summary)
    write(receipts / 'contact_sheet_source_validation.json', contacts)
    small = V / 'long_variant_small_receipts.tar.gz'
    names = ('status.json', 'actual_exit_code.txt', 'continuation_diagnostic.json', 'diagnostic_plan.json',
             'root_coordination_receipt.json', 'first_backward_reference_comparison.json',
             'first_step_reference_comparison.json', 'runtime_import_pins.json', 'step_losses.jsonl',
             'terminal_32000/diagnostic_snapshot_manifest.json', 'receipts/full49_summary.json',
             'receipts/common49_inputs_after_V_cpu.json', 'receipts/contact_sheet_source_validation.json',
             'receipts/paired_raw10_32000.png', 'receipts/outside32000_fixed5.png', 'receipts/raw10_long_recovery.png')
    paths = [V / name for name in names] + list((V / 'extra_eval').rglob('*.json')) + list((V / 'extra_eval').rglob('0017010.png'))
    with tarfile.open(small, 'w:gz', compresslevel=3) as tar:
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(V)))
        tar.add(P / 'opacity_outside_launcher_status.json', arcname='receipts/actual_launcher_status.json')
    full = V / 'long_variant_receipts_and_true49_png.tar.gz'
    with tarfile.open(full, 'w:gz', compresslevel=3) as tar:
        for path in V.rglob('*'):
            if path.is_file() and path.suffix not in ('.npz', '.pth', '.gz'):
                tar.add(path, arcname=str(path.relative_to(V)))
    print(json.dumps({'status': summary['status'], 'summary': pin(receipts / 'full49_summary.json'),
                      'common_input_receipt': pin(receipts / 'common49_inputs_after_V_cpu.json'),
                      'small_archive': pin(small), 'full_archive': pin(full),
                      'final49_full': summary['final49_regions']['full'],
                      'final49_non_sky_outside_actor': summary['final49_regions']['non_sky_outside_actor']}))


if __name__ == '__main__':
    main()

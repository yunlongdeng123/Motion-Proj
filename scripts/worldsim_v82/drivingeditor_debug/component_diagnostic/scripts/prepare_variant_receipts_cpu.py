"""CPU-only final actual-file verification and concise paired receipts."""
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
D = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic'
C = BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic/opacity_outside30k_to30300'
B = C.parent / 'original30k_to30300'
PLAN = D / 'continuation_plan'


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    helper = module_from_file('verified_CPU_actual_png_receipts', D / 'scripts/prepare_continuation_receipts_cpu.py')
    read, pin, sha, write = helper.read, helper.pin, helper.sha, helper.write
    status = read(C / 'status.json')
    launcher = read(PLAN / 'opacity_outside_launcher_status.json')
    if status['status'] != 'complete' or status['actual_exit_code'] != 0 or status['gpu_lock_held'] or launcher.get('actual_exit_code') != 0 or int((C / 'actual_exit_code.txt').read_text()) != 0:
        raise RuntimeError('Variant has not completely exited and released lock')
    report = read(C / 'continuation_diagnostic.json')
    if report['step_count'] != 300 or not report['all_node_eval_rng_preserved'] or not report['terminal_full49_evaluation'] or not report['all300_camera_time_shift_LR_exactly_paired'] or not report['original_algorithm_changed'] or report['formal_original_source_changed']:
        raise RuntimeError('Single-variable experiment final invariant failed')
    first = read(C / 'first_backward_reference_comparison.json')
    if not first['original_backward_returned'] or len(first['comparisons']) != 10 or not all(row['exact_equal'] for row in first['comparisons'].values()):
        raise RuntimeError('Real first10 pre-backward exact gate failed')
    rows = [json.loads(line) for line in (C / 'step_losses.jsonl').read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in (B / 'step_losses.jsonl').read_text().splitlines()]
    if [row['iteration'] for row in rows] != list(range(30001, 30301)) or len(baseline_rows) != 300:
        raise RuntimeError('Real300-step log incomplete')
    for actual, reference in zip(rows, baseline_rows):
        if not actual['paired_original_camera_time_shift_LR_exact'] or any(actual[key] != reference[key] for key in ('iteration', 'camera_id', 'time_shift', 'learning_rates')):
            raise RuntimeError('Actual paired300 camera/time_shift/LR sequence differs')
    checkpoints = {}
    for branch, saved in read(C / 'terminal_30300/diagnostic_snapshot_manifest.json')['checkpoints'].items():
        current = pin(saved['path'])
        if current['bytes'] != saved['bytes'] or current['sha256'] != saved['sha256'] or not saved['cpu_load_iteration_schema_validated']:
            raise RuntimeError('Terminal component hash/bytes/schema mismatch: ' + branch)
        checkpoints[branch] = current | {'matches_saved_manifest': True, 'CPU_load_iteration_schema_validated_by_harness': True}
    if len(checkpoints) != 4:
        raise RuntimeError('Terminal must contain all4 complete components/state')
    plan = read(C / 'diagnostic_plan.json')
    for path, expected in plan['original_file_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Protected original/reference changed: ' + path)
    final = read(C / 'extra_eval/30300/diagnostic_eval.json')
    baseline_final = read(B / 'extra_eval/30300/diagnostic_eval.json')
    variant = module_from_file('variant_CPU_pair_verification_only', PLAN / 'continuation_variant_harness.py')
    if [frame['frame_id'] for frame in final['frames']] != [f'image_0/0017{i:03}' for i in range(49)]:
        raise RuntimeError('Final actual49 camera list differs')
    paired49 = variant.verify_paired_evaluation(final, baseline_final, np)
    receipts = C / 'receipts'
    receipts.mkdir(exist_ok=True)
    lookup = {frame['frame_id']: frame for frame in final['frames']}
    baseline_lookup = {frame['frame_id']: frame for frame in baseline_final['frames']}
    contacts = [helper.sheet([('original30300', baseline_lookup['image_0/0017010']['outputs']), ('outside30300', lookup['image_0/0017010']['outputs'])], receipts / 'paired_raw10_30300.png'), helper.sheet([(frame_id, lookup[frame_id]['outputs']) for frame_id in helper.SELECTED], receipts / 'outside30300_fixed5.png')]
    summary = {'status': 'complete_paired_variant_receipts_verified_CPU', 'training_or_CUDA_started_by_this_script': False, 'parent_iteration': 30000, 'stop_iteration': 30300, 'step_count': 300, 'horizon': 50000, 'hypothesis_weight': 0.05, 'scientific_status': plan['scientific_status'], 'actual_first10_original_scalars_exact_before_backward': first, 'all300_camera_time_shift_LR_exactly_paired': True, 'all49_GT_float32_and_region_counts_exactly_paired': paired49, 'all6_node_eval_RNG_preserved': report['all_node_eval_rng_preserved'], 'all_protected_SHA_unchanged': True, 'checkpoint_validation': checkpoints, 'final49_regions': {region: helper.aggregate(final['frames'], region) for region in ('full', 'sky', 'non_sky', 'actor', 'outside_actor', 'non_sky_outside_actor')}, 'final_model_statistics': final['model_statistics'], 'contact_sheets': contacts, 'terminal49_png_count': 294, 'GPU_peak_allocated_bytes': report['cuda_peak_allocated_bytes'], 'GPU_peak_reserved_bytes': report['cuda_peak_reserved_bytes'], 'elapsed_seconds': report['elapsed_seconds'], 'harness_exit_code': 0, 'wrapper_exit_code': 0, 'gpu_lock_released_by_harness': True, 'normal_algorithm_success_claimed': False}
    write(receipts / 'full49_summary.json', summary)
    write(receipts / 'contact_sheet_source_validation.json', contacts)
    small_paths = [C / name for name in ('status.json', 'actual_exit_code.txt', 'continuation_diagnostic.json', 'diagnostic_plan.json', 'root_coordination_receipt.json', 'first_backward_reference_comparison.json', 'first_step_reference_comparison.json', 'runtime_import_pins.json', 'step_losses.jsonl', 'terminal_30300/diagnostic_snapshot_manifest.json', 'receipts/full49_summary.json', 'receipts/paired_raw10_30300.png', 'receipts/outside30300_fixed5.png', 'receipts/contact_sheet_source_validation.json')]
    small_paths += [path for path in (C / 'extra_eval').rglob('*.json')]
    small_paths += [path for path in (C / 'extra_eval').rglob('0017010.png')]
    small = C / 'variant_small_receipts.tar.gz'
    with tarfile.open(small, 'w:gz', compresslevel=3) as tar:
        for path in small_paths:
            tar.add(path, arcname=str(path.relative_to(C)))
        tar.add(PLAN / 'opacity_outside_launcher_status.json', arcname='receipts/actual_launcher_status.json')
    full = C / 'variant_receipts_and_true49_png.tar.gz'
    with tarfile.open(full, 'w:gz', compresslevel=3) as tar:
        for path in C.rglob('*'):
            if not path.is_file() or path.suffix in ('.npz', '.pth', '.gz'):
                continue
            tar.add(path, arcname=str(path.relative_to(C)))
    print(json.dumps({'status': summary['status'], 'summary': pin(receipts / 'full49_summary.json'), 'small_archive': pin(small), 'full_archive': pin(full), 'final49_full': summary['final49_regions']['full'], 'final49_non_sky_outside_actor': summary['final49_regions']['non_sky_outside_actor']}))


if __name__ == '__main__':
    main()

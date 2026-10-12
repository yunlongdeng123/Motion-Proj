"""CPU-only receipts for a completed approved B1 or B2 long original trajectory."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
CAMPAIGN = BASE / 'campaign/vgpu_bugfix_20261012'
D = CAMPAIGN / 'component_diagnostic'
PLAN = CAMPAIGN / 'loss_audit/long_pair_30k_32000/materialized_plan'
NODES = (30001, 30300, 31000, 32000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('B1', 'B2'), required=True)
    args = parser.parse_args()
    output_name = 'original30k_to32000' if args.arm == 'B1' else 'original30k_to32000_noise_reference'
    c = CAMPAIGN / 'continuation_diagnostic' / output_name
    plan_dir = PLAN if args.arm == 'B1' else PLAN / 'baseline_noise_reference'
    spec = importlib.util.spec_from_file_location('original_CPU_truePNG_receipt_helper', D / 'scripts/prepare_continuation_receipts_cpu.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    read, sha, pin, write = helper.read, helper.sha, helper.pin, helper.write
    status = read(c / 'status.json')
    launcher = read(plan_dir / 'continuation_launcher_status.json')
    if status['status'] != 'complete' or status['actual_exit_code'] != 0 or status['gpu_lock_held'] or launcher.get('actual_exit_code') != 0 or int((c / 'actual_exit_code.txt').read_text()) != 0:
        raise RuntimeError('Long original arm is not completed with actual exit0 and GPU lock release')
    report = read(c / 'continuation_diagnostic.json')
    if report['first_iteration'] != 30001 or report['stop_iteration'] != 32000 or report['horizon'] != 50000 or report['step_count'] != 2000 or report['original_algorithm_changed'] or not report['all_node_eval_rng_preserved'] or not report['terminal_full49_evaluation']:
        raise RuntimeError('Completed original long-window invariant failed')
    rows = [json.loads(line) for line in (c / 'step_losses.jsonl').read_text().splitlines()]
    if [row['iteration'] for row in rows] != list(range(30001, 32001)) or not all(math.isfinite(value) for row in rows for value in row['losses'].values()):
        raise RuntimeError('Exactly2000 finite genuine original steps required')
    first = read(c / 'first_step_reference_comparison.json')
    if len(first['comparisons']) != 10 or not all(row['exact_equal'] for row in first['comparisons'].values()):
        raise RuntimeError('First actual original10 comparison is not exact; preserve evidence and inform root')
    snapshot = read(c / 'terminal_32000/diagnostic_snapshot_manifest.json')
    if snapshot['iteration'] != 32000 or len(snapshot['checkpoints']) != 4 or not snapshot['not_a_formal_paper_checkpoint_event']:
        raise RuntimeError('Long terminal four-state manifest schema failed')
    checkpoint_validation = {}
    for branch, saved in snapshot['checkpoints'].items():
        actual = pin(saved['path'])
        if actual['bytes'] != saved['bytes'] or actual['sha256'] != saved['sha256'] or not saved['cpu_load_iteration_schema_validated']:
            raise RuntimeError('Long terminal snapshot hash/bytes/CPU schema failed: ' + branch)
        checkpoint_validation[branch] = actual | {'matches_saved_manifest': True, 'CPU_load_iteration_schema_validated_by_harness': True}
    plan = read(c / 'diagnostic_plan.json')
    protected = {}
    for path, expected in plan['original_file_sha256'].items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError('Protected original/reference input changed: ' + path)
        protected[path] = {'sha256': actual, 'matches_before_training': True}
    final = read(c / 'extra_eval/32000/diagnostic_eval.json')
    if [frame['frame_id'] for frame in final['frames']] != [f'image_0/0017{i:03}' for i in range(49)] or not final['all_rng_unchanged']:
        raise RuntimeError('Final original49 camera/RNG evaluation invariant failed')
    artifacts = []
    for frame in final['frames']:
        for channel in helper.CHANNELS:
            path = Path(frame['outputs'][channel])
            with Image.open(path) as im:
                if im.size != (480, 320):
                    raise RuntimeError('Actual native PNG dimensions differ')
            artifacts.append(pin(path) | {'frame_id': frame['frame_id'], 'channel': channel})
        artifacts.append(pin(frame['float32_arrays']) | {'frame_id': frame['frame_id'], 'artifact': 'raw_float32_arrays'})
    pair = {'B1_B2_complete_pair_validated': False, 'reason': 'Only B1 has completed at its first postprocess; no noise estimate is inferred.'}
    if args.arm == 'B2':
        b1 = c.parent / 'original30k_to32000'
        b1_rows = [json.loads(line) for line in (b1 / 'step_losses.jsonl').read_text().splitlines()]
        if len(b1_rows) != 2000 or any(any(actual[k] != reference[k] for k in ('iteration', 'camera_id', 'time_shift', 'learning_rates')) for actual, reference in zip(rows, b1_rows)):
            raise RuntimeError('B1/B2 actual2000 camera/time_shift/LR pairing failed')
        b1_final = read(b1 / 'extra_eval/32000/diagnostic_eval.json')
        domains = []
        for frame, reference in zip(final['frames'], b1_final['frames']):
            counts = {name: region['pixel_count'] for name, region in frame['regions'].items()}
            reference_counts = {name: region['pixel_count'] for name, region in reference['regions'].items()}
            with np.load(frame['float32_arrays'], allow_pickle=False) as actual_arrays, np.load(reference['float32_arrays'], allow_pickle=False) as reference_arrays:
                gt = actual_arrays['gt']
                ref_gt = reference_arrays['gt']
                if frame['frame_id'] != reference['frame_id'] or gt.dtype != np.float32 or ref_gt.dtype != np.float32 or gt.shape != (3, 320, 480) or not np.array_equal(gt, ref_gt) or counts != reference_counts:
                    raise RuntimeError('Actual B1/B2 final49 GT/region domain differs')
            domains.append({'frame_id': frame['frame_id'], 'GT_float32_exact': True, 'dimensions': [3, 320, 480], 'region_pixel_counts': counts})
        pair = {'B1_B2_complete_pair_validated': True, 'all2000_camera_time_shift_LR_exact': True, 'all49_GT_float32_dimensions_region_counts_exact': True, 'frames': domains, 'noise_reference_is_one_observed_repeat_difference_not_confidence_interval': True}
    receipts = c / 'receipts'
    receipts.mkdir(exist_ok=True)
    lookup = {frame['frame_id']: frame for frame in final['frames']}
    contacts = [helper.sheet([(fid, lookup[fid]['outputs']) for fid in helper.SELECTED], receipts / 'terminal32000_fixed5.png')]
    recovery = []
    raw10_rows = []
    for iteration in NODES:
        node = read(c / f'extra_eval/{iteration}/diagnostic_eval.json')
        fixed = [frame for frame in node['frames'] if frame['frame_id'] in helper.SELECTED]
        if not node['all_rng_unchanged'] or len(fixed) != 5:
            raise RuntimeError('Long recovery node fixed5/RNG invariant failed')
        raw10_rows.append((f'{args.arm}_{iteration}', next(frame for frame in node['frames'] if frame['frame_id'] == 'image_0/0017010')['outputs']))
        recovery.append({'iteration': iteration, 'all_rng_unchanged': True, 'actual_camera_count': len(node['frames']), 'fixed5_full': helper.aggregate(fixed, 'full'), 'fixed5_non_sky_outside_actor': helper.aggregate(fixed, 'non_sky_outside_actor'), 'model_statistics': node['model_statistics']})
    contacts.append(helper.sheet(raw10_rows, receipts / 'raw10_long_recovery.png'))
    summary = {'schema_version': 1, 'status': 'all_long_original_receipts_verified_CPU', 'arm': args.arm, 'training_or_CUDA_started_by_this_script': False, 'parent_iteration': 30000, 'first_iteration': 30001, 'stop_iteration': 32000, 'step_count': 2000, 'retained_horizon': 50000, 'original_algorithm_changed': False, 'first_step_original10_exact': True, 'all4_node_eval_RNG_preserved': True, 'final49_regions': {region: helper.aggregate(final['frames'], region) for region in ('full', 'sky', 'non_sky', 'actor', 'outside_actor', 'non_sky_outside_actor')}, 'recovery_nodes_same_fixed5': recovery, 'terminal_model_statistics': final['model_statistics'], 'checkpoint_validation': checkpoint_validation, 'protected_original_files': protected, 'all_protected_SHA_unchanged': True, 'paired_original_repeat_evidence': pair, 'harness_exit_code': 0, 'wrapper_exit_code': 0, 'gpu_lock_released_by_harness': True, 'CUDA_peak_allocated_bytes': report['cuda_peak_allocated_bytes'], 'CUDA_peak_reserved_bytes': report['cuda_peak_reserved_bytes'], 'elapsed_seconds': report['elapsed_seconds'], 'terminal49_png_count': 294, 'terminal49_float32_count': 49, 'contact_sheets': contacts, 'artifact_pins': artifacts, 'normal_algorithm_success_claimed': False}
    write(receipts / 'full49_summary.json', summary)
    write(receipts / 'contact_sheet_source_validation.json', contacts)
    small = c / 'long_original_small_receipts.tar.gz'
    small_paths = [c / name for name in ('status.json', 'actual_exit_code.txt', 'continuation_diagnostic.json', 'diagnostic_plan.json', 'root_coordination_receipt.json', 'first_step_reference_comparison.json', 'runtime_import_pins.json', 'step_losses.jsonl', 'terminal_32000/diagnostic_snapshot_manifest.json', 'receipts/full49_summary.json', 'receipts/contact_sheet_source_validation.json', 'receipts/terminal32000_fixed5.png', 'receipts/raw10_long_recovery.png')]
    small_paths += [path for path in (c / 'extra_eval').rglob('*.json')]
    small_paths += [path for path in (c / 'extra_eval').rglob('0017010.png')]
    with tarfile.open(small, 'w:gz', compresslevel=3) as tar:
        for path in small_paths:
            tar.add(path, arcname=str(path.relative_to(c)))
        tar.add(plan_dir / 'continuation_launcher_status.json', arcname='receipts/actual_launcher_status.json')
    full = c / 'long_original_receipts_and_true49_png.tar.gz'
    with tarfile.open(full, 'w:gz', compresslevel=3) as tar:
        for path in c.rglob('*'):
            if path.is_file() and path.suffix not in ('.npz', '.pth', '.gz'):
                tar.add(path, arcname=str(path.relative_to(c)))
    print(json.dumps({'status': summary['status'], 'arm': args.arm, 'summary': pin(receipts / 'full49_summary.json'), 'small_archive': pin(small), 'full_archive': pin(full), 'final49_full': summary['final49_regions']['full'], 'final49_non_sky_outside_actor': summary['final49_regions']['non_sky_outside_actor']}))


if __name__ == '__main__':
    main()

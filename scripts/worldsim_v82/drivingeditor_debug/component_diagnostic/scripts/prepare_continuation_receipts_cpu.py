"""CPU-only final receipt verification and actual-PNG contact sheets."""
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
D = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic'
C = BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic/original30k_to30300'
CHANNELS = ('gt', 'composite', 'static', 'dynamic', 'alpha_s', 'alpha_d')
SELECTED = [f'image_0/0017{i:03}' for i in (0, 10, 20, 30, 48)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def pin(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def aggregate(frames, region):
    total = sum(frame['regions'][region]['pixel_count'] for frame in frames)
    result = {'pixel_count_summed': total, 'frame_count': len(frames)}
    for channel in ('composite', 'static'):
        rows = [frame['regions'][region]['metrics'][channel] for frame in frames]
        mse = sum(frame['regions'][region]['pixel_count'] * row['clamped_rgb_mse_vs_gt'] for frame, row in zip(frames, rows)) / total
        result[channel] = {'mean_frame_psnr_db': sum(row['clamped_rgb_psnr_db'] for row in rows) / len(rows), 'pooled_mse': mse, 'pooled_psnr_db': -10 * math.log10(mse)}
    for channel in ('alpha_s', 'alpha_d', 'environment_sample', 'static_foreground_contribution', 'static_env_contribution', 'dynamic'):
        rows = [frame['regions'][region]['metrics'][channel] for frame in frames]
        fields = ('mean', 'fraction_lt0', 'fraction_gt1')
        result[channel] = {key: sum(frame['regions'][region]['pixel_count'] * row[key] for frame, row in zip(frames, rows)) / total for key in fields}
        if channel == 'dynamic':
            key = 'rgb_support_gt2_uint8_pixel_fraction'
            result[channel][key] = sum(frame['regions'][region]['pixel_count'] * row[key] for frame, row in zip(frames, rows)) / total
    return result


def sheet(rows, destination):
    """Paste source PNG cells at native480x320 with separate labels; verify cells."""
    width, height = 480, 320
    margin, label = 10, 30
    canvas = Image.new('RGB', (6 * (width + margin) + margin, len(rows) * (height + label + margin) + margin), '#15191d')
    draw = ImageDraw.Draw(canvas)
    entries = []
    for row_index, (row_name, outputs) in enumerate(rows):
        for col_index, channel in enumerate(CHANNELS):
            path = Path(outputs[channel])
            source = Image.open(path).convert('RGB')
            if source.size != (width, height):
                raise ValueError('Unexpected original PNG dimensions')
            left, top = margin + col_index * (width + margin), margin + row_index * (height + label + margin)
            draw.text((left, top + 7), row_name + ' / ' + channel, fill='white')
            canvas.paste(source, (left, top + label))
            box = (left, top + label, left + width, top + label + height)
            if not np.array_equal(np.asarray(source), np.asarray(canvas.crop(box))):
                raise ValueError('Contact sheet did not preserve true source PNG pixels')
            entries.append({'row': row_name, 'channel': channel, 'source': pin(path), 'box': box, 'exact_png_cell_pixels': True})
    canvas.save(destination)
    return {'output': pin(destination), 'no_resampling_gamma_or_replacement': True, 'cells': entries}


def main():
    status = read(C / 'status.json')
    launcher = read(D / 'continuation_plan/continuation_launcher_status.json')
    if status['status'] != 'complete' or status['actual_exit_code'] != 0 or status['gpu_lock_held']:
        raise RuntimeError('Baseline training has not completed and released GPU')
    if int((C / 'actual_exit_code.txt').read_text()) != 0 or launcher.get('actual_exit_code') != 0:
        raise RuntimeError('Harness or wrapper exit receipt failed')
    report = read(C / 'continuation_diagnostic.json')
    if report['step_count'] != 300 or not report['all_node_eval_rng_preserved'] or not report['terminal_full49_evaluation'] or report['original_algorithm_changed']:
        raise RuntimeError('Controlled baseline report invariant failed')
    rows = [json.loads(line) for line in (C / 'step_losses.jsonl').read_text().splitlines()]
    if [row['iteration'] for row in rows] != list(range(30001, 30301)):
        raise RuntimeError('Every genuine training step must be logged exactly once')
    snapshot = read(C / 'terminal_30300/diagnostic_snapshot_manifest.json')
    if snapshot['iteration'] != 30300 or len(snapshot['checkpoints']) != 4 or not snapshot['not_a_formal_paper_checkpoint_event']:
        raise RuntimeError('Terminal diagnostic snapshot schema mismatch')
    checkpoint_validation = {}
    for branch, saved in snapshot['checkpoints'].items():
        current = pin(saved['path'])
        if current['bytes'] != saved['bytes'] or current['sha256'] != saved['sha256'] or not saved['cpu_load_iteration_schema_validated']:
            raise RuntimeError('Terminal diagnostic checkpoint changed/truncated: ' + branch)
        checkpoint_validation[branch] = current | {'matches_saved_manifest': True, 'CPU_load_iteration_schema_validated_by_harness': True}
    plan = read(C / 'diagnostic_plan.json')
    protected = {}
    for path, expected in plan['original_file_sha256'].items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError('Protected original file changed: ' + path)
        protected[path] = {'sha256': actual, 'matches_before_training': True}
    final = read(C / 'extra_eval/30300/diagnostic_eval.json')
    expected_ids = [f'image_0/0017{i:03}' for i in range(49)]
    if [frame['frame_id'] for frame in final['frames']] != expected_ids or not final['all_rng_unchanged']:
        raise RuntimeError('Final evaluation must preserve RNG and cover all49 exact cameras')
    artifacts = []
    for frame in final['frames']:
        for channel in CHANNELS:
            path = Path(frame['outputs'][channel])
            with Image.open(path) as image:
                if image.size != (480, 320):
                    raise RuntimeError('Terminal PNG is not actual480x320')
            artifacts.append(pin(path) | {'frame_id': frame['frame_id'], 'channel': channel})
        artifacts.append(pin(frame['float32_arrays']) | {'frame_id': frame['frame_id'], 'artifact': 'raw_float32_arrays'})
    receipts = C / 'receipts'
    receipts.mkdir(exist_ok=True)
    lookup = {frame['frame_id']: frame for frame in final['frames']}
    contacts = [sheet([(frame_id, lookup[frame_id]['outputs']) for frame_id in SELECTED], receipts / 'terminal30300_fixed5.png')]
    comparison = []
    for case in ('baseline20', 'baseline30'):
        row = next(frame for frame in read(D / f'cases/{case}/case_manifest.json')['frames'] if frame['frame']['frame_id'] == 'image_0/0017010')
        comparison.append((case, row['outputs']))
    comparison.append(('original30300', lookup['image_0/0017010']['outputs']))
    contacts.append(sheet(comparison, receipts / 'raw10_20k_30k_30300.png'))
    trajectories = []
    for iteration in (30001, 30010, 30050, 30100, 30200, 30300):
        node = read(C / f'extra_eval/{iteration}/diagnostic_eval.json')
        fixed = [frame for frame in node['frames'] if frame['frame_id'] in SELECTED]
        trajectories.append({'iteration': iteration, 'all_rng_unchanged': node['all_rng_unchanged'], 'actual_eval_camera_count': len(node['frames']), 'model_statistics': node['model_statistics'], 'fixed5_full': aggregate(fixed, 'full'), 'fixed5_non_sky_outside_actor': aggregate(fixed, 'non_sky_outside_actor')})
    summary = {'schema_version': 1, 'status': 'all_controlled_baseline_receipts_verified_CPU', 'training_started_or_cuda_initialized_by_this_script': False, 'parent_iteration': 30000, 'stop_iteration': 30300, 'step_count': 300, 'retained_horizon': 50000, 'first_step_all_TB_values_exact': all(row['exact_equal'] for row in read(C / 'first_step_reference_comparison.json')['comparisons'].values()), 'training_PSNR_blocks': {f'{begin}..{begin+99}': sum(row['losses']['psnr'] for row in rows if begin <= row['iteration'] < begin + 100) / 100 for begin in (30001, 30101, 30201)}, 'final49_regions': {region: aggregate(final['frames'], region) for region in ('full', 'sky', 'non_sky', 'actor', 'outside_actor', 'non_sky_outside_actor')}, 'recovery_nodes_same_fixed5': trajectories, 'terminal_model_statistics': final['model_statistics'], 'checkpoint_validation': checkpoint_validation, 'protected_original_files': protected, 'all_protected_SHA_unchanged': True, 'harness_exit_code': 0, 'wrapper_exit_code': 0, 'gpu_lock_released_by_harness': True, 'CUDA_peak_allocated_bytes': report['cuda_peak_allocated_bytes'], 'CUDA_peak_reserved_bytes': report['cuda_peak_reserved_bytes'], 'elapsed_seconds': report['elapsed_seconds'], 'terminal49_png_count': 49 * 6, 'terminal49_float32_count': 49, 'contact_sheets': contacts, 'artifact_pins': artifacts, 'normal_algorithm_success_claimed': False}
    write(receipts / 'full49_summary.json', summary)
    write(receipts / 'contact_sheet_source_validation.json', contacts)
    archive = C / 'baseline_receipts_and_true49_png.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        for path in C.rglob('*'):
            if not path.is_file() or path == archive or 'terminal_30300' in path.parts and path.suffix == '.pth' or path.suffix == '.npz':
                continue
            tar.add(path, arcname=str(path.relative_to(C)))
    print(json.dumps({'status': summary['status'], 'summary': pin(receipts / 'full49_summary.json'), 'archive': pin(archive), 'final49_full': summary['final49_regions']['full'], 'final49_non_sky_outside_actor': summary['final49_regions']['non_sky_outside_actor'], 'training_PSNR_blocks': summary['training_PSNR_blocks']}))


if __name__ == '__main__':
    main()

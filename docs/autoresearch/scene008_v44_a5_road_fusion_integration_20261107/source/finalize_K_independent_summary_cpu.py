"""Final K CPU evidence digest and unchanged, fixed diagnostic NR image crops."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
C49 = ROOT.parent / 'scene008_v44_a5_c49_rgb_20261107'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def main():
    review = ROOT / 'evidence/review'
    files = [review / 'K_candidate_fixed4560_cpu.json', review / 'K_actual_pools_bins_cpu_v1.json',
             review / 'K_components_vs_H2_C2000_v1.json', review / 'K_components_vs_latest_v1.json',
             review / 'J_failure_pixels_v1.json', ROOT / 'evidence/K_LP_candidate/K_single_v1/result.json',
             ROOT / 'evidence/K_LP_candidate/receipt.json', ROOT / 'evidence/K_LP_candidate/K_execution_ledger_v1.json']
    gates, pools, h2_components, latest_components, failure_pixels, solve, native, ledger = map(read, files)
    source = {str(p): sha(p) for p in files}
    counts = Counter('unsupported' if g['passed'] is None else 'passed' if g['passed'] else 'failed'
                     for g in gates['numeric_gates'])
    failed = [g for g in gates['numeric_gates'] if g['passed'] is False]
    categories = []
    for name, count in sorted(Counter(g['name'] for g in failed).items()):
        rows = [g for g in failed if g['name'] == name]
        worst = max(rows, key=lambda g: g['value'])
        categories.append({'name': name, 'failed_count': count, 'threshold': worst['threshold'],
                           'maximum_failed_value': worst['value'], 'worst_raw': worst.get('raw'),
                           'worst_state': worst.get('state')})
    assert dict(counts) == {'passed': 55, 'failed': 31, 'unsupported': 8}
    assert len(pools['actual_pools']) == 6 and all(r['original_pass'] for r in pools['actual_pools'])
    assert pools['per_bin_original_pass_count'] == 106 and not pools['per_bin_original_failed']
    bins = pools['all106_bins']
    min_margin = min(bins, key=lambda row: row['margin_to_original'])
    reserve_worst = min(bins, key=lambda row: row['margin_to_tightened'])
    cases = [('Delivered21K', ROOT / 'evidence/baseline4'),
             ('H2 + C2000', C49 / 'evidence/H2_material2000_retry1'),
             ('J actual8F', ROOT / 'evidence/J_LP_candidate'),
             ('K actual8F', ROOT / 'evidence/K_LP_candidate')]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    board = Image.new('RGB', (1260, 650), '#f4f7f8')
    draw = ImageDraw.Draw(board)
    draw.text((15, 8), 'Fixed protectedNR diagnostic crops: latest21K / H2+C2000 / J / K', font=font, fill='#173b4a')
    draw.text((15, 31), 'Prior J worst-pixel locations; original source PNG crops with NEAREST display zoom. These are not new score masks.', font=small, fill='#415863')
    for col, (label, case) in enumerate(cases):
        draw.text((15 + col * 310, 57), label, font=font, fill='#173b4a')
    whitepoints = []
    for row, raw in enumerate([128, 137]):
        prior = next(r for r in failure_pixels['records'] if r['raw'] == raw and r['metric'] == 'protected_nonroad_error_increase')
        box = tuple(prior['display_crop_xyxy'])
        values = {}
        for label, case in cases[1:]:
            path = case / f'frames/{raw:03d}/delete_float.npz'
            source[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                values[label] = z['rgb'].copy()
        samples = []
        for point in prior['top20']:
            x, y = point['pixel_xy']
            gt = np.asarray(point['observed_RGB'], np.float64)
            h = values['H2 + C2000'][:, y, x]
            j = values['J actual8F'][:, y, x]
            k = values['K actual8F'][:, y, x]
            samples.append({'rank_in_preexisting_J_audit': point['rank'], 'pixel_xy': [x, y],
                            'source_instance_actor_id': point['source_instance_actor_id'],
                            'original_roles': point['original_roles'], 'observed_RGB': point['observed_RGB'],
                            'latest21K_RGB': point['latest21K_RGB'], 'H2_RGB': h.tolist(), 'J_RGB': j.tolist(), 'K_RGB': k.tolist(),
                            'K_equals_H2_float32': k.tobytes() == h.tobytes(), 'K_equals_J_float32': k.tobytes() == j.tobytes(),
                            'K_observed_RGB_error': float(abs(k.astype(np.float64) - gt).mean()),
                            'J_observed_RGB_error': point['J_observed_RGB_error'],
                            'latest21K_observed_RGB_error': point['latest21K_observed_RGB_error']})
        x, y = samples[0]['pixel_xy']
        whitepoints.append({'raw': raw, 'state': 'delete', 'fixed_display_crop_xyxy': list(box),
                            'selection_provenance': 'Prior J protected_nonroad maximum error-increase point and its original top20 set; not selected from K.',
                            'maximum_J_point': samples[0],
                            'prior_J_top20_K_equals_J_count': sum(p['K_equals_J_float32'] for p in samples),
                            'prior_J_top20_K_equals_H2_count': sum(p['K_equals_H2_float32'] for p in samples),
                            'fixed_top20_samples': samples,
                            'crop_RGB_K_minus_J_max_abs': float(abs(values['K actual8F'][:, box[1]:box[3], box[0]:box[2]] - values['J actual8F'][:, box[1]:box[3], box[0]:box[2]]).max()),
                            'crop_RGB_K_minus_H2_max_abs': float(abs(values['K actual8F'][:, box[1]:box[3], box[0]:box[2]] - values['H2 + C2000'][:, box[1]:box[3], box[0]:box[2]]).max())})
        top = 90 + row * 275
        for col, (label, case) in enumerate(cases):
            path = case / f'frames/{raw:03d}/delete.png'
            source[str(path)] = sha(path)
            with Image.open(path) as im:
                assert im.size == (480, 320)
                im.load()
                crop = im.convert('RGB').crop(box)
            crop = crop.resize((crop.width * 6, crop.height * 6), Image.Resampling.NEAREST)
            draw.text((15 + col * 310, top), f'raw{raw} xyxy={box}', font=small, fill='#173b4a')
            board.paste(crop, (15 + col * 310, top + 22))
    draw.text((15, 628), 'The bright artifacts at both fixed maximum-error coordinates are unchanged in H2/J/K. No whole-image identity claim.', font=small, fill='#415863')
    fig = ROOT / 'figures/K_protected_nonroad_fixed_crops.png'
    board.save(fig)
    same_geometry = all(value['bitwise_equal'] for record in h2_components['records']
                        for value in record['global_geometry_vs_initial'].values())
    foreground_max = max(value['RGB_pixel_max_abs_difference']['max'] or 0.0
                         for record in h2_components['records'] for value in record['foreground'].values())
    local_latest = [r for r in pools['per_bin_summary'] if r['reference'] == 'latest']
    assert native['actual_native_forwards'] == 8 and native['optimizer_updates'] == 0
    assert solve['actual_solver_calls'] == 1 and solve['float64_certificate_pass'] and solve['float32_certificate_pass']
    summary = {
        'schema': 'K_FINAL_INDEPENDENT_CPU_SUMMARY_V1',
        'status': 'SIX_POOL_AND_106_LOCAL_TARGETS_PASSED_FULL_ORIGINAL_QUALITY_FAILED',
        'raws': [94, 105, 128, 137], 'states': ['keep', 'delete'],
        'cost': {'new_native_forwards': 8, 'new_backwards': 0, 'new_CPU_LP_calls': 1,
                 'Adam_updates': 0, 'Gaussian_SH_updates': 0, 'new_operator_forwards': 0,
                 'J_operator_reused': True, 'palette_parent_rows_changed': solve['eligible_parent_rows_changed'],
                 'palette_max_absolute_RGB_change': solve['max_absolute_RGB_change']},
        'original4560': {'total_gates': 94, 'counts': dict(counts), 'failed_categories': categories,
                         'all31_failures_retained': True, 'gates_not_redefined': True},
        'six_original_pool_targets': {'pass_count': 6, 'records': pools['actual_pools']},
        '106_local_no_regression_targets': {'original_pass_count': 106, 'failed_count': 0,
             'formula': 'mean(abs(signed_RGB_residual)) <= min(current_H2_C2000_bin_score, latest21K_bin_score)+1e-6',
             'LP_reserve': 1e-7, 'minimum_original_margin': min_margin['margin_to_original'],
             'minimum_original_margin_bin': min_margin,
             'actual_bins_exceeding_extra_reserve': sum(not r['tightened_reserve_pass'] for r in bins),
             'worst_reserve_margin': reserve_worst['margin_to_tightened'],
             'reserve_is_not_the_original_quality_tolerance': True,
             'latest21K_strict_sign_and_1e_6_diagnostic_counts': local_latest},
        'same_H2_C2000_geometry_and_identity': {'all_saved_alpha_depth_8groups_semanticRGB_bitwise_equal': same_geometry,
             'fixed_model_foreground_RGB_max_abs_change': foreground_max,
             'fixed_model_foreground_newroad_mass_all_zero': all(r['foreground_union_newroad_mass']['zero_mass'] for r in h2_components['records']),
             'reference_limit': 'Relative to H2+C2000 only. H2 road geometry/effective visibility differ from delivered21K.'},
        'protectedNR_bright_artifact_diagnostic': {'conclusion': 'NO_REDUCTION_AT_TWO_FIXED_PRIOR_J_MAX_ERROR_COORDINATES',
             'maximum_coordinates_unchanged_vs_J_and_H2': all(r['maximum_J_point']['K_equals_H2_float32'] and r['maximum_J_point']['K_equals_J_float32'] for r in whitepoints),
             'records': whitepoints, 'figure': {'path': str(fig.relative_to(ROOT)).replace('\\', '/'),
                                             'sha256': sha(fig), 'size': list(board.size)}},
        'source_files': [{'path': path, 'sha256': digest} for path, digest in source.items()],
        'quality_accepted': False, 'agent_visual_verdict': 'ROOT_REVIEW_SEPARATE', 'human_verdict': None,
        'automatic49': False,
        'limits': ['Local tolerance passing permits strict micro-regressions below1e-6; no pixel-exact or all-bin-improved claim.',
                   'Pooled and local seam feasibility do not remove31 original photo/geometry failures.',
                   'ProtectedNR fixed crops are post-result diagnostics of existing failures, not a changed training or scoring domain.',
                   'Source O includes a37 and is not hidden-road DELETE ground truth. Label/source-group roles are recorded without semantic reassignment.']}
    assert all(sha(Path(path)) == digest for path, digest in source.items())
    out = ROOT / 'evidence/K_final_independent_summary.json'
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'summary': str(out), 'sha256': sha(out), 'figure': str(fig),
                      'original_counts': dict(counts), 'local_pass': 106,
                      'whitepoint_top20_unchanged_counts': [(r['raw'], r['prior_J_top20_K_equals_J_count'], r['prior_J_top20_K_equals_H2_count']) for r in whitepoints]}))


if __name__ == '__main__':
    main()

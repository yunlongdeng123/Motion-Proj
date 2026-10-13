"""Summarize copied terminal evidence using the local standard library only."""
from collections import Counter
import datetime
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

STAGE = Path(sys.argv[1]).resolve()
RAW = STAGE / 'original_snapshot'
GI_B = 2 ** 30
NPS_NODES = [1, 1000, 10000, 15000, 20000, 30000]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def read_lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write(name, data):
    with (STAGE / name).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def stats(values):
    values = [value for value in values if finite(value)]
    return {'finite_count': len(values), 'equal_frame_mean': math.fsum(values) / len(values) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None,
            'median': statistics.median(values) if values else None}

contract = read(STAGE / 'scene0029_10000_20000_30000_completion_contract.json')
config = read(RAW / 'effective_config.json')
spec = config['followup_scene_spec']
terminal_status = read(RAW / 'status.json')
memory = read(RAW / 'memory_summary.json')
phase_rows = read_lines(RAW / 'memory_phases.jsonl')
samples = read_lines(RAW / 'memory_samples.jsonl')
gradients = read_lines(RAW / 'gradient_diagnostics.jsonl')
training = read(RAW / 'sgcm_nps_training_rows.json')
assert contract['status'] == 'FOLLOWUP_COMPLETE_CONTRACT_PASS'
assert terminal_status['status'] == 'complete' and terminal_status['iteration'] == 30000
assert memory['monitor_status'] == 'closed' and memory['run_status'] == 'complete' and memory['iteration'] == 30000
assert not contract['owned_process_observation']['alive']
assert memory['error'] is None and memory['error_count'] == 0
assert training['all_30000_logged_numeric_losses_finite']
assert training['sampler_contiguous_1_through_30000']
assert training['all_30000_NPS_loss_finite_enabled_weight_0p01_scaling_pass']
assert not training['log_error_signature_hits']
assert set(training['sampler_unique_frame_ids']) == set(spec['frame_ids_with_camera'])

frame_rows = [read(path) for path in sorted((RAW / 'eval_30000/frame_metrics').glob('*.json'))]
assert [row['frame_id'] for row in frame_rows] == spec['frame_ids_with_camera']
assert len(frame_rows) == 49
event = read(RAW / 'iteration_030000.json')
event_by_id = {row['frame_id']: row for row in event['frames']}
regions = sorted(set().union(*(row['regions'] for row in frame_rows)))
region_summary = {}
all49_rows = []
for row in frame_rows:
    source = event_by_id[row['frame_id']]
    all49_rows.append({'frame_id': row['frame_id'], 'source_frame': row['frame_id'].split('/')[-1],
                       'camera': source['camera'], 'timestamp_model': source['timestamp'],
                       'source_image': source['source_image'], 'metric_json': source['hybrid_metrics'],
                       'raw_arrays': row['arrays'], 'regions': row['regions']})
for region in regions:
    objects = [row['regions'][region] for row in frame_rows]
    keys = sorted(set().union(*(obj for obj in objects)) - {'pixel_count'})
    nonempty = [index for index, obj in enumerate(objects) if obj['pixel_count'] > 0]
    empty = [frame_rows[index]['frame_id'] for index, obj in enumerate(objects) if obj['pixel_count'] == 0]
    metric_summary = {}
    for key in keys:
        values = [obj.get(key) for obj in objects]
        metric_summary[key] = {**stats(values), 'missing_or_nonfinite_count': 49 - sum(finite(value) for value in values)}
    region_summary[region] = {
        'all_retained_frames': 49, 'nonempty_region_frames': len(nonempty), 'empty_region_frames': empty,
        'pixel_count_all49': stats([obj['pixel_count'] for obj in objects]),
        'pixel_count_nonempty': stats([objects[index]['pixel_count'] for index in nonempty]),
        'metrics': metric_summary,
    }
global_metrics = read(RAW / 'eval_30000/metrics.json')
paper_metrics = {str(step): read(RAW / ('eval_%d/paper_region_metrics.json' % step)) for step in (10000, 20000, 30000)}
full_psnr_mean = region_summary['full']['metrics']['composite_psnr_db']['equal_frame_mean']
metric_delta = full_psnr_mean - global_metrics['psnr']
assert abs(metric_delta) < 1e-5
assert math.isclose(region_summary['actor']['metrics']['composite_psnr_db']['equal_frame_mean'],
                    paper_metrics['30000']['summary']['dynamic_psnr_star']['psnr_roi_mean'], abs_tol=1e-8)
worst_frames = {}
for region in ('full', 'actor', 'non_sky_outside_actor', 'actor_inner_boundary_5px', 'outside_boundary_5px'):
    ranked = [row for row in all49_rows if finite(row['regions'][region].get('composite_psnr_db'))]
    ranked.sort(key=lambda row: row['regions'][region]['composite_psnr_db'])
    worst_frames[region] = [{'frame_id': row['frame_id'], 'psnr_db': row['regions'][region]['composite_psnr_db'],
                            'pixel_count': row['regions'][region]['pixel_count']} for row in ranked[:5]]
quality = {
    'scene_id': spec['scene_id'], 'iteration': 30000, 'frame_count': 49,
    'evaluation_scope': '49 all-observed development reconstruction views; no independent test or edited GT',
    'RGB_metric_source': 'Copied existing hybrid_raw JSON: clamped RGB, float64 error from stored FP32; pixels not recomputed in this audit',
    'alpha_metric_source': 'Copied existing raw unmasked alpha metrics, no body clipping',
    'aggregation': 'Arithmetic means of available finite per-frame values; empty regions remain missing, never filled with zero or dropped from the 49-row artifact',
    'global_metrics_recorded': global_metrics,
    'full_PSNR_region_json_minus_global_metrics_db': metric_delta,
    'regions': region_summary, 'worst_existing_metric_frames': worst_frames,
    'paper_region_summary_recorded': {step: {k: v for k, v in obj.items() if k != 'frames'} for step, obj in paper_metrics.items()},
    'human_verdict': None,
}
write('scene0029_30000_all49_region_metrics.json', {'summary': quality, 'frames': all49_rows, 'human_verdict': None})

static_time = {}
for step in (10000, 20000, 30000):
    summary = read(RAW / ('static_time_%d_summary.json' % step))
    rows = summary['rows']
    assert [row['source_timestamp_frame'] for row in rows] == spec['frame_ids']
    assert len(rows) == 49 and summary['anchor_pose_frame'] == spec['anchor_frame']
    assert all(math.isclose(row['timestamp_model'], stamp, abs_tol=1e-12) for row, stamp in zip(rows, spec['model_timestamps']))
    keys = ('static_rgb_MAE_from_anchor', 'static_rgb_max_abs_from_anchor', 'static_alpha_MAE_from_anchor')
    metric_summaries = {key: stats([row[key] for row in rows]) for key in keys}
    for key in keys:
        maximum_row = max(rows, key=lambda row: row[key])
        metric_summaries[key]['maximum_source_timestamp_frame'] = maximum_row['source_timestamp_frame']
        metric_summaries[key]['maximum_timestamp_model'] = maximum_row['timestamp_model']
    assert all(rows[spec['anchor_index']][key] == 0 for key in keys)
    static_time[str(step)] = {
        'source_summary': str(summary_path := RAW / ('static_time_%d_summary.json' % step)),
        'source_remote': str(PurePosixPath(contract['output']) / ('static_time_%d/summary.json' % step)),
        'actual_saved_summary_metadata': {k: v for k, v in summary.items() if k != 'rows'},
        'fixed_pose': spec['anchor_frame'], 'camera': 'image_0', 'all49_model_times': [row['timestamp_model'] for row in rows],
        'all49_equal_frame_statistics': metric_summaries, 'rows': rows,
        'scope': 'Fixed rays/pose with 49 observed model timestamps; counterfactual pose-time combinations have no GT quality score',
        'unit': 'Model coordinates/time, not physical velocity or seconds',
    }
write('scene0029_fixed_pose_49times_summary.json', {'scene_id': spec['scene_id'], 'nodes': static_time, 'human_verdict': None})

assert len(gradients) == 6 and [row['iteration'] for row in gradients] == NPS_NODES
assert [row['iteration'] for row in training['NPS_loss_nodes']] == NPS_NODES
gradient_nodes = []
scale_failures = []
finite_failures = []
inputs = ('static_feature', 'static_alpha', 'alpha', 'dynamic_alpha', 'render', 'render_s', 'render_d')
for row, loss in zip(gradients, training['NPS_loss_nodes']):
    d = row['diagnostic']
    filtered = {key: value for key, value in d.items() if key.startswith('grad_hybrid/nps/')}
    for key, value in filtered.items():
        if key.endswith('_finite') and value is not True:
            finite_failures.append({'iteration': row['iteration'], 'field': key, 'value': value})
        if type(value) in (int, float) and not finite(value):
            finite_failures.append({'iteration': row['iteration'], 'field': key, 'value': str(value)})
    for target in inputs:
        prefix = 'grad_hybrid/nps/' + target + '/'
        for statistic in ('l2', 'abs_mean', 'abs_max'):
            raw = d[prefix + 'raw_' + statistic]
            weighted = d[prefix + 'weighted_' + statistic]
            if not math.isclose(weighted, raw * .01, rel_tol=1e-5, abs_tol=1e-8):
                scale_failures.append({'iteration': row['iteration'], 'input': target,
                                       'statistic': statistic, 'raw': raw, 'weighted': weighted})
    gradient_nodes.append({'iteration': row['iteration'], 'NPS_loss': loss, 'NPS_gradient': filtered,
                           'static_RGB_feature_raw_connected': d['grad_hybrid/static_rgb/static_feature/raw_connected'],
                           'static_RGB_alpha_raw_connected': d['grad_hybrid/static_rgb/static_alpha/raw_connected']})
assert not scale_failures and not finite_failures
init = training['SGCM_INIT'][0]
calls = training['SGCM_STATS']
assert training['SGCM_INIT_count'] == 1 and len(calls) == 145
assert [row['iteration'] for row in calls] == list(range(550, 15000, 100))
sgcm_failures = []
cfg = init['config']
for row in calls:
    tests = {
        'point_conservation': row['points_after'] == row['points_before'] - row['pruned'] + row['cloned'] + row['split_parents'],
        'guard_enabled': row['addition_guard_enabled'] is True,
        'all49_training_views': row['training_view_count'] == 49,
        'point_cap': row['points_after'] <= cfg['max_points'],
        'addition_cap': row['cloned'] + row['split_parents'] <= cfg['max_new_points'],
        'metric_radius': row['dr_m'] == cfg['dr_m'] and math.isclose(row['dr_model'], init['dr_model'], abs_tol=1e-12),
        'clone_candidate_accounting': row['candidate_clone'] == row['geometric_candidate_clone'] - row['excluded_clone_training_view_support'] - row['excluded_cloned_lineage'],
        'split_candidate_accounting': row['candidate_split'] == row['geometric_candidate_split'] - row['excluded_split_training_view_support'] - row['excluded_split_child_training_view_support'],
        'clone_action_bounded_by_candidates': row['cloned'] <= row['candidate_clone'],
        'split_action_bounded_by_candidates': row['split_parents'] <= row['candidate_split'],
    }
    sgcm_failures.extend({'iteration': row['iteration'], 'field': name} for name, passed in tests.items() if not passed)
assert not sgcm_failures
counter_keys = ('inside_region', 'pruned', 'cloned', 'split_parents', 'candidate_clone', 'candidate_split',
                'geometric_candidate_clone', 'geometric_candidate_split', 'excluded_clone_training_view_support',
                'excluded_split_training_view_support', 'excluded_split_child_training_view_support',
                'excluded_cloned_lineage', 'excluded_duplicate_lineage_in_step')
sgcm = {
    'status': 'PASS_LOGGED_SGCM_ACCOUNTING', 'calls': len(calls), 'schedule': {'first': 550, 'last': 14950, 'step': 100},
    'actual_config': cfg, 'meter_to_model_scale': init['meter_to_model_scale'], 'dr_model': init['dr_model'],
    'ground_fit': init['ground_fit'], 'addition_guard': {k: v for k, v in init['addition_guard'].items() if k != 'training_view_ids'},
    'exact49_training_view_id_set_matches_scene_source': set(init['addition_guard']['training_view_ids']) == {str(PurePosixPath(spec['source_path']) / ('image/image_0/' + name + '.png')) for name in spec['frame_ids']},
    'all_point_conservation_guard_cap_candidate_checks_pass': not sgcm_failures,
    'total_per_call_counters': {key: sum(row.get(key, 0) for row in calls) for key in counter_keys},
    'counter_present_call_counts': {key: sum(key in row for row in calls) for key in counter_keys},
    'counter_total_semantics': 'Totals sum recorded per-call counters; an absent optional same-step duplicate field adds no observed count and is listed as missing, not independently certified zero.',
    'per_call_counter_ranges': {key: {'min': min(row.get(key, 0) for row in calls), 'max': max(row.get(key, 0) for row in calls)} for key in counter_keys},
    'same_step_duplicate_counter_present_calls': sum('excluded_duplicate_lineage_in_step' in row for row in calls),
    'same_step_duplicate_counter_missing_calls': sum('excluded_duplicate_lineage_in_step' not in row for row in calls),
    'maximum_additions_per_call': max(row['cloned'] + row['split_parents'] for row in calls),
    'maximum_net_delta_per_call': max(row['points_after'] - row['points_before'] for row in calls),
    'minimum_headroom_to_point_cap': min(cfg['max_points'] - row['points_after'] for row in calls),
    'logged_SGCM_net_topology_delta': sum(row['cloned'] + row['split_parents'] - row['pruned'] for row in calls),
    'first_call': calls[0], 'last_call': calls[-1], 'failure_fields': sgcm_failures,
    'scope': 'Counts sum per-call observations, not unique points. Center-frustum/temporal-alpha support is a proxy; occlusion and exact per-Gaussian contribution are unmeasured. Logged counts do not independently prove every historical lineage identity.',
}
assert sgcm['exact49_training_view_id_set_matches_scene_source']
nps_sgcm = {
    'scene_id': spec['scene_id'], 'arm': 'GN11', 'NPS': {
        'diagnostic_node_count': 6, 'nodes': gradient_nodes,
        'all_logged_NPS_gradients_finite': not finite_failures,
        'all_logged_raw_to_weighted_gradients_scale_0p01': not scale_failures,
        'all30000_logged_numeric_losses_finite': training['all_30000_logged_numeric_losses_finite'],
        'logged_numeric_loss_fields': training['logged_numeric_loss_fields'],
        'all30000_NPS_losses_finite_enabled_scaled_0p01': training['all_30000_NPS_loss_finite_enabled_weight_0p01_scaling_pass'],
        'static_feature_to_static_RGB_ratios_valid': [row['NPS_gradient']['grad_hybrid/nps/static_feature/ratio_to_static_rgb_valid'] for row in gradient_nodes],
        'boundary': 'Static-feature/static-RGB ratios are invalid where static-RGB is disconnected; recorded zero is not a gradient balance measurement. No complete Gaussian parameter-gradient comparison is certified.',
    }, 'SGCM': sgcm, 'sampler': {
        'rows': training['sampler_rows'], 'contiguous_1_to_30000': training['sampler_contiguous_1_through_30000'],
        'unique_frame_count': len(training['sampler_unique_frame_ids']),
        'scope': 'Single-scene configured sampler evidence; no between-arm paired sampler claim',
    }, 'log_error_signature_hits': training['log_error_signature_hits'], 'human_verdict': None,
}
write('scene0029_NPS6_SGCM145_accounting_summary.json', nps_sgcm)

assert len(samples) == memory['sampling']['samples'] and len(samples) > 0
assert len(phase_rows) == sum(obj['count'] for obj in memory['phases'].values())
assert all(row['status'] == 'complete' and row.get('error') is None for row in phase_rows)
assert all(not row['errors'] for row in samples)
assert max(row['process_used_bytes'] for row in samples) == memory['nvidia_smi']['process_peak_used_bytes']
assert max(row['device_used_bytes'] for row in samples) == memory['nvidia_smi']['device_peak_used_bytes']
peak_fields = ('peak_allocated_bytes', 'peak_reserved_bytes', 'process_peak_used_bytes', 'device_peak_used_bytes')
all_phase_aggregates = {}
for name, phase in memory['phases'].items():
    rows = [row for row in phase_rows if row['name'] == name]
    normalized = dict(phase)
    for field in peak_fields:
        available = [row[field] for row in rows if field in row]
        expected = max(available) if available else None
        assert phase.get(field) == expected
        normalized[field] = expected
    normalized['peaks_GiB'] = {field.removesuffix('_bytes'): normalized[field] / GI_B if normalized[field] is not None else None for field in peak_fields}
    normalized['nvidia_samples_unavailable'] = phase['sample_count'] == 0
    all_phase_aggregates[name] = normalized
process_peak_samples = [row for row in samples if row['process_used_bytes'] == memory['nvidia_smi']['process_peak_used_bytes']]
device_peak_samples = [row for row in samples if row['device_used_bytes'] == memory['nvidia_smi']['device_peak_used_bytes']]
all_stage_speed = 30000 / terminal_status['wall_seconds']
resources = {
    'scene_id': spec['scene_id'], 'arm': 'GN11', 'terminal_status': terminal_status['status'],
    'iteration': 30000, 'pid': terminal_status['pid'], 'owned_process_alive_at_audit': False,
    'terminal_wall_seconds': terminal_status['wall_seconds'], 'terminal_wall_minutes': terminal_status['wall_seconds'] / 60,
    'memory_monitor_closed_wall_seconds': memory['wall_seconds'],
    'all_stage_wall_iterations_per_second': all_stage_speed,
    'all_stage_speed_includes': ['initialization', 'training', 'gradient diagnostics', 'checkpoint', 'photo/LPIPS eval', 'fixed-pose static time eval'],
    'training_exclusive_phase_iterations_per_second': 30000 / memory['phases']['training']['wall_seconds_exclusive'],
    'training_exclusive_phase_speed_definition': '30k iterations divided by recorded training host phase wall, subtracting recorded nested child-phase wall',
    'last100_speed_window': terminal_status['speed_window'],
    'last_iter_forward_backward_ms': terminal_status['iter_forward_backward_ms'],
    'final_point_counts': {'static': memory['extra']['render_static_points'], 'dynamic': memory['extra']['render_dynamic_points']},
    'overall_pytorch': memory['pytorch'], 'overall_recorded_nvidia_smi': memory['nvidia_smi'],
    'overall_peaks_GiB': {
        'allocated': memory['pytorch']['peak_allocated_bytes'] / GI_B,
        'reserved': memory['pytorch']['peak_reserved_bytes'] / GI_B,
        'PID_sampled_used': memory['nvidia_smi']['process_peak_used_bytes'] / GI_B,
        'device_sampled_used': memory['nvidia_smi']['device_peak_used_bytes'] / GI_B,
    },
    'phases_all_aggregate': all_phase_aggregates, 'all_phase_instances_count': len(phase_rows),
    'all_phase_instances': phase_rows, 'sampling': memory['sampling'], 'phase_attribution': memory['phase_attribution'],
    'sampling_recomputed_from_single_copied_source': {
        'rows': len(samples), 'all_sampling_errors_absent': True,
        'PID_presence_sample_count': sum(row['process_present'] for row in samples),
        'PID_sampled_peak_matches_summary': True, 'device_sampled_peak_matches_summary': True,
        'all_phase_allocated_reserved_PID_device_peaks_match_summary': True,
        'all_phase_instances_complete_without_error': True,
        'PID_peak_samples': process_peak_samples, 'device_peak_samples': device_peak_samples,
    },
    'memory_summary_error_count': memory['error_count'], 'memory_summary_error': memory['error'],
    'rtx3090_assessment_recorded': memory['rtx3090_assessment'],
    'capacity_interpretation': 'Complete single-run observations within preliminary 22 GiB budget. Sampled PID/device peaks are lower bounds, allocator peaks exclude non-PyTorch CUDA memory. Target RTX3090 fit or speed is not certified.',
    'authoritative_terminal_memory_source': str(RAW / 'memory_summary.json'),
    'embedded_status_memory_note': 'status.json memory field is a pre-close running snapshot; the separately copied closed memory_summary.json supplies terminal phase counts and monitor status',
    'source_copy_count': {'status.json': 1, 'memory_summary.json': 1, 'memory_phases.jsonl': 1, 'memory_samples.jsonl': 1, 'gradient_diagnostics.jsonl': 1},
    'source_scope': 'Existing terminal monitor logs only; audit made no nvidia-smi query or GPU call',
    'human_verdict': None,
}
write('scene0029_terminal_resources_summary.json', resources)

copied_manifest = [{'local_relative_path': str(path.relative_to(STAGE)).replace('\\', '/'),
                    'stat_bytes': path.stat().st_size,
                    'local_mtime_utc': datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat(),
                    'new_hash_computed': False}
                   for path in sorted(RAW.rglob('*')) if path.is_file()]
summary = {
    'schema_version': 1, 'status': 'SCENE0029_GN11_TERMINAL_READONLY_AUDIT_PASS',
    'scene_id': spec['scene_id'], 'arm': 'GN11', 'scheduled_check': '2026-10-13 final red terminal after shared lock',
    'compiled_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'contract_audit_observed_singapore': contract['observed_singapore'],
    'original_completion_contract_result': contract['original_completion_contract_actual_result'],
    'completion_contract_receipt': str(STAGE / 'scene0029_10000_20000_30000_completion_contract.json'),
    'terminal': {key: resources[key] for key in ('terminal_status', 'iteration', 'pid', 'owned_process_alive_at_audit',
                                               'terminal_wall_seconds', 'terminal_wall_minutes', 'all_stage_wall_iterations_per_second',
                                               'training_exclusive_phase_iterations_per_second', 'last100_speed_window',
                                               'last_iter_forward_backward_ms', 'final_point_counts', 'overall_peaks_GiB')},
    'regional_metrics': quality,
    'fixed_pose_30000_equal49_statistics': static_time['30000']['all49_equal_frame_statistics'],
    'NPS': {key: nps_sgcm['NPS'][key] for key in ('diagnostic_node_count', 'all_logged_NPS_gradients_finite',
                                              'all_logged_raw_to_weighted_gradients_scale_0p01', 'all30000_NPS_losses_finite_enabled_scaled_0p01',
                                              'static_feature_to_static_RGB_ratios_valid', 'boundary')},
    'SGCM': sgcm,
    'resources_summary_file': str(STAGE / 'scene0029_terminal_resources_summary.json'),
    'all49_region_metrics_file': str(STAGE / 'scene0029_30000_all49_region_metrics.json'),
    'fixed_pose_49times_summary_file': str(STAGE / 'scene0029_fixed_pose_49times_summary.json'),
    'NPS6_SGCM145_summary_file': str(STAGE / 'scene0029_NPS6_SGCM145_accounting_summary.json'),
    'original_snapshot_manifest': copied_manifest,
    'restoration_verified': False, 'actual_model_Adam_RNG_restore_verified': False,
    'human_verdict': None,
    'audit_boundaries': {'GPU_used': False, 'Torch_imported': False, 'CUDA_initialized': False,
                         'models_or_checkpoint_payload_loaded': False, 'raw_pixel_arrays_loaded': False,
                         'rerender_or_training_started': False, 'continuous_progress_polling': False,
                         'new_hash_or_fingerprint_computed': False, 'remote_training_assets_or_repository_modified': False, 'new_report_metadata_written': True,
                         'red_mask_queue_touched': False},
}
write('scene0029_terminal_summary.json', summary)
print(json.dumps({
    'status': summary['status'], 'summary_file': str(STAGE / 'scene0029_terminal_summary.json'),
    'terminal': summary['terminal'], 'global_metrics': global_metrics,
    'region_finite_composite_PSNR': {region: {'nonempty_frames': obj['nonempty_region_frames'],
                                             **obj['metrics'].get('composite_psnr_db', {})} for region, obj in region_summary.items()},
    'fixed_pose_30000': summary['fixed_pose_30000_equal49_statistics'],
    'SGCM_counter_totals': sgcm['total_per_call_counters'],
    'SGCM_net_topology_delta': sgcm['logged_SGCM_net_topology_delta'],
    'phase_instance_count': len(phase_rows), 'sample_count': len(samples),
    'human_verdict': None,
}, ensure_ascii=False, indent=2, allow_nan=False))

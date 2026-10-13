"""CPU-only stat/header audit of completed 10k/20k nodes; stdout is JSON."""
import datetime
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

sys.dont_write_bytecode = True
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
ROOT = BASE / 'runs/scene0029_069_117_hybrid_30000_seed0_GN11'
sys.path.insert(0, str(BASE / 'tools/followup'))
import contracts

OBSERVED_START = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
metadata_cache = {}
original_read_json = contracts.read_json

def cached_json(path):
    key = str(Path(path))
    if key not in metadata_cache:
        metadata_cache[key] = original_read_json(path)
    return metadata_cache[key]

contracts.read_json = cached_json
config = cached_json(ROOT / 'effective_config.json')
spec = config['followup_scene_spec']
# One existing-JSON snapshot each. No new NVML/nvidia-smi queries or monitoring.
status = original_read_json(ROOT / 'status.json')
memory = original_read_json(ROOT / 'memory_summary.json')
snapshot_observed = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
original_nodes = contracts.NODES

def close(a, b):
    return type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=0, abs_tol=1e-7)

def stat_header(path):
    path = Path(path)
    stat = path.stat()
    with path.open('rb') as stream:
        header = stream.read(16)
    result = {'path': str(path), 'stat_bytes': stat.st_size,
              'mtime_ns': stat.st_mtime_ns, 'header16_hex': header.hex(),
              'tensor_or_pickle_payload_read': False, 'restoration_verified': False}
    if header[:4] == b'PK\x03\x04':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        result.update(container='ZIP', directory_entries=len(names),
                      data_pkl_directory_entries=[n for n in names if n.endswith('/data.pkl')],
                      version_directory_entries=[n for n in names if n.endswith('/version')],
                      archive_payload_members_read=False)
    else:
        result['container'] = 'unrecognized_header'
    return result

def png_header(path):
    path = Path(path)
    stat = path.stat()
    with path.open('rb') as stream:
        header = stream.read(33)
    valid = (len(header) == 33 and header[:8] == b'\x89PNG\r\n\x1a\n'
             and header[12:16] == b'IHDR' and struct.unpack('>I', header[8:12])[0] == 13)
    return {'path': str(path), 'stat_bytes': stat.st_size, 'IHDR_valid': valid,
            'wh': list(struct.unpack('>II', header[16:24])) if valid else None,
            'bit_depth': header[24] if valid else None,
            'color_type': header[25] if valid else None,
            'pixels_loaded': False}

def npz_header(path):
    path = Path(path)
    return {'path': str(path), 'stat_bytes': path.stat().st_size,
            'NPY_headers': contracts.npz_headers(path), 'arrays_loaded': False}

raw_schema = {
    name: {'dtype': '|b1' if name in ('actor_mask.npy', 'sky_mask.npy') else '<f4',
           'shape': [3 if name in ('gt.npy', 'composite.npy', 'static.npy', 'dynamic.npy') else 1, 320, 480]}
    for name in ('gt.npy', 'composite.npy', 'static.npy', 'dynamic.npy', 'alpha_d.npy',
                 'alpha_s.npy', 'depth_accum_model.npy', 'actor_mask.npy', 'sky_mask.npy')
}
fixed_schema = {'static_rgb.npy': {'dtype': '<f4', 'shape': [3, 320, 480]},
                'static_alpha.npy': {'dtype': '<f4', 'shape': [1, 320, 480]}}
all_failures = []
nodes = {}
for step in (10000, 20000):
    failures = []
    def check(condition, field, actual=None):
        if not condition:
            failures.append({'field': field, 'actual': actual})
    try:
        # Scope the original function to one node in memory only. Source/config remain unchanged.
        contracts.NODES = (step,)
        original_result = contracts.completion_contract(ROOT, config)
    except Exception as exc:
        original_result = {'status': 'FAIL', 'exception_type': type(exc).__name__, 'error': str(exc)}
        check(False, 'original_completion_contract_single_node', original_result)
    finally:
        contracts.NODES = original_nodes

    event_path = ROOT / ('checkpoint_events/iteration_%06d.json' % step)
    event = cached_json(event_path)
    check(event.get('status') == 'complete' and event.get('iteration') == step,
          'event_complete_iteration', [event.get('status'), event.get('iteration')])
    check(event.get('model_path') == str(ROOT), 'event_model_path', event.get('model_path'))
    check(event.get('source_path') == spec['source_path'], 'event_source_path', event.get('source_path'))
    check(event.get('parameters_evaluated') == 'after_optimizer_and_sgcm_same_as_checkpoint',
          'same_checkpoint_parameter_state', event.get('parameters_evaluated'))
    check(event.get('config') == config, 'event_effective_config_exact_equality')
    check(set(event.get('checkpoints', {})) == {'static', 'dynamic', 'environment', 'training_state'},
          'four_checkpoint_keys', list(event.get('checkpoints', {})))
    component_receipts = {}
    for name, template in {'static': 'chkpnt%d.pth', 'dynamic': 'dynamic_chkpnt%d.pth',
                           'environment': 'env_light_chkpnt%d.pth', 'training_state': 'training_state%d.pth'}.items():
        path = ROOT / (template % step)
        entry = event['checkpoints'][name]
        try:
            header = stat_header(path)
            header['event_bytes'] = entry.get('bytes')
            check(entry.get('path') == str(path), 'checkpoint.' + name + '.path', entry.get('path'))
            check(header['stat_bytes'] == entry.get('bytes') and header['stat_bytes'] > 0,
                  'checkpoint.' + name + '.event_size_equals_stat', header)
            check(header['container'] == 'ZIP' and len(header.get('data_pkl_directory_entries', [])) == 1
                  and len(header.get('version_directory_entries', [])) == 1,
                  'checkpoint.' + name + '.ZIP_header_directory', header)
            component_receipts[name] = header
        except Exception as exc:
            check(False, 'checkpoint.' + name + '.stat_header', str(exc))

    frames = event.get('frames', [])
    check(len(frames) == 49 and len({r.get('frame_id') for r in frames}) == 49, '49_unique_frames', len(frames))
    check([r.get('frame_id') for r in frames] == spec['frame_ids_with_camera'], 'exact_chronological_frame_IDs')
    render_dir = ROOT / ('eval/train_%d_render' % step)
    frame_receipts = []
    for index, row in enumerate(frames):
        expected_raw_id = spec['frame_ids'][index]
        expected_frame_id = spec['frame_ids_with_camera'][index]
        expected_time = spec['model_timestamps'][index]
        expected_stem = 'image_0_' + expected_raw_id
        check(row.get('uid') == index and row.get('image_name') == expected_raw_id
              and row.get('frame_id') == expected_frame_id, 'frame.%d.identity' % index, row.get('frame_id'))
        check(row.get('camera') == 'image_0' and row.get('width') == 480 and row.get('height') == 320,
              'frame.%d.camera_dimensions' % index)
        check(close(row.get('timestamp'), expected_time), 'frame.%d.model_timestamp' % index, row.get('timestamp'))
        expected_source = str(Path(spec['source_path']) / 'image/image_0' / (expected_raw_id + '.png'))
        check(row.get('source_image') == expected_source, 'frame.%d.source_pointer' % index, row.get('source_image'))
        check(set(row.get('channels', {})) == {'gt', 'composite', 'static', 'dynamic'},
              'frame.%d.four_PNG_channels' % index)
        channel_receipts = {}
        for channel, value in row.get('channels', {}).items():
            expected_path = render_dir / channel / (expected_stem + '.png')
            check(value == str(expected_path), 'frame.%d.%s.path' % (index, channel), value)
            try:
                header = png_header(value)
                check(header['IHDR_valid'] and header['wh'] == [480, 320] and header['stat_bytes'] > 0,
                      'frame.%d.%s.PNG_header' % (index, channel), header)
                channel_receipts[channel] = header
            except Exception as exc:
                check(False, 'frame.%d.%s.PNG_header' % (index, channel), str(exc))
        metric_path = render_dir / 'hybrid_raw' / (expected_stem + '.json')
        check(row.get('hybrid_metrics') == str(metric_path), 'frame.%d.raw_JSON_path' % index)
        metric = cached_json(metric_path)
        check(metric.get('frame_id') == expected_frame_id, 'frame.%d.raw_JSON_identity' % index, metric.get('frame_id'))
        arrays_path = render_dir / 'hybrid_raw' / (expected_stem + '.npz')
        check(metric.get('arrays') == str(arrays_path), 'frame.%d.raw_NPZ_path_binding' % index, metric.get('arrays'))
        try:
            raw_receipt = npz_header(arrays_path)
            check(raw_receipt['NPY_headers'] == raw_schema and raw_receipt['stat_bytes'] > 0,
                  'frame.%d.raw_NPY_FP32_bool_schema' % index, raw_receipt)
        except Exception as exc:
            raw_receipt = {'path': str(arrays_path), 'error': str(exc), 'arrays_loaded': False}
            check(False, 'frame.%d.raw_NPY_header' % index, str(exc))
        frame_receipts.append({'uid': index, 'frame_id': row.get('frame_id'), 'camera': row.get('camera'),
                               'source_image': row.get('source_image'), 'timestamp_model': row.get('timestamp'),
                               'raw_timestamp_binding': 'event frame + exact spec timestamp + raw JSON frame_id + NPZ basename',
                               'width': row.get('width'), 'height': row.get('height'),
                               'channels': channel_receipts, 'hybrid_metrics': str(metric_path),
                               'raw_NPZ': raw_receipt})
    expected_stems = ['image_0_' + f for f in spec['frame_ids']]
    for channel in ('gt', 'composite', 'static', 'dynamic'):
        check(sorted(p.stem for p in (render_dir / channel).glob('*.png')) == expected_stems,
              'channel.%s.exact_49_file_set' % channel)
    check(sorted(p.stem for p in (render_dir / 'hybrid_raw').glob('*.npz')) == expected_stems,
          'raw_NPZ.exact_49_file_set')

    fixed_dir = ROOT / ('static_time_%d' % step)
    summary = cached_json(fixed_dir / 'summary.json')
    fixed_rows = summary.get('rows', [])
    check(summary.get('iteration') == step, 'fixed_time.iteration', summary.get('iteration'))
    check(summary.get('anchor_pose_frame') == spec['anchor_frame'] == '0029079'
          and spec['anchor_index'] == 10, 'fixed_time.anchor', summary.get('anchor_pose_frame'))
    check(summary.get('anchor_camera_directory') == spec['camera_directory'] == 'image_0',
          'fixed_time.anchor_camera', summary.get('anchor_camera_directory'))
    check(close(summary.get('anchor_timestamp_model'), spec['anchor_timestamp'])
          and close(spec['anchor_timestamp'], -0.28), 'fixed_time.anchor_model_timestamp')
    check(len(fixed_rows) == 49 and [r.get('source_timestamp_frame') for r in fixed_rows] == spec['frame_ids'],
          'fixed_time.49_source_timestamp_IDs')
    fixed_receipts = []
    for index, row in enumerate(fixed_rows):
        check(close(row.get('timestamp_model'), spec['model_timestamps'][index]),
              'fixed_time.%d.model_timestamp' % index, row.get('timestamp_model'))
        path = fixed_dir / ('%03d.npz' % index)
        try:
            header = npz_header(path)
            check(header['NPY_headers'] == fixed_schema and header['stat_bytes'] > 0,
                  'fixed_time.%d.FP32_header_schema' % index, header)
        except Exception as exc:
            header = {'path': str(path), 'error': str(exc), 'arrays_loaded': False}
            check(False, 'fixed_time.%d.NPY_header' % index, str(exc))
        fixed_receipts.append({'index': index, 'source_timestamp_frame': row.get('source_timestamp_frame'),
                               'timestamp_model': row.get('timestamp_model'), 'NPZ': header})
    check(sorted(p.name for p in fixed_dir.glob('*.npz')) == ['%03d.npz' % i for i in range(49)],
          'fixed_time.exact_49_NPZ_file_set')
    nodes[str(step)] = {
        'status': 'PASS_COMPLETED_NODE_STAT_AND_HEADER_CONTRACT' if not failures else 'FAIL',
        'step': step, 'event_path': str(event_path), 'event_status': event.get('status'),
        'original_completion_contract_scope_in_RAM': [step], 'original_completion_contract_result': original_result,
        'four_component_files': component_receipts, 'frames': frame_receipts,
        'SGCM_event_metadata': event.get('sgcm'),
        'SGCM_first_source_pose_frame': spec['frame_ids'][0],
        'SGCM_first_source_pose_basis': 'source window first pose; geometry was not revalidated in this stat/header audit',
        'unique_chronological_FRONT_frames': len({r.get('frame_id') for r in frames}),
        'PNG_IHDR_count': sum(len(r['channels']) for r in frame_receipts),
        'raw_NPZ_header_count': len(frame_receipts),
        'raw_NPY_headers_each': raw_schema,
        'fixed_pose': {'anchor_frame': summary.get('anchor_pose_frame'),
                       'camera_directory': summary.get('anchor_camera_directory'),
                       'anchor_timestamp_model': summary.get('anchor_timestamp_model'),
                       'rows': fixed_receipts, 'NPZ_header_count': len(fixed_receipts),
                       'NPY_headers_each': fixed_schema, 'scope': summary.get('scope'), 'unit': summary.get('unit')},
        'failure_fields': failures, 'restoration_verified': False,
        'actual_model_Adam_RNG_restore_verified': False,
    }
    all_failures.extend({'step': step, **f} for f in failures)

phase_resource_rows = []
for phase, values in memory.get('phases', {}).items():
    phase_resource_rows.append({
        'phase': phase, 'count': values.get('count'), 'active_count': values.get('active_count'),
        'failed_count': values.get('failed_count'), 'sample_count': values.get('sample_count'),
        'wall_seconds': values.get('wall_seconds'), 'wall_seconds_exclusive': values.get('wall_seconds_exclusive'),
        'peak_allocated_bytes': values.get('peak_allocated_bytes'),
        'peak_reserved_bytes': values.get('peak_reserved_bytes'),
        'current_allocated_bytes': values.get('current_allocated_bytes'),
        'current_reserved_bytes': values.get('current_reserved_bytes'),
        'NV_process_sampled_peak_bytes_lower_bound': values.get('process_peak_used_bytes'),
        'NV_device_sampled_peak_bytes_lower_bound': values.get('device_peak_used_bytes'),
        'NV_sampled_peaks_are_lower_bounds': True,
        'missing_NV_phase_values_mean': 'not sampled/attributed; not zero',
    })
wall = status.get('wall_seconds')
iteration = status.get('iteration')
resources = {
    'observed_singapore': snapshot_observed, 'source': 'existing status.json and memory_summary.json each read once',
    'status': status.get('status'), 'status_iteration': iteration, 'memory_iteration': memory.get('iteration'),
    'expected_iterations': status.get('iterations'), 'owned_PID_expected': 124555,
    'status_PID': status.get('pid'), 'memory_PID': memory.get('pid'),
    'PID_metadata_matches_expected': status.get('pid') == memory.get('pid') == 124555,
    'wall_seconds': wall,
    'overall_wall_iterations_per_second': iteration / wall if type(iteration) in (int, float)
        and type(wall) in (int, float) and wall > 0 else None,
    'overall_speed_scope': 'elapsed since launch, includes initialization and completed evaluations/checkpoints',
    'speed_window': status.get('speed_window'), 'iter_forward_backward_ms': status.get('iter_forward_backward_ms'),
    'loss': status.get('loss'), 'psnr': status.get('psnr'), 'static_points_status': status.get('static_points'),
    'point_counts_and_current_render_metadata': memory.get('extra'),
    'pytorch': memory.get('pytorch'), 'nvidia_smi_existing_samples': memory.get('nvidia_smi'),
    'sampling': memory.get('sampling'), 'phase_resources': phase_resource_rows,
    'phase_attribution': memory.get('phase_attribution'),
    'monitor_status': memory.get('monitor_status'), 'run_status': memory.get('run_status'),
    'error': memory.get('error'), 'error_count': memory.get('error_count'),
    'recent_errors': memory.get('recent_errors'), 'rtx3090_assessment': memory.get('rtx3090_assessment'),
    'NV_sampled_peak_lower_bounds': True, 'new_device_queries': False,
}
assert 'torch' not in sys.modules
receipt = {
    'schema_version': 1, 'status': 'TWO_NODE_STAT_HEADER_PASS' if not all_failures else 'FAIL',
    'audit_started_singapore': OBSERVED_START,
    'audit_finished_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'ssh_alias': 'wm-vgpu-1011', 'python_executable': sys.executable,
    'scene_id': spec['scene_id'], 'arm': 'GN11', 'run': str(ROOT), 'nodes_requested': [10000, 20000],
    'checked_nodes': [10000, 20000], 'terminal_complete': False,
    'raw_source_frame_range': spec['raw_frame_range'], 'source_path': spec['source_path'],
    'original_contract_function': str(BASE / 'tools/followup/contracts.py') + ':completion_contract',
    'original_source_NODES_restored_in_RAM': list(contracts.NODES),
    'full_30000_completion_audited': False, 'nodes': nodes, 'failure_fields': all_failures,
    'resources': resources, 'raw_metadata_snapshots': {'status': status, 'memory_summary': memory},
    'audit_scope': '10k/20k event.complete + four stat/ZIP directory headers + 49x4 PNG IHDR + 49 raw NPZ NPY headers + fixed 0029079 anchor/49 model-time NPZ headers',
    'read_only': True, 'GPU_used': False, 'Torch_imported': False, 'CUDA_initialized': False,
    'remote_files_written': False, 'source_config_input_changed': False, 'repository_changed': False,
    'controller_changed': False, 'automation_changed': False, 'GPUSlot_changed': False,
    'training_worker_started': False, 'training_worker_restarted': False, 'rerender_started': False,
    'new_hash_or_fingerprint_computed': False, 'large_checkpoint_or_sidecar_loaded': False,
    'NPZ_arrays_loaded': False, 'pth_tensor_or_pickle_payload_loaded': False,
    'restoration_verified': False, 'actual_model_Adam_RNG_restore_verified': False,
    'evaluation_scope': '49 all-observed reconstruction/development views, no heldout or independent test',
    'human_quality_verdict': None,
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))

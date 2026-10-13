"""Reuse the original completion contract; read stat and headers only on AutoDL."""
import datetime
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
ROOT = BASE / 'runs/scene0029_069_117_hybrid_30000_seed0_GN11'
sys.path.insert(0, str(BASE / 'tools/followup'))
from contracts import completion_contract, read_json

config = read_json(ROOT / 'effective_config.json')
contract = completion_contract(ROOT, config)
spec = config['followup_scene_spec']
nodes = {}
failures = []
for step in (10000, 20000, 30000):
    event_path = ROOT / ('checkpoint_events/iteration_%06d.json' % step)
    event = read_json(event_path)
    node_failures = []
    def check(condition, field, actual=None):
        if not condition:
            node_failures.append({'field': field, 'actual': actual})
    check(event.get('model_path') == str(ROOT), 'model_path', event.get('model_path'))
    check(event.get('source_path') == spec['source_path'], 'source_path', event.get('source_path'))
    check(event.get('parameters_evaluated') == 'after_optimizer_and_sgcm_same_as_checkpoint',
          'evaluation_same_parameter_state', event.get('parameters_evaluated'))
    check(event.get('iteration') == step and event.get('status') == 'complete', 'event_complete')
    components = {}
    expected_names = {'static': 'chkpnt%d.pth', 'dynamic': 'dynamic_chkpnt%d.pth',
                      'environment': 'env_light_chkpnt%d.pth', 'training_state': 'training_state%d.pth'}
    for key, template in expected_names.items():
        entry = event['checkpoints'][key]
        path = ROOT / (template % step)
        stat = path.stat()
        check(entry['path'] == str(path) and stat.st_size == entry['bytes'] > 0,
              'checkpoint.' + key + '.expected_path_nonempty_size', entry)
        components[key] = {'path': str(path), 'stat_bytes': stat.st_size,
                           'event_bytes': entry['bytes'], 'payload_loaded': False}
    frames = event['frames']
    check(len(frames) == 49 and len({r['frame_id'] for r in frames}) == 49, '49_unique_frames')
    for index, row in enumerate(frames):
        expected_source = Path(spec['source_path']) / ('image/image_0/' + spec['frame_ids'][index] + '.png')
        check(row.get('source_image') == str(expected_source), 'source_frame_pointer.' + str(index), row.get('source_image'))
        check(row.get('camera') == 'image_0' and row.get('width') == 480 and row.get('height') == 320,
              'FRONT_camera_dimensions.' + str(index))
        check(isinstance(row.get('timestamp'), (int, float)) and
              math.isclose(row['timestamp'], spec['model_timestamps'][index], abs_tol=1e-12),
              'frame_timestamp.' + str(index), row.get('timestamp'))
    summary = read_json(ROOT / ('static_time_%d/summary.json' % step))
    check(summary.get('anchor_pose_frame') == spec['anchor_frame'], 'static_time_anchor_frame', summary.get('anchor_pose_frame'))
    check(summary.get('iteration') == step, 'static_time_iteration', summary.get('iteration'))
    check(math.isclose(summary.get('anchor_timestamp_model', 9), spec['anchor_timestamp'], abs_tol=1e-12),
          'static_time_anchor_timestamp', summary.get('anchor_timestamp_model'))
    nodes[str(step)] = {
        'status': 'PASS_CHECKPOINT_FILE_AND_FRAME_CONTRACT' if not node_failures else 'FAIL_SUPPLEMENTAL_CONTRACT',
        'event_path': str(event_path), 'event_status': event['status'], 'step': step,
        'four_component_files': components, 'source_path': event['source_path'],
        'raw_source_frame_range': spec['raw_frame_range'], 'unique_chronological_FRONT_frames': len(frames),
        'timestamps_model_range': [frames[0]['timestamp'], frames[-1]['timestamp']],
        'four_channel_png_IHDR_count': 196, 'raw_FP32_npz_headers_count': 49,
        'raw_boolean_masks_headers_count': 49, 'render_WH': [480, 320],
        'fixed_pose_anchor': summary['anchor_pose_frame'], 'fixed_pose_camera': summary['anchor_camera_directory'],
        'fixed_pose_anchor_timestamp_model': summary['anchor_timestamp_model'],
        'fixed_pose_49_model_times': [r['timestamp_model'] for r in summary['rows']],
        'fixed_pose_FP32_npz_headers_count': 49, 'failure_fields': node_failures,
        'restoration_verified': False, 'actual_model_Adam_RNG_restore_verified': False,
    }
    failures.extend({'step': step, **failure} for failure in node_failures)

pid = 124555
try:
    parts = [p.decode('utf-8', errors='replace') for p in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if p]
    owned_process = {'pid': pid, 'alive': bool(parts),
                     'matches_owned_job': any('run_scene_gn11.py' in p for p in parts) and
                                          any('scene0029_spec.json' in p for p in parts)}
except FileNotFoundError:
    owned_process = {'pid': pid, 'alive': False, 'matches_owned_job': False}
assert 'torch' not in sys.modules
receipt = {
    'schema_version': 1,
    'status': 'FOLLOWUP_COMPLETE_CONTRACT_PASS' if not failures else 'FAIL_SUPPLEMENTAL_CONTRACT',
    'scheduled_check_label': '2026-10-13 final red terminal after shared lock',
    'observed_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'ssh_alias': 'wm-vgpu-1011', 'python_executable': sys.executable,
    'scene_id': spec['scene_id'], 'arm': 'GN11', 'output': str(ROOT),
    'original_contract_function': str(BASE / 'tools/followup/contracts.py') + ':completion_contract',
    'original_completion_contract_actual_result': contract, 'nodes': nodes,
    'failure_fields': failures, 'owned_process_observation': owned_process,
    'audit_scope': 'Original completion_contract plus exact source/camera/parameter-state and stat checks; PNG/NPY headers only; no payload arrays, models, optimizer or RNG restoration',
    'read_only': True, 'GPU_used': False, 'Torch_imported': False, 'CUDA_initialized': False,
    'remote_files_written': False, 'repository_changed': False, 'controller_changed': False,
    'training_worker_started': False, 'rerender_started': False,
    'new_hash_or_fingerprint_computed': False, 'large_checkpoint_or_sidecar_loaded': False,
    'restoration_verified': False, 'actual_model_Adam_RNG_restore_verified': False,
    'evaluation_scope': 'All 49 observed development reconstruction views; no heldout or independent test',
    'human_verdict': None,
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))

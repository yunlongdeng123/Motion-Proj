"""CPU-only live receipt for the one approved running long B1 trajectory."""
import datetime
import fcntl
import json
from pathlib import Path
import subprocess

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
CAMPAIGN = BASE / 'campaign/vgpu_bugfix_20261012'
PLAN = CAMPAIGN / 'loss_audit/long_pair_30k_32000/materialized_plan'
OUT = CAMPAIGN / 'continuation_diagnostic/original30k_to32000'


def read(path):
    return json.loads(path.read_text())


status = read(OUT / 'status.json')
launcher = read(PLAN / 'continuation_launcher_status.json')
first = read(OUT / 'first_step_reference_comparison.json')
node = read(OUT / 'extra_eval/30001/diagnostic_eval.json')
assert launcher['launcher_pid'] == 36252 and launcher['worker_pid'] == 36255
assert status['restore']['status'] == 'actual_restore_pass'
assert first['iteration'] == 30001 and len(first['comparisons']) == 10
assert all(row['exact_equal'] and row['abs_delta'] == 0 for row in first['comparisons'].values())
assert node['all_rng_unchanged'] and len(node['frames']) == 5
rows = [json.loads(line) for line in (OUT / 'step_losses.jsonl').read_text().split('\n')[:-1] if line]
assert rows and rows[0]['iteration'] == 30001
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                '--format=csv,noheader'], text=True).strip().splitlines()
assert apps and all(int(row.split(',')[0]) == launcher['worker_pid'] for row in apps), apps
processes = subprocess.check_output(['ps', '-p', '36252,36255', '-o', 'pid=,stat=,args='], text=True)
assert len(processes.strip().splitlines()) == 2
with (BASE / 'campaign/gpu_slot.lock').open('r+') as fd:
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held = True
    else:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        raise RuntimeError('Live GPU worker does not retain exclusive slot lock')
receipt = {
    'schema_version': 1,
    'status': 'B1_running_restore_and_first10_actual_PASS',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'GPU_owner_agent': '/root/vgpu_component_diagnostic_solmax',
    'host_alias': 'wm-vgpu-1011',
    'GPU_started_by_this_receipt_script': False,
    'sole_authorized_active_arm': 'B1_original_math_30k_to32000',
    'output': str(OUT), 'launcher_pid': 36252, 'worker_pid': 36255,
    'both_actual_processes_present': True, 'actual_GPU_compute_apps': apps,
    'exclusive_GPU_slot_lock_actually_held': held,
    'actual_step_count_at_receipt': len(rows), 'last_actual_iteration_at_receipt': rows[-1]['iteration'],
    'status_file_updates_only_at_eval_nodes': [30001, 30300, 31000, 32000],
    'progress_source': str(OUT / 'step_losses.jsonl'),
    'first10_actual_scalar_comparison': first,
    'actual_restore': status['restore'],
    'first_fixed5_eval_preserved_all_RNG': True,
    'first_iteration': 30001, 'stop_iteration': 32000, 'horizon': 50000,
    'seed': 0, 'algorithm_changes': [], 'formal_run_resumed': False,
    'approved_source_pins_observed_before_launch': {
        'root_coordination_receipt.json': 'f9a8b649da27f3626daa6ce2e27a3a2bec50a173ea026daf06742a90995b8214',
        'prepared_plan.json': '359b9d62ee9aec28951532cedbaed45401838dd6ea2b961d5704bec2d303a2d5',
        'continuation_harness.py': '88a2df38e5ac4df5905f1de030701189763d469f9eb3daf9e3339f1931aba829',
        'launch_controlled_continuation.py': '6337b0c660d1eedb95b822d87d0feaa6d5691ba22da4de8d8d54fb750a7e68b0',
    },
    'poll_files': {
        'worker_status': str(OUT / 'status.json'),
        'worker_exit': str(OUT / 'actual_exit_code.txt'),
        'launcher_status': str(PLAN / 'continuation_launcher_status.json'),
        'launcher_exit': str(PLAN / 'continuation_launcher.exit'),
        'log': str(PLAN / 'continuation_training.log'),
        'diagnostic_report': str(OUT / 'continuation_diagnostic.json'),
        'final49_metrics': str(OUT / 'extra_eval/32000/diagnostic_eval.json'),
        'terminal_four_state_manifest': str(OUT / 'terminal_32000/diagnostic_snapshot_manifest.json'),
    },
    'completion_gates': [
        'Worker and launcher actual exit0, statuses complete, lock released and actual compute apps empty.',
        'Exactly2000 completed step records30001..32000, finite full original objective/state.',
        'All4 extra evaluations preserve Python/NumPy/TorchCPU/allCUDA RNG.',
        'Final32000 all49 original cameras, each six truePNG channels and original metrics/NPZ.',
        'Terminal static/dynamic/environment/trainingstate4 snapshot hashes/bytes/CPU schema/config validated.',
        'Protected original source, checkpoint, data and kernel hashes remain unchanged.',
    ],
    'next_actions_after_completion': [
        'CPU verify final evidence; sync smallJSON/raw10 first, then fullPNG; do not download bigcheckpoint/NPZ.',
        'B2 requires a new exact root approval and fresh idle/process/output/flock gate; never infer it from this B1 receipt.',
        'Long outside variant requires genuine complete B1/B2 pairing references, new binding/plan/review/root approval.',
    ],
    'failure_policy': 'Keep actual failure evidence and inform root; no automatic retry, coefficient/window/source change or formal-run resume.',
    'normal_algorithm_success_claimed': False,
}
(OUT / 'receipts').mkdir(exist_ok=True)
path = OUT / 'receipts/gpu_owner_handoff.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'path': str(path),
                  'last_actual_iteration': rows[-1]['iteration'], 'actual_GPU_compute_apps': apps}))

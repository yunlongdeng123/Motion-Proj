"""Record actual GPU/lock release after the completed bounded variant; CPU only."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
CAMPAIGN = BASE / 'campaign/vgpu_bugfix_20261012'
OUT = CAMPAIGN / 'continuation_diagnostic/opacity_outside30k_to30300'
PLAN = CAMPAIGN / 'component_diagnostic/continuation_plan'


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


status = read(OUT / 'status.json')
launcher = read(PLAN / 'opacity_outside_launcher_status.json')
summary = read(OUT / 'receipts/full49_summary.json')
assert status['status'] == launcher['status'] == 'complete'
assert status['actual_exit_code'] == launcher['actual_exit_code'] == 0
assert not status['gpu_lock_held'] and summary['all_protected_SHA_unchanged']
apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                       '--format=csv,noheader'], text=True, capture_output=True, check=True)
assert not apps.stdout.strip(), apps.stdout
pids = [launcher['launcher_pid'], launcher['worker_pid']]
processes = subprocess.run(['ps', '-p', ','.join(map(str, pids)), '-o', 'pid=,stat=,args='],
                           text=True, capture_output=True)
assert processes.returncode == 1 and not processes.stdout.strip(), processes.stdout
with (BASE / 'campaign/gpu_slot.lock').open('a') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
receipt = {
    'status': 'completed_variant_GPU_available_for_root_coordination',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'host_alias': 'wm-vgpu-1011', 'actual_iteration': status['iteration'],
    'actual_worker_exit_code': 0, 'actual_launcher_exit_code': 0,
    'launcher_pid': pids[0], 'worker_pid': pids[1],
    'both_completed_processes_absent': True, 'actual_compute_apps': [],
    'exclusive_flock_available_and_immediately_released': True,
    'GPU_or_training_started_by_this_script': False,
    'no_new_GPU_experiment_authorized_or_launched_by_this_receipt': True,
    'normal_algorithm_success_claimed': False,
    'summary': pin(OUT / 'receipts/full49_summary.json'),
    'status_file': pin(OUT / 'status.json'),
    'launcher_status_file': pin(PLAN / 'opacity_outside_launcher_status.json'),
    'root_approval': pin(PLAN / 'opacity_outside_root_coordination_receipt.json'),
    'terminal_four_component_manifest': pin(OUT / 'terminal_30300/diagnostic_snapshot_manifest.json'),
}
path = OUT / 'receipts/gpu_release_handoff.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(pin(path)))

"""Read current original long-arm process, GPU lock, and actual saved step health."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
C = BASE / 'campaign/vgpu_bugfix_20261012'
P = C / 'loss_audit/long_pair_30k_32000/materialized_plan'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('B1', 'B2'), required=True)
    arm = parser.parse_args().arm
    output = C / 'continuation_diagnostic' / ('original30k_to32000' if arm == 'B1' else 'original30k_to32000_noise_reference')
    plan = P if arm == 'B1' else P / 'baseline_noise_reference'
    read = lambda path: json.loads(path.read_text())
    worker_status = read(output / 'status.json')
    launcher = read(plan / 'continuation_launcher_status.json')
    rows = [json.loads(s) for s in (output / 'step_losses.jsonl').read_text().split('\n')[:-1]]
    assert [r['iteration'] for r in rows] == list(range(30001, 30001 + len(rows)))
    assert all(math.isfinite(value) for row in rows for value in row['losses'].values())
    pair = None
    if arm == 'B2':
        reference = [json.loads(s) for s in (output.parent / 'original30k_to32000/step_losses.jsonl').read_text().splitlines()]
        assert all(all(r[k] == q[k] for k in ('iteration', 'camera_id', 'time_shift', 'learning_rates'))
                   for r, q in zip(rows, reference))
        pair = {'completed_prefix_rows_checked': len(rows), 'camera_time_shift_LR_exact_to_B1': True}
    pid = launcher['worker_pid']
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(apps) <= 1 and all(int(line.split(',')[0]) == pid for line in apps), apps
    with (BASE / 'campaign/gpu_slot.lock').open('r+') as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = True
        else:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            held = False
    nodes = []
    for step in (30001, 30300, 31000, 32000):
        path = output / f'extra_eval/{step}/diagnostic_eval.json'
        if path.exists():
            node = read(path)
            assert node['all_rng_unchanged']
            nodes.append({'iteration': step, 'camera_count': len(node['frames']), 'all_rng_unchanged': True})
    result = {'schema_version': 1, 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'arm': arm, 'host_alias': 'wm-vgpu-1011', 'output': str(output),
              'actual_launcher_status': launcher['status'], 'actual_worker_status': worker_status['status'],
              'launcher_pid': launcher['launcher_pid'], 'worker_pid': pid,
              'launcher_process_exists': Path('/proc', str(launcher['launcher_pid'])).exists(),
              'worker_process_exists': Path('/proc', str(pid)).exists(),
              'GPU_compute_apps': apps, 'exclusive_lock_is_held': held,
              'complete_step_rows': len(rows), 'last_actual_iteration': rows[-1]['iteration'] if rows else None,
              'last_actual_losses': rows[-1]['losses'] if rows else None,
              'completed_logged_losses_all_finite': True, 'completed_prefix_pairing': pair,
              'saved_nodes_rng_preserved': nodes,
              'worker_exit_code': int((output / 'actual_exit_code.txt').read_text()) if (output / 'actual_exit_code.txt').exists() else None,
              'launcher_exit_code': launcher.get('actual_exit_code'),
              'worker_status_updates_only_at_eval_nodes': True, 'progress_source': str(output / 'step_losses.jsonl'),
              'CPU_only_observation': True, 'new_training_started': False,
              'V_GPU_authorized': False, 'shutdown_authorized_now': False,
              'whole_debug_task_complete': False, 'normal_algorithm_success_claimed': False}
    if worker_status['status'] == 'complete' and launcher.get('actual_exit_code') == 0:
        assert len(rows) == 2000 and not held and not apps
        assert result['worker_exit_code'] == 0
        assert not result['worker_process_exists']
        result['actual_exit0_GPU_released'] = True
    path = output / 'receipts/live_health_latest.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    result['receipt_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(result))


if __name__ == '__main__':
    main()

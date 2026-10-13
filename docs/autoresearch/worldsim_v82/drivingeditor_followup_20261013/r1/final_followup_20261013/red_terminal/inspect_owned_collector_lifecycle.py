"""Final process exit observation for the two owned CPU collection attempts and children."""
import datetime
import json
from pathlib import Path

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
paths = [BASE / 'metadata/red_terminal_20261013_1056', BASE / 'metadata/red_terminal_20261013_1056_attempt2']
pids = {124555: ['authorized_last_red_training']}
attempts = []
for meta in paths:
    status = json.loads((meta / 'status.json').read_text(encoding='utf-8'))
    attempts.append({'metadata': str(meta), 'status': status['status'], 'collector_pid': status['collector_pid'],
                     'error': status.get('error'), 'active_cpu_child_pid': status.get('active_cpu_child_pid')})
    pids.setdefault(status['collector_pid'], []).append(meta.name + ':collector')
    for stage in status['stages']:
        pids.setdefault(stage['pid'], []).append(meta.name + ':' + stage['name'])
rows = []
for pid, roles in sorted(pids.items()):
    proc = Path('/proc') / str(pid)
    try:
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().decode('utf-8', errors='replace').replace('\0', ' ').strip()
        alive = state not in ('Z', 'X')
    except FileNotFoundError:
        state, command, alive = 'absent', '', False
    rows.append({'pid': pid, 'roles': roles, 'alive': alive, 'proc_state': state, 'command': command})
assert not any(row['alive'] for row in rows), 'An owned process is still alive; power handoff not ready'
assert attempts[-1]['status'] == 'PASS_RED_30K_TERMINAL_AND_REAL_SAVED_CPU_REPORT'
assert attempts[0]['status'] == 'engineering_stopped_actual_partial_assets_preserved'
assert all(item['active_cpu_child_pid'] is None for item in attempts)
record = {'status': 'PASS_ALL_OWNED_RED_TRAINING_AND_CPU_COLLECTOR_PROCESSES_EXITED',
          'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'processes': rows, 'attempts': attempts, 'no_training_progress_read': True,
          'GPU_used': False, 'training_started': False, 'human_verdict': None}
print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))

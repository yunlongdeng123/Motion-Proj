"""Wait only for the already-started owned CPU collector; never read training status."""
import json
from pathlib import Path
import time

PID = 146207
META = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/metadata/red_terminal_20261013_1056')
proc = Path('/proc') / str(PID)
while True:
    try:
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().decode('utf-8', errors='replace')
    except FileNotFoundError:
        break
    if state in ('Z', 'X'):
        break
    assert 'collect_red_terminal_cpu.py' in command, 'Owned CPU collector PID identity changed'
    time.sleep(2)
status = json.loads((META / 'status.json').read_text(encoding='utf-8'))
print(json.dumps({'owned_cpu_collector_pid': PID, 'owned_cpu_collector_exited': True,
                  'collector_terminal_status': status['status'], 'collector_result': str(META / 'result.json'),
                  'failure': status.get('error'), 'training_status_read_by_waiter': False}, ensure_ascii=False), flush=True)
if status['status'] != 'PASS_RED_30K_TERMINAL_AND_REAL_SAVED_CPU_REPORT':
    raise SystemExit(1)

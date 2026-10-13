"""Launch exactly one owned CPU terminal collector in a new metadata directory."""
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
META = BASE / 'metadata/red_terminal_20261013_1056'
REPORT = BASE / 'reports/red_gn11_30k_saved_cpu_20261013'
SCRIPT = BASE / 'tools/red_terminal_20261013/collect_red_terminal_cpu.py'
PY = '/root/autodl-tmp/envs/drivingeditor/bin/python'
assert SCRIPT.is_file() and not META.exists() and not REPORT.exists()
META.mkdir()
command = [PY, '-u', str(SCRIPT)]
with (META / 'collector.stdout.log').open('xb') as output:
    child = subprocess.Popen(command, cwd=str(BASE), stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                             close_fds=True, start_new_session=True,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0'))
record = {'status': 'OWNED_RED_TERMINAL_CPU_COLLECTOR_DISPATCHED', 'pid': child.pid, 'command': command,
          'metadata': str(META), 'log': str(META / 'collector.stdout.log'), 'report': str(REPORT),
          'blocking_shared_lock': '/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock',
          'GPU_used': False, 'training_started': False, 'continuous_training_polling': False, 'human_verdict': None}
with (META / 'launch.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
print(json.dumps(record, ensure_ascii=False))

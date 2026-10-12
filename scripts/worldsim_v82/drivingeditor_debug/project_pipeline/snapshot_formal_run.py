import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path('/root/autodl-tmp/drivingeditor_repro')
run = base/'runs/0017085_demo_000_020_dynamic_30000'
log_path = base/'logs/formal_demo_30000.log'
log = log_path.read_text(errors='replace')
steps = re.findall(r'(\d+)/30000', log)
pid = int((base/'logs/formal_demo.pid').read_text())
try:
    os.kill(pid, 0)
    running = True
except ProcessLookupError:
    running = False
acc = EventAccumulator(str(run), size_guidance={'scalars': 0})
acc.Reload()
scalars = {}
for tag in ('train/loss', 'train/total_points', 'train/psnr'):
    if tag in acc.Tags().get('scalars', []):
        events = acc.Scalars(tag)
        scalars[tag] = {'events': len(events), 'all_finite': all(math.isfinite(e.value) for e in events),
                        'first': {'step': events[0].step, 'value': events[0].value},
                        'latest': {'step': events[-1].step, 'value': events[-1].value},
                        'min': min(e.value for e in events), 'max': max(e.value for e in events)}
result = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'running' if running else 'exited', 'launcher_pid': pid,
    'started': (base/'logs/formal_demo.started').read_text().strip(),
    'last_logged_iteration': int(steps[-1]) if steps else None,
    'target_iterations': 30000, 'run_directory': str(run), 'log': str(log_path),
    'source_scene': str(base/'data/waymo_scenes/0017085_demo_000_020'),
    'including_dynamic': True, 'training_protocol': 'Official code reconstruction defaults, FRONT, scale4, seed0; 21 verified display frames, unknown author training context',
    'scalars': scalars,
    'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip(),
    'completion_record': str(base/'logs/formal_demo_completion.json'),
    'scope': 'Formal code baseline in progress; no completed quality or paper reproduction claim',
}
(base/'logs/formal_run_status.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

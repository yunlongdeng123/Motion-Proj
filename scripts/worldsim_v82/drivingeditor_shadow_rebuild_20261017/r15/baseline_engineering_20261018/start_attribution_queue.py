"""Launch a single registered zero-update follow-up; no GPU at launch."""
from pathlib import Path
import json, subprocess, sys

out = Path(__file__).resolve().parent
assert not (out/'attribution_queue_started.json').exists()
assert not (out/'attribution_controller.log').exists()
with (out/'attribution_controller.log').open('x') as log:
    process = subprocess.Popen([sys.executable,'-B','-u',str(out/'run_baseline_attribution.py')],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(controller_pid=process.pid, optimizer_updates=0, GPU_acquired_by_launcher=False)))

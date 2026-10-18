"""Launch exactly one registered follow-up controller, without acquiring GPU."""
from pathlib import Path
import json, subprocess, sys

out = Path(__file__).resolve().parent
assert not (out / 'review_queue_started.json').exists(), 'queue already registered'
assert not (out / 'review_controller.log').exists(), 'preserve prior attempt'
with (out / 'review_controller.log').open('x') as log:
    worker = subprocess.Popen([sys.executable, '-B', '-u', str(out / 'run_engineering_review.py')],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(controller_pid=worker.pid, GPU_acquired_by_launcher=False)))

"""Only launch this task's finite frozen probe; never resume the historical queue."""
import json
import subprocess
import sys
from pathlib import Path

root = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
task = root / 'editable_ownership_20261030'
name = sys.argv[1]
assert name.replace('_', '').isalnum()
raws = sys.argv[2:]
assert raws and set(map(int, raws)) <= {98, 113, 114}
output = task / name
assert not output.exists()
log = task / (name + '.log')
assert not log.exists()
cmd = [sys.executable, str(task / 'run_scene1_probe.py'), '--output', str(output),
       '--input-contract', str(root / 'unknown_white_20261030/render_inputs/input_contract.json'),
       '--code-dir', str(root / 'code'), '--scope-file', str(task / 'scope.py'),
       '--authorization', str(task / 'authorization.json'), '--raws', *raws, '--split-max', '8']
with log.open('wb') as stream:
    proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, cwd=task)
receipt = dict(pid=proc.pid, command=cmd, log=str(log), output=str(output))
(task / (name + '_launch.json')).write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))

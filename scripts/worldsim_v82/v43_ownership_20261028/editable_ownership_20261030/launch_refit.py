import json
import subprocess
import sys
from pathlib import Path

root = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
task = root / 'editable_ownership_20261030'
name = sys.argv[1]
assert name.replace('_', '').isalnum()
assert not (task / name).exists() and not (task / (name + '.log')).exists()
cmd = [sys.executable, str(task / 'run_split_refit.py'), '--output', str(task / name),
       '--probe-dir', str(task / 'smoke_raw113_r1'),
       '--input-contract', str(root / 'unknown_white_20261030/render_inputs/input_contract.json'),
       '--code-dir', str(root / 'code'), '--scope-file', str(task / 'scope.py'),
       '--authorization', str(task / 'authorization.json')]
if len(sys.argv) > 2:
    checkpoint = Path(sys.argv[2])
    assert checkpoint.is_file() and checkpoint.parent.parent == task
    cmd += ['--resume-checkpoint', str(checkpoint)]
with (task / (name + '.log')).open('wb') as stream:
    p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=task)
receipt = dict(pid=p.pid, command=cmd, log=str(task / (name + '.log')))
(task / (name + '_launch.json')).write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))

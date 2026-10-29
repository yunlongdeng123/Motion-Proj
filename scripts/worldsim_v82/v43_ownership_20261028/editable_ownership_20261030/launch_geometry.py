import subprocess
import sys
from pathlib import Path

root = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
task = root / 'editable_ownership_20261030'
cmd = [sys.executable, str(task / 'audit_support_geometry.py'),
       '--input-contract', str(root / 'unknown_white_20261030/render_inputs/input_contract.json'),
       '--code-dir', str(root / 'code'), '--authorization', str(task / 'authorization.json'),
       '--probe', str(task / 'probe_anchors_r1'), '--output', str(task / 'support_geometry_r1')]
with (task / 'support_geometry_r1.log').open('wb') as stream:
    r = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
print((task / 'support_geometry_r1.log').read_text()[-6000:])
sys.exit(r.returncode)

"""Capture final versions, source patches and script hashes without using the GPU."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

base = Path('/root/autodl-tmp/drivingeditor_repro')
def command(args):
    return subprocess.check_output(args, text=True).strip()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

repositories = [base/'DrivingEditor', base/'DrivingEditor-run']
repositories.extend(p for p in (base/'deps').iterdir() if (p/'.git').exists())
patch = subprocess.check_output(['git', '-C', str(base/'DrivingEditor-run'), 'diff', '--binary'])
(base/'logs/compatibility.patch').write_bytes(patch)
freeze = command([sys.executable, '-m', 'pip', 'freeze']) + '\n'
(base/'logs/environment.freeze.txt').write_text(freeze)
result = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'environment_python': sys.executable,
    'python_version': sys.version,
    'host': platform.node(),
    'pip_check': command([sys.executable, '-m', 'pip', 'check']),
    'source_repositories': {str(p): {'commit': command(['git', '-C', str(p), 'rev-parse', 'HEAD']),
        'changes': command(['git', '-C', str(p), 'status', '--porcelain'])} for p in repositories},
    'compatibility_patch_sha256': hashlib.sha256(patch).hexdigest(),
    'environment_freeze_sha256': hashlib.sha256(freeze.encode()).hexdigest(),
    'scripts_sha256': {p.name: digest(p) for p in sorted((base/'scripts').iterdir())
        if p.is_file() and p.suffix in {'.py', '.sh'}},
    'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
    'data_disk': command(['df', '-h', str(base)]),
}
(base/'logs/reproduction_state.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

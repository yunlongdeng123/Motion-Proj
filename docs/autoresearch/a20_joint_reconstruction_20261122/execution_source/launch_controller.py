"""Frozen smoke -> clean G40, leaving later stages behind geometry review."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/a20_joint_reconstruction_20261122_r1')
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'
RUN = ROOT / ('execution_' + sys.argv[1])
RUN.mkdir(exist_ok=False)
SOURCE = RUN / 'source'
shutil.copytree(ROOT / 'runtime', SOURCE, ignore=shutil.ignore_patterns('__pycache__', 'cpu_preflight'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

expected = '1e7964eda4a06ae835cc0d8483c5bdce06a21c707dc35120a44b04914a937dc4'
actual = digest(ROOT / 'initialization/joint_actor_init.npz')
assert actual == expected, (actual, expected)
write('execution_entry.json', dict(pid=os.getpid(), started=time.time(), GPU_keep_on=True,
    initializer_sha256=actual, source_sha256={p.name: digest(p) for p in SOURCE.glob('*') if p.is_file()},
    formal_budget=dict(geometry=40, appearance=80, joint=40), automatic_stages=sys.argv[2:] or ['smoke', 'geometry'],
    next_stage_requires_actual_geometry_gate_and_grey_review=True))
base = [PYTHON, '-u', str(SOURCE / 'run_joint_reconstruction.py'),
    '--init', str(ROOT / 'initialization/joint_actor_init.npz'),
    '--manifest', str(ROOT / 'initialization/manifest.json'),
    '--source-manifest', str(ROOT / 'native_bindings.json'),
    '--prior-contract', '/root/autodl-tmp/drivingeditor_repro/campaign/a20_sd15_sameview_20261122_r1/student/contract.json',
    '--context-plan', str(ROOT / 'context_plan.json'),
    '--path-map-file', str(SOURCE / 'remote_path_map.json'),
    '--config', str(SOURCE / 'training_contract.frozen.json')]
results = []
for mode in (sys.argv[2:] or ('smoke', 'geometry')):
    command = base + ['--mode', 'evaluate' if mode == 'initial' else mode, '--output', str(RUN / mode)]
    if mode == 'initial': command += ['--asset', str(ROOT / 'initialization/joint_actor_init.npz')]
    started = time.time()
    with (RUN / (mode + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(mode=mode, returncode=result.returncode, seconds=time.time()-started))
    write('controller_progress.json', dict(results=results, GPU_keep_on=True))
    if result.returncode:
        write('controller_completion.json', dict(completed=False, failed_stage=mode, results=results, GPU_keep_on=True))
        sys.exit(result.returncode)
write('controller_completion.json', dict(completed=True, results=results,
    stopped_for_geometry_review=True, GPU_keep_on=True))

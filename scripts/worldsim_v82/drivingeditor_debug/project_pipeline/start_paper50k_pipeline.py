#!/usr/bin/env python3
"""Start the one-run report worker and, when requested, the validated50k job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
RUN = BASE / 'runs/scene0017_000_048_paper_50000_seed0'
SCENE = BASE / 'data/waymo_paper_scenes/scene0017_000_048'
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'


def spawn(name, command):
    pidfile = BASE / f'logs/{name}.pid'
    if pidfile.is_file():
        pid = int(pidfile.read_text().strip())
        process_file = Path(f'/proc/{pid}/cmdline')
        if process_file.exists():
            running = process_file.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if Path(command[1]).name in running:
                return {'pid': pid, 'state': 'already_running'}
            raise RuntimeError(f'PID file belongs to a different process:{pid}; inspect before relaunch')
    with (BASE / f'logs/{name}.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
    pidfile.write_text(str(process.pid) + '\n')
    return {'pid': process.pid, 'state': 'launched', 'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-training', action='store_true')
    args = parser.parse_args()
    results = {}
    if args.start_training:
        ready = SCENE / 'formal_training_ready.json'
        if not ready.is_file():
            raise RuntimeError('49-frame dataset readiness marker is missing')
        if json.loads(ready.read_text()).get('status') != 'ready':
            raise RuntimeError('49-frame dataset readiness check did not pass')
        if RUN.exists():
            raise RuntimeError('Training output already exists; do not overwrite or silently restart')
        subprocess.run([PYTHON, str(BASE / 'scripts/run_paper50k_training.py'), '--repository',
                        str(BASE / 'DrivingEditor-paper'), '--scene', str(SCENE), '--output', str(RUN),
                        '--check-only'], check=True)
        memory = subprocess.check_output(['nvidia-smi', '--query-compute-apps=used_gpu_memory',
                                          '--format=csv,noheader,nounits']).decode()
        allocated = sum(int(row.strip()) for row in memory.splitlines() if row.strip().isdigit())
        if allocated > 1024:
            raise RuntimeError(f'GPU still occupied ({allocated}MiB); wait for the known diagnostic job')
        results['training'] = spawn('paper50k', ['bash', str(BASE / 'scripts/run_paper50k_job.sh')])
    results['report_worker'] = spawn('paper50k_comparison', [PYTHON, str(BASE / 'scripts/build_checkpoint_comparison.py'),
        '--run', str(RUN), '--output', str(BASE / 'reports/paper50k_comparison'), '--watch', '--poll-seconds', '30'])
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Preserve the failed attempt and restart the identical protocol once."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

BASE = Path('/root/autodl-tmp/drivingeditor_repro').resolve()
RUN = BASE / 'runs/scene0017_000_048_paper_50000_seed0'
ARCHIVE = BASE / 'runs/scene0017_000_048_paper_50000_seed0_failed_oom_20261011T123448'
RECORD = BASE / 'logs/paper50k_allocator_recovery.json'
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # A previous successful launch must never be repeated by this helper.
    assert not RECORD.exists(), 'Recovery already recorded; inspect it instead of relaunching'
    assert RUN.parent.resolve() == (BASE / 'runs').resolve()
    assert ARCHIVE.parent.resolve() == RUN.parent.resolve()
    assert not ARCHIVE.exists(), 'Failed-attempt destination already exists'
    state = json.loads((RUN / 'progress_status.json').read_text())
    assert state['state'] == 'failed' and 'OutOfMemoryError' in state.get('error', '')
    assert not Path(f"/proc/{state['pid']}").exists(), 'Training PID is still alive'
    assert (BASE / 'logs/paper50k.exit').read_text().strip() == '1'
    assert not list(RUN.glob('*chkpnt*.pth')), 'Unexpected checkpoint: use a reviewed resume instead'
    assert not list((RUN / 'checkpoint_events').glob('*.json'))
    for pid_name in ['paper50k.pid', 'paper50k_comparison.pid']:
        pidfile = BASE / 'logs' / pid_name
        if pidfile.exists():
            pid = pidfile.read_text().strip()
            assert not Path(f'/proc/{pid}').exists(), f'Pipeline process still alive: {pid}'
    memory = subprocess.check_output(['nvidia-smi', '--query-compute-apps=used_gpu_memory',
                                      '--format=csv,noheader,nounits']).decode()
    assert sum(int(row.strip()) for row in memory.splitlines() if row.strip().isdigit()) < 1024
    assert shutil.disk_usage(BASE).free > 18 * 1024 ** 3

    names = ['paper50k.log', 'paper50k.pid', 'paper50k.started', 'paper50k.launcher.pid',
             'paper50k.exit', 'paper50k.finished', 'paper50k_comparison.log', 'paper50k_comparison.pid']
    record = {
        'timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'status': 'archiving_failed_attempt',
        'failed_run': str(ARCHIVE), 'active_run': str(RUN), 'failure': state,
        'old_effective_config_sha256': digest(RUN / 'effective_config.yaml'),
        'old_code_snapshot_sha256': digest(RUN / 'code_snapshot.tar.gz'),
        'recovery': 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        'restart_from_step': 0,
        'reason': 'No complete checkpoint before first 10k; preserve evidence and repeat same protocol.',
        'training_code_and_parameters_changed': False,
        'preserved_logs': names,
    }
    # Only this exact failed run is relocated; nothing is removed or overwritten.
    RUN.rename(ARCHIVE)
    log_archive = ARCHIVE / 'failure_logs'
    log_archive.mkdir()
    for name in names:
        old = BASE / 'logs' / name
        if old.exists():
            old.rename(log_archive / name)
    shutil.copy2(BASE / 'reports/paper50k_comparison/comparison_manifest.json',
                 log_archive / 'comparison_manifest_failed.json')
    RECORD.write_text(json.dumps(record, indent=2) + '\n')
    result = subprocess.run([PYTHON, str(BASE / 'scripts/start_paper50k_pipeline.py'),
                             '--start-training'], capture_output=True, text=True)
    record['launcher_returncode'] = result.returncode
    record['launcher_stdout'] = result.stdout
    record['launcher_stderr'] = result.stderr
    record['status'] = 'restarted' if result.returncode == 0 else 'launch_failed'
    RECORD.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    result.check_returncode()


if __name__ == '__main__':
    main()

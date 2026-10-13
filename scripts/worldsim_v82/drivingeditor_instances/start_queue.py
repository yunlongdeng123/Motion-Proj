"""启动独立GMC等待controller，默认plan-only。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--queue-directory', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root = Path(args.queue_directory)
    command = ['nice', '-n', '10', sys.executable, '-B', '-u', str(Path(__file__).with_name('queue_gmc.py')),
               '--spec', args.spec, '--queue-directory', str(root), '--steps', '300', '--cpu-threads', '4', '--run']
    if not args.run:
        print(json.dumps({'status': 'PLAN_ONLY', 'command': command})); return
    if root.exists(): raise FileExistsError('queue目录已存在，不重复controller')
    root.mkdir(parents=True)
    with (root / 'controller.log').open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), start_new_session=True)
    receipt = {'status': 'CONTROLLER_LAUNCHED_WAITING_IS_NOT_GPU_TRAINING', 'pid': child.pid,
               'command': command, 'utc': datetime.now(timezone.utc).isoformat(), 'GPU_started_by_launcher': False,
               'max_wait_hours': 24, 'scope': 'GMC2actor300stepfirstGPUcompatibilitytest', 'other_tasks_modified': False}
    (root / 'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__': main()

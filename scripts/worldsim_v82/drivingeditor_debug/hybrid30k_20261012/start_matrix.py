"""明确--run才启动独立会话；SSH断开不终止固定串行矩阵。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_matrix import ROOT, check_config, check_prepared, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    cli = parser.parse_args()
    check_prepared()
    for arm in ('C00', 'G10', 'N01', 'GN11'):
        check_config(arm)
    if not cli.run:
        print(json.dumps({'status': 'PREPARED_WAITING_FOR_GPU', 'GPU_started': False}))
        return
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / 'controller.log').open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).with_name('run_matrix.py')), '--run'],
            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment)
    receipt = {'status': 'controller_dispatched', 'pid': process.pid,
               'arms': ['C00', 'G10', 'N01', 'GN11'], 'iterations_each': 30000,
               'checkpoints': [10000, 20000, 30000], 'memory_monitor_enabled': True,
               'root': str(ROOT), 'completion_verified': False}
    write(ROOT / 'launch.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()

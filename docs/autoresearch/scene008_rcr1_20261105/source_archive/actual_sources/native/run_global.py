"""单独补回一项global RGB，先跑1K并实际恢复两步。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'
ASSIGNMENT = CAMPAIGN / 'scene008_a6_a37_dynamic_20261104_r1/inputs/init_assignment.npz'
MASKS = CAMPAIGN / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137/dynamic_actor_assets/actor_3/body_masks'


def run(name, mode, checkpoint=None):
    output = ROOT / name
    if output.exists():
        raise RuntimeError(f'拒绝覆盖既有结果 {output}')
    command = [PYTHON, str(ROOT/'source/train_loss_probe.py'), '--mode', mode,
               '--loss-probe-mode', 'global', '--output', str(output),
               '--assignment', str(ASSIGNMENT), '--a6-masks', str(MASKS)]
    if checkpoint:
        command += ['--checkpoint', str(checkpoint), '--extra-updates', '2']
    record = dict(name=name, start_time=time.time(), command=command, status='RUNNING')
    (ROOT/'runner_progress.json').write_text(json.dumps(record, indent=2))
    with (ROOT/f'{name}.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    record.update(status='PASS' if result.returncode == 0 else 'FAIL',
                  returncode=result.returncode, seconds=time.time()-record['start_time'])
    (ROOT/f'{name}_execution.json').write_text(json.dumps(record, indent=2))
    (ROOT/'runner_progress.json').write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f'{name}失败，保留现场与日志')


if __name__ == '__main__':
    with (CAMPAIGN/'gpu_slot.lock').open('a') as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run('global_1k', 'formal')
        run('global_restore_1002', 'resume', ROOT/'global_1k/chkpnt1000.pth')

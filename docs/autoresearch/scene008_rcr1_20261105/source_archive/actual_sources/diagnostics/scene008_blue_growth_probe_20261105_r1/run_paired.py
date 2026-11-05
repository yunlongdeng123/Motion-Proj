"""短程配对增长试验；保留原检查点，每个节点独立保存完整状态。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'
BASE = CAMPAIGN / 'scene008_native_ownership_20261105/run_r3/formal'
ASSIGNMENT = CAMPAIGN / 'scene008_a6_a37_dynamic_20261104_r1/inputs/init_assignment.npz'
MASKS = CAMPAIGN / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137/dynamic_actor_assets/actor_3/body_masks'


def run(name, mode, checkpoint, updates, native_eval=False):
    output = ROOT / name
    if output.exists():
        raise RuntimeError(f'保留既有结果，拒绝覆盖 {output}')
    command = [PYTHON, str(ROOT / 'source/train_growth_probe.py'),
               '--mode', 'resume', '--growth-rgb-mode', mode,
               '--output', str(output), '--checkpoint', str(checkpoint),
               '--assignment', str(ASSIGNMENT), '--a6-masks', str(MASKS),
               '--extra-updates', str(updates)]
    if native_eval:
        command += ['--native-eval-at-end']
    record = dict(name=name, mode=mode, checkpoint=str(checkpoint), extra_updates=updates,
                  command=command, start_time=time.time(), status='RUNNING')
    (ROOT / 'runner_progress.json').write_text(json.dumps(record, indent=2))
    with (ROOT / f'{name}.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    record.update(status='PASS' if result.returncode == 0 else 'FAIL',
                  returncode=result.returncode, seconds=time.time()-record['start_time'])
    (ROOT / f'{name}_execution.json').write_text(json.dumps(record, indent=2))
    (ROOT / 'runner_progress.json').write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f'{name}失败，请查看独立日志')
    return output


def main():
    with (CAMPAIGN / 'gpu_slot.lock').open('a') as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = sys.argv[1]
        if stage == 'preflight':
            first = run('preflight_1003', 'joint', BASE/'chkpnt1000.pth', 3, True)
            run('preflight_restore_1005', 'joint', first/'chkpnt1003.pth', 2)
        elif stage == 'paired_2k':
            run('control_2k', 'control', BASE/'chkpnt1000.pth', 1000)
            run('joint_2k', 'joint', BASE/'chkpnt1000.pth', 1000)
        elif stage == 'paired_3k':
            run('control_3k', 'control', ROOT/'control_2k/chkpnt2000.pth', 1000)
            run('joint_3k', 'joint', ROOT/'joint_2k/chkpnt2000.pth', 1000)
        else:
            raise ValueError(stage)


if __name__ == '__main__':
    main()

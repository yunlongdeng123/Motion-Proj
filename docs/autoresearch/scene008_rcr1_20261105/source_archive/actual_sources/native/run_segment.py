"""从完整状态按1K窗口继续同一global RGB试验。"""
from pathlib import Path
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--updates', type=int, default=1000)
    args = parser.parse_args()
    output = ROOT / args.name
    assert output.parent.resolve() == ROOT.resolve() and not output.exists()
    command = [PYTHON, str(ROOT/'source/train_loss_probe.py'), '--mode', 'resume',
               '--loss-probe-mode', 'global', '--output', str(output),
               '--assignment', str(ASSIGNMENT), '--a6-masks', str(MASKS),
               '--checkpoint', str(args.checkpoint), '--extra-updates', str(args.updates)]
    with (CAMPAIGN/'gpu_slot.lock').open('a') as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = dict(name=args.name, start_time=time.time(), command=command, status='RUNNING')
        (ROOT/'segment_progress.json').write_text(json.dumps(record, indent=2))
        with (ROOT/f'{args.name}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        record.update(status='PASS' if result.returncode == 0 else 'FAIL',
                      returncode=result.returncode, seconds=time.time()-record['start_time'])
        (ROOT/f'{args.name}_execution.json').write_text(json.dumps(record, indent=2))
        (ROOT/'segment_progress.json').write_text(json.dumps(record, indent=2))
        if result.returncode:
            raise RuntimeError(f'{args.name}失败，保留现场')


if __name__ == '__main__':
    main()

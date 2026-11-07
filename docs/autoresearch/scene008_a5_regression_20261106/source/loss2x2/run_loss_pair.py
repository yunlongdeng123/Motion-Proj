"""Run the two registered arms sequentially; preserve native long-run concurrency."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def status(stage, **extra):
    write(ROOT / 'pair_status.json', dict(stage=stage, pid=os.getpid(), updated_unix=time.time(),
          shutdown_allowed=False, gpu_reset_allowed=False, **extra))


def free_mib():
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    assert len(values) == 1
    return int(values[0])


def main():
    assert ROOT.name == 'scene008_a5_static_rgb5_ablation_20261107_r2'
    assert json.loads((ROOT / 'cpu_validation.json').read_text())['status'] == 'PASS_NATIVE_COMPOSER_ONE_WEIGHT_ONLY_CPU'
    for arm in ['control', 'support']:
        directory = ROOT / arm
        assert not (directory / 'queue_status.json').exists(), 'Do not overwrite or automatically replay an existing arm'
        while free_mib() < 12288:
            status('WAITING_FOR_REAL_GPU_HEADROOM', next_arm=arm, free_mib=free_mib(), required_free_mib=12288)
            time.sleep(30)
        before = free_mib()
        with (directory / 'queue.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(directory / 'source/run_loss_arm.py')],
                                       stdout=log, stderr=subprocess.STDOUT)
            status('RUNNING_SINGLE_LOSS_ARM', arm=arm, child_pid=process.pid,
                   free_mib_at_launch=before, minimum_launch_free_mib=12288,
                   serial_new_arms=True, concurrent_native_longrun=True)
            assert process.wait() == 0, (arm, process.returncode)
        result = json.loads((directory / 'queue_status.json').read_text())
        assert result['stage'] == 'COMPLETE_STATIC_RGB5_FRESH5K_ABLATION_ARM'
    status('COMPLETE_STATIC_RGB5_PAIR', new_formal_updates=10000, new_engineering_updates=0)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('FAILED', error=repr(error))
        raise

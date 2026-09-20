"""GT反馈与逐帧参考通过后的三个固定对照；子进程失败即停。"""
import json
from pathlib import Path
import subprocess
import sys
import time
from run_following_closed_loop import OUT


def main():
    path = OUT/'queue_result.json'; assert not path.exists()
    assert json.loads((OUT/'gt_clean/result.json').read_text())['baseline_admitted']
    assert json.loads((OUT/'gt_clean/dense_reference_result.json').read_text())['status'] == 'passed'
    result = {'status': 'started', 'completed': [], 'human_verdict': None, 'failure_ledger_delta': 'none'}
    started = time.monotonic()
    try:
        for arm in ['dvgt_metric', 'dvgt_lidar_scaled', 'reference_lidar']:
            print(json.dumps({'stage': 'generate', 'arm': arm}), flush=True)
            with (OUT/f'{arm}.log').open('w') as log:
                run = subprocess.run([sys.executable, str(Path(__file__).with_name('run_following_closed_loop.py')), '--arm', arm], stdout=log, stderr=subprocess.STDOUT)
            if run.returncode: raise RuntimeError(f'{arm} stopped: returncode={run.returncode}; no further arms')
            with (OUT/f'{arm}.assessment.log').open('w') as log:
                run = subprocess.run([sys.executable, str(Path(__file__).with_name('assess_following_closed_loop.py')), '--arm', arm], stdout=log, stderr=subprocess.STDOUT)
            if run.returncode: raise RuntimeError(f'{arm} reference audit failed: returncode={run.returncode}')
            result['completed'].append(arm)
            path.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'stage': 'complete', 'arm': arm}), flush=True)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed_stopped', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        result['wall_s'] = time.monotonic()-started
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

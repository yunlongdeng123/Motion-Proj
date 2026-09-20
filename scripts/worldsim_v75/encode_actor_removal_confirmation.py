"""为冻结的独立确认初帧编码一次共享文本和图像条件。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from run_prepared import encode


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r1'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(); out = args.output
    q = json.loads((out/'result.json').read_text())
    assert q['status'] in ['passed_pending_raster', 'qualified']
    source = Path(q['selected']['base'])/'initial_rgb.png'
    dest = out/'conditioning'; assert not dest.exists(); dest.mkdir()
    shutil.copy2(source, dest/'initial_rgb.png')
    (dest/'prompt.txt').write_text('A forward-facing driving camera on an urban road in daylight. Vehicles, buildings, road markings and sidewalks are visible.\n')
    result = {'status': 'started', 'phase': 'encode', 'human_verdict': None,
              'world_model_generation_calls': 0, 'failure_ledger_delta': 'none'}
    began = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    try:
        result.update(encode(dest), status='complete')
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(wall_s=time.monotonic()-began,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(dest/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Unified a20 field: CPU preflight, real native-Gaussian smoke, G40/A80/J40 and fresh reload."""
from pathlib import Path
import argparse
import os
import traceback
from joint_core import load_inputs, preflight, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['preflight', 'smoke', 'geometry', 'appearance', 'joint', 'evaluate', 'reload'], default='preflight')
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--prior-contract', type=Path)
    p.add_argument('--context-plan', type=Path)
    p.add_argument('--path-map', action='append', default=[])
    p.add_argument('--path-map-file', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--geometry-gate', type=Path)
    p.add_argument('--asset', type=Path)
    args = p.parse_args()
    if args.mode == 'preflight': os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        inp = load_inputs(args)
        if args.mode == 'preflight':
            report = preflight(args, inp)
            print('CPU_PREFLIGHT_PASSED', report['unified_rows'], flush=True)
            return
        from joint_gpu import execute
        execute(args, inp)
    except Exception as exc:
        write(args.output / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), PID=os.getpid(), mode=args.mode))
        raise


if __name__ == '__main__': main()

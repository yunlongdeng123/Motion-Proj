"""GMC 双实例短测试排队。等待时不导入Torch，不占GPU、不持共享计算锁。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .priority import require_gpu_slot
    from .runtime import validate_spec
else:
    from priority import require_gpu_slot
    from runtime import validate_spec


def write_status(path, value):
    value.update(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def queue(args):
    import fcntl
    spec, report = validate_spec(args.spec)
    if spec['scene_id'] != 'scene0008_023_071' or report['instance_count'] != 2 or not report['full_mask_supervision_ready']:
        raise ValueError('此次只允许真实GMC两个已审查实例和98个真实mask配对')
    root = Path(args.queue_directory)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / 'controller.lock').open('a+')
    status = root / 'status.json'
    output = root / 'gpu_smoke'
    started = time.monotonic()
    owned_child = None
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or (root / 'training.log').exists():
            raise FileExistsError('测试输出已存在，保留旧资产；不自动重跑或续训')
        while time.monotonic() - started < args.max_wait_hours * 3600:
            try:
                slot, admission = require_gpu_slot(args.gpu_lock)
                slot.close()
            except (BlockingIOError, RuntimeError) as busy:
                write_status(status, {'state': 'waiting_for_other_tasks', 'GPU_allocated': False,
                                      'reason': str(busy), 'poll_seconds': args.poll_seconds})
                time.sleep(args.poll_seconds)
                continue
            command = [sys.executable, '-B', str(Path(__file__).with_name('train.py')),
                       '--spec', args.spec, '--output', str(output), '--iterations', str(args.steps),
                       '--gpu-lock', args.gpu_lock, '--run']
            write_status(status, {'state': 'dispatching_GMC_only_GPU_smoke', 'command': command, 'admission': admission})
            environment = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
            # nice降低CPU调度优先级；不调整其他worker的CPU、显存或优先级。
            with (root / 'training.log').open('x', encoding='utf-8') as log:
                owned_child = subprocess.Popen(['nice', '-n', '10', *command], stdout=log, stderr=subprocess.STDOUT, env=environment)
                result = owned_child.wait()
            if result:
                write_status(status, {'state': 'engineering_attention_required', 'returncode': result,
                                      'training_output': str(output), 'auto_retry': False})
                return 1
            real_status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
            if real_status['state'] != 'complete':
                write_status(status, {'state': real_status['state'], 'training_status': real_status,
                                      'automatic_resume': False, 'GPU_released': True})
                return 0
            write_status(status, {'state': 'GMC_GPU_SMOKE_COMPLETE_EDIT_VISUAL_REVIEW_PENDING', 'training_status': real_status,
                                  'checkpoint': str(output / ('checkpoint_%06d.pt' % args.steps)),
                                  'test_scope': 'fixed_trajectory_two_instances; photoreal_edit_acceptance_still_pending',
                                  'GPU_released': True, 'other_tasks_modified': False})
            return 0
        write_status(status, {'state': 'resource_wait_timeout', 'GPU_allocated': False,
                              'max_wait_hours': args.max_wait_hours, 'automatic_retry': False})
        return 3
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--queue-directory', required=True)
    parser.add_argument('--gpu-lock', default='/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--max-wait-hours', type=float, default=24)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or not 30 <= args.poll_seconds <= 3600 or not 1 <= args.cpu_threads <= 4 or not 0 < args.max_wait_hours <= 24:
        parser.error('短测1..1000步、CPU1..4线程、等待不超过24小时')
    validate_spec(args.spec)
    if not args.run:
        print(json.dumps({'status': 'QUEUE_PLAN_ONLY', 'arguments': vars(args), 'GPU_allocated': False}, ensure_ascii=False, indent=2))
        return 0
    return queue(args)


if __name__ == '__main__': raise SystemExit(main())

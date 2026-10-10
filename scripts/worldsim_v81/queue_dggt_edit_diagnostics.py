"""在当前 Seen-to-Scene 12500 六窗退出后，串行运行一次 DGGT 参考修复诊断。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
RUN = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1')
P1 = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1/paper_bidirectional_m4')
OUTPUT = RUN / 'diagnostics/reference_refinement_r1'
STATE = RUN / 'diagnostics/reference_refinement_queue.json'
PYTHON = Path('/root/autodl-tmp/envs/dggt_inference/bin/python')
TRIGGER = 12500


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def gpu_processes():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True)
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def ready():
    """只信正式保存、六窗和真实控制器退出，不凭旧状态启动。"""
    import continue_p1_review_cycles as p1
    later = [p for pattern in ('launch_step*.json', 'states/step*.json')
             for p in (P1/'review_cycles').glob(pattern)
             if int(p.stem.split('step')[-1]) > TRIGGER]
    later += [p for p in (P1/'review_cycles/train').glob('p1-checkpoint-*.pt')
              if int(p.stem.split('-')[-1]) > TRIGGER]
    if later:
        raise RuntimeError('后继S2S段已启动；12500间隙已过，不将旧完成记录当本次空闲')
    status = read(P1/f'review_cycles/states/step{TRIGGER:06d}.json')
    if status.get('status') == 'review_cycle_engineering_failure':
        raise RuntimeError('S2S 当前段工程失败，先修复训练/验证，不越过失败启动 DGGT')
    verified = read(P1/f'review_cycles/checkpoint_step{TRIGGER:06d}_verified.json')
    checkpoint = P1/f'review_cycles/train/p1-checkpoint-{TRIGGER:06d}.pt'
    if not (status.get('status') == 'waiting_assistant_training_review'
            and status.get('training_step') == TRIGGER
            and status.get('checkpoint') == str(checkpoint)
            and verified.get('step') == TRIGGER
            and verified.get('source_step') == TRIGGER - 2500
            and verified.get('updates') == 2500
            and verified.get('protocol_unchanged') is True):
        return False
    if (checkpoint.is_symlink() or not checkpoint.is_file()
            or checkpoint.stat().st_size != verified.get('full_checkpoint_bytes')):
        raise RuntimeError('正式12500完整断点原件与验证记录不匹配')
    p1.fixed_windows(TRIGGER, checkpoint)
    return not p1.active_controllers() and not gpu_processes()


@contextmanager
def wait_for_slot():
    """锁冲突只等待；取得双锁后再次核对生产者和GPU状态。"""
    deadline = time.monotonic() + 12*3600
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError('等待12小时仍未就绪；保存现场，不停 S2S、不越过GPU锁')
        if ready():
            with ExitStack() as stack:
                ctl = stack.enter_context((P1/'controller.lock').open('a'))
                launch = stack.enter_context((P1/'review_cycles/launch.lock').open('a'))
                try:
                    fcntl.flock(ctl, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(launch, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if ready():
                        yield
                        return
        time.sleep(30)


def preparation():
    paths = [REPO/'scripts/worldsim_v81/probe_dggt_reference_refinement.py', PYTHON,
             RUN/'gaussian_edits/manifest.json', RUN/'difix_refined/manifest.json',
             Path('/root/autodl-tmp/dggt/pretrained/diffusion_model.pth')]
    paths += [RUN/f'gaussian_edits/{branch}/{i:03d}.png'
              for branch in ('noop', 'delete') for i in range(4)]
    paths += [RUN/'processed/validation/128/images/000_0.jpg']
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    return not missing, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    prepared, missing = preparation()
    if args.check_only:
        print(json.dumps({'status': 'cpu_queue_check', 'prepared': prepared, 'missing': missing,
                          'trigger_step': TRIGGER, 'ready_for_gpu': ready(),
                          'gpu_processes': gpu_processes(), 'gpu_started': False}))
        return
    if not args.worker:
        raise ValueError('需要显式 --worker；不自动增加 DGGT 或 S2S 训练')
    if not prepared:
        raise FileNotFoundError(f'准备缺文件: {missing}')
    if STATE.exists():
        raise FileExistsError('诊断队列已存在；先核对实际PID、状态、错误和产物，禁止重复启动')
    if (OUTPUT/'run.json').exists():
        raise FileExistsError('已有诊断结果或失败证据，不能覆盖重跑')
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent/'reference_refinement_queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'waiting_s2s_12500_and_six_windows', 'trigger_step': TRIGGER,
                 'controller_pid': os.getpid(), 'child_pid': None, 'started_at': now(),
                 'human_verdict': None, 'training_updates': 0, 'output': str(OUTPUT),
                 's2s_priority': True, 's2s_preempted': False}

        def update(**fields):
            state.update(fields, updated_at=now())
            write(STATE, state)

        try:
            update()
            with wait_for_slot():
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
                env.pop('CUDA_VISIBLE_DEVICES', None)
                argv = [str(PYTHON), str(REPO/'scripts/worldsim_v81/probe_dggt_reference_refinement.py'),
                        '--external', '/root/autodl-tmp/dggt', '--edits-dir', str(RUN/'gaussian_edits'),
                        '--prior-dir', str(RUN/'difix_refined'), '--output-dir', str(OUTPUT)]
                log_path = STATE.parent/'reference_refinement_gpu.log'
                update(status='running_reference_probe', command=argv, log=str(log_path))
                with log_path.open('a') as log:
                    child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                    update(child_pid=child.pid)
                    try:
                        code = child.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                update(surviving_child_pid=child.pid)
                        raise TimeoutError('当前诊断超过30分钟，已仅停止其子作业，保留日志和部分产物')
                update(child_pid=None, returncode=code)
                if code:
                    raise RuntimeError(f'参考诊断返回{code}；先读原栈，不能重复已完成分支')
                result = read(OUTPUT/'run.json')
                if result.get('status') != 'completed':
                    raise RuntimeError('子作业退出但实际完成记录缺失')
                update(probe_complete=True, result_json=str(OUTPUT/'run.json'))
                # 新媒体只做CPU解码/复制和逐帧页；缺助手审核时明确显示待审核。
                builder = [str(Path('/root/autodl-tmp/envs/motionproj/bin/python')),
                           str(REPO/'scripts/worldsim_v81/build_dggt_reference_review.py'),
                           '--run-dir', str(RUN), '--output-dir', str(OUTPUT/'review')]
                delivery_env = env.copy()
                delivery_env['CUDA_VISIBLE_DEVICES'] = ''
                delivery = subprocess.run(builder, env=delivery_env, cwd=REPO,
                                          capture_output=True, text=True, timeout=60)
                (STATE.parent/'reference_delivery_cpu.log').write_text(delivery.stdout+'\n'+delivery.stderr)
                if delivery.returncode:
                    raise RuntimeError('GPU诊断已完成，CPU交付构建失败；仅修复交付，不重复GPU')
                audit = json.loads(delivery.stdout)
                update(status='complete_pending_assistant_review', completed_at=now(),
                       controller_pid=None, result_json=str(OUTPUT/'run.json'),
                       review_html=str(OUTPUT/'review/index.html'), delivery=audit)
        except Exception as exc:
            update(status='engineering_failure', controller_pid=None,
                   error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()

"""等待现有7500控制器结束，再串行执行零更新传播评估；永不启动训练。"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


RUN = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1')
PHASE = RUN / 'paper_bidirectional_m4'
PYTHON = Path('/root/autodl-tmp/envs/motionproj/bin/python')
SCRIPT = Path('/root/autodl-tmp/motion_proj_v81/scripts/worldsim_v81/ablate_p1_step7500_propagation.py')
STATE = RUN / 'evaluation_7500_queue_state.json'
MARKER = RUN / 'evaluation_7500_queue_launch.json'
POLICY = RUN / 'evaluation_7500_policy.json'
OUTPUT = RUN / 'propagation_ablation_step7500'
REPO = Path('/root/autodl-tmp/motion_proj_v81')
REVIEW = Path('/root/autodl-tmp/reviews/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/progress-preview')
SEQUENCES = ('00f88c4f0a', '7e625db8c4', 'ff6eb95840')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def update(**fields):
    payload = read(STATE) if STATE.exists() else {}
    payload.update(fields, updated_at_utc=now(),
                   training_after_7500_allowed=False, optimizer_updates=0)
    if 'controller_pid' not in fields:
        payload['controller_pid'] = os.getpid()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(STATE)


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    stat = Path('/proc') / str(pid) / 'stat'
    if not stat.exists():
        return False
    # /proc/stat names can contain spaces; the state follows the final close paren.
    return stat.read_text().rsplit(')', 1)[1].strip().split()[0] != 'Z'


def gpu_pids():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True)
    return [int(line.strip()) for line in result.splitlines() if line.strip()]


def policy_check():
    policy = read(POLICY)
    if (policy.get('max_formal_training_step') != 7500
            or policy.get('training_after_7500_allowed') is not False
            or policy.get('evaluation_optimizer_updates') != 0):
        raise RuntimeError('用户7500后仅评估的范围未正确落盘')
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)
    ast.parse(SCRIPT.read_text(encoding='utf-8'))


def formal_ready():
    state = read(PHASE / 'controller_state.json')
    status = state.get('status', '')
    if 'failure' in status or 'error' in status:
        raise RuntimeError(f'正式7500控制器失败，需保留证据后修复: {status}')
    if status != 'bounded7500_complete_waiting_assistant_quality':
        return False
    if any(process_alive(state.get(key)) for key in ('controller_pid', 'child_pid')):
        return False
    marker = read(MARKER)
    if any(process_alive(marker.get(key)) for key in ('source_controller_pid', 'source_child_pid')):
        return False
    for sequence in SEQUENCES:
        for side in (.125, .33):
            path = PHASE / 'validation/step007500' / sequence / f'side_{side:g}' / 'run.json'
            if not path.is_file() or read(path).get('status') != 'complete':
                return False
    return not gpu_pids()


def worker():
    deadline = time.monotonic() + 24 * 3600
    try:
        policy_check()
        update(status='waiting_existing_7500_training_and_six_validation_windows', child_pid=None)
        while not formal_ready():
            if time.monotonic() > deadline:
                raise TimeoutError('7500正式控制器24小时内未完成，评估未启动')
            time.sleep(30)
        policy_check()
        if OUTPUT.exists():
            raise FileExistsError(f'传播输出已存在，拒绝重复GPU作业: {OUTPUT}')
        # This script's preflight validates the retained checkpoint and all six protocols.
        checked = subprocess.run([str(PYTHON), str(SCRIPT), '--check-only'],
                                 text=True, capture_output=True, check=True)
        evidence = json.loads(checked.stdout)
        if evidence.get('status') != 'check_only_passed' or evidence.get('cases') != 6:
            raise RuntimeError(f'正式传播预检未通过: {evidence}')
        if gpu_pids():
            raise RuntimeError('启动前GPU已被占用，未启动第二个作业')
        log_path = RUN / 'logs/propagation_ablation_step7500.log'
        with log_path.open('x', encoding='utf-8') as log:
            child = subprocess.Popen([str(PYTHON), str(SCRIPT)], stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            update(status='zero_update_propagation_ablation_running', child_pid=child.pid,
                   log=str(log_path), preflight=evidence, output=str(OUTPUT))
            code = child.wait()
        if code != 0:
            raise RuntimeError(f'传播评估退出码{code}；先检查日志与已完成产物，勿重复启动')
        report = read(OUTPUT / 'run.json')
        if (report.get('status') != 'complete' or report.get('optimizer_updates') != 0
                or report.get('checkpoint_step') != 7500 or len(report.get('cases', [])) != 6):
            raise RuntimeError('传播评估未完整完成六窗零更新')
        update(status='cpu_metrics_and_review_build', child_pid=None)
        metrics = RUN / 'step7500_window_metrics.json'
        cpu_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        subprocess.run([str(PYTHON), str(REPO/'scripts/worldsim_v81/summarize_p1_step7500_windows.py'),
                        '--media', str(RUN), '--output', str(metrics)], check=True, env=cpu_env)
        destination = REVIEW / 'paper_bidirectional_m4/validation/step007500'
        if destination.exists():
            raise FileExistsError(f'审核页新媒体路径已存在，须检查后再恢复CPU交付: {destination}')
        shutil.copytree(PHASE/'validation/step007500', destination)
        shutil.copytree(OUTPUT, REVIEW/'propagation_ablation_step7500',
                        ignore=shutil.ignore_patterns('paired_inputs.pt'))
        shutil.copy2(metrics, REVIEW/metrics.name)
        page = REVIEW / 'step7500_review.html'
        subprocess.run([str(PYTHON), str(REPO/'scripts/worldsim_v81/build_p1_step7500_review.py'),
                        '--media', str(REVIEW), '--propagation-root', str(REVIEW/'propagation_ablation_step7500'),
                        '--output', str(page)], check=True, env=cpu_env)
        update(status='complete_waiting_assistant_full_frame_reviews_and_delivery',
               child_pid=None, controller_pid=None, assistant_review_required=True,
               human_verdict=None, report=str(OUTPUT / 'run.json'),
               window_metrics=str(metrics), remote_review_page=str(page))
    except Exception as exc:
        stack = traceback.format_exc()
        update(status='evaluation_engineering_failure', child_pid=None,
               error=f'{type(exc).__name__}: {exc}', traceback=stack)
        print(stack, flush=True)
        raise


def launch():
    policy_check()
    if MARKER.exists() or STATE.exists() or OUTPUT.exists():
        raise FileExistsError('7500评估队列/产物已有记录，禁止重复启动')
    parent = read(PHASE / 'controller_state.json')
    if parent.get('training_budget') != 7500 or parent.get('source_step') != 5000:
        raise RuntimeError('现有训练控制器不是正式5000→7500')
    with MARKER.open('x', encoding='utf-8') as stream:
        json.dump({'created_at_utc': now(), 'source_controller_pid': parent.get('controller_pid'),
                   'source_child_pid': parent.get('child_pid'), 'max_training_step': 7500,
                   'evaluation_optimizer_updates': 0, 'training_after_7500_allowed': False,
                   'script': str(SCRIPT)}, stream, ensure_ascii=False, indent=2)
    log_path = RUN / 'logs/evaluation_7500_queue_parent.log'
    with log_path.open('x', encoding='utf-8') as log:
        child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker'],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print(json.dumps({'launched_pid': child.pid, 'status': 'waiting_existing_controller',
                      'max_training_step': 7500, 'evaluation_optimizer_updates': 0,
                      'parent_log': str(log_path)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        ast.parse(Path(__file__).read_text(encoding='utf-8'))
        print(json.dumps({'status': 'static_pass', 'gpu_started': False,
                          'training_argv_present': False, 'max_formal_training_step': 7500}))
    elif args.worker:
        worker()
    else:
        launch()


if __name__ == '__main__':
    main()

"""按用户授权串行运行 R30K→RP30K；异常即停，保留现场与所有10K节点。"""

from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_complete(run):
    returned = read(run/'v4_training_returned.json')
    if returned['V1_updates'] != 30000 or returned['road_updates'] != 30000:
        raise RuntimeError('未完成30K V1与道路更新')
    for step in (10000, 20000, 30000):
        node = read(run/f'v4_checkpoint{step}.json')
        if node['V1_updates'] != step or node['road_updates'] != step:
            raise RuntimeError('V4节点更新数不匹配')
        for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state', 'road_chkpnt'):
            path = run/f'{prefix}{step}.pth'
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError('缺少完整 checkpoint：'+str(path))
    return returned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    campaign = options.campaign.resolve()
    configs = [read(campaign/'configs'/f'{arm}.json') for arm in ('R', 'RP')]
    for config in configs:
        if (config['iterations'] != 30000 or config['start_checkpoint'] is not None
                or any(config[key] != [10000, 20000, 30000]
                       for key in ('save_iterations', 'checkpoint_iterations', 'test_iterations'))):
            raise RuntimeError('必须保持共同fresh30K和10K/20K/30K节点')
    print(json.dumps(dict(order=['R', 'RP'], iterations_each=30000,
        checkpoint_steps=[10000, 20000, 30000], execute=options.execute)), flush=True)
    if not options.execute:
        return
    lock = (campaign.parent/'gpu_slot.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    if applications:
        raise RuntimeError('GPU已有任务，不抢占')
    for arm in ('R', 'RP'):
        if (campaign/'arms'/arm/'run30k').exists():
            raise FileExistsError('不覆盖、重复启动或自动从零重跑 '+arm)
    state = dict(status='STARTING', pid=os.getpid(), started_utc=now(),
        process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
        order=['R', 'RP'], completed_arms=[], active_arm=None, training_pid=None,
        checkpoint_steps=[10000, 20000, 30000],
        authorization='user_2026-10-22_serial_R_RP_30k_10k_checkpoints',
        quality_accepted=False, human_verdict=None)
    status_path = campaign/'serial_status.json'
    try:
        for index, arm in enumerate(('R', 'RP')):
            config = configs[index]
            node_bytes = config['densify_until_num_points']*640+config['dynamic_densify_until_num_points']*800+230_000_000
            required = int((2-index)*3*node_bytes*1.15+8e9)
            free = shutil.disk_usage(campaign).free
            if free < required:
                raise RuntimeError(f'剩余存储不足以保留全部节点：{free} < {required}')
            command = [sys.executable, str(campaign/'tools/run_v4_arm.py'),
                       '--campaign', str(campaign), '--arm', arm, '--execute']
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            log_path = campaign/'arms'/arm/'training.log'
            with log_path.open('xb') as log:
                process = subprocess.Popen(command, cwd=campaign, env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
                state.update(status='RUNNING', active_arm=arm, training_pid=process.pid,
                    arm_started_utc=now(), log=str(log_path), free_bytes_at_arm_start=free,
                    required_bytes_at_arm_start=required)
                write(status_path, state)
                code = process.wait()
            if code:
                raise RuntimeError(f'{arm}进程异常退出，exit={code}，不启动下一组')
            check_complete(campaign/'arms'/arm/'run30k')
            state['completed_arms'].append(arm)
            state.update(active_arm=None, training_pid=None, last_arm_completed_utc=now())
            write(status_path, state)
        state.update(status='COMPLETE_R_RP_30K_CHECKPOINTS_VERIFIED', completed_utc=now())
        write(status_path, state)
    except BaseException:
        state.update(status='FAILED_STOPPED_NO_AUTOMATIC_REPLAY', failed_utc=now(), error=traceback.format_exc())
        write(status_path, state)
        raise


if __name__ == '__main__':
    main()

"""用户授权2+2资源调度；采用正在运行的full，不暂停或重跑其训练。"""

import argparse
import datetime
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from run_v3_four_arms import command
from v3_gpu_authorization import require_gpu_authorization
from v3_pair_runtime import PAIR_POLICY, PAIR_VERSION, read, write, process, matches_worker, change_lease, validate_pair_plan


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'], text=True)
    free, used = map(int, output.strip().split(','))
    return dict(free_bytes=free*1024**2, used_bytes=used*1024**2)


def completed(run):
    state = read(run/'status.json')
    return (state.get('status') == 'complete' and state.get('iteration') == 30000
            and all(state.get('optimizer_step_counts', {}).get(key) == 30000 for key in ['static', 'dynamic', 'environment']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--execute-after-user-gpu-approval', action='store_true')
    parser.add_argument('--adopt-full-pid', type=int, required=True)
    parser.add_argument('--replace-controller-pid', type=int, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(); plan = read(campaign/'plan.json')
    approval = require_gpu_authorization(campaign/'plan.json', plan, args.execute_after_user_gpu_approval)
    validate_pair_plan(plan, approval)
    if (campaign/'pair_registry.json').exists():
        raise FileExistsError('pair controller already attempted; no automatic restart')
    full = process(args.adopt_full_pid); old = process(args.replace_controller_pid)
    state = read(campaign/'arms/full/run30k/status.json')
    gate = read(campaign/'arms/full/gate_001/gpu_gate_receipt.json')
    old_argv = old['argv'] if old else []
    old_matches = (any(Path(part).name == 'run_v3_four_arms.py' for part in old_argv)
                   and '--campaign' in old_argv and old_argv[old_argv.index('--campaign')+1] == str(campaign)
                   and Path('/proc', str(args.replace_controller_pid), 'cwd').resolve() == campaign)
    if (not matches_worker(full, campaign, 'full', 'train') or not old or not old_matches
            or full.get('ppid') != args.replace_controller_pid
            or state['pid'] != args.adopt_full_pid or state.get('status') not in {'training', 'evaluating'}
            or gate.get('optimizer_updates') != 0
            or gate.get('road_version') != plan['road_version']):
        raise RuntimeError('live full/controller/gate identity cannot be adopted')
    for arm in ['real_only', 'prior_only', 'ownership_only']:
        if any((campaign/'arms'/arm/name).exists() for name in ['gate_001', 'run30k']):
            raise FileExistsError('non-fresh pending arm: '+arm)
    if shutil.disk_usage(campaign).free < plan['minimum_campaign_free_bytes']:
        raise RuntimeError('campaign disk admission incomplete')
    backup = campaign/'queue_pairs_r1'; backup.mkdir()
    for name in ['four_arm_registry.json', 'status.json', 'plan.json', 'GPU_APPROVAL.json']:
        shutil.copy2(campaign/name, backup/name)
    # 只终止旧控制器PID，不对进程组发信号；subprocess.run不会把该信号转发给full。
    os.kill(old['pid'], signal.SIGTERM)
    for _ in range(50):
        if not process(old['pid']):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('old controller did not yield; full untouched')
    after = process(full['pid'])
    if not after or after['start_ticks'] != full['start_ticks']:
        raise RuntimeError('adopted full exited during handoff; inspect without replay')
    controller_lock = (campaign/'controller.lock').open('a+')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own = process(os.getpid())
    lease = dict(version=PAIR_VERSION, coordinator_pid=os.getpid(), coordinator_start_ticks=own['start_ticks'],
        batch_index=0, global_guard=dict(pid=full['pid'], start_ticks=full['start_ticks'], kind='inherited_full'),
        slots={'full': dict(pid=full['pid'], start_ticks=full['start_ticks'], phase='train')})
    write(campaign/'pair_lease.json', lease)
    guard_ready = threading.Event()
    guard_handle = (Path(plan['gpu_lock'])).open('a+')
    def inherit_guard():
        fcntl.flock(guard_handle, fcntl.LOCK_EX)
        change_lease(campaign, lambda value: value.update(global_guard=dict(
            pid=os.getpid(), start_ticks=own['start_ticks'], kind='coordinator_exclusive')))
        guard_ready.set()
    # full现有全局锁不动；内核等待者在full结束释放后接管，期间real_only使用同一cohort准入。
    threading.Thread(target=inherit_guard, daemon=True).start()
    registry = dict(status='RUNNING_TWO_PAIRS', version=PAIR_VERSION, policy=PAIR_POLICY,
        coordinator_pid=os.getpid(), adopted_full_pid=full['pid'], replaced_controller_pid=old['pid'],
        adopted_full_iteration_snapshot=state['iteration'], adopted_full_optimizer_counts=state['optimizer_step_counts'],
        full_training_restarted=False, started_utc=utc(), completed_arms=[], arm_phases={'full':'train'},
        training_updates_planned=120000, automatic_restart=False, human_verdict=None)
    def save():
        registry['updated_utc'] = utc()
        write(campaign/'pair_registry.json', registry); write(campaign/'status.json', registry)
    save()
    failure = None
    def start(arm, mode):
        nonlocal failure
        resources = memory()
        if resources['free_bytes'] < PAIR_POLICY['minimum_launch_free_bytes']:
            registry['waiting_admission'] = dict(arm=arm, mode=mode, **resources)
            save(); return None
        registry.pop('waiting_admission', None)
        def admit(value):
            value['slots'][arm] = dict(pid=None, phase=mode)
        change_lease(campaign, admit)
        destination = campaign/'arms'/arm/('gate_001' if mode == 'gate' else 'run30k')
        log = destination.with_name(destination.name+'.log')
        if destination.exists() or log.exists():
            raise FileExistsError('never overwrite gate/train output: '+str(destination))
        gate_path = campaign/'arms'/arm/'gate_001/gpu_gate_receipt.json' if mode == 'train' else None
        with log.open('x') as handle:
            child = subprocess.Popen(command(campaign, plan, arm, mode, destination, gate_path),
                cwd=campaign, stdout=handle, stderr=subprocess.STDOUT)
        job = dict(pid=child.pid, phase=mode, child=child)
        active[arm] = job
        registry['arm_phases'][arm] = mode; save()
        return job
    try:
        for batch_index, batch in enumerate(PAIR_POLICY['batches']):
            if batch_index:
                while not guard_ready.wait(1.):
                    registry.update(status='WAITING_GLOBAL_GPU_GUARD', batch_index=batch_index); save()
                change_lease(campaign, lambda value: value.update(batch_index=batch_index, slots={}))
            registry.update(status='RUNNING_TWO_PAIRS', batch_index=batch_index, current_batch=batch)
            active = {'full':dict(pid=full['pid'], phase='train', child=None)} if batch_index == 0 else {}
            pending = [arm for arm in batch if arm not in active]
            waiting_train = []
            while active or pending or waiting_train:
                for arm, job in list(active.items()):
                    code = job['child'].poll() if job['child'] is not None else (None if process(job['pid']) else 0)
                    if code is None:
                        continue
                    del active[arm]
                    if code != 0:
                        failure = f'{arm}/{job["phase"]} exited {code}'
                    elif job['phase'] == 'gate':
                        receipt = read(campaign/'arms'/arm/'gate_001/gpu_gate_receipt.json')
                        if receipt.get('status') != 'PASSED_ACTUAL_CUDA_V3_FORWARD_BACKWARD_ZERO_UPDATES' or receipt.get('optimizer_updates') != 0:
                            failure = f'{arm} CUDA gate did not pass'
                        elif not failure:
                            waiting_train.append(arm)
                    elif not completed(campaign/'arms'/arm/'run30k'):
                        failure = f'{arm} terminal state is not exact30K'
                    else:
                        registry['completed_arms'].append(arm); registry['arm_phases'][arm] = 'complete'
                    change_lease(campaign, lambda value: value['slots'].pop(arm, None))
                if failure:
                    pending.clear(); waiting_train.clear()
                    registry.update(status='FAILED_DRAINING_HEALTHY_PEERS', error=failure)
                elif len(active) < 2:
                    # 同时最多一个gate；另一个槽可继续正常训练/保存/评估。
                    candidates = waiting_train if waiting_train else (pending if not any(j['phase'] == 'gate' for j in active.values()) else [])
                    if candidates:
                        arm = candidates[0]; mode = 'train' if candidates is waiting_train else 'gate'
                        job = start(arm, mode)
                        if job:
                            candidates.pop(0); active[arm] = job
                registry['active'] = {arm:dict(pid=job['pid'], phase=job['phase']) for arm,job in active.items()}
                registry['completed_optimizer_updates_in_finished_arms'] = 30000*len(registry['completed_arms'])
                save()
                if active or pending or waiting_train:
                    time.sleep(2.)
            if failure:
                break
    except BaseException as error:
        # 不杀健康训练，不自动重试；保留控制器错误给心跳/用户处理。
        registry.update(status='COORDINATOR_ERROR_PEERS_NOT_TERMINATED', error=repr(error)); save()
        while any(process(job['pid']) for job in active.values()):
            time.sleep(2.)
        raise
    registry.update(status='FAILED_PRESERVED_NO_RESTART' if failure else 'FOUR_ARMS_TRAINING_COMPLETE_EVALUATION_PENDING',
        current_batch=None, active={}, completed_utc=utc(), quality_accepted=False)
    save()
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

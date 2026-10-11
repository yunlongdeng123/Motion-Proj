"""正式P1每次续训2500步；整千保存，助手复盘后才允许下一段。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from run_p1 import Controller, PYTHON, REPO, atomic_json


RUN = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1')
PHASE = RUN / 'paper_bidirectional_m4'
CYCLES = PHASE / 'review_cycles'
TRAIN = CYCLES / 'train'
INITIAL = PHASE / 'checkpoint_retained_step007500/p1-checkpoint-007500.pt'
REVIEW = Path('/root/autodl-tmp/reviews/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/progress-preview')
SEQUENCES = ('00f88c4f0a', '7e625db8c4', 'ff6eb95840')
SIDES = (.125, .33)
VALID = Path('/root/autodl-tmp/data/worldsim_v81/youtube_vos_2019/valid/JPEGImages')
FORMAT = 'worldsim_v81_seen_to_scene_p1_youtube_vos'
STABLE_ARGS = ('data_root', 'svd', 'raft_weight', 'fcnet_weight', 'external',
               'seed', 'lr', 'amp', 'propagation_protocol', 'raft_iters', 'raft_pair_chunk')
FULL_KEYS = ('models', 'optimizer', 'scheduler', 'scaler', 'torch_rng', 'cuda_rng',
             'numpy_rng', 'python_rng', 'cfg_rng', 'data_profile', 'args')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_checkpoint(path):
    import torch
    if path.is_symlink() or not path.is_file() or path.name.endswith('.tmp'):
        raise ValueError(f'需要完整断点原件: {path}')
    obj = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (obj.get('format') != FORMAT or obj.get('optimizer_name') != 'Adam'
            or obj.get('propagation_protocol') != 'paper-bidirectional-m4'
            or any(key not in obj for key in FULL_KEYS)
            or set(obj['models']) != {'unet_temporal', 'fcnet', 'propagator'}):
        raise ValueError('不是正式Adam/双向m4完整训练断点；不接受诊断models-only')
    if path.name != f"p1-checkpoint-{obj['step']:06d}.pt":
        raise ValueError('断点文件名与真实步数不一致')
    args = obj['args']
    if (args['seed'] != 123 or args['lr'] != 1e-5 or args['amp'] != 'bf16'
            or args['raft_iters'] != 20 or args['raft_pair_chunk'] != 2
            or obj['scheduler']['last_epoch'] != obj['step']):
        raise ValueError('正式训练配置/调度器步数不符')
    return obj


def formal_path(path):
    """允许当前正式断点；排除诊断训练权重、外部模型与P0。"""
    exact = {INITIAL.resolve(), Path('/root/v81_checkpoint_retained/paper_bidirectional_m4_step005000/p1-checkpoint-005000.pt')}
    return path.resolve() in exact or path.resolve().parent in {
        TRAIN.resolve(), (PHASE / 'train').resolve()}


def gate_check(gate, step, checkpoint, comparison_step=None):
    expected = {(seq, side) for seq in SEQUENCES for side in SIDES}
    if (gate.get('step') != step or gate.get('reviewer') != 'assistant'
            or gate.get('human_verdict') is not None
            or gate.get('train_decision') not in ('continue', 'diagnose', 'complete')
            or gate.get('quality_decision') not in ('continue', 'hold')
            or Path(gate.get('checkpoint', '')).resolve() != checkpoint.resolve()
            or not isinstance(gate.get('reason'), str) or not gate['reason'].strip()
            or (comparison_step is not None and gate.get('comparison_step') != comparison_step)):
        raise ValueError('助手收益决策的步数/来源/理由/权限不符')
    cases = gate.get('cases', [])
    if (len(cases) != 6 or {(c.get('sequence_id'), c.get('side_ratio')) for c in cases} != expected
            or any(c.get('frames_reviewed') != 25 or c.get('native_reviewed') is not True
                   or c.get('composite_reviewed') is not True or not c.get('evidence') for c in cases)):
        raise ValueError('需要同一六窗原生及写回全部150帧的真实助手复盘')
    for case in cases:
        expected_run = PHASE/f'validation/step{step:06d}'/case['sequence_id']/f"side_{case['side_ratio']:g}/run.json"
        if Path(case.get('run_json', '')).resolve() != expected_run.resolve():
            raise ValueError('逐窗收益决策未绑定当前同序列/倍率的run.json')
        evidence = Path(case['evidence'])
        if not evidence.is_file() or evidence.resolve().parent != expected_run.resolve().parent:
            raise ValueError('每例复盘证据须位于当前相同序列/倍率目录')
        reviewed = read(evidence)
        if (reviewed.get('step') != step or reviewed.get('sequence_id') != case['sequence_id']
                or reviewed.get('side_ratio') != case['side_ratio']
                or reviewed.get('reviewer') != 'assistant' or reviewed.get('frames_reviewed') != 25
                or reviewed.get('native_reviewed') is not True
                or reviewed.get('composite_reviewed') is not True):
            raise ValueError('实际助手记录未逐例绑定全帧覆盖')
    return gate


def fixed_windows(step, checkpoint):
    cases = []
    for sequence in SEQUENCES:
        expected_frames = [str(p) for p in sorted((VALID/sequence).iterdir())
                           if p.suffix.lower() in ('.jpg', '.jpeg', '.png')][:25]
        if len(expected_frames) != 25:
            raise ValueError('固定valid原始帧不足25')
        for side in SIDES:
            path = PHASE/f'validation/step{step:06d}'/sequence/f'side_{side:g}/run.json'
            obj = read(path)
            if (obj.get('status') != 'complete' or obj.get('checkpoint_step') != step
                    or Path(obj.get('checkpoint', '')).resolve() != checkpoint.resolve()
                    or obj.get('data_root') != str(VALID) or obj.get('source_frames') != expected_frames
                    or obj.get('side_ratio_each') != side or obj.get('num_frames') != 25
                    or obj.get('seed') != 2026 or obj.get('steps') != 25
                    or obj.get('mode') != 'paper-feedforward' or obj.get('amp') != 'bf16'
                    or obj.get('propagation_protocol') != 'paper-bidirectional-m4'):
                raise ValueError(f'固定验证来源或协议变化: {path}')
            for kind in ('gt', 'visible', 'pred', 'comp'):
                folder = Path(obj['output_paths'][kind]['frames'])
                if (len(list(folder.glob('*.png'))) != 25
                        or not Path(obj['output_paths'][kind]['mp4']).is_file()):
                    raise ValueError('每窗四组25PNG/视频未齐')
            cases.append({'sequence_id': sequence, 'side_ratio': side, 'run_json': str(path)})
    return cases


def gpu_pids():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def active_controllers():
    tokens = ('v81_run_bounded_', 'queue_p1_step7500_evaluation.py --worker',
              'continue_p1_review_cycles.py --worker', 'probe_dggt_reference_refinement.py')
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            command = (folder/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, ProcessLookupError):
            continue
        if any(token in command for token in tokens):
            found.append({'pid': int(folder.name), 'command': command})
    return found


def expected_saves(source, target):
    return sorted(set(range((source//1000 + 1)*1000, target+1, 1000)) | {target})


def checkpoint_users(rows):
    """兼顾CPU读取和mmap；不把GPU空闲等同于断点无人使用。"""
    identities = {(row['device'], row['inode']) for row in rows}
    paths = {row['realpath'] for row in rows}
    users = set()
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            for fd in (folder/'fd').iterdir():
                try:
                    stat = fd.stat()
                    if (stat.st_dev, stat.st_ino) in identities:
                        users.add(int(folder.name))
                except OSError:
                    pass
            mappings = (folder/'maps').read_text()
            if any(line.split(maxsplit=5)[-1] in paths for line in mappings.splitlines()
                   if len(line.split(maxsplit=5)) == 6):
                users.add(int(folder.name))
        except (OSError, ProcessLookupError):
            continue
    return sorted(users)


def preflight(source, target):
    campaign = CYCLES/'campaign_step020000_policy.json'
    policy = read(campaign) if campaign.exists() else {}
    if policy.get('training_resume_authorized') is False or policy.get('status') == 'paused_by_user':
        raise ValueError('用户已暂停Seen-to-Scene并优先DGGT；重新明确授权前禁止启动训练')
    maximum = min(20000, policy.get('maximum_step', 20000))
    if target > maximum:
        raise ValueError(f'当前用户授权训练上限为{maximum}；不得自动启动下一段')
    # 新授权的 DGGT 短诊断只占当前12500复盘间隙；不抢占在途训练。
    diagnostic = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1/diagnostics/reference_refinement_queue.json')
    if diagnostic.is_file():
        pending = read(diagnostic)
        if pending.get('status') in ('waiting_s2s_12500_and_six_windows', 'running_reference_probe'):
            command_file = Path(f"/proc/{pending.get('controller_pid')}/cmdline")
            try:
                command = command_file.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                command = ''
            if ('queue_dggt_edit_diagnostics.py --worker' in command
                    and target >= pending.get('trigger_step', 12500) + 2500):
                raise RuntimeError('已授权DGGT参考诊断等待当前复盘间隙；先完成短诊断后续S2S，不抢占在途作业')
    if not formal_path(source):
        raise ValueError('恢复路径不属于明确的正式断点目录')
    obj = load_checkpoint(source)
    step = obj['step']
    if step < 7500 or target != step + 2500 or target > 100000:
        raise ValueError('每次仅授权从已复盘正式点新增2500；首轮7500→10000')
    if step == 7500 and source.resolve() != INITIAL.resolve():
        raise ValueError('首轮只接受已审过的正式7500原件')
    if step != 7500:
        previous = read(CYCLES/f'states/step{step:06d}.json')
        verified = read(CYCLES/f'checkpoint_step{step:06d}_verified.json')
        if (previous.get('status') != 'waiting_assistant_training_review'
                or previous.get('checkpoint') != str(source) or verified.get('step') != step
                or verified.get('protocol_unchanged') is not True):
            raise ValueError('上一段正式保存/六窗未完整完成')
        gate = gate_check(read(CYCLES/f'gates/step{step:06d}.json'), step, source,
                          previous['comparison_step'])
        if gate['train_decision'] != 'continue':
            raise ValueError('上一复盘未明确决定继续，禁止自动追加训练')
        if step % 5000 == 0 and read(CYCLES/f'cleanup/step{step:06d}.json').get('status') != 'complete':
            raise ValueError('全局5000倍数须完成旧断点清理，才可进入下一段')
    fixed_windows(step, source)
    for key in ('data_root', 'svd', 'raft_weight', 'fcnet_weight', 'external'):
        if not Path(obj['args'][key]).exists():
            raise FileNotFoundError(obj['args'][key])
    if (TRAIN/f'p1-checkpoint-{target:06d}.pt').exists():
        raise FileExistsError('目标完整断点已存在；先检查已有阶段，不重复更新')
    saves = expected_saves(step, target)
    free = shutil.disk_usage(PHASE).free
    required = source.stat().st_size * len(saves) + 2_000_000_000
    if free < required:
        raise OSError(f'完整断点与审核媒体空间不足: free={free}, required={required}')
    active = active_controllers()
    if active or gpu_pids():
        raise RuntimeError(f'仍有控制器或GPU作业，未启动: {active}')
    selection = read(RUN/'validation_selection.json')
    # 后续推理继续使用已冻结的valid路径与同三个序列。
    valid_root = VALID
    if tuple(selection['sequence_ids']) != SEQUENCES:
        raise ValueError('固定验证名单变化')
    return {'status': 'preflight_pass', 'source': str(source), 'source_step': step,
            'target_step': target, 'expected_checkpoint_steps': saves,
            'data_free_bytes': free, 'required_bytes': required,
            'valid_root': str(valid_root), 'training_args': {k: obj['args'][k] for k in STABLE_ARGS}}


def verify_target(source, checkpoint, target):
    before, after = load_checkpoint(source), load_checkpoint(checkpoint)
    if after['step'] != target or any(before['args'][key] != after['args'][key] for key in STABLE_ARGS):
        raise ValueError('新完整断点步数或训练协议变化')
    if before['data_profile'] != after['data_profile']:
        raise ValueError('训练数据profile变化')
    for module, tensors in before['models'].items():
        current = after['models'][module]
        if tensors.keys() != current.keys() or any(t.shape != current[k].shape for k, t in tensors.items()):
            raise ValueError(f'参数名/形状变化: {module}')
    if before['optimizer']['param_groups'] != after['optimizer']['param_groups']:
        raise ValueError('Adam参数组或学习率变化')
    old, new = before['optimizer']['state'], after['optimizer']['state']
    if old.keys() != new.keys():
        raise ValueError('Adam状态参数名单变化')
    for key, item in new.items():
        previous = int(old[key]['step'].item())
        current = int(item['step'].item())
        if current < previous or current > previous + target - before['step']:
            raise ValueError(f'Adam参数{key}计数非法；条件丢弃可合法不递增')
    return {'step': target, 'full_checkpoint_bytes': checkpoint.stat().st_size,
            'source_step': before['step'], 'adam_states': len(new), 'protocol_unchanged': True,
            'scheduler_step': after['scheduler']['last_epoch'], 'rng_keys_present': True}


class ReviewController(Controller):
    def __init__(self, target):
        self.cycle_state = CYCLES/f'states/step{target:06d}.json'
        self.cycle_state.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(PHASE, 'paper-bidirectional-m4', 'paper-feedforward')

    def update(self, **changes):
        super().update(**changes)
        atomic_json(self.cycle_state, self.state)


def worker(source, target):
    evidence = preflight(source, target)
    CYCLES.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    backup = CYCLES/'prior_controller_states'
    backup.mkdir(exist_ok=True)
    if (PHASE/'controller_state.json').exists():
        shutil.copy2(PHASE/'controller_state.json', backup/f'{stamp}.json')
    ctl = ReviewController(target)
    try:
        ctl.update(training_budget=target, source_step=evidence['source_step'],
                   checkpoint_source=str(source), target_step=target, save_every=1000,
                   cleanup_every=5000, review_every=2500, human_verdict=None,
                   next_training_requires_assistant_benefit_decision=True, preflight=evidence,
                   training_after_7500_allowed=True, research_continues=True,
                   automatic_100k=False, power_policy='keep_autodl_running')
        atomic_json(CYCLES/'continuation_policy.json', {
            'authorized_at': '2026-10-10', 'first_source_step': 7500, 'first_target_step': 10000,
            'save_every': 1000, 'review_every': 2500, 'cleanup_every_global': 5000,
            'keep_latest_and_previous_review_checkpoint': True, 'human_in_loop': False,
            'training_after_7500_allowed': True, 'automatic_100k': False,
            'power_policy': 'keep_autodl_running', 'updated_at_utc': now()})
        TRAIN.mkdir(exist_ok=True)
        argv = [str(PYTHON), '-m', 'motion_proj.worldsim_v81.train_p1',
                '--output-dir', str(TRAIN), '--resume', str(source), '--max-steps', str(target),
                '--save-every', '1000', '--keep-checkpoints', '100000']
        for key, value in evidence['training_args'].items():
            argv += ['--'+key.replace('_', '-'), str(value)]
        ctl.command(f'train_to_{target:06d}', argv)
        checkpoint = TRAIN/f'p1-checkpoint-{target:06d}.pt'
        verification = verify_target(source, checkpoint, target)
        for step in evidence['expected_checkpoint_steps']:
            saved = TRAIN/f'p1-checkpoint-{step:06d}.pt'
            obj = load_checkpoint(saved)
            if obj['step'] != step:
                raise ValueError('整千保存缺失')
            del obj
        gc.collect()
        log = PHASE/f'logs/train_to_{target:06d}.log'
        events = []
        for line in log.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'train_step':
                events.append(row)
        if [e['step'] for e in events] != list(range(evidence['source_step']+1, target+1)):
            raise ValueError('新增更新不是精确连续2500步；保留日志排查')
        if any(e['frozen_grad_tensors'] != 0 or not math.isfinite(e['loss'])
               or any(g['status'] not in ('ok', 'skipped_condition_dropout')
                      or g['finite'] is False
                      or (g['status'] == 'ok' and g['finite'] is not True)
                      for g in e['gradients'].values()) for e in events):
            raise ValueError('训练数值或冻结梯度异常')
        verification.update(updates=len(events), finite_loss=True, frozen_grad_tensors=0,
                            mean_seconds_per_step=sum(e['wall_seconds'] for e in events)/len(events))
        atomic_json(CYCLES/f'checkpoint_step{target:06d}_verified.json', verification)
        ctl.update(training_step=target, checkpoint=str(checkpoint), checkpoint_verified=verification)
        output = PHASE/f'validation/step{target:06d}'
        for sequence in SEQUENCES:
            for side in SIDES:
                ctl.infer(checkpoint, Path(evidence['valid_root']), sequence, side, output,
                          f'validation_{target:06d}_{sequence}_{side}')
        cases = fixed_windows(target, checkpoint)
        destination = REVIEW/f'paper_bidirectional_m4/validation/step{target:06d}'
        if destination.exists():
            raise FileExistsError('审核媒体目的路径已存在，先核对避免覆盖')
        shutil.copytree(output, destination)
        ctl.update(status='waiting_assistant_training_review', child_pid=None,
                   controller_pid=None, comparison_step=evidence['source_step'], cases=cases,
                   gate_file=str(CYCLES/f'gates/step{target:06d}.json'),
                   remote_review_media=str(destination), training_beyond_target_started=False)
    except Exception as exc:
        ctl.update(status='review_cycle_engineering_failure', child_pid=None,
                   error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise


def cleanup(target):
    """只删除明确清单的旧正式文件；软链与hardlink逐项记录，不递归删除。"""
    campaign = CYCLES/'campaign_step020000_policy.json'
    if campaign.exists() and (read(campaign).get('training_resume_authorized') is False
                              or read(campaign).get('status') == 'paused_by_user'):
        raise ValueError('用户暂停期间保留现有完整断点，不执行旧训练周期清理')
    if target % 5000:
        raise ValueError('只在全局5000倍数清理')
    checkpoint = TRAIN/f'p1-checkpoint-{target:06d}.pt'
    state = read(CYCLES/f'states/step{target:06d}.json')
    if state.get('status') != 'waiting_assistant_training_review' or len(state.get('cases', [])) != 6:
        raise ValueError('先完成目标完整保存及六窗')
    comparison = state['comparison_step']
    source = Path(state['checkpoint_source'])
    verify_target(source, checkpoint, target)
    fixed_windows(target, checkpoint)
    gate_check(read(CYCLES/f'gates/step{target:06d}.json'), target, checkpoint, comparison)
    if active_controllers() or gpu_pids():
        raise RuntimeError('训练/推理仍在使用断点，先不清理')
    manifest = CYCLES/f'cleanup/step{target:06d}.json'
    if manifest.exists():
        previous_plan = read(manifest)
        if previous_plan.get('status') == 'complete':
            print(json.dumps(previous_plan, ensure_ascii=False), flush=True)
            return
        if previous_plan.get('status') != 'planned':
            raise ValueError('已有清理清单状态未知，未做删除')
    else:
        previous_plan = None
    candidates = []
    for folder in (TRAIN, PHASE/'train'):
        candidates += list(folder.glob('p1-checkpoint-*.pt'))
    candidates += [INITIAL, Path('/root/v81_checkpoint_retained/paper_bidirectional_m4_step005000/p1-checkpoint-005000.pt')]
    keep = {checkpoint.resolve(), source.resolve()}
    rows = []
    for path in sorted(set(candidates)):
        if not path.exists() or path.resolve() in keep:
            continue
        real = path.resolve()
        if not formal_path(path):
            raise ValueError(f'清理路径越界: {path}')
        # 诊断副本与P0的目录不扫描；正式train旧文件有hardlink时不夸大释放量。
        obj = load_checkpoint(real)
        if obj['step'] >= target:
            del obj
            continue
        stat = path.stat()
        rows.append({'path': str(path), 'realpath': str(real), 'step': obj['step'],
                     'symlink': path.is_symlink(), 'inode': stat.st_ino, 'device': stat.st_dev,
                     'links': stat.st_nlink, 'bytes': stat.st_size, 'deleted': False})
        del obj
    gc.collect()
    users = checkpoint_users(rows)
    if users:
        raise RuntimeError(f'旧断点仍被CPU/内存映射使用，暂不删除: {users}')
    manifest.parent.mkdir(exist_ok=True)
    before_data, before_root = shutil.disk_usage(PHASE).free, shutil.disk_usage('/root').free
    payload = {'status': 'planned', 'step': target, 'kept': sorted(map(str, keep)),
               'created_at_utc': now(), 'rows': rows, 'data_free_before': before_data,
               'root_free_before': before_root}
    if previous_plan is not None:
        if previous_plan['step'] != target or previous_plan['kept'] != payload['kept']:
            raise ValueError('中断清单与当前保留点不一致')
        payload = previous_plan
        rows = payload['rows']
        before_data, before_root = payload['data_free_before'], payload['root_free_before']
    atomic_json(manifest, payload)
    # 先移除软链接，再移除原件；否则原件删除会使待删除软链无法stat。
    rows.sort(key=lambda row: (not row['symlink'], row['path']))
    for row in rows:
        path = Path(row['path'])
        if not formal_path(path) or path.resolve() in keep:
            raise ValueError('中断清单的待删路径超范围或落在保留点')
        if row['deleted']:
            if os.path.lexists(path):
                raise RuntimeError('已删路径出现新的文件，不触碰')
            continue
        if not os.path.lexists(path):
            row.update(deleted=True, recovered_missing_after_interruption=True)
            atomic_json(manifest, payload)
            continue
        stat = path.stat()
        if (str(path.resolve()), stat.st_ino, stat.st_dev, path.is_symlink()) != (
                row['realpath'], row['inode'], row['device'], row['symlink']):
            raise RuntimeError('清理计划之后路径变化；保留清单停止')
        if checkpoint_users([row]):
            raise RuntimeError('待删断点出现新的读取进程，保留计划停止')
        path.unlink()
        row['deleted'] = True
        atomic_json(manifest, payload)
    payload.update(status='complete', completed_at_utc=now(),
                   data_free_after=shutil.disk_usage(PHASE).free,
                   root_free_after=shutil.disk_usage('/root').free)
    payload['data_free_delta'] = payload['data_free_after'] - before_data
    payload['root_free_delta'] = payload['root_free_after'] - before_root
    atomic_json(manifest, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def launch(source, target):
    CYCLES.mkdir(exist_ok=True)
    with (CYCLES/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = preflight(source, target)
        marker = CYCLES/f'launch_step{target:06d}.json'
        if marker.exists():
            raise FileExistsError('本段已有launch记录，先核对PID/断点，禁止重复启动')
        log = PHASE/f'logs/review_cycle_parent_{target:06d}.log'
        with log.open('x', encoding='utf-8') as output:
            child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker',
                                      '--resume', str(source), '--target', str(target)],
                                     cwd=REPO, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(marker, {**evidence, 'launched_pid': child.pid, 'parent_log': str(log),
                             'launched_at_utc': now()})
        print(json.dumps(read(marker), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, default=INITIAL)
    parser.add_argument('--target', type=int, default=10000)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--worker', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--cleanup-reviewed', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(args.resume, args.target), ensure_ascii=False))
    elif args.worker:
        worker(args.resume, args.target)
    elif args.cleanup_reviewed:
        cleanup(args.target)
    else:
        launch(args.resume, args.target)


if __name__ == '__main__':
    main()

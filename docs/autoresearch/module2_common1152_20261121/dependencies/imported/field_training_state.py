"""冻结世界材质的完整训练状态；不创建场景、GPU 上下文或后台任务。"""
from pathlib import Path
import copy
import hashlib
import json
import random
import time

import numpy as np
import torch


ARM_SPEC = {
    'M00': ('avg', 'observed', False),
    'M10': ('avg', 'teacher', False),
    'M01': ('pixel', 'observed', False),
    'M11': ('pixel', 'teacher', False),
    'T11': ('pixel', 'teacher', True),
}


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = str((tuple(value.shape), str(value.dtype))).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def rng_state(device):
    result = dict(torch=torch.get_rng_state().clone(), numpy=copy.deepcopy(np.random.get_state()),
                  python=copy.deepcopy(random.getstate()))
    if torch.device(device).type == 'cuda':
        result['CUDA'] = cpu_clone(torch.cuda.get_rng_state_all())
    return result


def restore_rng(saved, device):
    torch.set_rng_state(saved['torch'])
    np.random.set_state(copy.deepcopy(saved['numpy']))
    random.setstate(copy.deepcopy(saved['python']))
    if torch.device(device).type == 'cuda':
        if 'CUDA' not in saved:
            raise ValueError('CUDA checkpoint 缺完整 RNG')
        torch.cuda.set_rng_state_all(saved['CUDA'])
    elif 'CUDA' in saved:
        raise ValueError('CPU/CUDA checkpoint 角色不同')


def balanced_schedule(length, seed=101):
    if length % 2:
        raise ValueError('固定采样预算须为偶数')
    sampler, trace = random.Random(seed), []
    for _ in range(length // 2):
        pair = [86, 110]
        sampler.shuffle(pair)
        trace.extend(pair)
    return trace


def max_tree_difference(a, b, path='root'):
    """完整 snapshot 对比；非数值身份必须 exact，浮点返回实际最大差。"""
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('tensor 身份不同: ' + path)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('nonfinite snapshot: ' + path)
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('numpy 身份不同: ' + path)
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError('dict 身份不同: ' + path)
        return max((max_tree_difference(a[k], b[k], path + '.' + str(k)) for k in a), default=0.)
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('sequence 身份不同: ' + path)
        return max((max_tree_difference(x, y, path + '.' + str(i)) for i, (x, y) in enumerate(zip(a, b))), default=0.)
    if a != b:
        raise ValueError('scalar/metadata 不同: ' + path)
    return 0.


def rgb_l1(prediction, reference, mask):
    if prediction.shape != reference.shape or prediction.shape[0] != 3 or mask.shape != prediction.shape[1:]:
        raise ValueError('固定监督 shape 不同')
    denominator = 3 * mask.sum()
    if not bool(denominator > 0):
        raise ValueError('固定源监督分母为零')
    return (prediction - reference).abs()[:, mask].sum() / denominator


class TrainingState:
    """reader 只负责渲染/读取；仅 model.low/high 是优化参数。"""
    def __init__(self, model, reader, targets, identity, optimizer_spec, *, role, arm,
                 stage_counts=(200, 300, 300), seed=101, loss_weights=None):
        if role not in ('smoke', 'formal') or arm not in (*ARM_SPEC, 'common_avg', 'common_pixel'):
            raise ValueError('训练角色或 arm 错误')
        if role == 'formal' and tuple(stage_counts) != (200, 300, 300):
            raise ValueError('正式阶段预算冻结为 200/300/300')
        self.model, self.reader, self.targets = model, reader, targets
        self.identity = copy.deepcopy(identity)
        self.role, self.arm, self.stage_counts = role, arm, tuple(int(x) for x in stage_counts)
        if any(x <= 0 for x in self.stage_counts):
            raise ValueError('阶段预算须为正数')
        self.optimizer_spec = copy.deepcopy(optimizer_spec)
        self.seed = int(seed)
        self.loss_weights = dict(A=1., C=.5, S=1.) if loss_weights is None else dict(loss_weights)
        if self.loss_weights != dict(A=1., C=.5, S=1.):
            raise ValueError('源权重合同不同')
        self.schedule = balanced_schedule(sum(self.stage_counts), seed)
        self.trace, self.cursor = [], 0
        self.optimizer_step_invocations = 0  # 实际调用单调累计；restore 不回退计算账。
        self.optimizer = torch.optim.Adam([
            dict(params=[model.low], lr=optimizer_spec['low_lr']),
            dict(params=[model.high], lr=optimizer_spec['high_lr'])],
            betas=tuple(optimizer_spec['betas']), eps=optimizer_spec['eps'],
            weight_decay=optimizer_spec.get('weight_decay', 0.))
        self.frozen_signature = {key: tensor_sha(value) for key, value in model.named_buffers()}
        self.frozen_versions = {key: value._version for key, value in model.named_buffers()}
        self.cursor_stage = 'A'
        self.model.set_stage('A')

    def planned_stage(self):
        if self.cursor >= len(self.schedule):
            raise StopIteration('固定预算已完成')
        return 'A' if self.cursor < self.stage_counts[0] else 'B' if self.cursor < sum(self.stage_counts[:2]) else 'C'

    def objective(self, raw):
        packet = self.reader.render(self.model, int(raw), joint=False)
        pred = packet['road_rgb']
        target = self.targets[int(raw)]
        alpha = target['fixed_alpha'][None]
        terms = {key: rgb_l1(pred, alpha * target['target_' + key], target[key]) for key in ('A', 'C', 'S')}
        reg = self.model.regularizers()
        if set(reg) != set(self.optimizer_spec['regularizers']):
            raise ValueError('正则名称与冻结 P 合同不同')
        total = sum(self.loss_weights[key] * terms[key] for key in terms)
        total = total + sum(self.optimizer_spec['regularizers'][key] * reg[key] for key in reg)
        return total, dict(**terms, **reg)

    def step(self):
        self.reader.require_execution(self.role)
        stage, raw = self.planned_stage(), self.schedule[self.cursor]
        self.cursor_stage = stage
        self.model.set_stage(stage)
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss, terms = self.objective(raw)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite loss')
        loss.backward()
        gradients = {}
        for name in ('low', 'high'):
            parameter = getattr(self.model, name)
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError('nonfinite 世界场梯度: ' + name)
            gradients[name] = 0. if parameter.grad is None else float(parameter.grad.abs().max())
        # B 冻 L 参数和 Adam moments；A 冻 H。requires_grad=False 且 grad=None。
        frozen = self.model.high if stage == 'A' else self.model.low if stage == 'B' else None
        if frozen is not None and frozen.grad is not None:
            raise RuntimeError('冻结阶段参数出现梯度')
        self.optimizer.step()
        self.optimizer_step_invocations += 1
        self.cursor += 1
        self.trace.append(raw)
        self.assert_frozen()
        return dict(cursor=self.cursor, stage=stage, raw=raw, loss=float(loss.detach()),
                    gradients=gradients, elapsed_seconds=time.perf_counter() - started,
                    **{key: float(value.detach()) for key, value in terms.items()})

    def snapshot(self):
        return dict(schema='teacher_detail_complete_state.v1', role=self.role, arm=self.arm,
                    identity=copy.deepcopy(self.identity), params=cpu_clone(dict(low=self.model.low, high=self.model.high)),
                    frozen_signature=self.frozen_signature.copy(), settings=dict(low_stride=self.model.low_stride,
                        low_bound=self.model.low_bound, high_bound=self.model.high_bound),
                    optimizer_spec=copy.deepcopy(self.optimizer_spec), optimizer=cpu_clone(self.optimizer.state_dict()),
                    schedule=self.schedule.copy(), stage_counts=self.stage_counts, seed=self.seed,
                    cursor=self.cursor, trace=self.trace.copy(), stage=self.cursor_stage,
                    logical_updates=self.cursor, RNG=cpu_clone(rng_state(self.model.low.device)))

    def restore(self, saved):
        if saved.get('schema') != 'teacher_detail_complete_state.v1':
            raise ValueError('checkpoint schema 不同')
        for key, expected in [('role', self.role), ('arm', self.arm), ('identity', self.identity),
                              ('frozen_signature', self.frozen_signature), ('optimizer_spec', self.optimizer_spec),
                              ('stage_counts', self.stage_counts), ('schedule', self.schedule)]:
            if saved[key] != expected:
                raise ValueError('checkpoint ' + key + ' 不同')
        expected_settings = dict(low_stride=self.model.low_stride, low_bound=self.model.low_bound,
                                 high_bound=self.model.high_bound)
        if saved['settings'] != expected_settings or saved['seed'] != self.seed:
            raise ValueError('checkpoint bounds/stride/seed 不同')
        cursor = saved['cursor']
        if not isinstance(cursor, int) or not 0 <= cursor <= len(self.schedule):
            raise ValueError('checkpoint cursor 非法')
        if saved['trace'] != self.schedule[:cursor] or saved['logical_updates'] != cursor:
            raise ValueError('checkpoint sampler/逻辑步数不完整')
        groups = saved['optimizer']['param_groups']
        if len(groups) != 2 or [g['lr'] for g in groups] != [self.optimizer_spec['low_lr'], self.optimizer_spec['high_lr']]:
            raise ValueError('Adam group/LR 不同')
        if [g['params'] for g in groups] != [[0], [1]]:
            raise ValueError('Adam 参数身份不同')
        for index, adam in saved['optimizer']['state'].items():
            if index not in (0, 1):
                raise ValueError('Adam 未知参数')
            shape = self.model.low.shape if index == 0 else self.model.high.shape
            for key, value in adam.items():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise ValueError('Adam 非有限值')
                if key in ('exp_avg', 'exp_avg_sq') and value.shape != shape:
                    raise ValueError('Adam moment shape 不同')
        with torch.no_grad():
            for key in ('low', 'high'):
                target, source = getattr(self.model, key), saved['params'][key]
                if target.shape != source.shape or not bool(torch.isfinite(source).all()):
                    raise ValueError('参数 shape/finite 不同')
                target.copy_(source.to(target.device))
        # 必须 deepcopy：Adam CPU step tensor 不得别名污染调用方磁盘 snapshot。
        self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        restore_rng(saved['RNG'], self.model.low.device)
        self.cursor, self.trace = cursor, saved['trace'].copy()
        self.cursor_stage = saved['stage']
        self.model.set_stage(self.cursor_stage)
        self.assert_frozen()

    def fork_common_A(self, common, *, high_seed=None):
        """只拷参数/Adam/RNG，绝不 load_state_dict 覆盖 arm 的 H_gate。"""
        if common['cursor'] != self.stage_counts[0] or common['stage'] != 'A':
            raise ValueError('不是指定共同 A 完整端点')
        if common['role'] != self.role or common['identity'] != self.identity or common['schedule'] != self.schedule:
            raise ValueError('共同 A 来源/采样/角色不同')
        if common['arm'] not in ('common_avg', 'common_pixel') or common['arm'] != 'common_' + self.reader.kind:
            raise ValueError('共同 A reader 不同')
        if bool((common['params']['high'] != 0).any()) or common['optimizer']['state'].get(1):
            raise ValueError('共同 A 必须 H=0 且 AdamH 未使用')
        current = self.snapshot()
        shared = copy.deepcopy(common)
        shared['arm'] = self.arm
        shared['frozen_signature'] = current['frozen_signature']
        # 除 H 输出门外，底色/valid/latent/正则/low support 必须逐值相同。
        for key, value in common['frozen_signature'].items():
            if key != 'H_gate' and current['frozen_signature'][key] != value:
                raise ValueError('共同 A 非许可字段不同: ' + key)
        self.restore(shared)
        if high_seed is not None:
            seed = torch.as_tensor(high_seed, device=self.model.high.device, dtype=self.model.high.dtype)
            if seed.shape != self.model.high.shape or not bool(torch.isfinite(seed).all()):
                raise ValueError('T11 seed shape/finite 不同')
            with torch.no_grad():
                self.model.high.copy_(seed)
        self.model.set_stage('B')
        self.cursor_stage = 'B'

    def assert_frozen(self):
        for key, value in self.model.named_buffers():
            if value._version != self.frozen_versions[key]:
                raise RuntimeError('冻结 field buffer 改变: ' + key)
        self.reader.assert_frozen()

    def save(self, directory, metadata=None):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        self.assert_frozen()
        torch.save(self.snapshot(), output / 'checkpoint.pt')
        record = dict(metadata or {}, role=self.role, arm=self.arm, reader=self.reader.kind,
                      identity=self.identity, cursor=self.cursor, renderer_mode=self.reader.renderer_mode,
                      authoritative_world_field=True, independent_row_RGB_parameters=False)
        self.model.save(output / 'field.pt', record)
        (output / 'trace.json').write_text(json.dumps(dict(cursor=self.cursor, trace=self.trace,
            schedule=self.schedule, stage_counts=self.stage_counts, physical_step_invocations_this_instance=self.optimizer_step_invocations), indent=2) + '\n')
        (output / 'renderer.json').write_text(json.dumps(self.reader.manifest(), indent=2) + '\n')
        return output

"""Scene008 分阶段拟合及可恢复的真实环境 smoke。"""

import argparse
import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from instance_renderer import CAMPAIGN, GROUPS, PVD_FILE, ROAD_RUN, InstanceRenderer
from object_layer import ObjectLayer

PILOTS = (105, 113, 130)
PROBE = CAMPAIGN / 'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(x, y) for x, y in zip(left, right))
    return left == right


def read_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def mean_valid(value, mask):
    selected = value[..., mask]
    return selected.sum() / max(selected.numel(), 1)


def photo(delta):
    return torch.sqrt(delta.square() + 1e-6) - .001


class Experiment:
    def __init__(self, bank, settings):
        self.bank = bank
        self.settings = settings
        self.objects = torch.nn.ModuleDict({name: ObjectLayer.from_measurements(bank.asset(int(name[1:])))
                                            for name in ('a6', 'a37')})
        self.pvd = ObjectLayer.from_fixed_pvd(dict(np.load(PVD_FILE, allow_pickle=False)), self.objects['a6'])
        self.objects.cuda()
        self.pvd.cuda()
        self.renderer = InstanceRenderer(bank, self.objects, self.pvd)
        self.depth_scale = self.renderer.scale
        self.baselines = {}
        self.replacement = None
        self.replacement_identity = None
        self.fit_scope = None
        self.schedule_steps = settings['steps_c']
        self.identity = dict(native=self.renderer.n.before, inputs={str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in (PVD_FILE, ROAD_RUN/'formal/checkpoint001000.pt', *sorted(bank.root.rglob('*.npz')))})

    @lru_cache(maxsize=4)
    def tensor_frame(self, raw):
        from evaluate_scene008 import _road_edit
        frame = self.bank.frame(raw)
        if raw not in self.renderer.poses:
            self.renderer.poses[raw] = torch.as_tensor(frame['model_from_local'], dtype=torch.float32, device='cuda')
        result = {name: torch.as_tensor(frame[name], device='cuda') for name in
                  ('visible', 'depth_m', 'depth_valid', 'free', 'protect', 'road')}
        result['rgb'] = torch.as_tensor(frame['rgb'], device='cuda').permute(2, 0, 1)
        result['road_edit'] = torch.as_tensor(_road_edit(frame), device='cuda')
        result['road_edit_rgb'] = torch.as_tensor(frame['road_edit_rgb'], device='cuda').permute(2, 0, 1)
        return result

    def instance_terms(self, out, d, keep=(0, 1)):
        losses = {}
        for index in keep:
            name = GROUPS[index]
            weight = out['component_alpha_T'][index]
            # epsilon 放在 log 内，允许弱贡献获得梯度。没有投影支持时仍需独立覆盖项。
            losses[f'{name}_joint'] = mean_valid(-(weight + 1e-5).log(), d['visible'][index])
            depth = out['component_depth_sum'][index] / (weight.clamp_min(1e-5) * self.depth_scale)
            valid = d['depth_valid'][index] & d['visible'][index] & (weight.detach() > .05)
            losses[f'{name}_joint_depth'] = mean_valid(
                F.smooth_l1_loss(depth, d['depth_m'][index], reduction='none', beta=.15), valid)
        return losses

    def step_b(self, raw, joint):
        d = self.tensor_frame(raw)
        losses = {}
        for index, name in enumerate(('a6', 'a37')):
            out = self.renderer.render(raw, f'{name}_only', check=True)
            alpha = out['alpha'][0]
            # 孤立渲染使用黑背景。用直通颜色拟合，显式要求高覆盖。
            colour = out['rgb_premultiplied'] / alpha.clamp_min(.05)
            visible = d['visible'][index]
            losses[f'{name}_rgb'] = mean_valid(photo(colour - d['rgb']), visible & (alpha.detach() > .05))
            losses[f'{name}_coverage'] = mean_valid((1 - alpha).square(), visible)
            losses[f'{name}_free'] = mean_valid(alpha.square(), d['free'][index])
            depth = out['depth_sum'][0] / (alpha.clamp_min(1e-5) * self.depth_scale)
            losses[f'{name}_depth'] = mean_valid(F.smooth_l1_loss(depth, d['depth_m'][index], reduction='none', beta=.15),
                                               d['depth_valid'][index] & visible & (alpha.detach() > .05))
            losses[f'{name}_anchor'] = self.objects[name].anchor_loss()
        if joint:
            out = self.renderer.render(raw, variant='objects', check=True)
            losses.update(self.instance_terms(out, d))
        weights = self.settings['loss_b']
        total = sum(weights[key.split('_', 1)[1]] * value for key, value in losses.items())
        return total, losses

    def baseline(self, raw, state):
        key = raw, state
        if key not in self.baselines:
            with torch.no_grad():
                out = self.renderer.render(raw, state, variant='strong_baseline')
                self.baselines[key] = out['rgb'].cpu()
        return self.baselines[key].cuda()

    def step_c(self, raw, step):
        d = self.tensor_frame(raw)
        state = 'unedited' if step % 2 else 'delete_a37'
        out = self.renderer.render(raw, state, variant='replacement', check=True)
        losses = self.instance_terms(out, d, (0, 1) if state == 'unedited' else (0,))
        reference = self.baseline(raw, state)
        reveal = torch.zeros_like(d['protect'])
        if state == 'delete_a37':
            reveal = torch.as_tensor(self.reveal[raw], dtype=torch.bool, device='cuda')
            weight = out['component_alpha_T'][0]
            losses['reveal'] = mean_valid(-(weight + 1e-5).log(), reveal)
            target_z = torch.as_tensor(self.reveal_z[raw], dtype=torch.float32, device='cuda')
            z = out['component_depth_sum'][0] / (weight.clamp_min(1e-5) * self.depth_scale)
            losses['reveal_depth'] = mean_valid(F.smooth_l1_loss(z, target_z, reduction='none', beta=.25),
                                                reveal & (weight.detach() > .05))
            losses['road_edit'] = mean_valid(photo(out['rgb'] - d['road_edit_rgb']), d['road_edit'] & ~reveal)
        losses['protect'] = mean_valid(photo(out['rgb'] - reference), d['protect'] & ~reveal)
        losses['road'] = mean_valid(photo(out['rgb'] - reference), d['road'] & ~d['visible'].any(0) & ~reveal)
        losses['gate_prior'] = self.replacement.regularizer()
        losses['old_a37'] = 1 - self.replacement.dynamic_retirement.clamp(0, 1)
        cfg = self.settings['loss_c']
        return sum(cfg[key.split('_', 1)[1] if key.startswith(('a6_', 'a37_')) else key] * value
                   for key, value in losses.items()), losses

    def attach_ledger(self, path, cpu_ledger, projections):
        from local_replacement import PersistentReplacement, load_ledger
        count = len(self.renderer.n.static.get_xyz)
        base = load_ledger(cpu_ledger, static_count=count)
        ledger = load_ledger(path, static_count=count, checkpoint_sha256=base['checkpoint_sha256'].item(),
                             road_checkpoint_sha256=base['road_checkpoint_sha256'].item())
        if not np.array_equal(ledger['ids'], base['ids']) or not np.array_equal(ledger['road_keep'], base['road_keep']):
            raise ValueError('归因账本的候选 ID 或既有道路门控改变')
        self.replacement = PersistentReplacement(ledger, count).cuda()
        self.renderer.replacement = self.replacement
        self.replacement_identity = {str(p.resolve()): (p.stat().st_size, p.stat().st_mtime_ns)
                                     for p in (path, cpu_ledger, projections)}
        with np.load(projections, allow_pickle=False) as data:
            self.reveal = {raw: data[f'reveal_{raw:03d}'] for raw in self.bank.raw_ids}
            self.reveal_z = {raw: data[f'reveal_z_{raw:03d}'] for raw in self.bank.raw_ids}

    def optimizer(self, stage):
        self.objects.requires_grad_(stage == 'B')
        if stage == 'B':
            return torch.optim.Adam([
                {'params': [layer.colour_logit for layer in self.objects.values()], 'lr': self.settings['lr_b_colour']},
                {'params': [p for layer in self.objects.values() for p in (layer.offset, layer.log_size, layer.coverage_logit)],
                 'lr': self.settings['lr_b_shape']}])
        return torch.optim.Adam(self.replacement.parameters(), lr=self.settings['lr_c'])

    def state(self, stage, step, opt, sampler):
        return dict(stage=stage, step=step, settings=self.settings, identity=self.identity,
                    schedule_steps=self.schedule_steps, fit_scope=self.fit_scope,
                    replacement_identity=self.replacement_identity,
                    objects=self.objects.state_dict(), pvd=self.pvd.state_dict(),
                    replacement=self.replacement.state_dict() if self.replacement is not None else None,
                    optimizer=opt.state_dict() if opt is not None else None,
                    rng=rng_state(), sampler=sampler.getstate() if sampler is not None else None)

    def load(self, state, opt=None, sampler=None):
        if state['identity'] != self.identity or state['settings'] != self.settings:
            raise ValueError('恢复源或训练配置发生改变')
        self.objects.load_state_dict(state['objects'], strict=True)
        self.pvd.load_state_dict(state['pvd'], strict=True)
        if state['replacement'] is not None:
            if state['replacement_identity'] != self.replacement_identity:
                raise ValueError('恢复用退役账本或固定重投影证据发生改变')
            # 候选 ID/来源不得由另一份账本悄悄覆盖，具体绑定由账本比较负责。
            self.replacement.verify_state_binding(state['replacement'])
            self.replacement.load_state_dict(state['replacement'], strict=True)
        if opt is not None:
            if state['fit_scope'] != self.fit_scope:
                raise ValueError('续跑采样视图或 smoke / 正式范围发生改变')
            opt.load_state_dict(state['optimizer'])
            sampler.setstate(state['sampler'])
            restore_rng(state['rng'])
            if not equal(state, self.state(state['stage'], state['step'], opt, sampler)):
                raise RuntimeError('完整训练状态恢复与检查点不一致')
        else:
            self.fit_scope = state['fit_scope']

    def save(self, destination, stage, step, opt=None, sampler=None):
        temporary = destination.with_suffix('.tmp')
        torch.save(self.state(stage, step, opt, sampler), temporary)
        os.replace(temporary, destination)
        return destination

    def fit(self, stage, destination, until, *, start=None, resume=False, smoke=False, schedule_steps=None):
        seed(self.settings['seed'])
        sampler = random.Random(self.settings['seed'])
        fit_raws = tuple(self.bank.raw_ids)
        if not fit_raws:
            raise ValueError('没有可采样视角')
        if smoke:
            fit_raws = PILOTS
        self.schedule_steps = schedule_steps or self.settings['steps_c']
        if start is not None and not resume:
            previous = read_checkpoint(start)
            if stage != 'C' or previous['stage'] != 'B':
                raise ValueError('跨阶段初始化只允许 B 进入 C')
            if previous['fit_scope']['smoke'] != smoke:
                raise ValueError('B 与 C 必须同属 smoke 或正式训练')
            self.load(previous)
        self.fit_scope = dict(smoke=smoke, raw_ids=fit_raws)
        opt = self.optimizer(stage)
        first = 0
        if resume:
            state = read_checkpoint(start)
            if state['stage'] != stage:
                raise ValueError('续跑检查点必须来自同一阶段')
            if stage == 'C' and state['schedule_steps'] != self.schedule_steps:
                raise ValueError('续跑的 a37 接管进度表发生改变')
            self.load(state, opt, sampler)
            first = state['step']
        if until <= first:
            raise ValueError('目标步数必须大于已完成步数')
        if stage == 'C' and until > self.schedule_steps:
            raise ValueError('阶段 C 超出预定的接管步数')
        destination.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        isolated_steps = 1 if smoke else self.settings['b_isolated_steps']
        with (destination/'training.jsonl').open('x', encoding='utf8') as log:
            for step in range(first + 1, until + 1):
                raw = sampler.choice(fit_raws)
                opt.zero_grad(set_to_none=True)
                total, losses = self.step_b(raw, step > isolated_steps) if stage == 'B' else self.step_c(raw, step)
                if not bool(total.isfinite()):
                    raise FloatingPointError(f'{stage}/{step} raw{raw} loss={float(total.detach())}')
                total.backward()
                gradients = [p.grad for g in opt.param_groups for p in g['params'] if p.grad is not None]
                if not gradients or not all(bool(g.isfinite().all()) for g in gradients):
                    raise FloatingPointError(f'{stage}/{step} 梯度缺失或非有限')
                opt.step()
                if stage == 'C':
                    self.replacement.project_parameters()
                    with torch.no_grad():
                        fraction = min(step / self.schedule_steps, 1.)
                        self.replacement.dynamic_retirement.clamp_(min=fraction)
                        if step == self.schedule_steps:
                            self.replacement.retire_dynamic()
                row = dict(stage=stage, step=step, raw=raw, loss=float(total.detach()),
                           elapsed_s=time.monotonic() - started, max_cuda_bytes=torch.cuda.max_memory_allocated(),
                           terms={key: float(value.detach()) for key, value in losses.items()})
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                if step == first + 1 or step % 50 == 0:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                if step == until or step % self.settings['checkpoint_every'] == 0:
                    self.save(destination/f'checkpoint{step:06d}.pt', stage, step, opt, sampler)
        if not self.renderer.base.unchanged():
            raise RuntimeError('父检查点或冻结静态模型发生改变')
        return destination/f'checkpoint{until:06d}.pt'

    @torch.no_grad()
    def packet(self, variant, state, raw):
        from evaluate_scene008 import make_packet
        out = self.renderer.render(raw, state, variant=variant, check=True)
        if out is None:
            return None
        packet = make_packet(out, self.depth_scale, GROUPS)
        if variant == 'strong_baseline':
            packet['weights']['a6'] = None
            packet['component_depth_m'].pop('a6', None)
        return packet

    def evaluate(self, output, raw_ids, groups):
        from evaluate_scene008 import evaluate
        evaluate(self.bank, self.packet, output, raw_ids=raw_ids, groups=groups,
                 probe_path=PROBE if self.settings['fixed_probe'] else None)

    def attribute(self, cpu_ledger, projection_path, output, raws):
        from local_replacement import ReplacementAttribution, frame_attribution, load_ledger
        count = len(self.renderer.n.static.get_xyz)
        ledger = load_ledger(cpu_ledger, static_count=count)
        with np.load(projection_path, allow_pickle=False) as archive:
            projections = dict(archive)
        attribution = ReplacementAttribution(ledger, projections, raw_ids=raws)
        ids = torch.as_tensor(ledger['ids'], dtype=torch.long, device='cuda')
        started = time.monotonic()
        for raw in raws:
            frame = self.tensor_frame(raw)
            for include, state in ((True, 'unedited'), (False, 'delete_a37')):
                marker = torch.zeros((len(ids), 3), device='cuda', requires_grad=True)
                dense = marker.new_zeros((count, 3)).index_copy(0, ids, marker)
                raster = self.renderer.render(raw, state, variant='native_attribution', marker=dense, check=True)
                d6, d37 = frame['visible']
                guard = frame['protect'] | (frame['road'] & ~frame['visible'].any(0))
                if not include:
                    reveal = torch.as_tensor(projections[f'reveal_{raw:03d}'], device='cuda', dtype=torch.bool)
                    d6 = d6 | reveal
                    guard = (guard | frame['road_edit']) & ~reveal
                domains = [d6, d37 if include else torch.zeros_like(d37), guard]
                score = frame_attribution(marker, raster['marker'], domains).cpu().numpy()
                attribution.add(raw, include, score)
                del marker, dense, raster
            print(f'raw{raw} attributed in both edit states', flush=True)
        result = attribution.finish()
        if not np.any(result['max_retirement'] > 0):
            raise RuntimeError('本次原生贡献归因没有得到可退出的旧静态行')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            np.savez_compressed(stream, **result)
        write(output.with_suffix('.json'), dict(raw_ids=list(raws), candidates=len(ids),
              eligible=int((result['max_retirement'] > 0).sum()), mixed=int(result['mixed'].sum()),
              elapsed_s=time.monotonic() - started))

    def final_pvd(self, checkpoint, output):
        state = read_checkpoint(checkpoint)
        if state['stage'] != 'C':
            raise ValueError('PVD 阶段必须从 C 的候选接管场景开始')
        self.load(state)
        if not bool(self.replacement.dynamic_retired):
            raise ValueError('旧 a37 尚未在两个编辑状态中退出')
        self.objects.requires_grad_(False)
        self.pvd.update_inferred_colour(self.objects['a6'])
        output.mkdir(parents=True, exist_ok=False)
        return self.save(output/'checkpoint000000.pt', 'D', 0)

    @torch.no_grad()
    def validate_native_replay(self, output):
        from PIL import Image
        differences = {}
        for raw in PILOTS:
            out = self.renderer.render(raw, 'delete_a37', variant='strong_baseline', check=True)
            rgb = out['rgb'].cpu().numpy().transpose(1, 2, 0)
            pixels = np.uint8(np.rint(np.clip(rgb, 0, 1)*255))
            path = ROAD_RUN/f'evaluation/all49/{raw:03d}_repaired.png'
            with Image.open(path) as reference:
                differences[raw] = int(np.abs(pixels.astype(np.int16) - np.asarray(reference)).max())
        write(output, differences)
        if max(differences.values()) > 1:
            raise RuntimeError('强编辑基线未复现归档 PNG')


def preflight(bank, settings, path, projections):
    from local_replacement import load_ledger, sha256
    from evaluate_scene008 import FixedProbes
    probes = FixedProbes(PROBE) if settings['fixed_probe'] else None
    data = {}
    for raw in bank.raw_ids:
        frame = bank.frame(raw)
        for key in ('visible', 'occluded', 'free', 'depth_valid'):
            if frame[key].shape != (2, 320, 480):
                raise ValueError(f'raw{raw} {key} 形状错误')
        if (frame['visible'][0] & frame['visible'][1]).any():
            raise ValueError(f'raw{raw} 实例可见监督重叠')
        if (frame['visible'] & (frame['occluded'] | frame['free'])).any() or (frame['occluded'] & frame['free']).any():
            raise ValueError(f'raw{raw} 观测三种状态冲突')
        if not np.isfinite(frame['depth_m']).all() or not np.isfinite(frame['rgb']).all():
            raise ValueError(f'raw{raw} 观测包含非有限数值')
        if probes is not None:
            probes.for_frame(raw, frame)
        data[raw] = {key: frame[key].sum((1, 2)).astype(int).tolist() for key in ('visible', 'occluded', 'free', 'depth_valid')}
    layers = {name: ObjectLayer.from_measurements(bank.asset(int(name[1:]))) for name in ('a6', 'a37')}
    pvd = ObjectLayer.from_fixed_pvd(dict(np.load(PVD_FILE, allow_pickle=False)), layers['a6'])
    for name, layer in layers.items():
        pose = bank.frame(113)['model_from_local'][GROUPS.index(name)]
        camera = type('CpuCamera', (), {'world_view_transform': torch.as_tensor(bank.frame(113)['model_to_camera'].T, dtype=torch.float32)})()
        pipe = type('CpuPipe', (), {'compute_cov3D_python': False})()
        rows = layer.raster_rows(torch.tensor(pose, dtype=torch.float32), camera, pipe, 16, False)
        loss = rows['means3D'].square().mean() + rows['scales'].mean() + rows['opacities'].mean() + rows['shs'].square().mean()
        loss.backward()
        if not all(p.grad is not None and bool(p.grad.isfinite().all()) for p in layer.parameters()):
            raise RuntimeError(f'{name} CPU 锚点参数梯度异常')
    ledger = load_ledger(path)
    if ledger['observation_fingerprint'].item() != sha256(bank.root/'receipt.json'):
        raise ValueError('CPU 候选账本与观测库不匹配')
    if np.any(ledger['max_retirement'] > 0):
        raise ValueError('CPU 候选账本意外开放了额外退役权限')
    reveal_pixels = 0
    with np.load(projections, allow_pickle=False) as projection:
        for raw in bank.raw_ids:
            mask, z = (projection[f'{name}_{raw:03d}'] for name in ('reveal', 'reveal_z'))
            if mask.shape != (320, 480) or z.shape != mask.shape or not np.isfinite(z).all() or not np.array_equal(z > 0, mask):
                raise ValueError(f'raw{raw} 固定重投影证据无效')
            reveal_pixels += int(mask.sum())
    return dict(status='PASS', eval_frames=len(data), observations=data, object_points={k: len(v.anchor) for k,v in layers.items()},
                pvd_generated_points=len(pvd.anchor), ledger_rows=len(ledger['ids']), reveal_witness_pixels=reveal_pixels, gpu_calls=0)


def main():
    from object_observations import ObservationBank
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'smoke', 'attribute', 'train-b', 'train-c', 'attach-pvd', 'evaluate'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--confirm-gpu', action='store_true')
    parser.add_argument('--anchors', action='store_true')
    parser.add_argument('--smoke-minutes', type=float, default=10.)
    parser.add_argument('--groups', nargs='+', default=['strong_baseline', 'objects', 'replacement', 'pvd'])
    args = parser.parse_args()
    if args.resume and (args.checkpoint is None or args.action not in ('train-b', 'train-c')):
        parser.error('--resume 需要训练入口和同阶段检查点')
    root, action = args.root, args.action
    settings = json.loads((root/'source/settings.json').read_text(encoding='utf8'))
    bank = ObservationBank(root/'observations')
    cpu_ledger = root/'inputs/ledger_cpu.npz'
    projections = root/'inputs/ledger_cpu_projections.npz'
    if action == 'preflight':
        result = preflight(bank, settings, cpu_ledger, projections)
        write(args.out, result)
        print(json.dumps({k:v for k,v in result.items() if k != 'observations'}, ensure_ascii=False), flush=True)
        return
    if not args.confirm_gpu or not torch.cuda.is_available():
        parser.error('此入口需要用户授权 GPU 后显式传入 --confirm-gpu')
    torch.set_num_threads(1)
    seed(settings['seed'])
    if action == 'smoke':
        run_smoke(bank, settings, cpu_ledger, projections, args.out, args.smoke_minutes)
        return
    experiment = Experiment(bank, settings)
    if args.ledger:
        experiment.attach_ledger(args.ledger, cpu_ledger, projections)
    if action == 'attribute':
        experiment.attribute(cpu_ledger, projections, args.out, bank.raw_ids)
    elif action == 'train-b':
        experiment.fit('B', args.out, settings['steps_b'], start=args.checkpoint, resume=args.resume)
    elif action == 'train-c':
        if args.ledger is None or args.checkpoint is None:
            parser.error('阶段 C 需要 49 帧归因账本和阶段检查点')
        from local_replacement import load_ledger
        if tuple(load_ledger(args.ledger)['attributed_raw_ids']) != tuple(bank.raw_ids):
            parser.error('正式阶段 C 需要全部 49 帧、两种编辑状态的归因')
        experiment.fit('C', args.out, settings['steps_c'], start=args.checkpoint, resume=args.resume)
    elif action == 'attach-pvd':
        if args.ledger is None or args.checkpoint is None:
            parser.error('阶段 D 需要接管检查点和对应账本')
        experiment.final_pvd(args.checkpoint, args.out)
    elif action == 'evaluate':
        stage = None
        if args.checkpoint:
            state = read_checkpoint(args.checkpoint)
            experiment.load(state)
            stage = state['stage']
        if 'pvd' in args.groups and stage != 'D':
            parser.error('PVD 组需要阶段 D 检查点')
        if 'replacement' in args.groups and (stage not in ('C', 'D') or not bool(experiment.replacement.dynamic_retired)):
            parser.error('替换组需要已完成持久接管的检查点')
        experiment.evaluate(args.out, PILOTS if args.anchors else None, args.groups)


def run_smoke(bank, settings, cpu_ledger, projections, out, minutes):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    experiment = Experiment(bank, settings)
    experiment.validate_native_replay(out/'baseline_replay.json')
    b1 = experiment.fit('B', out/'b_initial', 2, smoke=True)
    peaks = [torch.cuda.max_memory_allocated()]
    experiment.tensor_frame.cache_clear()
    del experiment
    torch.cuda.empty_cache()
    experiment = Experiment(bank, settings)
    b2 = experiment.fit('B', out/'b_restored', 4, start=b1, resume=True, smoke=True)
    experiment.evaluate(out/'eval_b', PILOTS, ('strong_baseline', 'objects'))
    ledger = out/'ledger_pilots.npz'
    attr_start = time.monotonic()
    experiment.attribute(cpu_ledger, projections, ledger, PILOTS)
    rough_step_time = max((time.monotonic() - attr_start) / (2*len(PILOTS)), .1)
    peaks.append(torch.cuda.max_memory_allocated())
    experiment.attach_ledger(ledger, cpu_ledger, projections)
    # 测得的单次光栅代价仅用于 smoke 的资源预算，正式步数不受影响。
    remaining = max(minutes * 60 - (time.monotonic() - started) - 35, 0)
    smoke_steps = min(experiment.settings['steps_c'], max(4, int(remaining / rough_step_time)))
    c1 = experiment.fit('C', out/'c_initial', 2, start=b2, smoke=True, schedule_steps=smoke_steps)
    peaks.append(torch.cuda.max_memory_allocated())
    experiment.tensor_frame.cache_clear()
    del experiment
    torch.cuda.empty_cache()
    experiment = Experiment(bank, settings)
    experiment.attach_ledger(ledger, cpu_ledger, projections)
    c2 = experiment.fit('C', out/'c_restored', smoke_steps, start=c1, resume=True, smoke=True, schedule_steps=smoke_steps)
    d = experiment.final_pvd(c2, out/'d_pvd')
    experiment.load(read_checkpoint(d))
    experiment.evaluate(out/'eval_d', PILOTS, ('strong_baseline', 'objects', 'replacement', 'pvd'))
    write(out/'receipt.json', dict(status='PASS', scope='engineering_smoke', raw_ids=list(PILOTS),
          updates_b=4, updates_c=smoke_steps, restored_b=True, restored_c=True, d_entry=True,
          elapsed_s=time.monotonic()-started, peak_cuda_bytes=max(peaks + [torch.cuda.max_memory_allocated()]),
          settings=experiment.settings, formal_updates=0, quality_accepted=False))


if __name__ == '__main__':
    main()

"""C2：冻结三维点条件先验、持久a6表面和原始观测下的局部优化。

不写入B10k或道路r3；新a6无自由时间参数，按真实逐帧刚体位姿运动。
先验几何是proposal，不是无车/完整车身真值。颜色为有效外观，不宣称BRDF分解。
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parent.parent
R3 = RUN.parent / 'scene008_a37_only_B10k_road_20261103_r3'
sys.path.insert(0, str(R3 / 'source'))
import repair37
from editor_common import cpu, save, prepare_rasterization, road_kwargs, merge_rows, raster
from utils.loss_utils import ssim, psnr


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def tensor(value, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype, device='cuda')


class ActorSurface(torch.nn.Module):
    """固定表面支撑的薄高斯数值离散；无独立可学习opacity和寿命。"""
    def __init__(self, path):
        super().__init__()
        a = dict(np.load(path, allow_pickle=False))
        self.register_buffer('canonical', tensor(a['canonical_xyz']))
        normals = tensor(a['canonical_normals'])
        self.register_buffer('normals', normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-6))
        self.register_buffer('sigma', tensor(a['sigma_tangent_m']).reshape(-1))
        self.register_buffer('observed', tensor(a['observed_anchor'], torch.bool))
        self.register_buffer('colour_known', tensor(a['colour_observed'], torch.bool))
        colour = tensor(a['colour_srgb']).clamp(.005, .995)
        self.register_buffer('initial_colour', colour)
        self.colour_logit = torch.nn.Parameter(torch.logit(colour))
        self.normal_offset = torch.nn.Parameter(torch.zeros(len(colour), device='cuda'))
        self.register_buffer('point_nearest', tensor(a['observed_nearest_anchor'], torch.long))
        self.register_buffer('observed_points', tensor(a['observed_points']))
        self.opacity = tensor(np.full((len(colour), 1), .995, np.float32))
        self.active = torch.ones(len(colour), dtype=torch.bool, device='cuda')

    def xyz_canonical(self):
        delta = .03 * self.normal_offset.tanh() * (~self.observed)
        return self.canonical + self.normals * delta[:, None]

    def colors(self):
        return self.colour_logit.sigmoid()

    def kwargs(self, camera, matrix, pipe, channels, precomputed):
        points = self.xyz_canonical()
        scale = matrix[:3, :3].norm(dim=0).mean()
        xyz = points @ matrix[:3, :3].T + matrix[:3, 3]
        normals = self.normals @ matrix[:3, :3].T / scale
        normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-6)
        q = torch.stack((1 + normals[:, 2], -normals[:, 1], normals[:, 0], torch.zeros_like(normals[:, 0])), 1)
        opposite = q.norm(dim=1) < 1e-5
        q = torch.where(opposite[:, None], tensor([0., 1., 0., 0.])[None], q)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-6)
        scales = torch.stack((self.sigma, self.sigma, torch.full_like(self.sigma, .003)), 1) * scale
        homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1)
        depth = homogeneous @ camera.world_view_transform[:, 2:3]
        cov = None
        if pipe.compute_cov3D_python:
            from utils.general_utils import build_scaling_rotation, strip_symmetric
            basis = build_scaling_rotation(scales, q)
            cov = strip_symmetric(basis @ basis.transpose(1, 2))
        dc = ((self.colors() - .5) / .28209479177387814)[:, None]
        shs = None if precomputed else torch.cat((dc, dc.new_zeros((len(dc), channels - 1, 3))), 1)
        return dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=shs,
                    colors_precomp=self.colors() if precomputed else None,
                    features=torch.cat((depth, torch.ones_like(depth)), 1),
                    opacities=self.opacity, scales=None if cov is not None else scales,
                    rotations=None if cov is not None else q, cov3D_precomp=cov, mask=self.active)

    def prior_loss(self):
        # 同一持久latent/表面在所有时刻共享；隐藏部分不设逐帧RGB残差。
        geometry = (.03 * self.normal_offset.tanh()).square().mean()
        known = self.colour_known
        colour = (self.colors()[known] - self.initial_colour[known]).square().mean() if known.any() else geometry * 0
        return 10 * geometry + .002 * colour


class Experiment:
    def __init__(self, settings, asset):
        self.settings = settings
        self.base = repair37.Experiment()
        checkpoint = torch.load(str(R3 / 'formal/checkpoint001000.pt'), map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 1000 and checkpoint['parent_identity'] == self.base.n.before
        self.base.repair.load_state_dict(checkpoint['repair'], strict=True)
        self.base.repair.requires_grad_(False)
        self.n = self.base.n
        self.actor = ActorSurface(asset)
        self.actor_gates = bool(settings.get('actor_gates', False))
        if self.actor_gates:
            selection = dict(np.load(settings['actor_contributors'], allow_pickle=False))
            self.actor.register_buffer('contributor_ids', tensor(selection['ids'], torch.long))
            self.actor.gate_raw = torch.nn.Parameter(torch.full((len(selection['ids']),), 2., device='cuda'))
        self.geometry = dict(np.load(RUN / 'inputs/a6_geometry_minimal.npz', allow_pickle=False))
        self.views, self.static_cache, self.truth_cache = {}, {}, {}
        self.visibility = None
        if settings.get('road_visibility', False):
            from road_visibility import RoadVisibility
            self.visibility = RoadVisibility(self.base, settings=settings.get('road_visibility_settings'))
        self.parent_versions = {name: getattr(self.n.static, name)._version for name in
                                ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', '_t', '_scaling_t', '_velocity')}

    def parameter_groups(self):
        groups = [{'params': [self.actor.colour_logit], 'lr': self.settings['colour_lr']},
                  {'params': [self.actor.normal_offset], 'lr': self.settings['geometry_lr']}]
        if self.actor_gates:
            groups.append({'params': [self.actor.gate_raw], 'lr': self.settings['gate_lr']})
        return groups

    def view(self, raw):
        if raw not in self.views:
            self.views[raw] = dict(np.load(RUN / 'inputs/a6_views' / ('%03d.npz' % raw), allow_pickle=False))
        return self.views[raw]

    def truth(self, raw):
        if raw not in self.truth_cache:
            self.truth_cache[raw] = tensor(self.view(raw)['image_srgb'].transpose(2, 0, 1).copy())
        return self.truth_cache[raw]

    def matrix(self, raw):
        return tensor(self.geometry['model_from_canonical'][raw])

    def native_mask(self, raw, kw):
        if raw not in self.static_cache:
            # 仅退出物理a6盒内的旧贡献；不把宽核候选集合当对象真值。
            # 记录跨边界残留并验收域外损伤，中心裁剪不是完整语义分割。
            with torch.no_grad():
                inv = torch.linalg.inv(self.matrix(raw))
                local = kw['means3D'] @ inv[:3, :3].T + inv[:3, 3]
                size = tensor(self.geometry['box_lwh_m'])
                margin = float(self.settings['replacement_margin_m'])
                inside = (local.abs() <= size[None] / 2 + margin).all(1)
                self.static_cache[raw] = inside.cpu()
        return self.static_cache[raw].to('cuda')

    def render(self, raw, include_target=True, enabled=True, camera=None):
        e, n = self.base, self.n
        c = camera or e.camera(raw)
        kw = prepare_rasterization(c, n.static, n.args)
        kw['means2D'] = torch.zeros_like(kw['means3D'])
        ids = e.repair.ids
        kw['opacities'] = kw['opacities'].index_copy(0, ids, kw['opacities'][ids] * e.repair.gate_raw.sigmoid()[:, None])
        if enabled:
            kw['mask'] = kw['mask'] & ~self.native_mask(raw, kw)
            if self.actor_gates:
                actor_ids = self.actor.contributor_ids
                kw['opacities'] = kw['opacities'].index_copy(0, actor_ids,
                    kw['opacities'][actor_ids] * self.actor.gate_raw.sigmoid()[:, None])
        if include_target:
            dy = prepare_rasterization(c, n.dynamic, n.args, dynamic=True)
            dy['means2D'] = torch.zeros_like(dy['means3D'])
            kw = merge_rows(kw, dy)
        channels = kw['shs'].shape[1] if kw['shs'] is not None else 1
        precomputed = kw['colors_precomp'] is not None
        road = road_kwargs(c, e.repair.road, n.args, channels, precomputed_color=precomputed)
        if self.visibility is not None:
            road = self.visibility.apply(raw, include_target, road, camera=c)
        # 两个贡献通道只用于诊断，不乘图像mask，不参与颜色条件路由。
        kw['features'] = torch.cat((kw['features'], torch.zeros_like(kw['features'][:, :1]).repeat(1, 2)), 1)
        road['features'] = torch.cat((road['features'], torch.ones_like(road['features'][:, :1]), torch.zeros_like(road['features'][:, :1])), 1)
        kw = merge_rows(kw, road)
        if enabled:
            actor = self.actor.kwargs(c, self.matrix(raw), n.args, channels, precomputed)
            actor['features'] = torch.cat((actor['features'], torch.zeros_like(actor['features'][:, :1]), torch.ones_like(actor['features'][:, :1])), 1)
            kw = merge_rows(kw, actor)
        out = raster(c, n.args, n.static.active_sh_degree, kw, n.bg)
        environment = n.env(c.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        out['rgb'] = out['rgb'] + (1 - out['alpha']) * environment
        out['road_alpha'] = out['extra'][0:1]
        out['actor_alpha'] = out['extra'][1:2]
        return out

    def loss(self, raw):
        d = self.view(raw)
        out = self.render(raw, True)
        truth = self.truth(raw)
        evidence = tensor(d['visible_evidence_mask'], torch.bool)
        boundary = tensor(d['uncertain_boundary_mask'], torch.bool)
        labels = tensor(d['actor_labels'], torch.long)
        target = (labels == 4).float()
        positive = (labels == 4) & ~boundary
        negative = (labels != 4) & ~boundary
        photo = (out['rgb'] - truth).abs()[:, evidence].mean() if evidence.any() else out['rgb'].sum() * 0
        global_photo = (out['rgb'] - truth).abs().mean()
        # 在含a37的完整场景中监督可见alpha*T；删车后隐藏面无伪mask真值。
        alpha = out['actor_alpha'][0]
        inside = (alpha[positive] - 1).square().mean() if positive.any() else alpha.sum() * 0
        outside = alpha[negative].square().mean() if negative.any() else alpha.sum() * 0
        regularizer = self.actor.prior_loss()
        cfg = self.settings['loss_weights']
        preserve = global_photo * 0
        gate_prior = global_photo * 0
        if self.actor_gates:
            with torch.no_grad():
                base_full = self.render(raw, True, enabled=False)['rgb']
            # 保留真实其余车辆、a37和a6之外的已重建区域；候选并非归属GT。
            # 与a6光度使用相同像素分母，避免整图平均稀释域外损伤。
            preserve_mask = negative
            preserve = (out['rgb'] - base_full).abs()[:, preserve_mask].sum() / (3 * evidence.sum().clamp_min(1))
            gate_prior = .0005 * (1 - self.actor.gate_raw.sigmoid()).mean()
        total = (cfg['evidence_rgb'] * photo + cfg['full_rgb'] * global_photo + cfg['alpha_inside'] * inside
                 + cfg['alpha_outside'] * outside + regularizer + cfg.get('native_keep', 0.) * preserve + gate_prior)
        return total, dict(photo=float(photo.detach()), full=float(global_photo.detach()),
                           alpha_inside=float(inside.detach()), alpha_outside=float(outside.detach()),
                           prior=float(regularizer.detach()), preserve=float(preserve.detach()), gate_prior=float(gate_prior.detach()))

    def unchanged(self):
        return self.base.unchanged() and all(getattr(self.n.static, k)._version == v for k, v in self.parent_versions.items())

    @torch.no_grad()
    def evaluate(self, dest, raws, tag):
        folder = Path(dest) / tag
        folder.mkdir(parents=True, exist_ok=False)
        rows = []
        from baseline_audit import render_baseline
        for raw in raws:
            full = self.render(raw, True)
            edit = self.render(raw, False)
            native = render_baseline(self.base, raw, True, False)
            baseline = render_baseline(self.base, raw, False, True)
            truth = self.truth(raw)
            d, v = self.base.view(raw), self.view(raw)
            actor = tensor(v['visible_evidence_mask'], torch.bool)
            rec = dict(raw=raw, full={}, road={}, actor_samples=int(actor.sum()),
                       replaced_native_rows=int(self.static_cache[raw].sum()))
            for name, image in [('native', native['rgb']), ('C2', full['rgb'])]:
                pred = image.clamp(0, 1)
                rec['full'][name] = dict(psnr=float(psnr(pred, truth)), ssim=float(ssim(pred[None], truth[None])),
                    a6_observed_MAE=float((pred-truth).abs()[:, actor].mean()) if actor.any() else None)
            for name, image in [('baseline', baseline['rgb']), ('C2', edit['rgb'])]:
                pred = repair37.r4.linear(image)
                metrics = {}
                for key, mask, refkey in [('fit', d['repair_mask'] & ~d['inferred_fine'], 'recovery_reference_linear'),
                                          ('new_dark', d['new_shadow_proxy'] & d['repair_mask'] & ~d['inferred_fine'], 'recovery_reference_linear'),
                                          ('original_shadow', d['shadow_proxy'] & d['domain_a37'], 'reference_linear')]:
                    m = tensor(mask, torch.bool)
                    reference = tensor(d[refkey].transpose(2, 0, 1))
                    metrics[key] = dict(samples=int(m.sum()), MAE=float((pred-reference).abs()[:, m].mean()) if m.any() else None)
                rec['road'][name] = metrics
            labels = tensor(v['actor_labels'], torch.long)
            for key, mask in [('other_retained', (labels > 0) & (labels < 4)), ('outside_actor', labels != 4)]:
                rec[key + '_change'] = float((edit['rgb'] - baseline['rgb']).abs()[:, mask].mean()) if mask.any() else None
            rows.append(rec)
            save(folder / ('%03d_C2_full.png' % raw), full['rgb'])
            save(folder / ('%03d_C2_delete37.png' % raw), edit['rgb'])
            save(folder / ('%03d_actor_alpha_full.png' % raw), full['actor_alpha'].expand(3, -1, -1))
            save(folder / ('%03d_actor_alpha_delete.png' % raw), edit['actor_alpha'].expand(3, -1, -1))
        write(folder / 'metrics.json', dict(rows=rows, parent_unchanged=self.unchanged(), input_role='BUILD',
              prior_is_frozen=True, opacity_is_fixed=True, no_per_frame_material=True,
              only_target_a37_deleted=True, a6_retained=True, quality_accepted=False, human_verdict=None))
        return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--settings', type=Path, default=RUN / 'settings.json')
    p.add_argument('--asset', type=Path, default=RUN / 'inputs/actor_surface_seed0.npz')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--resume', type=Path)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--diagnostic', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    sampler = random.Random(0)
    settings = json.loads(args.settings.read_text())
    if settings.get('road_visibility') and 'road_visibility_settings' not in settings:
        from road_visibility import SETTINGS as visibility_path
        settings['road_visibility_settings'] = json.loads(Path(visibility_path).read_text())
    receipt_path = args.asset.with_name(args.asset.stem + '_receipt.json')
    asset_receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if asset_receipt.get('formal_eligible') is False and not args.diagnostic:
        raise RuntimeError('该先验候选未通过几何验收；仅允许显式diagnostic接线，不得进入正式实验。')
    dest = RUN / args.output
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'runtime_source.py').write_text(Path(__file__).read_text())
    write(dest / 'settings_consumed.json', settings)
    e = Experiment(settings, args.asset)
    opt = torch.optim.Adam(e.parameter_groups(), eps=1e-15)
    contract = dict(task_id='scene008_a37_C2_20261103_r1', baseline='single-a37 B10k + r3 road1k',
        frozen_base=str(repair37.Native.__module__), target=[37], retained_actor=6, formal_steps=settings['formal_steps'],
        diagnostic_only=args.diagnostic, proposal_receipt=str(receipt_path),
        actor_asset=str(args.asset), settings=settings, prior=settings.get('prior_label', 'Hunyuan3D-Omni fixed point-conditioned geometry proposal'),
        trainable='canonical colour + bounded missing-surface normal offsets',
        fixed='B10k, road-r3 state, camera, pose, environment, learned prior, observed geometry, surface opacity',
        alpha_supervision='actual a6 visible alpha*T in noedit full scene; reviewed visible mask positive and negative domains separately normalized, excluding 1px uncertain boundary',
        noedit_branch_enabled=True, no_per_frame_material=True, independent_shadow_layer=False,
        all_199_BUILD=True, reference_is_proxy=True, quality_accepted=False, human_verdict=None,
        optimizer_raws=[r for r in range(89, 138) if r not in (101, 113, 125)],
        excluded_optimizer_anchors_are_not_independent_test=True)
    startstep = 0
    if e.actor_gates:
        contract['trainable'] += ' + local static contribution gates (candidate, not ownership GT)'
        contract['replacement_selection'] = settings['actor_contributors']

    def state(step):
        return dict(step=step, actor=e.actor.state_dict(), optimizer=opt.state_dict(), sampler=sampler.getstate(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(), parent_identity=e.n.before, contract=contract)

    if args.resume:
        ck = torch.load(str(args.resume), map_location='cpu', weights_only=False)
        assert ck['parent_identity'] == e.n.before
        e.actor.load_state_dict(ck['actor'], strict=True)
        opt.load_state_dict(ck['optimizer'])
        sampler.setstate(ck['sampler'])
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        startstep = ck['step']
        restored = state(startstep)
        from compare_state import differences
        errors = differences(ck, restored)
        write(dest / 'restore.json', dict(step=startstep, exact_full_state=not errors, differences=errors, updates=0))
        assert not errors, errors
    write(dest / 'contract.json', contract)
    if args.evaluate:
        e.evaluate(dest, range(89, 138), 'all49')
        return
    e.evaluate(dest, [101, 105, 113], 'initial')
    started = time.monotonic()
    with (dest / 'training.jsonl').open('x') as log:
        for step in range(startstep + 1, args.steps + 1):
            raw = sampler.choice(contract['optimizer_raws'])
            opt.zero_grad(set_to_none=True)
            loss, rec = e.loss(raw)
            assert torch.isfinite(loss), rec
            loss.backward()
            for name, param in e.actor.named_parameters():
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
            opt.step()
            rec.update(step=step, raw=raw, loss=float(loss.detach()), seconds=time.monotonic() - started,
                       peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(rec) + '\n')
            log.flush()
            if step == startstep + 1 or step % 100 == 0:
                print(json.dumps(rec), flush=True)
            if step in (32, 100, 500, 1000) or step == args.steps:
                torch.save(state(step), str(dest / ('checkpoint%06d.pt' % step)))
                e.evaluate(dest, [101, 105, 113], 'step%06d' % step)
    write(dest / 'receipt.json', dict(updates=args.steps - startstep, step=args.steps,
          seconds=time.monotonic() - started, parent_unchanged=e.unchanged(), quality_accepted=False, human_verdict=None))


if __name__ == '__main__':
    main()

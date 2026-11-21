"""真实高斯路块的局部接缝拟合；默认仅CPU预检，GPU须显式启用。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
DEFAULT_ROOT = CAMPAIGN / 'road_gaussian_transfer_20261122_r1'
SCHEMA = 'retrieved_gaussian_local_seam_fit.v1'
RUN_ROOT_ENV = 'ROAD_TRANSFER_RUN_ROOT'
RUN_PATH_FIELDS = ('transfer_asset', 'gate_receipt', 'parent_rendered', 'output', 'smoke', 'recovery')
PATH_FIELDS = RUN_PATH_FIELDS + ('parent_asset', 'common_teacher', 'domain_root', 'same_source_root')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in (saved.files if keys is None else keys)}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def defaults(root=None):
    root = Path(root if root is not None else os.environ.get(RUN_ROOT_ENV, DEFAULT_ROOT)).resolve()
    return dict(schema=SCHEMA, root=str(root),
        parent_asset=str(CAMPAIGN / 'road_surface_residual_20261121_r2/appearance_fit/asset'),
        transfer_asset=str(root / 'transfer/asset'), gate_receipt=str(root / 'transfer/receipt.json'),
        parent_rendered=str(root / 'parent/rendered'),
        common_teacher=str(CAMPAIGN / 'road_common1152_20261121/teacher'),
        domain_root=str(CAMPAIGN / 'road_surface_residual_20261121_r2/proposal/domains'),
        same_source_root=str(CAMPAIGN / 'road_multisource_20261122_r1/sources/targets'),
        output=str(root / 'appearance'), smoke=str(root / 'smoke/fit_seams'), recovery=str(root / 'recovery'),
        train_frames=[86, 110], evaluation_frames=[86, 110, 134, 179, 198], seed=101,
        formal_steps=400, smoke_steps=8, checkpoint_interval=50,
        gain_bounds=[.85, 1.15], bias_bounds=[-.02, .02], opacity_bounds=[1e-4, .999],
        tangent_sigma_multiplier_bounds=[.9, 1.15],
        learning_rates=dict(gain_bias=.005, opacity_logit=.02, log_sigma=.002),
        Adam=dict(betas=[.9, .999], eps=1e-8),
        stable_alpha=.95, photo_new_contribution=1 / 255, coverage_new_contribution=1e-5,
        coverage_target_alpha=.99, alpha_denominator_min=1e-4, normalizer_floor=(1 / 255)**2,
        prior_weights=dict(gain_bias=.02, opacity=.001, log_sigma=.001),
        public_reload_tolerance=1e-6, resume_tolerance=1e-6,
        target_policy='observed PNG plus trusted same-position RGB only; no diffusion RGB',
        mask_policy='TRANSFER-initial masks frozen; parent and TRANSFER alpha>=stable_alpha for photo',
        photograph_error='MSE of current native road_RGB/current_alpha.clamp_min(1e-4) versus observed RGB',
        appearance_parameters='one global scalar gain plus one RGB bias for new rows only',
        coverage_parameters='only active new seam-row opacity and common tangent-sigma multiplier',
        new_diffusion_calls=0)


def configuration(path, root):
    provided = read(path) if path is not None else {}
    selected_root = Path(root if root is not None else provided.get('root', os.environ.get(RUN_ROOT_ENV, DEFAULT_ROOT))).resolve()
    value = defaults(selected_root)
    if provided:
        unknown = set(provided) - set(value)
        need(not unknown, 'Unknown configuration fields: ' + str(sorted(unknown)))
        old_root = Path(provided.get('root', selected_root)).resolve()
        provided = copy.deepcopy(provided)
        # 显式--root可复用旧配置；仅重新定位run所属路径，外部只读输入保留。
        if selected_root != old_root:
            for key in RUN_PATH_FIELDS:
                if key in provided:
                    previous = Path(provided[key])
                    if previous.is_absolute() and previous.is_relative_to(old_root):
                        provided[key] = str(selected_root / previous.relative_to(old_root))
        for key, item in provided.items():
            if key == 'root':
                continue
            if isinstance(value.get(key), dict) and isinstance(item, dict):
                need(not (set(item) - set(value[key])), 'Unknown nested configuration field: ' + key)
                value[key] = dict(value[key], **item)
            else:
                value[key] = item
    value['root'] = str(selected_root)
    for key in PATH_FIELDS:
        candidate = Path(value[key])
        value[key] = str(candidate if candidate.is_absolute() else selected_root / candidate)
    for key in ('output', 'smoke', 'recovery'):
        need(Path(value[key]).resolve().is_relative_to(selected_root), 'Mutable output must stay within selected run: ' + key)
    need(value['schema'] == SCHEMA and value['new_diffusion_calls'] == 0, 'Configuration scope differs')
    need(value['formal_steps'] == 400 and value['smoke_steps'] == 8 and value['checkpoint_interval'] == 50,
         'This registered fit requires formal400/smoke8/checkpoint50')
    need(value['train_frames'] and set(value['train_frames']) <= set(value['evaluation_frames']), 'Invalid frame lists')
    need(len(set(value['train_frames'])) == len(value['train_frames']), 'Duplicate training frame')
    for name in ('gain_bounds', 'bias_bounds', 'opacity_bounds', 'tangent_sigma_multiplier_bounds'):
        bounds = value[name]
        need(len(bounds) == 2 and all(np.isfinite(bounds)) and bounds[0] < bounds[1], 'Invalid ' + name)
    need(0 < value['opacity_bounds'][0] < value['opacity_bounds'][1] < 1, 'Invalid opacity bounds')
    need(value['tangent_sigma_multiplier_bounds'][0] <= 1 <= value['tangent_sigma_multiplier_bounds'][1], 'Sigma bounds omit identity')
    need(value['gain_bounds'][0] <= 1 <= value['gain_bounds'][1] and value['bias_bounds'][0] <= 0 <= value['bias_bounds'][1], 'RGB bounds omit identity')
    need(value['opacity_bounds'][0] >= 1e-4 and value['opacity_bounds'][1] <= .999 and
         value['tangent_sigma_multiplier_bounds'][0] >= .9 and value['tangent_sigma_multiplier_bounds'][1] <= 1.15,
         'Fit bounds exceed the registered K qualification envelope')
    return value


def input_files(cfg):
    transfer = Path(cfg['transfer_asset'])
    paths = [Path(__file__), Path(cfg['parent_asset']) / 'road.npz',
             transfer / 'road.npz', transfer / 'provenance.npz',
             transfer / 'persistent/scene.json', transfer / 'persistent/proof.json',
             Path(cfg['gate_receipt']), Path(cfg['root']) / 'qualification/seam_rows.npz']
    for raw in cfg['train_frames']:
        paths += [Path(cfg['common_teacher']) / f'raw{raw:03d}/observation_1152.png',
                  Path(cfg['domain_root']) / f'{raw:03d}.npz',
                  Path(cfg['same_source_root']) / f'{raw:03d}.npz']
    for raw in cfg['evaluation_frames']:
        paths.append(Path(cfg['parent_rendered']) / f'{raw:03d}.npz')
    return paths


def preflight(cfg, require_ready=False):
    missing = [str(path) for path in input_files(cfg) if not path.is_file()]
    gate = read(cfg['gate_receipt']) if Path(cfg['gate_receipt']).is_file() else {}
    ready = not missing and gate.get('complete') is True
    result = dict(schema=SCHEMA, ready=ready, missing=missing, gate_complete=gate.get('complete') is True,
                  GPU_used=False, new_diffusion_calls=0, configuration=cfg,
                  target_fields_loaded=['observation_1152.png', 'same_mask', 'same_rgb', 'same_confidence'],
                  diffusion_RGB_loaded=False)
    if require_ready:
        need(ready, 'Final TRANSFER/K gate is not ready: ' + json.dumps(result, ensure_ascii=False))
        need(all(row.get('RGB_max') == 0 and row.get('alpha_max') == 0 for row in gate.get('checks', [])),
             'TRANSFER receipt contains a failed K check')
    return result


def runtime(cfg):
    import native
    import asset_io
    native.ROOT = Path(cfg['root']); native.PARENT = Path(cfg['parent_asset'])
    native.FRAMES = tuple(cfg['evaluation_frames'])
    native.m.ROOT = native.ROOT; native.m.PARENT = native.PARENT
    asset_io.ROOT = native.ROOT; asset_io.PARENT = native.PARENT
    return native, asset_io


def array_signature(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), count=int(value.sum()), sha256=hashlib.sha256(value.tobytes()).hexdigest())


class FitState:
    def __init__(self, ctx, cfg, native):
        self.ctx, self.cfg, self.native = ctx, cfg, native
        self.torch = native.torch
        torch = self.torch
        device = ctx.base_rgb.device
        self.device = device
        self.cursor = 0
        self.base_rgb = ctx.base_rgb.detach().clone()
        packet = ctx.base_packet(cfg['train_frames'][0])
        self.base_opacity = packet['opacities'].detach().reshape(-1).clone()
        self.base_scales = packet['scales'].detach().clone()
        self.parent_rows = len(load(Path(cfg['parent_asset']) / 'road.npz', ['active'])['active'])
        new = np.asarray(ctx.provenance['transfer_new_row'], bool)
        need(np.array_equal(new, np.arange(len(new)) >= self.parent_rows), 'New-row identity must follow parent prefix')
        active = np.asarray(ctx.active_world_rows, np.int64)
        self.new_world = active[active >= self.parent_rows]
        self.new_pos = torch.as_tensor(np.flatnonzero(active >= self.parent_rows), device=device, dtype=torch.long)
        from seam_domain import seam_rows
        seam = np.asarray(seam_rows(cfg['root'])['new_seam_mask'])
        need(seam.dtype.kind == 'b' and seam.shape == (len(new) - self.parent_rows,), 'Invalid full new_seam_mask')
        seam_active = (active >= self.parent_rows)
        seam_active[seam_active] &= seam[active[seam_active] - self.parent_rows]
        self.seam_world = active[seam_active]
        self.seam_pos = torch.as_tensor(np.flatnonzero(seam_active), device=device, dtype=torch.long)
        need(len(self.new_pos) > 0, 'No active transferred Gaussian rows')
        self.gain = torch.nn.Parameter(torch.ones((), device=device))
        self.bias = torch.nn.Parameter(torch.zeros(3, device=device))
        # 直接沿用原提供器logit，避免初值概率再取logit引入F32往返差。
        raw_logits = ctx.provider.state_objects[0].opacity.detach().reshape(-1)
        need(torch.equal(raw_logits.sigmoid(), self.base_opacity), 'Provider logits differ from native opacity template')
        self.opacity_logits = torch.nn.Parameter(raw_logits.index_select(0, self.seam_pos).clone())
        self.log_sigma = torch.nn.Parameter(torch.zeros(len(self.seam_pos), device=device))
        self.initial_logits = self.opacity_logits.detach().clone()
        lo, hi = cfg['opacity_bounds']
        initial_probability = self.opacity_logits.detach().sigmoid()
        need(bool(((initial_probability >= lo) & (initial_probability <= hi)).all()), 'TRANSFER seam opacity is outside registered fit bounds')
        self.optimizer = torch.optim.Adam([
            dict(params=[self.gain, self.bias], lr=cfg['learning_rates']['gain_bias']),
            dict(params=[self.opacity_logits], lr=cfg['learning_rates']['opacity_logit']),
            dict(params=[self.log_sigma], lr=cfg['learning_rates']['log_sigma'])],
            betas=tuple(cfg['Adam']['betas']), eps=cfg['Adam']['eps'])
        self.frames = {}
        self.mask_signature = {}
        with torch.no_grad():
            for raw in cfg['train_frames']:
                with Image.open(Path(cfg['common_teacher']) / f'raw{raw:03d}/observation_1152.png') as im:
                    observation = np.asarray(im.convert('RGB')).astype(np.float32) / 255
                domain = load(Path(cfg['domain_root']) / f'{raw:03d}.npz',
                              ['appearance_A', 'appearance_C', 'appearance_S', 'canonical_valid'])
                source = load(Path(cfg['same_source_root']) / f'{raw:03d}.npz', ['same_mask', 'same_rgb', 'same_confidence'])
                CS = domain['appearance_C'] | domain['appearance_S']
                same = CS & source['same_mask'] & (source['same_confidence'] > 0)
                target = observation.copy(); target[same] = source['same_rgb'][same]
                need(np.isfinite(target).all() and target.min() >= 0 and target.max() <= 1, 'Invalid observation target')
                parent_alpha = load(Path(cfg['parent_rendered']) / f'{raw:03d}.npz', ['road_alpha'])['road_alpha']
                initial = native.render(ctx, raw, feature_world=self.new_world)
                alpha = initial['alpha']; contribution = initial['new_weight']
                alpha_np = alpha.cpu().numpy(); weight_np = contribution.cpu().numpy()
                need(observation.shape == (*alpha_np.shape, 3), 'Observation/native raster grid differs')
                stable = (parent_alpha >= cfg['stable_alpha']) & (alpha_np >= cfg['stable_alpha'])
                impact = weight_np > cfg['photo_new_contribution']
                photo_A = domain['appearance_A'] & stable & impact
                photo_same = same & ~domain['appearance_A'] & stable & impact
                photo = photo_A | photo_same
                cover = CS & domain['canonical_valid'] & (weight_np > cfg['coverage_new_contribution'])
                cv = lambda a: torch.as_tensor(np.array(a, copy=True), device=device)
                self.frames[raw] = dict(target=cv(target.transpose(2, 0, 1)), photo=cv(photo), cover=cv(cover),
                    photo_A=photo_A, photo_same=photo_same, same_requested=same,
                    initial_alpha=alpha.detach().clone(), initial_new_contribution=contribution.detach().clone())
                self.mask_signature[str(raw)] = {name: array_signature(value) for name, value in
                    [('photo_A', photo_A), ('photo_same', photo_same), ('photo', photo), ('cover', cover)]}
        self.normalizers = {}
        with torch.no_grad():
            photo_values, cover_values = [], []
            for raw, f in self.frames.items():
                initial = native.render(ctx, raw)
                photo, cover = self.data_terms(initial, f)
                photo_values.append(float(photo)); cover_values.append(float(cover))
            initial_photo = float(np.mean(photo_values)); initial_cover = float(np.mean(cover_values))
            self.normalizers = dict(photo=max(initial_photo, cfg['normalizer_floor']),
                                    cover=max(initial_cover, cfg['normalizer_floor']),
                                    initial_photo=initial_photo, initial_cover=initial_cover)
        self.binding = dict(schema=SCHEMA, config_sha256=canonical_sha(cfg),
            files={str(path): sha(path) for path in input_files(cfg)},
            parent_rows=self.parent_rows, new_world_ids_sha256=hashlib.sha256(self.new_world.tobytes()).hexdigest(),
            seam_world_ids_sha256=hashlib.sha256(self.seam_world.tobytes()).hexdigest(), masks=self.mask_signature)
        self.assert_frozen()

    def colors(self):
        value = (self.base_rgb.index_select(0, self.new_pos) * self.gain + self.bias[None]).clamp(0, 1)
        return self.base_rgb.index_copy(0, self.new_pos, value)

    def opacity(self):
        return self.base_opacity.index_copy(0, self.seam_pos, self.opacity_logits.sigmoid())

    def scales(self):
        selected = self.base_scales.index_select(0, self.seam_pos)
        value = self.torch.cat((selected[:, :2] * self.log_sigma.exp()[:, None], selected[:, 2:]), dim=1)
        return self.base_scales.index_copy(0, self.seam_pos, value)

    def data_terms(self, output, frame):
        zero = output['rgb'].sum() * 0
        surface = output['rgb'] / output['alpha'].clamp_min(self.cfg['alpha_denominator_min'])[None]
        photo = (surface[:, frame['photo']] - frame['target'][:, frame['photo']]).square().mean() if bool(frame['photo'].any()) else zero
        cover = self.torch.relu(self.cfg['coverage_target_alpha'] - output['alpha'][frame['cover']]).square().mean() if bool(frame['cover'].any()) else zero
        return photo, cover

    def objective(self):
        torch = self.torch
        colors, opacity, scales = self.colors(), self.opacity(), self.scales()
        photo, cover = [], []
        for raw, frame in self.frames.items():
            output = self.native.render(self.ctx, raw, colors, opacity, scales)
            a, b = self.data_terms(output, frame); photo.append(a); cover.append(b)
        photo = torch.stack(photo).mean(); cover = torch.stack(cover).mean()
        gain_scale = max(abs(self.cfg['gain_bounds'][0] - 1), abs(self.cfg['gain_bounds'][1] - 1))
        bias_scale = max(abs(v) for v in self.cfg['bias_bounds'])
        gb = torch.cat(((self.gain - 1).reshape(1) / gain_scale, self.bias / bias_scale)).square().mean()
        sigma = self.log_sigma.square().mean() if self.log_sigma.numel() else self.gain * 0
        op = (self.opacity_logits.sigmoid() - self.initial_logits.sigmoid()).square().mean() if self.opacity_logits.numel() else self.gain * 0
        priors = self.cfg['prior_weights']
        total = photo / self.normalizers['photo'] + cover / self.normalizers['cover'] + priors['gain_bias'] * gb + priors['log_sigma'] * sigma + priors['opacity'] * op
        return total, dict(photo=float(photo.detach()), cover=float(cover.detach()), gain_bias_prior=float(gb.detach()),
                           sigma_prior=float(sigma.detach()), opacity_prior=float(op.detach()))

    def assert_frozen(self):
        torch = self.torch
        colors, op, scales = self.colors(), self.opacity(), self.scales()
        old = torch.ones(len(colors), dtype=torch.bool, device=self.device); old[self.new_pos] = False
        nonseam = torch.ones(len(colors), dtype=torch.bool, device=self.device); nonseam[self.seam_pos] = False
        need(torch.equal(colors[old], self.base_rgb[old]), 'Old RGB changed')
        need(torch.equal(op[nonseam], self.base_opacity[nonseam]), 'Opacity changed outside new seam')
        need(torch.equal(scales[nonseam], self.base_scales[nonseam]) and
             torch.equal(scales[:, 2], self.base_scales[:, 2]), 'Frozen sigma changed')

    def step(self):
        start = time.monotonic(); self.optimizer.zero_grad(set_to_none=True)
        total, terms = self.objective()
        need(bool(self.torch.isfinite(total)), 'Nonfinite objective')
        total.backward()
        for name, value in [('gain', self.gain), ('bias', self.bias), ('opacity', self.opacity_logits), ('sigma', self.log_sigma)]:
            if value.numel():
                need(value.grad is not None and bool(self.torch.isfinite(value.grad).all()), 'Missing/nonfinite gradient: ' + name)
        self.optimizer.step()
        with self.torch.no_grad():
            self.gain.clamp_(*self.cfg['gain_bounds']); self.bias.clamp_(*self.cfg['bias_bounds'])
            lo, hi = self.cfg['opacity_bounds']; self.opacity_logits.clamp_(math.log(lo / (1 - lo)), math.log(hi / (1 - hi)))
            lo, hi = self.cfg['tangent_sigma_multiplier_bounds']; self.log_sigma.clamp_(math.log(lo), math.log(hi))
        self.cursor += 1; self.assert_frozen()
        return dict(cursor=self.cursor, loss=float(total.detach()), gain=float(self.gain.detach()),
                    bias=self.bias.detach().cpu().tolist(), seconds=time.monotonic() - start, **terms)

    def snapshot(self):
        cpu = self.native.m.cpu_clone
        return dict(schema=SCHEMA, binding=self.binding, cursor=self.cursor,
                    gain=cpu(self.gain), bias=cpu(self.bias), opacity_logits=cpu(self.opacity_logits),
                    log_sigma=cpu(self.log_sigma), RGB=cpu(self.colors()), opacity=cpu(self.opacity()),
                    scales=cpu(self.scales()), optimizer=cpu(self.optimizer.state_dict()),
                    RNG=cpu(self.native.m.rng_state('cuda')), normalizers=copy.deepcopy(self.normalizers))

    def restore(self, value):
        need(value['schema'] == SCHEMA and value['binding'] == self.binding, 'Saved state binding differs')
        with self.torch.no_grad():
            for name in ('gain', 'bias', 'opacity_logits', 'log_sigma'):
                getattr(self, name).copy_(value[name].to(self.device))
        self.optimizer.load_state_dict(copy.deepcopy(value['optimizer'])); self.cursor = int(value['cursor'])
        self.normalizers = copy.deepcopy(value['normalizers']); self.native.m.restore_rng(value['RNG'], 'cuda')
        for key, tensor in [('RGB', self.colors()), ('opacity', self.opacity()), ('scales', self.scales())]:
            need(self.torch.equal(tensor.detach().cpu(), value[key]), 'Full restored native state differs: ' + key)
        self.assert_frozen()


def total_native_calls(native):
    current = native.m.CURRENT
    return int(native.m.PRIOR_CALLS + (current.native_calls if current is not None else 0))


def kcheck(ctx, state, cfg, native):
    rows = []
    with native.torch.no_grad():
        colors, opacity, scales = state.colors(), state.opacity(), state.scales()
        for raw in cfg['evaluation_frames']:
            output = native.render(ctx, raw, colors, opacity, scales, joint=True)
            parent = load(Path(cfg['parent_rendered']) / f'{raw:03d}.npz', ['joint_rgb', 'joint_alpha', 'K'])
            K = np.asarray(ctx.K_masks[raw], bool)
            need(np.array_equal(K, parent['K']), 'Retained-object K mask changed')
            rgb = output['rgb'].detach().cpu().numpy(); alpha = output['alpha'].detach().cpu().numpy()
            dr = float(np.abs(rgb - parent['joint_rgb'])[:, K].max(initial=0))
            da = float(np.abs(alpha - parent['joint_alpha'])[K].max(initial=0))
            rows.append(dict(raw=raw, RGB_max=dr, alpha_max=da))
            need(dr == 0 and da == 0, 'Joint K must remain strictly exact: ' + str(rows[-1]))
    return rows


def expected_road_arrays(ctx, state, asset_io):
    """只用于已有资产复核；保存规则与write_state_asset使用同一native差分。"""
    result = {k: v.copy() for k, v in ctx.road.items()}
    active = np.asarray(ctx.active_world_rows, np.int64)
    packet = ctx.base_packet(state.cfg['train_frames'][0])
    values, changed = asset_io._native_difference(state.colors(), ctx.base_rgb, 'colors')
    rows = changed.any(1); result['reference_srgb'][active[rows]] = values[rows]
    values, changed = asset_io._native_difference(state.opacity(), packet['opacities'], 'opacity')
    rows = changed.any(1); result['road_opacity'][active[rows]] = values.reshape(-1)[rows]
    values, changed = asset_io._native_difference(state.scales(), packet['scales'], 'scales')
    rr, axis = np.nonzero(changed)
    result['gaussian_sigma_m'][active[rr], axis] = values.astype(np.float64)[rr, axis] / float(ctx.road['meter_to_model'])
    return result


def ensure_asset(ctx, state, folder, asset_io, record):
    asset = Path(folder) / 'asset'
    if asset.exists():
        stored = load(asset / 'road.npz'); wanted = expected_road_arrays(ctx, state, asset_io)
        need(set(stored) == set(wanted) and all(np.array_equal(stored[k], wanted[k]) for k in wanted),
             'Existing asset differs from restored full state; preserve existing asset')
        provenance = load(asset / 'provenance.npz', ['known', 'unknown', 'source', 'transfer_new_row'])
        new = provenance['transfer_new_row']
        need(not provenance['known'][new].any() and provenance['unknown'][new].all() and
             (provenance['source'][new] == 2).all(), 'Transferred inference source labels changed')
        return asset, True
    asset_io.write_state_asset(ctx, asset, state.colors(), state.opacity(), state.scales(), record=record)
    return asset, False


def export_saved_or_verify(ctx, state, folder, cfg, native):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    packets = {}; records = []
    with native.torch.no_grad():
        colors, opacity, scales = state.colors(), state.opacity(), state.scales()
        for raw in cfg['evaluation_frames']:
            road = native.render(ctx, raw, colors, opacity, scales)
            joint = native.render(ctx, raw, colors, opacity, scales, joint=True)
            value = dict(road_rgb=road['rgb'].cpu().numpy(), road_alpha=road['alpha'].cpu().numpy(),
                         joint_rgb=joint['rgb'].cpu().numpy(), joint_alpha=joint['alpha'].cpu().numpy(),
                         K=np.asarray(ctx.K_masks[raw], bool))
            parent = load(Path(cfg['parent_rendered']) / f'{raw:03d}.npz', ['joint_rgb', 'joint_alpha', 'K'])
            need(np.array_equal(value['K'], parent['K']), 'K mask changed at export')
            K = value['K']
            need(float(np.abs(value['joint_rgb'] - parent['joint_rgb'])[:, K].max(initial=0)) == 0 and
                 float(np.abs(value['joint_alpha'] - parent['joint_alpha'])[K].max(initial=0)) == 0,
                 'Joint K changed at export')
            path = folder / f'{raw:03d}.npz'; reused = path.exists(); maximum = 0.
            if reused:
                stored = load(path)
                need(set(stored) == set(value), 'Saved render packet fields changed')
                for key in value:
                    if value[key].dtype.kind == 'b':
                        need(np.array_equal(stored[key], value[key]), 'Saved render mask changed')
                    else:
                        maximum = max(maximum, float(np.abs(stored[key] - value[key]).max(initial=0)))
                need(maximum <= cfg['public_reload_tolerance'], 'Existing render differs; preserve it')
            else:
                np.savez_compressed(path, **value)
                for name in ('road', 'joint'):
                    image = np.rint(np.clip(value[name + '_rgb'].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(image).save(folder / f'{raw:03d}_{name}.png')
                Image.fromarray(np.rint(np.clip(value['road_alpha'], 0, 1) * 255).astype(np.uint8)).save(folder / f'{raw:03d}_alpha.png')
            packets[raw] = value
            records.append(dict(raw=raw, existing_render_preserved=reused, maximum_readback_difference=maximum, joint_K_RGB_alpha_exact=True))
    return packets, records


def persist_and_public_reload(ctx, state, folder, cfg, native, asset_io, review_path):
    start = total_native_calls(native)
    state.assert_frozen()
    record = dict(stage='local_seam_fit_state', schema=SCHEMA, cursor=state.cursor,
                  gain=float(state.gain.detach()), RGB_bias=state.bias.detach().cpu().tolist(),
                  binding=state.binding, normalizers=state.normalizers, new_diffusion_calls=0,
                  target_policy=cfg['target_policy'], photograph_error=cfg['photograph_error'],
                  old_parameters_frozen=True, new_core_opacity_sigma_frozen=True,
                  per_Gaussian_free_RGB_parameters=0, global_RGB_parameters=4)
    asset, reused = ensure_asset(ctx, state, folder, asset_io, record)
    expected, renders = export_saved_or_verify(ctx, state, Path(folder) / 'rendered', cfg, native)
    loaded = native.open_context(asset, frames=tuple(cfg['evaluation_frames']))
    comparisons = []
    with native.torch.no_grad():
        for raw in cfg['evaluation_frames']:
            road = native.render(loaded, raw)
            joint = native.render(loaded, raw, joint=True)
            actual = dict(road_rgb=road['rgb'].cpu().numpy(), road_alpha=road['alpha'].cpu().numpy(),
                          joint_rgb=joint['rgb'].cpu().numpy(), joint_alpha=joint['alpha'].cpu().numpy())
            errors = {key: float(np.abs(value - expected[raw][key]).max(initial=0)) for key, value in actual.items()}
            need(max(errors.values()) <= cfg['public_reload_tolerance'], 'Public asset reload mismatch: ' + str(errors))
            K = expected[raw]['K']; parent = load(Path(cfg['parent_rendered']) / f'{raw:03d}.npz', ['joint_rgb', 'joint_alpha'])
            need(float(np.abs(actual['joint_rgb'] - parent['joint_rgb'])[:, K].max(initial=0)) == 0 and
                 float(np.abs(actual['joint_alpha'] - parent['joint_alpha'])[K].max(initial=0)) == 0,
                 'Public loaded asset changed K')
            comparisons.append(dict(raw=raw, **errors, joint_K_RGB_alpha_exact=True))
    public = dict(passed=True, asset=str(asset), existing_asset_preserved=reused, renders=renders,
                  comparisons=comparisons, nativeF=total_native_calls(native) - start,
                  optimizer_updates=0, new_diffusion_calls=0)
    write(review_path, public)
    return asset, public


def save_checkpoint(path, state, native):
    native.save(Path(path), state.snapshot())


def verify_input_binding(state):
    for path, digest in state.binding['files'].items():
        need(sha(path) == digest, 'A frozen input changed during fitting: ' + path)


def frame_records(state):
    return {str(raw): dict(photo_A_pixels=int(f['photo_A'].sum()),
                          photo_cross_frame_same_pixels=int(f['photo_same'].sum()),
                          trusted_cross_frame_same_candidates=int(f['same_requested'].sum()),
                          photo_pixels=int(f['photo'].sum()), coverage_pixels=int(f['cover'].sum()))
            for raw, f in state.frames.items()}


def smoke(cfg, native, asset_io):
    folder = Path(cfg['smoke']); folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pt'; receipt_path = folder / 'receipt.json'
    start = time.monotonic(); calls = total_native_calls(native)
    ctx = native.open_context(Path(cfg['transfer_asset']), frames=tuple(cfg['evaluation_frames']))
    state = FitState(ctx, cfg, native)
    if receipt_path.exists():
        receipt = read(receipt_path)
        need(receipt.get('passed') is True, 'Preserve failed smoke; do not silently reuse it')
        saved = native.torch.load(checkpoint, map_location='cpu', weights_only=False)
        need(saved['binding'] == state.binding and saved['cursor'] == cfg['smoke_steps'], 'Passed smoke binding differs')
        need(receipt['binding'] == state.binding, 'Smoke receipt binding differs')
        return dict(reused_passed_smoke=True, additional_optimizer_updates=0, nativeF=total_native_calls(native) - calls)
    need(not checkpoint.exists(), 'Preserve unfinished smoke checkpoint; resolve engineering replay explicitly')
    trace = []
    for _ in range(cfg['smoke_steps']):
        trace.append(state.step())
    save_checkpoint(checkpoint, state, native)
    saved = state.snapshot()
    state.step(); expected = state.snapshot()
    restored = FitState(ctx, cfg, native); restored.restore(saved); restored.step()
    difference = native.m.max_tree_difference(expected, restored.snapshot())
    need(difference < cfg['resume_tolerance'], 'Smoke next-state replay differs: ' + str(difference))
    restored.restore(saved)
    K = kcheck(ctx, restored, cfg, native)
    asset, public = persist_and_public_reload(ctx, restored, folder, cfg, native, asset_io, folder / 'public_reload.json')
    verify_input_binding(restored)
    receipt = dict(schema=SCHEMA, passed=True, checkpoint_cursor=cfg['smoke_steps'],
        smoke_optimizer_updates=cfg['smoke_steps'], disposable_next_step_updates=2,
        total_discarded_updates=cfg['smoke_steps'] + 2, native_trainingF=(cfg['smoke_steps'] + 2) * len(cfg['train_frames']),
        native_trainingB=(cfg['smoke_steps'] + 2) * len(cfg['train_frames']), actual_nativeF=total_native_calls(native) - calls,
        next_state_max_difference=difference, K=K, public_reload=public, binding=restored.binding,
        normalizers=restored.normalizers, frames=frame_records(restored), trace=trace,
        asset=str(asset), formal_start='fresh TRANSFER, gain1/bias0/native opacity/sigma multiplier1',
        seconds=time.monotonic() - start, new_diffusion_calls=0)
    write(receipt_path, receipt)
    return receipt


def formal(cfg, native, asset_io):
    folder = Path(cfg['output']); folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pt'; receipt_path = folder / 'receipt.json'
    start = time.monotonic(); calls = total_native_calls(native)
    ctx = native.open_context(Path(cfg['transfer_asset']), frames=tuple(cfg['evaluation_frames']))
    state = FitState(ctx, cfg, native)
    write(folder / 'executed_config.json', cfg)
    if checkpoint.exists():
        state.restore(native.torch.load(checkpoint, map_location='cpu', weights_only=False))
    else:
        need(not receipt_path.exists(), 'Formal receipt exists without checkpoint')
        need(native.torch.equal(state.colors(), state.base_rgb) and
             native.torch.equal(state.opacity(), state.base_opacity) and
             native.torch.equal(state.scales(), state.base_scales), 'Formal fresh state differs from TRANSFER')
        save_checkpoint(folder / 'initial_state.pt', state, native)
        save_checkpoint(checkpoint, state, native)
    need(0 <= state.cursor <= cfg['formal_steps'], 'Formal cursor outside fixed budget')
    initial_cursor = state.cursor
    if receipt_path.exists() and read(receipt_path).get('complete') is True:
        need(state.cursor == cfg['formal_steps'], 'Completed formal receipt cursor differs')
        _, public = persist_and_public_reload(ctx, state, folder, cfg, native, asset_io,
            Path(cfg['recovery']) / 'completed_entry_public_reload.json')
        verify_input_binding(state)
        result = dict(passed=True, checkpoint_cursor=state.cursor, completed_asset_reused=True,
            optimizer_updates=0, original_formal_receipt_preserved=True, checkpoint_preserved=True,
            actual_nativeF=total_native_calls(native) - calls, public_reload=public, new_diffusion_calls=0)
        write(Path(cfg['recovery']) / 'completed_entry.json', result)
        return result
    history_path = folder / 'history.json'
    history = read(history_path) if history_path.exists() else []
    while state.cursor < cfg['formal_steps']:
        result = state.step()
        if state.cursor % cfg['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint, state, native)
            K = kcheck(ctx, state, cfg, native)
            history.append(dict(**result, K=K))
            write(history_path, history)
            write(folder / 'progress.json', dict(state='running', **result))
            print(json.dumps(result), flush=True)
    asset, public = persist_and_public_reload(ctx, state, folder, cfg, native, asset_io, folder / 'public_reload.json')
    verify_input_binding(state)
    receipt = dict(schema=SCHEMA, complete=True, formal_optimizer_updates=state.cursor,
        current_process_start_cursor=initial_cursor, current_process_optimizer_updates=state.cursor - initial_cursor,
        formal_training_nativeF=state.cursor * len(cfg['train_frames']), formal_training_nativeB=state.cursor * len(cfg['train_frames']),
        current_process_actual_nativeF=total_native_calls(native) - calls,
        checkpoint=str(checkpoint), asset=str(asset), rendered=str(folder / 'rendered'), binding=state.binding,
        gain=float(state.gain.detach()), RGB_bias=state.bias.detach().cpu().tolist(), normalizers=state.normalizers,
        frames=frame_records(state), active_new_rows=len(state.new_world), active_new_seam_rows=len(state.seam_world),
        public_reload=public, old_all_parameters_frozen=True, new_core_opacity_sigma_frozen=True,
        XYZ_normal_normalSigma_quaternion_frozen=True, per_Gaussian_free_RGB_parameters=0,
        global_RGB_parameters=4, new_diffusion_calls=0, diffusion_RGB_loaded=False,
        production_promoted=False, human_verdict=None, seconds=time.monotonic() - start)
    write(receipt_path, receipt)
    return receipt


def resume_probe(cfg, native):
    checkpoint = Path(cfg['output']) / 'checkpoint.pt'
    need(checkpoint.is_file(), 'Formal checkpoint is missing')
    stat = checkpoint.stat(); start = time.monotonic(); calls = total_native_calls(native)
    saved = native.torch.load(checkpoint, map_location='cpu', weights_only=False)
    ctx = native.open_context(Path(cfg['transfer_asset']), frames=tuple(cfg['evaluation_frames']))
    first = FitState(ctx, cfg, native); first.restore(saved); first.step(); expected = first.snapshot()
    del first
    second = FitState(ctx, cfg, native); second.restore(saved); second.step(); actual = second.snapshot()
    difference = native.m.max_tree_difference(expected, actual)
    need(difference < cfg['resume_tolerance'], 'Independent next-state probe differs: ' + str(difference))
    verify_input_binding(second)
    now = checkpoint.stat()
    need(now.st_size == stat.st_size and now.st_mtime_ns == stat.st_mtime_ns, 'Probe changed formal checkpoint')
    result = dict(schema=SCHEMA, passed=True, source_cursor=saved['cursor'], next_cursor=second.cursor,
        maximum_full_state_difference=difference, discarded_optimizer_updates=2,
        native_trainingF=2 * len(cfg['train_frames']), native_trainingB=2 * len(cfg['train_frames']),
        actual_nativeF=total_native_calls(native) - calls, checkpoint_preserved=True,
        persistent_asset_preserved=True, normalizers_restored=True, masks_binding=second.mask_signature,
        seconds=time.monotonic() - start, new_diffusion_calls=0)
    write(Path(cfg['recovery']) / 'fresh_process_next_step.json', result)
    return result


class ProcessLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            old = read(self.path); pid = int(old['pid'])
            process = Path('/proc') / str(pid) / 'cmdline'
            if process.exists() and 'fit_seams.py' in process.read_text(errors='replace'):
                raise RuntimeError('Existing fit process remains active: ' + str(old))
            self.path.rename(self.path.with_name(self.path.name + '.stale.' + str(time.time_ns())))
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(dict(pid=os.getpid(), started_at=time.time()), stream)
        return self

    def __exit__(self, *unused):
        if self.path.exists() and read(self.path).get('pid') == os.getpid():
            self.path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, help='Overrides config.root, then ROAD_TRANSFER_RUN_ROOT, then the legacy default')
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--resume-probe', action='store_true')
    parser.add_argument('--write-default-config', type=Path)
    args = parser.parse_args()
    cfg = configuration(args.config, args.root)
    if args.write_default_config is not None:
        write(args.write_default_config, cfg)
    if not args.execute_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        result = preflight(cfg)
        write(Path(cfg['root']) / 'qualification/seam_fit_preflight.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    preflight(cfg, require_ready=True)
    native, asset_io = runtime(cfg)
    torch = native.torch
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    np.random.seed(cfg['seed'])
    import random
    random.seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with ProcessLock(Path(cfg['output']) / 'fit.lock'):
        try:
            if args.resume_probe:
                result = resume_probe(cfg, native)
            else:
                smoke_result = smoke(cfg, native, asset_io)
                # 正式从TRANSFER重建，所有smoke参数和Adam均丢弃。
                torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
                np.random.seed(cfg['seed']); random.seed(cfg['seed'])
                result = formal(cfg, native, asset_io)
                result = dict(result, smoke_reused=smoke_result.get('reused_passed_smoke', False))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except BaseException:
            write(Path(cfg['root']) / 'failures' / ('fit_seams_' + str(time.time_ns()) + '.json'),
                  dict(error=traceback.format_exc(), configuration=cfg, new_diffusion_calls=0))
            raise
        finally:
            native.close()


if __name__ == '__main__':
    main()

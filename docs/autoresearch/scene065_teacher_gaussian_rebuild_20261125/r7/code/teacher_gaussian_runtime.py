"""Teacher-colour Gaussian completion with five trainable parameter groups.

Original native30k rows are frozen. New rows are ordinary 3D Gaussians in the
same depth-sorted native raster. Topology changes migrate survivor Adam moments
and initialize newborn moments to zero; no permanent green-plane holes exist.
"""
from pathlib import Path
from collections import OrderedDict
import copy, hashlib, importlib.util, json, math, sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

C0 = .28209479177387814
PARAMETERS = ('xyz', 'log_scales', 'raw_quaternion', 'opacity_logits', 'dc')
BUFFERS = ('prior_xyz', 'prior_log_scales', 'prior_raw_quaternion', 'prior_opacity_logits', 'prior_dc',
           'row_ids', 'parent_ids', 'source_candidate_ids', 'confidence', 'teacher_color', 'source_raw', 'source_uv',
           'normal_local', 'grad_accum', 'vis_count', 'max_screen_radius', 'born_step')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def pinned(spec):
    if isinstance(spec, (str, Path)):
        return Path(spec)
    path = Path(spec['path']); assert path.is_file() and sha(path) == spec['sha256'], path
    return path

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def subset(packet, indices):
    n = len(packet['means3D'])
    return {k: v[indices] if v is not None and hasattr(v, 'ndim') and v.ndim and len(v) == n else v for k, v in packet.items()}

def qmul(a, b):
    w, x, y, z = a.unbind(-1); v, i, j, k = b.unbind(-1)
    return torch.stack([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v], -1)

def rotation_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)

@torch.no_grad()
def frozen_fragment_packet(parents, offsets, scale_factors, opacity_factors, view, near_model):
    """Derive frozen 3D children; offsets use each parent's local sigma basis."""
    n = len(parents['means3D'])
    result = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in parents.items()}
    offsets = offsets.to(parents['means3D']); factors = scale_factors.to(offsets).reshape(n)
    opacity_factors = opacity_factors.to(offsets).reshape(n)
    covariance = parents['cov3D_precomp']
    if covariance is not None:
        if covariance.ndim == 2 and covariance.shape[1] == 6:
            xx, xy, xz, yy, yz, zz = covariance.unbind(-1)
            full = torch.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], -1).reshape(n, 3, 3)
        else:
            assert covariance.shape == (n, 3, 3)
            full = covariance
        chol, info = torch.linalg.cholesky_ex(full)
        if bool((info != 0).any()):
            # Finite precision can make very thin learned covariances singular.
            bad = info != 0
            jitter = full[bad].diagonal(dim1=-2, dim2=-1).abs().amax(-1).clamp_min(torch.finfo(full.dtype).tiny)*1e-6
            chol[bad] = torch.linalg.cholesky(full[bad]+torch.diag_embed(jitter[:, None].expand(-1, 3)))
        delta = torch.bmm(chol, offsets[..., None]).squeeze(-1)
        result['cov3D_precomp'] = covariance*factors.reshape(n, *([1]*(covariance.ndim-1))).square()
        result['scales'] = result['rotations'] = None
    else:
        assert parents['scales'] is not None and parents['rotations'] is not None
        rotation = rotation_matrix(torch.nn.functional.normalize(parents['rotations'], dim=-1))
        delta = torch.bmm(rotation, (parents['scales']*offsets)[..., None]).squeeze(-1)
        result['scales'] = parents['scales']*factors[:, None]
    result['means3D'] = parents['means3D']+delta
    result['means2D'] = torch.zeros_like(result['means3D'])
    alpha = parents['opacities'].clamp(0, 1)
    result['opacities'] = -torch.expm1(torch.log1p(-alpha)*opacity_factors.reshape(n, *([1]*(alpha.ndim-1))))
    depth = result['means3D'] @ view[:3, 2]+view[3, 2]
    result['features'][:, 0] = depth
    # Raw packet mask is temporal eligibility; native center-near is applied
    # separately to the retained prefix, never inherited by fragment children.
    result['mask'] = parents['mask'] & (depth > near_model)
    assert all(not v.requires_grad for v in result.values() if torch.is_tensor(v))
    assert all(bool(torch.isfinite(v).all()) for v in result.values() if torch.is_tensor(v) and v.is_floating_point())
    return result

def candidate_ids(arrays):
    if 'source_candidate_ids' in arrays:
        return np.asarray(arrays['source_candidate_ids'], np.int64)
    n = len(arrays['xyz']); raw = np.asarray(arrays.get('source_raw', np.zeros(n)), np.int64)
    uv = np.asarray(arrays.get('source_uv', arrays.get('source_pixel_xy', np.zeros((n, 2)))), np.float64)
    xy = np.rint(uv).astype(np.int64)
    return raw * 100000000 + xy[:, 1] * 10000 + xy[:, 0]

def initial_tensors(arrays, device, born_step=0):
    n = len(arrays['xyz']); asf = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device).clone()
    xyz = asf(arrays['xyz']); scales = asf(arrays['scales']).clamp_min(1e-5)
    quaternion = asf(arrays['rotation']); quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    opacity = asf(arrays.get('opacity', np.full(n, .2))).reshape(n, 1).clamp(1e-5, 1-1e-5)
    rgb = asf(arrays.get('teacher_color', arrays['rgb']))
    if 'coefficients' in arrays:
        dc = asf(arrays['coefficients'][:, 0])
    else:
        srgb = torch.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb.clamp_min(1e-10).pow(1/2.4)-.055)
        dc = (srgb-.5)/C0
    values = OrderedDict(xyz=xyz, log_scales=scales.log(), raw_quaternion=quaternion,
                         opacity_logits=torch.logit(opacity), dc=dc)
    values.update(prior_xyz=asf(arrays.get('prior_xyz', arrays['xyz'])), prior_log_scales=scales.log().clone(),
                  prior_raw_quaternion=quaternion.clone(), prior_opacity_logits=torch.logit(opacity).clone(), prior_dc=dc.clone(),
                  row_ids=torch.arange(n, dtype=torch.int64, device=device),
                  parent_ids=torch.full((n,), -1, dtype=torch.int64, device=device),
                  source_candidate_ids=torch.as_tensor(candidate_ids(arrays), dtype=torch.int64, device=device),
                  confidence=asf(arrays.get('confidence', np.ones(n))).reshape(n), teacher_color=rgb,
                  source_raw=torch.as_tensor(arrays.get('source_raw', np.zeros(n)), dtype=torch.int64, device=device),
                  source_uv=asf(arrays.get('source_uv', arrays.get('source_pixel_xy', np.zeros((n, 2))))),
                  normal_local=asf(arrays.get('normal_local', arrays.get('prior_normal', np.tile([0, 0, 1], (n, 1))))),
                  grad_accum=torch.zeros(n, device=device), vis_count=torch.zeros(n, device=device),
                  max_screen_radius=torch.zeros(n, device=device), born_step=torch.full((n,), born_step, dtype=torch.int64, device=device))
    assert all(bool(torch.isfinite(v).all()) for k, v in values.items() if v.is_floating_point())
    return values

class FreshGaussianField(torch.nn.Module):
    def __init__(self, arrays, device='cuda'):
        super().__init__(); values = initial_tensors(arrays, device)
        for name in PARAMETERS:
            self.register_parameter(name, torch.nn.Parameter(values[name], requires_grad=True))
        for name in BUFFERS:
            self.register_buffer(name, values[name])
        self.next_row_id = len(self.xyz)

    def named_new_parameters(self):
        return OrderedDict((name, getattr(self, name)) for name in PARAMETERS)

    @property
    def scales(self):
        return self.log_scales.exp()

    @property
    def rotation(self):
        return torch.nn.functional.normalize(self.raw_quaternion, dim=-1, eps=1e-12)

    @property
    def opacity(self):
        return self.opacity_logits.sigmoid()

    def values(self):
        return {name: getattr(self, name) for name in (*PARAMETERS, *BUFFERS)}

    def capture(self):
        return dict(parameters={k: v.detach().cpu().clone() for k, v in self.named_new_parameters().items()},
                    **{k: getattr(self, k).detach().cpu().clone() for k in BUFFERS}, next_row_id=self.next_row_id)

    def replace(self, values, next_row_id=None):
        device = self.xyz.device; n = len(values['xyz'])
        for name in PARAMETERS:
            value = values[name].detach().to(device=device, dtype=torch.float32).clone()
            assert len(value) == n
            self._parameters[name] = torch.nn.Parameter(value, requires_grad=True)
        for name in BUFFERS:
            value = values[name].detach().to(device=device).clone(); assert len(value) == n
            self._buffers[name] = value
        if next_row_id is not None:
            self.next_row_id = int(next_row_id)

    def restore(self, state):
        values = dict(state['parameters']); values.update({k: state[k] for k in BUFFERS})
        self.replace(values, state['next_row_id'])

    def append(self, arrays, optimizer=None, born_step=0):
        rows = initial_tensors(arrays, self.xyz.device, born_step); n = len(self.xyz); added = len(rows['xyz'])
        rows['row_ids'] = torch.arange(self.next_row_id, self.next_row_id+added, device=self.xyz.device, dtype=torch.int64)
        old_named = self.named_new_parameters(); old = self.values()
        values = {k: torch.cat([old[k].detach(), rows[k]], 0) for k in (*PARAMETERS, *BUFFERS)}
        self.replace(values, self.next_row_id+added)
        if optimizer is not None:
            from full_gaussian_lifecycle import remap_adam_rows
            source = torch.cat([torch.arange(n, device=self.xyz.device), torch.full((added,), -1, device=self.xyz.device)])
            remap_adam_rows(optimizer, old_named, self.named_new_parameters(), source)
        return added

    def prune(self, keep, optimizer=None):
        keep = torch.as_tensor(keep, device=self.xyz.device)
        indices = torch.nonzero(keep).reshape(-1) if keep.dtype == torch.bool else keep.long()
        assert len(indices) > 0
        old_named = self.named_new_parameters(); old = self.values(); removed = len(self.xyz)-len(indices)
        self.replace({k: old[k][indices] for k in (*PARAMETERS, *BUFFERS)}, self.next_row_id)
        if optimizer is not None:
            from full_gaussian_lifecycle import remap_adam_rows
            remap_adam_rows(optimizer, old_named, self.named_new_parameters(), indices)
        return removed

class TeacherGaussianRuntime:
    def __init__(self, contract, output, train_guard=None):
        self.cfg = cfg = read(contract); self.torch = torch
        assert cfg['schema'] in ('scene065.teacher_gaussian.runtime.v1', 'scene065.teacher_gaussian_reconstruction.runtime.v1', 'scene065.teacher_visibility.runtime.v1')
        adapter_path = pinned(cfg['native_adapter_code']); sys.path.insert(0, str(adapter_path.parent))
        spec = importlib.util.spec_from_file_location('_teacher_original_native30k_adapter', adapter_path)
        native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        self.base = base = native.Native30KAdapter(pinned(cfg['native30k_contract']), output/'native30k')
        self.frames = list(cfg['frames']); assert set(self.frames) <= set(base.frames)
        seed_spec = cfg.get('teacher_seed', cfg.get('new_field')); assert seed_spec is not None
        seed = load(pinned(seed_spec)); self.candidates = load(pinned(cfg['candidate_pool']))
        for data in [seed, self.candidates]:
            assert bool(np.asarray(data['source_independent_teacher']).all())
            assert not bool(np.asarray(data['source_is_reprojection']).any())
            assert 'active' not in data or bool(np.asarray(data['active']).all()), 'No inherited permanent active carve'
        self.field = FreshGaussianField(seed)
        self.road_height_sigma_m = float(cfg.get('road_height_sigma_m', .05))
        assert self.road_height_sigma_m > 0
        self.road_plane_origin = torch.as_tensor(seed.get('prior_xyz', seed['xyz'])[0], dtype=torch.float32, device='cuda').clone()
        self.road_plane_normal = torch.as_tensor(seed.get('normal_local', seed['prior_normal'])[0], dtype=torch.float32, device='cuda').clone() if 'normal_local' not in seed else torch.as_tensor(seed['normal_local'][0], dtype=torch.float32, device='cuda').clone()
        self.road_plane_normal /= self.road_plane_normal.norm()
        self.candidate_ids = candidate_ids(self.candidates); self.candidate_cursor = 0
        self.used_candidate_ids = set(candidate_ids(seed).tolist()); self.topology_events = []
        self.densify_cfg = dict(cfg.get('densification', {})); self._render_records = []; self._residual_by_raw = {}
        pose_data = read(pinned(cfg['pose'])); self.world_from_patch = np.asarray(pose_data['world_from_patch'], np.float64)
        model_from_patch = base.transform @ self.world_from_patch
        self.pose = torch.as_tensor(model_from_patch.astype(np.float32), device='cuda')
        self.unit = float(np.linalg.norm(model_from_patch[:3, :3], axis=0).mean())
        quaternion = Rotation.from_matrix(model_from_patch[:3, :3]/self.unit).as_quat()[[3, 0, 1, 2]]
        self.pose_q = torch.as_tensor(quaternion.astype(np.float32), device='cuda')
        sys.path.insert(0, str(Path(base.cfg['repo'])/'scripts/worldsim_v82'))
        from drivingeditor_dynamic_edit.v44_metric_raster import metric_raster
        self.raster = metric_raster(base.scene._bindings().raster, 1/base.meter_to_model)
        retirement = load(pinned(cfg['context_retirement']))
        removed = torch.as_tensor(retirement['remove_global_row_ids'], device='cuda')
        semantic = base.owners(self.frames[0], semantic=False)
        protected = torch.as_tensor(retirement['protected_green_core_global_row_ids'], device='cuda')
        assert len(protected) > 0 and not torch.isin(protected, removed).any()
        fragment_spec = cfg.get('retained_fragments')
        self.retained_fragments = None; fragment_hash = None; fragment_parent_ids = None
        if fragment_spec is not None:
            fragment_path = pinned(fragment_spec); recipe = load(fragment_path); fragment_hash = sha(fragment_path)
            parent_ids = np.asarray(recipe['parent_global_row_ids'])
            assert parent_ids.ndim == 1 and parent_ids.dtype.kind in 'iu'
            count = len(parent_ids)
            assert recipe['offsets'].shape == (count, 3)
            assert recipe['scale_factors'].shape == recipe['opacity_factors'].shape == (count,)
            assert all(np.isfinite(recipe[k]).all() for k in ('offsets', 'scale_factors', 'opacity_factors'))
            assert np.all(recipe['scale_factors'] > 0) and np.all(recipe['opacity_factors'] > 0)
            fragment_parent_ids = torch.as_tensor(parent_ids.astype(np.int64), device=base.global_row_ids.device)
            assert bool(((fragment_parent_ids >= 0) & (fragment_parent_ids < len(base.global_row_ids))).all())
            assert torch.equal(base.global_row_ids[fragment_parent_ids], fragment_parent_ids)
            assert bool(torch.isin(fragment_parent_ids, removed).all()), 'Fragment parents must be retired, never duplicated'
            self.retained_fragments = dict(parent_global_row_ids=fragment_parent_ids,
                **{k: torch.as_tensor(recipe[k], dtype=torch.float32, device=fragment_parent_ids.device) for k in ('offsets', 'scale_factors', 'opacity_factors')})
            self._fragment_signature = {k: (id(v), v.data_ptr(), v._version) for k, v in self.retained_fragments.items()}
        self.context = {}; self.context_indices = {}; self.calls = 0; self.backward_calls = 0; self.last_local_means2D = None
        drop = torch.isin(base.global_row_ids, removed)
        with torch.no_grad():
            for raw in self.frames:
                full_packet = base.packet(raw); ids = torch.nonzero(base.native_visible_mask(raw, full_packet) & ~drop).reshape(-1)
                packet = subset(full_packet, ids); features = packet['features'].new_zeros((len(ids), 24))
                features[:, 0] = packet['features'][:, 0]; features[:, 1:3] = 1
                features[:, 3] = (semantic[ids] == 21).float()
                features[:, 5] = ((semantic[ids] > 0) & (semantic[ids] != 21)).float()
                packet = dict(packet, features=features)
                if self.retained_fragments is not None and len(fragment_parent_ids):
                    parents = subset(full_packet, fragment_parent_ids)
                    f = parents['features'].new_zeros((len(fragment_parent_ids), 24)); f[:, 1:3] = 1
                    f[:, 3] = (semantic[fragment_parent_ids] == 21).float()
                    f[:, 5] = ((semantic[fragment_parent_ids] > 0) & (semantic[fragment_parent_ids] != 21)).float()
                    parents = dict(parents, features=f)
                    fragments = frozen_fragment_packet(parents, self.retained_fragments['offsets'],
                        self.retained_fragments['scale_factors'], self.retained_fragments['opacity_factors'],
                        base.cameras[raw].world_view_transform, .2*base.meter_to_model)
                    packet = base.scene._bindings().merge_rows(packet, fragments)
                    ids = torch.cat([ids, fragment_parent_ids])
                self.context[raw] = packet; self.context_indices[raw] = ids
        self.frozen_assets = dict(base.frozen_assets)
        for key in ['native_adapter_code', 'native30k_contract', 'context_retirement', 'pose', 'candidate_pool']:
            if isinstance(cfg[key], dict): self.frozen_assets[key] = cfg[key]['sha256']
        if isinstance(seed_spec, dict): self.frozen_assets['teacher_seed'] = seed_spec['sha256']
        if fragment_hash is not None: self.frozen_assets['retained_fragments'] = fragment_hash
        self.assert_fixed()
        write(output/'composition.json', dict(schema='scene065.teacher_gaussian.composition.v1', original_native30k_frozen=True,
            new_parameter_groups=list(PARAMETERS), initial_new_rows=len(self.field.xyz), candidate_rows=len(self.candidate_ids),
            retained_fragments=None if fragment_hash is None else dict(path=str(fragment_path), sha256=fragment_hash,
                rows=len(fragment_parent_ids), unique_parents=int(torch.unique(fragment_parent_ids).numel()),
                frozen=True, near_m=.2, original_prefix_native_near=True),
            geometry_optimizer_enabled=True, permanent_green_carve=False, raster='one native depth-sort retained30k+new3DGaussians'))

    def named_new_parameters(self):
        return self.field.named_new_parameters()

    @property
    def trainable(self):
        return list(self.named_new_parameters().values())

    @property
    def initial_parameters(self):
        return [self.field.prior_xyz, self.field.prior_log_scales, self.field.prior_raw_quaternion, self.field.prior_opacity_logits, self.field.prior_dc]

    def make_optimizer(self, cfg):
        rates = dict(xyz=.003, log_scales=.005, raw_quaternion=.001, opacity_logits=.05, dc=.03544907701811032)
        rates.update(cfg.get('learning_rates', {}))
        return torch.optim.Adam([dict(params=[p], name=name, lr=float(rates[name])) for name, p in self.named_new_parameters().items()],
                                eps=float(cfg.get('adam_eps', 1e-8)))

    def local_packet(self, raw):
        field = self.field; camera = self.base.cameras[raw]
        means = field.xyz @ self.pose[:3, :3].T + self.pose[:3, 3]
        color = (.5 + C0*field.dc).clamp(0, 1)
        linear = torch.where(color <= .04045, color/12.92, ((color+.055)/1.055).pow(2.4))
        transform = self.cfg.get('photometric_by_raw', {}).get(str(raw), {})
        gain = torch.as_tensor(transform.get('gain', [1, 1, 1]), dtype=color.dtype, device=color.device)
        bias = torch.as_tensor(transform.get('bias', [0, 0, 0]), dtype=color.dtype, device=color.device)
        linear = ((linear-bias)/gain).clamp(0, 1)
        color = torch.where(linear <= .0031308, 12.92*linear, 1.055*linear.clamp_min(1e-10).pow(1/2.4)-.055)
        depth = means @ camera.world_view_transform[:3, 2] + camera.world_view_transform[3, 2]
        features = means.new_zeros((len(means), 24)); features[:, 0] = depth; features[:, 1] = 1; features[:, 4] = 1
        screen = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()); self.last_local_means2D = screen
        rotations = qmul(self.pose_q.expand_as(field.rotation), field.rotation); scales = field.scales*self.unit
        covariance = None
        if self.context[raw]['cov3D_precomp'] is not None:
            matrix = rotation_matrix(rotations) * scales[:, None, :]
            covariance_full = matrix @ matrix.transpose(1, 2)
            covariance = torch.stack([covariance_full[:, 0, 0], covariance_full[:, 0, 1], covariance_full[:, 0, 2], covariance_full[:, 1, 1], covariance_full[:, 1, 2], covariance_full[:, 2, 2]], -1)
            scales = rotations = None
        return dict(means3D=means, means2D=screen, shs=None, colors_precomp=color, features=features, opacities=field.opacity,
                    scales=scales, rotations=rotations, cov3D_precomp=covariance, mask=depth > .2*self.base.meter_to_model)

    def packet(self, raw, component='shared'):
        if component == 'context': return self.context[raw]
        local = self.local_packet(raw)
        if component in ('local', 'road'): return local
        assert component == 'shared'
        return self.base.scene._bindings().merge_rows(self.context[raw], local)

    def render(self, raw, component='shared'):
        packet = self.packet(raw, component); camera = self.base.cameras[raw]
        out = self.raster(camera, self.base.pipe, 0, packet, self.base.scene._background); self.calls += 1
        if component != 'context' and torch.is_grad_enabled():
            self._render_records.append((self.last_local_means2D, out['radii'][-len(self.field.xyz):].detach()))
        rgb = out['rgb']
        if component in ('shared', 'context'):
            rgb = rgb + (1-out['alpha'])*self.base.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        return dict(rgb=rgb, alpha=out['alpha'], local_effective_alpha=out['extra'][2], actor_alpha=out['extra'][1],
                    context_alpha=out['extra'][0], other_dynamic_alpha=out['extra'][3],
                    retained_foreground_alpha=(out['extra'][1]+out['extra'][3]).clamp(0,1),
                    depth_sum_m=out['depth_sum']/self.base.meter_to_model)

    def accumulate_render_stats(self):
        with torch.no_grad():
            for screen, radii in self._render_records:
                if screen.grad is None: continue
                assert len(screen) == len(self.field.xyz), 'Collect screen gradients before changing topology'
                seen = radii > 0; grad = screen.grad[:, :2].norm(dim=-1)
                self.field.grad_accum[seen] += grad[seen]; self.field.vis_count[seen] += 1
                self.field.max_screen_radius = torch.maximum(self.field.max_screen_radius, radii.float())
        self._render_records.clear()

    def observe_residual(self, raw, residual):
        self._residual_by_raw[int(raw)] = residual.detach() if torch.is_tensor(residual) else np.asarray(residual)

    def regularization_losses(self):
        f = self.field
        prior_rotation = torch.nn.functional.normalize(f.prior_raw_quaternion, dim=-1)
        return dict(geometry_prior=(f.xyz-f.prior_xyz).square().mean(), scale_prior=(f.log_scales-f.prior_log_scales).square().mean(),
                    rotation_prior=(1-(f.rotation*prior_rotation).sum(-1).square()).clamp_min(0).mean(),
                    color_prior=(f.dc-f.prior_dc).square().mean(),
                    road_height_prior=(((f.xyz-self.road_plane_origin)*self.road_plane_normal).sum(-1)/self.road_height_sigma_m).square().mean())

    def post_optimizer_step(self):
        """Numerical safeguards without fixing any geometric degree of freedom."""
        with torch.no_grad():
            f = self.field; norm = f.raw_quaternion.norm(dim=-1, keepdim=True)
            identity = torch.zeros_like(f.raw_quaternion); identity[:, 0] = 1
            f.raw_quaternion.copy_(torch.where(norm > 1e-8, f.raw_quaternion/norm.clamp_min(1e-8), identity))
            f.log_scales.clamp_(math.log(float(self.cfg.get('minimum_scale_m', .0005))), math.log(float(self.cfg.get('maximum_scale_m', 1.0))))
            f.opacity_logits.clamp_(-12, 12); f.dc.clamp_(-.5/C0, .5/C0)

    def _birth_candidates(self, count, residual):
        if count <= 0 or not len(self.candidate_ids): return np.empty(0, np.int64)
        n = len(self.candidate_ids); window = int(self.densify_cfg.get('candidate_window', 8192))
        ids = (np.arange(min(window, n)) + self.candidate_cursor) % n
        self.candidate_cursor = int((self.candidate_cursor + len(ids)) % n)
        confidence = np.asarray(self.candidates.get('confidence', np.ones(n)), np.float64).reshape(n)
        eligible = np.array([int(self.candidate_ids[i]) not in self.used_candidate_ids for i in ids]) & (confidence[ids] >= self.densify_cfg.get('min_candidate_confidence', .05))
        ids = ids[eligible]; score = confidence[ids].copy()
        maps = residual if isinstance(residual, dict) else self._residual_by_raw
        source = np.asarray(self.candidates.get('source_raw', np.zeros(n)), np.int64)
        if 'source_pixel_xy' in self.candidates:
            uv = np.asarray(self.candidates['source_pixel_xy'])
        elif 'source_pinhole_uv' in self.candidates:
            uv = np.asarray(self.candidates['source_pinhole_uv'])-.5
        else:
            raise ValueError('Candidate residual lookup requires native pinhole source_pixel_xy; Brown source_uv is lineage only')
        for raw in np.unique(source[ids]):
            error = maps.get(int(raw), maps.get(str(int(raw)))) if maps else None
            if isinstance(error, dict): error = error.get('residual', error.get('error'))
            if error is None: continue
            error = error.detach().cpu().numpy() if torch.is_tensor(error) else np.asarray(error)
            error = np.squeeze(error)
            if error.ndim != 2: continue
            selected = np.flatnonzero(source[ids] == raw); xy = np.rint(uv[ids[selected]]).astype(np.int64)
            valid = (xy[:, 0]>=0)&(xy[:, 0]<error.shape[1])&(xy[:, 1]>=0)&(xy[:, 1]<error.shape[0])
            measured = np.zeros(len(selected)); measured[valid] = np.maximum(error[xy[valid, 1], xy[valid, 0]], 0)
            score[selected] *= measured + 1e-12
        chosen = ids[np.argsort(-score, kind='stable')[:count]]
        for i in chosen: self.used_candidate_ids.add(int(self.candidate_ids[i]))
        return chosen

    def densify_and_prune(self, optimizer, logical_step, targets=None, residual=None):
        self.accumulate_render_stats(); step = int(logical_step)
        calendar = step in (16, 32) or (100 <= step <= int(self.densify_cfg.get('until_step', 1500)) and step % int(self.densify_cfg.get('every', 100)) == 0)
        if not calendar: return None
        f = self.field; n = len(f.xyz); device = f.xyz.device; config = self.densify_cfg
        with torch.no_grad():
            opacity_bad = f.opacity[:, 0] < float(config.get('min_opacity', .005))
            scale_bad = f.scales.max(-1).values > float(config.get('max_scale_m', 1.0))
            old_invisible = (f.vis_count == 0) & ((step-f.born_step) >= int(config.get('invisible_prune_age', 300)))
            bad = (opacity_bad | scale_bad | old_invisible) if step >= 32 else torch.zeros(n, dtype=torch.bool, device=device)
            minimum = int(config.get('min_points', 128))
            if int((~bad).sum()) < minimum:
                rescue = torch.argsort(f.opacity[:, 0], descending=True, stable=True)[:min(minimum, n)]; bad[rescue] = False
            capacity = max(0, int(config.get('max_points', 120000)) - n + int(bad.sum()))
            birth_count = min(int(config.get('birth_per_event', 256)), capacity) if step == 16 or step >= 100 else 0
            births = self._birth_candidates(birth_count, residual); capacity -= len(births)
            gradient = f.grad_accum/f.vis_count.clamp_min(1)
            eligible = (~bad) & (f.vis_count > 0) & (gradient > float(config.get('grad_threshold', 1e-9)))
            order = torch.argsort(gradient, descending=True, stable=True)
            order = order[eligible[order]]
            count = min(int(config.get('max_densify_parents', 256)), capacity, len(order)) if step >= 32 else 0
            parents = order[:count]; large = f.scales[parents].max(-1).values > float(config.get('split_scale_m', .025))
            split_parents = parents[large]; clone_parents = parents[~large]
            split_drop = torch.zeros(n, dtype=torch.bool, device=device); split_drop[split_parents] = True
            keep = torch.nonzero(~bad & ~split_drop).reshape(-1); old = f.values(); chunks = []
            if len(births):
                source = {k: v[births] if np.asarray(v).ndim and len(v) == len(self.candidate_ids) else v for k, v in self.candidates.items()}
                chunks.append(initial_tensors(source, device, step))
            for selected, split in [(clone_parents, False), (split_parents, True)]:
                if not len(selected): continue
                index = selected.repeat_interleave(2) if split else selected
                chunk = {k: old[k][index].detach().clone() for k in (*PARAMETERS, *BUFFERS)}
                scales = f.scales[index]; axes = scales.argmax(-1)
                rotation = rotation_matrix(f.rotation[index]); direction = rotation[torch.arange(len(index), device=device), :, axes]
                sign = torch.tensor([-1., 1.], device=device).repeat(len(selected)) if split else torch.ones(len(index), device=device)
                offset = direction*scales.max(-1).values[:, None]*sign[:, None]*(.5 if split else .15)
                chunk['xyz'] += offset; chunk['prior_xyz'] += offset
                if split:
                    chunk['log_scales'] -= math.log(1.6); chunk['prior_log_scales'] -= math.log(1.6)
                    alpha = 1-torch.sqrt((1-f.opacity[index]).clamp_min(1e-7))
                    chunk['opacity_logits'] = torch.logit(alpha.clamp(1e-5, 1-1e-5)); chunk['prior_opacity_logits'] = chunk['opacity_logits'].clone()
                chunk['parent_ids'] = f.row_ids[index].clone(); chunk['born_step'].fill_(step)
                chunks.append(chunk)
            born = sum(len(x['xyz']) for x in chunks); next_id = f.next_row_id
            for chunk in chunks:
                size = len(chunk['xyz']); chunk['row_ids'] = torch.arange(next_id, next_id+size, dtype=torch.int64, device=device); next_id += size
                for key in ['grad_accum', 'vis_count', 'max_screen_radius']: chunk[key].zero_()
            values = {key: torch.cat([old[key][keep].detach(), *(x[key] for x in chunks)], 0) for key in (*PARAMETERS, *BUFFERS)}
            old_named = f.named_new_parameters(); f.replace(values, next_id)
            from full_gaussian_lifecycle import remap_adam_rows
            source_rows = torch.cat([keep, torch.full((born,), -1, dtype=torch.int64, device=device)])
            remap_adam_rows(optimizer, old_named, f.named_new_parameters(), source_rows)
            f.grad_accum.zero_(); f.vis_count.zero_(); f.max_screen_radius.zero_()
        event = dict(logical_step=step, rows_before=n, birth=len(births), clone=len(clone_parents), split=2*len(split_parents),
                     split_parents=len(split_parents), prune=int(bad.sum())+len(split_parents), rows_after=len(f.xyz),
                     low_opacity_pruned=int((bad & opacity_bad).sum()), candidate_cursor=self.candidate_cursor,
                     split_parent_row_ids=old['row_ids'][split_parents].detach().cpu().tolist())
        assert event['rows_after'] == event['rows_before']+event['birth']+event['clone']+event['split']-event['prune']
        self.topology_events.append(copy.deepcopy(event)); self.assert_fixed(); return event

    def capture_model_state(self):
        return dict(schema='scene065.teacher_gaussian.model_state.v1', field=self.field.capture(), candidate_cursor=self.candidate_cursor,
                    used_candidate_ids=torch.as_tensor(sorted(self.used_candidate_ids), dtype=torch.int64), topology_events=copy.deepcopy(self.topology_events),
                    residual_by_raw={k: v.detach().cpu().clone() if torch.is_tensor(v) else np.asarray(v).copy() for k, v in self._residual_by_raw.items()},
                    road_plane_state=dict(origin=self.road_plane_origin.detach().cpu().clone(),normal=self.road_plane_normal.detach().cpu().clone(),sigma_m=self.road_height_sigma_m))

    def restore_model_state(self, state, optimizer=None):
        assert state['schema'] == 'scene065.teacher_gaussian.model_state.v1'
        plane = state['road_plane_state']
        assert torch.equal(self.road_plane_origin.cpu(), plane['origin'].cpu())
        assert torch.equal(self.road_plane_normal.cpu(), plane['normal'].cpu())
        assert self.road_height_sigma_m == plane['sigma_m']
        self._render_records.clear(); self._residual_by_raw.clear(); old = self.field.named_new_parameters(); self.field.restore(state['field'])
        if optimizer is not None:
            from full_gaussian_lifecycle import remap_adam_rows
            remap_adam_rows(optimizer, old, self.field.named_new_parameters(), torch.full((len(self.field.xyz),), -1, dtype=torch.int64, device=self.field.xyz.device))
        self.candidate_cursor = int(state['candidate_cursor']); self.used_candidate_ids = set(state['used_candidate_ids'].tolist())
        self._residual_by_raw = {int(k): v.clone() if torch.is_tensor(v) else np.asarray(v).copy() for k, v in state.get('residual_by_raw', {}).items()}
        self.topology_events = copy.deepcopy(state['topology_events']); self.assert_fixed(); return optimizer

    def assert_fixed(self):
        self.base.assert_fixed()
        if self.retained_fragments is not None:
            assert self._fragment_signature == {k: (id(v), v.data_ptr(), v._version) for k, v in self.retained_fragments.items()}
            assert all(not v.requires_grad for v in self.retained_fragments.values())
        assert list(self.named_new_parameters()) == list(PARAMETERS)
        assert all(p.requires_grad and bool(torch.isfinite(p).all()) for p in self.trainable)
        assert all(len(p) == len(self.field.xyz) for p in self.trainable)

    def export_field(self, path):
        self.assert_fixed(); f = self.field
        xyz = f.xyz.detach().cpu().numpy(); srgb = np.clip(.5+C0*f.dc.detach().cpu().numpy(), 0, 1)
        rgb = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
        coeff = np.zeros((len(xyz), 16, 3), np.float32); coeff[:, 0] = f.dc.detach().cpu().numpy()
        arrays = dict(xyz=xyz, world_xyz=xyz @ self.world_from_patch[:3, :3].T+self.world_from_patch[:3, 3], rgb=rgb, coefficients=coeff,
                      scales=f.scales.detach().cpu().numpy(), rotation=f.rotation.detach().cpu().numpy(), opacity=f.opacity.detach().cpu().numpy()[:, 0],
                      active=np.ones(len(xyz), bool))
        arrays.update({k: getattr(f, k).detach().cpu().numpy() for k in BUFFERS})
        np.savez_compressed(path, **arrays)

Runtime = TeacherGaussianRuntime
CleanRuntimeV2 = TeacherGaussianRuntime

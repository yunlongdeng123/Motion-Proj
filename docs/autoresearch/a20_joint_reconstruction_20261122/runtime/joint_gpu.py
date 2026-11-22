"""Actual native 3DGS forward/backward for one editable object and frozen scene context.

This module is deliberately absent from the CPU preflight import graph. Neither
the previous fusion class nor its preservation/selection rules are imported.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from joint_core import BUILD, FIELDS, bank, read, ref, sha, write


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def tree_equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(cpu(a), cpu(b))
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return isinstance(b, type(a)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def tree_max_difference(a, b):
    if torch.is_tensor(a): return float((cpu(a).double()-cpu(b).double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict): return max([tree_max_difference(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, (tuple, list)): return max([tree_max_difference(x, y) for x, y in zip(a, b)] or [0.])
    return 0. if a == b else float('inf')


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def matrix_quaternion(r):
    """Torch-only wxyz conversion; frozen pose is the only matrix input."""
    # The four-candidate form is stable at 180 degrees and does not detach q.
    qabs = torch.sqrt(torch.clamp(torch.stack((1+r[0,0]+r[1,1]+r[2,2],
          1+r[0,0]-r[1,1]-r[2,2], 1-r[0,0]+r[1,1]-r[2,2],
          1-r[0,0]-r[1,1]+r[2,2])), min=0))
    rows = torch.stack((torch.stack((qabs[0]**2, r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1])),
           torch.stack((r[2,1]-r[1,2], qabs[1]**2, r[1,0]+r[0,1], r[0,2]+r[2,0])),
           torch.stack((r[0,2]-r[2,0], r[1,0]+r[0,1], qabs[2]**2, r[2,1]+r[1,2])),
           torch.stack((r[1,0]-r[0,1], r[2,0]+r[0,2], r[2,1]+r[1,2], qabs[3]**2))))
    return F.normalize((rows/(2*qabs.clamp_min(0.1))[:, None])[qabs.argmax()], dim=0)


def mean_mask(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def huber(value, delta):
    return torch.where(value.abs() < delta, 0.5*value.square()/delta, value.abs()-0.5*delta)


class JointField(torch.nn.Module):
    def __init__(self, a, device='cuda'):
        super().__init__()
        for key in FIELDS:
            self.register_buffer('base_' + key, torch.as_tensor(a[key], device=device).clone())
        self.xyz = torch.nn.Parameter(self.base_xyz.clone())
        self.log_scales = torch.nn.Parameter(torch.zeros_like(self.base_scales))
        identity = torch.zeros_like(self.base_rotation); identity[:, 0] = 1
        self.raw_rotation = torch.nn.Parameter(identity)
        # Activated source opacities, including exact 0 and 1, are unchanged at entry.
        self.opacity = torch.nn.Parameter(self.base_opacity.clone())
        self.dc = torch.nn.Parameter(self.base_coefficients[:, 0].clone())
        for key, value in a.items():
            if key not in FIELDS:
                self.register_buffer('meta_' + key, torch.as_tensor(value, device=device).clone())
        self.register_buffer('active', torch.as_tensor(a.get('active', np.ones(len(a['xyz']), bool)), device=device).bool())
        self.register_buffer('conflict_score', torch.zeros(len(a['xyz']), device=device))
        self.register_buffer('support_contribution', torch.zeros(len(a['xyz']), device=device))
        self.register_buffer('conflict_contribution', torch.zeros(len(a['xyz']), device=device))
        self.metadata_keys = [k for k in a if k not in FIELDS]

    def scales(self): return self.base_scales * self.log_scales.exp()
    def rotation(self): return qmul(F.normalize(self.raw_rotation, dim=-1), self.base_rotation)
    def coefficients(self): return torch.cat((self.dc[:, None], self.base_coefficients[:, 1:]), 1)

    def set_stage(self, stage):
        for key, p in self.named_parameters(): p.requires_grad_(stage in ('geometry', 'joint') if key != 'dc' else stage in ('appearance', 'joint'))

    def export(self):
        result = {k: cpu(getattr(self, k)() if k in ('scales', 'rotation', 'coefficients') else getattr(self, k)).numpy() for k in FIELDS}
        result.update({k: cpu(getattr(self, 'meta_' + k)).numpy() for k in self.metadata_keys})
        result['active'] = cpu(self.active).numpy()
        result['opacity'] = result['opacity'] * result['active']
        return result


class Runtime:
    def __init__(self, args, inp):
        self.args = args; self.i = inp; self.cfg = inp['config']; self.s = inp['sources']
        self.started = time.monotonic(); self.calls = 0; self.backward_calls = 0; self.vjp_calls = 0
        self.records = []; self.retirements = []; self.contribution_history = []; self.source = inp['init']; self.mode = args.mode
        if not torch.cuda.is_available(): raise RuntimeError('A real native CUDA renderer is required')
        torch.set_num_threads(2); self.seed()
        deps = inp['native']['native_renderer']
        legacy = self.s.get(deps['legacy'], 'frozen native scene/camera adapter')
        # Validate source files before native load; no surrogate scene banks.
        for item in deps['native_source_bindings']: self.s.get(item, 'full frozen native ' + item.get('role', 'source'))
        spec = importlib.util.spec_from_file_location('_a20_joint_native_adapter', legacy)
        m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); self.m = m
        if (m.TARGET, m.DELETE, m.TRACK) != (21, 10, 'bzFbWRrSGOgEaXxnuGV0tw'):
            raise ValueError('Unexpected native actor identity')
        m.ROOT = args.output / 'native_loader'; m.FRAMES = BUILD; m.TRAIN = BUILD
        m.ROOT.mkdir(parents=True, exist_ok=True)
        rt = self
        class FrozenSource(torch.nn.Module):
            def __init__(owner, a):
                super().__init__()
                for k in FIELDS: owner.register_buffer(k, m.ten(a[k]))
        m.Vehicle = FrozenSource
        class Experiment(m.Experiment):
            def build_vehicle(e, path):
                import shutil
                shutil.copy2(inp['teacher_path'], path)
            def build_domains(e): return
        self.e = Experiment(self.s.path(deps['road_root']))
        self.e.road_color.requires_grad_(False)
        self.frozen_source_tensors = []
        for role, model in [('static', self.e.scene.static), ('dynamic', self.e.scene.dynamic), ('environment', self.e.scene.environment)]:
            if hasattr(model, 'parameters'):
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(False); self.frozen_source_tensors.append((role+'.'+name, parameter))
            for name, value in vars(model).items():
                if torch.is_tensor(value) and value.is_leaf:
                    value.requires_grad_(False); self.frozen_source_tensors.append((role+'.'+name, value))
        self.actor = JointField(inp['init'])
        self.teacher = FrozenSource(inp['teacher']).cuda()
        self.teacher_unknown = m.ten(inp['init']['teacher_prior_weight'][inp['init']['source_id'] == 1][np.argsort(inp['init']['source_row_id'][inp['init']['source_id'] == 1])])
        self.identity = torch.eye(4, device='cuda'); self.blank = torch.zeros(3, device='cuda')
        self.context = {}; self.context_evidence = []
        self.context_plan = self.load_context_plan()
        write(args.output/'context_plan.json', self.context_plan)
        self.K_preserve = m.ten(self.context_plan.get('preserved_native_context_global_row_ids', []), torch.long)
        side = bank(self.s.get(deps['sidecar'], 'fixed a9 identity guard'))
        guard = m.ten(side['global_row_ids'], torch.long)
        remove_ids = np.asarray(self.context_plan['remove_native_global_row_ids'], np.int64)
        remove = m.ten(remove_ids, torch.long)
        for raw in BUILD:
            packet, owners, ri = self.e.native_packets[raw]
            native_rows, _, _, _ = self.e.scene._rows(raw, (21,), (), False)
            global_ids = torch.nonzero(native_rows['mask']).reshape(-1)
            if not torch.equal(native_rows['means3D'][global_ids], packet['means3D']):
                raise ValueError('Native packed/global identity mismatch')
            owners = owners.clone(); guarded = torch.isin(global_ids, guard)
            if ((owners[guarded] > 0) & (owners[guarded] != 10)).any(): raise ValueError('a9 guard conflicts with retained actor')
            owners[guarded] = 10
            replace = torch.isin(global_ids, remove)
            if ((owners[replace] > 0) & (owners[replace] != 21) & (owners[replace] != 10)).any():
                raise ValueError('Context replacement would remove a retained other actor')
            road_ids = torch.nonzero(ri >= 0).reshape(-1)
            packet = dict(packet, colors_precomp=packet['colors_precomp'].index_copy(0, road_ids, self.e.colors()[ri[road_ids]]))
            keep = ~replace & (owners != 10)
            p = m.subset(packet, keep)
            self.context[raw] = (p, owners[keep], ri[keep], global_ids[keep])
            self.context_evidence.append(dict(raw=raw, native_rows=len(global_ids), removed_replaced_rows=int(replace.sum()),
                     frozen_context_rows=int(keep.sum()), retained_other_actor_rows=int(((owners[keep] > 0) & (owners[keep] != 21)).sum())))
        self.domains = {raw: self.load_domains(raw) for raw in BUILD}
        self.K_reference = self.cache_K_reference()
        self.lidar = self.prepare_lidar()
        self.specs = self.synthetic_specs()
        self.novel_pipe = copy.deepcopy(self.e.scene.pipe); self.novel_pipe.neg_fov = False
        self.synthetic_cameras = {v['key']: self.make_camera(v) for v in self.specs}
        self.teacher_cache = self.cache_teacher()
        self.neighbor, self.cross_duplicate = self.prepare_neighbors()
        self.initial_forward = self.verify_initial_forward()
        self.frozen_fingerprint = self.frozen_state_fingerprint()
        write(args.output / 'entry.json', dict(runtime=ref(__file__), init=ref(inp['init_path']), manifest=ref(inp['manifest_path']),
              source_manifest=ref(inp['native_path']), frozen_teacher=ref(inp['teacher_path']), context=self.context_evidence,
              initial_forward=self.initial_forward, config=self.cfg, read_ledger=self.s.opened,
              heldout_frames_opened=[], one_object_field=True, previous_P0_filter_used=False, PID=os.getpid()))

    def seed(self):
        seed = self.cfg['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)

    def load_context_plan(self):
        path = self.args.context_plan
        if not path:
            for key in ('context_plan', 'native_context_plan', 'context_retirement'):
                if self.i['manifest'].get(key):
                    path = self.s.get(self.i['manifest'][key], 'unified actor native-context handoff'); break
        if path:
            if str(path).endswith('.npz'):
                p = bank(path)
                return dict(remove_native_global_row_ids=p['remove_native_global_row_ids'].tolist(), source=ref(path))
            value = read(path)
            if 'remove_native_global_row_ids' in value: return value
            for key in ('indices', 'index_npz', 'context_index'):
                if key in value:
                    a = bank(self.s.get(value[key], 'native context exact row indices'))
                    return dict(value, remove_native_global_row_ids=a['remove_native_global_row_ids'].tolist())
            raise ValueError('Context plan must enumerate exact remove_native_global_row_ids')
        # Cleanup exclusions alone never authorize deleting native context.
        # Certified pre-existing DELETE retirement remains independently valid.
        a = self.source; clean = a['native_global_row_id'][a['source_id'] == 0]
        if (clean < 0).any(): raise ValueError('Old actor lacks exact native global mapping')
        frozen = bank(self.s.get(self.i['native']['frozen_context_selection'], 'frozen mode-specific retirement identities'))
        retired = frozen['global_row_ids'][frozen['retire_DELETE']]
        if 'preserve_any_K' not in frozen: raise ValueError('Missing complete native K-positive preservation union')
        preserve = frozen['global_row_ids'][frozen['preserve_any_K']]
        if len(preserve) != 1922: raise ValueError('Expected all1922 native K-positive context rows')
        cleanup = self.i['manifest'].get('cleanup')
        cleanup_path = None
        if cleanup:
            item = dict(cleanup)
            if not Path(item['path']).is_absolute(): item['path'] = str(self.i['manifest_path'].parent/item['path'])
            cleanup_path = self.s.get(item, 'clean initialization native-context preservation identities')
            rows = bank(cleanup_path)
            if not np.array_equal(np.sort(clean), np.sort(rows['native_joint_copy_dedup_candidate_global_row_ids'])):
                raise ValueError('Clean native-context dedup IDs differ from unified old source rows')
        clear = np.union1d(clean, retired)
        if np.intersect1d(clear, preserve).size: raise ValueError('Native K-positive context intersects actor clear set')
        return dict(schema='a20.unified_field.native_context_plan.v1', remove_native_global_row_ids=clear.tolist(),
                    preserved_native_context_global_row_ids=preserve.tolist(),
                    cleanup=None if cleanup_path is None else ref(cleanup_path),
                    mapped_clean_actor_rows=len(clean), certified_DELETE_retire_rows=len(retired),
                    preserved_K_positive_union_rows=len(preserve),
                    origin='mapped clean actor union certified fixed DELETE; clear intersection K is empty',
                    cleanup_exclusion_policy='No batch native deletion from actor cleanup. Certified DELETE independently applies to non-K rows; remaining excluded context stays frozen.')

    def load_domains(self, raw):
        from PIL import Image
        f = self.i['fixed']['real_V'][f'{raw:03d}']
        mapping = read(self.s.get(f['mapping'], f'BUILD {raw} mask mapping'))
        width, height = mapping['image_native_size']; x0, y0, x1, y1 = mapping['crop_native_xyxy_half_open']
        d = {}
        for key, field in [('V', 'visible_mask'), ('U', 'unknown_potential'), ('B', 'known_background')]:
            full = np.full((height, width), 255 if key == 'B' else 0, np.uint8)
            full[y0:y1, x0:x1] = np.asarray(Image.open(self.s.get(f[field], f'BUILD{raw} {key}')).convert('L'))
            d[key] = self.m.ten(np.asarray(Image.fromarray(full).resize((1152, 768), Image.Resampling.NEAREST)) > 0, torch.bool)
        labels_image = Image.open(self.s.get(self.i['fixed']['real_K_labels'][f'{raw:03d}'], f'BUILD{raw} fixed categorical labels'))
        source_size = list(labels_image.size)
        if labels_image.size == (480, 320): labels_image = labels_image.resize((1152, 768), Image.Resampling.NEAREST)
        elif labels_image.size != (1152, 768): raise ValueError('Unexpected fixed label resolution')
        labels = np.asarray(labels_image)
        d['K'] = self.m.ten((labels > 0) & (labels != 21) & (labels != 10), torch.bool)
        d['B'] &= ~d['K'] & ~d['U'] & ~d['V']
        d['rgb'] = self.e.targets[raw]['rgb'].detach()
        d['label_sampling'] = dict(source_size=source_size, render_size=[1152, 768], interpolation='NEAREST')
        if not d['V'].any(): raise ValueError('Empty real V')
        return d

    def synthetic_specs(self):
        if not self.args.prior_contract: raise ValueError('--prior-contract is required for the six frozen synthetic views')
        path = self.s.get(self.args.prior_contract, 'six-view frozen teacher camera contract'); c = read(path)
        v = c['views']; specs = []
        for i, az in enumerate(v['synthetic_azimuth_degrees']):
            a = math.radians(az); e = math.radians(v['synthetic_elevation_degrees'])
            target = np.asarray(v['synthetic_target_actor_m'], float)
            eye = target + v['synthetic_distance_m']*np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)])
            forward = target-eye; forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right); down = np.cross(forward, right)
            W = np.eye(4); W[:3, :3] = np.stack([right, down, forward]); W[:3, 3] = -W[:3, :3]@eye
            near, far = 0.1, 100.; P = np.zeros((4, 4)); f = 1/math.tan(math.radians(v['synthetic_fov_degrees'])/2)
            P[0, 0] = P[1, 1] = f; P[2, 2] = far/(far-near); P[2, 3] = -far*near/(far-near); P[3, 2] = 1
            specs.append(dict(key=f'novel_{i:02d}', camera_from_actor=W.tolist(), projection_matrix=P.tolist(),
                 camera_center_actor_m=eye.tolist(), fov_degrees=v['synthetic_fov_degrees'], azimuth_degrees=az))
        if len(specs) != 6: raise ValueError('Expected six fixed synthetic views')
        write(self.args.output / 'synthetic_cameras.json', dict(specs=specs, source=ref(path)))
        return specs

    def make_camera(self, v):
        W = self.m.ten(v['camera_from_actor']).T; P = self.m.ten(v['projection_matrix']).T
        return SimpleNamespace(image_width=512, image_height=512, FoVx=math.radians(v['fov_degrees']),
              FoVy=math.radians(v['fov_degrees']), world_view_transform=W, full_proj_transform=W@P,
              camera_center=self.m.ten(v['camera_center_actor_m']))

    def actor_rows(self, a, pose, cam, teacher=False, row_weights=None):
        xyz = a.xyz; scales = a.scales if teacher else a.scales(); q = a.rotation if teacher else a.rotation()
        coeff = a.coefficients if teacher else a.coefficients(); opacity = a.opacity
        scale = pose[:3, :3].norm(dim=0).mean(); R = pose[:3, :3]/scale
        world = xyz@pose[:3, :3].T + pose[:3, 3]
        local = F.normalize(F.normalize(world-cam.camera_center, dim=1)@R, dim=1)
        color = (torch.einsum('ni,nic->nc', self.m.basis3(local), coeff) + 0.5).clamp_min(0)
        qp = matrix_quaternion(R)
        depth = world@cam.world_view_transform[:3, 2] + cam.world_view_transform[3, 2]
        features = world.new_zeros((len(world), 24)); features[:, 0] = depth; features[:, 1] = 1
        features[:, 2] = 1 if row_weights is None else row_weights
        if teacher:
            features[:, 4] = 1; features[:, 5] = self.teacher_unknown
        else:
            sid = a.meta_source_id
            features[:, 3] = (sid == 0).float(); features[:, 4] = (sid == 1).float()
            features[:, 5] = a.meta_teacher_prior_weight
        mask = depth > (0.00377 if scale < 0.5 else 0.1)
        if not teacher: mask = mask & a.active
        return dict(means3D=world, means2D=torch.zeros_like(world), colors_precomp=color, shs=None,
              features=features, opacities=opacity[:, None], scales=scales*scale,
              rotations=qmul(qp.expand_as(q), q), cov3D_precomp=None, mask=mask)

    def render(self, raw=None, novel=None, teacher=False, actor_only=False, grey=False, row_weights=None, context_only=False):
        if (raw is None) == (novel is None): raise ValueError('Choose exactly one real or synthetic view')
        if raw is not None and raw not in BUILD: raise ValueError('Sealed frame read forbidden')
        cam = self.e.cameras[raw] if raw is not None else self.synthetic_cameras[novel]
        pose = self.e.poses[raw] if raw is not None else self.identity
        pipe = self.e.scene.pipe if raw is not None else self.novel_pipe
        rows = self.actor_rows(self.teacher if teacher else self.actor, pose, cam, teacher, row_weights)
        if grey: rows = dict(rows, colors_precomp=torch.full_like(rows['colors_precomp'], 0.5))
        scene = raw is not None and not actor_only and not teacher
        if context_only and not scene: raise ValueError('Context-only reference requires a real scene view')
        if scene:
            p, owner, road, gids = self.context[raw]
            features = p['features'].clone(); features[:, 2:] = 0
            features[:, 6] = (((owner > 0) & (owner != 21)) | torch.isin(gids, self.K_preserve)).float()
            context = dict(p, features=features)
            rows = context if context_only else self.e.scene._bindings().merge_rows(context, rows)
        out = self.e.scene._bindings().raster(cam, pipe, 0, rows, self.e.scene._background if scene else self.blank)
        self.calls += 1
        rgb = out['rgb'] + (1-out['alpha'])*self.e.env[raw] if scene else out['rgb']
        alpha = out['alpha'].squeeze(); unit_scale = self.e.scale if raw is not None else 1.
        # RGB is always premultiplied. Training depth uses a fixed domain and
        # a safe denominator; evaluation normalizes only stable-alpha values.
        return dict(rgb=rgb, alpha=alpha, depth_sum_m=out['depth_sum'].squeeze()/unit_scale,
                    actor_alpha=out['extra'][0], old_alpha=out['extra'][1], trellis_alpha=out['extra'][2],
                    unknown_alpha=out['extra'][3], K_alpha=out['extra'][4])

    def cache_K_reference(self):
        result = {}; records = []
        with torch.no_grad():
            for raw in BUILD:
                out = self.render(raw=raw, context_only=True)
                pixels = int(self.domains[raw]['K'].sum())
                if pixels == 0: raise ValueError('Empty fixed retained K domain')
                result[raw] = dict(rgb=out['rgb'].detach().clone(), K_alpha=out['K_alpha'].detach().clone(),
                                  topk=min(self.cfg['retained_K']['topk_pixels'], pixels))
                self.save_render(self.args.output/'K_reference'/f'raw{raw:03d}', out)
                records.append(dict(raw=raw, fixed_K_pixels=pixels, topk=result[raw]['topk']))
        write(self.args.output/'K_reference/receipt.json', dict(reference='once-cached context-only native DELETE render',
            actor_clear_K_intersection=0, frozen=True, views=records))
        return result

    def retained_K_losses(self, raw, out):
        fixed = self.domains[raw]['K']; reference = self.K_reference[raw]
        rgb = (out['rgb']-reference['rgb']).abs()[:, fixed]
        alpha = (out['K_alpha']-reference['K_alpha']).abs()[fixed]
        k = reference['topk']; threshold = self.cfg['retained_K']['changed_threshold']
        losses = dict(retained_K_rgb=rgb.mean(0).topk(k).values.mean(),
                      retained_K_alpha=alpha.topk(k).values.mean())
        statistics = dict(fixed_pixels=int(fixed.sum()), topk=k,
            RGB_float_max=float(rgb.detach().max()), RGB_float_mean=float(rgb.detach().mean()),
            RGB_changed_pixels_gt_1over255=int((rgb.detach().amax(0)>threshold).sum()),
            K_alpha_float_max=float(alpha.detach().max()), K_alpha_float_mean=float(alpha.detach().mean()),
            K_alpha_changed_pixels_gt_1over255=int((alpha.detach()>threshold).sum()))
        return losses, statistics

    def cache_teacher(self):
        result = {}; records = []
        with torch.no_grad():
            for key in self.synthetic_cameras:
                out = self.render(novel=key, teacher=True)
                unknown_fraction = out['unknown_alpha']/out['alpha'].clamp_min(1e-8)
                fixed = (out['alpha'] >= self.cfg['teacher']['stable_alpha']) & (unknown_fraction >= self.cfg['teacher']['unknown_attribution_minimum'])
                out['fixed_unknown'] = fixed; out['unknown_fraction'] = unknown_fraction
                support = out['alpha'] > self.cfg['teacher']['background_support_alpha']
                radius = self.cfg['teacher']['background_dilation_pixels']
                dilated = F.max_pool2d(support.float()[None, None], 2*radius+1, 1, radius)[0, 0] > 0
                out['fixed_background'] = ~dilated
                result[key] = {k: v.detach() for k, v in out.items()}
                records.append(dict(key=key, fixed_unknown_pixels=int(fixed.sum()), frozen=True,
                    fixed_background_pixels=int((~dilated).sum()), source_support_pixels=int(support.sum()),
                    applicability='APPLICABLE' if fixed.any() else 'N/A: no high-confidence unknown teacher pixels'))
                self.save_render(self.args.output/'teacher_cache'/key, out)
        write(self.args.output/'teacher_cache/receipt.json', dict(teacher=ref(self.i['teacher_path']), views=records,
              source='pure original full410080 T', attribution='frozen source-row teacher_prior_weight; unknown_alpha/teacher_alpha >= .95',
              fixed_background='outside pure-T alpha>.01 dilated by fixed8px with17x17 maxpool; alpha-only supervision', real_V_precedes_prior=True))
        return result

    def prepare_neighbors(self):
        from scipy.spatial import cKDTree
        xyz = self.source['xyz']; sid = self.source['source_id']; neighbor = np.empty(len(xyz), np.int64)
        for source in (0, 1):
            ids = np.flatnonzero(sid == source)
            _, found = cKDTree(xyz[ids]).query(xyz[ids], k=2, workers=2)
            neighbor[ids] = ids[found[:, 1]]
        old = np.flatnonzero(sid == 0); t = np.flatnonzero(sid == 1)
        dist_t, _ = cKDTree(xyz[old]).query(xyz[t], k=1, workers=2)
        dist_o, _ = cKDTree(xyz[t]).query(xyz[old], k=1, workers=2)
        duplicate = np.zeros(len(xyz), bool); radius = self.cfg['redundancy']['cross_source_radius_m']
        duplicate[t] = dist_t < radius; duplicate[old] = dist_o < radius
        return self.m.ten(neighbor, torch.long), self.m.ten(duplicate, torch.bool)

    def prepare_lidar(self):
        a = self.i['measured']; result = {}; matrices = {}
        eligible_path = self.s.path(self.args.prior_contract).parent/'opaque_anchor_selection.json'
        eligible = read(self.s.get(eligible_path, 'frozen source-eligible235 audit subset'))
        if eligible.get('grouped_returns_sha256') != sha(self.i['measured_path']):
            raise ValueError('Source-eligible return IDs bind another measured bank')
        eligible_ids = np.asarray(eligible['opaque_anchor_ids'], np.int64)
        if len(eligible_ids) != 235 or not (a['RGB_region_group'][eligible_ids] == 2).all():
            raise ValueError('Expected frozen source-eligible235 opaque audit subset')
        self.eligible_mask = np.isin(np.arange(1030), eligible_ids)
        for raw in BUILD:
            ids = np.flatnonzero(a['source_raw'] == raw); cam = self.e.cameras[raw]; pose = self.e.poses[raw]
            world = self.m.ten(a['xyz_canonical_m'][ids])@pose[:3, :3].T + pose[:3, 3]
            uv, z = self.m.project(world, cam)
            view = world@cam.world_view_transform[:3, :3] + cam.world_view_transform[3, :3]
            ij = torch.floor(uv+0.5).long(); valid = (ij[:, 0]>=0)&(ij[:, 0]<1152)&(ij[:, 1]>=0)&(ij[:, 1]<768)&(z>0)
            result[raw] = dict(bank_ids=self.m.ten(ids, torch.long), uv=ij, projection_in_frame=valid,
                truth_range_m=torch.linalg.vector_norm(view, dim=1)/self.e.scale,
                rayfactor=torch.linalg.vector_norm(view, dim=1)/z,
                opaque=self.m.ten(a['RGB_region_group'][ids] == 2, torch.bool),
                group=self.m.ten(a['RGB_region_group'][ids], torch.long))
            matrices[str(raw)] = dict(world_view_transform=cpu(cam.world_view_transform).tolist(),
                full_proj_transform=cpu(cam.full_proj_transform).tolist(), camera_center=cpu(cam.camera_center).tolist(),
                model_from_actor=cpu(pose).tolist(), fx=float(cam.fx), fy=float(cam.fy), cx=float(cam.cx), cy=float(cam.cy),
                model_units_per_meter=float(self.e.scale), image_size=[1152, 768])
        write(self.args.output/'native_camera_matrices.json', dict(cameras=matrices, point_uv='fixed original point source_raw, floor(uv+0.5)',
                 range='norm(native camera view point)/pose_scale', depth='native target-only depth_sum/alpha * point rayfactor'))
        return result

    def verify_initial_forward(self):
        out = self.actor.export(); fields = {}
        for key in FIELDS:
            diff = float(np.max(np.abs(out[key]-self.source[key])))
            fields[key] = dict(max_abs_error=diff, bit_exact=bool(np.array_equal(out[key], self.source[key])))
            if diff > 1e-6: raise ValueError('Initial activated forward mismatch: '+key)
        return fields

    def frozen_state_fingerprint(self):
        import hashlib
        h = hashlib.sha256()
        for raw in BUILD:
            for value in (self.e.poses[raw], self.e.cameras[raw].world_view_transform,
                          self.e.cameras[raw].full_proj_transform): h.update(cpu(value).numpy().tobytes())
        h.update(cpu(self.e.road_color).numpy().tobytes())
        return h.hexdigest()

    def check_frozen(self):
        self.e.scene.assert_persistent_unchanged()
        if self.frozen_state_fingerprint() != self.frozen_fingerprint: raise RuntimeError('Frozen road/camera/trajectory changed')
        gradients = {}
        for name, value in self.frozen_source_tensors:
            gradients[name] = None if value.grad is None else float(value.grad.abs().max())
        if any(value not in (None, 0.) for value in gradients.values()):
            raise RuntimeError('A frozen source-model tensor accumulated gradients')
        return dict(persistent_native_scene_unchanged=True, road_camera_trajectory_unchanged=True,
                    other_actors_frozen=True, source_model_gradient_max=gradients,
                    all_source_model_gradients_None_or_zero=True)

    def save_render(self, path, out):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        for key in ('rgb', 'alpha', 'actor_alpha', 'old_alpha', 'trellis_alpha', 'unknown_alpha', 'fixed_unknown', 'fixed_background'):
            if key in out: self.m.png(path/(key+'.png'), out[key])
        np.savez_compressed(path/'render_float.npz', **{k: cpu(v).numpy() for k, v in out.items()})

    def optimizer(self, stage):
        self.actor.set_stage(stage)
        return torch.optim.Adam([dict(params=[p], lr=self.cfg['learning_rates'][k], name=k)
                   for k, p in self.actor.named_parameters() if p.requires_grad], betas=(0.9, 0.999), eps=1e-8)

    def parameter_losses(self):
        a = self.actor; r = self.cfg['regularizer']; floor = r['confidence_floor']
        g = floor+(1-floor)*a.meta_geometry_confidence; c = floor+(1-floor)*a.meta_appearance_confidence
        dq = F.normalize(a.raw_rotation, dim=-1)
        geometry = ((a.xyz-a.base_xyz)/r['xyz_scale_m']).square().mean(1)
        geometry = geometry + (a.log_scales/r['log_scale_scale']).square().mean(1)
        geometry = geometry + (1-dq[:, 0].square())/r['quaternion_scale']**2
        geometry = geometry + ((a.opacity-a.base_opacity)/r['opacity_scale']).square()
        # Each source contributes one mean so 410080 T rows do not swamp old evidence.
        by_source = lambda value: sum(value[a.meta_source_id == s].mean() for s in (0, 1))/2
        geom = by_source(g*geometry); appearance = by_source(c*((a.dc-a.base_coefficients[:, 0])/r['dc_scale']).square().mean(1))
        delta = a.xyz-a.base_xyz
        smooth = ((delta-delta[self.neighbor])/r['xyz_scale_m']).square().mean()
        duplicate = (a.opacity*a.conflict_score*self.cross_duplicate.float()).sum()/self.cross_duplicate.sum().clamp_min(1)
        return dict(parameter_geometry=geom, parameter_appearance=appearance, delta_smoothness=smooth, duplicate_opacity=duplicate)

    def lidar_losses(self, raw, out):
        q = self.lidar[raw]; valid = q['projection_in_frame']; uv = q['uv'][valid]; ids = torch.nonzero(valid).reshape(-1)
        alpha = out['alpha'][uv[:, 1], uv[:, 0]]; depth_sum = out['depth_sum_m'][uv[:, 1], uv[:, 0]]
        opaque = q['opaque'][valid]; stable = alpha >= self.cfg['lidar']['stable_alpha']
        # The opaque training domain is frozen independently of student alpha.
        # Lowering alpha never removes a difficult fixed return from the loss.
        residual = depth_sum[opaque]/alpha[opaque].clamp_min(self.cfg['lidar']['training_denominator_floor'])*q['rayfactor'][ids[opaque]]-q['truth_range_m'][ids[opaque]]
        depth_loss = huber(residual, self.cfg['lidar']['huber_delta_m']).mean() if opaque.any() else depth_sum.sum()*0
        alpha_loss = (1-alpha[opaque]).square().mean() if opaque.any() else alpha.sum()*0
        return dict(lidar_range=depth_loss, lidar_alpha=alpha_loss), dict(opaque=int(opaque.sum()), training_points=int(opaque.sum()), stable_evaluation_points=int((opaque & stable).sum()))

    def data_losses(self, stage, step):
        raw = BUILD[step % len(BUILD)]; novel = self.specs[step % len(self.specs)]['key']; d = self.domains[raw]
        shared = self.render(raw=raw); isolated = self.render(raw=raw, actor_only=True)
        prior = self.render(novel=novel); teacher = self.teacher_cache[novel]; fixed = teacher['fixed_unknown']
        losses = dict(real_rgb=mean_mask((isolated['rgb']-d['rgb']).abs().mean(0), d['V']),
            silhouette_positive=mean_mask((isolated['alpha']-1).square(), d['V']),
            silhouette_negative=mean_mask(shared['actor_alpha'].square(), d['B']))
        K_losses, K_statistics = self.retained_K_losses(raw, shared); losses.update(K_losses)
        measured, counts = self.lidar_losses(raw, isolated); losses.update(measured)
        losses['teacher_alpha'] = mean_mask((prior['alpha']-teacher['alpha']).abs(), fixed)
        background_alpha = prior['alpha'][teacher['fixed_background']]
        background_k = min(self.cfg['teacher']['background_topk_pixels'], background_alpha.numel())
        losses['teacher_background'] = background_alpha.square().topk(background_k).values.mean() if background_k else prior['alpha'].sum()*0
        if fixed.any():
            error = prior['depth_sum_m'][fixed]/prior['alpha'][fixed].clamp_min(self.cfg['teacher']['training_denominator_floor']) - teacher['depth_sum_m'][fixed]/teacher['alpha'][fixed]
            losses['teacher_depth'] = huber(error, self.cfg['lidar']['huber_delta_m']).mean()
        else: losses['teacher_depth'] = prior['depth_sum_m'].sum()*0
        # Fixed neutral-background, premultiplied RGB; no low-alpha colour normalization.
        rgb_prior = prior['rgb'] + 0.5*(1-prior['alpha'])
        rgb_teacher = teacher['rgb'] + 0.5*(1-teacher['alpha'])
        losses['teacher_rgb'] = mean_mask((rgb_prior-rgb_teacher).abs().mean(0), fixed) if stage != 'geometry' else prior['rgb'].sum()*0
        return losses, dict(raw=raw, novel=novel, lidar=counts, fixed_unknown_pixels=int(fixed.sum()),
                            fixed_teacher_background_pixels=int(teacher['fixed_background'].sum()),
                            shared_V_rgb_L1=float(mean_mask((shared['rgb']-d['rgb']).abs().mean(0), d['V']).detach()),
                            shared_V_alpha95_fraction=float((shared['actor_alpha'][d['V']]>=0.95).float().mean().detach()),
                            teacher_unknown_applicable=bool(fixed.any()), retained_K=K_statistics)

    def gradient_stats(self):
        report = {}
        for key, p in self.actor.named_parameters():
            if not p.requires_grad: continue
            if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError('Disconnected/nonfinite gradient: '+key)
            report[key] = {str(s): float(p.grad[self.actor.meta_source_id == s].abs().max()) for s in (0, 1)}
        return report

    def step(self, stage, step, opt, role='formal', verify_raster=False):
        opt.zero_grad(set_to_none=True)
        data, metadata = self.data_losses(stage, step)
        weights = self.cfg['loss']; raster_loss = sum(weights[k]*v for k, v in data.items())
        raster_grads = {}
        if verify_raster:
            params = [(k, p) for k, p in self.actor.named_parameters() if p.requires_grad]
            values = torch.autograd.grad(raster_loss, [p for k, p in params], retain_graph=True, allow_unused=True)
            for (key, p), g in zip(params, values):
                if g is None or not torch.isfinite(g).all(): raise RuntimeError('Native raster geometry gradient disconnected: '+key)
                raster_grads[key] = {str(s): float(g[self.actor.meta_source_id == s].abs().max()) for s in (0, 1)}
            # Occluded sources and isotropic rotations may correctly have zero
            # gradients. Record those values; require connected finite tensors.
            major = [k for k in ('xyz', 'log_scales', 'opacity') if k in raster_grads]
            if major and not any(max(raster_grads[k].values()) > 0 for k in major):
                raise RuntimeError('All major geometry gradients from the actual raster are zero')
            if not major and not any(max(v.values()) > 0 for v in raster_grads.values()):
                raise RuntimeError('All active appearance gradients from the actual raster are zero')
        reg = self.parameter_losses(); all_losses = dict(data, **reg)
        loss = sum(weights[k]*v for k, v in all_losses.items())
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite actual native training loss')
        loss.backward(); self.backward_calls += 1; grads = self.gradient_stats()
        opt.step()
        with torch.no_grad():
            r = self.cfg['regularizer']; delta = self.actor.xyz-self.actor.base_xyz
            norm = delta.norm(dim=1).clamp_min(1e-12); delta *= (r['safety_xyz_bound_m']/norm).clamp_max(1)[:, None]
            self.actor.xyz.copy_(self.actor.base_xyz+delta)
            self.actor.log_scales.clamp_(math.log(r['safety_scale_ratio_bounds'][0]), math.log(r['safety_scale_ratio_bounds'][1]))
            self.actor.raw_rotation.copy_(F.normalize(self.actor.raw_rotation, dim=-1)); self.actor.opacity.clamp_(0, 1)
        record = dict(stage=stage, step=step+1, role=role, total_loss=float(loss),
            losses={k: float(v.detach()) for k, v in all_losses.items()}, raster_gradient_max=raster_grads,
            gradient_max=grads, elapsed_seconds=time.monotonic()-self.started, **metadata)
        self.records.append(record); write(self.args.output/'progress.json', dict(PID=os.getpid(), native_calls=self.calls,
            backward_calls=self.backward_calls, feature_vjp_calls=self.vjp_calls, latest=record, heldout_frames_opened=[]))
        print(__import__('json').dumps(record), flush=True)
        return record

    def snapshot(self, stage, step, opt, role):
        return dict(schema='a20.unified_field.full_state.v1', init_sha256=sha(self.i['init_path']),
            manifest_sha256=sha(self.i['manifest_path']), teacher_sha256=sha(self.i['teacher_path']),
            runtime_sha256=sha(__file__), config=copy.deepcopy(self.cfg), actor=cpu(self.actor.state_dict()),
            source_metadata_keys=self.actor.metadata_keys, optimizer=cpu(opt.state_dict()),
            phase=stage, step=step, role=role, records=copy.deepcopy(self.records), retirements=copy.deepcopy(self.retirements),
            contribution_history=copy.deepcopy(self.contribution_history),
            frozen_fingerprint=self.frozen_fingerprint, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), sampler=dict(next_real=BUILD[step % 2], next_novel=self.specs[step % 6]['key']))

    def save_state(self, path, stage, step, opt, role):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp')
        torch.save(self.snapshot(stage, step, opt, role), tmp); tmp.replace(path)

    def restore(self, state, opt=None):
        for key, path in [('init_sha256', self.i['init_path']), ('manifest_sha256', self.i['manifest_path']), ('teacher_sha256', self.i['teacher_path'])]:
            if state[key] != sha(path): raise ValueError('Source-bound checkpoint mismatch: '+key)
        if state['config'] != self.cfg: raise ValueError('Checkpoint config mismatch; no silent protocol change')
        self.actor.load_state_dict(state['actor'])
        if opt is not None: opt.load_state_dict(copy.deepcopy(state['optimizer']))
        self.records = copy.deepcopy(state['records']); self.retirements = copy.deepcopy(state['retirements'])
        self.contribution_history = copy.deepcopy(state.get('contribution_history', []))
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if state['frozen_fingerprint'] != self.frozen_fingerprint: raise ValueError('Frozen scene bindings changed')

    def refresh_redundancy(self, step):
        (self.args.output/'contribution').mkdir(parents=True, exist_ok=True)
        support = torch.zeros(len(self.actor.xyz), device='cuda'); conflict = torch.zeros_like(support)
        real_V_raw = torch.zeros_like(support); lidar_bad_raw = torch.zeros_like(support)
        novel_outside_views = torch.zeros_like(support, dtype=torch.int32)
        evidence = {}; view_records = []; r = self.cfg['redundancy']
        before = self.retirement_V_snapshot()

        def contribution(out, weights, domain, key, kind):
            pixels = int(domain.sum())
            if pixels:
                g = torch.autograd.grad(out['actor_alpha'][domain].sum(), weights, retain_graph=True)[0]
                if not torch.isfinite(g).all() or (g < -1e-7).any():
                    raise RuntimeError('Invalid actual native feature contribution: '+key)
                g = g.detach().clamp_min(0); self.vjp_calls += 1
            else: g = torch.zeros_like(support)
            evidence[key] = cpu(g).numpy()
            per_source = {}
            for sid in (0, 1):
                values = g[self.actor.meta_source_id == sid]
                per_source[str(sid)] = dict(raw_sum=float(values.sum()), raw_max=float(values.max()),
                    contributing_rows=int((values>0).sum()), raw_above_001_rows=int((values>0.001).sum()),
                    normalized_sum=float(values.sum()/max(pixels, 1)))
            view_records.append(dict(key=key, role=kind, domain_pixels=pixels, per_source=per_source))
            return g, g/max(pixels, 1)

        for raw in BUILD:
            weights = torch.ones_like(support, requires_grad=True)
            out = self.render(raw=raw, actor_only=True, row_weights=weights)
            d = self.domains[raw]
            g, normalized = contribution(out, weights, d['V'], f'raw{raw:03d}_real_V_support', 'observed support')
            real_V_raw += g; support += normalized
            _, normalized = contribution(out, weights, d['B'], f'raw{raw:03d}_real_B_conflict', 'fixed known-background conflict')
            conflict += normalized
            q = self.lidar[raw]; valid = q['projection_in_frame'] & q['opaque']; uv = q['uv'][valid]
            alpha = out['alpha'][uv[:, 1], uv[:, 0]]
            stable = alpha >= self.cfg['lidar']['stable_alpha']
            residual = torch.zeros_like(alpha)
            residual[stable] = (out['depth_sum_m'][uv[stable, 1], uv[stable, 0]]/alpha[stable]*q['rayfactor'][valid][stable]-q['truth_range_m'][valid][stable]).abs()
            bad = torch.zeros_like(d['V'])
            bad_uv = uv[stable & (residual > self.cfg['redundancy']['conflict_range_m'])]
            if len(bad_uv): bad[bad_uv[:, 1], bad_uv[:, 0]] = True
            # Sparse return residuals are expanded only for contribution attribution.
            bad = F.max_pool2d(bad.float()[None, None], 5, 1, 2)[0, 0] > 0
            g, normalized = contribution(out, weights, bad, f'raw{raw:03d}_lidar_conflict', 'stable opaque-range conflict')
            lidar_bad_raw += g; conflict += normalized
            del out, weights
        for key in self.synthetic_cameras:
            weights = torch.ones_like(support, requires_grad=True)
            out = self.render(novel=key, row_weights=weights); teacher = self.teacher_cache[key]
            _, normalized = contribution(out, weights, teacher['fixed_unknown'], key+'_unknown_support', 'frozen teacher unknown support')
            support += normalized
            g, normalized = contribution(out, weights, teacher['fixed_background'], key+'_outside_conflict', 'frozen teacher outside conflict')
            conflict += normalized
            novel_outside_views += (g > r['novel_outside_raw_contribution_minimum']).to(torch.int32)
            del out, weights
        with torch.no_grad():
            self.actor.support_contribution.copy_(support); self.actor.conflict_contribution.copy_(conflict)
            fraction = conflict/(support+conflict).clamp_min(1e-12)
            self.actor.conflict_score.copy_(fraction)
            common = self.actor.active & (fraction >= r['conflict_fraction_minimum']) & (real_V_raw <= r['support_contribution_max'])
            novel_retire = common & (novel_outside_views >= r['novel_outside_view_minimum'])
            lidar_retire = common & self.cross_duplicate & (lidar_bad_raw > 0) & (self.actor.opacity <= r['inactive_opacity_max'])
            retire = novel_retire | lidar_retire
            reason_bits = lidar_retire.to(torch.uint8) + 2*novel_retire.to(torch.uint8)
            if r['automatic_global_retirement'] and retire.any():
                ids = torch.nonzero(retire).reshape(-1); self.actor.active[ids] = False
                for j in cpu(ids).tolist():
                    self.retirements.append(dict(step=step, unified_row=j, source_id=int(self.source['source_id'][j]),
                        source_row_id=int(self.source['source_row_id'][j]), native_global_row_id=int(self.source['native_global_row_id'][j]),
                        reason_bits=int(reason_bits[j]), novel_outside_views=int(novel_outside_views[j]),
                        reason='novel outside >=2 views' if novel_retire[j] else 'low-opacity LiDAR depth conflict',
                        real_V_support_raw=float(real_V_raw[j]), normalized_support=float(support[j]),
                        normalized_conflict=float(conflict[j]), conflict_fraction=float(fraction[j]), opacity=float(self.actor.opacity[j])))
        after = self.retirement_V_snapshot()
        V_path, V_changes = self.save_retirement_V_change(step, before, after)
        evidence_path = self.args.output/f'contribution/step{step:04d}.npz'
        np.savez_compressed(evidence_path, **evidence, support=cpu(support).numpy(), conflict=cpu(conflict).numpy(),
            real_V_support_raw=cpu(real_V_raw).numpy(), lidar_conflict_raw=cpu(lidar_bad_raw).numpy(),
            conflict_fraction=cpu(fraction).numpy(), novel_outside_view_count=cpu(novel_outside_views).numpy(),
            retirement_reason_bits=cpu(reason_bits).numpy(), cross_source_duplicate=cpu(self.cross_duplicate).numpy(),
            active=cpu(self.actor.active).numpy(), source_id=self.source['source_id'], source_row_id=self.source['source_row_id'],
            native_global_row_id=self.source['native_global_row_id'], unified_row_id=np.arange(len(support)))
        source_counts = {str(s): dict(active=int((self.actor.active & (self.actor.meta_source_id==s)).sum()),
                    retired_this_refresh=int((retire & (self.actor.meta_source_id==s)).sum()),
                    novel_retired=int((novel_retire & (self.actor.meta_source_id==s)).sum()),
                    lidar_retired=int((lidar_retire & (self.actor.meta_source_id==s)).sum())) for s in (0, 1)}
        refresh = dict(step=step, evidence=ref(evidence_path), per_view=view_records, per_source=source_counts,
                       real_V_before_after=ref(V_path), real_V_changes=V_changes)
        self.contribution_history.append(refresh)
        write(self.args.output/f'contribution/step{step:04d}_receipt.json', refresh)
        write(self.args.output/'retirement_ledger.json', dict(rows=self.retirements, refreshes=self.contribution_history,
             per_camera_selection=False, reason_bits={'1':'low-opacity LiDAR branch', '2':'multi-view novel outside branch'},
             rule='domain-mean actual raster contribution; raw real-V <=.001; novel outside raw>.001 in >=2 views; conflict fraction>=.8; teacher unknown support protects'))

    def retirement_V_snapshot(self):
        result = {}
        with torch.no_grad():
            for raw in BUILD:
                domain = self.domains[raw]['V']; views = {}
                for name, kwargs in [('isolated', dict(actor_only=True)), ('shared', {})]:
                    out = self.render(raw=raw, **kwargs)
                    alpha = out['alpha'] if name == 'isolated' else out['actor_alpha']
                    views[name] = dict(rgb=cpu(out['rgb'][:, domain]).numpy(), alpha=cpu(alpha[domain]).numpy())
                result[raw] = views
        return result

    def save_retirement_V_change(self, step, before, after):
        arrays = {}; changes = []
        for raw in BUILD:
            target = cpu(self.domains[raw]['rgb'][:, self.domains[raw]['V']]).numpy()
            for name in ('isolated', 'shared'):
                b = before[raw][name]; a = after[raw][name]
                for field in ('rgb', 'alpha'):
                    arrays[f'raw{raw:03d}_{name}_{field}_before'] = b[field]
                    arrays[f'raw{raw:03d}_{name}_{field}_after'] = a[field]
                rgb_delta = np.abs(a['rgb']-b['rgb']); alpha_delta = np.abs(a['alpha']-b['alpha'])
                changes.append(dict(raw=raw, render=name, V_pixels=len(a['alpha']),
                    RGB_change_max=float(rgb_delta.max()), RGB_change_mean=float(rgb_delta.mean()),
                    RGB_changed_pixels_gt_1over255=int((rgb_delta.max(0)>1/255).sum()),
                    alpha_change_max=float(alpha_delta.max()), alpha_change_mean=float(alpha_delta.mean()),
                    alpha95_fraction_before=float((b['alpha']>=.95).mean()), alpha95_fraction_after=float((a['alpha']>=.95).mean()),
                    RGB_L1_before=float(np.abs(b['rgb']-target).mean()), RGB_L1_after=float(np.abs(a['rgb']-target).mean())))
        path = self.args.output/f'contribution/step{step:04d}_real_V_before_after.npz'
        np.savez_compressed(path, **arrays)
        return path, changes

    def evaluate(self, label, save=True):
        folder = self.args.output/'evaluation'/label; records = []; observed_errors = []; stable_errors = []
        samples = dict(bank_ids=np.arange(1030), source_raw=self.i['measured']['source_raw'], RGB_region_group=self.i['measured']['RGB_region_group'])
        for key, shape in [('render_expected_range_m', (1030,)), ('observed_range_m', (1030,)), ('render_alpha', (1030,)), ('pixel_uv', (1030, 2))]:
            samples[key] = np.full(shape, np.nan, np.float64)
        samples['projection_in_frame'] = np.zeros(1030, bool)
        samples['source_eligible_mask'] = self.eligible_mask
        with torch.no_grad():
            for raw in BUILD:
                shared = self.render(raw=raw); isolated = self.render(raw=raw, actor_only=True); d = self.domains[raw]
                q = self.lidar[raw]; valid = q['projection_in_frame']; ids = q['bank_ids'][valid]; uv = q['uv'][valid]
                _, K_statistics = self.retained_K_losses(raw, shared)
                alpha = isolated['alpha'][uv[:, 1], uv[:, 0]]; stable = alpha >= self.cfg['lidar']['stable_alpha']
                estimate = torch.full_like(alpha, float('nan'))
                estimate[stable] = isolated['depth_sum_m'][uv[stable, 1], uv[stable, 0]]/alpha[stable]*q['rayfactor'][valid][stable]
                opaque = q['opaque'][valid]; errors = (estimate-q['truth_range_m'][valid]).abs(); use = opaque & stable
                stable_errors.extend(cpu(errors[use]).tolist())
                records.append(dict(raw=raw, visible_pixels=int(d['V'].sum()),
                    visible_alpha95_fraction=float((shared['actor_alpha'][d['V']]>=0.95).float().mean()),
                    isolated_visible_alpha95_fraction=float((isolated['alpha'][d['V']]>=0.95).float().mean()),
                    real_rgb_L1=float(mean_mask((isolated['rgb']-d['rgb']).abs().mean(0), d['V'])),
                    isolated_real_rgb_L1=float(mean_mask((isolated['rgb']-d['rgb']).abs().mean(0), d['V'])),
                    shared_real_rgb_L1=float(mean_mask((shared['rgb']-d['rgb']).abs().mean(0), d['V'])),
                    opaque_total=int(opaque.sum()), opaque_stable=int(use.sum()),
                    opaque_stable_mae_m=float(errors[use].mean()) if use.any() else None,
                    retained_K=K_statistics, label_sampling=d['label_sampling']))
                j = cpu(ids).numpy(); samples['render_expected_range_m'][j] = cpu(estimate).numpy()
                samples['observed_range_m'][j] = cpu(q['truth_range_m'][valid]).numpy(); samples['render_alpha'][j] = cpu(alpha).numpy()
                samples['pixel_uv'][j] = cpu(uv).numpy(); samples['projection_in_frame'][j] = True
                if save:
                    self.save_render(folder/f'raw{raw:03d}/shared', shared); self.save_render(folder/f'raw{raw:03d}/actor', isolated)
                    self.save_render(folder/f'raw{raw:03d}/grey', self.render(raw=raw, actor_only=True, grey=True))
            novel_records = []
            for key in self.synthetic_cameras:
                out = self.render(novel=key); teacher = self.teacher_cache[key]; fixed = teacher['fixed_unknown']; background = teacher['fixed_background']
                novel_records.append(dict(key=key, fixed_unknown_pixels=int(fixed.sum()),
                    fixed_background_pixels=int(background.sum()),
                    background_alpha_square_mean=float(mean_mask(out['alpha'].square(), background)),
                    background_alpha_max=float(out['alpha'][background].max()) if background.any() else None,
                    background_alpha_gt_01_pixels=int((out['alpha'][background] > 0.01).sum()),
                    applicability='APPLICABLE' if fixed.any() else 'N/A: no high-confidence unknown teacher pixels',
                    unknown_alpha95_fraction=float((out['alpha'][fixed] >= 0.95).float().mean()) if fixed.any() else None,
                    prior_rgb_L1=float(mean_mask((out['rgb']-teacher['rgb']).abs().mean(0), fixed)) if fixed.any() else None))
                if save:
                    self.save_render(folder/key/'actor', out)
                    self.save_render(folder/key/'grey', self.render(novel=key, grey=True))
        errors = np.asarray(stable_errors); gate = self.cfg['geometry_gate']
        stable = samples['render_alpha'] >= self.cfg['lidar']['stable_alpha']
        eligible_use = samples['source_eligible_mask'] & stable & np.isfinite(samples['render_expected_range_m'])
        eligible_errors = np.abs(samples['render_expected_range_m'][eligible_use]-samples['observed_range_m'][eligible_use])
        conditions = dict(real_visible_coverage=all(r['visible_alpha95_fraction'] >= gate['visible_alpha95_fraction_minimum'] for r in records),
            opaque_all246_stable=sum(r['opaque_stable'] for r in records) >= gate['opaque_stable_count_minimum'],
            opaque_mae=len(errors)>0 and float(errors.mean()) <= gate['opaque_mae_m_maximum'],
            opaque_p90=len(errors)>0 and float(np.quantile(errors, 0.9)) <= gate['opaque_p90_m_maximum'],
            unknown_teacher_coverage=all(r['unknown_alpha95_fraction'] >= gate['novel_unknown_alpha95_fraction_minimum'] for r in novel_records if r['fixed_unknown_pixels'] > 0))
        vehicle_conditions = {k: v for k, v in conditions.items() if k != 'real_visible_coverage'}
        vehicle_conditions['isolated_real_visible_coverage'] = all(r['isolated_visible_alpha95_fraction'] >= gate['visible_alpha95_fraction_minimum'] for r in records)
        K_limit = self.cfg['retained_K']['changed_threshold']
        scene_conditions = dict(shared_real_visible_coverage=conditions['real_visible_coverage'],
            retained_K_RGB_within_1over255=all(r['retained_K']['RGB_float_max'] <= K_limit for r in records),
            retained_K_alpha_within_1over255=all(r['retained_K']['K_alpha_float_max'] <= K_limit for r in records))
        # The original full-scene gate remains strict; actor qualification is a
        # separate finding and never implies scene-handoff qualification.
        conditions.update({k: v for k, v in scene_conditions.items() if k != 'shared_real_visible_coverage'})
        report = dict(schema='a20.unified_field.geometry_evaluation.v1', label=label, real=records, novel=novel_records,
            opaque_stable_count=len(errors), opaque_mae_m=float(errors.mean()) if len(errors) else None,
            opaque_p90_m=float(np.quantile(errors, 0.9)) if len(errors) else None,
            source_eligible235=dict(total=235, stable=int(eligible_use.sum()),
                mae_m=float(eligible_errors.mean()) if len(eligible_errors) else None,
                p90_m=float(np.quantile(eligible_errors, 0.9)) if len(eligible_errors) else None),
            gate=dict(passed=all(conditions.values()), conditions=conditions), heldout_frames_opened=[],
            vehicle_geometry_gate=dict(passed=False, numeric_passed=all(vehicle_conditions.values()),
                 conditions=vehicle_conditions, grey_review='PENDING_ROOT_GREY_REVIEW'),
            scene_handoff_gate=dict(passed=all(scene_conditions.values()), conditions=scene_conditions),
            low_alpha_range_values_omitted=True, glass_used_as_exterior_shell=False, frozen=self.check_frozen())
        folder.mkdir(parents=True, exist_ok=True); np.savez_compressed(folder/'all1030_range.npz', **samples)
        write(folder/'metrics.json', report); return report

    def export_asset(self, label):
        folder = self.args.output/'assets'/label; folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/'actor.npz', **self.actor.export())
        write(folder/'scene_binding.json', dict(schema='a20.unified_field.scene_binding.v1', actor=ref(folder/'actor.npz'),
             source_init=ref(self.i['init_path']), source_manifest=ref(self.i['native_path']), unified_rows=len(self.actor.xyz),
             active_rows=int(self.actor.active.sum()), native_context_plan=self.context_plan, fixed_build_raws=list(BUILD),
             native_camera_matrices=ref(self.args.output/'native_camera_matrices.json'), source_bound=True,
             expected_renderer='actual shared native Gaussian raster; road/other actors/context frozen'))
        return folder/'actor.npz'

    def reload_check(self, asset, label='fresh_reload'):
        current = self.actor; before = {}
        with torch.no_grad():
            for raw in BUILD: before[raw] = {k: v.detach().clone() for k, v in self.render(raw=raw).items()}
        loaded = bank(asset); self.actor = JointField(loaded)
        differences = []
        with torch.no_grad():
            for raw in BUILD:
                out = self.render(raw=raw); diff = {k: float((out[k]-before[raw][k]).abs().max()) for k in out}
                differences.append(dict(raw=raw, maximum=diff)); self.save_render(self.args.output/'evaluation'/label/f'raw{raw:03d}', out)
        self.actor = current
        passed = all(v <= 2e-5 for row in differences for v in row['maximum'].values())
        write(self.args.output/(label+'_proof.json'), dict(passed=passed, asset=ref(asset), rows=differences))
        if not passed: raise RuntimeError('Exported actor fresh reload rendered mismatch')
        return differences

    def smoke(self):
        initial = cpu(self.actor.state_dict()); opt = self.optimizer('geometry'); n = self.cfg['stages']['smoke']
        if n != 10: raise ValueError('Frozen short smoke is exactly10 geometry steps')
        for step in range(n): self.step('geometry', step, opt, role='smoke', verify_raster=step == 0)
        self.refresh_redundancy(n)
        checkpoint = self.args.output/'checkpoints/smoke_step0010.pt'; self.save_state(checkpoint, 'geometry', n, opt, 'smoke')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        first = self.step('geometry', n, opt, role='smoke_replay'); state_first = cpu(self.actor.state_dict()); opt_first = cpu(opt.state_dict())
        self.restore(saved, opt)
        restored_exact = all(torch.equal(cpu(v), saved['actor'][k]) for k, v in self.actor.state_dict().items())
        optimizer_restored_exact = tree_equal(cpu(opt.state_dict()), saved['optimizer'])
        rng_restored_exact = tree_equal(torch.get_rng_state(), saved['torch_rng']) and tree_equal(torch.cuda.get_rng_state_all(), saved['cuda_rng'])
        rng_restored_exact = rng_restored_exact and tree_equal(np.random.get_state(), saved['numpy_rng']) and tree_equal(random.getstate(), saved['python_rng'])
        if not (restored_exact and optimizer_restored_exact and rng_restored_exact):
            raise RuntimeError('Checkpoint actor/Adam/RNG state failed bit-exact restore')
        second = self.step('geometry', n, opt, role='smoke_replay'); state_second = cpu(self.actor.state_dict())
        parameter_differences = {k: float((state_first[k].float()-state_second[k].float()).abs().max()) for k in state_first}
        maximum = max(parameter_differences.values()); loss_difference = abs(first['total_loss']-second['total_loss'])
        optimizer_maximum = tree_max_difference(opt_first, cpu(opt.state_dict()))
        if maximum > 2e-5 or loss_difference > 1e-6 or optimizer_maximum > 2e-5:
            raise RuntimeError('Smoke same-next-step replay mismatch')
        evaluation = self.evaluate('smoke_geometry'); asset = self.export_asset('smoke_geometry'); reload_diff = self.reload_check(asset, 'smoke_reload')
        # Discarded next-stage probe covers material and joint optimizer wiring.
        stage_state = cpu(self.actor.state_dict()); probes = {}
        for stage in ('appearance', 'joint'):
            probe_opt = self.optimizer(stage); probes[stage] = self.step(stage, 0, probe_opt, role='discarded_next_stage_probe', verify_raster=True)
            self.actor.load_state_dict(stage_state)
        proof = dict(schema='a20.unified_field.smoke.v1', passed=True, real_native_cuda=True,
             finite_forward_backward=True, every_source_geometry_gradient_from_raster=True,
             full_source_bound_checkpoint_saved=ref(checkpoint), bit_exact_actor_restore=restored_exact,
             bit_exact_Adam_restore=optimizer_restored_exact, bit_exact_all_rng_restore=rng_restored_exact,
             next_step_parameter_max=maximum, next_step_loss_abs=loss_difference,
             next_step_optimizer_max=optimizer_maximum,
             parameter_differences=parameter_differences, discarded_next_stage_probes=probes,
             evaluation=True, exported_actor_reload=True, actual_geometry_smoke_updates=10,
             extra_replay_updates=2, discarded_next_stage_updates=2, formal_updates=0,
             smoke_contribution_refreshes=len(self.contribution_history), smoke_feature_vjp_calls=self.vjp_calls,
             smoke_retired_rows=len(self.retirements), smoke_contribution_history=self.contribution_history,
             geometry_gate=evaluation['gate'], vehicle_geometry_gate=evaluation['vehicle_geometry_gate'],
             scene_handoff_gate=evaluation['scene_handoff_gate'], heldout_frames_opened=[], frozen=self.check_frozen())
        write(self.args.output/'smoke_proof.json', proof); self.actor.load_state_dict(initial)
        return proof

    def formal(self, stage):
        if stage in ('appearance', 'joint'):
            if not self.args.geometry_gate: raise ValueError('Appearance/joint require frozen geometry evaluation gate')
            gate = read(self.args.geometry_gate)
            if not gate.get('vehicle_geometry_gate', {}).get('passed'):
                raise ValueError('Vehicle geometry gate or grey review failed/pending; stop before appearance/joint')
        opt = self.optimizer(stage); start = 0
        if self.args.resume:
            state = torch.load(self.args.resume, map_location='cpu', weights_only=False)
            if state['phase'] == stage:
                self.restore(state, opt); start = int(state['step'])
            else: self.restore(state)
        elif stage != 'geometry': raise ValueError('Next stage requires a source-bound prior-stage checkpoint')
        if stage == 'geometry' and not self.args.resume: self.seed()
        budget = self.cfg['stages'][stage]
        (self.args.output/'contribution').mkdir(parents=True, exist_ok=True)
        for step in range(start, budget):
            self.step(stage, step, opt)
            if stage in ('geometry', 'joint') and (step+1) % self.cfg['redundancy']['refresh_every'] == 0:
                self.refresh_redundancy(step+1)
            if (step+1) % 10 == 0 or step+1 == budget:
                self.save_state(self.args.output/f'checkpoints/{stage}_step{step+1:04d}.pt', stage, step+1, opt, 'formal')
        metrics = self.evaluate(stage+'_final'); asset = self.export_asset(stage+'_final'); self.reload_check(asset)
        write(self.args.output/'training_records.json', dict(rows=self.records))
        write(self.args.output/'completion.json', dict(stage=stage, steps=budget, completed=True,
            geometry_gate=metrics['gate'], vehicle_geometry_gate=metrics['vehicle_geometry_gate'],
            scene_handoff_gate=metrics['scene_handoff_gate'], scene_qualified=metrics['gate']['passed'] and metrics['scene_handoff_gate']['passed'] and metrics['vehicle_geometry_gate']['passed'],
            can_enter_appearance=metrics['vehicle_geometry_gate']['passed'], actor=ref(asset),
            active_rows=int(self.actor.active.sum()), retired_rows=len(self.retirements), native_calls=self.calls,
            backward_calls=self.backward_calls, feature_vjp_calls=self.vjp_calls, frozen=self.check_frozen(),
            heldout_frames_opened=[], GPU_keep_on=True, elapsed_seconds=time.monotonic()-self.started))


def execute(args, inp):
    if (args.output/'entry.json').exists(): raise FileExistsError('Preserve existing run; choose a new output directory')
    write(args.output/'training_contract.json', inp['config'])
    rt = Runtime(args, inp)
    if args.mode == 'smoke': rt.smoke()
    elif args.mode in ('geometry', 'appearance', 'joint'): rt.formal(args.mode)
    else:
        if args.asset:
            rt.actor = JointField(bank(args.asset))
        elif args.resume:
            rt.restore(torch.load(args.resume, map_location='cpu', weights_only=False))
        else: raise ValueError('Evaluation/reload requires --asset or --resume')
        rt.evaluate('fresh_process')
        if args.asset: rt.reload_check(args.asset, 'fresh_process_reload')
    torch.cuda.synchronize()
    write(args.output/'resource_ledger.json', dict(native_calls=rt.calls, backward_calls=rt.backward_calls,
          feature_vjp_calls=rt.vjp_calls, wall_seconds=time.monotonic()-rt.started,
          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), GPU_keep_on=True, power_operations=0,
          heldout_frames_opened=[]))

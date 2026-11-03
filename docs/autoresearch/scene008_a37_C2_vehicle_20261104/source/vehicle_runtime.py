"""受控车辆读出：冻结PVD与道路，仅替换明确登记的旧静态行。

不使用图像mask路由颜色/深度；所有保留行和新a6始终共同进行原生深度排序。
本模块的候选/诊断开关不等于已确认对象归属，需由独立证据登记后才能正式使用。
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PARENT = ROOT / 'scene008_a37_C2_20261103_r1'
sys.path.insert(0, str(PARENT / 'source'))
import c2_runtime as old
from editor_common import prepare_rasterization, road_kwargs, merge_rows, raster

NAMES = ('static', 'road', 'a37', 'measured', 'generated_upper', 'generated_lower')


class VehicleExperiment:
    def __init__(self, config):
        self.config = config
        self.checkpoint = torch.load(str(PARENT / 'formal/checkpoint001000.pt'), map_location='cpu', weights_only=False)
        self.old = old.Experiment(self.checkpoint['contract']['settings'], Path(self.checkpoint['contract']['actor_asset']))
        assert self.checkpoint['parent_identity'] == self.old.n.before
        self.old.actor.load_state_dict(self.checkpoint['actor'], strict=True)
        self.old.actor.requires_grad_(False)
        self.n = self.old.n
        a = dict(np.load(PARENT / 'inputs/actor_surface_pvd_seed0.npz', allow_pickle=False))
        lower = a['prior_origin'] & (a['canonical_xyz'][:, 2] < a['observed_points'][:, 2].min())
        groups = np.where(a['observed_anchor'], 3, np.where(lower, 5, 4))
        self.groups = old.tensor(groups, torch.long)
        selections = {'all': np.ones(len(groups), bool), 'missing_lower': lower,
                      'generated': a['prior_origin'], 'measured': a['observed_anchor'],
                      'none': np.zeros(len(groups), bool)}
        self.old.actor.active = old.tensor(selections[config.get('actor_subset', 'all')], torch.bool)
        if config.get('colour_source') == 'observed_initial':
            with torch.no_grad():
                self.old.actor.colour_logit.copy_(torch.logit(self.old.actor.initial_colour.clamp(.005, .995)))
        ids = np.empty(0, dtype=np.int64)
        self.rule = None
        if config.get('replacement_file'):
            self.rule = dict(np.load(config['replacement_file'], allow_pickle=False))
            ids = self.rule['ids'].astype(np.int64)
            assert len(np.unique(ids)) == len(ids)
            assert int(ids.min(initial=0)) >= 0 and int(ids.max(initial=0)) < len(self.n.static.get_xyz)
        self.ids = old.tensor(ids, torch.long)
        self.keep = old.tensor(self.rule['keep'] if self.rule is not None and 'keep' in self.rule else np.zeros(len(ids)))
        assert bool(((self.keep >= 0) & (self.keep <= 1)).all())
        self.before = self.n.before
        self.corridor_cache = {}
        self.corridor_stats = {}
        self.road_tree = None
        self.protected_static = None
        if config.get('protected_static_file'):
            pids = np.load(config['protected_static_file'],allow_pickle=False)['ids']
            self.protected_static = torch.zeros(len(self.n.static.get_xyz),device='cuda',dtype=torch.bool)
            self.protected_static[old.tensor(pids,torch.long)] = True

    def corridor_remove(self, raw, means):
        """相机无关局部3D替换体积；保留实测a6邻域及实测道路表面。"""
        if raw not in self.corridor_cache:
            from scipy.spatial import ConvexHull,cKDTree
            import itertools
            b = self.old.visibility.box(raw, 37)
            corners = np.asarray(list(itertools.product((-1.,1.),repeat=3)))
            target = (corners*np.asarray(b['half'])) @ np.asarray(b['axes']).T + np.asarray(b['center'])
            T = np.asarray(self.old.geometry['model_from_canonical'][raw])
            half = np.asarray(self.old.geometry['box_lwh_m'])/2
            measured_min = float(self.old.actor.observed_points[:,0].min())
            endpoint = half[0] + .10 if self.config.get('corridor_replace_whole_a6',False) else measured_min - .10
            rear = np.array([[endpoint,y*half[1],z*half[2]] for y,z in itertools.product((-1.,1.),repeat=2)])
            rear = rear @ T[:3,:3].T + T[:3,3]
            planes = ConvexHull(np.concatenate((target,rear))).equations
            with torch.no_grad():
                p = old.tensor(planes)
                inside = (means @ p[:,:3].T + p[:,3] <= 1e-6).all(1)
                inv = old.tensor(np.linalg.inv(T))
                local = means @ inv[:3,:3].T + inv[:3,3]
                keep_a6 = (local.abs() <= old.tensor(half+np.array([.20,.10,.10]))).all(1)
                if self.config.get('corridor_replace_whole_a6',False):
                    keep_a6 = torch.zeros_like(keep_a6)
                ids = torch.nonzero(inside & ~keep_a6).flatten()
                keep_road = torch.zeros(len(means),device=means.device,dtype=torch.bool)
                if len(ids):
                    if self.road_tree is None:
                        self.road_tree = cKDTree(self.old.base.repair.road.xyz.detach().cpu().numpy())
                    dist = self.road_tree.query(means[ids].detach().cpu().numpy(),workers=1)[0]
                    near = torch.as_tensor(dist < self.config.get('corridor_road_protect_m',.20)*self.old.visibility.scale,device=means.device)
                    keep_road[ids[near]] = True
                remove = inside & ~keep_a6 & ~keep_road
                if self.protected_static is not None:
                    remove &= ~self.protected_static
                self.corridor_cache[raw] = remove.cpu()
                self.corridor_stats[raw] = dict(inside=int(inside.sum()),keep_a6=int((inside&keep_a6).sum()),keep_road=int(keep_road.sum()),removed=int(remove.sum()),
                    corridor_endpoint_canonical_x_m=endpoint,planes_model=planes.tolist())
        return self.corridor_cache[raw].to(means.device)

    @staticmethod
    def tagged(kw, group, marker=None, marker_dim=0):
        depth = kw['features'][:, :1]
        if isinstance(group, int):
            hot = depth.new_zeros((len(depth), len(NAMES)))
            hot[:, group] = 1
        else:
            hot = torch.nn.functional.one_hot(group, len(NAMES)).to(depth.dtype)
        pieces = [kw['features'][:, :2], hot, hot * depth]
        if marker_dim:
            pieces.append(depth.new_zeros((len(depth), marker_dim)) if marker is None else marker)
        kw['features'] = torch.cat(pieces, 1)
        return kw

    def render(self, raw, include_target=True, component_tags=True, static_marker=None):
        e, n = self.old, self.n
        c = e.base.camera(raw)
        if include_target and self.config.get('completion_state','always') == 'delete_only':
            # 明确的编辑条件分支：无编辑沿用原合法Native路径，不能记成重建优化收益。
            out = n.render(c,dynamic=True)
            out['actor_alpha'] = torch.zeros_like(out['alpha']) if 'alpha' in out else out['rgb'].new_zeros((1,out['rgb'].shape[1],out['rgb'].shape[2]))
            return out
        kw = prepare_rasterization(c, n.static, n.args)
        kw['means2D'] = torch.zeros_like(kw['means3D'])
        rid = e.base.repair.ids
        repair_active = (not include_target) or self.config.get('road_policy', 'always') == 'always'
        if repair_active:
            kw['opacities'] = kw['opacities'].index_copy(0, rid,
                kw['opacities'][rid] * e.base.repair.gate_raw.sigmoid()[:, None])
        if self.config.get('legacy_replacement', 'old_c2') == 'old_c2':
            kw['mask'] = kw['mask'] & ~e.native_mask(raw, kw)
            if e.actor_gates:
                aid = e.actor.contributor_ids
                kw['opacities'] = kw['opacities'].index_copy(0, aid,
                    kw['opacities'][aid] * e.actor.gate_raw.sigmoid()[:, None])
        if len(self.ids):
            kw['opacities'] = kw['opacities'].index_copy(0, self.ids, kw['opacities'][self.ids] * self.keep[:, None])
        if not include_target and self.config.get('target_static_volume_margin_lwh_m') is not None:
            b = e.visibility.box(raw, 37)
            center, axes, half = (old.tensor(b[k]) for k in ('center', 'axes', 'half'))
            local = (kw['means3D'] - center) @ axes
            margin = old.tensor(self.config['target_static_volume_margin_lwh_m']) * e.visibility.scale
            remove = (local.abs() <= half + margin).all(1)
            kw['mask'] = kw['mask'] & ~remove
        if not include_target and self.config.get('replacement_corridor',False):
            kw['mask'] = kw['mask'] & ~self.corridor_remove(raw,kw['means3D'])
        channels = kw['shs'].shape[1] if kw['shs'] is not None else 1
        precomputed = kw['colors_precomp'] is not None
        mdim = 0 if static_marker is None else static_marker.shape[1]
        if component_tags:
            self.tagged(kw, 0, static_marker, mdim)
        if include_target:
            dy = prepare_rasterization(c, n.dynamic, n.args, dynamic=True)
            dy['means2D'] = torch.zeros_like(dy['means3D'])
            if component_tags:
                self.tagged(dy, 2, marker_dim=mdim)
            kw = merge_rows(kw, dy)
        road = road_kwargs(c, e.base.repair.road, n.args, channels, precomputed_color=precomputed)
        if e.visibility is not None:
            road = e.visibility.apply(raw, include_target, road, camera=c)
        if component_tags:
            self.tagged(road, 1, marker_dim=mdim)
        if repair_active:
            kw = merge_rows(kw, road)
        actor = e.actor.kwargs(c, e.matrix(raw), n.args, channels, precomputed)
        if component_tags:
            self.tagged(actor, self.groups, marker_dim=mdim)
        kw = merge_rows(kw, actor)
        out = raster(c, n.args, n.static.active_sh_degree, kw, n.bg)
        env = n.env(c.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        out['rgb'] = out['rgb'] + (1 - out['alpha']) * env
        if component_tags:
            out['component_alpha_T'] = out['extra'][:len(NAMES)]
            out['component_depth_sum'] = out['extra'][len(NAMES):2*len(NAMES)]
            out['component_mean_depth_model'] = out['component_depth_sum'] / out['component_alpha_T'].clamp_min(1e-8)
            out['actor_alpha'] = out['component_alpha_T'][3:].sum(0, keepdim=True)
            if mdim:
                out['static_marker'] = out['extra'][2*len(NAMES):]
            assert float((out['component_alpha_T'].sum(0, keepdim=True) - out['alpha']).detach().abs().max()) < 1e-4
        return out

    def unchanged(self):
        return self.old.unchanged()

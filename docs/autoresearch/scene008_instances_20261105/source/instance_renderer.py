"""在同一原生光栅中合成对象、道路和旧行。标签只参与贡献读出。"""

import sys
from pathlib import Path

import torch

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROAD_RUN = CAMPAIGN / 'scene008_a37_only_B10k_road_20261103_r3'
PVD_FILE = CAMPAIGN / 'scene008_a37_C2_20261103_r1/inputs/actor_surface_pvd_seed0.npz'
GROUPS = ('a6', 'a37', 'road', 'legacy')


def tagged(rows, group, marker=None, marker_dim=0):
    depth = rows['features'][:, :1]
    hot = depth.new_zeros((len(depth), len(GROUPS)))
    hot[:, GROUPS.index(group)] = 1
    pieces = [rows['features'], hot, hot * depth]
    if marker_dim:
        pieces.append(depth.new_zeros((len(depth), marker_dim)) if marker is None else marker)
    return dict(rows, features=torch.cat(pieces, 1))


class InstanceRenderer:
    def __init__(self, bank, objects, pvd):
        sys.path.insert(0, str(ROAD_RUN / 'source'))
        import editor_common
        import repair37

        self.runtime = editor_common
        self.base = repair37.Experiment()
        saved = torch.load(ROAD_RUN / 'formal/checkpoint001000.pt', map_location='cpu', weights_only=False)
        if saved['parent_identity'] != self.base.n.before or saved['step'] != 1000:
            raise ValueError('道路检查点与原生 10K 的绑定不匹配')
        self.base.repair.load_state_dict(saved['repair'], strict=True)
        self.base.repair.requires_grad_(False)
        self.bank = bank
        self.objects = objects
        self.pvd = pvd
        self.n = self.base.n
        self.scale = float(self.base.meta['meter_to_model'])
        self.replacement = None
        self.poses = {}

    def object_rows(self, raw, name, camera, channels, precomputed, include_pvd):
        if raw not in self.poses:
            self.poses[raw] = torch.as_tensor(self.bank.frame(raw)['model_from_local'], dtype=torch.float32, device='cuda')
        matrix = self.poses[raw][GROUPS.index(name)]
        rows = self.objects[name].raster_rows(matrix, camera, self.n.args, channels, precomputed)
        if name == 'a6' and include_pvd:
            rows = self.runtime.merge_rows(rows, self.pvd.raster_rows(matrix, camera, self.n.args, channels, precomputed))
        return rows

    def render(self, raw, state='unedited', *, variant='objects', marker=None, check=False):
        if variant not in ('strong_baseline', 'objects', 'replacement', 'pvd', 'native_attribution'):
            raise ValueError(variant)
        if state not in ('unedited', 'delete_a37', 'a6_only', 'a37_only'):
            raise ValueError(state)
        n, rt = self.n, self.runtime
        camera = self.base.camera(raw)
        isolated = state.endswith('_only')
        baseline = variant == 'strong_baseline'
        native = variant == 'native_attribution'
        replace = variant in ('replacement', 'pvd')
        if replace and self.replacement is None:
            raise RuntimeError('尚未建立局部替换账本')
        if baseline and state == 'a6_only':
            return None
        marker_dim = 0 if marker is None else marker.shape[1]
        layers = []
        precomputed = bool(n.args.convert_SHs_python)
        channels = 1 if precomputed else n.static.get_max_sh_channels
        if not isolated:
            static = rt.prepare_rasterization(camera, n.static, n.args)
            static['means2D'] = torch.zeros_like(static['means3D'])
            if replace:
                static = self.replacement.apply(static)
            else:
                ids = self.base.repair.ids
                keep = self.base.repair.gate_raw.sigmoid()
                static['opacities'] = static['opacities'].index_copy(0, ids, static['opacities'][ids] * keep[:, None])
            layers.append(tagged(static, 'legacy', marker, marker_dim))
        # 合成控制保留原 a37；完成接管后的旧 a37 在两种编辑状态均退出。
        retain_old_a37 = (not isolated and not bool(self.replacement.dynamic_retired)) if replace else state == 'unedited'
        if retain_old_a37 or (baseline and state == 'a37_only'):
            dynamic = rt.prepare_rasterization(camera, n.dynamic, n.args, dynamic=True)
            dynamic['means2D'] = torch.zeros_like(dynamic['means3D'])
            if replace:
                dynamic = self.replacement.apply_dynamic(dynamic)
            layers.append(tagged(dynamic, 'a37' if baseline else 'legacy', marker_dim=marker_dim))
        if not isolated:
            road = rt.road_kwargs(camera, self.base.repair.road, n.args, channels, precomputed_color=precomputed)
            layers.append(tagged(road, 'road', marker_dim=marker_dim))
        if not (baseline or native):
            names = (state[:-5],) if isolated else ('a6', 'a37') if state == 'unedited' else ('a6',)
            for name in names:
                rows = self.object_rows(raw, name, camera, channels, precomputed, variant == 'pvd')
                layers.append(tagged(rows, name, marker_dim=marker_dim))
        rows = layers[0]
        for layer in layers[1:]:
            rows = rt.merge_rows(rows, layer)
        out = rt.raster(camera, n.args, n.static.active_sh_degree, rows, n.bg.new_zeros(3) if isolated else n.bg)
        out['rgb_premultiplied'] = out['rgb']
        if not isolated:
            env = n.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
            out['environment_weight'] = 1 - out['alpha']
            out['environment_contribution'] = out['environment_weight'] * env
            out['rgb'] = out['rgb'] + out['environment_contribution']
        n_groups = len(GROUPS)
        out['component_alpha_T'] = out['extra'][:n_groups]
        out['component_depth_sum'] = out['extra'][n_groups:2*n_groups]
        out['marker'] = out['extra'][2*n_groups:]
        if check and float((out['component_alpha_T'].sum(0, keepdim=True) - out['alpha']).detach().abs().max()) > 1e-4:
            raise RuntimeError('同一次光栅的分组贡献和累计透明度不守恒')
        return out

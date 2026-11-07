"""Persistent measured-road extension around a frozen V44 runtime.

No screen-space replacement, depth priority, original parameter write or model fit.
The original V44 runtime is imported from its immutable source directory.
"""
from pathlib import Path
import hashlib
import json
import sys

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
V44 = C/'scene008_v44_rgb_asset_20261107_r1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MeasuredExtension:
    def __init__(self, rt, path):
        import numpy as np
        import torch
        self.path = Path(path)
        self.sha256 = sha(path)
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        required = ('active', 'local_xyz', 'model_xyz', 'normal_world',
                    'gaussian_sigma_m', 'reference_srgb')
        if any(k not in a for k in required):
            raise ValueError('Road asset lacks required provenance/geometry fields')
        boundary = float(rt.road_xy[:, 0].max())
        active = a['active'].astype(bool)
        # A fixed world-region extension: old asset's forward edge, independent
        # of camera/time/edit. No changes to the existing road or its colour field.
        selected = active & (a['local_xyz'][:, 0] > boundary)
        self.ids = np.flatnonzero(selected)
        if not len(self.ids):
            raise ValueError('No measured road exists beyond the frozen road domain')
        cv = lambda x: torch.as_tensor(x, dtype=torch.float32, device=rt.n.bg.device)
        self.xyz = cv(a['model_xyz'][self.ids])
        self.scaling = cv(a['gaussian_sigma_m'][self.ids]*rt.scale)
        n = a['normal_world'][self.ids]@np.asarray(rt.exp.meta['world_to_model'])[:3, :3].T/rt.scale
        n /= np.linalg.norm(n, axis=1)[:, None]
        q = np.c_[1+n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))]
        if np.any(np.linalg.norm(q, axis=1) < 1e-8):
            raise ValueError('Road normals point away from the declared upper hemisphere')
        q /= np.linalg.norm(q, axis=1)[:, None]
        self.rotation = cv(q)
        self._rgb = cv(a['reference_srgb'][self.ids])
        self.dc = ((self._rgb-.5)/.28209479177387814)[:, None]
        self.opacity = cv(np.full((len(self.ids), 1), np.log(.99/.01)))
        self.active = torch.ones(len(self.ids), dtype=torch.bool, device=self.xyz.device)
        for t in (self.xyz, self.scaling, self.rotation, self._rgb, self.opacity):
            if not bool(torch.isfinite(t).all()):
                raise ValueError('Nonfinite extension parameter')
        if bool((self.scaling <= 0).any()) or bool(((self._rgb < 0)|(self._rgb > 1)).any()):
            raise ValueError('Invalid extension scale or sRGB')
        self.record = dict(path=str(path), sha256=self.sha256, rows=len(self.ids),
            selected_ids_sha256=hashlib.sha256(self.ids.tobytes()).hexdigest(),
            selection='active measured-support cells forward of frozen-road max local-x; fixed across camera/time/edit',
            frozen_boundary_local_x_m=boundary, local_min=a['local_xyz'][self.ids].min(0).tolist(),
            local_max=a['local_xyz'][self.ids].max(0).tolist(), peak_opacity=.99,
            geometry='LiDAR-supported surface interpolation; not per-cell new LiDAR returns',
            colour='source-qualified stable real-FRONT RGB proxy; no generator or extrapolated old colour field',
            original_road_rows_unchanged=True)
        self._versions = [t._version for t in (self.xyz, self.scaling, self.rotation, self._rgb, self.opacity)]

    def colors(self):
        return self._rgb

    def output_mask(self):
        return self.active

    def unchanged(self):
        return (sha(self.path) == self.sha256 and self._versions ==
                [t._version for t in (self.xyz, self.scaling, self.rotation, self._rgb, self.opacity)])


class R0Runtime:
    def __init__(self, asset_path, contract=None, retirement=None, material=None):
        import numpy as np
        import torch
        sys.path.insert(0, str(V44/'source'))
        from v44_runtime import V44Runtime
        self.rt = V44Runtime(contract or V44/'contract/input_contract.json')
        self.extension = MeasuredExtension(self.rt, asset_path)
        self.material_colours = None
        self.material_path = None
        self.material_hash = None
        if material:
            self.material_path = Path(material)
            self.material_hash = sha(material)
            with np.load(material, allow_pickle=False) as z:
                if str(z['asset_sha256'].item()) != self.extension.sha256:
                    raise ValueError('Material belongs to a different measured surface')
                if not np.array_equal(z['extension_ids'], self.extension.ids):
                    raise ValueError('Material addresses a different world-cell selection')
                colours = z['colours_srgb'].copy()
            if colours.shape != (len(self.extension.ids), 3) or not np.isfinite(colours).all():
                raise ValueError('Invalid calibrated road material')
            if np.any((colours < 0)|(colours > 1)):
                raise ValueError('Calibrated road material out of sRGB range')
            self.material_colours = torch.as_tensor(colours, dtype=torch.float32, device='cuda')
        self.retirement = None
        self.retirement_hash = None
        if retirement:
            with np.load(retirement, allow_pickle=False) as z:
                ids = z['ids'].astype(np.int64)
                coefficients = z['coefficients'].astype(np.float32)
                if len(np.unique(ids)) != len(ids) or len(ids) != len(coefficients):
                    raise ValueError('Invalid persistent replacement ledger')
                if len(ids) and (ids.min() < 0 or ids.max() >= len(self.rt.n.static.get_xyz)):
                    raise ValueError('Replacement row ID out of bounds')
                if np.any((coefficients < 0)|(coefficients > 1)):
                    raise ValueError('Replacement coefficient out of range')
            self.retirement = (torch.as_tensor(ids, device='cuda'), torch.as_tensor(coefficients, device='cuda'))
            self.retirement_hash = sha(retirement)

    def extension_rows(self, raw, material=False):
        rt = self.rt
        rows = rt.road_kwargs(rt.raw(raw)['cam'], self.extension, rt.pipe,
                              sh_channels=1, precomputed_color=True)
        if material:
            if self.material_colours is None:
                raise ValueError('No persistent calibrated road material was loaded')
            rows['colors_precomp'] = self.material_colours
        rows = rt._tag(rows, 'road')
        # These channels are unused in V44. Group alpha/RGB still attribute all
        # extension primitives to the road, preserving the original conservation.
        rows['features'][:, 18] = 1
        rows['features'][:, 19:22] = rows['colors_precomp']
        return rows

    def scene(self, raw, delete=False, extension=True, retire=False, material=False):
        rt = self.rt
        scene = rt.scene(raw, delete=delete)
        if retire:
            if self.retirement is None:
                raise ValueError('No persistent replacement ledger was provided')
            ids, coefficients = self.retirement
            # V44 packs original static rows first, with stable native row IDs.
            opacity = scene['rows']['opacities'].clone()
            opacity[ids] = opacity[ids]*coefficients[:, None]
            scene['rows'] = dict(scene['rows'], opacities=opacity)
        if extension:
            scene['rows'] = rt.merge(scene['rows'], self.extension_rows(raw, material=material))
        scene['meta'].update(R0_extension_enabled=extension, persistent_retirement_enabled=retire,
            extension_rows=len(self.extension.ids) if extension else 0,
            extension_sha256=self.extension.sha256, retirement_sha256=self.retirement_hash if retire else None,
            calibrated_road_material_sha256=self.material_hash if material else None)
        return scene

    def render(self, raw, delete=False, extension=True, retire=False, solo=False, material=False):
        rt = self.rt
        if solo:
            data = rt.raw(raw)
            scene = dict(rows=self.extension_rows(raw, material=material), cam=data['cam'], env=data['env']*0,
                         bg=rt.n.bg*0, meta=dict(raw=raw, road_extension_solo=True))
        else:
            scene = self.scene(raw, delete, extension, retire, material=material)
        out = rt.render(raw, scene)
        out['extension_alpha_T'] = out['extra'][16]
        out['extension_rgb_T'] = out['extra'][17:20]
        return out

    def receipt(self):
        assert self.extension.unchanged()
        if self.material_path:
            assert sha(self.material_path) == self.material_hash
        return dict(V44=self.rt.receipt(), extension=self.extension.record,
                    retirement_sha256=self.retirement_hash, material_sha256=self.material_hash, optimizer_updates=0,
                    original_model_updates=0, image_compositing=False,
                    separate_foreground_priority=False, shared_sort=True)

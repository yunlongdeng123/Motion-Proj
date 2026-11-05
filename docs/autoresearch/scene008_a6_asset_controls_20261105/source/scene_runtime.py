"""Independent native10K + road1K scene with a complete replacement a6 asset.

This wrapper never imports the clipping/retirement experiment chain. It uses
the original native rasterizer, freezes the complete background/a37 state, and
physically excludes every native-a6 row from replacement scene inputs.
"""
from pathlib import Path
import hashlib
import os
import sys

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import torch

RUN = Path(__file__).resolve().parent.parent
ROAD = RUN.parent / 'scene008_native_road1k_20261105_r1'
FINE = RUN.parent / 'scene008_ray_road_colour_fine_20261105'
OBS = RUN.parent / 'scene008_instances_20261105_r1/observations/frames'
sys.path.insert(0, str(ROAD / 'source'))
from repair37 import Experiment
from editor_common import prepare_rasterization, road_kwargs, merge_rows, raster

GROUPS = ('asset', 'a37', 'road', 'legacy_static')
RAW_IDS = tuple(range(89, 138))
DC = .28209479177387814


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_rows(rows, keep):
    return {key: None if value is None else value[keep].contiguous()
            for key, value in rows.items()}


def tagged(rows, group=None, owners=None):
    """24-feature ABI: depth, one, four group flags, then eighteen zeros."""
    source = rows['features']
    feature = source.new_zeros((len(source), 24))
    feature[:, :2] = source[:, :2]
    if owners is not None:
        feature[owners, 2] = 1
        feature[~owners, 3] = 1
    elif group is not None:
        feature[:, 2 + GROUPS.index(group)] = 1
    return dict(rows, features=feature)


def world_basis(xy, knots):
    """Pure copy of the saved 11x9 nonnegative bilinear world-local field."""
    low, high = np.asarray(knots['low_m']), np.asarray(knots['high_m'])
    grid = np.asarray(knots['grid'], dtype=np.int64)
    if tuple(grid) != (11, 9) or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Expected the saved 11x9 road field and xy[N,2]')
    scaled = np.clip((xy-low)/np.maximum(high-low, 1e-8)*(grid-1), 0, grid-1)
    index = np.minimum(scaled.astype(np.int64), grid-2)
    fraction = scaled-index
    result = np.zeros((len(xy), int(grid.prod())), np.float32)
    for ox in (0, 1):
        for oy in (0, 1):
            weight = ((fraction[:, 0] if ox else 1-fraction[:, 0]) *
                      (fraction[:, 1] if oy else 1-fraction[:, 1]))
            result[np.arange(len(xy)), (index[:, 0]+ox)*grid[1]+index[:, 1]+oy] = weight
    return result


class Runtime:
    """Own an Experiment through self.exp; do not override its render API.

    raw() exposes GPU tensors with the original observation shapes, plus the
    untouched NumPy observation dictionary under ['obs']. scene() returns a
    dictionary containing rows/cam/env/bg/sameasset/meta. Asset objects need
    rows(matrix, camera, pipe, channels, precomputed=False).
    """

    def __init__(self, fine_road=True):
        import json
        self.exp = Experiment()
        self.n = self.exp.n
        self.pipe = self.n.args
        self.scale = float(self.exp.meta['meter_to_model'])
        self.road_checkpoint = ROAD / 'formal/checkpoint001000.pt'
        checkpoint = torch.load(self.road_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['step'] != 1000 or checkpoint['parent_identity'] != self.n.before:
            raise ValueError('Road1K checkpoint/native10K identity mismatch')
        self.exp.repair.load_state_dict(checkpoint['repair'], strict=True)
        self.exp.repair.requires_grad_(False)
        self.n.env.requires_grad_(False)
        self._versions = self._tensor_versions()
        self.parent_identity = dict(self.n.before)
        self.repair_state_sha256 = self.repair_hash()
        self._data = {}
        self.road_colors = self.exp.repair.road.colors().detach().clone()
        self.meta = dict(native_parent=self.n.before, road_checkpoint=str(self.road_checkpoint),
                         road_checkpoint_sha256=sha256(self.road_checkpoint), groups=GROUPS,
                         repair_state_sha256=self.repair_state_sha256,
                         native_actor_rows=dict(self.n.rows), fine_road=bool(fine_road),
                         replacement_native_a6_rows=0, renderer='original editor_common.raster')
        if fine_road:
            path = FINE / 'coefficients.npz'
            with np.load(path, allow_pickle=False) as saved:
                baseline = saved['colour_baseline'].copy()
                coefficients = saved['world'].copy()
            current = self.road_colors.cpu().numpy()
            if not np.array_equal(baseline, current) or coefficients.shape != (99, 3):
                raise ValueError('Fine-road coefficients do not bind to the loaded road colors')
            knots = json.loads((FINE / 'design.json').read_text())['knots']
            matrix = np.asarray(self.exp.meta['world_to_model']) @ np.asarray(self.exp.meta['coordinate_anchor'])
            inverse = np.linalg.inv(matrix)
            xyz = self.exp.repair.road.xyz.detach().cpu().numpy()
            local = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            np.testing.assert_allclose(local[:, :2].min(0), knots['low_m'], atol=1e-6, rtol=0)
            np.testing.assert_allclose(local[:, :2].max(0), knots['high_m'], atol=1e-6, rtol=0)
            basis = self.road_colors.new_tensor(world_basis(local[:, :2], knots))
            assert torch.allclose(basis.sum(1), torch.ones_like(basis[:, 0]), atol=2e-6, rtol=0)
            self.road_colors = (self.road_colors + basis @ basis.new_tensor(coefficients)).clamp(0, 1)
            self.meta.update(fine_road_coefficients=str(path), fine_road_sha256=sha256(path),
                             road_knots=knots, road_rows=len(xyz))
        self.road_colors_sha256 = self._color_hash()
        self._road_color_version = (id(self.road_colors), self.road_colors._version)
        self.meta['rendered_road_colors_sha256'] = self.road_colors_sha256
        self.assert_unchanged()

    def _tensor_versions(self):
        versions = {}
        for name, model in (('static', self.n.static), ('dynamic', self.n.dynamic)):
            for key, value in vars(model).items():
                if torch.is_tensor(value):
                    versions[(name, key)] = (id(value), value._version)
        for key, value in self.exp.repair.state_dict(keep_vars=True).items():
            versions[('repair', key)] = (id(value), value._version)
        for key, value in self.n.env.named_parameters():
            versions[('environment_parameter', key)] = (id(value), value._version)
        for key, value in self.n.env.named_buffers():
            versions[('environment_buffer', key)] = (id(value), value._version)
        return versions

    def assert_unchanged(self):
        assert self.n.unchanged(), 'Native checkpoint files changed'
        assert self.n.before == self.parent_identity, 'Native baseline identity changed'
        assert self._tensor_versions() == self._versions, 'Frozen native/road tensors changed'
        assert (id(self.road_colors), self.road_colors._version) == self._road_color_version

    def _color_hash(self):
        value = self.road_colors.detach().cpu().contiguous().numpy()
        return hashlib.sha256(value.tobytes()).hexdigest()

    def repair_hash(self):
        """Content hash of the complete frozen road/gate state, for endpoint QA."""
        digest = hashlib.sha256()
        for key, value in sorted(self.exp.repair.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode('utf-8'))
            digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def modelunchanged(self):
        """Check native identity/versions and the exact loaded repair contents."""
        return (self.n.unchanged() and self.n.before == self.parent_identity
                and self._tensor_versions() == self._versions
                and self.repair_hash() == self.repair_state_sha256
                and self._color_hash() == self.road_colors_sha256)

    def raw(self, raw):
        if raw not in RAW_IDS:
            raise KeyError(raw)
        if raw not in self._data:
            with np.load(OBS / f'{raw:03d}.npz', allow_pickle=False) as saved:
                original = {key: saved[key].copy() for key in saved.files}
            camera = self.exp.camera(raw)
            assert abs(float(original['model_time'])-camera.timestamp) < 1e-6
            np.testing.assert_allclose(original['model_to_camera'].T,
                                       camera.world_view_transform.cpu().numpy(), atol=1e-6, rtol=0)
            device = self.n.bg.device
            data = {key: torch.as_tensor(value, device=device) for key, value in original.items()
                    if value.dtype.kind in 'biuf'}
            data['rgb'] = data['rgb'].float().permute(2, 0, 1).contiguous()
            if float(data['rgb'].max()) > 1:
                data['rgb'] = data['rgb']/255.
            data.update(raw=raw, cam=camera, pose=data['model_from_local'][0].float(), obs=original,
                        a6_visible=data['visible'][0], a37_visible=data['visible'][1],
                        a6_depth_valid=data['depth_valid'][0], depth=data['depth_m'][0])
            with torch.no_grad():
                directions = camera.get_world_directions(False).permute(1, 2, 0)
                data['env'] = self.n.env(directions).permute(2, 0, 1).detach()
            self._data[raw] = data
        return self._data[raw]

    def _static(self, camera):
        rows = prepare_rasterization(camera, self.n.static, self.pipe)
        rows['means2D'] = torch.zeros_like(rows['means3D'])
        ids = self.exp.repair.ids
        rows['opacities'] = rows['opacities'].index_copy(
            0, ids, rows['opacities'][ids]*self.exp.repair.gate_raw.sigmoid()[:, None])
        return tagged(rows, 'legacy_static')

    def _road(self, camera, channels, precomputed):
        rows = road_kwargs(camera, self.exp.repair.road, self.pipe, channels,
                           precomputed_color=precomputed)
        if precomputed:
            rows['colors_precomp'] = self.road_colors
        else:
            rows['shs'] = rows['shs'].clone()
            rows['shs'][:, 0] = (self.road_colors-.5)/DC
        return tagged(rows, 'road')

    def scene(self, raw, asset=None, delete=False, baseline=False):
        self.assert_unchanged()
        data = self.raw(raw)
        static = self._static(data['cam'])
        dynamic = self.n.actor_kwargs(data['cam'], delete_a37=False)
        channels = static['shs'].shape[1] if static['shs'] is not None else 1
        precomputed = static['colors_precomp'] is not None
        if baseline:
            # Preserve original packed row order and the parent's deletion mask.
            dynamic = dict(dynamic, mask=dynamic['mask'] & (self.n.keep_a6 if delete else True))
            rows = merge_rows(static, tagged(dynamic, owners=self.n.keep_a6))
            asset_rows, native_a6 = None, int(self.n.keep_a6.sum())
            a37_rows = int((~self.n.keep_a6).sum())
        else:
            if asset is None:
                raise ValueError('A complete asset is required for replacement scenes')
            # Physical row selection removes ALL native-a6 primitives before raster input.
            a37 = select_rows(dynamic, ~self.n.keep_a6)
            if delete:
                a37 = select_rows(a37, torch.zeros_like(a37['mask']))
            asset_rows = asset.rows(data['pose'], data['cam'], self.pipe, channels,
                                    precomputed=precomputed)
            if len(asset_rows['means3D']) != len(asset.xyz) or len(asset.xyz)==0:
                raise ValueError('Replacement must use every row of one complete asset')
            rows = merge_rows(merge_rows(static, tagged(asset_rows, 'asset')), tagged(a37, 'a37'))
            native_a6, a37_rows = 0, len(a37['mask'])
        rows = merge_rows(rows, self._road(data['cam'], channels, precomputed))
        assert bool((rows['features'][:, 6:] == 0).all()), 'Unexpected old clip metadata'
        meta = dict(raw=raw, delete_a37=bool(delete), native_baseline=bool(baseline),
                    native_a6_rows_input=native_a6, native_a37_packed_rows=a37_rows,
                    asset_rows=0 if asset_rows is None else len(asset_rows['means3D']),
                    groups=GROUPS, same_material_both_states=True)
        return dict(rows=rows, cam=data['cam'], env=data['env'], bg=self.n.bg,
                    sameasset=None if baseline else asset, meta=meta)

    def render_rows(self, raw, scene):
        data = self.raw(raw)
        if 'rows' in scene:
            rows, background, environment = scene['rows'], scene['bg'], scene['env']
        else:
            rows, background, environment = scene, self.n.bg, data['env']
        out = raster(data['cam'], self.pipe, self.n.static.active_sh_degree, rows, background)
        out['rgb_premultiplied'] = out['rgb']
        if environment is not None:
            out['rgb'] = out['rgb']+(1-out['alpha'])*environment
        out['component_alpha_T'] = out['extra'][:4]
        assert float((out['component_alpha_T'].sum(0)-out['alpha'][0]).detach().abs().max()) < 2e-5
        if 'meta' in scene:
            out['meta'] = scene['meta']
        return out

    def render_scene(self, raw, asset=None, delete=False, baseline=False):
        return self.render_rows(raw, self.scene(raw, asset, delete=delete, baseline=baseline))

    def native_reference(self, raw):
        """Native a6 alone, black background, no static/a37/road/PVD/environment.

        Restrict appearance-reference use to observed_visible. Predictions
        outside that real observation mask are not unknown-surface RGB truth.
        """
        data = self.raw(raw)
        dynamic = self.n.actor_kwargs(data['cam'], delete_a37=False)
        a6 = tagged(select_rows(dynamic, self.n.keep_a6), 'asset')
        out = raster(data['cam'], self.pipe, self.n.static.active_sh_degree, a6,
                     self.n.bg.new_zeros(3))
        out['rgb_premultiplied'] = out['rgb']
        out['rgb_unpremultiplied'] = out['rgb']/out['alpha'].clamp_min(1e-5)
        out['component_alpha_T'] = out['extra'][:4]
        out['observed_visible'] = data['a6_visible']
        out['meta'] = dict(raw=raw, native_reference=True, native_a6_rows_input=len(a6['mask']),
                           reference_scope='Real observed-visible a6 domain only')
        return out

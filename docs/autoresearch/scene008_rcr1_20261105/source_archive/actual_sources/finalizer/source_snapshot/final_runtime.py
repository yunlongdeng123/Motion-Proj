"""Explicit new-native/new-road binding, complete fixed M2, one depth order.

Only import this module in an execution worker. It does not import the legacy
ray-switch chain or any old gate/colour file. Keep/delete changes only a37.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ASSET_SOURCE = CAMPAIGN/'scene008_a6_asset_controls_20261105/source'
ASSET = ASSET_SOURCE.parent/'m2/complete_a6.npz'
OBS = CAMPAIGN/'scene008_instances_20261105_r1/observations/frames'
FIXED = CAMPAIGN/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
RAWS = tuple(range(89, 138))
GROUPS = ('asset', 'a37', 'road', 'native_static')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensors_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def select(rows, keep):
    return {key: None if value is None else value[keep].contiguous() for key, value in rows.items()}


def tagged(rows, group):
    feature = rows['features'].new_zeros((len(rows['features']), 24))
    feature[:, :2] = rows['features'][:, :2]
    feature[:, 2+GROUPS.index(group)] = 1
    return dict(rows, features=feature)


class FixedM2RoadRuntime:
    def __init__(self, native_parent, native_iteration, road_run, asset=ASSET,
                 asset_source=ASSET_SOURCE, observations=OBS, fixed_reference=FIXED,
                 road_checkpoint=None, purpose='final'):
        if purpose not in ('final', 'preflight'):
            raise ValueError('Unknown evaluation purpose')
        if purpose == 'final' and int(native_iteration) != 10000:
            raise ValueError('Final acceptance requires native10K; use purpose=preflight for earlier parents')
        self.purpose = purpose
        self.parent = Path(native_parent).resolve()
        self.iteration = int(native_iteration)
        self.road_run = Path(road_run).resolve()
        self.asset_path = Path(asset).resolve()
        self.asset_source = Path(asset_source).resolve()
        self.observations = Path(observations).resolve()
        self.fixed_reference = Path(fixed_reference).resolve()
        binding_path = self.road_run/'replay_binding.json'
        binding = read(binding_path)  # An old hard-bound road has no such file.
        if Path(binding['native_parent']).resolve() != self.parent or int(binding['native_iteration']) != self.iteration:
            raise ValueError('Requested native is not the parent used by this road replay')
        metadata_paths = [self.parent/name for name in
                          ('effective_config.json', 'cameras.json', 'actual_scene_binding.json')]
        for path in metadata_paths:
            if binding['parent_metadata_sha256'].get(str(path)) != sha(path):
                raise ValueError(f'Frozen replay parent metadata differs: {path}')
        for name in ('editor_common', 'native_binding', 'repair37', 'r4_runtime', 'road_runtime'):
            if name in sys.modules:
                raise RuntimeError(f'Use an isolated worker; scene module already imported: {name}')
        sys.path[:0] = [str(self.road_run/'source'), str(self.asset_source)]
        import repair37
        import native_binding
        from editor_common import prepare_rasterization, road_kwargs, merge_rows, raster
        from utils.sh_utils import eval_sh
        from sh_asset import SHAsset
        for module in (repair37, native_binding):
            if Path(module.__file__).resolve().parent != self.road_run/'source':
                raise ValueError('Imported a different road source tree')
        if Path(native_binding.PARENT).resolve() != self.parent:
            raise ValueError('Road module resolved a different native parent')
        if not torch.cuda.is_available():
            raise RuntimeError('Original CUDA rasterizer unavailable')
        torch.set_num_threads(1)
        self.exp = repair37.Experiment()
        self.n = self.exp.n
        self.pipe = self.n.args
        self.prepare = prepare_rasterization
        self.road_kwargs = road_kwargs
        self.merge = merge_rows
        self.raster = raster
        self.eval_sh = eval_sh
        expected = {key: tuple(value) for key, value in binding['parent_identity'].items()}
        if self.n.before != expected or self.n.checkpoint_iteration != self.iteration:
            raise ValueError('Loaded native identity is not the frozen replay parent')
        self.road_checkpoint = Path(road_checkpoint).resolve() if road_checkpoint else self.road_run/'formal/checkpoint001000.pt'
        if self.road_checkpoint.parent.parent != self.road_run:
            raise ValueError('Road checkpoint must belong to the explicit fresh road run')
        state = torch.load(str(self.road_checkpoint), map_location='cpu', weights_only=False)
        allowed_steps = (1000,) if purpose == 'final' else (32, 1000)
        if state['step'] not in allowed_steps or state['parent_identity'] != self.n.before:
            raise ValueError('Road checkpoint step/native identity mismatch')
        self.road_iteration = int(state['step'])
        contract = state['contract']
        if Path(contract['native_parent']).resolve() != self.parent or int(contract['native_iteration']) != self.iteration:
            raise ValueError('Road checkpoint contract belongs to another native')
        if not contract['only_static_opacity_gate_trainable'] or not contract['road_colour_geometry_opacity_fixed']:
            raise ValueError('Unexpected road method contract')
        contributors = self.road_run/'inputs/contributors.npz'
        with np.load(contributors, allow_pickle=False) as data:
            selected_ids = data['ids'].copy()
        checkpoint_ids = state['repair']['ids'].numpy()
        if not np.array_equal(selected_ids, checkpoint_ids):
            raise ValueError('Checkpoint gates do not address this replay contributor list')
        if (selected_ids.ndim != 1 or len(np.unique(selected_ids)) != len(selected_ids)
                or (len(selected_ids) and (selected_ids.min() < 0 or selected_ids.max() >= len(self.n.static.get_xyz)))):
            raise ValueError('Invalid new-native contributor indices')
        self.exp.repair.load_state_dict(state['repair'], strict=True)
        self.exp.repair.requires_grad_(False)
        self.n.env.requires_grad_(False)
        self.asset = SHAsset.from_geometry(self.asset_path).cuda().requires_grad_(False)
        if self.asset.degree != 3:
            raise ValueError('M2 must always use its complete SH3 material')
        self.asset_before = {key: value.detach().cpu().clone() for key, value in
                             (*self.asset.named_parameters(), *self.asset.named_buffers())}
        if len(self.asset_before) != 27:
            raise ValueError(f'Unexpected M2 tensor schema: {len(self.asset_before)} instead of27')
        self.baseline_colours = self.exp.repair.road.colors().detach().clone()
        self.road_colours = self.baseline_colours  # Pure output colour table; never written into the road model.
        self.repair_hash = tensors_hash(self.exp.repair.state_dict())
        self.baseline_hash = tensors_hash({'colours': self.baseline_colours})
        self.asset_file_hash = sha(self.asset_path)
        self.parent_identity = dict(self.n.before)
        self._versions = self.tensor_versions()
        self._colour_versions = (id(self.baseline_colours), self.baseline_colours._version)
        paths = [binding_path, self.road_checkpoint, contributors, self.asset_path, self.fixed_reference,
                 self.road_run/'inputs/road_reference.npz', *metadata_paths]
        paths += [self.road_run/'source'/name for name in
                  ('editor_common.py', 'native_binding.py', 'repair37.py', 'road_runtime.py', 'r4_runtime.py')]
        paths += [self.asset_source/'sh_asset.py', Path(self.raster.__code__.co_filename)]
        self.file_identities = {str(path): identity(path) for path in paths}
        self.content_hashes = {str(path): sha(path) for path in paths}
        with np.load(self.fixed_reference, allow_pickle=False) as fixed:
            if fixed['layer_names'].tolist()[3] != 'ground_beneath_a37_proxy':
                raise ValueError('Original delete road supervision layer changed')
            np.testing.assert_array_equal(fixed['raw_ids'], np.arange(89, 138))
            self.fixed_ground = fixed['depth_m'][:, 3] > 0
            self.fixed_cameras = fixed['model_to_camera'].copy()
            self.fixed_times = fixed['model_time'].copy()
        matrix = np.asarray(self.exp.meta['world_to_model']) @ np.asarray(self.exp.meta['coordinate_anchor'])
        inverse = np.linalg.inv(matrix)
        xyz = self.exp.repair.road.xyz.detach().cpu().numpy()
        self.road_xy = (xyz @ inverse[:3, :3].T+inverse[:3, 3])[:, :2]
        self._data = {}
        self.fine_binding = None
        self.binding = dict(purpose=purpose, native_parent=str(self.parent), native_iteration=self.iteration,
                            parent_identity=self.parent_identity, road_run=str(self.road_run),
                            road_checkpoint_sha256=sha(self.road_checkpoint), road_iteration=self.road_iteration,
                            contributors_sha256=sha(contributors), repair_state_sha256=self.repair_hash,
                            road_colour_baseline_sha256=self.baseline_hash, M2_sha256=self.asset_file_hash,
                            native_metadata_sha256={str(path): sha(path) for path in metadata_paths},
                            road_source_sha256={path.name: sha(path) for path in paths if path.parent == self.road_run/'source'},
                            groups=GROUPS, native_static_opacity='own_native_opacity * fresh_road1K_gate',
                            native_a6_rows_input=0, complete_M2=True, M2_degree=3,
                            native_degrees=dict(static=self.n.static.active_sh_degree, a37=self.n.dynamic.active_sh_degree),
                            raster='original native GaussianRasterizer; one real shared depth sort',
                            colour_evaluation='per-component precomputed RGB at its own degree; M2 always SH3')
        self.binding_sha256 = stable_hash(self.binding)
        self.assert_unchanged()

    def tensor_versions(self):
        result = {}
        for name, model in (('static', self.n.static), ('dynamic', self.n.dynamic)):
            for key, value in vars(model).items():
                if torch.is_tensor(value):
                    result[name, key] = (id(value), value._version)
        for name, module in (('road_repair', self.exp.repair), ('env', self.n.env), ('M2', self.asset)):
            for key, value in (*module.named_parameters(), *module.named_buffers()):
                result[name, key] = (id(value), value._version)
        return result

    def assert_unchanged(self, full=False):
        assert self.n.unchanged() and self.n.before == self.parent_identity, 'Frozen native changed'
        assert self.tensor_versions() == self._versions, 'Frozen model tensor identity/version changed'
        assert (id(self.baseline_colours), self.baseline_colours._version) == self._colour_versions
        assert all(identity(path) == value for path, value in self.file_identities.items()), 'Input file changed'
        if full:
            assert all(sha(path) == value for path, value in self.content_hashes.items()), 'Input content changed'
            assert tensors_hash(self.exp.repair.state_dict()) == self.repair_hash, 'Frozen road/gates changed'
            assert tensors_hash({'colours': self.baseline_colours}) == self.baseline_hash
            current = dict((*self.asset.named_parameters(), *self.asset.named_buffers()))
            assert set(current) == set(self.asset_before)
            for key, before in self.asset_before.items():
                after = current[key].detach().cpu().contiguous()
                assert before.dtype == after.dtype and before.shape == after.shape, key
                assert torch.equal(before.contiguous().reshape(-1).view(torch.uint8),
                                   after.reshape(-1).view(torch.uint8)), f'M2 tensor changed: {key}'
            assert self.asset.assert_geometry_unchanged()
        return True

    def raw(self, raw):
        if raw not in RAWS:
            raise KeyError(raw)
        if raw not in self._data:
            obs_path = self.observations/f'{raw:03d}.npz'
            shadow_path = self.road_run/'inputs/views'/f'{raw:03d}.npz'
            for path in (obs_path, shadow_path):
                self.file_identities[str(path)] = identity(path)
                self.content_hashes[str(path)] = sha(path)
            with np.load(obs_path, allow_pickle=False) as saved:
                obs = {key: saved[key].copy() for key in saved.files}
            camera = self.exp.camera(raw)
            np.testing.assert_allclose(obs['model_to_camera'].T, camera.world_view_transform.cpu().numpy(), atol=1e-6, rtol=0)
            np.testing.assert_allclose(self.fixed_cameras[raw-89], obs['model_to_camera'], atol=1e-6, rtol=0)
            assert abs(float(obs['model_time'])-camera.timestamp) < 1e-6
            assert abs(float(self.fixed_times[raw-89])-camera.timestamp) < 1e-6
            record = next(row for row in self.n.records if row['img_name'] == f'0008{raw:03d}')
            np.testing.assert_allclose(obs['K'], [[record['fx']/4, 0, record['cx']/4],
                                                [0, record['fy']/4, record['cy']/4], [0, 0, 1]], atol=1e-6, rtol=0)
            rgb = torch.as_tensor(obs['rgb'], device='cuda').float().permute(2, 0, 1)
            if float(rgb.max()) > 1:
                rgb = rgb/255.
            pose = torch.as_tensor(obs['model_from_local'][0], device='cuda').float()
            env = self.n.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1).detach()
            self._data[raw] = dict(cam=camera, pose=pose, rgb=rgb, obs=obs, env=env)
        return self._data[raw]

    def native_rows(self, camera, dynamic=False):
        model = self.n.dynamic if dynamic else self.n.static
        rows = self.prepare(camera, model, self.pipe, dynamic=dynamic)
        if rows['colors_precomp'] is None:
            direction = rows['means3D']-camera.camera_center
            direction = direction/direction.norm(dim=-1, keepdim=True)
            colour = (self.eval_sh(model.active_sh_degree, rows['shs'].transpose(1, 2), direction)+.5).clamp_min(0)
        else:
            colour = rows['colors_precomp']
        return dict(rows, means2D=torch.zeros_like(rows['means3D']), shs=None, colors_precomp=colour)

    def scene(self, raw, delete=False, road_colours=None):
        self.assert_unchanged()
        data = self.raw(raw)
        camera = data['cam']
        static = self.native_rows(camera)
        ids = self.exp.repair.ids
        static['opacities'] = static['opacities'].index_copy(
            0, ids, static['opacities'][ids]*self.exp.repair.gate_raw.sigmoid()[:, None])
        dynamic = self.native_rows(camera, dynamic=True)
        a37 = select(dynamic, ~self.n.keep_a6)  # Physical exclusion of every native a6, including descendants.
        if delete:
            a37 = select(a37, torch.zeros_like(a37['mask']))
        asset = self.asset.rows(data['pose'], camera, self.pipe, channels=16, precomputed=True)
        assert len(asset['means3D']) == len(self.asset.xyz) > 0
        joint = self.merge(self.merge(tagged(static, 'native_static'), tagged(asset, 'asset')), tagged(a37, 'a37'))
        road = self.road_kwargs(camera, self.exp.repair.road, self.pipe, sh_channels=1, precomputed_color=True)
        road['colors_precomp'] = self.baseline_colours if road_colours is None else road_colours
        start = len(joint['features'])
        joint = self.merge(joint, tagged(road, 'road'))
        assert joint['shs'] is None and joint['colors_precomp'] is not None
        assert bool((joint['features'][:, 6:] == 0).all())
        return (joint, slice(start, len(joint['features'])), data['env'])

    def render(self, raw, scene):
        rows, _, env = scene
        data = self.raw(raw)
        out = self.raster(data['cam'], self.pipe, 0, rows, self.n.bg)
        out['rgb_premultiplied'] = out['rgb']
        out['rgb'] = out['rgb']+(1-out['alpha'])*env
        return out

    def render_scene(self, raw, delete=False, colours=None):
        scene = self.scene(raw, delete, self.road_colours if colours is None else colours)
        out = self.render(raw, scene)
        assert float((out['extra'][:4].sum(0)-out['alpha'][0]).abs().max()) < 2e-5
        return out

    def load_fine(self, folder, basis_function):
        folder = Path(folder).resolve()
        meta = read(folder/'scene_binding.json')
        if meta['binding_sha256'] != self.binding_sha256 or meta['stage'] != 'fine11x9':
            raise ValueError('Fine field was not calibrated on this exact new native/M2/road scene')
        if self.purpose == 'final' and meta['preflight_only']:
            raise ValueError('Preflight coefficients cannot be used for final acceptance')
        with np.load(folder/'coefficients.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['colour_baseline'], self.baseline_colours.cpu().numpy()):
                raise ValueError('Fine-road baseline differs')
            coeff = saved['world'].copy()
        if coeff.shape != (99, 3) or not np.isfinite(coeff).all() or np.max(np.abs(coeff)) > .09000001:
            raise ValueError('Unexpected fine-road coefficient field')
        basis, knots = basis_function(self.road_xy)
        stored_knots = read(folder/'design.json')['knots']
        assert tuple(stored_knots['grid']) == (11, 9)
        for key in ('low_m', 'high_m'):
            np.testing.assert_allclose(knots[key], stored_knots[key], atol=1e-6, rtol=0)
        gpu_basis = self.baseline_colours.new_tensor(basis)
        assert torch.allclose(gpu_basis.sum(1), torch.ones_like(gpu_basis[:, 0]), atol=2e-6, rtol=0)
        self.road_colours = (self.baseline_colours+gpu_basis@gpu_basis.new_tensor(coeff)).clamp(0, 1)
        self.fine_binding = dict(path=str(folder), coefficients_sha256=sha(folder/'coefficients.npz'),
                                 colour_table_sha256=tensors_hash({'colours': self.road_colours}), knots=knots)
        self._fine_colour_version = (id(self.road_colours), self.road_colours._version)

    def receipt(self):
        self.assert_unchanged(full=True)
        if self.fine_binding:
            assert (id(self.road_colours), self.road_colours._version) == self._fine_colour_version
            assert tensors_hash({'colours': self.road_colours}) == self.fine_binding['colour_table_sha256']
        return dict(binding=self.binding, binding_sha256=self.binding_sha256,
                    M2_tensor_count=len(self.asset_before), M2_all_27_tensors_bitwise_unchanged_CPU=True,
                    frozen_native_and_road_unchanged=True, fine_binding=self.fine_binding,
                    native_a6_rows_removed=int(self.n.keep_a6.sum()), native_a37_rows=int((~self.n.keep_a6).sum()),
                    input_identities=self.file_identities, input_sha256=self.content_hashes,
                    training_updates=0, quality_accepted=False, human_verdict=None)

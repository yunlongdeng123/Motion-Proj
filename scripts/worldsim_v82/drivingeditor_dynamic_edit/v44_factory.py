"""Build the shared renderer from data manifests, without scene-specific branches."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
try:
    from .v44_manifest import SceneManifest, make_native_camera, _pose, _matrix, _finite_number, _frame_key
except ImportError:
    from v44_manifest import SceneManifest, make_native_camera, _pose, _matrix, _finite_number, _frame_key


def _owner_array(value, count, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iu' or a.shape != (count,):
        raise ValueError(name + ' must be an integer array matching native row count; no float casting')
    if (a.dtype.kind == 'i' and (a < -1).any()) or (a.size and int(a.max()) > np.iinfo(np.int64).max):
        raise ValueError(name + ' must fit int64 with -1 unknown, 0 background, or positive IDs')
    return a.astype(np.int64, copy=True)


def _load_ownership(path, ns, nd):
    with np.load(path, allow_pickle=False) as z:
        required = {'static_actor_ids', 'dynamic_actor_ids'}
        if not required <= set(z.files): raise ValueError('Ownership NPZ needs both native row branches')
        return {'static': _owner_array(z['static_actor_ids'], ns, 'static_actor_ids'),
                'dynamic': _owner_array(z['dynamic_actor_ids'], nd, 'dynamic_actor_ids')}


def _load_road_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    required = {'active', 'model_from_world', 'meter_to_model', 'model_xyz', 'gaussian_sigma_m',
                'normal_world', 'reference_srgb', 'road_opacity'}
    if not required <= set(arrays): raise ValueError('Road NPZ lacks required Gaussian geometry/material fields')
    active = arrays['active']
    if active.ndim != 1 or active.dtype.kind not in 'biu' or not np.isin(active, [0, 1]).all():
        raise ValueError('Road active must be a bool or integer 0/1 row mask')
    n = len(active); arrays['active'] = active.astype(bool, copy=True)
    for key in ('model_xyz', 'gaussian_sigma_m', 'normal_world', 'reference_srgb'):
        arrays[key] = _matrix(arrays[key], (n, 3), 'road ' + key)
    arrays['road_opacity'] = _matrix(arrays['road_opacity'], (n,), 'road_opacity')
    scalar = np.asarray(arrays['meter_to_model'])
    if scalar.shape != () or scalar.dtype.kind not in 'iuf': raise ValueError('meter_to_model must be a real scalar')
    scale = _finite_number(scalar.item(), 'meter_to_model', positive=True)
    transform = _pose(arrays['model_from_world'], 'road model_from_world', similarity=True)
    if not np.allclose(np.linalg.norm(transform[:3, :3], axis=0) / scale, 1., atol=2e-6, rtol=0):
        raise ValueError('Road transform and meter_to_model scales disagree')
    normals = np.linalg.norm(arrays['normal_world'], axis=1)
    if ((arrays['gaussian_sigma_m'] <= 0).any() or not np.isfinite(normals).all() or (normals <= 0).any()
            or (arrays['road_opacity'] < 0).any() or (arrays['road_opacity'] > 1).any()):
        raise ValueError('Road needs positive scales/nonzero normals and opacity in [0,1]')
    arrays['meter_to_model'], arrays['model_from_world'] = scale, transform
    return arrays


def _load_actor_poses(path, track_id, frame_keys):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 'v44.actor_poses.v1' or data.get('track_id') != track_id:
        raise ValueError('Object pose identity/schema mismatch')
    poses = {}
    for row in data['frames']:
        key = _frame_key(row['key']); value = np.asarray(row['world_from_actor'])
        if key in poses or key not in frame_keys: raise ValueError('Object pose has duplicate or unregistered frame key')
        if value.shape == (16,): value = value.reshape(4, 4)
        poses[key] = _pose(value, 'object world_from_actor')
    return poses


class SurfaceRows:
    def __init__(self, asset, poses, world_to_model):
        self.asset, self.poses = asset, poses
        self.world_to_model = _pose(world_to_model, 'surface world_to_model', similarity=True)

    def rows(self, frame_key, camera, pipe):
        import torch
        if frame_key not in self.poses:
            return None
        pose = torch.as_tensor(self.world_to_model @ self.poses[frame_key],
                               device=self.asset.base_asset.xyz.device if hasattr(self.asset, 'base_asset') else self.asset.xyz.device,
                               dtype=torch.float32)
        return self.asset.rows(pose, camera, pipe, 1, precomputed=True)


class RoadRows:
    def __init__(self, path, device):
        arrays = _load_road_arrays(path)
        import torch
        active = arrays['active']
        tensor = lambda a: torch.as_tensor(np.array(a, copy=True), device=device, dtype=torch.float32)
        transform = arrays['model_from_world']; scale = float(arrays['meter_to_model'])
        self.xyz = tensor(arrays['model_xyz'][active])
        self.scaling = tensor(arrays['gaussian_sigma_m'][active] * scale)
        normal = arrays['normal_world'][active] @ transform[:3, :3].T / scale
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        q = np.c_[1 + normal[:, 2], -normal[:, 1], normal[:, 0], np.zeros(len(normal))]
        singular = np.linalg.norm(q, axis=1) < 1e-8
        q[singular] = [0., 1., 0., 0.]
        q /= np.linalg.norm(q, axis=1)[:, None]
        self.rotation = tensor(q); self.rgb = tensor(arrays['reference_srgb'][active])
        self.opacity = torch.logit(tensor(arrays['road_opacity'][active, None]).clamp(1e-5, 1-1e-5))
        self.active = torch.ones(len(self.xyz), dtype=torch.bool, device=device)

    def colors(self):
        return self.rgb

    def output_mask(self):
        return self.active

    def rows(self, frame_key, camera, pipe):
        if not len(self.xyz):
            return None
        from utils.v4_road_render import road_kwargs
        return road_kwargs(camera, self, pipe, precomputed_color=True)


def _row_mask(path, count):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('Native row count must be a nonnegative integer')
    if path is None: return np.zeros(count, dtype=bool)
    if Path(path).suffix.lower() == '.npy':
        value = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as z:
            if not {'mask', 'ids'} & set(z.files): raise ValueError('Retirement NPZ needs mask or ids')
            key = 'mask' if 'mask' in z else 'ids'
            value = z[key]
    if value.dtype.kind == 'b':
        if value.shape != (count,): raise ValueError('Retirement mask topology mismatch')
        return value.copy()
    if value.dtype.kind not in 'iu' or value.ndim != 1 or (value < 0).any() or (value >= count).any():
        raise ValueError('Retirement rows must be valid integer indices')
    mask = np.zeros(count, dtype=bool); mask[value] = True
    return mask


def from_manifest(manifest):
    """CUDA worker factory; the same function serves every scene and edit target."""
    import torch
    from .scene_io import load_native_models
    from .ownership import actor_ids
    from .v44_scene import V44Scene, RowProvider
    from .v44_sh_asset import SHAsset
    from .v44_surface_lod import SurfaceLODAsset
    m = SceneManifest.load(manifest) if not isinstance(manifest, SceneManifest) else manifest
    pipe, models, _ = load_native_models(m.model, m.iteration, engine=m.engine)
    static, dynamic = models['static'], models['dynamic']
    ns, nd = len(static.get_xyz), len(dynamic.get_xyz)
    source = {'static': np.full(ns, -1, dtype=np.int64),
              'dynamic': _owner_array(actor_ids(dynamic).detach().cpu().numpy(), nd, 'native dynamic actor IDs')}
    source['dynamic'][source['dynamic'] == 0] = -1
    semantic = None
    if m.ownership is not None:
        semantic = _load_ownership(m.ownership, ns, nd)
        for branch, count in [('static', ns), ('dynamic', nd)]:
            if semantic[branch].shape != (count,): raise ValueError('Ownership/model topology mismatch')
            zero = semantic[branch] == 0
            semantic[branch][zero] = source[branch][zero] if m.zero_semantic_policy == 'source_fallback' else -1
    fixed_owners = source if semantic is None else semantic
    valid_ids = set(m.actor_to_track)
    for branch in ('static', 'dynamic'):
        if not set(np.unique(fixed_owners[branch][fixed_owners[branch] > 0])).issubset(valid_ids):
            raise ValueError('Ownership contains actor IDs absent from the explicit registry')
    headers = m.camera_headers(); cameras = {}
    def camera_provider(key):
        if key not in headers: raise KeyError('Frame is not in this saved scene')
        if key not in cameras: cameras[key] = make_native_camera(headers[key])
        return cameras[key]
    providers = []
    for item in m.surfaces:
        if m.world_to_model is None: raise ValueError('Object surfaces require world_to_model')
        actor = m.track_to_actor[item['track_id']]
        base = SHAsset.from_geometry(item['geometry']).to(static.get_xyz.device).requires_grad_(False)
        asset = SurfaceLODAsset(base, item['posterior'], item['reference_hw']) if item.get('posterior') else base
        poses = _load_actor_poses(item['poses'], item['track_id'], set(headers))
        wrapped = SurfaceRows(asset, poses, m.world_to_model)
        retire_s, retire_d = _row_mask(item.get('retire_static_rows'), ns), _row_mask(item.get('retire_dynamic_rows'), nd)
        if item.get('retire_native_owner', False):
            retire_s |= fixed_owners['static'] == actor
            retire_d |= fixed_owners['dynamic'] == actor
        if item.get('retire_dynamic_source_owner', False): retire_d |= source['dynamic'] == actor
        providers.append(RowProvider(wrapped.rows, actor, 'surface', retire_s, retire_d,
                                     state_objects=(base,), require_complete_coverage=item.get('require_complete_coverage', False),
                                     assert_persistent_unchanged=getattr(asset, 'assert_persistent_unchanged', base.assert_geometry_unchanged)))
    if m.road is not None:
        road = RoadRows(m.road, static.get_xyz.device)
        providers.append(RowProvider(road.rows, 0, 'road', state_objects=(road,)))
    gates = None
    if m.static_gates is not None:
        with np.load(m.static_gates['indices'], allow_pickle=False) as z: ids = z['ids']
        state = torch.load(m.static_gates['checkpoint'], map_location='cpu', weights_only=False)
        if not torch.isfinite(state['gate']).all(): raise ValueError('Fixed static-gate logits must be finite')
        values = state['gate'].sigmoid().detach().cpu().numpy()
        if (ids.dtype.kind not in 'iu' or ids.ndim != 1 or (ids < 0).any() or (ids >= ns).any()
                or values.shape != ids.shape or not np.isfinite(values).all()
                or len(np.unique(ids)) != len(ids)):
            raise ValueError('Fixed static-gate topology mismatch')
        gates = np.ones(ns, dtype=np.float32); gates[ids] = values
    scene = V44Scene(static, dynamic, models['environment'], pipe, camera_provider,
                     source_owner_ids=source, semantic_owner_ids=semantic,
                     row_providers=providers, static_gates=gates)
    return scene, m

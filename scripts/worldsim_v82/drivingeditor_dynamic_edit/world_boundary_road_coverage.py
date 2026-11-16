"""Fixed world-boundary coverage on a preloaded, fixed-H2 C49 scene."""
from pathlib import Path
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class WorldBoundaryRoadCoverage:
    """No optimizer, qualification pass or CUDA change; one native public render."""
    def __init__(self, fixed_h2_runtime, proof_path):
        self.h2 = fixed_h2_runtime
        self.road = fixed_h2_runtime.road
        self.base = self.road.base
        self.torch = self.road.torch
        self.proof_path = Path(proof_path).resolve()
        self.proof = json.loads(self.proof_path.read_text())
        p = self.proof
        if p['schema'] != 'fixed-world-boundary-road-coverage-v1' or p['band_m'] != .4:
            raise ValueError('only registered fixed0.4m world coverage is supported')
        if (self.road.manifest['new_road']['sha256'] != p['material_asset_sha256']
                or sha(self.road.manifest_path) != p['material_scene_manifest_sha256']):
            raise ValueError('material scene differs from registered formal2K')
        for key, expected in p['geometry_array_sha256'].items():
            if array_sha(self.road.asset[key]) != expected:
                raise ValueError('fixed geometry/opacity/map/coordinates changed: ' + key)
        gain_path = self.proof_path.parent / p['gain_file']
        if sha(gain_path) != p['gain_sha256']:
            raise ValueError('fixed coverage gain file changed')
        with np.load(gain_path, allow_pickle=False) as z:
            gain = z['child_gain'].copy()
        if gain.shape != (40392,) or gain.dtype != np.float32 or np.any(gain < 0) or np.any(gain > 1):
            raise ValueError('invalid fixed child coverage')
        inverse = np.linalg.inv(self.road.asset['world_to_model'] @ self.road.asset['coordinate_anchor'])
        xyz = self.road.asset['render_model_xyz'].astype(np.float64)
        local = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        bounds = p['source_crop_planes_m']
        distance = np.minimum.reduce((local[:, 0]-bounds['near_x'],
            local[:, 1]-bounds['lower_y'], bounds['upper_y']-local[:, 1]))
        u = np.clip(distance/.4, 0, 1)
        expected_gain = (u*u*(3-2*u)).astype(np.float32)
        if not np.array_equal(gain, expected_gain):
            raise ValueError('gain does not equal the fixed source-boundary formula')
        self.gain = self.torch.as_tensor(gain, device=self.base.n.bg.device).detach()
        self._gain_version = self.gain._version
        self._files = {str(self.proof_path): sha(self.proof_path), str(gain_path): sha(gain_path)}
        self._cache = {}
        for item in p['H2_cached_visibility']:
            key = (int(item['raw']), item['state'] == 'delete')
            if key not in self.h2.cache:
                raise ValueError('preload the registered H2 cache; new probes are forbidden')
            path = Path(item['path'])
            if sha(path) != item['sha256']:
                raise ValueError('H2 visibility file changed')
            with np.load(path, allow_pickle=False) as z:
                mask = self.torch.as_tensor(z['selected_road_child_rows'], device=self.gain.device)
                guard = self.torch.as_tensor(z['model_foreground_guard'], device=self.gain.device)
            actual = self.h2.cache[key]
            if not actual['record']['converged'] or not self.torch.equal(mask, actual['mask']) or not self.torch.equal(guard, actual['guard']):
                raise ValueError('H2 runtime cache differs from the fixed source')
            self._cache[key] = (mask.detach().clone(), guard.detach().clone())
            self._files[str(path)] = item['sha256']
        self.render_calls = 0
        self.assert_unchanged(full=True)

    def __getattr__(self, name):
        return getattr(self.road, name)

    def scene(self, raw, delete=False):
        key = (int(raw), bool(delete))
        if type(delete) is not bool or key not in self._cache:
            raise ValueError('requires a registered raw/state and preloaded fixed H2 mask')
        self.assert_unchanged()
        with self.torch.no_grad():
            scene = self.h2.scene(int(raw), delete=delete)
            rows = scene['rows']
            offset = int(scene['meta']['latest_semantic_prefix_rows'])
            if len(rows['mask']) != offset + 40392:
                raise RuntimeError('complete canonical C road tail changed')
            expected_mask, _ = self._cache[key]
            if not self.torch.equal(rows['mask'][offset:], expected_mask):
                raise RuntimeError('fixed H2 child mask changed')
            alpha = rows['opacities'].clone()
            alpha[offset:] = alpha[offset:] * self.gain[:, None]
            # No original tensor is modified. Effective child alpha is the
            # only packet field changed after original relative-Mip and H2.
            packet = dict(rows, opacities=alpha)
            meta = dict(scene['meta'], world_boundary_coverage=dict(
                proof_path=str(self.proof_path), proof_sha256=self._files[str(self.proof_path)],
                band_m=.4, formula='smoothstep(clip(min(source_three_inward_distances)/.4,0,1))',
                source_crop_planes_m=self.proof['source_crop_planes_m'],
                applied_after_original_relativeMip=True, fixed_across_view_and_edit=True,
                persistent_intrinsic_opacity_and_geometry_unchanged=True,
                H2_fixed_visibility_unchanged=True, material_RGB_unchanged=True,
                original_parameter_updates=0, trainable_parameters=0),
                effective_newroad_opacity_changed=True)
            return dict(scene, rows=packet, meta=meta)

    def render(self, raw, scene):
        key = (int(raw), bool(scene['meta']['delete_a37']))
        if key not in self._cache:
            raise ValueError('unregistered scene state')
        self.render_calls += 1
        out = self.road.render(raw, scene)
        guard = self._cache[key][1]
        mass = float(out['group_alpha_T'][7][guard].double().sum())
        if mass != 0:
            raise RuntimeError('boundary coverage violates the original H2 model foreground')
        out['meta']['public_foreground_road_alpha_mass'] = mass
        self.assert_unchanged()
        return out

    def render_scene(self, raw, delete=False):
        return self.render(raw, self.scene(raw, delete))

    def assert_unchanged(self, full=False):
        self.road.assert_unchanged(full=full)
        if self.gain._version != self._gain_version:
            raise RuntimeError('fixed world coverage buffer changed')
        for key, (mask, guard) in self._cache.items():
            current = self.h2.cache[key]
            if not self.torch.equal(mask, current['mask']) or not self.torch.equal(guard, current['guard']):
                raise RuntimeError('cached H2 visibility changed')
        if full:
            for path, expected in self._files.items():
                if sha(path) != expected:
                    raise RuntimeError('world coverage input file changed: ' + path)
        return True

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(schema='fixed-world-boundary-road-coverage-runtime-v1',
            proof_path=str(self.proof_path), proof_sha256=self._files[str(self.proof_path)],
            native_road_receipt=self.road.receipt(), band_m=.4,
            source_crop_planes_m=self.proof['source_crop_planes_m'],
            gain_summary=self.proof['gain_summary'], actual_native_forwards=self.render_calls,
            actual_dummy_backwards=0, optimizer_updates=0,
            intrinsic_opacity_geometry_RGB_and_identity_unchanged=True,
            fixed_H2_visibility_reused=True, adoption_claim=False, human_verdict=None)

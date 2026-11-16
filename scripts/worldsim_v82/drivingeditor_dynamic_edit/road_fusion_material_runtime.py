"""Differentiable parent RGB for the frozen 21K Semantic-a5/V44 plus C scene.

This adapter creates no optimizer and changes no persistent source tensor. The
caller supplies a complete 10098x3 parent table, normally assembled from the
fixed C table and a separately qualified trainable subset by ``index_copy``.
Only that table has a gradient path. All 40392 child rows retain the original
geometry, opacity, relative-Mip policy, masks and joint depth ordering.
"""
from pathlib import Path
import hashlib

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RoadFusionMaterialRuntime:
    def __init__(self, road_runtime):
        # Structural checks allow the exact same frozen class to be imported
        # through a content-addressed campaign snapshot or the repo package.
        for name in ('scene', 'base', 'asset', 'parent_indices',
                     'assert_unchanged', 'receipt'):
            if not hasattr(road_runtime, name):
                raise TypeError('expected frozen RoadFusionV44Integration: ' + name)
        self.road = road_runtime
        self.base = road_runtime.base
        self.torch = road_runtime.torch
        self.np = road_runtime.np
        asset = road_runtime.asset
        parents = asset['parent_render_colours_srgb']
        children = asset['render_colours_srgb']
        mapping = asset['child_to_parent_ids']
        if (parents.shape != (10098, 3) or parents.dtype != np.float32
                or children.shape != (40392, 3)
                or not np.array_equal(mapping, np.repeat(np.arange(10098), 4))
                or not np.array_equal(parents[mapping], children)):
            raise ValueError('C parent colours must reproduce every child exactly')
        if not np.isfinite(parents).all() or np.any(parents < 0) or np.any(parents > 1):
            raise ValueError('invalid C warm-start RGB')
        self.initial_parent_rgb = self.torch.as_tensor(parents.copy(),
            dtype=self.torch.float32, device=self.base.n.bg.device).detach()
        self.child_to_parent_ids = road_runtime.parent_indices.detach()
        self._initial_rgb_version = int(self.initial_parent_rgb._version)
        self._map_version = int(self.child_to_parent_ids._version)
        self._initial_parent_sha = hashlib.sha256(parents.tobytes()).hexdigest()
        self.render_calls = 0
        self.assert_unchanged(full=True)

    def __getattr__(self, name):
        return getattr(self.road, name)

    def parent_table(self, eligible_parent_ids, eligible_rgb):
        """Assemble fixed C + qualified RGB; gradient flows only to eligible_rgb."""
        if (eligible_parent_ids.dtype != self.torch.long
                or eligible_parent_ids.ndim != 1
                or eligible_parent_ids.device != self.initial_parent_rgb.device
                or eligible_rgb.shape != (len(eligible_parent_ids), 3)
                or eligible_rgb.dtype != self.initial_parent_rgb.dtype
                or eligible_rgb.device != self.initial_parent_rgb.device):
            raise ValueError('invalid eligible parent IDs/RGB tensor')
        if len(eligible_parent_ids):
            ids = eligible_parent_ids.detach()
            if (int(ids.min()) < 0 or int(ids.max()) >= 10098
                    or len(self.torch.unique(ids)) != len(ids)):
                raise ValueError('eligible parents must be unique valid row IDs')
        return self.initial_parent_rgb.index_copy(0, eligible_parent_ids, eligible_rgb)

    def scene(self, raw, delete=False, parent_rgb=None):
        self.assert_unchanged()
        parent_rgb = self.initial_parent_rgb if parent_rgb is None else parent_rgb
        if (parent_rgb.shape != (10098, 3)
                or parent_rgb.dtype != self.initial_parent_rgb.dtype
                or parent_rgb.device != self.initial_parent_rgb.device):
            raise ValueError('parent_rgb must be a full 10098x3 float32 device table')
        if not bool(self.torch.isfinite(parent_rgb).all()):
            raise ValueError('nonfinite parent RGB')
        # The frozen scene constructor performs its native no-grad preparation.
        # Detach explicitly as some source fields may alias original Parameters.
        original = self.road.scene(raw, delete=delete)
        rows = {key: value.detach() if self.torch.is_tensor(value) else value
                for key, value in original['rows'].items()}
        offset = int(original['meta']['latest_semantic_prefix_rows'])
        if len(rows['mask']) != offset + 40392 or rows['features'].shape[1] != 24:
            raise RuntimeError('frozen road tail/feature ABI changed')
        child_rgb = parent_rgb.index_select(0, self.child_to_parent_ids)
        # These two replacements MUST remain outside no_grad. Existing prefix
        # values are detached and remain exact; only parent RGB can backprop.
        rows['colors_precomp'] = self.torch.cat(
            (rows['colors_precomp'][:offset], child_rgb), dim=0)
        features = rows['features'].clone()
        features[offset:, 13:16] = child_rgb
        rows['features'] = features
        meta = dict(original['meta'],
            material_runtime='road-fusion-parent-rgb-v1',
            material_parent_rows=10098, material_child_rows=40392,
            material_warm_start='saved_C_parent_render_colours_srgb',
            geometry_opacity_masks_mip_and_original_prefix_frozen=True,
            original_parameter_updates=0)
        return dict(original, rows=rows, env=original['env'].detach(),
                    bg=original['bg'].detach(), meta=meta)

    def render(self, raw, scene):
        # Same single native joint kernel/env composition as frozen fd755;
        # deliberately no no_grad decorator/context in this method.
        out = self.base.raster(scene['cam'], self.base.pipe, 0,
                               scene['rows'], scene['bg'])
        self.render_calls += 1
        if out['extra'].shape[0] < 14:
            raise RuntimeError('24-channel native feature ABI changed')
        out['rgb_premultiplied'] = out['rgb']
        out['group_alpha_T'] = out['extra'][:8]
        out['component_alpha_T'] = out['group_alpha_T']
        rgb = out['extra'][8:11]
        out['added_owner_rgb_T'] = self.torch.stack(
            (rgb, self.torch.zeros_like(rgb), self.torch.zeros_like(rgb)))
        out['added_owner_rgb_actor_ids'] = [5, 18, 26]
        out['newroad_rgb_T'] = out['extra'][11:14]
        out['env_rgb_T'] = (1-out['alpha']) * scene['env']
        out['constant_background_rgb_T'] = (1-out['alpha']) * scene['bg'][:, None, None]
        out['rgb'] = out['rgb_premultiplied'] + out['env_rgb_T']
        error = float((out['group_alpha_T'].sum(0)-out['alpha'][0]).abs().max().detach())
        if error >= 2e-5:
            raise RuntimeError('eight joint alphaT groups do not conserve alpha')
        for key in ('rgb', 'alpha', 'depth_sum', 'extra',
                    'added_owner_rgb_T', 'newroad_rgb_T'):
            if not bool(self.torch.isfinite(out[key]).all()):
                raise RuntimeError('nonfinite render: ' + key)
        out['group_conservation_maxabs'] = error
        out['meta'] = scene['meta']
        return out

    def render_scene(self, raw, delete=False, parent_rgb=None):
        return self.render(raw, self.scene(raw, delete, parent_rgb))

    def assert_unchanged(self, full=False):
        self.road.assert_unchanged(full=full)
        if (int(self.initial_parent_rgb._version) != self._initial_rgb_version
                or int(self.child_to_parent_ids._version) != self._map_version):
            raise RuntimeError('fixed material initialization/mapping was mutated')
        if full:
            initial = self.initial_parent_rgb.detach().cpu().contiguous().numpy()
            if hashlib.sha256(initial.tobytes()).hexdigest() != self._initial_parent_sha:
                raise RuntimeError('fixed C parent RGB bytes changed')
        return True

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(schema='road-fusion-parent-rgb-runtime-v1',
            frozen_road=self.road.receipt(), source_file=str(Path(__file__).resolve()),
            source_sha256=_sha(__file__), parent_rows=10098, child_rows=40392,
            initial_parent_rgb_sha256=self._initial_parent_sha,
            initial_child_rgb_exact_to_saved_C=True,
            selection_owned_by_external_all49_visibility_qualification=True,
            optimizer_created=False, original_parameter_updates=0,
            shared_joint_sort=True, render_calls=self.render_calls,
            physical_prefix_frozen=True, geometry_opacity_mask_mip_frozen=True,
            adoption_claim=False)

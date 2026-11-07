"""Conservative whole-road-primitive culling against frozen 3D foreground.

This is a separate engineering candidate. It does not replace the unmodified C
material control. The foreground support comes only from the baseline model's
native actor alpha*T, never from query RGB or observed masks.
"""
import hashlib
import json

from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration


class ForegroundRoadCulling:
    def __init__(self, road_runtime, max_iterations=64):
        if not isinstance(road_runtime, RoadFusionV44Integration):
            raise TypeError('Expected the frozen21K v44+semantic-a5+C runtime')
        if max_iterations != 64:
            raise ValueError('This registered candidate fixes the cap at64')
        self.road = road_runtime
        self.torch = road_runtime.torch
        self.cache = {}
        self.max_iterations = max_iterations
        self.actual_forwards = 0
        self.actual_dummy_backwards = 0
        self.actual_public_forwards = 0

    def __getattr__(self, name):
        return getattr(self.road, name)

    def _foreground(self, raw, delete):
        with self.torch.no_grad():
            original = self.road.semantic.render_scene(raw, delete=delete)
        self.actual_forwards += 1
        groups = original['group_alpha_T']
        # Existing native components: C2/a6, appended semantic-a5, retained a37.
        guard = (groups[0] > 0) | (groups[4] > 0)
        if not delete:
            guard = guard | (groups[1] > 0)
        return guard.detach(), original

    def _qualify(self, raw, delete):
        key = (int(raw), bool(delete))
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        guard, baseline = self._foreground(raw, delete)
        scene = self.road.scene(raw, delete=delete)
        offset = int(scene['meta']['latest_semantic_prefix_rows'])
        n = len(self.road.asset['active'])
        selected = scene['rows']['mask'][offset:].detach().clone()
        if n != 40392 or len(selected) != n:
            raise ValueError('C geometry topology must remain complete40392')
        iterations = []
        converged = False
        for iteration in range(self.max_iterations):
            rows = {k: v.detach() if torch.is_tensor(v) else v
                    for k, v in scene['rows'].items()}
            mask = rows['mask'].clone()
            mask[offset:] &= selected
            probe = torch.ones((n,1), device=mask.device,
                               dtype=rows['features'].dtype, requires_grad=True)
            zeros = probe.new_zeros((offset,1))
            features = rows['features']
            if bool((features[:,16] != 0).any()):
                raise ValueError('Probe requires unused feature16')
            rows = dict(rows, mask=mask,
                features=torch.cat((features[:,:16],torch.cat((zeros,probe),0),features[:,17:]),1))
            out = self.road.base.raster(scene['cam'],self.road.base.pipe,0,rows,scene['bg'])
            self.actual_forwards += 1
            road_w = out['extra'][7].detach()
            dummy_w = out['extra'][14]
            dummy_error = float((dummy_w.detach()-road_w).abs().max())
            if dummy_error > 1e-6:
                raise RuntimeError('Native per-row contribution probe is not linear')
            loss = dummy_w[guard].sum()
            mass = torch.autograd.grad(loss,probe,create_graph=False,retain_graph=False)[0][:,0].detach()
            self.actual_dummy_backwards += 1
            if not bool(torch.isfinite(mass).all()) or bool((mass<0).any()):
                raise RuntimeError('Invalid native road contribution mass')
            expected = float(road_w[guard].double().sum())
            measured = float(mass.double().sum())
            if abs(expected-measured) > max(1e-6,abs(expected)*1e-5):
                raise RuntimeError('Contribution mass does not match native guard readout')
            rejected = selected & (mass > 0)
            iterations.append(dict(iteration=iteration,active_before=int(selected.sum()),
                rejected_rows=int(rejected.sum()),foreground_pixels=int(guard.sum()),
                guard_road_alpha_mass=expected,probe_mass=measured,
                dummy_alpha_maxabs=dummy_error))
            del out, rows, loss, probe, zeros, features, dummy_w, road_w
            if not bool(rejected.any()):
                converged = True
                break
            selected = selected & ~rejected
        record = dict(raw=int(raw),delete=bool(delete),converged=converged,
            active_rows=int(selected.sum()),total_asset_rows=n,iterations=iterations,
            foreground_components=['C2_a6','semantic_a5']+([] if delete else ['retained_a37']),
            guard_predicate='native baseline alpha_T>0; model-derived only',
            rows_only_culled=True,gaussian_parameters_updated=0,
            visibility_policy_fixed=True,colour_used_for_selection=False,
            observed_mask_used_for_selection=False)
        self.cache[key] = dict(mask=selected.detach().clone(),guard=guard,
                               baseline=baseline,record=record)
        return self.cache[key]

    def scene(self, raw, delete=False):
        qualified = self._qualify(raw,delete)
        if not qualified['record']['converged']:
            raise RuntimeError('Culling did not reach zero foreground road contribution within64 iterations')
        scene = self.road.scene(raw,delete=delete)
        offset = int(scene['meta']['latest_semantic_prefix_rows'])
        rows = scene['rows'];mask = rows['mask'].clone()
        mask[offset:] &= qualified['mask']
        return dict(scene,rows=dict(rows,mask=mask),meta=dict(scene['meta'],
            foreground_road_culling=qualified['record'],
            query_visibility_from_frozen_model=True,
            road_geometry_and_material_tensors_unchanged=True,
            screen_colour_composite=False))

    def render_scene(self, raw, delete=False):
        scene = self.scene(raw,delete)
        out = self.road.render(raw,scene)
        self.actual_forwards += 1
        self.actual_public_forwards += 1
        guard = self.cache[(int(raw),bool(delete))]['guard']
        road_mass = float(out['group_alpha_T'][7][guard].double().sum())
        if road_mass != 0:
            raise RuntimeError('Public render exposes newly unmasked road on the protected foreground')
        out['meta']['public_foreground_road_alpha_mass'] = road_mass
        self.road.assert_unchanged()
        return out

    def receipt(self):
        self.road.assert_unchanged(full=True)
        return dict(status='FOREGROUND_ROAD_CULLING_RESEARCH_CANDIDATE_PENDING_QUALITY',
            actual_native_forwards=self.actual_forwards,
            actual_dummy_feature_backwards=self.actual_dummy_backwards,
            actual_public_forwards=self.actual_public_forwards,
            model_optimizer_updates=0,material_optimizer_updates=0,
            records=[self.cache[k]['record'] for k in sorted(self.cache)],
            native_road_receipt=self.road.receipt(),
            query_RGB_or_observed_masks_used_for_selection=False,
            inferred_visibility_changes_only=True,known_static_parked_cars_not_explicitly_guarded=True,
            quality_accepted=False,human_verdict=None)

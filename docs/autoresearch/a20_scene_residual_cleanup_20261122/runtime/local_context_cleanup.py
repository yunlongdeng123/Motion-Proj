"""Zero-update local context cleanup via source-bound, global-ID inactive sidecar.

Only explicitly supplied native static background/owner21 candidates are tested.
Every complete candidate set is rerendered against one fixed original scene.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import numpy as np

SEEDS = np.asarray([1503286, 670512, 842596], np.int64)
K_LIMIT = 1e-5
RGB_LIMIT = 1/255


def load_candidate_ids(path):
    if path is None: return np.sort(SEEDS.copy())
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            key = next((k for k in ('global_row_ids', 'candidate_global_row_ids') if k in z.files), None)
            if key is None: raise ValueError('Candidate NPZ requires global_row_ids')
            values = z[key].copy()
    else:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        values = value['global_row_ids'] if isinstance(value, dict) else value
        values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or (values < 0).any():
        raise ValueError('global_row_ids must be a one-dimensional nonnegative integer array')
    return np.union1d(SEEDS, values).astype(np.int64)


def protection_metrics(before, after, truth, K, V, allowed, ground):
    arrays = [before[k] for k in ('rgb', 'alpha', 'K_alpha', 'road_alpha')]
    arrays += [after[k] for k in ('rgb', 'alpha', 'K_alpha', 'road_alpha')]
    if not all(np.isfinite(v).all() for v in arrays): return dict(passed=False, finite=False)
    difference = np.abs(after['rgb']-before['rgb']); outside = ~allowed
    K_rgb = float(difference[:, K].max()); K_alpha = float(np.abs(after['K_alpha']-before['K_alpha'])[K].max())
    outside_rgb = float(difference[:, outside].max()) if outside.any() else 0.
    increment = np.abs(after['rgb'][:, V]-truth[:, V])-np.abs(before['rgb'][:, V]-truth[:, V])
    V_increment = max(0., float(increment.max()))
    new_holes = ground & (before['alpha'] >= .95) & (after['alpha'] < .95)
    conditions = dict(K_RGB=K_rgb <= K_LIMIT, K_visible_alpha=K_alpha <= K_LIMIT,
        outside_edit_RGB=outside_rgb <= RGB_LIMIT, true_V_absolute_RGB_error_increment=V_increment <= RGB_LIMIT,
        no_new_ground_alpha95_holes=not new_holes.any())
    return dict(passed=all(conditions.values()), finite=True, conditions=conditions,
        K_RGB_max=K_rgb, K_visible_alpha_max=K_alpha, outside_edit_RGB_max=outside_rgb,
        V_absolute_RGB_error_increment_max=V_increment,
        V_RGB_L1_before=float(np.abs(before['rgb'][:, V]-truth[:, V]).mean()),
        V_RGB_L1_after=float(np.abs(after['rgb'][:, V]-truth[:, V]).mean()),
        new_ground_alpha_below95_pixels=int(new_holes.sum()), ground_pixels=int(ground.sum()),
        road_shared_alpha_mean_before=float(before['road_alpha'][ground].mean()) if ground.any() else None,
        road_shared_alpha_mean_after=float(after['road_alpha'][ground].mean()) if ground.any() else None,
        changed_pixels_gt_1over255=int((difference.max(0)>RGB_LIMIT).sum()))


def cpu_self_check():
    h, w = 3, 4; before = dict(rgb=np.full((3, h, w), .4), alpha=np.ones((h, w)),
        K_alpha=np.ones((h, w)), road_alpha=np.full((h, w), .5))
    K = np.zeros((h, w), bool); K[0, 0] = True
    V = np.zeros((h, w), bool); V[0, 1] = True
    allowed = np.zeros((h, w), bool); allowed[1:] = True
    ground = allowed.copy(); after = {k: v.copy() for k, v in before.items()}; truth = before['rgb'].copy()
    after['rgb'][:, 1, 1] = .2
    assert protection_metrics(before, after, truth, K, V, allowed, ground)['passed']
    after['alpha'][1, 1] = .9
    assert not protection_metrics(before, after, truth, K, V, allowed, ground)['passed']
    after = {k: v.copy() for k, v in before.items()}; after['rgb'][0, 0, 0] += 2e-5
    assert not protection_metrics(before, after, truth, K, V, allowed, ground)['passed']
    return dict(passed=True, K_fixed_baseline_enforced=True, outside_domain_enforced=True,
                new_ground_holes_rejected=True, GPU_used=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-self-check', action='store_true')
    for key in ('runtime-dir', 'checkpoint', 'actor', 'init', 'manifest', 'source-manifest', 'prior-contract',
                'context-plan', 'path-map-file', 'roi-file', 'output', 'attribution-dir', 'candidate-file'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--teacher', type=Path); p.add_argument('--path-map', action='append', default=[])
    p.add_argument('--expected-actor-sha-prefix', default='ab4fa110'); p.add_argument('--top-rows', type=int, default=64)
    return p.parse_args()


class Cleanup:
    def __init__(self, args):
        if args.attribution_dir is None: raise ValueError('--attribution-dir is required')
        module_path = Path(__file__).with_name('source_attribution.py')
        spec = importlib.util.spec_from_file_location('_a20_readonly_source_attribution', module_path)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        self.audit = module.Audit(args); self.args = args; self.rt = self.audit.rt
        self.t = self.audit.t; self.core = self.audit.core; self.gpu = self.audit.gpu
        self.started = time.monotonic(); self.base_rows = {}; self.metadata = {}; self.native_gids = {}
        self.before = {}; self.actor_before = {}; self.road_alone = {}; self.domains = {}
        proof_path = args.attribution_dir/'proof.json'; proof = self.core.read(proof_path)
        if not proof.get('complete') or proof['actor']['sha256'] != self.core.sha(args.actor):
            raise ValueError('Attribution evidence binds a different actor or incomplete run')
        self.attribution_proof = self.core.ref(proof_path)
        self.road_path = self.rt.s.path(self.rt.i['native']['native_renderer']['road_root'])/'road.npz' if hasattr(self.rt, 'i') else self.rt.s.path(self.audit.i['native']['native_renderer']['road_root'])/'road.npz'
        self.road_hash = self.core.sha(self.road_path)
        with self.t.no_grad():
            for raw in self.core.BUILD:
                rows, metadata, layers, qa = self.audit.scene_rows(raw)
                features = self.t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]
                actor = metadata['actor_source_id'] >= 0; road = metadata['road_active_index'] >= 0
                protectedK = metadata['retained_other_actor'] | metadata['protected_native_K']
                for channel, mark in [(2, actor), (3, road), (4, protectedK)]:
                    features[:, channel] = self.t.as_tensor(mark, device='cuda', dtype=self.t.float32)
                self.base_rows[raw] = dict(rows, features=features); self.metadata[raw] = metadata
                self.native_gids[raw] = metadata['scene_global_row_id'][metadata['scene_global_row_id'] >= 0]
                if not np.all(np.diff(self.native_gids[raw]) > 0): raise ValueError('Native global IDs must retain their original packing order')
                self.before[raw] = self.render(raw, [])
                car = self.rt.render(raw=raw, actor_only=True)
                self.actor_before[raw] = {k: self.audit.arr(car[k]) for k in ('rgb', 'alpha', 'depth_sum_m')}
                only = self.audit.raster(raw, dict(rows, mask=rows['mask'] & self.t.as_tensor(road, device='cuda')), environment=False)
                self.road_alone[raw] = self.audit.arr(only['alpha'])
                self.domains[raw] = self.fixed_domains(raw, self.actor_before[raw]['alpha'])
                self.save_view(raw, 'before', self.before[raw])
                self.audit.save_image_set(raw, 'road_standalone', self.audit.arr(only['rgb']), self.road_alone[raw])
                dp = args.output/'fixed_domains'; dp.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dp/f'raw{raw:03d}.npz', **self.domains[raw], road_standalone_alpha=self.road_alone[raw],
                    actor_alpha=self.actor_before[raw]['alpha'], before_RGB=self.before[raw]['rgb'],
                    before_alpha=self.before[raw]['alpha'], before_K_alpha=self.before[raw]['K_alpha'])
                for key in ('allowed', 'K', 'V', 'ground'):
                    self.rt.m.png(dp/f'raw{raw:03d}_{key}.png', self.domains[raw][key])
        self.requested = load_candidate_ids(args.candidate_file)
        self.candidates, qualification = self.qualify_candidates(self.requested)
        self.core.write(args.output/'candidate_qualification.json', dict(requested=self.requested.tolist(),
            qualified=self.candidates.tolist(), rows=qualification, attribution=self.attribution_proof))
        self.core.write(args.output/'cleanup_protocol.json', dict(schema='a20.local_context_cleanup.protocol.v1',
            source_attribution=self.attribution_proof, actor=self.core.ref(args.actor), checkpoint=self.core.ref(args.checkpoint),
            script=self.core.ref(__file__), candidate_file=None if args.candidate_file is None else self.core.ref(args.candidate_file),
            requested_rows=len(self.requested), qualified_rows=len(self.candidates), seeds=SEEDS.tolist(),
            candidate_class='explicit global IDs; native static background/owner21; no retained other actor, no K1922, no actual K contribution',
            sidecar='global native row inactive flag / opacity multiplier0; no camera-specific selection',
            reference='one fixed original shared scene for every batch and split',
            allowed_domain='fixed label10 dilated8px union circled/strip/shadow union fixed actor-only alpha>.05, then excluding fixed K',
            ground_domain='allowed domain excluding fixed actor-only alpha>.05 and K; no newly alpha<.95 at previously alpha>=.95 pixels',
            thresholds=dict(K_RGB_max=K_LIMIT, K_alpha_max=K_LIMIT, outside_RGB_max=RGB_LIMIT, V_absolute_RGB_error_increment_max=RGB_LIMIT),
            actor_only_tolerance=0, actor_and_road_bytes_frozen=True, global_ID_sort=True,
            optimizer_updates=0, heldout_frames_opened=[], ROI_is_erase_GT=False))

    def fixed_domains(self, raw, actor_alpha):
        from PIL import Image
        item = self.audit.i['fixed']['real_K_labels'][f'{raw:03d}']
        labels = Image.open(self.rt.s.get(item, f'BUILD{raw} fixed original labels'))
        if labels.size != (1152, 768): labels = labels.resize((1152, 768), Image.Resampling.NEAREST)
        a9 = self.t.as_tensor(np.asarray(labels) == 10, device='cuda', dtype=self.t.float32)
        allowed = self.t.nn.functional.max_pool2d(a9[None, None], 17, 1, 8)[0, 0] > 0
        for name, mask in self.audit.domains(raw).items():
            if name in ('circled', 'strip', 'shadow'): allowed |= mask
        allowed |= self.t.as_tensor(actor_alpha > .05, device='cuda')
        K = self.rt.domains[raw]['K']; allowed &= ~K
        V = self.rt.domains[raw]['V']
        ground = allowed & ~self.t.as_tensor(actor_alpha > .05, device='cuda') & ~K
        return dict(allowed=self.audit.arr(allowed), K=self.audit.arr(K), V=self.audit.arr(V), ground=self.audit.arr(ground),
                    truth=self.audit.arr(self.rt.domains[raw]['rgb']))

    def render(self, raw, disabled):
        rows = self.base_rows[raw]; mask = rows['mask'].clone()
        if len(disabled):
            ids = np.asarray(disabled, np.int64); packed = self.native_gids[raw]
            position = np.searchsorted(packed, ids); good = position < len(packed)
            good[good] &= packed[position[good]] == ids[good]
            indices = position[good]
            meta = self.metadata[raw]
            if len(indices) and (np.any(meta['actor_source_id'][indices] >= 0) or np.any(meta['road_active_index'][indices] >= 0)):
                raise RuntimeError('Context sidecar would change a20 or road rows')
            mask[self.t.as_tensor(indices, device='cuda', dtype=self.t.long)] = False
        out = self.audit.raster(raw, dict(rows, mask=mask))
        return dict(rgb=self.audit.arr(out['rgb']), alpha=self.audit.arr(out['alpha']),
            actor_alpha=self.audit.arr(out['extra'][0]), road_alpha=self.audit.arr(out['extra'][1]), K_alpha=self.audit.arr(out['extra'][2]))

    def qualify_candidates(self, requested):
        status = {int(g): dict(global_row_id=int(g), seen=False, ROI_alpha_sum=0., V_alpha_sum=0., K_alpha_sum=0., reasons=[]) for g in requested}
        for raw in self.core.BUILD:
            path = self.args.attribution_dir/'evidence'/f'raw{raw:03d}_exact_row_contributions.npz'
            with np.load(path, allow_pickle=False) as z:
                gids = z['scene_global_row_id']; n = int((gids >= 0).sum()); packed = gids[:n]
                positions = np.searchsorted(packed, requested); present = positions < n
                present[present] &= packed[positions[present]] == requested[present]
                layer = z['layer_id']; owner = z['owner']; protected = z['retained_other_actor'] | z['protected_native_K']
                K = z['retained_K_alpha_raw']; V = z['real_V_alpha_raw']
                roi = np.zeros_like(K)
                for name in ('circled', 'strip', 'shadow'):
                    if name+'_alpha_raw' in z.files: roi += z[name+'_alpha_raw']
                for gid, j in zip(requested[present], positions[present]):
                    row = status[int(gid)]; row['seen'] = True
                    row['ROI_alpha_sum'] += float(roi[j]); row['V_alpha_sum'] += float(V[j]); row['K_alpha_sum'] += float(K[j])
                    if gid >= self.audit.ns or layer[j] not in (1, 6): row['reasons'].append(f'raw{raw}:not eligible native static background/owner21')
                    if protected[j] or (owner[j] > 0 and owner[j] != 21): row['reasons'].append(f'raw{raw}:retained other-actor/K source')
        qualified = []
        for gid in requested:
            row = status[int(gid)]
            if not row['seen']: row['reasons'].append('not present in BUILD context')
            if row['ROI_alpha_sum'] <= 0: row['reasons'].append('no actual diagnostic-ROI contribution')
            if row['K_alpha_sum'] > 0: row['reasons'].append('actual retained-K contribution is positive')
            row['qualified'] = not row['reasons']
            if row['qualified']: qualified.append(int(gid))
        return np.asarray(sorted(qualified), np.int64), list(status.values())

    def save_view(self, raw, name, result):
        self.audit.save_image_set(raw, name, result['rgb'], result['alpha'])
        self.audit.save_image_set(raw, name+'_road_shared_alpha', np.repeat(result['road_alpha'][None], 3, axis=0))
        self.audit.save_image_set(raw, name+'_a20_visible_alpha', np.repeat(result['actor_alpha'][None], 3, axis=0))
        path = self.args.output/'evaluation'; path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path/f'raw{raw:03d}_{name}.npz', **result)

    def assess(self, disabled):
        results = {}; reports = []
        with self.t.no_grad():
            for raw in self.core.BUILD:
                after = self.render(raw, disabled); results[raw] = after; d = self.domains[raw]
                reports.append(dict(raw=raw, **protection_metrics(self.before[raw], after, d['truth'], d['K'], d['V'], d['allowed'], d['ground'])))
        return all(r['passed'] for r in reports), reports, results

    def run(self):
        accepted = []; rejected = []; trials = 0
        stack = [(0, len(self.candidates), 0, None)] if len(self.candidates) else []
        with (self.args.output/'cleanup_trials.jsonl').open('w', encoding='utf-8') as file:
            while stack:
                lo, hi, depth, parent = stack.pop(); group = self.candidates[lo:hi].tolist()
                disabled = sorted(accepted+group); passed, reports, _ = self.assess(disabled)
                trials += 1; record = dict(trial=trials, parent=parent, depth=depth, global_row_ids=group,
                    whole_disabled_global_row_ids=disabled, accepted=passed, views=reports)
                file.write(json.dumps(record, allow_nan=False)+'\n'); file.flush()
                if passed: accepted.extend(group)
                elif hi-lo == 1: rejected.extend(group)
                else:
                    mid = lo+(hi-lo)//2; stack.append((mid, hi, depth+1, trials)); stack.append((lo, mid, depth+1, trials))
                self.core.write(self.args.output/'cleanup_progress.json', dict(PID=os.getpid(), trials=trials,
                    accepted_rows=len(accepted), rejected_rows=len(rejected), remaining_batches=len(stack), latest=record, optimizer_updates=0))
                print('LOCAL_CLEANUP_TRIAL', trials, len(group), passed, flush=True)
        accepted = np.asarray(sorted(accepted), np.int64); final_passed, final_reports, after = self.assess(accepted)
        if not final_passed: raise RuntimeError('Accepted whole-set sidecar does not satisfy fixed-original protection gates')
        sidecar_path = self.args.output/'context_inactive_sidecar.npz'
        np.savez_compressed(sidecar_path, native_global_row_ids=accepted, opacity_multiplier=np.zeros(len(accepted), np.float32),
            source_branch=np.zeros(len(accepted), np.int8), global_per_camera_selection=np.asarray(False))
        sidecar = self.core.bank(sidecar_path)
        if not np.array_equal(sidecar['native_global_row_ids'], accepted): raise RuntimeError('Sidecar ID round trip failed')
        actor_differences = []; reload_differences = []; road_exposure = []
        with self.t.no_grad():
            for raw in self.core.BUILD:
                # Apply only a derived context mask, never any source Gaussian/actor/road value.
                p, owner, road, gids = self.rt.context[raw]
                updated = dict(p, mask=p['mask'] & ~self.t.isin(gids, self.t.as_tensor(accepted, device='cuda')))
                self.rt.context[raw] = (updated, owner, road, gids)
                reloaded = self.rt.render(raw=raw)
                delta = {k: float(np.abs(self.audit.arr(reloaded[k])-after[raw][k]).max()) for k in ('rgb', 'alpha', 'actor_alpha', 'K_alpha')}
                reload_differences.append(dict(raw=raw, maximum=delta))
                if max(delta.values()) > 5e-6: raise RuntimeError('Sidecar-render reload parity failed')
                car = self.rt.render(raw=raw, actor_only=True)
                car_delta = {k: float(np.abs(self.audit.arr(car[k])-self.actor_before[raw][k]).max()) for k in ('rgb', 'alpha', 'depth_sum_m')}
                actor_differences.append(dict(raw=raw, maximum=car_delta))
                if any(v != 0 for v in car_delta.values()): raise RuntimeError('Frozen a20 isolated image/depth changed')
                self.save_view(raw, 'after', after[raw])
                diff = np.abs(after[raw]['rgb']-self.before[raw]['rgb']).max(0)
                self.audit.save_image_set(raw, 'before_after_difference', np.stack([np.clip(diff*5, 0, 1), np.clip(diff*2, 0, 1), np.zeros_like(diff)]))
                for name, mask_t in self.audit.domains(raw).items():
                    mask = self.audit.arr(mask_t); before = self.before[raw]; current = after[raw]
                    road_exposure.append(dict(raw=raw, domain=name, pixels=int(mask.sum()),
                        road_standalone_alpha95_fraction=float((self.road_alone[raw][mask]>=.95).mean()),
                        road_shared_alpha_mean_before=float(before['road_alpha'][mask].mean()), road_shared_alpha_mean_after=float(current['road_alpha'][mask].mean()),
                        opaque_road_hidden_before_pixels=int(((self.road_alone[raw]>=.95)&(before['road_alpha']<.05)&mask).sum()),
                        opaque_road_hidden_after_pixels=int(((self.road_alone[raw]>=.95)&(current['road_alpha']<.05)&mask).sum()),
                        total_alpha95_fraction_before=float((before['alpha'][mask]>=.95).mean()), total_alpha95_fraction_after=float((current['alpha'][mask]>=.95).mean()),
                        newly_alpha_below95_pixels=int(((before['alpha']>=.95)&(current['alpha']<.95)&mask).sum())))
        actor_unchanged = self.gpu.tree_equal(self.gpu.cpu(self.rt.actor.state_dict()), self.audit.actor_before)
        Adam_unchanged = self.gpu.tree_equal(self.gpu.cpu(self.audit.opt.state_dict()), self.audit.Adam_before)
        sources_unchanged = {str(p.resolve()): self.core.sha(p) == self.audit.source_hashes[str(p.resolve())] for p in self.audit.source_paths}
        road_bytes_unchanged = self.core.sha(self.road_path) == self.road_hash
        if not actor_unchanged or not Adam_unchanged or not road_bytes_unchanged or not all(sources_unchanged.values()):
            raise RuntimeError('Actor, road, source or Adam state changed during context cleanup')
        binding = dict(schema='a20.context_inactive_sidecar.v1', sidecar=self.core.ref(sidecar_path),
            actor=self.core.ref(self.args.actor), actor_checkpoint=self.core.ref(self.args.checkpoint),
            native_source_manifest=self.core.ref(self.args.source_manifest), original_road=self.core.ref(self.road_path),
            scope='global native static row IDs, applied identically to BUILD86/110; sealed frames unvalidated',
            renderer='reuse exact Runtime context with context_mask &= ~isin(global_row_id, sidecar_ids)',
            optimizer_updates=0, camera_specific_mask=False, actor_and_road_values_unchanged=True)
        self.core.write(self.args.output/'scene_context_binding.json', binding)
        self.core.write(self.args.output/'cleanup_ledger.json', dict(accepted_global_row_ids=accepted.tolist(),
            rejected_global_row_ids=sorted(rejected), requested_global_row_ids=self.requested.tolist(),
            trials=trials, fixed_original_gates=final_reports, road_exposure=road_exposure, optimizer_updates=0))
        self.core.write(self.args.output/'proof.json', dict(schema='a20.local_context_cleanup.proof.v1', complete=True,
            accepted_context_rows=len(accepted), accepted_global_row_ids=accepted.tolist(), rejected_rows=len(rejected),
            full_two_view_protection_passed=final_passed, gates=final_reports, actor_and_global_active_unchanged=actor_unchanged,
            actor_isolated_RGB_alpha_depth_exact_zero_change=actor_differences, actor_asset=self.core.ref(self.args.actor),
            road_asset_bytes_unchanged=road_bytes_unchanged, source_files_unchanged=sources_unchanged,
            Adam_unchanged=Adam_unchanged, context_sidecar=self.core.ref(sidecar_path), sidecar_reload_parity=reload_differences,
            road_exposure=road_exposure, optimizer_updates=0, other_actor_rows_retired=0, actor_rows_retired=0, road_rows_retired=0,
            no_static_wholesale_deletion=True, no_new_model=True, heldout_frames_opened=[], frozen_scene=self.rt.check_frozen(),
            visual_review='PENDING_ROOT_REVIEW', native_calls=self.rt.calls, elapsed_seconds=time.monotonic()-self.started,
            GPU_keep_on=True, power_operations=0))
        self.core.write(self.args.output/'resource_ledger.json', dict(native_calls=self.rt.calls, optimizer_updates=0,
            feature_VJP_calls=0, trials=trials, elapsed_seconds=time.monotonic()-self.started,
            cuda_peak_allocated_bytes=self.t.cuda.max_memory_allocated(), GPU_keep_on=True, power_operations=0))
        print('LOCAL_CONTEXT_CLEANUP_COMPLETE', json.dumps(dict(accepted=len(accepted), rejected=len(rejected), trials=trials)), flush=True)


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check: print(json.dumps(cpu_self_check(), indent=2))
    else:
        try: Cleanup(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'cleanup_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

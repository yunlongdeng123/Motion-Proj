"""Extend an accepted context sidecar without changing its original reference.

Same CLI as local_context_cleanup.py plus --initial-sidecar. The initial set is
verified first, retained in every trial, and excluded from candidate bisection.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import local_context_cleanup as base


class ExtendedCleanup(base.Cleanup):
    def __init__(self, args):
        super().__init__(args)
        path = args.initial_sidecar
        if path is None: raise ValueError('--initial-sidecar is required')
        sidecar = self.core.bank(path); ids = sidecar['native_global_row_ids']
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or ((ids < 0)|(ids >= self.audit.ns)).any():
            raise ValueError('Initial sidecar must contain strict native static global IDs')
        if len(np.unique(ids)) != len(ids): raise ValueError('Initial sidecar contains duplicate global IDs')
        if sidecar['opacity_multiplier'].shape != ids.shape or not (sidecar['opacity_multiplier'] == 0).all():
            raise ValueError('Only accepted inactive sidecars are supported')
        self.initial_ids = np.sort(ids.astype(np.int64, copy=False))
        binding_path = path.parent/'scene_context_binding.json'; proof_path = path.parent/'proof.json'
        binding = self.core.read(binding_path); proof = self.core.read(proof_path)
        if binding['sidecar']['sha256'] != self.core.sha(path): raise ValueError('Initial sidecar hash mismatch')
        if binding['actor']['sha256'] != self.core.sha(args.actor): raise ValueError('Initial sidecar binds another actor')
        if binding['actor_checkpoint']['sha256'] != self.core.sha(args.checkpoint): raise ValueError('Initial checkpoint differs')
        if binding['native_source_manifest']['sha256'] != self.core.sha(args.source_manifest): raise ValueError('Native source bindings differ')
        if binding['original_road']['sha256'] != self.road_hash: raise ValueError('Original road asset differs')
        if not proof.get('complete') or not proof.get('full_two_view_protection_passed'):
            raise ValueError('Initial sidecar lacks a completed passing two-view proof')
        if not np.array_equal(np.sort(np.asarray(proof['accepted_global_row_ids'], np.int64)), self.initial_ids):
            raise ValueError('Initial sidecar IDs and passing proof differ')
        if np.intersect1d(self.initial_ids, self.audit.arr(self.rt.K_preserve)).size:
            raise ValueError('Initial sidecar intersects native K1922')
        self.original_anchor = []; watched = [path, binding_path, proof_path]
        for raw in self.core.BUILD:
            # Anchor all cumulative decisions to the saved original scene, not
            # to either a freshly cleaned render or the previous accepted batch.
            before_path = path.parent/'evaluation'/f'raw{raw:03d}_before.npz'
            domains_path = path.parent/'fixed_domains'/f'raw{raw:03d}.npz'
            original = self.core.bank(before_path); original_domains = self.core.bank(domains_path)
            for key in ('allowed', 'K', 'V', 'ground', 'truth'):
                if not np.array_equal(original_domains[key], self.domains[raw][key]):
                    raise ValueError(f'Original fixed domain/truth changed: raw{raw} {key}')
            parity = {k: float(np.abs(original[k]-self.before[raw][k]).max()) for k in ('rgb', 'alpha', 'actor_alpha', 'road_alpha', 'K_alpha')}
            if max(parity.values()) > 5e-6: raise ValueError('Fresh original scene does not match saved original reference')
            self.before[raw] = original
            self.save_view(raw, 'before', original)
            # Future extensions inherit these same original arrays byte-for-byte.
            np.savez_compressed(args.output/'fixed_domains'/f'raw{raw:03d}.npz', **original_domains)
            self.original_anchor.append(dict(raw=raw, original_before=self.core.ref(before_path),
                original_domains=self.core.ref(domains_path), reconstruction_parity=parity))
            watched.extend([before_path, domains_path])
            p, owner, road, gids = self.rt.context[raw]
            mark = self.t.isin(gids, self.t.as_tensor(self.initial_ids, device='cuda'))
            if ((owner[mark] > 0)&(owner[mark] != 21)).any() or (road[mark] >= 0).any():
                raise ValueError('Initial sidecar would affect another actor or road')
        for p in watched:
            self.audit.source_paths.append(p)
            self.audit.source_hashes[str(p.resolve())] = self.core.sha(p)
        self.initial_passed, self.initial_reports, self.initial_render = self.assess(self.initial_ids)
        if not self.initial_passed: raise RuntimeError('Initial complete sidecar failed the fixed original-scene guard')
        for raw in self.core.BUILD: self.save_view(raw, 'initial', self.initial_render[raw])
        self.candidates = np.setdiff1d(self.candidates, self.initial_ids)
        self.core.write(args.output/'extension_protocol.json', dict(schema='a20.local_context_cleanup.extend.v1',
            initial_sidecar=self.core.ref(path), exact_initial_global_row_ids=self.initial_ids.tolist(),
            initial_sidecar_revalidated=True, initial_original_guards=self.initial_reports,
            original_scene_anchor=self.original_anchor, remaining_qualified_global_row_ids=self.candidates.tolist(),
            invariant='Every candidate trial includes all initial IDs; all guards use the saved original scene/domain/truth',
            source_candidate_file=None if args.candidate_file is None else self.core.ref(args.candidate_file),
            optimizer_updates=0, actor_and_road_unchanged=True, sealed_frames_opened=[]))

    def finish_extension_metadata(self):
        sidecar_path = self.args.output/'context_inactive_sidecar.npz'
        complete = self.core.bank(sidecar_path)['native_global_row_ids']
        if not np.isin(self.initial_ids, complete).all(): raise RuntimeError('An inherited inactive ID was lost')
        additional = np.setdiff1d(complete, self.initial_ids)
        payload = dict(initial_sidecar=self.core.ref(self.args.initial_sidecar),
            inherited_context_rows=len(self.initial_ids), inherited_global_row_ids=self.initial_ids.tolist(),
            newly_accepted_context_rows=len(additional), newly_accepted_global_row_ids=additional.tolist(),
            complete_context_rows=len(complete), original_scene_anchor=self.original_anchor,
            initial_original_guard_passed=True, initial_original_guards=self.initial_reports,
            baseline_shifted=False, optimizer_updates=0)
        for name in ('scene_context_binding.json', 'proof.json', 'cleanup_ledger.json'):
            value = self.core.read(self.args.output/name); value['extension'] = payload
            self.core.write(self.args.output/name, value)
        changes = []
        with self.t.no_grad():
            for raw in self.core.BUILD:
                final = self.render(raw, complete); initial = self.initial_render[raw]
                delta = np.abs(final['rgb']-initial['rgb']).max(0)
                self.audit.save_image_set(raw, 'initial_to_after_difference', np.stack([
                    np.clip(delta*5, 0, 1), np.clip(delta*2, 0, 1), np.zeros_like(delta)]))
                changes.append(dict(raw=raw, RGB_change_max=float(delta.max()), RGB_change_mean=float(delta.mean())))
        self.core.write(self.args.output/'extension_completion.json', dict(**payload, completed=True,
            context_sidecar=self.core.ref(sidecar_path), initial_to_final=changes))
        print('LOCAL_CONTEXT_EXTENSION_COMPLETE', json.dumps(dict(inherited=len(self.initial_ids), additional=len(additional), total=len(complete))), flush=True)


# The frozen r1 run method is copied below with only the initial accepted set
# changed and a final extension metadata call. Its guard/sidecar code is intact.

    def run(self):
        accepted = self.initial_ids.tolist(); rejected = []; trials = 0
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

        self.finish_extension_metadata()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--initial-sidecar', type=Path, required=True)
    initial, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    args = base.parse_args(); args.initial_sidecar = initial.initial_sidecar
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check: print(json.dumps(base.cpu_self_check(), indent=2))
    else:
        try: ExtendedCleanup(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'cleanup_extension_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

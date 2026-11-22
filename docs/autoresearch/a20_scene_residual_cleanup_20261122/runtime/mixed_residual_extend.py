"""Explicit <=512 residual IDs over a proved mixed sidecar; no source edits.

Allowed new rows are native static background, or native dynamic owner8/14.
All current-query and original V/K contributions must be exactly zero.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
import numpy as np
import local_context_cleanup as base
import local_context_cleanup_extend as extension
import local_dynamic_stray_cleanup as dynamic
import cold_reload_cleanup as cold


def explicit_ids(path):
    keys = ('global_row_ids', 'static_global_row_ids', 'dynamic_global_row_ids', 'dynamic')
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as z: values = [z[k].copy() for k in keys if k in z.files]
    else:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        values = [value[k] for k in keys if k in value] if isinstance(value, dict) else [value]
    if not values: raise ValueError('Explicit candidate file needs global_row_ids or dynamic_global_row_ids')
    ids = np.unique(np.concatenate([cold.id_array(np.asarray(v), 'explicit residual IDs') for v in values]))
    if len(ids) > 512: raise ValueError('Residual extension is limited to <=512 explicit query IDs')
    return ids


def allowed_row(gid, layer, owner, protected, ns, nd):
    return not protected and ((0 <= gid < ns and layer == 6 and owner <= 0) or
        (ns <= gid < ns+nd and layer == 5 and owner in (8, 14)))


def cpu_self_check():
    assert allowed_row(2, 6, -1, False, 10, 10)
    assert allowed_row(12, 5, 8, False, 10, 10) and allowed_row(13, 5, 14, False, 10, 10)
    assert not allowed_row(12, 5, 7, False, 10, 10)
    assert not allowed_row(2, 4, 8, False, 10, 10)
    assert not allowed_row(2, 6, -1, True, 10, 10)
    return dict(passed=True, only_static_background_or_dynamic_owner8_14=True, K1922_protected=True, GPU_used=False)


class MixedResidualExtend(base.Cleanup):
    run = extension.ExtendedCleanup.run
    owner_visible_fields = dynamic.DynamicStrayCleanup.owner_visible_fields

    def __init__(self, args):
        if args.initial_sidecar is None or args.candidate_file is None: raise ValueError('--initial-sidecar and --candidate-file are required')
        self.explicit_ids = explicit_ids(args.candidate_file); self.allowed_owners = {8, 14}
        super().__init__(args)
        self.requested = self.explicit_ids
        path = args.initial_sidecar; parent = path.parent
        binding_path = parent/'scene_context_binding.json'; proof_path = parent/'proof.json'
        binding = self.core.read(binding_path); proof = self.core.read(proof_path)
        self.initial_ids, self.initial_branches, _ = cold.validate_sidecar(self.core.bank(path), binding, proof, self.audit.ns, self.audit.nd)
        for key, actual in [('sidecar', path), ('actor', args.actor), ('actor_checkpoint', args.checkpoint),
                            ('native_source_manifest', args.source_manifest), ('original_road', self.road_path)]:
            if binding[key]['sha256'] != self.core.sha(actual): raise ValueError('Initial mixed binding mismatch: '+key)
        if np.intersect1d(self.initial_ids, self.audit.arr(self.rt.K_preserve)).size: raise ValueError('Initial sidecar intersects K1922')
        query_dir = args.query_dir or args.candidate_file.parent
        protocol_path = next((query_dir/k for k in ('rgb_query_protocol.json', 'mixed_residual_query_protocol.json', 'dark_query_protocol.json') if (query_dir/k).is_file()), None)
        top_path = next((query_dir/k for k in ('top_rgb_residual_rows.json', 'top_mixed_residual_rows.json', 'top_dark_static_rows.json') if (query_dir/k).is_file()), None)
        if protocol_path is None or top_path is None: raise ValueError('Bound query protocol and top-row evidence are required')
        query_proof_path = query_dir/'proof.json'; q = self.core.read(protocol_path); qp = self.core.read(query_proof_path)
        if not qp.get('complete') or qp.get('newly_retired_context_rows') != 0: raise ValueError('Query must be complete and diagnostic-only')
        if q['sidecar']['sha256'] != self.core.sha(path) or q['actor']['sha256'] != self.core.sha(args.actor) or q['road']['sha256'] != self.road_hash:
            raise ValueError('Query does not bind this exact initial sidecar, actor and original road')
        ranked = {int(v['global_row_id']): v for v in self.core.read(top_path)['rows']}
        for gid in self.explicit_ids:
            row = ranked.get(int(gid))
            if row is None or not row.get('current_two_view_V_K_exact_zero') or row.get('cleanup_eligible', row.get('legacy_cleanup_eligible')) is not True:
                raise ValueError('Candidate lacks current/original V/K qualification: '+str(gid))
            if row.get('current_V_alpha_sum') != 0 or row.get('current_K_alpha_sum') != 0 or row.get('original_V_alpha_sum') != 0 or row.get('original_K_alpha_sum') != 0:
                raise ValueError('Candidate has nonzero current/original V/K support: '+str(gid))
        self.candidates = np.intersect1d(np.setdiff1d(self.candidates, self.initial_ids), self.explicit_ids)
        self.original_anchor = []; watched = [path, binding_path, proof_path, protocol_path, query_proof_path, top_path, args.candidate_file]
        for raw in self.core.BUILD:
            before_path = parent/'evaluation'/f'raw{raw:03d}_before.npz'; domain_path = parent/'fixed_domains'/f'raw{raw:03d}.npz'
            original = self.core.bank(before_path); domains = self.core.bank(domain_path)
            for key in ('allowed', 'K', 'V', 'ground', 'truth'):
                if not np.array_equal(domains[key], self.domains[raw][key]): raise ValueError('Original mask/GT changed: '+key)
            parity = {k: float(np.abs(original[k]-self.before[raw][k]).max()) for k in ('rgb', 'alpha', 'actor_alpha', 'road_alpha', 'K_alpha')}
            if max(parity.values()) > 5e-6: raise ValueError('Fresh original scene differs from the fixed original reference')
            self.before[raw] = original; self.save_view(raw, 'before', original)
            np.savez_compressed(args.output/'fixed_domains'/f'raw{raw:03d}.npz', **domains)
            self.original_anchor.append(dict(raw=raw, original_before=self.core.ref(before_path), original_domains=self.core.ref(domain_path), reconstruction_parity=parity))
            watched.extend([before_path, domain_path])
        for p in watched:
            self.audit.source_paths.append(p); self.audit.source_hashes[str(p.resolve())] = self.core.sha(p)
        self.initial_passed, self.initial_reports, self.initial_render = self.assess(self.initial_ids)
        if not self.initial_passed: raise RuntimeError('The inherited mixed sidecar failed the fixed original-scene guard')
        for raw in self.core.BUILD: self.save_view(raw, 'initial', self.initial_render[raw])
        self.dynamic_owner_original = self.owner_visible_fields([])
        self.core.write(args.output/'mixed_residual_extension_protocol.json', dict(schema='a20.explicit_mixed_residual_extend.v1',
            script=self.core.ref(__file__), initial_sidecar=self.core.ref(path), inherited_global_row_ids=self.initial_ids.tolist(),
            inherited_static_rows=int((self.initial_branches==0).sum()), inherited_dynamic_rows=int((self.initial_branches==1).sum()),
            initial_original_guards=self.initial_reports, qualified_explicit_ids=self.candidates.tolist(),
            bound_query=self.core.ref(protocol_path), original_scene_anchor=self.original_anchor,
            candidate_scope='native static background OR explicit dynamic owner8/14; not whole owners',
            current_and_original_V_K_exact_zero=True, fixed_original_baseline=True, candidate_budget=512,
            optimizer_updates=0, actor_and_road_unchanged=True, heldout_frames_opened=[]))

    def qualify_candidates(self, requested):
        requested = self.explicit_ids
        status = {int(g): dict(global_row_id=int(g), seen=False, ROI_alpha_sum=0., V_alpha_sum=0., K_alpha_sum=0., reasons=[]) for g in requested}
        for raw in self.core.BUILD:
            path = self.args.attribution_dir/'evidence'/f'raw{raw:03d}_exact_row_contributions.npz'
            with np.load(path, allow_pickle=False) as z:
                gids = z['scene_global_row_id']; n = int((gids>=0).sum()); pos = np.searchsorted(gids[:n], requested)
                present = pos < n; present[present] &= gids[pos[present]] == requested[present]
                layer = z['layer_id']; owner = z['owner']; protected = z['protected_native_K']
                V = z['real_V_alpha_raw']; K = z['retained_K_alpha_raw']; roi = np.zeros_like(V)
                for name in ('circled', 'strip', 'shadow'):
                    if name+'_alpha_raw' in z.files: roi += z[name+'_alpha_raw']
                for gid, j in zip(requested[present], pos[present]):
                    row = status[int(gid)]; row['seen'] = True
                    row['ROI_alpha_sum'] += float(roi[j]); row['V_alpha_sum'] += float(V[j]); row['K_alpha_sum'] += float(K[j])
                    if not allowed_row(int(gid), int(layer[j]), int(owner[j]), bool(protected[j]), self.audit.ns, self.audit.nd):
                        row['reasons'].append(f'raw{raw}:outside explicit static-background/dynamic8,14 scope or K1922')
        good = []
        for gid in requested:
            row = status[int(gid)]
            if not row['seen'] or row['ROI_alpha_sum'] <= 0: row['reasons'].append('no original BUILD ROI contribution')
            if row['V_alpha_sum'] != 0 or row['K_alpha_sum'] != 0: row['reasons'].append('original actual V/K contribution is nonzero')
            row['qualified'] = not row['reasons']
            if row['qualified']: good.append(int(gid))
        return np.asarray(sorted(good), np.int64), list(status.values())

    def finish_extension_metadata(self):
        path = self.args.output/'context_inactive_sidecar.npz'; sidecar = self.core.bank(path)
        complete = sidecar['native_global_row_ids']; branch = (complete >= self.audit.ns).astype(np.int8)
        additional = np.setdiff1d(complete, self.initial_ids); added_dynamic = additional[additional >= self.audit.ns]; added_static = additional[additional < self.audit.ns]
        if not np.isin(self.initial_ids, complete).all(): raise RuntimeError('Inherited IDs were lost')
        if not np.isin(additional, self.explicit_ids).all() or (complete >= self.audit.ns+self.audit.nd).any(): raise RuntimeError('An unrequested/non-native row was introduced')
        np.savez_compressed(path, native_global_row_ids=complete, opacity_multiplier=sidecar['opacity_multiplier'], source_branch=branch,
            source_row_id=complete-branch.astype(np.int64)*self.audit.ns, global_per_camera_selection=np.asarray(False))
        ref = self.core.ref(path); binding = self.core.read(self.args.output/'scene_context_binding.json')
        binding.update(schema='a20.native_context_visibility_sidecar.v2', sidecar=ref,
            scope='inherited mixed cleanup plus explicit query-verified static background/dynamic owner8,14 IDs; BUILD86/110',
            source_branch_convention={'0':'native static', '1':'native dynamic'}, whole_actor_cleanup=False, other_actor_source_parameters_unchanged=True)
        self.core.write(self.args.output/'scene_context_binding.json', binding)
        proof = self.core.read(self.args.output/'proof.json'); proof['context_sidecar'] = ref; self.core.write(self.args.output/'proof.json', proof)
        extension.ExtendedCleanup.finish_extension_metadata(self)
        after_fields = self.owner_visible_fields(complete); comparisons = []; arrays = {}
        for raw in self.core.BUILD:
            for owner in (8, 14):
                before = self.dynamic_owner_original[raw][owner]; after = after_fields[raw][owner]; mask = before['visible_mask']; n = int(mask.sum())
                rgb = float(np.abs(after['rgb']-before['rgb'])[:, mask].max()) if n else None; alpha = float(np.abs(after['alpha']-before['alpha'])[mask].max()) if n else None
                comparisons.append(dict(raw=raw, owner=owner, fixed_visible_pixels=n, applicability='APPLICABLE' if n else 'N/A',
                    visible_scene_RGB_change_max=rgb, visible_owner_alpha_change_max=alpha, exact_zero_change=(rgb==0 and alpha==0) if n else None))
                arrays[f'raw{raw:03d}_owner{owner}_visible_mask'] = mask
                arrays[f'raw{raw:03d}_owner{owner}_alpha_before'] = before['alpha']; arrays[f'raw{raw:03d}_owner{owner}_alpha_after'] = after['alpha']
        evidence = self.args.output/'evaluation/retained_owner_visible_alpha_before_after.npz'; np.savez_compressed(evidence, **arrays)
        proof = self.core.read(self.args.output/'proof.json'); dyn = complete[branch==1]
        K_exact = all(g['K_RGB_max']==0 and g['K_visible_alpha_max']==0 for g in proof['gates']); owner_exact = all(v['exact_zero_change'] for v in comparisons if v['fixed_visible_pixels'])
        proof.update(new_static_context_rows_retired=len(added_static), new_static_global_row_ids=added_static.tolist(),
            newly_added_dynamic_context_rows=len(added_dynamic), newly_added_dynamic_global_row_ids=added_dynamic.tolist(),
            cumulative_static_context_rows=int((branch==0).sum()), cumulative_dynamic_context_rows=len(dyn),
            inherited_dynamic_context_rows=int((self.initial_branches==1).sum()), new_dynamic_global_row_ids=dyn.tolist(), dynamic_source_branch=1,
            other_actor_stray_rows_retired=len(dyn), other_actor_rows_retired=len(dyn), new_other_actor_rows_retired=len(added_dynamic), whole_actor_deletions=0,
            other_actor_source_files_and_parameters_unchanged=True, explicitly_checked_owner_visible_masks=comparisons,
            checked_owner_masks_exact_zero=owner_exact, retained_full_K_RGB_alpha_exact_zero_change=K_exact, owner_alpha_evidence=self.core.ref(evidence))
        self.core.write(self.args.output/'proof.json', proof)
        cold.validate_sidecar(self.core.bank(path), binding, proof, self.audit.ns, self.audit.nd)
        self.core.write(self.args.output/'mixed_residual_completion.json', dict(completed=True, inherited_rows=len(self.initial_ids),
            new_static_rows=len(added_static), new_dynamic_rows=len(added_dynamic), complete_rows=len(complete),
            full_K_exact_zero=K_exact, owner8_14_exact_zero=owner_exact, original_baseline_shifted=False,
            mixed_sidecar_cold_schema_validated=True, sidecar=ref, optimizer_updates=0))
        print('MIXED_RESIDUAL_EXTENSION_COMPLETE', len(added_static), 'static', len(added_dynamic), 'dynamic;', len(complete), 'total', flush=True)


def parse_args():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--initial-sidecar', type=Path); p.add_argument('--query-dir', type=Path)
    extra, remaining = p.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = base.parse_args(); args.initial_sidecar = extra.initial_sidecar; args.query_dir = extra.query_dir
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check:
        print(json.dumps(dict(cpu_self_check(), guards=base.cpu_self_check(), mixed_sidecar=cold.cpu_self_check(),
            parsed_init=None if args.init is None else str(args.init), parsed_initial_sidecar=None if args.initial_sidecar is None else str(args.initial_sidecar)), indent=2))
    else:
        try: MixedResidualExtend(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'mixed_residual_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(), optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

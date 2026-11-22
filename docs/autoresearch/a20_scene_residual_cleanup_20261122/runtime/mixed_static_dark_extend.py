"""Extend an accepted mixed sidecar using only the bound dark-query static IDs.

No new cleanup engine: reuse r1 construction/guards and the r2 cumulative run.
Same CLI as r1 plus --initial-sidecar and optional --query-dir (candidate parent).
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


class MixedStaticDarkExtend(base.Cleanup):
    run = extension.ExtendedCleanup.run
    owner_visible_fields = dynamic.DynamicStrayCleanup.owner_visible_fields

    def __init__(self, args):
        if args.initial_sidecar is None or args.candidate_file is None:
            raise ValueError('--initial-sidecar and explicit dark-query --candidate-file are required')
        self.allowed_owners = {8, 14}
        super().__init__(args)
        path = args.initial_sidecar; parent = path.parent
        binding_path = parent/'scene_context_binding.json'; proof_path = parent/'proof.json'
        binding = self.core.read(binding_path); proof = self.core.read(proof_path); sidecar = self.core.bank(path)
        self.initial_ids, self.initial_branches, _ = cold.validate_sidecar(sidecar, binding, proof, self.audit.ns, self.audit.nd)
        for key, actual in [('sidecar', path), ('actor', args.actor), ('actor_checkpoint', args.checkpoint),
                            ('native_source_manifest', args.source_manifest), ('original_road', self.road_path)]:
            if binding[key]['sha256'] != self.core.sha(actual): raise ValueError('Initial mixed binding mismatch: '+key)
        if np.intersect1d(self.initial_ids, self.audit.arr(self.rt.K_preserve)).size: raise ValueError('Initial sidecar intersects K1922')
        query_dir = args.query_dir or args.candidate_file.parent
        query_protocol_path = query_dir/'dark_query_protocol.json'; query_proof_path = query_dir/'proof.json'
        query_protocol = self.core.read(query_protocol_path); query_proof = self.core.read(query_proof_path)
        if not query_proof.get('complete') or query_proof.get('newly_retired_context_rows') != 0:
            raise ValueError('Expected a complete diagnostic-only dark query')
        if query_protocol['sidecar']['sha256'] != self.core.sha(path) or query_protocol['actor']['sha256'] != self.core.sha(args.actor):
            raise ValueError('Dark query was not performed on this exact mixed sidecar and actor')
        if query_protocol['road']['sha256'] != self.road_hash: raise ValueError('Dark query road reference differs')
        compact = self.core.read(args.candidate_file)['global_row_ids']
        compact = cold.id_array(np.asarray(compact), 'explicit dark-query candidate IDs')
        if len(compact) > 256: raise ValueError('This extension accepts only the <=256 ranked dark-query compact IDs')
        top_path = query_dir/'top_dark_static_rows.json'
        ranked = {int(v['global_row_id']): v for v in self.core.read(top_path)['rows']}
        for gid in compact:
            row = ranked.get(int(gid))
            if row is None or not row['current_two_view_V_K_exact_zero'] or row.get('legacy_cleanup_eligible') is not True:
                raise ValueError('Dark candidate lacks current and original source qualification: '+str(gid))
        self.candidates = np.intersect1d(np.setdiff1d(self.candidates, self.initial_ids), compact)
        for raw in self.core.BUILD:
            packed = self.native_gids[raw]; pos = np.searchsorted(packed, self.candidates); present = pos < len(packed)
            present[present] &= packed[pos[present]] == self.candidates[present]
            if np.any(self.metadata[raw]['layer_id'][pos[present]] != 6):
                raise ValueError('Dark query compact IDs contain a non-static-background current row')
        self.original_anchor = []; watched = [path, binding_path, proof_path, query_protocol_path, query_proof_path, top_path, args.candidate_file]
        for raw in self.core.BUILD:
            before_path = parent/'evaluation'/f'raw{raw:03d}_before.npz'; domain_path = parent/'fixed_domains'/f'raw{raw:03d}.npz'
            original = self.core.bank(before_path); domains = self.core.bank(domain_path)
            for key in ('allowed', 'K', 'V', 'ground', 'truth'):
                if not np.array_equal(domains[key], self.domains[raw][key]): raise ValueError('Original mask/GT changed: '+key)
            parity = {k: float(np.abs(original[k]-self.before[raw][k]).max()) for k in ('rgb', 'alpha', 'actor_alpha', 'road_alpha', 'K_alpha')}
            if max(parity.values()) > 5e-6: raise ValueError('Fresh original scene does not match the fixed original reference')
            self.before[raw] = original; self.save_view(raw, 'before', original)
            np.savez_compressed(args.output/'fixed_domains'/f'raw{raw:03d}.npz', **domains)
            self.original_anchor.append(dict(raw=raw, original_before=self.core.ref(before_path), original_domains=self.core.ref(domain_path), reconstruction_parity=parity))
            watched.extend([before_path, domain_path])
        for p in watched:
            self.audit.source_paths.append(p); self.audit.source_hashes[str(p.resolve())] = self.core.sha(p)
        self.initial_passed, self.initial_reports, self.initial_render = self.assess(self.initial_ids)
        if not self.initial_passed: raise RuntimeError('The complete inherited mixed sidecar failed original-scene guards')
        for raw in self.core.BUILD: self.save_view(raw, 'initial', self.initial_render[raw])
        self.dynamic_owner_original = self.owner_visible_fields([])
        self.core.write(args.output/'mixed_static_extension_protocol.json', dict(
            schema='a20.mixed_init_static_dark_extend.v1', script=self.core.ref(__file__),
            initial_sidecar=self.core.ref(path), inherited_global_row_ids=self.initial_ids.tolist(),
            inherited_static_rows=int((self.initial_branches==0).sum()), inherited_dynamic_rows=int((self.initial_branches==1).sum()),
            initial_original_guards=self.initial_reports, initial_mixed_cold_validation=True,
            bound_dark_query=self.core.ref(query_protocol_path), qualified_new_static_ids=self.candidates.tolist(),
            original_scene_anchor=self.original_anchor, fixed_original_baseline=True,
            optimizer_updates=0, actor_and_road_values_unchanged=True, heldout_frames_opened=[]))

    def finish_extension_metadata(self):
        path = self.args.output/'context_inactive_sidecar.npz'; sidecar = self.core.bank(path)
        complete = sidecar['native_global_row_ids']; branch = (complete >= self.audit.ns).astype(np.int8)
        additional = np.setdiff1d(complete, self.initial_ids)
        if (additional >= self.audit.ns).any(): raise RuntimeError('Static dark extension introduced a dynamic/road row')
        if not np.isin(self.initial_ids, complete).all(): raise RuntimeError('Inherited mixed IDs were lost')
        np.savez_compressed(path, native_global_row_ids=complete, opacity_multiplier=sidecar['opacity_multiplier'],
            source_branch=branch, source_row_id=complete-branch.astype(np.int64)*self.audit.ns,
            global_per_camera_selection=np.asarray(False))
        ref = self.core.ref(path)
        binding = self.core.read(self.args.output/'scene_context_binding.json')
        binding.update(schema='a20.native_context_visibility_sidecar.v2', sidecar=ref,
            scope='inherited proved mixed sidecar plus explicit bound dark-query static background IDs; BUILD86/110',
            source_branch_convention={'0':'native static', '1':'native dynamic'}, whole_actor_cleanup=False,
            other_actor_source_parameters_unchanged=True)
        self.core.write(self.args.output/'scene_context_binding.json', binding)
        proof = self.core.read(self.args.output/'proof.json'); proof['context_sidecar'] = ref
        self.core.write(self.args.output/'proof.json', proof)
        extension.ExtendedCleanup.finish_extension_metadata(self)
        after_fields = self.owner_visible_fields(complete); comparisons = []; arrays = {}
        for raw in self.core.BUILD:
            for owner in (8, 14):
                before = self.dynamic_owner_original[raw][owner]; after = after_fields[raw][owner]; mask = before['visible_mask']; n = int(mask.sum())
                rgb = float(np.abs(after['rgb']-before['rgb'])[:, mask].max()) if n else None
                alpha = float(np.abs(after['alpha']-before['alpha'])[mask].max()) if n else None
                comparisons.append(dict(raw=raw, owner=owner, fixed_visible_pixels=n,
                    applicability='APPLICABLE' if n else 'N/A', visible_scene_RGB_change_max=rgb,
                    visible_owner_alpha_change_max=alpha, exact_zero_change=(rgb==0 and alpha==0) if n else None))
                arrays[f'raw{raw:03d}_owner{owner}_visible_mask'] = mask
                arrays[f'raw{raw:03d}_owner{owner}_alpha_before'] = before['alpha']
                arrays[f'raw{raw:03d}_owner{owner}_alpha_after'] = after['alpha']
        evidence = self.args.output/'evaluation/retained_owner_visible_alpha_before_after.npz'; np.savez_compressed(evidence, **arrays)
        proof = self.core.read(self.args.output/'proof.json')
        K_exact = all(g['K_RGB_max']==0 and g['K_visible_alpha_max']==0 for g in proof['gates'])
        owner_exact = all(v['exact_zero_change'] for v in comparisons if v['fixed_visible_pixels'])
        dynamic_ids = complete[branch==1]
        proof.update(new_static_context_rows_retired=len(additional), new_static_global_row_ids=additional.tolist(),
            cumulative_static_context_rows=int((branch==0).sum()), inherited_dynamic_context_rows=len(dynamic_ids),
            new_dynamic_global_row_ids=dynamic_ids.tolist(), dynamic_source_branch=1,
            other_actor_stray_rows_retired=len(dynamic_ids), other_actor_rows_retired=len(dynamic_ids),
            new_other_actor_rows_retired=0, whole_actor_deletions=0,
            other_actor_source_files_and_parameters_unchanged=True, explicitly_checked_owner_visible_masks=comparisons,
            checked_owner_masks_exact_zero=owner_exact, retained_full_K_RGB_alpha_exact_zero_change=K_exact,
            owner_alpha_evidence=self.core.ref(evidence))
        self.core.write(self.args.output/'proof.json', proof)
        # Ensure this new complete artifact remains accepted by the existing cold reloader.
        cold.validate_sidecar(self.core.bank(path), binding, proof, self.audit.ns, self.audit.nd)
        self.core.write(self.args.output/'mixed_static_extension_completion.json', dict(completed=True,
            inherited_rows=len(self.initial_ids), newly_accepted_static_rows=len(additional), complete_rows=len(complete),
            preserved_dynamic_rows=len(dynamic_ids), full_K_exact_zero=K_exact, owner8_14_exact_zero=owner_exact,
            mixed_sidecar_cold_schema_validated=True, context_sidecar=ref, optimizer_updates=0))
        print('MIXED_STATIC_DARK_EXTENSION_COMPLETE', len(additional), 'new static rows;', len(complete), 'total', flush=True)


def parse_args():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--initial-sidecar', type=Path); p.add_argument('--query-dir', type=Path)
    extra, remaining = p.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = base.parse_args(); args.initial_sidecar = extra.initial_sidecar; args.query_dir = extra.query_dir
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check:
        print(json.dumps(dict(base.cpu_self_check(), mixed_sidecar=cold.cpu_self_check(),
            parsed_init=None if args.init is None else str(args.init),
            parsed_initial_sidecar=None if args.initial_sidecar is None else str(args.initial_sidecar)), indent=2))
    else:
        try: MixedStaticDarkExtend(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'mixed_static_extension_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

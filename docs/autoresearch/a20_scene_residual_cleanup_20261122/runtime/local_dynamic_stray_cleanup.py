"""Explicit dynamic stray pilot over an accepted static context sidecar.

Same CLI as local_context_cleanup_extend.py, plus --allowed-owners (default8,14).
Only explicit dynamic global IDs with exact two-view V/K contribution0 qualify.
No other-actor source parameter or checkpoint is edited; only a derived global
visibility sidecar changes. No whole actor candidate is generated.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
import numpy as np
import local_context_cleanup_extend as previous


def classify_dynamic_candidate(global_id, layer, owner, protected_K, V, K, ns, nd, allowed):
    reasons = []
    if not (ns <= global_id < ns+nd) or layer != 5:
        reasons.append('not native_dynamic_other_actor by global topology and actual row layer')
    if owner not in allowed: reasons.append('owner is not explicitly permitted for this tiny stray pilot')
    if protected_K: reasons.append('native K1922 protected source')
    if V != 0 or K != 0: reasons.append('actual true-V or retained-K contribution is not exactly zero')
    return reasons


def cpu_self_check():
    assert not classify_dynamic_candidate(12, 5, 8, False, 0., 0., 10, 10, {8, 14})
    assert classify_dynamic_candidate(12, 5, 8, False, 0., 1e-12, 10, 10, {8, 14})
    assert classify_dynamic_candidate(2, 4, 8, False, 0., 0., 10, 10, {8, 14})
    assert classify_dynamic_candidate(12, 5, 8, True, 0., 0., 10, 10, {8, 14})
    assert classify_dynamic_candidate(12, 5, 7, False, 0., 0., 10, 10, {8, 14})
    return dict(passed=True, topology_checked=True, exact_V_K_zero_required=True,
                K1922_protected=True, explicit_owner_scope=True, GPU_used=False)


class DynamicStrayCleanup(previous.ExtendedCleanup):
    def __init__(self, args):
        if args.candidate_file is None: raise ValueError('An explicit tiny --candidate-file global_row_ids JSON/NPZ is required')
        self.allowed_owners = set(args.allowed_owners)
        super().__init__(args)
        self.dynamic_owner_original = self.owner_visible_fields([])
        self.core.write(args.output/'dynamic_stray_protocol.json', dict(
            schema='a20.explicit_dynamic_stray_cleanup.v1', script=self.core.ref(__file__),
            initial_sidecar=self.core.ref(args.initial_sidecar), explicit_candidates=self.core.ref(args.candidate_file),
            allowed_owners=sorted(self.allowed_owners), qualified_global_row_ids=self.candidates.tolist(),
            source_branch=1, dynamic_global_interval=[self.audit.ns, self.audit.ns+self.audit.nd],
            actual_two_view_V_K_contribution='exactly zero, per present row; absent-view contribution0 recorded',
            preserved='original complete scene baseline, all K masks, a20 values, road values and native checkpoint bytes',
            derived_only='global visibility sidecar for explicit stray rows; not a whole retained actor deletion',
            optimizer_updates=0, heldout_frames_opened=[]))

    def qualify_candidates(self, requested):
        rows = {int(g): dict(global_row_id=int(g), source_branch=1, seen=False,
            ROI_alpha_sum=0., V_alpha_sum=0., K_alpha_sum=0., reasons=[], per_view=[]) for g in requested}
        for raw in self.core.BUILD:
            path = self.args.attribution_dir/'evidence'/f'raw{raw:03d}_exact_row_contributions.npz'
            with np.load(path, allow_pickle=False) as z:
                gids = z['scene_global_row_id']; n = int((gids >= 0).sum()); packed = gids[:n]
                positions = np.searchsorted(packed, requested); present = positions < n
                present[present] &= packed[positions[present]] == requested[present]
                layer = z['layer_id']; owner = z['owner']; protected = z['protected_native_K']
                V = z['real_V_alpha_raw']; K = z['retained_K_alpha_raw']; roi = np.zeros_like(K)
                for name in ('circled', 'strip', 'shadow'):
                    if name+'_alpha_raw' in z.files: roi += z[name+'_alpha_raw']
                for gid, j in zip(requested[present], positions[present]):
                    item = rows[int(gid)]; item['seen'] = True
                    item['ROI_alpha_sum'] += float(roi[j]); item['V_alpha_sum'] += float(V[j]); item['K_alpha_sum'] += float(K[j])
                    reasons = classify_dynamic_candidate(int(gid), int(layer[j]), int(owner[j]), bool(protected[j]),
                        float(V[j]), float(K[j]), self.audit.ns, self.audit.nd, self.allowed_owners)
                    item['reasons'].extend([f'raw{raw}: '+v for v in reasons])
                    item['per_view'].append(dict(raw=raw, present=True, owner=int(owner[j]),
                        layer_id=int(layer[j]), V_alpha_sum=float(V[j]), K_alpha_sum=float(K[j]), ROI_alpha_sum=float(roi[j])))
                for gid in requested[~present]:
                    rows[int(gid)]['per_view'].append(dict(raw=raw, present=False, V_alpha_sum=0., K_alpha_sum=0.))
        qualified = []
        for gid in requested:
            item = rows[int(gid)]
            if not item['seen']: item['reasons'].append('not present in either BUILD native context')
            if item['ROI_alpha_sum'] <= 0: item['reasons'].append('no actual diagnostic-ROI contribution')
            item['qualified'] = not item['reasons']
            if item['qualified']: qualified.append(int(gid))
        return np.asarray(sorted(qualified), np.int64), list(rows.values())

    def owner_visible_fields(self, disabled):
        """Actual shared-sort owner alpha; masks are fixed categorical labels."""
        from PIL import Image
        fields = {}
        with self.t.no_grad():
            for raw in self.core.BUILD:
                rows = self.base_rows[raw]; meta = self.metadata[raw]; mask = rows['mask'].clone()
                if len(disabled):
                    packed = self.native_gids[raw]; ids = np.asarray(disabled, np.int64)
                    pos = np.searchsorted(packed, ids); present = pos < len(packed)
                    present[present] &= packed[pos[present]] == ids[present]
                    mask[self.t.as_tensor(pos[present], device='cuda', dtype=self.t.long)] = False
                ref = self.audit.i['fixed']['real_K_labels'][f'{raw:03d}']
                image = Image.open(self.rt.s.get(ref, f'fixed visible other-actor labels {raw}'))
                if image.size != (1152, 768): image = image.resize((1152, 768), Image.Resampling.NEAREST)
                labels = np.asarray(image); fields[raw] = {}
                for owner in sorted(self.allowed_owners):
                    features = self.t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]
                    features[:, 2] = self.t.as_tensor(meta['owner'] == owner, device='cuda', dtype=self.t.float32)
                    out = self.audit.raster(raw, dict(rows, features=features, mask=mask))
                    fields[raw][owner] = dict(rgb=self.audit.arr(out['rgb']), alpha=self.audit.arr(out['extra'][0]),
                        visible_mask=(labels == owner)&self.domains[raw]['K'])
        return fields

    def finish_extension_metadata(self):
        # The inherited renderer/guards already tested the complete static+dynamic
        # set. Correct the newly written derived sidecar provenance before binding.
        path = self.args.output/'context_inactive_sidecar.npz'; old = self.core.bank(path)
        complete = old['native_global_row_ids']; dynamic = complete >= self.audit.ns
        if ((complete < 0)|(complete >= self.audit.ns+self.audit.nd)).any(): raise RuntimeError('Sidecar contains a non-native row')
        additional = np.setdiff1d(complete, self.initial_ids)
        if len(additional) and ((additional < self.audit.ns)|(additional >= self.audit.ns+self.audit.nd)).any():
            raise RuntimeError('Dynamic pilot unexpectedly introduced a static or road deletion')
        np.savez_compressed(path, native_global_row_ids=complete, opacity_multiplier=old['opacity_multiplier'],
            source_branch=dynamic.astype(np.int8), source_row_id=np.where(dynamic, complete-self.audit.ns, complete),
            global_per_camera_selection=np.asarray(False))
        binding = self.core.read(self.args.output/'scene_context_binding.json')
        binding.update(schema='a20.native_context_visibility_sidecar.v2', sidecar=self.core.ref(path),
            scope='inherited static cleanup plus explicit source-verified dynamic stray rows; BUILD86/110 only',
            source_branch_convention={'0':'native static', '1':'native dynamic'},
            whole_actor_cleanup=False, other_actor_source_parameters_unchanged=True)
        self.core.write(self.args.output/'scene_context_binding.json', binding)
        proof = self.core.read(self.args.output/'proof.json'); proof['context_sidecar'] = self.core.ref(path)
        self.core.write(self.args.output/'proof.json', proof)
        super().finish_extension_metadata()
        fields = self.owner_visible_fields(complete); comparisons = []; arrays = {}
        for raw in self.core.BUILD:
            for owner in sorted(self.allowed_owners):
                before = self.dynamic_owner_original[raw][owner]; after = fields[raw][owner]; visible = before['visible_mask']
                n = int(visible.sum())
                rgb_max = float(np.abs(after['rgb']-before['rgb'])[:, visible].max()) if n else None
                alpha_max = float(np.abs(after['alpha']-before['alpha'])[visible].max()) if n else None
                comparisons.append(dict(raw=raw, owner=owner, fixed_visible_pixels=n,
                    applicability='APPLICABLE' if n else 'N/A: owner has no fixed visible mask in this frame',
                    visible_scene_RGB_change_max=rgb_max, visible_owner_alpha_change_max=alpha_max,
                    exact_zero_change=(rgb_max == 0 and alpha_max == 0) if n else None))
                arrays[f'raw{raw:03d}_owner{owner}_visible_mask'] = visible
                arrays[f'raw{raw:03d}_owner{owner}_alpha_before'] = before['alpha']
                arrays[f'raw{raw:03d}_owner{owner}_alpha_after'] = after['alpha']
        evidence_path = self.args.output/'evaluation/retained_owner_visible_alpha_before_after.npz'
        np.savez_compressed(evidence_path, **arrays)
        proof = self.core.read(self.args.output/'proof.json')
        original_K_exact = all(g['K_RGB_max'] == 0 and g['K_visible_alpha_max'] == 0 for g in proof['gates'])
        owner_exact = all(v['exact_zero_change'] for v in comparisons if v['fixed_visible_pixels'])
        proof.update(other_actor_stray_rows_retired=len(additional), other_actor_rows_retired=len(additional),
            whole_actor_deletions=0, inherited_static_context_rows=len(self.initial_ids),
            new_dynamic_global_row_ids=additional.tolist(), dynamic_source_branch=1,
            other_actor_source_files_and_parameters_unchanged=True,
            retained_full_K_RGB_alpha_exact_zero_change=original_K_exact,
            explicitly_checked_owner_visible_masks=comparisons, checked_owner_masks_exact_zero=owner_exact,
            owner_alpha_evidence=self.core.ref(evidence_path))
        self.core.write(self.args.output/'proof.json', proof)
        ledger = self.core.read(self.args.output/'cleanup_ledger.json')
        ledger['dynamic_stray'] = dict(new_dynamic_global_row_ids=additional.tolist(), source_branch=1,
            allowed_owners=sorted(self.allowed_owners), actual_V_K_zero_qualification='candidate_qualification.json',
            retained_visible_comparisons=comparisons)
        self.core.write(self.args.output/'cleanup_ledger.json', ledger)
        print('DYNAMIC_STRAY_PILOT_COMPLETE', json.dumps(dict(inherited=len(self.initial_ids), dynamic_strays=len(additional),
            full_K_exact_zero=original_K_exact, owner_masks_exact_zero=owner_exact)), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--allowed-owners', default='8,14')
    parser.add_argument('--initial-sidecar', type=Path)
    own, remaining = parser.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = previous.base.parse_args(); args.initial_sidecar = own.initial_sidecar
    args.allowed_owners = [int(v) for v in own.allowed_owners.split(',')]
    if not args.cpu_self_check and args.initial_sidecar is None:
        raise ValueError('--initial-sidecar is required for GPU cleanup')
    if not args.allowed_owners or any(v in (10, 21) or v <= 0 for v in args.allowed_owners):
        raise ValueError('Explicit retained other-actor owners are required')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check:
        print(json.dumps(dict(cpu_self_check(), parsed_init=None if args.init is None else str(args.init),
            parsed_initial_sidecar=None if args.initial_sidecar is None else str(args.initial_sidecar)), indent=2))
    else:
        try: DynamicStrayCleanup(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'dynamic_stray_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

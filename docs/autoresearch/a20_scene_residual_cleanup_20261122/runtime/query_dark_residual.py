"""Diagnostic only: actual road-float dark residual -> static background VJP.

Same runtime/actor CLI as source_attribution.py, plus --cleanup-sidecar and
optional --attribution-dir for original ROI/V/K qualification evidence.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import source_attribution as source

LUMA = (0.2126, 0.7152, 0.0722)


def dark_row_score(gA, gRef, source_luma):
    reference = np.divide(gRef, gA, out=np.zeros_like(gRef), where=gA > 0)
    gap = reference-source_luma
    return (gA > .05) & (gap > .08), reference, gap, gRef-gA*source_luma


def cpu_self_check():
    valid, ref, gap, mass = dark_row_score(np.array([1., 1., .01]), np.array([.5, .05, .01]), np.array([.2, .03, .1]))
    assert np.array_equal(valid, [True, False, False])
    assert gap[0] > .08 and mass[0] > 0
    return dict(passed=True, weighted_float_reference=True, dark_reference_does_not_nominate_dark_source=True,
                minimum_actual_contribution_checked=True, GPU_used=False)


def parse_args():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--cleanup-sidecar', type=Path)
    p.add_argument('--attribution-dir', type=Path)
    extra, remaining = p.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = source.parse_args(); args.cleanup_sidecar = extra.cleanup_sidecar; args.attribution_dir = extra.attribution_dir
    if not args.cpu_self_check and args.cleanup_sidecar is None: raise ValueError('--cleanup-sidecar is required')
    return args


class DarkQuery:
    def __init__(self, args):
        self.audit = source.Audit(args); self.args = args; self.rt = self.audit.rt
        self.t = self.audit.t; self.core = self.audit.core; self.gpu = self.audit.gpu
        self.started = time.monotonic(); self.vjp_calls = 0; self.results = {}; self.by_id = {}
        sidecar = self.core.bank(args.cleanup_sidecar); ids = sidecar['native_global_row_ids']
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or ((ids < 0)|(ids >= self.audit.ns+self.audit.nd)).any():
            raise ValueError('Mixed sidecar needs native static/dynamic global IDs only')
        if sidecar['opacity_multiplier'].shape != ids.shape or not (sidecar['opacity_multiplier'] == 0).all():
            raise ValueError('Expected the accepted inactive visibility sidecar')
        branch = (ids >= self.audit.ns).astype(np.int8)
        if 'source_branch' in sidecar and not np.array_equal(sidecar['source_branch'], branch):
            raise ValueError('Sidecar static/dynamic branch labels do not match topology')
        binding_path = args.cleanup_sidecar.parent/'scene_context_binding.json'; binding = self.core.read(binding_path)
        if binding['sidecar']['sha256'] != self.core.sha(args.cleanup_sidecar): raise ValueError('Sidecar SHA mismatch')
        if binding['actor']['sha256'] != self.core.sha(args.actor): raise ValueError('Sidecar actor mismatch')
        road_path = self.rt.s.path(self.audit.i['native']['native_renderer']['road_root'])/'road.npz'
        if binding['original_road']['sha256'] != self.core.sha(road_path): raise ValueError('Road differs from the accepted original road; no inactive donor patch is permitted')
        if np.intersect1d(ids, self.audit.arr(self.rt.K_preserve)).size: raise ValueError('Sidecar intersects protected native K1922')
        for raw in self.core.BUILD:
            p, owner, road, gids = self.rt.context[raw]
            remove = self.t.isin(gids, self.t.as_tensor(ids, device='cuda'))
            if (road[remove] >= 0).any(): raise ValueError('Sidecar includes road rows')
            self.rt.context[raw] = (dict(p, mask=p['mask']&~remove), owner, road, gids)
        for path in (args.cleanup_sidecar, binding_path, road_path):
            self.audit.source_paths.append(path); self.audit.source_hashes[str(path.resolve())] = self.core.sha(path)
        self.core.write(args.output/'dark_query_protocol.json', dict(schema='a20.dark_residual_query.v1',
            script=self.core.ref(__file__), actor=self.core.ref(args.actor), sidecar=self.core.ref(args.cleanup_sidecar),
            inherited_context_rows=len(ids), road=self.core.ref(road_path),
            diagnostic_domain='fixed shadow/circled/strip union; non-K; frozen actor-only alpha<.05; actual road-only alpha>=.95',
            dark_mask='actual road-only RGB/alpha luma minus current shared luma >.08',
            ranking='same-sort unit-feature gA and road-reference-luma-weighted gRef; deficit=gRef-gA*source RGB luma',
            candidate_gate='native_static_background; gA>.05 and gRef/gA-source_luma>.08 in at least one BUILD view; both-view current V/K0; not K1922 or another actor',
            dark_pixels_are_erase_GT=False, automatic_cleanup=False, optimizer_updates=0,
            original_attribution=None if args.attribution_dir is None else str(args.attribution_dir),
            inactive_unknown_road_donors_used=False, heldout_frames_opened=[]))

    def query_view(self, raw):
        t = self.t; rows, meta, layers, qa = self.audit.scene_rows(raw)
        domains = self.audit.domains(raw); roi = t.zeros((768, 1152), device='cuda', dtype=t.bool)
        for name in ('shadow', 'circled', 'strip'):
            if name in domains: roi |= domains[name]
        with t.no_grad():
            is_road = t.as_tensor(meta['road_active_index'] >= 0, device='cuda')
            road = self.audit.raster(raw, dict(rows, mask=rows['mask']&is_road), environment=False)
            actor = self.rt.render(raw=raw, actor_only=True)
            road_stable = road['alpha'] >= .95
            road_RGB = t.zeros_like(road['rgb'])
            road_RGB[:, road_stable] = road['rgb'][:, road_stable]/road['alpha'][road_stable]
            luma = t.tensor(LUMA, device='cuda')[:, None, None]
            road_luma = (road_RGB*luma).sum(0)
            reliable = roi & ~domains['retained_K'] & (actor['alpha'] < .05) & road_stable
        weights = t.ones(len(rows['means3D']), device='cuda', requires_grad=True)
        features = t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]; features[:, 2] = weights
        out = self.audit.raster(raw, dict(rows, features=features))
        shared_luma = (out['rgb'].detach()*luma).sum(0)
        dark = reliable & ((road_luma-shared_luma) > .08)
        bright = reliable & ((shared_luma-road_luma) > .08)

        def vjp(domain, pixel_weight=None):
            if not domain.any(): return np.zeros(len(weights), np.float32)
            values = out['extra'][0][domain]
            if pixel_weight is not None: values = values*pixel_weight[domain]
            g = t.autograd.grad(values.sum(), weights, retain_graph=True)[0]
            if not t.isfinite(g).all() or (g < -1e-7).any(): raise RuntimeError('Invalid actual unit-feature VJP')
            self.vjp_calls += 1
            return self.audit.arr(g).clip(min=0)

        gA = vjp(dark); gRef = vjp(dark, road_luma)
        gV = vjp(domains['real_V']); gK = vjp(domains['retained_K']); gROI = vjp(roi)
        RGB = self.audit.arr(rows['colors_precomp']); source_luma = RGB@np.asarray(LUMA, np.float32)
        row_ok, reference_luma, gap, deficit = dark_row_score(gA, gRef, source_luma)
        static_background = (meta['layer_id'] == 6)&(meta['scene_global_row_id'] >= 0)&(meta['scene_global_row_id'] < self.audit.ns)
        static_background &= ~meta['protected_native_K'] & ~meta['retained_other_actor'] & meta['actual_row_mask']
        # Keep per-row support in both views, including rows only dark in one.
        ids = np.flatnonzero(static_background & (gA > 0))
        for j in ids.tolist():
            gid = int(meta['scene_global_row_id'][j])
            item = self.by_id.setdefault(gid, dict(global_row_id=gid, source_branch=0, owner=int(meta['owner'][j]),
                weighted_dark_deficit=0., current_V_alpha_sum=0., current_K_alpha_sum=0., nominated=False, per_view=[]))
            item['weighted_dark_deficit'] += max(0., float(deficit[j])); item['current_V_alpha_sum'] += float(gV[j]); item['current_K_alpha_sum'] += float(gK[j])
            item['nominated'] |= bool(row_ok[j])
            if gA[j] > 0 or row_ok[j]:
                item['per_view'].append(dict(raw=raw, gA=float(gA[j]), gRef=float(gRef[j]),
                    road_reference_luma_weighted=float(reference_luma[j]), source_RGB=RGB[j].tolist(),
                    source_RGB_luma=float(source_luma[j]), weighted_luma_gap=float(gap[j]),
                    weighted_dark_deficit=float(deficit[j]), current_ROI_alpha_sum=float(gROI[j]),
                    actual_V_alpha_sum=float(gV[j]), actual_K_alpha_sum=float(gK[j]), qualifies_in_view=bool(row_ok[j])))
        self.results[raw] = dict(meta=meta, gV=gV, gK=gK, gROI=gROI)
        folder = self.args.output/'evidence'; folder.mkdir(parents=True, exist_ok=True)
        evidence_path = folder/f'raw{raw:03d}_dark_source_vjp.npz'
        np.savez_compressed(evidence_path, scene_global_row_id=meta['scene_global_row_id'], layer_id=meta['layer_id'],
            owner=meta['owner'], protected_native_K=meta['protected_native_K'], actual_row_mask=meta['actual_row_mask'],
            gA=gA, gRef=gRef, source_RGB_luma=source_luma, weighted_reference_luma=reference_luma,
            weighted_luma_gap=gap, weighted_dark_deficit=deficit, real_V_alpha_raw=gV, retained_K_alpha_raw=gK,
            diagnostic_ROI_alpha_raw=gROI, candidate_in_view=row_ok&static_background)
        shared_RGB = self.audit.arr(out['rgb']); road_ref = self.audit.arr(road_RGB)
        reliable_np = self.audit.arr(reliable); dark_np = self.audit.arr(dark)
        np.savez_compressed(folder/f'raw{raw:03d}_actual_float_reference.npz', shared_RGB=shared_RGB,
            shared_alpha=self.audit.arr(out['alpha']), actor_alpha=self.audit.arr(actor['alpha']),
            road_RGB_premultiplied=self.audit.arr(road['rgb']), road_alpha=self.audit.arr(road['alpha']),
            road_RGB_normalized_stable=road_ref, road_luma=self.audit.arr(road_luma), reliable_ground=reliable_np,
            dark_mask=dark_np, bright_mask=self.audit.arr(bright), ROI_union=self.audit.arr(roi))
        self.audit.save_image_set(raw, 'current_shared', shared_RGB, self.audit.arr(out['alpha']))
        self.audit.save_image_set(raw, 'actual_road_reference', road_ref, self.audit.arr(road['alpha']))
        self.audit.save_image_set(raw, 'fixed_dark_mask', np.repeat(dark_np[None].astype(np.float32), 3, 0))
        overlay = shared_RGB.copy(); overlay[:, dark_np] = np.array([1., .15, .1])[:, None]
        self.audit.save_image_set(raw, 'dark_pixel_locations', overlay)
        per_roi = {}
        for name, mask in domains.items():
            if name not in ('shadow', 'circled', 'strip'): continue
            per_roi[name] = dict(reliable_ground_pixels=int((mask&reliable).sum()), dark_pixels=int((mask&dark).sum()),
                                bright_pixels=int((mask&bright).sum()))
        report = dict(raw=raw, reliable_ground_pixels=int(reliable.sum()), dark_pixels=int(dark.sum()),
            bright_pixels=int(bright.sum()), per_ROI=per_roi,
            road_luma_mean=float(road_luma[reliable].mean()) if reliable.any() else None,
            shared_luma_mean=float(shared_luma[reliable].mean()) if reliable.any() else None,
            chroma_gap_mean=float(np.abs((shared_RGB-self.audit.arr(shared_luma)[None])-(road_ref-self.audit.arr(road_luma)[None]))[:, reliable_np].mean()) if reliable.any() else None,
            camera_and_row_QA=qa, actual_evidence=self.core.ref(evidence_path))
        self.core.write(self.args.output/f'raw{raw:03d}_dark_metrics.json', report)
        print('DARK_QUERY_VIEW_COMPLETE', raw, json.dumps(per_roi), flush=True)
        return report

    def finish_candidates(self):
        # Fetch the other view's support even if a row only entered the ranking later.
        for gid, item in self.by_id.items():
            V = 0.; K = 0.; current_ROI = 0.
            for raw, values in self.results.items():
                g = values['meta']['scene_global_row_id']; n = int((g >= 0).sum()); pos = int(np.searchsorted(g[:n], gid))
                if pos < n and g[pos] == gid:
                    V += float(values['gV'][pos]); K += float(values['gK'][pos]); current_ROI += float(values['gROI'][pos])
            item.update(current_V_alpha_sum=V, current_K_alpha_sum=K, current_ROI_alpha_sum=current_ROI,
                        current_two_view_V_K_exact_zero=V == 0 and K == 0)
        ranked = [v for v in self.by_id.values() if v['nominated'] and v['current_two_view_V_K_exact_zero']]
        ranked.sort(key=lambda v: (-v['weighted_dark_deficit'], v['global_row_id']))
        top = ranked[:256]
        if self.args.attribution_dir is not None and top:
            lookup = {v['global_row_id']: v for v in top}
            for item in top: item.update(original_ROI_alpha_sum=0., original_V_alpha_sum=0., original_K_alpha_sum=0.)
            wanted = np.asarray(sorted(lookup), np.int64)
            for raw in self.core.BUILD:
                path = self.args.attribution_dir/'evidence'/f'raw{raw:03d}_exact_row_contributions.npz'
                with np.load(path, allow_pickle=False) as z:
                    g = z['scene_global_row_id']; n = int((g >= 0).sum()); pos = np.searchsorted(g[:n], wanted)
                    present = pos < n; present[present] &= g[pos[present]] == wanted[present]
                    V = z['real_V_alpha_raw']; K = z['retained_K_alpha_raw']; roi = np.zeros_like(V)
                    for name in ('shadow', 'circled', 'strip'):
                        if name+'_alpha_raw' in z.files: roi += z[name+'_alpha_raw']
                    for gid, j in zip(wanted[present], pos[present]):
                        item = lookup[int(gid)]; item['original_ROI_alpha_sum'] += float(roi[j])
                        item['original_V_alpha_sum'] += float(V[j]); item['original_K_alpha_sum'] += float(K[j])
            for item in top:
                item['legacy_cleanup_eligible'] = item['original_ROI_alpha_sum'] > 0 and item['original_V_alpha_sum'] == 0 and item['original_K_alpha_sum'] == 0
        else:
            for item in top: item['legacy_cleanup_eligible'] = None
        compact = [v['global_row_id'] for v in top if v['legacy_cleanup_eligible'] is not False]
        self.core.write(self.args.output/'top_dark_static_rows.json', dict(diagnostic_only=True, rows=top,
            total_current_candidates=len(ranked), top_limit=256, no_cleanup_applied=True))
        self.core.write(self.args.output/'candidate_ids_dark_static.json', dict(global_row_ids=compact,
            diagnostic_only=True, criterion='actual weighted road/source luma deficit with fixed dark pixels; current two-view V/K0; original ROI/V/K qualification when available'))
        return top, compact, len(ranked)

    def run(self):
        views = [self.query_view(raw) for raw in self.core.BUILD]
        top, compact, total = self.finish_candidates()
        unchanged = self.gpu.tree_equal(self.gpu.cpu(self.rt.actor.state_dict()), self.audit.actor_before)
        Adam_unchanged = self.gpu.tree_equal(self.gpu.cpu(self.audit.opt.state_dict()), self.audit.Adam_before)
        sources = {str(p.resolve()): self.core.sha(p) == self.audit.source_hashes[str(p.resolve())] for p in self.audit.source_paths}
        if not unchanged or not Adam_unchanged or not all(sources.values()): raise RuntimeError('A frozen source changed during dark diagnostic')
        self.core.write(self.args.output/'proof.json', dict(schema='a20.dark_residual_query.proof.v1', complete=True,
            actual_road_float_reference=True, safe_road_normalization_alpha_minimum=.95,
            actual_shared_sort_feature_VJP=True, actor_and_active_unchanged=unchanged, Adam_unchanged=Adam_unchanged,
            source_files_unchanged=sources, frozen_scene=self.rt.check_frozen(),
            dark_pixels_are_cleanup_GT=False, unknown_inactive_road_donors_used=False,
            optimizer_updates=0, newly_retired_context_rows=0, total_diagnostic_candidates=total,
            compact_candidate_rows=len(compact), top_rows=len(top), views=views,
            native_calls=self.rt.calls, feature_VJP_calls=self.vjp_calls, heldout_frames_opened=[],
            elapsed_seconds=time.monotonic()-self.started, GPU_keep_on=True, power_operations=0))
        print('DARK_RESIDUAL_QUERY_COMPLETE', len(compact), 'compact candidates; no cleanup', flush=True)


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check: print(json.dumps(cpu_self_check(), indent=2))
    else:
        try: DarkQuery(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'dark_query_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

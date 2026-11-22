"""Frozen BUILD86/110 RGB residual attribution; no cleanup or optimization.

Reuses source_attribution.py's exact actor/native runtime and CLI. Extra flags:
--cleanup-sidecar, --attribution-dir (original evidence directory),
--rgb-top-rows {256,512}. The unchanged road-only float reference is diagnostic.
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

LUMA = np.asarray((.2126, .7152, .0722), np.float32)
WATCH_IDS = (541991, 566431)


def rgb_row_score(gA, gE, gTE, source_RGB):
    """gE and gTE are signed, detached-pixel-weight unit-feature VJPs."""
    score = np.einsum('ij,ij->i', source_RGB, gE)-gTE
    return (gA > .05) & (score > 0), score


def row_positions(gids, wanted):
    """Join actual scene IDs, never actor lineage IDs or semantic owners."""
    native = np.flatnonzero(gids >= 0)
    values = gids[native]
    if len(values) > 1 and np.any(values[1:] <= values[:-1]):
        order = np.argsort(values, kind='stable'); values = values[order]; native = native[order]
        if np.any(values[1:] == values[:-1]): raise ValueError('Duplicate actual scene global IDs')
    pos = np.searchsorted(values, wanted); present = pos < len(values)
    present[present] &= values[pos[present]] == wanted[present]
    result = np.full(len(wanted), -1, np.int64); result[present] = native[pos[present]]
    return result


def allowed_source(layer, owner, branch):
    return ((layer == 6) & (branch == 0)) | ((layer == 5) & (branch == 1) & np.isin(owner, [8, 14]))


def cpu_self_check():
    # Direct alpha*T sum agrees with signed VJP formula, including purple/bright
    # residual with little luma contrast. Negative weighted gradients stay signed.
    alpha = np.array([[.2, .1, .4], [.5, .3, .2]], np.float32)
    colors = np.array([[.8, .2, .9], [.1, .2, .1]], np.float32)
    target = np.array([[.2, .3, .1], [.1, .3, .2], [.1, .3, .2]], np.float32)
    residual = np.array([[.3, -.1, .1], [-.1, .1, -.2], [.4, -.2, .2]], np.float32)
    gA = alpha.sum(1); gE = alpha@residual.T
    gTE = alpha@np.sum(target*residual, axis=0)
    valid, score = rgb_row_score(gA, gE, gTE, colors)
    direct = np.sum(alpha[:, None, :]*(colors[:, :, None]-target[None])*residual[None], axis=(1, 2))
    assert np.allclose(score, direct, atol=1e-7)
    assert (gE < 0).any() and valid[0]
    assert np.array_equal(row_positions(np.array([7, 2, -1, 4]), np.array([2, 7, 9])), [1, 0, -1])
    assert np.array_equal(allowed_source(np.array([6, 5, 5, 0, 8]), np.array([0, 8, 21, 0, 21]),
                                        np.array([0, 1, 1, 2, 3])), [True, True, False, False, False])
    return dict(passed=True, signed_RGB_VJP_matches_direct_sum=True,
                native_ID_join_independent_of_packet_order=True, source_scope_checked=True, GPU_used=False)


def parse_args():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--cleanup-sidecar', type=Path)
    p.add_argument('--attribution-dir', type=Path)
    p.add_argument('--rgb-top-rows', type=int, choices=(256, 512), default=256)
    extra, remaining = p.parse_known_args(); sys.argv = [sys.argv[0]]+remaining
    args = source.parse_args()
    for key, value in vars(extra).items(): setattr(args, key, value)
    if not args.cpu_self_check and (args.cleanup_sidecar is None or args.attribution_dir is None):
        raise ValueError('--cleanup-sidecar and --attribution-dir are required; original V/K evidence is mandatory')
    return args


class RGBQuery:
    def __init__(self, args):
        self.audit = source.Audit(args); self.args = args; self.rt = self.audit.rt
        self.t = self.audit.t; self.core = self.audit.core; self.gpu = self.audit.gpu
        self.started = time.monotonic(); self.vjp_calls = 0; self.results = {}; self.by_id = {}
        sidecar = self.core.bank(args.cleanup_sidecar); ids = sidecar['native_global_row_ids']
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or ((ids < 0)|(ids >= self.audit.ns+self.audit.nd)).any():
            raise ValueError('Accepted mixed sidecar must contain only native static/dynamic global IDs')
        if len(np.unique(ids)) != len(ids): raise ValueError('Duplicate sidecar IDs')
        if sidecar['opacity_multiplier'].shape != ids.shape or not (sidecar['opacity_multiplier'] == 0).all():
            raise ValueError('Expected an accepted inactive visibility sidecar')
        branch = (ids >= self.audit.ns).astype(np.int8)
        if 'source_branch' not in sidecar or not np.array_equal(sidecar['source_branch'], branch):
            raise ValueError('Sidecar source branch disagrees with native topology')
        binding_path = args.cleanup_sidecar.parent/'scene_context_binding.json'; binding = self.core.read(binding_path)
        for key, path in [('sidecar', args.cleanup_sidecar), ('actor', args.actor),
                          ('actor_checkpoint', args.checkpoint), ('native_source_manifest', args.source_manifest)]:
            if binding[key]['sha256'] != self.core.sha(path): raise ValueError('Accepted sidecar '+key+' SHA mismatch')
        road_path = self.rt.s.path(self.audit.i['native']['native_renderer']['road_root'])/'road.npz'
        if binding['original_road']['sha256'] != self.core.sha(road_path):
            raise ValueError('Road differs from accepted unchanged parent; no road patch is permitted')
        if np.intersect1d(ids, self.audit.arr(self.rt.K_preserve)).size:
            raise ValueError('Sidecar intersects protected native K1922')
        for raw in self.core.BUILD:
            packet, owner, road, gids = self.rt.context[raw]
            remove = self.t.isin(gids, self.t.as_tensor(ids, device='cuda'))
            if (road[remove] >= 0).any(): raise ValueError('Sidecar includes road rows')
            self.rt.context[raw] = (dict(packet, mask=packet['mask']&~remove), owner, road, gids)
        for path in (args.cleanup_sidecar, binding_path, road_path): self.track(path)
        self.core.write(args.output/'rgb_query_protocol.json', dict(schema='a20.rgb_residual_query.v1',
            script=self.core.ref(__file__), actor=self.core.ref(args.actor), sidecar=self.core.ref(args.cleanup_sidecar),
            inherited_context_rows=len(ids), road=self.core.ref(road_path), original_attribution=str(args.attribution_dir),
            diagnostic_domain='fixed shadow/circled/strip union AND non-K AND frozen actor-only alpha<.05 AND unchanged road-only alpha>=.95',
            residual_mask='max_channel(abs(current_shared_RGB-road_RGB_normalized_stable))>.08',
            VJP='gA=sum(alpha*T); gE[c]=sum(alpha*T*(R[c]-T[c])); gTE=sum(alpha*T*dot(T,R-T)); score=dot(source_RGB,gE)-gTE',
            signed_pixel_weights_and_scores=True, ranking='sum of positive per-view alignment scores; signed total also recorded',
            candidate_gate='gA>.05 and score>0 in at least one BUILD view; native static background or native dynamic other actor owner8/14; both current and original two-view V/K exact0; nonK1922',
            top_limit=args.rgb_top_rows, withheld_support_rows_ranked=True,
            road_reference_is_GT=False, actual_row_deletion_effect_not_assumed=True,
            automatic_cleanup=False, optimizer_updates=0, inactive_road_donors_used=False, heldout_frames_opened=[]))

    def track(self, path):
        path = Path(path)
        if str(path.resolve()) not in self.audit.source_hashes:
            self.audit.source_paths.append(path); self.audit.source_hashes[str(path.resolve())] = self.core.sha(path)

    def layer_images(self, raw, rows, layer_ids, shared_RGB, shared_alpha, roi, residual_domain, residual):
        t = self.t; contributions = {}; parity = []
        with t.no_grad():
            for start in range(0, len(source.LAYERS), 5):
                features = t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]
                for j, layer in enumerate(range(start, min(start+5, len(source.LAYERS)))):
                    mark = (layer_ids == layer).float()
                    features[:, 2+4*j] = mark; features[:, 3+4*j:6+4*j] = rows['colors_precomp']*mark[:, None]
                out = self.audit.raster(raw, dict(rows, features=features))
                rgb_error = float(np.abs(self.audit.arr(out['rgb'])-shared_RGB).max())
                alpha_error = float(np.abs(self.audit.arr(out['alpha'])-shared_alpha).max())
                if rgb_error > 5e-6 or alpha_error > 1e-7: raise RuntimeError('Same-sort layer features changed scene RGB/alpha')
                parity.append(dict(batch_start=start, RGB_max=rgb_error, alpha_max=alpha_error))
                for j, layer in enumerate(range(start, min(start+5, len(source.LAYERS)))):
                    contributions[source.LAYERS[layer]] = dict(alpha=self.audit.arr(out['extra'][4*j]),
                                                              RGB=self.audit.arr(out['extra'][4*j+1:4*j+4]))
        alpha_sum = sum(v['alpha'] for v in contributions.values()); rgb_sum = sum(v['RGB'] for v in contributions.values())
        background = self.audit.arr(self.rt.e.scene._background)[:, None, None]*(1-shared_alpha)
        environment = self.audit.arr(self.rt.e.env[raw])*(1-shared_alpha)
        closure = dict(alpha_max=float(np.abs(alpha_sum-shared_alpha).max()),
                       RGB_max=float(np.abs(rgb_sum+background+environment-shared_RGB).max()))
        if closure['alpha_max'] > 5e-5 or closure['RGB_max'] > 5e-5: raise RuntimeError('Same-sort source closure failed')
        yy, xx = np.nonzero(roi); y0, y1, x0, x1 = int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1
        saved = dict(xyxy=np.array([x0, y0, x1, y1]), ROI_union=roi[y0:y1, x0:x1],
                     residual_mask=residual_domain[y0:y1, x0:x1], RGB_residual=residual[:, y0:y1, x0:x1])
        for name, item in contributions.items():
            self.audit.save_image_set(raw, name+'_shared_source', item['RGB'], item['alpha'])
            self.audit.save_image_set(raw, name+'_source_on_grey', item['RGB']+.5*(1-item['alpha']))
            saved[name+'_alpha'] = item['alpha'][y0:y1, x0:x1]
            saved[name+'_RGB'] = item['RGB'][:, y0:y1, x0:x1]
        path = self.args.output/'evidence'/f'raw{raw:03d}_shared_layers_ROI.npz'
        np.savez_compressed(path, **saved)
        return dict(actual_shared_sort=True, closure=closure, parity=parity, ROI_float=self.core.ref(path))

    def query_view(self, raw):
        t = self.t; rows, meta, layers, qa = self.audit.scene_rows(raw); domains = self.audit.domains(raw)
        roi = t.zeros((768, 1152), device='cuda', dtype=t.bool)
        for name in ('shadow', 'circled', 'strip'):
            if name in domains: roi |= domains[name]
        with t.no_grad():
            is_road = t.as_tensor(meta['road_active_index'] >= 0, device='cuda')
            road = self.audit.raster(raw, dict(rows, mask=rows['mask']&is_road), environment=False)
            actor = self.rt.render(raw=raw, actor_only=True); actual = self.rt.render(raw=raw)
            road_stable = road['alpha'] >= .95; road_RGB = t.zeros_like(road['rgb'])
            road_RGB[:, road_stable] = road['rgb'][:, road_stable]/road['alpha'][road_stable]
            reliable = roi & ~domains['retained_K'] & (actor['alpha'] < .05) & road_stable
        weights = t.ones(len(rows['means3D']), device='cuda', requires_grad=True)
        features = t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]; features[:, 2] = weights
        out = self.audit.raster(raw, dict(rows, features=features))
        parity = dict(RGB_max=float((out['rgb']-actual['rgb']).abs().max()), alpha_max=float((out['alpha']-actual['alpha']).abs().max()))
        if parity['RGB_max'] > 5e-6 or parity['alpha_max'] > 1e-7: raise RuntimeError('Unit-feature render changed scene RGB/alpha')
        residual = out['rgb'].detach()-road_RGB.detach()
        residual_domain = reliable & (residual.abs().amax(0) > .08)

        def vjp(domain, pixel_weight=None, nonnegative=False):
            if not bool(domain.any()): return np.zeros(len(weights), np.float32)
            values = out['extra'][0][domain]
            if pixel_weight is not None: values = values*pixel_weight.detach()[domain]
            gradient = t.autograd.grad(values.sum(), weights, retain_graph=True)[0]; self.vjp_calls += 1
            if not bool(t.isfinite(gradient).all()): raise RuntimeError('Nonfinite unit-feature VJP')
            result = self.audit.arr(gradient)
            if nonnegative:
                if (result < -1e-7).any(): raise RuntimeError('Negative actual alpha unit-feature VJP')
                result = result.clip(min=0)
            return result

        gA = vjp(residual_domain, nonnegative=True)
        gE = np.stack([vjp(residual_domain, residual[c]) for c in range(3)], axis=1)
        gTE = vjp(residual_domain, (road_RGB*residual).sum(0))
        gV = vjp(domains['real_V'], nonnegative=True); gK = vjp(domains['retained_K'], nonnegative=True)
        gROI = vjp(roi, nonnegative=True); RGB = self.audit.arr(rows['colors_precomp'])
        nominated, score = rgb_row_score(gA, gE, gTE, RGB)
        allowed = allowed_source(meta['layer_id'], meta['owner'], meta['native_branch'])
        native = (meta['scene_global_row_id'] >= 0)&(meta['scene_global_row_id'] < self.audit.ns+self.audit.nd)
        for j in np.flatnonzero(native & (nominated | np.isin(meta['scene_global_row_id'], WATCH_IDS))).tolist():
            gid = int(meta['scene_global_row_id'][j]); item = self.by_id.setdefault(gid, dict(global_row_id=gid,
                source_branch=int(meta['native_branch'][j]), source_row_id=int(meta['source_row_id'][j]),
                source_class=source.LAYERS[int(meta['layer_id'][j])], layer_id=int(meta['layer_id'][j]),
                owner=int(meta['owner'][j]), allowed_source_class=bool(allowed[j]), protected_native_K=False))
            item['protected_native_K'] |= bool(meta['protected_native_K'][j])
        self.results[raw] = dict(meta=meta, gA=gA, gE=gE, gTE=gTE, score=score, RGB=RGB,
                                 gV=gV, gK=gK, gROI=gROI, nominated=nominated)
        folder = self.args.output/'evidence'; folder.mkdir(parents=True, exist_ok=True)
        evidence_path = folder/f'raw{raw:03d}_rgb_source_vjp.npz'
        np.savez_compressed(evidence_path, **meta, gA=gA, gE=gE, gTE=gTE, source_RGB=RGB,
            RGB_alignment_score=score, real_V_alpha_raw=gV, retained_K_alpha_raw=gK,
            diagnostic_ROI_alpha_raw=gROI, candidate_in_view=nominated&allowed&~meta['protected_native_K'])
        shared_RGB = self.audit.arr(out['rgb']); shared_alpha = self.audit.arr(out['alpha'])
        target_RGB = self.audit.arr(road_RGB); residual_np = self.audit.arr(residual)
        reliable_np = self.audit.arr(reliable); residual_np_mask = self.audit.arr(residual_domain)
        np.savez_compressed(folder/f'raw{raw:03d}_actual_float_reference.npz', shared_RGB=shared_RGB,
            shared_alpha=shared_alpha, actor_alpha=self.audit.arr(actor['alpha']),
            road_RGB_premultiplied=self.audit.arr(road['rgb']), road_alpha=self.audit.arr(road['alpha']),
            road_RGB_normalized_stable=target_RGB, reliable_ground=reliable_np,
            RGB_residual=residual_np, RGB_residual_mask=residual_np_mask, ROI_union=self.audit.arr(roi))
        self.audit.save_image_set(raw, 'current_shared', shared_RGB, shared_alpha)
        self.audit.save_image_set(raw, 'actual_road_reference', target_RGB, self.audit.arr(road['alpha']))
        self.audit.save_image_set(raw, 'RGB_absolute_residual', np.clip(np.abs(residual_np)*3, 0, 1)*reliable_np[None])
        self.audit.save_image_set(raw, 'RGB_signed_residual', np.clip(.5+residual_np*2, 0, 1)*reliable_np[None])
        self.audit.save_image_set(raw, 'fixed_RGB_residual_mask', np.repeat(residual_np_mask[None].astype(np.float32), 3, 0))
        layer_images = self.layer_images(raw, rows, layers, shared_RGB, shared_alpha, self.audit.arr(roi), residual_np_mask, residual_np)
        source_metrics = []
        for layer, name in enumerate(source.LAYERS):
            select = meta['layer_id'] == layer; positive = select & (score > 0)
            source_metrics.append(dict(layer_id=layer, source_class=name, actual_rows=int((select&meta['actual_row_mask']).sum()),
                residual_contributing_rows=int((select&(gA > 0)).sum()), nominated_rows=int((select&nominated).sum()),
                residual_alpha_sum=float(gA[select].sum(dtype=np.float64)),
                alignment_score_signed=float(score[select].sum(dtype=np.float64)),
                alignment_score_positive=float(score[positive].sum(dtype=np.float64)),
                alignment_score_negative=float(score[select&(score < 0)].sum(dtype=np.float64)),
                gE_sum=gE[select].sum(0, dtype=np.float64).tolist(), gTE_sum=float(gTE[select].sum(dtype=np.float64)),
                positive_rows_with_current_V_support=int((positive&(gV > 0)).sum()),
                positive_rows_with_current_K_support=int((positive&(gK > 0)).sum()),
                positive_rows_known_K1922=int((positive&meta['protected_native_K']).sum())))
        per_roi = {}
        luma_residual = np.einsum('c,chw->hw', LUMA, residual_np)
        for name in ('shadow', 'circled', 'strip'):
            if name in domains:
                mask = self.audit.arr(domains[name]); valid = mask&reliable_np
                per_roi[name] = dict(reliable_ground_pixels=int(valid.sum()), RGB_residual_pixels=int((mask&residual_np_mask).sum()),
                    dark_pixels=int((valid&(luma_residual < -.08)).sum()), bright_pixels=int((valid&(luma_residual > .08)).sum()))
        report = dict(raw=raw, reliable_ground_pixels=int(reliable_np.sum()), RGB_residual_pixels=int(residual_np_mask.sum()),
            dark_pixels=int((reliable_np&(luma_residual < -.08)).sum()),
            bright_pixels=int((reliable_np&(luma_residual > .08)).sum()), per_ROI=per_roi,
            source_classes=source_metrics, unit_feature_parity=parity, layer_images=layer_images,
            camera_and_row_QA=qa, actual_evidence=self.core.ref(evidence_path))
        self.core.write(self.args.output/f'raw{raw:03d}_rgb_metrics.json', report)
        print('RGB_QUERY_VIEW_COMPLETE', raw, json.dumps(per_roi), flush=True)
        return report

    def finish_candidates(self):
        wanted = np.asarray(sorted(self.by_id), np.int64)
        for item in self.by_id.values():
            item.update(positive_alignment_score=0., signed_alignment_score=0., nominated=False,
                        current_V_alpha_sum=0., current_K_alpha_sum=0., original_V_alpha_sum=0.,
                        original_K_alpha_sum=0., original_ROI_alpha_sum=0., current_views_present=0,
                        original_views_present=0, per_view=[])
        for raw, value in self.results.items():
            positions = row_positions(value['meta']['scene_global_row_id'], wanted)
            original_path = self.args.attribution_dir/'evidence'/f'raw{raw:03d}_exact_row_contributions.npz'
            self.track(original_path)
            with np.load(original_path, allow_pickle=False) as original:
                original_positions = row_positions(original['scene_global_row_id'], wanted)
                original_V = original['real_V_alpha_raw']; original_K = original['retained_K_alpha_raw']
                original_ROI = np.zeros_like(original_V)
                for name in ('shadow', 'circled', 'strip'):
                    key = name+'_alpha_raw'
                    if key in original.files: original_ROI += original[key]
                for k, gid in enumerate(wanted.tolist()):
                    item = self.by_id[gid]; j = int(positions[k]); oj = int(original_positions[k])
                    entry = dict(raw=int(raw), current_present=j >= 0, original_present=oj >= 0)
                    if j >= 0:
                        score = float(value['score'][j]); item['positive_alignment_score'] += max(0., score)
                        item['signed_alignment_score'] += score; item['nominated'] |= bool(value['nominated'][j])
                        item['current_V_alpha_sum'] += float(value['gV'][j]); item['current_K_alpha_sum'] += float(value['gK'][j])
                        item['current_views_present'] += 1; item['protected_native_K'] |= bool(value['meta']['protected_native_K'][j])
                        entry.update(gA=float(value['gA'][j]), gE=value['gE'][j].tolist(), gTE=float(value['gTE'][j]),
                            source_RGB=value['RGB'][j].tolist(), RGB_alignment_score=score,
                            actual_V_alpha_sum=float(value['gV'][j]), actual_K_alpha_sum=float(value['gK'][j]),
                            actual_ROI_alpha_sum=float(value['gROI'][j]), qualifies_in_view=bool(value['nominated'][j]))
                    if oj >= 0:
                        item['original_views_present'] += 1
                        item['original_V_alpha_sum'] += float(original_V[oj]); item['original_K_alpha_sum'] += float(original_K[oj])
                        item['original_ROI_alpha_sum'] += float(original_ROI[oj])
                        entry.update(original_V_alpha_sum=float(original_V[oj]), original_K_alpha_sum=float(original_K[oj]),
                                     original_ROI_alpha_sum=float(original_ROI[oj]))
                    item['per_view'].append(entry)
        for item in self.by_id.values():
            reasons = []
            if not item['allowed_source_class']: reasons.append('source_class_outside_static_background_or_dynamic_owner8_14')
            if item['protected_native_K']: reasons.append('protected_native_K1922')
            if item['current_views_present'] != len(self.core.BUILD): reasons.append('current_two_view_evidence_missing')
            if item['original_views_present'] != len(self.core.BUILD): reasons.append('original_two_view_evidence_missing')
            for prefix in ('current', 'original'):
                for support in ('V', 'K'):
                    if item[prefix+'_'+support+'_alpha_sum'] != 0: reasons.append(prefix+'_'+support+'_support_nonzero')
            if item['original_ROI_alpha_sum'] <= 0: reasons.append('no_original_ROI_support')
            if not item['nominated']: reasons.append('no_view_with_gA_above_0.05_and_positive_score')
            item.update(withheld_reasons=reasons, cleanup_eligible=not reasons,
                current_two_view_V_K_exact_zero=item['current_V_alpha_sum'] == 0 and item['current_K_alpha_sum'] == 0,
                original_two_view_V_K_exact_zero=item['original_V_alpha_sum'] == 0 and item['original_K_alpha_sum'] == 0)
        ranked = sorted((x for x in self.by_id.values() if x['nominated']),
                        key=lambda x: (-x['positive_alignment_score'], x['global_row_id']))
        eligible = [x for x in ranked if x['cleanup_eligible']]
        withheld = [x for x in ranked if not x['cleanup_eligible']]
        limit = self.args.rgb_top_rows; top = eligible[:limit]; blocked = withheld[:limit]
        support_blocked = [x for x in withheld if any('support_nonzero' in r or r == 'protected_native_K1922' for r in x['withheld_reasons'])]
        classes = []
        for name in source.LAYERS:
            group = [x for x in ranked if x['source_class'] == name]
            classes.append(dict(source_class=name, nominated_native_rows=len(group),
                eligible_rows=sum(x['cleanup_eligible'] for x in group), withheld_rows=sum(not x['cleanup_eligible'] for x in group),
                eligible_positive_alignment_score=sum(x['positive_alignment_score'] for x in group if x['cleanup_eligible']),
                withheld_positive_alignment_score=sum(x['positive_alignment_score'] for x in group if not x['cleanup_eligible'])))
        artifact = dict(schema='a20.rgb_residual_rank.v1', diagnostic_only=True, no_cleanup_applied=True,
            top_limit=limit, eligible_count=len(eligible), withheld_count=len(withheld),
            rows=top, withheld_rows=blocked, withheld_for_V_K_support_rows=support_blocked[:limit],
            watched_large_footprint_rows=[self.by_id[g] for g in WATCH_IDS if g in self.by_id],
            native_source_class_qualification=classes,
            rank='sum of positive per-view RGB alignment scores; signed total retained; whole-set actual protection still required')
        path = self.args.output/'top_rgb_residual_rows.json'; self.core.write(path, artifact)
        self.core.write(self.args.output/'candidate_ids_rgb_residual.json', dict(global_row_ids=[x['global_row_id'] for x in top],
            diagnostic_only=True, top_limit=limit, eligible_total=len(eligible), ranking_source=self.core.ref(path),
            original_and_current_two_view_V_K_exact_zero=True, protected_K1922_excluded=True,
            source_scope='native_static_background OR native_dynamic_other_actor owner8/14',
            no_cleanup_applied=True, road_reference_is_GT=False))
        return artifact

    def run(self):
        views = [self.query_view(raw) for raw in self.core.BUILD]; ranked = self.finish_candidates()
        unchanged = self.gpu.tree_equal(self.gpu.cpu(self.rt.actor.state_dict()), self.audit.actor_before)
        Adam_unchanged = self.gpu.tree_equal(self.gpu.cpu(self.audit.opt.state_dict()), self.audit.Adam_before)
        sources = {str(p.resolve()): self.core.sha(p) == self.audit.source_hashes[str(p.resolve())] for p in self.audit.source_paths}
        if not unchanged or not Adam_unchanged or not all(sources.values()): raise RuntimeError('Frozen source changed during RGB diagnostic')
        proof = dict(schema='a20.rgb_residual_query.proof.v1', complete=True, actor_and_active_unchanged=unchanged,
            Adam_unchanged=Adam_unchanged, source_files_unchanged=sources, frozen_scene=self.rt.check_frozen(),
            actual_shared_sort_feature_VJP=True, signed_RGB_weights_not_clipped=True,
            safe_road_normalization_alpha_minimum=.95, actor_only_alpha_maximum=.05,
            road_reference_is_GT=False, inactive_unknown_road_donors_used=False,
            optimizer_updates=0, newly_retired_context_rows=0, eligible_rows=ranked['eligible_count'],
            compact_rows=len(ranked['rows']), withheld_rows=ranked['withheld_count'], views=views,
            feature_VJP_calls=self.vjp_calls, native_calls=self.rt.calls,
            elapsed_seconds=time.monotonic()-self.started, heldout_frames_opened=[], GPU_keep_on=True, power_operations=0)
        self.core.write(self.args.output/'proof.json', proof)
        self.core.write(self.args.output/'metrics.json', dict(views=views, proof=proof,
            native_source_class_qualification=ranked['native_source_class_qualification']))
        print('RGB_RESIDUAL_QUERY_COMPLETE', len(ranked['rows']), 'bounded candidates; no cleanup', flush=True)


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check: print(json.dumps(cpu_self_check(), indent=2))
    else:
        try: RGBQuery(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'rgb_query_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise

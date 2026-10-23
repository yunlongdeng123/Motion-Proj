"""固定分母、逐身份退化及删车道路贡献；诊断不代替人工质量判定。"""
import numpy as np
from scipy.ndimage import label

CONTRACT = dict(vehicle_ids=[5, 26, 18, 6, 37],
    vehicle_metrics='each identity interior/edge: RGB MAE/MSE/PSNR; masked SSIM only with valid windows',
    full_metrics='V1 float clamp [0,1], full PSNR/SSIM/LPIPS, all 49 frames',
    degradation_flag='any per-frame composite MSE exceeds same-checkpoint native by 1e-8; diagnostic flag, not approval',
    coverage_thresholds=[.05, .25, .5], color_mae_threshold=.05,
    color_proxy='same world unit error on reliable visible pixels pooled across 49 frames; NOT hidden texture GT',
    topology='8-connected components in fixed projected query pixel domain; largest component/domain',
    success='final image AND each vehicle preserved AND actual road completion improves; transparency alone fails',
    a6='fixed geometry cannot add unsupported cells', human_verdict=None)

DELETION_CONTRACT = dict(version='V41_PAIRED_DELETION_R7',
    reference='fixed cross-view RGB proxy; legacy and additions also reported separately; NOT hidden-road ground truth',
    primary='float RGB clipped to [0,1]; raw float errors retained separately',
    delta='after static+road minus before native static; negative error delta is improvement',
    pixel_comparison='per-pixel channel-mean absolute error; epsilon=1e-6',
    denominator='full original actor silhouette, including unsupported/unlabelled pixels',
    aggregation='pixel-frame weighted errors and fractions; NOT world road area',
    unknown='unlabelled pixels have unknown color correctness, never zero error',
    continuity='8-connected projected pixel components; not world-surface connectivity',
    quality_accepted=False)


def paired_deleted_metrics(before, after, reference, target, labelled, contribution):
    """Same checkpoint/domain/reference. Never select a domain by the rendered alpha/error."""
    target, labelled = np.asarray(target, bool), np.asarray(labelled, bool)
    if target.shape != labelled.shape or np.any(labelled & ~target):
        raise ValueError('deleted reference must be a subset of the full fixed target')
    images = [np.asarray(x, np.float64) for x in (before, after, reference)]
    if any(x.shape != (3, *target.shape) or not np.isfinite(x).all() for x in images):
        raise ValueError('finite CHW RGB required')
    contribution = np.asarray(contribution)
    if contribution.shape != target.shape or not np.isfinite(contribution).all():
        raise ValueError('finite HW contribution required')
    n, total = int(labelled.sum()), int(target.sum())
    result = dict(target_pixels=total, labelled_pixels=n, unknown_pixels=total-n,
        labelled_fraction=n/total if total else None, before=None, after=None, delta=None,
        road_contribution_labelled=float(contribution[labelled].mean()) if n else None,
        road_contribution_full_target=float(contribution[target].mean()) if total else None,
        raw=None, pixel_change=dict(improved=0, degraded=0, unchanged=0,
            improved_fraction_labelled=None, degraded_fraction_labelled=None,
            improved_fraction_target=None, degraded_fraction_target=None), thresholds={})
    clipped = [np.clip(x, 0, 1) for x in images]
    before_error, after_error = [np.abs(x-clipped[2]).mean(0) for x in clipped[:2]]
    for clamp, values in ((True, clipped), (False, images)):
        pair = [{key:float(fn(x[:, labelled]-values[2][:, labelled]).mean()) if n else None
                 for key, fn in (('mae', np.abs), ('mse', np.square))} for x in values[:2]]
        delta = {key:pair[1][key]-pair[0][key] if n else None for key in ('mae', 'mse')}
        metrics = dict(before=pair[0], after=pair[1], delta=delta)
        if clamp: result.update(metrics)
        else: result['raw'] = metrics
    if n:
        diff = after_error[labelled]-before_error[labelled]
        improved, degraded = int((diff < -1e-6).sum()), int((diff > 1e-6).sum())
        result['pixel_change'] = dict(improved=improved, degraded=degraded,
            unchanged=n-improved-degraded, improved_fraction_labelled=improved/n,
            degraded_fraction_labelled=degraded/n, improved_fraction_target=improved/total,
            degraded_fraction_target=degraded/total)
    for t in CONTRACT['coverage_thresholds']:
        effective = contribution >= t
        verified = effective & labelled & (after_error <= CONTRACT['color_mae_threshold'])
        result['thresholds'][str(t)] = dict(
            effective_full_target=topology(target, effective),
            effective_labelled=topology(labelled, effective),
            verified_color_full_target=topology(target, verified),
            verified_color_labelled=topology(labelled, verified))
    return result


def deletion_summary(records, field='crossview_deleted'):
    applicable = [r for r in records if r[field]['applicable']]
    result = dict(contract=DELETION_CONTRACT, source_field=field, applicable_frames=len(applicable), actors={})
    for actor in CONTRACT['vehicle_ids']:
        rows = [dict(frame=r['frame'], **r[field]['actors'][str(actor)]) for r in applicable]
        known = [r for r in rows if r['labelled_pixels']]
        total = sum(r['target_pixels'] for r in rows)
        n = sum(r['labelled_pixels'] for r in rows)
        weighted = lambda side, key: sum(r['labelled_pixels']*r[side][key] for r in known)/n if n else None
        pair = {side:{k:weighted(side, k) for k in ('mae','mse')} for side in ('before','after','delta')}
        changes = {k:sum(r['pixel_change'][k] for r in rows) for k in ('improved','degraded','unchanged')}
        for k in ('improved','degraded'):
            changes[k+'_fraction_labelled'] = changes[k]/n if n else None
            changes[k+'_fraction_target'] = changes[k]/total if total else None
        worst = lambda k: max(known, key=lambda r:r['delta'][k]) if known else None
        result['actors'][str(actor)] = dict(target_pixel_frames=total, labelled_pixel_frames=n,
            unknown_pixel_frames=total-n, labelled_fraction=n/total if total else None,
            labelled_frames=len(known), zero_label_frames=[r['frame'] for r in rows if not r['labelled_pixels']],
            pixel_weighted=pair, pixel_change=changes,
            road_contribution_labelled=sum(r['labelled_pixels']*r['road_contribution_labelled'] for r in known)/n if n else None,
            road_contribution_full_target=sum(r['target_pixels']*r['road_contribution_full_target'] for r in rows if r['target_pixels'])/total if total else None,
            worst_mae_delta=worst('mae'), worst_mse_delta=worst('mse'),
            worst_labelled_fraction=min((r for r in rows if r['target_pixels']),
                key=lambda r:r['labelled_fraction'], default=None), frames=rows)
    return result


def regression_summary(records):
    result = {}
    regions = records[0]['regions']['v1']
    for region in regions:
        rows = []
        for record in records:
            before, after = [record['regions'][k][region] for k in ('v1', 'composite')]
            if not before['pixels']: continue
            rows.append(dict(frame=record['frame'], mse_delta=after['mse']-before['mse'],
                mae_delta=after['mae']-before['mae'],
                psnr_delta=None if before['psnr'] is None or after['psnr'] is None else after['psnr']-before['psnr']))
        result[region] = dict(nonempty_frames=len(rows),
            worsened_frames=sum(r['mse_delta'] > 1e-8 for r in rows),
            worst_mse_delta=max(rows, key=lambda r:r['mse_delta']) if rows else None,
            worst_psnr_delta=min((r for r in rows if r['psnr_delta'] is not None),
                                 key=lambda r:r['psnr_delta'], default=None), frames=rows)
    return result


def query_maps(arrays, frame_index, actor):
    rows = (arrays['query_frame'] == frame_index) & (arrays['query_actor'] == actor)
    xy = arrays['query_pixel'][rows]
    ids = np.full((320, 480), -1, np.int64)
    ids[xy[:, 1], xy[:, 0]] = arrays['query_unit_id'][rows]
    return ids >= 0, ids


def topology(domain, selected):
    count = int(domain.sum())
    parts, n = label(domain & selected, structure=np.ones((3, 3), bool))
    sizes = np.bincount(parts.ravel())[1:]
    return dict(domain_pixels=count, covered_pixels=int((domain & selected).sum()),
        coverage_fraction=float((domain & selected).sum()/count) if count else None,
        components=int(n), largest_component_pixels=int(sizes.max()) if n else 0,
        largest_component_fraction=float((sizes.max() if n else 0)/count) if count else None)


def road_query_metrics(arrays, frame_index, actor, contribution, deleted_contribution,
                       independent_alpha, color_error, color_count):
    domain, ids = query_maps(arrays, frame_index, actor)
    safe_ids = np.maximum(ids, 0)
    supported = domain & arrays['surface_supported'][safe_ids]
    observed_units = (arrays['surface_supported'] & (arrays['provenance'] == 1)
                      & ((arrays['base_weight'] > 0).sum(0) >= 2))
    eligible = domain & observed_units[safe_ids]
    verified_color = domain & (color_count[safe_ids] > 0) & (color_error[safe_ids] <= CONTRACT['color_mae_threshold'])
    result = dict(query_pixels=int(domain.sum()), supported_pixels=int(supported.sum()),
        unsupported_pixels=int((domain & ~supported).sum()), R_eligible_pixels=int(eligible.sum()),
        color_evaluated_pixels=int((domain & (color_count[safe_ids] > 0)).sum()),
        hidden_RGB_GT_available=False, thresholds={})
    for threshold in CONTRACT['coverage_thresholds']:
        result['thresholds'][str(threshold)] = dict(
            independent=topology(domain, independent_alpha >= threshold),
            full=topology(domain, contribution >= threshold),
            delete=topology(domain, deleted_contribution >= threshold),
            delete_with_verified_source_color=topology(domain,
                (deleted_contribution >= threshold) & verified_color),
            supported_delete=topology(supported, deleted_contribution >= threshold),
            eligible_delete=topology(eligible, deleted_contribution >= threshold))
    return result

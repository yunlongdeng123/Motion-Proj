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

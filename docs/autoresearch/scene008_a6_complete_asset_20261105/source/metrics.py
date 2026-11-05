"""Fixed-observation metrics for a complete a6 asset; no renderer/runtime imports.

All denominators come from the original OBS bank, never predicted alpha/depth.
RGB/depth errors use real visible measurements. A37-occluded contributions are
visibility diagnostics, not hidden-surface RGB/depth accuracy or reveal rates.
"""
import numpy as np
import torch

GROUPS = ('asset', 'a37', 'road', 'legacy_static')
SOURCE_LIMITS = (
    'visible[0] is the reviewed FRONT a6 mask eroded once, excluding the other '
    'actor dilated once, and excluding a measured return substantially in front '
    'of the padded a6 box. It is the original fixed full RGB evaluation core.',
    'visible_reviewed[0] is the complete reviewed visible FRONT silhouette, '
    'including boundary pixels; it is not a complete amodal silhouette.',
    'depth_m/depth_valid are real same-frame associated LiDAR camera-z returns '
    'with reviewed visibility/nearest-return checks, not native/PVD predictions. '
    'They may already have participated in parent training; they are not new test data.',
    'free outside box_near is an annotated-box-miss collar, not measured empty '
    '3-D space. free inside the padded box is separately labeled measured-through.',
    'On native49, a6 free contains 79146 box-miss pixels and zero measured-through '
    'pixels. Unknown/possible-occluded pixels are never negative supervision.',
    'road_edit_rgb is a fixed reobserved-road reference; use it only in the '
    'delete state, not as current-frame hidden RGB truth while a37 is present.',
    'No semantic rear-window/taillight ROI is invented here; inspect those '
    'features in the saved fixed-view images.',
)


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def numpy_metric(value, mask):
    value, mask = as_numpy(value), as_numpy(mask).astype(bool)
    if value.shape != mask.shape:
        raise ValueError(f'Metric/mask shapes differ: {value.shape} vs {mask.shape}')
    selected = value[mask].astype(np.float64, copy=False)
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite value on a fixed evaluation pixel')
    count = int(len(selected))
    total = float(selected.sum())
    return dict(count=count, sum=total, mean=total/count if count else None,
                max=float(selected.max()) if count else 0.,
                above_002=int((selected > .02).sum()), above_005=int((selected > .05).sum()))


def torch_metric(value, mask):
    mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
    if value.shape != mask.shape:
        raise ValueError(f'Metric/mask shapes differ: {tuple(value.shape)} vs {tuple(mask.shape)}')
    selected = value[mask].double()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError('Nonfinite value on a fixed evaluation pixel')
    count = int(selected.numel())
    total = float(selected.sum())
    return dict(count=count, sum=total, mean=total/count if count else None,
                max=float(selected.max()) if count else 0.,
                above_002=int((selected > .02).sum()), above_005=int((selected > .05).sum()))


def metric(value, mask):
    """Scalar count/sum/mean/max and strict .02/.05 exceedances; empty mean=None."""
    return torch_metric(value, mask) if torch.is_tensor(value) else numpy_metric(value, mask)


def pool(items):
    """Pool scalar metrics by fixed pixel counts, not by per-frame means."""
    items = list(items)
    count = sum(item['count'] for item in items)
    total = sum(item['sum'] for item in items)
    return dict(count=count, sum=total, mean=total/count if count else None,
                max=max((item['max'] for item in items), default=0.),
                above_002=sum(item['above_002'] for item in items),
                above_005=sum(item['above_005'] for item in items))


def observation(obs):
    """Accept either the original NumPy OBS dictionary or Runtime.raw(raw)."""
    source = obs['obs'] if 'obs' in obs and isinstance(obs['obs'], dict) else obs
    return {key: as_numpy(value) for key, value in source.items()
            if isinstance(value, np.ndarray) or torch.is_tensor(value) or np.isscalar(value)}


def fixed_masks(obs):
    source = observation(obs)
    visible = source['visible'].astype(bool)
    reviewed = source['visible_reviewed'].astype(bool)
    depth = source['depth_valid'].astype(bool)
    if visible.shape[0] != 2 or reviewed.shape != visible.shape or depth.shape != visible.shape:
        raise ValueError('Expected original two-actor OBS masks')
    if np.any(depth & ~visible) or np.any(visible[0] & visible[1]):
        raise ValueError('Contradictory original visible/depth masks')
    if not np.isfinite(source['depth_m'][depth]).all() or np.any(source['depth_m'][depth] <= 0):
        raise ValueError('Invalid original measured depth')
    zero = np.zeros_like(visible[0])
    free = source.get('free', np.stack((zero, zero)))[0].astype(bool)
    padded_hit = source['box_near_m'][0] > 0
    strict = source.get('occluded', np.stack((zero, zero)))[0].astype(bool)
    possible = source.get('possible_occluded', np.stack((zero, zero)))[0].astype(bool)
    outside = source['protect'].astype(bool)
    road = source['road'].astype(bool)
    # These are observed non-a6 IMAGE pixels, not empty-volume labels. Exclude
    # both original actor silhouettes in both states to keep one common domain.
    observed_non_a6 = (outside | road) & ~reviewed.any(0)
    masks = dict(a6_visible=visible[0], a6_reviewed=reviewed[0], a6_depth=depth[0],
                 a37_visible=visible[1], a37_reviewed=reviewed[1], a37_depth=depth[1],
                 strict_a37_foreground=strict, possible_occlusion_only=possible & ~strict,
                 road_visible=road, road_reobserved=source['road_edit'].astype(bool),
                 outside=outside, observed_non_a6=observed_non_a6,
                 negative_box_miss=free & ~padded_hit,
                 negative_measured_through=free & padded_hit)
    if np.any(masks['a6_visible'] & (free | strict)):
        raise ValueError('Original positive mask overlaps negative/occluded evidence')
    return masks


def _rgb(value):
    result = as_numpy(value)
    if result.ndim != 3:
        raise ValueError('Expected RGB with three dimensions')
    if result.shape[0] == 3:
        result = np.moveaxis(result, 0, -1)
    if result.shape[-1] != 3:
        raise ValueError('Expected RGB HxWx3 or 3xHxW')
    return result.astype(np.float64)/255. if result.dtype.kind in 'ui' else result.astype(np.float64)


def _scalar(value):
    result = as_numpy(value)
    return result[0] if result.ndim == 3 and result.shape[0] == 1 else result


def expected_depth(package, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive model-to-metre scale required')
    # Predicted misses stay in the fixed denominator. Coverage is reported too.
    return _scalar(package['depth_sum'])/np.maximum(_scalar(package['alpha']), 1e-5)/scale


def _groups(package, isolated=False):
    if isolated:
        return np.stack((_scalar(package['alpha']),) + (np.zeros_like(_scalar(package['alpha'])),)*3)
    values = as_numpy(package.get('component_alpha_T', package.get('extra')))
    if values.ndim != 3 or values.shape[0] < 4:
        raise ValueError('Scene package needs four groups: asset,a37,road,legacy_static')
    return values[:4]


def _rgb_record(predicted, baseline, target, mask):
    after = metric(np.abs(predicted-target).mean(-1), mask)
    before = metric(np.abs(baseline-target).mean(-1), mask)
    return dict(rgb_error=after, baseline_rgb_error=before,
                rgb_error_minus_baseline=(after['mean']-before['mean']) if after['count'] else None,
                parent_rgb_change=metric(np.abs(predicted-baseline).mean(-1), mask))


def evaluate_frame(obs, out, baseline, scale, state='keep'):
    """JSON-safe fixed-denominator frame evaluation, without a quality decision.

    state='keep'/'delete' evaluates joint scenes; 'solo', 'isolated' or 'black' evaluates
    a6 alone and omits scene-preservation metrics. out and baseline must use the
    same camera/state and frozen background. No mask uses predicted alpha.
    """
    if state not in ('keep', 'delete', 'solo', 'isolated', 'black'):
        raise ValueError(state)
    isolated = state in ('solo', 'isolated', 'black')
    source, masks = observation(obs), fixed_masks(obs)
    predicted, before, target = _rgb(out['rgb']), _rgb(baseline['rgb']), _rgb(source['rgb'])
    asset, base_asset = _groups(out, isolated)[0], _groups(baseline, isolated)[0]
    z, z0 = expected_depth(out, scale), expected_depth(baseline, scale)
    measured = source['depth_m'][0]
    a6 = _rgb_record(predicted, before, target, masks['a6_visible'])
    a6.update(visible_coverage=metric(asset, masks['a6_visible']),
              baseline_visible_coverage=metric(base_asset, masks['a6_visible']),
              reviewed_coverage=metric(asset, masks['a6_reviewed']),
              reviewed_below_half=metric((asset < .5).astype(float), masks['a6_reviewed']),
              measured_depth_error=metric(np.abs(z-measured), masks['a6_depth']),
              baseline_measured_depth_error=metric(np.abs(z0-measured), masks['a6_depth']),
              measured_parent_depth_change=metric(np.abs(z-z0), masks['a6_depth']))
    result = dict(raw=int(source['raw']) if 'raw' in source else None, state=state,
                  mask_counts={key: int(mask.sum()) for key, mask in masks.items()}, a6=a6)
    result['negative_diagnostics'] = {
        name: metric(asset, masks[name]) for name in
        ('negative_box_miss', 'negative_measured_through', 'observed_non_a6')}
    if isolated:
        return result
    groups = _groups(out)
    result['preservation'] = dict(
        road_visible=_rgb_record(predicted, before, target, masks['road_visible']),
        outside=_rgb_record(predicted, before, target, masks['outside']),
        road_reobserved_parent_change=metric(np.abs(predicted-before).mean(-1), masks['road_reobserved']))
    result['occlusion_diagnostics'] = dict(
        asset_on_strict_a37_foreground=metric(asset, masks['strict_a37_foreground']),
        asset_on_possible_occlusion_only=metric(asset, masks['possible_occlusion_only']))
    if state == 'keep':
        result['preservation']['a37'] = _rgb_record(predicted, before, target, masks['a37_visible'])
        result['occlusion_diagnostics']['a37_coverage'] = metric(groups[1], masks['a37_visible'])
        result['occlusion_diagnostics']['asset_on_a37_visible'] = metric(asset, masks['a37_visible'])
    else:
        result['preservation']['road_reobserved'] = _rgb_record(
            predicted, before, _rgb(source['road_edit_rgb']), masks['road_reobserved'])
        result['occlusion_diagnostics']['deleted_a37_full_image'] = metric(
            np.abs(groups[1]), np.ones_like(masks['a6_visible']))
    return result


def evaluate_pair(obs, keep, deleted, baseline_keep, baseline_delete, scale):
    """Paired-state contribution changes on a fixed measured-foreground domain.

    No hidden RGB target is used after a37 deletion. These values describe the
    complete asset's visibility change, not accuracy of its unseen geometry.
    """
    masks = fixed_masks(obs)
    contribution_change = _groups(deleted)[0]-_groups(keep)[0]
    original_change = _groups(baseline_delete)[0]-_groups(baseline_keep)[0]
    return dict(
        keep=evaluate_frame(obs, keep, baseline_keep, scale, 'keep'),
        delete=evaluate_frame(obs, deleted, baseline_delete, scale, 'delete'),
        paired_visibility_diagnostic=dict(
            strict_foreground_asset_change=metric(contribution_change, masks['strict_a37_foreground']),
            strict_foreground_native_change=metric(original_change, masks['strict_a37_foreground']),
            interpretation='Fixed foreground-witness pixels; not hidden RGB/depth truth'))

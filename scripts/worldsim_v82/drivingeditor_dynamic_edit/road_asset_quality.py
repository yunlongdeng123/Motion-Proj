"""道路四阶段独立验收；仅使用显式输入和真实浮点贡献，不推断隐藏真值。"""
from pathlib import Path
import json

import numpy as np

SCHEMA = 'road_asset_quality.v1'
SOURCE_NAMES = {0: 'unknown', 1: 'observed', 2: 'retrieved', 3: 'generated'}


def rgb_hwc(value):
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError('RGB must have three dimensions')
    if value.shape[-1] == 3:
        return value
    if value.shape[0] == 3:
        return value.transpose(1, 2, 0)
    raise ValueError('RGB requires three channels')


def mask_hw(value, shape):
    value = np.asarray(value).squeeze()
    if value.shape != tuple(shape) or value.dtype != np.bool_:
        raise ValueError('Mask must be boolean with matching image shape')
    return value


def alpha_hw(value, shape):
    value = np.asarray(value).squeeze()
    if value.shape != tuple(shape):
        raise ValueError('Alpha must match image shape')
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError('Numerical contribution audit requires float arrays')
    return value


def _mean(value, mask):
    return float(np.asarray(value)[mask].mean()) if mask.any() else None


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _record(stage, checks, metrics, *, acceptance=None, missing=()):
    failed = [key for key, value in checks.items() if value is False]
    unknown = [key for key, value in checks.items() if value is None]
    technical = 'failed' if failed else 'incomplete' if unknown else 'pass'
    return dict(schema=SCHEMA, stage=stage, status=acceptance or technical,
                technical_status=technical, checks=checks, failed_checks=failed,
                unknown_checks=unknown, metrics=metrics, missing_evidence=list(missing),
                visual_quality_status='incomplete', human_verdict=None,
                hidden_road_GT_used=False)


def rgb_difference(rgb, baseline, mask, tolerance=1 / 255):
    """完整掩码计数；不通过腐蚀掩码隐藏边界变化。"""
    rgb, baseline = rgb_hwc(rgb), rgb_hwc(baseline)
    if rgb.shape != baseline.shape:
        raise ValueError('RGB comparison shape mismatch')
    mask = mask_hw(mask, rgb.shape[:2])
    delta = np.abs(rgb.astype(np.float64) - baseline.astype(np.float64))
    maximum = delta.max(-1)
    n = int(mask.sum())
    return dict(pixels=n, mean_abs_rgb=_mean(delta.mean(-1), mask),
                max_abs_rgb=float(maximum[mask].max()) if n else None,
                over_tolerance_pixels=int((mask & (maximum > tolerance)).sum()),
                changed_pixels=int((mask & (maximum > 0)).sum()), tolerance=tolerance,
                mask_erosion_px=0)


def audit_completion(observation, decoded, composite, write_mask, protected_mask,
                     input_mask=None):
    """评价写回合法性；输入孔洞面积从不充当纹理质量或3D删除依据。"""
    observation, decoded, composite = map(rgb_hwc, (observation, decoded, composite))
    if observation.shape != decoded.shape or observation.shape != composite.shape:
        raise ValueError('Completion images must share shape')
    shape = observation.shape[:2]
    write = mask_hw(write_mask, shape)
    protected = mask_hw(protected_mask, shape)
    delta = np.abs(composite.astype(np.float64) - observation.astype(np.float64)).max(-1)
    finite = all(np.isfinite(a).all() for a in (observation, decoded, composite))
    checks = dict(finite_RGB=bool(finite), write_disjoint_protected=not bool((write & protected).any()),
                  outside_write_exact=bool(np.array_equal(composite[~write], observation[~write])),
                  protected_exact=bool(np.array_equal(composite[protected], observation[protected])),
                  inside_write_equals_same_decode=bool(np.array_equal(composite[write], decoded[write])))
    if input_mask is not None:
        checks['write_subset_input'] = not bool((write & ~mask_hw(input_mask, shape)).any())
    luminance = composite @ np.array([.2126, .7152, .0722])
    metrics = dict(write_pixels=int(write.sum()), protected_pixels=int(protected.sum()),
                   changed_pixels=int((delta > 0).sum()),
                   illegal_changed_pixels=int(((delta > 0) & ~write).sum()),
                   protected_changed_pixels=int(((delta > 0) & protected).sum()),
                   dark_write_pixels=int((write & (luminance < .05)).sum()),
                   dark_threshold_srgb_luminance=.05,
                   texture_quality_measured=False, marking_GT_available=False)
    return _record('2d_completion', checks, metrics,
                   missing=('Hidden texture and marking truth are unavailable.',))


def _neighbor_edges(keys):
    """显式世界格的四邻边；不把图像相邻像素当作世界邻接。"""
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != 2 or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError('World keys require an integer [N,2] array')
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('World keys must be unique')
    packed = np.ascontiguousarray(keys, dtype=np.int64).view([('x', '<i8'), ('y', '<i8')]).reshape(-1)
    order = np.argsort(packed)
    sorted_keys = packed[order]
    pairs = []
    for axis in range(2):
        shifted = keys.astype(np.int64, copy=True)
        shifted[:, axis] += 1
        query = np.ascontiguousarray(shifted).view(packed.dtype).reshape(-1)
        positions = np.searchsorted(sorted_keys, query)
        valid = positions < len(keys)
        valid[valid] &= sorted_keys[positions[valid]] == query[valid]
        first = np.flatnonzero(valid)
        pairs.append((first, order[positions[valid]]))
    return tuple(np.concatenate([pair[i] for pair in pairs]) for i in (0, 1))


def world_continuity(rgb, keys, valid, source=None):
    """颜色跳变和亮色线候选的断点仅是诊断量，不命名为车道线准确率。"""
    rgb = np.asarray(rgb)
    valid = np.asarray(valid, bool)
    a, b = _neighbor_edges(keys)
    selected = valid[a] & valid[b]
    a, b = a[selected], b[selected]
    delta = np.max(np.abs(rgb[a].astype(np.float64) - rgb[b].astype(np.float64)), axis=1)
    source_switch = np.asarray(source)[a] != np.asarray(source)[b] if source is not None else np.zeros(len(a), bool)
    bright = (rgb.min(-1) > .6) & ((rgb.max(-1) - rgb.min(-1)) < .2) & valid
    degree = np.bincount(np.r_[a[bright[a] & bright[b]], b[bright[a] & bright[b]]], minlength=len(rgb))
    return dict(world_neighbor_edges=len(a), edge_RGB_max_channel_mean=float(delta.mean()) if len(delta) else None,
                edge_RGB_jump_gt0p15=int((delta > .15).sum()),
                source_switch_edges=int(source_switch.sum()),
                source_switch_RGB_mean=float(delta[source_switch].mean()) if source_switch.any() else None,
                bright_neutral_candidate_cells=int(bright.sum()),
                bright_neutral_isolated_cells=int((bright & (degree == 0)).sum()),
                bright_neutral_endpoint_cells=int((bright & (degree == 1)).sum()),
                bright_neutral_definition='min(sRGB)>.6 & max(sRGB)-min(sRGB)<.2; no semantic truth',
                inferred_marking_accuracy=None)


def audit_world_material(road, provenance, *, base_road=None, atlas=None,
                         source_guards=None, source_images=None, coverage_domain=None,
                         observed_reference_srgb=None, observed_eligible=None,
                         max_retrieval_distance_m=.6):
    """source: 0未知、1实观测、2显式检索、3冻结生成；coverage与材质分开。"""
    rgb = np.asarray(provenance.get('row_rgb', road['reference_srgb']))
    n = len(rgb)
    if rgb.shape != (n, 3):
        raise ValueError('Row material RGB must have shape [N,3]')
    source = np.asarray(provenance['source'])
    unknown = np.asarray(provenance['unknown'], bool)
    legal = np.asarray(provenance['legal_write'], bool)
    known = np.asarray(provenance['known'], bool)
    confidence = np.asarray(provenance['confidence'])
    active = np.asarray(road['active'], bool)
    if any(a.shape != (n,) for a in (source, unknown, legal, known, confidence, active)):
        raise ValueError('Material and coverage fields must be row aligned')
    checks = dict(finite_row_RGB=bool(np.isfinite(rgb).all()),
                  RGB_in_unit_range=bool(((rgb >= 0) & (rgb <= 1)).all()),
                  source_enum_valid=bool(np.isin(source, list(SOURCE_NAMES)).all()),
                  unknown_matches_source=bool(np.array_equal(unknown, source == 0)),
                  known_is_observed=not bool((known & (source != 1)).any()),
                  confidence_finite_unit_range=bool(np.isfinite(confidence).all() & ((confidence >= 0) & (confidence <= 1)).all()),
                  generated_rows_legal=not bool(((source == 3) & ~legal).any()))
    observed_reference = provenance.get('observed_reference_srgb')
    checks['known_RGB_exact_observation'] = (bool(np.array_equal(rgb[known], np.asarray(observed_reference)[known]))
                                             if observed_reference is not None else None)
    if observed_reference_srgb is not None and observed_eligible is not None:
        actual_observed = source == 1
        checks['observed_source_bound_to_frozen_inputs'] = not bool((actual_observed & ~np.asarray(observed_eligible, bool)).any())
        checks['observed_RGB_exact_frozen_inputs'] = bool(np.array_equal(rgb[actual_observed], np.asarray(observed_reference_srgb)[actual_observed]))
    generated = source == 3
    source_raw = provenance.get('source_raw')
    uv = provenance.get('source_uv_native')
    checks['generated_four_neighbor_permission'] = None if generated.any() else True
    generated_rgb_max_error = None
    if source_guards is not None and source_raw is not None and uv is not None:
        raw, uv = np.asarray(source_raw), np.asarray(uv)
        ok = np.zeros(n, bool)
        for raw_id, guard in source_guards.items():
            ids = np.flatnonzero(generated & (raw == raw_id))
            if not len(ids):
                continue
            finite = np.isfinite(uv[ids]).all(-1)
            ids = ids[finite]
            xy = np.floor(uv[ids] - .5).astype(np.int64)
            in_frame = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] + 1 < guard.shape[1]) & (xy[:, 1] + 1 < guard.shape[0])
            ids, xy = ids[in_frame], xy[in_frame]
            x, y = xy.T
            ok[ids] = guard[y, x] & guard[y, x + 1] & guard[y + 1, x] & guard[y + 1, x + 1]
            if source_images is not None and raw_id in source_images and len(ids):
                original = rgb_hwc(source_images[raw_id])
                fraction = uv[ids] - .5 - xy
                fx, fy = fraction.T
                sample = (original[y, x] * ((1-fx)*(1-fy))[:, None]
                          + original[y, x+1] * (fx*(1-fy))[:, None]
                          + original[y+1, x] * ((1-fx)*fy)[:, None]
                          + original[y+1, x+1] * (fx*fy)[:, None])
                error = float(np.max(np.abs(sample - rgb[ids])))
                generated_rgb_max_error = max(generated_rgb_max_error or 0., error)
        checks['generated_four_neighbor_permission'] = not bool((generated & ~ok).any())
        if source_images is not None:
            checks['generated_RGB_matches_frozen_source'] = (generated_rgb_max_error is not None and generated_rgb_max_error <= 2e-4) if generated.any() else True
    donor = provenance.get('source_world_row')
    if donor is not None:
        ids = np.flatnonzero(source == 2)
        donor = np.asarray(donor)[ids]
        in_bounds = (donor >= 0) & (donor < n)
        checks['retrieval_donor_is_observed'] = bool(in_bounds.all() and (source[donor] == 1).all())
        checks['retrieval_RGB_exact_donor'] = bool(in_bounds.all() and np.array_equal(rgb[ids], rgb[donor]))
        distance = provenance.get('source_distance_m')
        checks['retrieval_metric_distance_bounded'] = (bool(np.isfinite(np.asarray(distance)[ids]).all() and
            (np.asarray(distance)[ids] >= 0).all() and (np.asarray(distance)[ids] <= max_retrieval_distance_m + 1e-6).all())
            if distance is not None else None if len(ids) else True)
    else:
        checks['retrieval_donor_is_observed'] = None if (source == 2).any() else True
    changed_geometry = {}
    if base_road is not None:
        updated = provenance.get('material_updated')
        if updated is not None:
            updated = np.asarray(updated, bool)
            checks['baked_updated_RGB_matches_atlas_query'] = bool(np.array_equal(np.asarray(road['reference_srgb'])[updated], rgb[updated]))
            checks['unmodified_baked_RGB_exact_parent'] = bool(np.array_equal(np.asarray(road['reference_srgb'])[~updated], np.asarray(base_road['reference_srgb'])[~updated]))
        for key in ('model_xyz', 'normal_world', 'model_from_world', 'meter_to_model'):
            if key in road and key in base_road:
                changed_geometry[key] = bool(np.array_equal(road[key], base_road[key]))
        allowed_movement = np.asarray(provenance.get('surface_interpolated', np.zeros(n, bool)), bool)
        trusted = np.asarray(provenance.get('trusted_geometry', np.ones(n, bool)), bool)
        protected = np.asarray(provenance.get('protected_geometry', np.zeros(n, bool)), bool)
        observed_initialization = np.asarray(provenance.get('observed_surface_activated', np.zeros(n, bool)), bool)
        moved = np.any(np.asarray(road['model_xyz']) != np.asarray(base_road['model_xyz']), axis=1)
        allowed_movement &= ~np.asarray(base_road['active'], bool) & ~trusted & ~protected
        checks['world_movement_only_declared_inactive_interpolation'] = not bool((moved & ~allowed_movement).any())
        checks['trusted_geometry_unchanged'] = all(np.array_equal(np.asarray(road[k])[trusted], np.asarray(base_road[k])[trusted])
            for k in ('model_xyz', 'normal_world') if k in road and k in base_road)
        state_trusted = trusted & ~observed_initialization
        checks['trusted_existing_representation_unchanged'] = all(np.array_equal(np.asarray(road[k])[state_trusted], np.asarray(base_road[k])[state_trusted])
            for k in ('gaussian_sigma_m', 'road_opacity', 'active') if k in road and k in base_road)
        if observed_initialization.any():
            checks['observed_initialization_original_inactive'] = not bool((observed_initialization & np.asarray(base_road['active'], bool)).any())
            checks['observed_initialization_is_known_supported_active'] = not bool((observed_initialization & (~known | (source != 1) | ~active | ~np.asarray(provenance['surface_supported'], bool))).any())
            checks['observed_initialization_RGB_consumes_F_exact'] = bool(np.array_equal(np.asarray(road['reference_srgb'])[observed_initialization], rgb[observed_initialization]))
            checks['observed_initialization_thickness_frozen'] = bool(np.array_equal(np.asarray(road['gaussian_sigma_m'])[observed_initialization, 2], np.asarray(base_road['gaussian_sigma_m'])[observed_initialization, 2]))
            checks['observed_initialization_coverage_request_explicit'] = ('coverage_requested' in provenance and
                not bool((observed_initialization & ~np.asarray(provenance['coverage_requested'], bool)).any()))
            boundary = provenance.get('coverage_boundary_distance_m')
            if boundary is not None:
                cap = np.minimum(.013, .30*np.asarray(boundary)[observed_initialization])
                checks['observed_initialization_metric_footprint_bounded'] = bool((np.asarray(road['gaussian_sigma_m'])[observed_initialization, :2] <= cap[:, None]+1e-9).all())
        checks['transform_unchanged'] = all(changed_geometry.get(k, False) for k in ('model_from_world', 'meter_to_model'))
        protected = provenance.get('protected_geometry')
        if protected is not None:
            protected = np.asarray(protected, bool)
            checks['protected_coverage_fields_unchanged'] = all(
                np.array_equal(np.asarray(road[k])[protected], np.asarray(base_road[k])[protected])
                for k in ('model_xyz', 'normal_world', 'active', 'gaussian_sigma_m', 'road_opacity') if k in road and k in base_road)
        if 'coverage_added' in provenance:
            added = np.asarray(provenance['coverage_added'], bool)
            checks['coverage_added_matches_active_delta'] = bool(np.array_equal(added, active & ~np.asarray(base_road['active'], bool)))
            checks['added_coverage_has_material'] = not bool((added & unknown).any())
        if 'legal_coverage' in provenance:
            changed = (active != np.asarray(base_road['active'], bool))
            for name in ('gaussian_sigma_m', 'road_opacity'):
                if name in road and name in base_road:
                    diff = np.asarray(road[name]) != np.asarray(base_road[name])
                    changed |= np.any(diff, axis=1) if diff.ndim > 1 else diff
            coverage_permission = np.asarray(provenance['legal_coverage'], bool) | observed_initialization
            checks['coverage_changes_only_legal_rows'] = not bool((changed & ~coverage_permission).any())
        if 'guarded_world_rows' in provenance:
            guarded = np.asarray(provenance['guarded_world_rows'], bool)
            checks['guarded_complete_rows_exact_parent'] = all(
                np.array_equal(np.asarray(road[key])[guarded], np.asarray(value)[guarded])
                for key, value in base_road.items() if np.asarray(value).ndim and np.asarray(value).shape[0] == n)
            checks['guarded_material_override_explicit'] = not bool((guarded & np.asarray(provenance['material_updated'], bool)).any())
            checks['guarded_coverage_override_explicit'] = all(not bool((guarded & np.asarray(provenance[key], bool)).any())
                for key in ('coverage_added', 'coverage_repaired') if key in provenance)
            for metadata, field in (('effective_row_rgb', 'reference_srgb'), ('effective_active', 'active'),
                                    ('effective_sigma_m', 'gaussian_sigma_m'), ('effective_model_xyz', 'model_xyz')):
                if metadata in provenance:
                    checks[metadata + '_matches_baked_asset'] = bool(np.array_equal(provenance[metadata], road[field]))
    domain = np.asarray(coverage_domain, bool) if coverage_domain is not None else np.ones(n, bool)
    unknown_domain = int((domain & unknown).sum())
    metrics = dict(world_rows=n, source_counts={name: int((source == key).sum()) for key, name in SOURCE_NAMES.items()},
                   requested_world_cells=int(domain.sum()), unknown_requested_cells=unknown_domain,
                   material_known_fraction=_fraction(int((domain & ~unknown).sum()), int(domain.sum())),
                   active_requested_cells=int((domain & active).sum()),
                   inactive_material_available_cells=int((domain & ~active & ~unknown).sum()),
                   active_unknown_material_cells=int((active & unknown).sum()),
                   geometry_exact_fields=changed_geometry, material_and_geometry_coverage_are_distinct=True)
    metrics['retrieval_max_distance_m'] = max_retrieval_distance_m
    metrics['generated_RGB_source_max_error'] = generated_rgb_max_error
    metrics['generated_RGB_source_tolerance'] = 2e-4
    metrics['baked_RGB_differs_atlas_query_cells'] = int(np.any(np.asarray(road['reference_srgb']) != rgb, axis=1).sum())
    metrics['baked_legacy_difference_is_not_unknown_atlas_material'] = True
    if 'guarded_world_rows' in provenance:
        guarded = np.asarray(provenance['guarded_world_rows'], bool)
        metrics['guarded_world_rows'] = int(guarded.sum())
        metrics['guarded_atlas_source_counts'] = {name: int((guarded & (source == key)).sum()) for key, name in SOURCE_NAMES.items()}
    metrics['observed_surface_initialized_rows'] = int(np.asarray(provenance.get('observed_surface_activated', np.zeros(n, bool))).sum())
    keys = provenance.get('world_keys')
    if keys is not None:
        metrics['continuity'] = world_continuity(rgb, keys, ~unknown, source)
    if atlas is not None:
        atlas_source = np.asarray(atlas['source'])
        atlas_unknown = np.asarray(atlas['unknown'], bool)
        checks['atlas_unknown_explicit'] = bool(np.array_equal(atlas_unknown, atlas_source == 0))
        checks['atlas_RGB_finite'] = bool(np.isfinite(atlas['rgb_srgb']).all())
        grid_rows = np.asarray(atlas['grid_world_row'])
        present = grid_rows >= 0
        row_ids = grid_rows[present]
        checks['atlas_grid_rows_in_range'] = bool((row_ids < n).all())
        if checks['atlas_grid_rows_in_range']:
            checks['atlas_RGB_matches_row_material'] = bool(np.array_equal(np.asarray(atlas['rgb_srgb'])[present], rgb[row_ids]))
            checks['atlas_source_matches_rows'] = bool(np.array_equal(atlas_source[present], source[row_ids]))
        metrics['atlas_shape'] = list(atlas_source.shape)
        metrics['atlas_source_counts'] = {name: int((atlas_source == key).sum()) for key, name in SOURCE_NAMES.items()}
    acceptance = 'failed' if any(v is False for v in checks.values()) else 'incomplete' if any(v is None for v in checks.values()) else 'pass'
    if unknown_domain:
        acceptance = 'failed'
    return _record('world_material_atlas', checks, metrics, acceptance=acceptance,
                   missing=('No hidden-road texture or marking truth; continuity values are diagnostics.',))


def audit_atlas_sampling(atlas_path, provenance, max_samples=2048):
    """真实保存图集上的节点、非节点、边界连续性和重载；期望RGB独立计算。"""
    from .road_asset import WorldRoadMaterialAtlas
    atlas = WorldRoadMaterialAtlas.load(atlas_path)
    reloaded = WorldRoadMaterialAtlas.load(atlas_path)
    world = np.asarray(provenance['world_xyz'])
    rgb = np.asarray(provenance['row_rgb'])
    ids = np.unique(np.linspace(0, len(world)-1, min(max_samples, len(world))).astype(np.int64))
    samples = atlas.query(world[ids])
    again = reloaded.query(world[ids])
    node_error = float(np.max(np.abs(samples['rgb_srgb'] - rgb[ids])))
    legal = atlas.legal & (atlas.source > 0)
    cells = legal[:-1, :-1] & legal[:-1, 1:] & legal[1:, :-1] & legal[1:, 1:]
    ys, xs = np.where(cells)
    choose = np.unique(np.linspace(0, len(xs)-1, min(max_samples, len(xs))).astype(np.int64)) if len(xs) else np.empty(0, np.int64)
    x, y = xs[choose], ys[choose]
    points = np.stack([atlas.surface_world_xyz[y, x], atlas.surface_world_xyz[y, x+1],
                       atlas.surface_world_xyz[y+1, x], atlas.surface_world_xyz[y+1, x+1]], axis=1).mean(1)
    colors = np.stack([atlas.rgb_srgb[y, x], atlas.rgb_srgb[y, x+1],
                       atlas.rgb_srgb[y+1, x], atlas.rgb_srgb[y+1, x+1]], axis=1).astype(np.float64)
    linear = np.where(colors <= .04045, colors/12.92, ((colors+.055)/1.055)**2.4)
    expected = linear.mean(1)
    midpoint = atlas.query(points)
    midpoint_error = float(np.max(np.abs(midpoint['rgb_linear'] - expected))) if len(points) else None
    # 横向跨格边界采样，所有六个支撑节点均合法，禁止用unknown边界制造虚假连续。
    edges = legal[:-1, :-2] & legal[1:, :-2] & legal[:-1, 1:-1] & legal[1:, 1:-1] & legal[:-1, 2:] & legal[1:, 2:]
    ey, ex = np.where(edges)
    select = np.unique(np.linspace(0, len(ex)-1, min(max_samples, len(ex))).astype(np.int64)) if len(ex) else np.empty(0, np.int64)
    ex, ey = ex[select], ey[select]
    epsilon = 1e-4
    left = (atlas.surface_world_xyz[ey, ex]+atlas.surface_world_xyz[ey+1, ex])/2
    edge = (atlas.surface_world_xyz[ey, ex+1]+atlas.surface_world_xyz[ey+1, ex+1])/2
    right = (atlas.surface_world_xyz[ey, ex+2]+atlas.surface_world_xyz[ey+1, ex+2])/2
    edge_a = atlas.query(edge*(1-epsilon)+left*epsilon)
    edge_b = atlas.query(edge*(1-epsilon)+right*epsilon)
    jump = float(np.max(np.abs(edge_a['rgb_srgb'] - edge_b['rgb_srgb']))) if len(ex) else None
    off_surface = points[:1].copy()
    if len(off_surface):
        off_surface[:, 2] += .1
    unknown_cells = ~cells
    uy, ux = np.where(unknown_cells)
    unknown_points = np.stack([atlas.surface_world_xyz[uy[:128], ux[:128]], atlas.surface_world_xyz[uy[:128], ux[:128]+1],
                               atlas.surface_world_xyz[uy[:128]+1, ux[:128]], atlas.surface_world_xyz[uy[:128]+1, ux[:128]+1]], axis=1).mean(1)
    invalid = atlas.query(unknown_points)
    checks = dict(saved_nodes_RGB_exact=bool(np.array_equal(samples['rgb_srgb'], rgb[ids])),
                  saved_nodes_unknown_flags_exact=bool(np.array_equal(samples['unknown'], np.asarray(provenance['unknown'])[ids])),
                  reload_query_exact=all(np.array_equal(samples[key], again[key]) for key in samples),
                  valid_midpoints_available=bool(len(points)), valid_midpoints_known=not bool(midpoint['unknown'].any()),
                  midpoint_linear_light_RGB_correct=bool(midpoint_error is not None and midpoint_error < 2e-6),
                  cross_cell_continuity=bool(jump is not None and jump < 1e-3),
                  unknown_neighbor_midpoints_rejected=bool(invalid['unknown'].all()),
                  off_surface_rejected=bool(atlas.query(off_surface)['unknown'].all()) if len(off_surface) else None)
    metrics = dict(node_samples=len(ids), node_RGB_max_error=node_error, midpoint_samples=len(points),
                   midpoint_linear_RGB_max_error=midpoint_error, edge_samples=len(ex),
                   edge_side_offset_texel=epsilon, edge_sRGB_max_jump=jump,
                   unknown_neighbor_samples=len(unknown_points), surface_offset_test_m=.1,
                   scope='deterministic saved-world samples; no video-frame input; not texture truth')
    return _record('continuous_world_sampling', checks, metrics)


def audit_training_asset(phase, reference, trained, allowed_world_rows, *, observed=None,
                         guarded=None, max_tangent_scale_factor=None, max_color_delta=None,
                         allowed_tangent_world_rows=None, tangent_upper_factors=None,
                         tangent_boundary_distance_m=None):
    """阶段1/2实际导出资产的独立差分，不以requires_grad清单代替冻结验证。"""
    if phase not in (1, 2):
        raise ValueError('Only module2 phases 1 and 2 are supported')
    n = len(reference['active'])
    allowed = np.asarray(allowed_world_rows, bool)
    guard = np.zeros(n, bool) if guarded is None else np.asarray(guarded, bool)
    known = np.zeros(n, bool) if observed is None else np.asarray(observed, bool)
    if any(value.shape != (n,) for value in (allowed, guard, known)):
        raise ValueError('Training permissions must use the full persistent world-row topology')
    permitted = allowed & ~guard
    tangent_permitted = (allowed if allowed_tangent_world_rows is None else np.asarray(allowed_tangent_world_rows, bool)) & ~guard
    if tangent_permitted.shape != (n,):
        raise ValueError('Tangent permissions must use the full world-row topology')
    if phase == 2:
        permitted &= ~known
    allowed_fields = {'road_opacity'} if phase == 1 else {'reference_srgb'}
    if phase == 1 and max_tangent_scale_factor is not None:
        if not np.isfinite(max_tangent_scale_factor) or max_tangent_scale_factor < 1:
            raise ValueError('Tangent scale bounds must be explicit and finite')
        allowed_fields.add('gaussian_sigma_m')
    checks = dict(asset_fields_unchanged=set(reference) == set(trained))
    field_deltas = {}
    for key, value in reference.items():
        if key not in trained:
            continue
        before, after = np.asarray(value), np.asarray(trained[key])
        checks[key + '_shape_dtype_fixed'] = before.shape == after.shape and before.dtype == after.dtype
        if not checks[key + '_shape_dtype_fixed']:
            continue
        if np.issubdtype(after.dtype, np.number):
            checks[key + '_finite'] = bool(np.isfinite(after).all())
        diff = after != before
        if diff.ndim and diff.shape[0] == n:
            row_changed = np.any(diff.reshape(n, -1), axis=1)
            field_deltas[key] = dict(changed_rows=int(row_changed.sum()),
                                     max_abs=float(np.abs(after.astype(np.float64)-before.astype(np.float64)).max()))
            if key in allowed_fields:
                field_permission = tangent_permitted if key == 'gaussian_sigma_m' else permitted
                checks[key + '_only_permitted_rows'] = not bool((row_changed & ~field_permission).any())
            else:
                checks[key + '_frozen_exact'] = not bool(row_changed.any())
            checks[key + '_guarded_exact'] = not bool((row_changed & guard).any())
        else:
            checks[key + '_frozen_exact'] = not bool(diff.any())
    opacity = np.asarray(trained['road_opacity'])
    checks['opacity_in_unit_range'] = bool(((opacity >= 0) & (opacity <= 1)).all())
    color = np.asarray(trained['reference_srgb'])
    checks['RGB_in_unit_range'] = bool(((color >= 0) & (color <= 1)).all())
    tangent_bound_excess = None
    if phase == 1 and max_tangent_scale_factor is not None:
        before, after = (np.asarray(a['gaussian_sigma_m']) for a in (reference, trained))
        checks['normal_thickness_exact'] = bool(np.array_equal(before[:, 2], after[:, 2]))
        upper = np.full(n, max_tangent_scale_factor) if tangent_upper_factors is None else np.asarray(tangent_upper_factors)
        checks['tangent_scope_within_opacity_scope'] = not bool((tangent_permitted & ~permitted).any())
        checks['declared_tangent_upper_factors_valid'] = bool(upper.shape == (n,) and np.isfinite(upper).all() and
            (upper >= 1).all() and (upper <= max_tangent_scale_factor+1e-7).all())
        # 光栅尺度与factor为float32，持久米制表为float64；显式允许两次舍入，物理3σ门仍严格。
        export_rounding = 2*np.finfo(np.float32).eps*np.maximum(np.abs(before[:, :2]),np.abs(after[:, :2])) + 1e-12
        tangent_bound_excess = float(np.max(after[:, :2]-before[:, :2]*upper[:, None]))
        checks['tangent_scale_bounded'] = bool((after[:, :2] <= before[:, :2]*upper[:, None]+export_rounding).all() and
                                              (after[:, :2] >= before[:, :2]-1e-12).all())
        ratio = after[:, :2] / np.maximum(before[:, :2], 1e-30)
        checks['tangent_axes_share_single_expansion_factor'] = bool(np.allclose(ratio[:, 0], ratio[:, 1], atol=1e-7, rtol=0))
        if tangent_boundary_distance_m is not None:
            distance = np.asarray(tangent_boundary_distance_m)
            checks['final_3sigma_inside_physical_support'] = bool((3*np.max(after[tangent_permitted, :2], axis=1) < distance[tangent_permitted]).all())
    if phase == 2:
        checks['observed_RGB_exact'] = bool(np.array_equal(color[known], np.asarray(reference['reference_srgb'])[known]))
        if max_color_delta is not None:
            checks['local_color_delta_bounded'] = bool((np.abs(color-np.asarray(reference['reference_srgb'])) <= max_color_delta+1e-7).all())
    return _record('module2_phase_' + str(phase) + '_asset_delta', checks,
                   dict(permitted_world_rows=int(permitted.sum()), field_deltas=field_deltas,
                        permitted_tangent_world_rows=int(tangent_permitted.sum()) if max_tangent_scale_factor is not None else None,
                        tangent_bound_max_raw_excess_m=tangent_bound_excess,
                        tangent_bound_export_relative_tolerance=2*float(np.finfo(np.float32).eps) if max_tangent_scale_factor is not None else None,
                        max_tangent_scale_factor=max_tangent_scale_factor, max_color_delta=max_color_delta,
                        evaluation='actual saved arrays; optimizer routing and lifecycle audited separately'))


def audit_local_texture_correction(base_atlas, effective_atlas, correction, provenance,
                                   reference_road, trained_road, maximum_linear_delta=.08):
    """独立重建baseF+线性delta，再核对effective图集和baked rows；基础来源另审。"""
    grid = np.asarray(base_atlas['grid_world_row'])
    base_rgb = np.asarray(base_atlas['rgb_srgb'])
    delta = np.asarray(correction['delta_linear'])
    eligible = np.asarray(correction['eligible'], bool)
    n = len(reference_road['active'])
    if delta.shape != base_rgb.shape or eligible.shape != grid.shape:
        raise ValueError('Correction requires the original world grid and HWC linear RGB delta')
    row_eligible = np.asarray(provenance['texture_correction_eligible'], bool)
    known = np.asarray(provenance['known'], bool)
    guard = np.asarray(provenance.get('guarded_world_rows', np.zeros(n, bool)), bool)
    allowed = (np.asarray(provenance['requested'], bool) & np.asarray(provenance['surface_supported'], bool)
               & ~np.asarray(provenance['unknown'], bool) & ~known & ~guard
               & np.asarray(reference_road['active'], bool) & np.asarray(provenance['material_updated'], bool))
    checks = dict(original_grid_topology=bool(np.array_equal(np.sort(grid.ravel()), np.arange(n))),
                  finite_linear_delta=bool(np.isfinite(delta).all()),
                  linear_delta_bound=bool((np.abs(delta) <= maximum_linear_delta+1e-7).all()),
                  correction_grid_matches_saved_world=bool(np.array_equal(correction['grid_world_row'], grid)),
                  correction_world_transform_fixed=bool(np.array_equal(correction['world_from_uv'], base_atlas['world_from_uv'])),
                  correction_surface_fixed=bool(np.array_equal(correction['surface_world_xyz'], base_atlas['surface_world_xyz'])),
                  grid_and_row_eligible_match=bool(np.array_equal(eligible, row_eligible[grid])),
                  correction_only_authorized_rows=not bool((row_eligible & ~allowed).any()),
                  delta_zero_outside_eligible=not bool(np.any(delta[~eligible] != 0)),
                  observed_delta_exact_zero=not bool(np.any(delta[known[grid]] != 0)),
                  guarded_delta_exact_zero=not bool(np.any(delta[guard[grid]] != 0)))
    for key in base_atlas:
        if key != 'rgb_srgb':
            checks['base_metadata_fixed_' + key] = bool(np.array_equal(base_atlas[key], effective_atlas[key]))
    base64 = base_rgb.astype(np.float64)
    linear = np.where(base64 <= .04045, base64/12.92, ((base64+.055)/1.055)**2.4)
    corrected_linear = np.clip(linear+delta, 0, 1)
    expected = np.where(corrected_linear <= .0031308, corrected_linear*12.92,
                        1.055*corrected_linear**(1/2.4)-.055).astype(np.float32)
    expected[~eligible] = base_rgb[~eligible]
    atlas_error = float(np.max(np.abs(expected-np.asarray(effective_atlas['rgb_srgb']))))
    checks['effective_RGB_reconstructs_base_plus_delta'] = atlas_error < 2e-6
    checks['untrained_texels_RGB_exact_base'] = bool(np.array_equal(np.asarray(effective_atlas['rgb_srgb'])[~eligible], base_rgb[~eligible]))
    expected_row = np.empty((n, 3), np.float32)
    base_row = np.empty_like(expected_row)
    row_delta = np.empty_like(expected_row)
    expected_row[grid.ravel()] = expected.reshape(-1, 3)
    base_row[grid.ravel()] = base_rgb.reshape(-1, 3)
    row_delta[grid.ravel()] = delta.reshape(-1, 3)
    checks['base_row_provenance_matches_immutable_atlas'] = bool(np.array_equal(provenance['base_row_rgb'], base_row))
    checks['row_delta_matches_saved_correction'] = bool(np.array_equal(provenance['texture_delta_linear'], row_delta))
    checks['effective_row_query_reconstructs'] = bool(np.max(np.abs(np.asarray(provenance['row_rgb'])-expected_row)) < 2e-6)
    baked_error = float(np.max(np.abs(np.asarray(trained_road['reference_srgb'])[row_eligible]-expected_row[row_eligible]))) if row_eligible.any() else 0.
    checks['baked_RGB_reconstructs_effective_world_query'] = baked_error < 2e-6
    checks['effective_baked_metadata_exact'] = bool(np.array_equal(provenance['effective_row_rgb'], trained_road['reference_srgb']))
    asset_delta = audit_training_asset(2, reference_road, trained_road, row_eligible, observed=known, guarded=guard)
    checks['only_authorized_asset_fields_changed'] = asset_delta['technical_status'] == 'pass'
    record = _record('module2_phase_2_local_texture_correction', checks,
        dict(eligible_texels=int(eligible.sum()), nonzero_delta_texels=int(np.any(delta != 0, axis=-1).sum()),
             maximum_saved_linear_delta=float(np.abs(delta).max()), declared_linear_delta_bound=maximum_linear_delta,
             effective_atlas_RGB_reconstruction_max_error=atlas_error, baked_RGB_reconstruction_max_error=baked_error,
             foundation_source_provenance_unchanged=all(value for key, value in checks.items() if key.startswith('base_metadata_fixed_')),
             interpretation='learned self-consistency appearance correction; not hidden road truth'))
    record['asset_delta'] = asset_delta
    return record


def audit_render(road_only_alpha, road_alphaT, static_alphaT, dynamic_alphaT, total_alpha,
                 domain, *, rgb=None, road_only_rgb=None, parts=None, protected=None,
                 baseline_rgb=None, branch_depth=None, min_supported_fraction=.99,
                 max_gap_pixels=0, protected_tolerance=1 / 255, tolerance=1e-5):
    """同一深度排序的αT验证与道路单层支撑分离；非零静态贡献不等于归属错误。"""
    shape = np.asarray(domain).squeeze().shape
    domain = mask_hw(domain, shape)
    only, road, static, dynamic, total = [alpha_hw(v, shape) for v in
                                         (road_only_alpha, road_alphaT, static_alphaT, dynamic_alphaT, total_alpha)]
    arrays = (only, road, static, dynamic, total)
    finite = all(np.isfinite(v).all() for v in arrays)
    conservation = float(np.max(np.abs(road + static + dynamic - total)))
    checks = dict(float_arrays_finite=bool(finite),
                  alpha_in_unit_range=bool(all((v >= -tolerance).all() and (v <= 1 + tolerance).all() for v in arrays)),
                  joint_branch_alphaT_sum=bool(conservation < tolerance),
                  joint_road_not_above_road_only=bool(np.all(road <= only + tolerance)))
    masks = dict(gap=domain & (only < .05), thin=domain & (only >= .05) & (only <= .9),
                 supported=domain & (only > .9),
                 supported_low_visible=domain & (only > .9) & (road < .5),
                 static_dominated=domain & (only > .9) & (road < .5) & (static >= dynamic),
                 dynamic_dominated=domain & (only > .9) & (road < .5) & (dynamic > static))
    n = int(domain.sum())
    metrics = dict(domain_pixels=n, **{key + '_pixels': int(value.sum()) for key, value in masks.items()},
                   supported_fraction=_fraction(int(masks['supported'].sum()), n),
                   gap_fraction=_fraction(int(masks['gap'].sum()), n),
                   road_only_alpha_mean=_mean(only, domain), road_alphaT_mean=_mean(road, domain),
                   static_alphaT_mean=_mean(static, domain), dynamic_alphaT_mean=_mean(dynamic, domain),
                   alphaT_conservation_max_error=conservation,
                   road_alphaT_above_only_pixels=int((road > only + tolerance).sum()),
                   visibility_interpretation='same-sort visible contribution, not ownership or hidden-road truth')
    if rgb is not None:
        rgb = rgb_hwc(rgb)
        checks['RGB_finite'] = bool(np.isfinite(rgb).all())
        masks['dark_joint'] = domain & ((rgb @ [.2126, .7152, .0722]) < .05)
        metrics['dark_joint_pixels'] = int(masks['dark_joint'].sum())
        for key in ('gap', 'thin', 'supported_low_visible'):
            metrics['dark_joint_' + key + '_pixels'] = int((masks['dark_joint'] & masks[key]).sum())
    if road_only_rgb is not None:
        road_rgb = rgb_hwc(road_only_rgb)
        # RGB为预乘；用alpha恢复材质亮度只在充分覆盖处解释。
        normalized = road_rgb / np.maximum(only[..., None], 1e-8)
        masks['dark_supported_material'] = masks['supported'] & ((normalized @ [.2126, .7152, .0722]) < .05)
        metrics['dark_supported_material_pixels'] = int(masks['dark_supported_material'].sum())
        checks['road_only_RGB_finite'] = bool(np.isfinite(road_rgb).all())
    if parts is not None and rgb is not None:
        summed = sum(rgb_hwc(v) for v in parts.values())
        error = float(np.max(np.abs(summed - rgb)))
        checks['joint_RGB_parts_conserve'] = bool(error < tolerance)
        metrics['RGB_conservation_max_error'] = error
    else:
        checks['joint_RGB_parts_conserve'] = None
    preservation_failed = False
    if protected is not None and baseline_rgb is not None and rgb is not None:
        metrics['protected'] = rgb_difference(rgb, baseline_rgb, protected, protected_tolerance)
        preservation_failed = metrics['protected']['over_tolerance_pixels'] > 0
    else:
        checks['protected_vehicle_comparison_available'] = None
    depth_metrics = {}
    if branch_depth:
        for name, depth in branch_depth.items():
            depth = np.asarray(depth).squeeze()
            finite_positive = np.isfinite(depth) & (depth > 0)
            depth_metrics[name] = dict(valid_pixels=int((domain & finite_positive).sum()),
                                       conditional_depth_mean=_mean(depth, domain & finite_positive))
    metrics['branch_depth'] = depth_metrics or None
    metrics['physical_depth_order_verified'] = False
    missing = ['Conservation alone does not verify physical sorting or correct ownership.']
    if not branch_depth:
        missing.append('No branch depth arrays provided; physical order remains unknown.')
    road_status = 'incomplete' if not n else ('pass' if int(masks['gap'].sum()) <= max_gap_pixels and metrics['supported_fraction'] >= min_supported_fraction else 'failed')
    joint_status = 'failed' if preservation_failed or any(v is False for v in checks.values()) else 'incomplete'
    result = _record('joint_rendering', checks, metrics, acceptance=joint_status, missing=missing)
    result['road_only_status'] = road_status
    result['road_only_thresholds'] = dict(gap_alpha_lt=.05, support_alpha_gt=.9,
                                          max_gap_pixels=max_gap_pixels, min_supported_fraction=min_supported_fraction)
    result['protected_status'] = 'failed' if preservation_failed else 'pass' if 'protected' in metrics else 'incomplete'
    return result, masks


def save_artifacts(output_dir, record, *, rgb=None, masks=None, scalar_maps=None, max_width=640):
    """浮点量先评分后制图；轻图从不回流为评分输入。"""
    from PIL import Image
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'metrics.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if rgb is not None:
        value = np.uint8(np.clip(rgb_hwc(rgb), 0, 1) * 255 + .5)
        image = Image.fromarray(value)
        if image.width > max_width:
            image = image.resize((max_width, round(image.height * max_width / image.width)), Image.Resampling.LANCZOS)
        image.save(directory / 'rgb.webp', quality=90)
    if masks:
        colors = dict(gap=(255, 77, 103), thin=(255, 186, 73), supported_low_visible=(57, 198, 230),
                      static_dominated=(157, 120, 244), dynamic_dominated=(250, 110, 187),
                      dark_supported_material=(140, 140, 140))
        shape = next(iter(masks.values())).shape
        overlay = np.uint8(np.clip(rgb_hwc(rgb), 0, 1) * 255 + .5) if rgb is not None else np.full((*shape, 3), 15, np.uint8)
        for name in ('gap', 'thin', 'supported_low_visible'):
            if name in masks:
                overlay[masks[name]] = np.uint8(.3 * overlay[masks[name]] + .7 * np.asarray(colors[name]))
        image = Image.fromarray(overlay)
        if image.width > max_width:
            image = image.resize((max_width, round(image.height * max_width / image.width)), Image.Resampling.NEAREST)
        image.save(directory / 'support_visibility.png')
    for name, value in (scalar_maps or {}).items():
        image = Image.fromarray(np.uint8(np.clip(np.asarray(value).squeeze(), 0, 1) * 255 + .5))
        if image.width > max_width:
            image = image.resize((max_width, round(image.height * max_width / image.width)), Image.Resampling.NEAREST)
        image.save(directory / (name + '.png'))

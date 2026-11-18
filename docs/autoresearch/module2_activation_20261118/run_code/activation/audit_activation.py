"""激活覆盖修复的独立CPU验收；不导入Torch/native，不启动GPU。"""
from pathlib import Path
import argparse
import json
import math
import numpy as np

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
CPU = CAMPAIGN / 'road_capacity_cpu_20261117_r1'
PREVIOUS = CAMPAIGN / 'road_capacity_gpu_20261118_r1'
CURRENT = CAMPAIGN / 'road_activation_20261118_r1'
PARENT = CAMPAIGN / 'road_asset_20261116_r1/phase2/formal_r1/asset'
FRAMES = (86, 110, 134, 179, 198)
LEDGER = ('source', 'known', 'unknown', 'confidence', 'observed_reference_srgb',
          'source_world_row', 'source_distance_m', 'source_raw', 'source_uv_native',
          'row_rgb', 'base_row_rgb', 'texture_delta_linear', 'world_xyz', 'world_keys',
          'atlas_uv', 'surface_supported', 'original_surface_supported',
          'original_world_xyz', 'original_active', 'trusted_geometry', 'protected_geometry',
          'old_requested', 'requested', 'W_world')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (keys if keys is not None else z.files)}


def equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b))


def changed_rows(a, b):
    if a.shape != b.shape:
        raise ValueError('字段形状发生变化')
    same = a == b
    if a.dtype.kind in 'fc':
        same |= np.isnan(a) & np.isnan(b)
    return ~same if same.ndim == 1 else ~same.reshape(len(a), -1).all(1)


def masks():
    result = {}
    K = load(PREVIOUS / 'check1/native_r1/K_masks.npz')
    for raw in FRAMES:
        D = load(CPU / f'ideal_targets_r2/constant/{raw:03d}.npz', ['mask'])['mask']
        row = dict(D=D, K=K[f'{raw:03d}'])
        if raw in (86, 110):
            row['full_S'] = load(CPU / f'supervision_v4_candidate_r1/{raw:03d}.npz', ['S_target_shadow'])['S_target_shadow']
            row['fixed_low_S'] = load(PREVIOUS / f'check1/remaining_floor_cpu_r1/{raw:03d}_source_state_maps.npz', ['fixed_low_S'])['fixed_low_S']
        result[raw] = row
    assert sum(int(r['D'].sum()) for r in result.values()) == 38416
    assert sum(int(r['K'].sum()) for r in result.values()) == 77266
    assert sum(int(r.get('full_S', np.zeros(0)).sum()) for r in result.values()) == 3187
    assert sum(int(r.get('fixed_low_S', np.zeros(0)).sum()) for r in result.values()) == 965
    return result


def prepare(output):
    fixed = masks()
    result = dict(schema='road_activation_independent_QA_plan.v1', GPU_executed=False,
                  parent_asset=str(PARENT), frames=list(FRAMES),
                  fixed_domains=dict(D=38416, full_S=3187, low_S=965, K=77266),
                  per_frame=[dict(raw=raw, **{k: int(v.sum()) for k, v in row.items()}) for raw, row in fixed.items()],
                  control_chain=[
                      dict(stage='parent', optimizer_updates=0, role='canonical oldS2'),
                      dict(stage='A0', optimizer_updates=0, role='active-only coverage mechanism control; legacy colors may be unqualified'),
                      dict(stage='A1', optimizer_updates=0, role='same activation plus new-row lawful observed material initialization'),
                      dict(stage='color_refinement', optimizer_updates=None, status='not registered or executed in this first round')],
                  asset_gates=['no topology creation or deletion; only old inactive rows become active',
                               'all-row xyz/normal/sigma/opacity/minimumdepth/transform exact',
                               'original active RGB exact; observed/source prior ledger exact',
                               'A0 all-row RGB exact; A1 RGB writes only accepted new rows and equals original observed reference',
                               'effective coverage request includes accepted new rows without expanding old generated-color W'],
                  native_gates=['new rows actual native final-state K contribution/near qualification',
                                'joint full K77266 RGB exact zero vs parent in all five views',
                                'actual public persistent loader renders RGB/alpha <=1e-6; RGB8 separately reported',
                                'A0/A1 alpha exact; activation delta separately from material RGB delta',
                                'report road-only alpha support separately from joint road alphaT and fixed full/low S'],
                  failure_rules='fail explicit violated invariant; incomplete for missing native evidence; no inferred human verdict',
                  human_verdict=None, visual_verdict=None, production_promoted=False)
    write(output, result)
    return result


def audit_asset(parent, candidate, stage, proposal_path, proposal_key, output):
    parent, candidate = Path(parent), Path(candidate)
    a = load(parent / 'road.npz')
    b = load(candidate / 'road.npz')
    assert a.keys() == b.keys(), 'road字段集合变化需显式审核'
    n = len(a['active'])
    old = a['active']
    new = b['active'] & ~old
    proposal = load(proposal_path, [proposal_key])[proposal_key].astype(bool)
    assert proposal.shape == old.shape
    checks = dict(no_old_active_deactivation=bool(b['active'][old].all()),
                  new_rows_only_from_proposal=bool(not np.any(new & ~proposal)))
    fields = {}
    for key in a:
        x, y = a[key], b[key]
        entry = dict(dtype_equal=x.dtype == y.dtype, shape_equal=x.shape == y.shape, all_equal=equal(x, y))
        if x.ndim and len(x) == n:
            change = changed_rows(x, y)
            entry.update(changed_rows=int(change.sum()), old_active_changed_rows=int((change & old).sum()),
                         new_active_changed_rows=int((change & new).sum()), other_changed_rows=int((change & ~new & ~old).sum()))
            if key == 'active':
                checks['active_changes_are_only_new_rows'] = bool(np.array_equal(change, new))
            elif key == 'reference_srgb' and stage == 'activation_material':
                checks['RGB_only_new_rows'] = bool(not np.any(change & ~new))
            else:
                checks[key + '_all_rows_frozen'] = entry['all_equal']
        else:
            checks[key + '_frozen'] = entry['all_equal']
        fields[key] = entry
    ledger = {}
    with np.load(parent / 'provenance.npz', allow_pickle=False) as p, np.load(candidate / 'provenance.npz', allow_pickle=False) as q:
        for key in LEDGER:
            if key in p.files:
                ledger[key] = key in q.files and equal(p[key], q[key])
        checks['prior_observed_and_source_ledger_frozen'] = all(ledger.values())
        source = p['source'][new]
        known = p['known'][new]
        unknown = p['unknown'][new]
        observed = p['observed_reference_srgb'][new]
        expected_material = observed if np.all((source == 1) & known) else p['effective_row_rgb'][new]
        actual_material = b['reference_srgb'][new]
        material_error = float(np.max(np.abs(actual_material - expected_material))) if new.any() else 0.0
        source_qualified = bool(np.all((source > 0) & ~unknown))
        material_qualified = source_qualified and bool(np.array_equal(actual_material, expected_material))
        if stage == 'activation_material':
            checks['new_rows_use_qualified_observed_material'] = bool(np.all((source == 1) & known & ~unknown)) and material_qualified
        effective_key = 'effective_coverage_requested' if 'effective_coverage_requested' in q.files else 'coverage_requested'
        effective_requested = q[effective_key] if effective_key in q.files else np.zeros(n, bool)
        if stage == 'activation_material':
            checks['new_rows_effective_coverage_requested'] = bool(effective_requested[new].all())
        source_counts = {str(int(k)): int((source == k).sum()) for k in np.unique(source)}
    status = 'pass' if all(checks.values()) else 'failed'
    result = dict(schema='road_activation_asset_independent_audit.v1', stage=stage, status=status,
                  parent=str(parent), candidate=str(candidate), GPU_executed=False, new_active_rows=int(new.sum()),
                  source_counts_new=source_counts, checks=checks, field_differences=fields, prior_ledger=ledger,
                  material=dict(qualified=material_qualified, max_abs_vs_qualified_source=material_error,
                                role='intentional legacy-color mechanism control' if stage == 'active_only' else 'qualified material candidate'),
                  native_acceptance='incomplete_pending_actual_newK_and_public_reload_and_joint_render',
                  human_verdict=None, visual_verdict=None, production_promoted=False)
    write(output, result)
    return result


def squeeze(array):
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def metrics(alpha, mask):
    values = alpha[mask].astype(np.float64)
    return dict(pixels=int(mask.sum()), gap=int((values < .05).sum()),
                exact_zero=int((values == 0).sum()), thin=int(((values >= .05) & (values <= .9)).sum()),
                supported=int((values > .9).sum()), mean_alpha=float(values.mean()) if len(values) else None,
                one_minus_alpha_MSE=float(((1 - values) ** 2).mean()) if len(values) else None)


def get_array(root, raw):
    root = Path(root)
    for p in (root / f'{raw:03d}/arrays.npz', root / f'{raw:03d}.npz'):
        if p.exists():
            return load(p), str(p)
    raise FileNotFoundError(str(root / f'{raw:03d}/arrays.npz'))


def audit_renders(plan_path, output):
    plan = read(plan_path)
    fixed = masks()
    baseline = next(s for s in plan['stages'] if s['name'] == plan['baseline_name'])
    base = {r: get_array(baseline['rendered'], r)[0] for r in FRAMES}
    rows = []
    native_missing = []
    checks = {}
    stage_arrays = {}
    for stage in plan['stages']:
        name = stage['name']
        stage_arrays[name] = {}
        for raw in FRAMES:
            arrays, path = get_array(stage['rendered'], raw)
            stage_arrays[name][raw] = arrays
            alpha = squeeze(arrays['road_only_alpha'])
            base_alpha = squeeze(base[raw]['road_only_alpha'])
            P = arrays.get('rgb', arrays.get('joint_rgb'))
            B = base[raw].get('rgb', base[raw].get('joint_rgb'))
            if P is None or B is None:
                native_missing.append(f'{name}:{raw}:jointRGB')
                Kmax = None
                Kchanged = None
            else:
                diff = np.abs(P[:, fixed[raw]['K']] - B[:, fixed[raw]['K']])
                Kmax = float(diff.max()) if diff.size else 0.0
                Kchanged = int((diff != 0).any(0).sum())
                checks[f'{name}:{raw}:joint_K_exact'] = Kmax == 0
            row = dict(stage=name, raw=raw, arrays=path,
                       regions={key: metrics(alpha, mask) for key, mask in fixed[raw].items() if key != 'K'},
                       alpha_delta_min=float((alpha - base_alpha).min()),
                       alpha_delta_max=float((alpha - base_alpha).max()), K_max=Kmax, K_changed_pixels=Kchanged)
            if 'road_alphaT' in arrays:
                roadT = squeeze(arrays['road_alphaT'])
                row['joint_road_alphaT'] = {k: metrics(roadT, mask) for k, mask in fixed[raw].items() if k != 'K'}
            else:
                native_missing.append(f'{name}:{raw}:joint_road_alphaT')
            if all(k in arrays for k in ['static_alphaT', 'dynamic_alphaT', 'road_alphaT', 'total_alpha']):
                error = float(np.abs(sum(squeeze(arrays[k]) for k in ['static_alphaT', 'dynamic_alphaT', 'road_alphaT'])
                                     - squeeze(arrays['total_alpha'])).max())
                row['joint_branch_alphaT_conservation_error'] = error
                checks[f'{name}:{raw}:alphaT_conservation'] = error <= 3e-6
            rows.append(row)
        reload_path = stage.get('public_reload')
        if name != plan['baseline_name']:
            if not reload_path or not Path(reload_path).exists():
                native_missing.append(name + ':public_reload_receipt')
            else:
                proof = read(reload_path)
                checks[name + ':actual_public_loader'] = proof.get('public_loader') is True and proof.get('GPU_executed') is True
                checks[name + ':public_reload_tolerance'] = len(proof.get('rows', [])) == 5 and all(
                    r['RGB_max'] <= 1e-6 and r['alpha_max'] <= 1e-6 for r in proof.get('rows', []))
    comparisons = []
    for first, second in plan.get('alpha_equal_pairs', []):
        for raw in FRAMES:
            a = squeeze(stage_arrays[first][raw]['road_only_alpha'])
            b = squeeze(stage_arrays[second][raw]['road_only_alpha'])
            same = equal(a, b)
            checks[f'{first}->{second}:{raw}:coverage_frozen'] = same
            comparisons.append(dict(before=first, after=second, raw=raw, alpha_exact=same,
                                    alpha_max_abs=float(np.abs(a - b).max()),
                                    road_RGB_max_abs=float(np.abs(stage_arrays[first][raw]['road_only_rgb']
                                                                   - stage_arrays[second][raw]['road_only_rgb']).max())))
    guard_path = plan.get('new_rows_K_evidence')
    if not guard_path or not Path(guard_path).exists():
        native_missing.append('new_rows_actual_native_final_state_K_guard')
    else:
        guard = read(guard_path)
        checks['new_rows_guard_uses_actual_native'] = guard.get('actual_native') is True
        checks['new_rows_guard_matches_final_opacity_and_sigma'] = guard.get('final_state') is True
        checks['new_rows_K_mass_zero'] = guard.get('accepted_new_rows_K_mass_max') == 0
    status = 'failed' if not all(checks.values()) else 'incomplete' if native_missing else 'pass'
    result = dict(schema='road_activation_float_render_independent_audit.v1', status=status,
                  GPU_executed_by_auditor=False, checks=checks, missing=native_missing,
                  rows=rows, comparisons=comparisons,
                  fixed_domain_counts=dict(D=38416, full_S=3187, fixed_low_S=965, K=77266),
                  attribution='activation changes road-only alpha; A1-A0 RGB is lawful material initialization, never training benefit',
                  human_verdict=None, visual_verdict=None, production_promoted=False)
    write(output, result)
    return result


def audit_native_delivery(output):
    """独立读取本轮实际交付的父/A0/A1浮点，不请求新的原生前向。"""
    root = CURRENT / 'native_r1'
    fixed = masks()
    source = read(root / 'results.json')
    reload = read(root / 'public_reload.json')
    guard = read(root / 'new_packet_guard.json')
    guard_arrays = load(root / 'new_packet_guards.npz')
    original = load(PARENT / 'road.npz', ['active'])['active']
    final = load(root / 'asset_A1_guarded/road.npz', ['active'])['active']
    new = final & ~original
    checks = dict(real_native_receipt=source['GPU_executed'] is True,
                  no_optimizer_updates=source['optimizer_updates'] == 0,
                  active_world_mapping_exact=bool(np.array_equal(guard_arrays['active_world_rows'], np.flatnonzero(final))),
                  no_new_rows_in_conservative_K_support=bool(not guard_arrays['world_K_guard'][new].any()),
                  no_new_rows_in_actual_packet_near_guard=bool(not guard_arrays['world_near_guard'][new].any()),
                  K_envelope_uses_actual_CUDA_radii='actual native CUDA radii' in guard['K']['method'],
                  public_loader_real=reload.get('public_loader') is True and reload.get('GPU_executed') is True,
                  five_frame_reload=len(reload['rows']) == 5,
                  reload_RGB_alpha_le_1e_6=all(r['RGB_max'] <= 1e-6 and r['alpha_max'] <= 1e-6 for r in reload['rows']))
    rows = []
    for raw in FRAMES:
        candidate = load(root / f'{raw:03d}_native.npz')
        active_only = load(root / f'A0/{raw:03d}_native.npz')
        parent = load(PARENT.parent / f'rendered/{raw:03d}/arrays.npz', ['rgb', 'road_only_alpha', 'road_only_rgb'])
        checks[f'{raw}:cached_parent_road_alpha_exact'] = equal(candidate['baseline_road_alpha'], squeeze(parent['road_only_alpha']))
        checks[f'{raw}:cached_parent_road_RGB_exact'] = equal(candidate['baseline_road_rgb'], parent['road_only_rgb'])
        checks[f'{raw}:A0_A1_road_alpha_exact'] = equal(candidate['road_alpha'], active_only['road_alpha'])
        checks[f'{raw}:Dmask_exact'] = equal(candidate['D_full'], fixed[raw]['D'])
        checks[f'{raw}:Kmask_exact'] = equal(candidate['K'], fixed[raw]['K'])
        protected = fixed[raw]['K']
        K1 = float(np.abs(candidate['joint_rgb'][:, protected] - parent['rgb'][:, protected]).max())
        K0 = float(np.abs(active_only['joint_rgb'][:, protected] - parent['rgb'][:, protected]).max())
        checks[f'{raw}:A1_joint_K_exact'] = K1 == 0
        checks[f'{raw}:A0_joint_K_exact'] = K0 == 0
        row = dict(raw=raw, K_pixels=int(protected.sum()), A0_joint_K_max=K0, A1_joint_K_max=K1, regions={})
        before, after = candidate['baseline_road_alpha'], candidate['road_alpha']
        for name, mask in fixed[raw].items():
            if name == 'K':
                continue
            b, a = metrics(before, mask), metrics(after, mask)
            delta = after.astype(np.float64) - before.astype(np.float64)
            decreases = mask & (delta < -1e-6)
            row['regions'][name] = dict(before=b, after=a,
                                       newly_supported=int((mask & (before <= .9) & (after > .9)).sum()),
                                       lost_supported=int((mask & (before > .9) & (after <= .9)).sum()),
                                       alpha_decreased_gt_1e_6=int(decreases.sum()),
                                       maximum_observed_alpha_drop=float(np.maximum(-delta[mask], 0).max()) if mask.any() else 0.0)
            if name == 'full_S':
                checks[f'{raw}:fullS_mask_exact'] = equal(candidate['S_full'], mask)
                checks[f'{raw}:no_old_fullS_support_lost'] = row['regions'][name]['lost_supported'] == 0
            if name == 'fixed_low_S':
                checks[f'{raw}:fixed_low_S_mask_exact'] = equal(candidate['fixed_low_S'], mask)
        row['A0_A1_road_RGB_max_abs'] = float(np.abs(candidate['road_rgb'] - active_only['road_rgb']).max())
        row['A0_A1_joint_RGB_max_abs'] = float(np.abs(candidate['joint_rgb'] - active_only['joint_rgb']).max())
        rows.append(row)
    full_S = [r['regions']['full_S'] for r in rows if 'full_S' in r['regions']]
    low_S = [r['regions']['fixed_low_S'] for r in rows if 'fixed_low_S' in r['regions']]
    D = [r['regions']['D'] for r in rows]
    totals = {}
    for name, source_rows in [('D', D), ('full_S', full_S), ('fixed_low_S', low_S)]:
        totals[name] = {state: {k: sum(r[state][k] for r in source_rows)
                               for k in ('pixels', 'gap', 'exact_zero', 'thin', 'supported')}
                        for state in ('before', 'after')}
    with np.load(CURRENT / 'supervision_r1/activation_proposal.npz') as proposal:
        missing_uv = int(proposal['material_source_original_uv_missing'].sum())
    asset_checks = {name: read(CURRENT / f'qa/asset_{name}_audit.json') for name in ('A0', 'A1')}
    checks['A0_asset_invariants'] = asset_checks['A0']['status'] == 'pass'
    checks['A1_asset_invariants'] = asset_checks['A1']['status'] == 'pass'
    checks['A1_new_material_qualified'] = asset_checks['A1']['material']['qualified'] is True
    result = dict(schema='road_activation_independent_acceptance.v1',
                  status='pass_mechanical' if all(checks.values()) else 'failed', checks=checks,
                  GPU_executed_by_auditor=False, additional_native_calls=0, optimizer_updates=0,
                  native_execution=dict(receipt=str(root / 'results.json'), calls=source['native_raster_calls'],
                                        seconds=source['seconds'], optimizer_updates=source['optimizer_updates']),
                  new_active_rows=int(new.sum()), active_before=int(original.sum()), active_after=int(final.sum()),
                  fixed_domains=dict(D=38416, full_S=3187, fixed_low_S=965, K=77266), totals=totals, rows=rows,
                  material=dict(A0='legacy placeholders intentionally retained, unqualified material control',
                                A1='765 new rows initialized from immutable source1 observed reference; old active RGB unchanged',
                                source_UV_missing_rows=missing_uv,
                                provenance_boundary='source ledger identity/RGB equality verified; original per-source UV reprojection cannot be verified'),
                  new_K_guard=dict(role='conservative support envelope using actual native radius and tile-expanded getRect',
                                   accepted_new_rows_intersecting_K=0, accepted_new_near_guard_rows=0,
                                   actual_per_row_alphaT_attribution_exported=False,
                                   source=str(root / 'new_packet_guard.json')),
                  joint_visibility=dict(joint_RGB_rendered=True, protected_K_exact=True,
                                        per_branch_road_alphaT_exported=False,
                                        added_jointly_visible_road_pixels=None,
                                        boundary='892 gained support is road-only; do not re-label it joint-visible contribution'),
                  alpha_drop=dict(monotonicity_claimed=False,
                                  wording='observed high-alpha decreases; no support threshold loss',
                                  source_level_candidate_mechanism='inherited CUDA exits before compositing a fragment when test_T<1e-4; this can invalidate monotonicity after insertion',
                                  causal_mechanism_proven=False,
                                  rounding_only_claimed=False),
                  control_attribution='parent->A0 isolates activation at fixed colors/opa/sigma; A0->A1 changes only new-row observed material, with alpha exact;0training updates',
                  remaining_coverage='73 fixed low-S pixels are thin, no remaining low-S gap; not a complete coverage success',
                  visual_verdict=None, human_verdict=None, production_promoted=False)
    write(output, result)
    return result


def audit_b32(output):
    root = CURRENT / 'B32_native_r1'
    parent = CURRENT / 'native_r1/asset_A1_guarded'
    final = root / 'asset_B32_guarded'
    a, b = load(parent / 'road.npz'), load(final / 'road.npz')
    init = load(CURRENT / 'supervision_B32_r1/activation_pseudo_init.npz')
    new = b['active'] & ~a['active']
    ids = np.flatnonzero(new)
    checks = dict(exact_registered_32_rows=bool(np.array_equal(ids, init['proposed_world_rows'])),
                  prior_active_not_deactivated=bool(b['active'][a['active']].all()))
    for key in a:
        if key == 'active':
            continue
        if key == 'reference_srgb':
            change = changed_rows(a[key], b[key])
            checks['RGB_only_new_rows'] = not bool((change & ~new).any())
            checks['new_RGB_exact_frozen_pseudo_source'] = equal(b[key][ids], init['proposed_effective_rgb_sparse'])
        else:
            checks[key + '_all_rows_exact'] = equal(a[key], b[key])
    prior = {}
    with np.load(parent / 'provenance.npz') as p, np.load(final / 'provenance.npz') as q:
        for key in LEDGER:
            if key in p.files:
                prior[key] = key in q.files and equal(p[key], q[key])
        checks['prior_observed_known_unknown_source_ledger_exact'] = all(prior.values())
        checks['effective_source_pseudo3'] = bool((q['effective_material_source'][ids] == 3).all())
        checks['effective_material_available'] = bool(q['effective_material_available'][ids].all())
        checks['effective_material_unknown_false'] = not bool(q['effective_material_unknown'][ids].any())
        checks['effective_coverage_legal'] = bool(q['effective_coverage_legal'][ids].all())
        checks['effective_coverage_request'] = bool(q['effective_coverage_requested'][ids].all())
    guards = load(root / 'new_packet_guards.npz')
    checks['new_K_guard_zero'] = not bool(guards['world_K_guard'][ids].any())
    checks['new_near_guard_zero'] = not bool(guards['world_near_guard'][ids].any())
    fixed = masks()
    rows = []
    for raw in FRAMES:
        data = load(root / f'{raw:03d}_native.npz')
        previous = load(CURRENT / f'native_r1/{raw:03d}_native.npz')
        K = fixed[raw]['K']
        maximum = float(np.abs(data['joint_rgb'][:, K] - previous['joint_rgb'][:, K]).max())
        checks[f'{raw}:full_joint_K_exact'] = maximum == 0
        checks[f'{raw}:Kmask_exact'] = equal(data['K'], K)
        checks[f'{raw}:parent_alpha_exact'] = equal(data['A1_road_alpha'], previous['road_alpha'])
        row = dict(raw=raw, K_pixels=int(K.sum()), K_max=maximum, regions={})
        for key, mask in fixed[raw].items():
            if key == 'K':
                continue
            row['regions'][key] = dict(before=metrics(previous['road_alpha'], mask), after=metrics(data['road_alpha'], mask))
        if raw in (86, 110):
            remaining = fixed[raw]['fixed_low_S'] & (previous['road_alpha'] <= .9)
            checks[f'{raw}:fixed_A1_remaining73_exact'] = equal(data['A1_remaining73'], remaining)
            related = data['B32_related_remaining15']
            row['B32_related_remaining_pixels'] = int(related.sum())
            row['B32_related_remaining_supported'] = int((related & (data['road_alpha'] > .9)).sum())
        rows.append(row)
    source = read(root / 'results.json')
    reload = read(root / 'public_reload.json')
    checks['native_GPU_executed'] = source['GPU_executed'] is True
    checks['zero_optimizer_updates'] = source['optimizer_updates'] == 0
    checks['actual_public_loader'] = reload['GPU_executed'] is True and reload['public_loader'] is True
    checks['fiveframe_public_reload_exact'] = len(reload['rows']) == 5 and all(r['RGB_max'] == 0 and r['alpha_max'] == 0 and r['RGB8_max'] == 0 for r in reload['rows'])
    result = dict(schema='road_activation_B32_independent_CPU_acceptance.v1',
                  status='pass_mechanical' if all(checks.values()) else 'failed', checks=checks, prior_ledger=prior,
                  new_rows=len(ids), parent=str(parent), final=str(final), rows=rows,
                  native_calls=source['native_calls'], optimizer_updates=0, GPU_executed_by_auditor=False,
                  effective_material_role='frozen2D shadow pseudo teacher, not real observed GT; prior source0/unknown remains truthful',
                  source_permission_q='native footprint validity, not calibrated teacher correctness',
                  fixed_low_S_remaining_thin=sum(r['regions']['fixed_low_S']['after']['thin'] for r in rows if 'fixed_low_S' in r['regions']),
                  human_verdict=None, visual_verdict=None, production_promoted=False)
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'asset', 'renders', 'native_delivery', 'B32'), default='prepare')
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--stage', choices=('active_only', 'activation_material'), default='active_only')
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--proposal-key', default='candidate')
    parser.add_argument('--render-plan', type=Path)
    parser.add_argument('--output', type=Path, default=CURRENT / 'qa/acceptance_plan.json')
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare(args.output)
    elif args.mode == 'asset':
        result = audit_asset(args.parent, args.candidate, args.stage, args.proposal, args.proposal_key, args.output)
    elif args.mode == 'renders':
        result = audit_renders(args.render_plan, args.output)
    elif args.mode == 'native_delivery':
        result = audit_native_delivery(args.output)
    else:
        result = audit_b32(args.output)
    print(json.dumps(dict(output=str(args.output), status=result.get('status', 'prepared'), GPU_executed_by_auditor=False)))


if __name__ == '__main__':
    main()

"""固定pseudo-S区域的只读材料/覆盖验收与状态图；无Torch/native/GPU。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
N = CAMPAIGN / 'road_activation_20261118_r1'
CPU = CAMPAIGN / 'road_capacity_cpu_20261117_r1'
G = CAMPAIGN / 'road_capacity_gpu_20261118_r1'
OLD = CAMPAIGN / 'road_asset_20261116_r1/phase2/formal_r1'
FRAMES = (86, 110, 134, 179, 198)
MATERIAL_FRAMES = (86, 110)
STAGES = {
    'parent': OLD / 'rendered',
    'A0': N / 'native_r1/A0',
    'A1': N / 'native_r1',
    'B32': N / 'B32_native_r1',
    'A1_beam95': N / 'A1_beam95_native_r1',
    'B32_beam95': N / 'B32_beam95_native_r1',
    'composed_I0': N / 'composed_I0_native_r1',
}
LEDGER = ('source', 'known', 'unknown', 'confidence', 'observed_reference_srgb', 'row_rgb',
          'source_world_row', 'source_distance_m', 'source_raw', 'source_uv_native',
          'world_xyz', 'world_keys', 'atlas_uv', 'surface_supported', 'original_surface_supported',
          'original_world_xyz', 'old_requested', 'requested', 'W_world')


def read(path):
    return json.loads(Path(path).read_text())


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (keys if keys else z.files)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def arrays(stage, raw):
    p = STAGES[stage] / (f'{raw:03d}/arrays.npz' if stage == 'parent' else f'{raw:03d}_native.npz')
    z = load(p)
    if stage == 'parent':
        return dict(P=z['road_only_rgb'], alpha=z['road_only_alpha'].squeeze(), J=z['rgb'], path=str(p))
    return dict(P=z['road_rgb'], alpha=z['road_alpha'], J=z['joint_rgb'], path=str(p))


def residual_metric(values):
    v = np.asarray(values, np.float64)
    return dict(MAE=float(np.abs(v).mean()), RMSE=float(np.sqrt((v*v).mean())))


def summarize(stacked):
    P = np.concatenate([x['P'] for x in stacked], axis=1).astype(np.float64)
    J = np.concatenate([x['J'] for x in stacked], axis=1).astype(np.float64)
    T = np.concatenate([x['T'] for x in stacked], axis=1).astype(np.float64)
    a = np.concatenate([x['a'] for x in stacked]).astype(np.float64)
    return dict(pixel_frames=len(a), RGB_denominator=3*len(a),
                road_only_opaque=residual_metric(P-T),
                road_only_premult=residual_metric(P-a[None]*T),
                joint_opaque=residual_metric(J-T),
                alpha=dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                           exact_zero=int((a == 0).sum()), gap=int((a < .05).sum()),
                           thin=int(((a >= .05) & (a <= .9)).sum()), supported=int((a > .9).sum())))


def build_material():
    teacher, masks = {}, {}
    for raw in MATERIAL_FRAMES:
        t = load(CPU / f'supervision_v4_candidate_r1/{raw:03d}.npz',
                 ['target_S_srgb', 'S_target_shadow', 'S_target_shadow_photo'])
        teacher[raw] = t['target_S_srgb']
        S, photo = t['S_target_shadow'], t['S_target_shadow_photo']
        final_alpha = arrays('B32_beam95', raw)['alpha']
        masks[raw] = dict(full_S=S, old_photo_S=photo,
                          fixed_low_S=S & ~photo,
                          current_new_support_909=S & ~photo & (final_alpha > .9),
                          current_remaining_56=S & ~photo & (final_alpha <= .9))
    totals = {name: sum(int(masks[r][name].sum()) for r in MATERIAL_FRAMES) for name in masks[86]}
    assert totals == dict(full_S=3187, old_photo_S=2222, fixed_low_S=965,
                          current_new_support_909=909, current_remaining_56=56)
    np.savez_compressed(N / 'qa/material_masks.npz', **{f'{raw:03d}_{key}': value for raw, group in masks.items() for key, value in group.items()})
    stages = {}
    for stage in STAGES:
        by_region = {name: [] for name in masks[86]}
        per_frame = []
        for raw in MATERIAL_FRAMES:
            z = arrays(stage, raw)
            frame = dict(raw=raw, source=z['path'], regions={})
            for name, mask in masks[raw].items():
                entry = dict(P=z['P'][:, mask], J=z['J'][:, mask], a=z['alpha'][mask], T=teacher[raw][:, mask])
                by_region[name].append(entry)
                frame['regions'][name] = summarize([entry])
            per_frame.append(frame)
        pooled = {name: summarize(entries) for name, entries in by_region.items()}
        for name, metrics in pooled.items():
            metrics['premult_across_coverage_comparable'] = name == 'old_photo_S'
            if stage == 'parent' and name == 'current_new_support_909':
                metrics['unsupported_parent_premult_residual_diagnostic'] = metrics['road_only_premult']
                metrics['road_only_premult'] = None
                metrics['color_ranking_qualified'] = False
                metrics['qualification_note'] = 'unsupported parent: low alpha causes artificial small premult residual; excluded from new-support material comparisons'
            else:
                metrics['color_ranking_qualified'] = metrics['alpha']['supported'] == metrics['pixel_frames']
        stages[stage] = dict(source_directory=str(STAGES[stage]), pooled=pooled, per_frame=per_frame)
    attribution = []
    for first, second, role in [('parent', 'A0', 'activation only at legacy colors'),
                                ('A0', 'A1', 'same-alpha observed material initialization'),
                                ('A1', 'B32', '32 frozen-pseudo row initialization and activation'),
                                ('A1', 'A1_beam95', 'observed95 fixed-footprint control without pseudo32'),
                                ('B32', 'B32_beam95', 'observed95 fixed-footprint control after pseudo32'),
                                ('B32_beam95', 'composed_I0', 'reuse previous I0 old-active learned RGB; no optimization')]:
        regions = {}
        for region in ('old_photo_S', 'full_S', 'current_new_support_909'):
            a, b = stages[first]['pooled'][region], stages[second]['pooled'][region]
            regions[region] = dict(road_opaque_MAE_delta=b['road_only_opaque']['MAE']-a['road_only_opaque']['MAE'],
                                    joint_opaque_MAE_delta=b['joint_opaque']['MAE']-a['joint_opaque']['MAE'],
                                    support_delta=b['alpha']['supported']-a['alpha']['supported'])
            if a['road_only_premult'] is not None and b['road_only_premult'] is not None:
                regions[region]['premult_MAE_delta'] = b['road_only_premult']['MAE']-a['road_only_premult']['MAE']
        attribution.append(dict(before=first, after=second, role=role, regions=regions))
    contribution = []
    for raw in MATERIAL_FRAMES:
        z = load(N / f'visibility_r2/{raw:03d}_groups.npz')
        for region, mask in masks[raw].items():
            masses = {name: float(z[name][mask].astype(np.float64).sum()) for name in
                      ['joint_static_alphaT', 'joint_dynamic_alphaT', 'joint_old_road_alphaT', 'joint_new_road_alphaT',
                       'road_only_old_road_alphaT', 'road_only_new_road_alphaT']}
            contribution.append(dict(raw=raw, region=region, pixels=int(mask.sum()), mass=masses,
                                     mean_joint_road_alphaT=(masses['joint_old_road_alphaT']+masses['joint_new_road_alphaT'])/mask.sum(),
                                     source_stage='A1 only', actual_native_measurement=True))
    material = dict(schema='road_activation_material_quality_CPU.v1', GPU_executed_by_auditor=False,
                    additional_native_calls=0, optimizer_updates=0,
                    teacher_role='same frozen target-shadow pseudo teachers, not GT; I0 already trained on two source views',
                    teacher_paths={str(raw): str(CPU / f'supervision_v4_candidate_r1/{raw:03d}.npz') for raw in MATERIAL_FRAMES},
                    target_key='target_S_srgb', measured_frames=list(MATERIAL_FRAMES), pilot_frames=list(FRAMES),
                    generalization_claimed=False, mask_counts=totals,
                    masks=str(N / 'qa/material_masks.npz'),
                    current_new_support_definition='fixed after B32+beam95 alpha>0.9 within original lowS; diagnostic post-defined stratum reused identically across comparisons, never substitutes fullS',
                    metrics=dict(accumulation='float64 pooled pixel/channel mean over both views, not equal-view macro',
                                 road_only_opaque='P-T', road_only_premult='P-alpha*T', joint_opaque='J-T',
                                 alpha='road-only alpha; not joint visibility',
                                 unsupported_parent_premult='explicitly excluded from new-support color ranking; never count original zero-alpha residual as improvement',
                                 fullS_premult='coverage-weighted residual only; do not rank different coverage by this number alone'),
                    stages=stages, attribution=attribution,
                    actual_contribution=dict(source=str(N / 'visibility_r2'), source_asset='A1',
                                              rows=contribution, later_stage_alphaT=None,
                                              boundary='actual native A1 group alphaT reused only at its source state; parent/B32/beam95/composed contributions are not inferred from road-only alpha'),
                    interpretation=[
                        'Activation+lawful initialization improves fullS opaque error, while old-photoS material error stays approximately unchanged until I0 color reuse.',
                        'Composed I0 greatly reduces old-photoS pseudo-teacher error; new909 material residual remains around0.0885.',
                        '56 fixed fullS pixels remain thin; coverage and material quality are separate outcomes.',
                        'I0 stripe-quality failure is inherited; teacher fitting and mechanical acceptance are not a visual or production pass.'
                    ], human_verdict=None, visual_verdict=None, production_promoted=False)
    write(N / 'qa/material_quality.json', material)
    return material, masks


def coverage_maps(masks):
    palette = dict(outside=[0, 0, 0], gap=[231, 76, 60], thin=[245, 158, 11], supported=[34, 197, 94])
    rows = []
    for label, stage in [('parent', 'parent'), ('final', 'composed_I0')]:
        for raw in MATERIAL_FRAMES:
            z = arrays(stage, raw)
            mask = masks[raw]['full_S']
            a = z['alpha']
            assert a.shape == mask.shape == (320, 480) and np.isfinite(a[mask]).all()
            classes = dict(gap=mask & (a < .05), thin=mask & (a >= .05) & (a <= .9), supported=mask & (a > .9))
            image = np.zeros((320, 480, 3), np.uint8)
            for name, selected in classes.items():
                image[selected] = palette[name]
            target = N / f'qa/coverage_maps/{label}/{raw:03d}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(target)
            rows.append(dict(label=label, stage=stage, raw=raw, path=str(target), source=z['path'],
                             fullS_pixels=int(mask.sum()), **{name: int(value.sum()) for name, value in classes.items()}))
    legend = dict(schema='fixed_fullS_coverage_state_legend.v1', palette_RGB=palette,
                  threshold=dict(gap='alpha<0.05', thin='0.05<=alpha<=0.9', supported='alpha>0.9'),
                  fixed_fullS=3187, original_WH=[480, 320], domains_and_dimensions_unchanged=True,
                  pixels_outside_domain_black=True, no_render_or_training=True, rows=rows)
    write(N / 'qa/coverage_maps/legend.json', legend)
    return legend


def field_audit(parent_path, child_path, expected_new_count, mode):
    parent_path, child_path = Path(parent_path), Path(child_path)
    a, b = load(parent_path / 'road.npz'), load(child_path / 'road.npz')
    checks = {}
    new = b['active'] & ~a['active']
    checks['no_deactivation'] = bool(b['active'][a['active']].all())
    checks['new_rows_registered_count'] = int(new.sum()) == expected_new_count
    for name in a:
        if name == 'active':
            if mode == 'composition':
                checks['active_exact'] = exact(a[name], b[name])
        elif name == 'reference_srgb':
            if mode != 'composition':
                checks['oldactive_RGB_exact'] = exact(a[name][a['active']], b[name][a['active']])
        else:
            checks[name + '_exact'] = exact(a[name], b[name])
    with np.load(parent_path / 'provenance.npz') as p, np.load(child_path / 'provenance.npz') as q:
        for name in LEDGER:
            checks['prior_' + name + '_bytes_exact'] = exact(p[name], q[name])
        if mode == 'observed':
            checks['new_source1_known'] = bool(((p['source'][new] == 1) & p['known'][new] & ~p['unknown'][new]).all())
            checks['new_RGB_exact_observed'] = exact(b['reference_srgb'][new], p['observed_reference_srgb'][new])
    return dict(parent=str(parent_path), child=str(child_path), new_rows=int(new.sum()), checks=checks)


def final_acceptance(material, legend):
    stages = [
        ('A1_beam95', N/'native_r1/asset_A1_guarded', N/'A1_beam95_native_r1/asset_guarded', 95, 'observed'),
        ('B32_beam95', N/'B32_native_r1/asset_B32_guarded', N/'B32_beam95_native_r1/asset_guarded', 95, 'observed'),
        ('composed_I0', N/'B32_beam95_native_r1/asset_guarded', N/'composed_I0_native_r1/asset_guarded', 0, 'composition')]
    audit = [dict(stage=name, **field_audit(a, b, count, mode)) for name,a,b,count,mode in stages]
    checks = {item['stage']+':'+key: value for item in audit for key,value in item['checks'].items()}
    base = load(OLD/'asset/road.npz')
    cover = load(N/'B32_beam95_native_r1/asset_guarded/road.npz')
    final = load(N/'composed_I0_native_r1/asset_guarded/road.npz')
    I0 = load(G/'check3/I0/formal_r1/asset/road.npz', ['reference_srgb'])['reference_srgb']
    eligible = load(G/'check1/native_r1/qualified_scope.npz', ['world_eligible_color'])['world_eligible_color']
    new = cover['active'] & ~base['active']
    changed = np.any(final['reference_srgb'] != cover['reference_srgb'], axis=1)
    checks.update(composed_exact_I0_on_eligible=exact(final['reference_srgb'][eligible], I0[eligible]),
                  composed_RGB_unchanged_outside_eligible=exact(final['reference_srgb'][~eligible], cover['reference_srgb'][~eligible]),
                  new892_RGB_exact=exact(final['reference_srgb'][new], cover['reference_srgb'][new]),
                  eligible83388=int(eligible.sum()) == 83388, actual_override83324=int(changed.sum()) == 83324,
                  eligible_new_row_disjoint=not bool((eligible & new).any()), activated_world_rows892=int(new.sum()) == 892)
    for name in base:
        if name not in ['active', 'reference_srgb']:
            checks['canonical_'+name+'_all_rows_exact'] = exact(base[name], final[name])
    K = load(G/'check1/native_r1/K_masks.npz')
    per_frame = []
    for raw in FRAMES:
        f, c, p = arrays('composed_I0', raw), arrays('B32_beam95', raw), arrays('parent', raw)
        k = K[f'{raw:03d}']
        checks[f'{raw}:composed_alpha_exact_coverage_parent'] = exact(f['alpha'], c['alpha'])
        error = float(np.abs(f['J'][:,k]-p['J'][:,k]).max())
        checks[f'{raw}:joint_K_exact_parent'] = error == 0
        d = load(CPU/f'ideal_targets_r2/constant/{raw:03d}.npz', ['mask'])['mask']
        dc = dict(pixels=int(d.sum()), gap=int((d & (f['alpha']<.05)).sum()),
                  thin=int((d & (f['alpha']>=.05) & (f['alpha']<=.9)).sum()), supported=int((d & (f['alpha']>.9)).sum()))
        per_frame.append(dict(raw=raw, K_pixels=int(k.sum()), joint_K_max=error, D=dc))
    call_sources=[('A0_A1','native_r1/results.json','native_raster_calls'),
                  ('actual_visibility','visibility_r2/results.json','native_calls'),
                  ('bounded_endpoints','endpoint_r1/results.json','native_calls'),
                  ('pseudo32','B32_native_r1/results.json','native_calls'),
                  ('isolated_fragment_calibration','footprint_calibration_native_r1/results.json','native_calls'),
                  ('A1_observed95','A1_beam95_native_r1/results.json','native_calls'),
                  ('B32_observed95','B32_beam95_native_r1/results.json','native_calls'),
                  ('I0_color_composition','composed_I0_native_r1/results.json','native_calls')]
    accounting=[]
    for role,path,key in call_sources:
        receipt=read(N/path)
        checks[role+':zero_optimizer'] = receipt['optimizer_updates'] == 0
        accounting.append(dict(role=role, receipt=str(N/path), native_forward_calls=receipt[key], optimizer_updates=0))
    total=sum(r['native_forward_calls'] for r in accounting)
    checks['campaign_actual_native194']=total == 194
    checks['all_K77266']=sum(r['K_pixels'] for r in per_frame) == 77266
    checks['D38416']=sum(r['D']['pixels'] for r in per_frame) == 38416
    checks['D_supported37404']=sum(r['D']['supported'] for r in per_frame) == 37404
    reloads=[]
    for name, _, child, _, _ in stages:
        folder=child.parent
        reload=read(folder/'public_reload.json')
        ok=reload.get('public_loader') is True and reload.get('GPU_executed') is True and len(reload['rows']) == 5 and all(r['RGB_max']==0 and r['alpha_max']==0 and r['RGB8_max']==0 for r in reload['rows'])
        checks[name+':actual_public_reload_exact']=ok
        reloads.append(dict(stage=name,path=str(folder/'public_reload.json'), exact=ok))
    previous=[N/'qa/acceptance.json',N/'qa/B32_acceptance.json']
    for p in previous:
        checks[p.name+':previous_independent_audit_pass']=read(p)['status']=='pass_mechanical'
    result=dict(schema='road_activation_final_mechanical_acceptance.v1',
                status='pass_mechanical' if all(checks.values()) else 'failed', checks=checks,
                fixed_domains=dict(pilot_frames=list(FRAMES), D=38416, full_S=3187, old_photo_S=2222, fixed_low_S=965,
                                   current_newly_supported_pixel_frames=909, remaining_thin_pixel_frames=56, K=77266),
                geometry=dict(activated_existing_world_rows=892, observed_material_rows=860,pseudo_material_rows=32,
                              geometry_created=0, all_positions_normals_sigma_opacity_frozen=True),
                field_audits=audit, per_frame=per_frame, public_reload=reloads,
                accounting=dict(total_native_forward_calls=total, registered_limit=200, reassessment_limit_after_composition=194,
                                remaining_automatic_authorization=False, optimizer_updates=0,new_generation_calls=0,
                                failed_visibility_r1_native_calls=0, additional_CPU_auditor_native_calls=0,
                                sources=accounting, inherited_I0_training='already charged in originalG:200updates, reused here with0newupdates; not hidden N training',
                                separate_M_solver='500formalFG+3smokeFG belongs M, not charged to N'),
                material_quality=str(N/'qa/material_quality.json'), coverage_maps=str(N/'qa/coverage_maps/legend.json'),
                pseudo_teacher_metric_summary={name:material['stages'][name]['pooled'] for name in ['parent','B32_beam95','composed_I0']},
                scope=dict(two_source_views_supervised=True, diagnostic_views=[134,179,198], blind_test_claimed=False,
                           human_visual_judgment='root/user only', coverage_success='partial:56thin remain',
                           material_success='old-photoS teacher error reduced; new909 residual remains about0.0885',
                           actual_joint_alphaT='A1 only existing visibility_r2; no invented later-state contribution',
                           I0_stripe_quality_failure_inherited=True, line_quality_unresolved=True),
                no_complete_solution_claim=True, human_verdict=None, visual_verdict=None, production_promoted=False)
    write(N/'qa/final_acceptance.json',result)
    return result


def main():
    (N/'qa').mkdir(exist_ok=True)
    material,masks=build_material()
    legend=coverage_maps(masks)
    result=final_acceptance(material,legend)
    print(json.dumps(dict(material_quality=str(N/'qa/material_quality.json'), final_acceptance=str(N/'qa/final_acceptance.json'),
                          status=result['status'], checks=len(result['checks']),
                          failures=[k for k,v in result['checks'].items() if not v], native_calls=result['accounting']['total_native_forward_calls'],
                          coverage_maps=str(N/'qa/coverage_maps/legend.json'))))


if __name__=='__main__':
    main()

"""Independent CPU-only fixed-domain evaluation; never loads a Gaussian model.
H entry preserves all original4560 gates and adds only the explicit H variant name.

Inputs: each variant's frames/RAW/{keep,delete}_float.npz, original observation
NPZs, original ground proxy, and the frozen world-mask registry. No mask is
selected from a candidate/alpha. Existing reports and outputs are never changed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EDGES = {'near_x': 1, 'far_x': 1, 'lateral_y': 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def mask(value):
    result = np.asarray(value).squeeze().astype(bool)
    assert result.shape == (320, 480)
    return result


def chw(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape == (320, 480, 3):
        value = value.transpose(2, 0, 1)
    assert value.shape == (3, 320, 480) and np.isfinite(value).all()
    return value


def hw(value):
    value = np.asarray(value).squeeze()
    assert value.shape == (320, 480) and np.isfinite(value).all()
    return value


def stats(value, domain):
    selected = np.asarray(value)[domain]
    assert np.isfinite(selected).all(), 'Never silently drop nonfinite fixed pixels'
    return dict(pixels=int(domain.sum()), sum=float(selected.astype(np.float64).sum()),
                mean=float(selected.astype(np.float64).mean()) if selected.size else None,
                max=float(selected.max()) if selected.size else None)


def upper(a5):
    result = np.zeros_like(a5)
    y, x = np.nonzero(a5)
    if y.size:
        result = a5 & (np.arange(320)[:, None] <= y.min()+.25*(y.max()-y.min()))
    return result


def registration(masks):
    result = {}
    xyz = masks['local_xyz']
    for edge, tangent_axis in EDGES.items():
        minus, plus = masks[edge+'_minus'], masks[edge+'_plus']
        selected = minus | plus
        bins = np.zeros(minus.shape, np.int32)
        bins[selected] = np.floor(xyz[..., tangent_axis][selected]/.2).astype(np.int32)
        pairs = []
        for number in np.unique(bins[selected]):
            left, right = minus & (bins == number), plus & (bins == number)
            if left.sum() >= 3 and right.sum() >= 3:
                pairs.append(dict(tangent_bin=int(number), minus_count=int(left.sum()), plus_count=int(right.sum())))
        result[edge] = dict(bins=pairs)
    return result


def seams(image, gt, masks, registered):
    # Same published residual-jump definition; do not replace it with direct
    # colour differences or a prediction-conditioned boundary.
    residual = image-gt
    xyz = masks['local_xyz']
    result = {}
    for edge, tangent_axis in EDGES.items():
        minus, plus = masks[edge+'_minus'], masks[edge+'_plus']
        selected = minus | plus
        bins = np.zeros(minus.shape, np.int32)
        bins[selected] = np.floor(xyz[..., tangent_axis][selected]/.2).astype(np.int32)
        values = []
        for pair in registered[edge]['bins']:
            left = minus & (bins == pair['tangent_bin'])
            right = plus & (bins == pair['tangent_bin'])
            assert int(left.sum()) == pair['minus_count'] and int(right.sum()) == pair['plus_count']
            jump = residual[:, right].mean(1)-residual[:, left].mean(1)
            assert np.isfinite(jump).all()
            values.append(dict(**pair, signed_residual_jump_rgb=jump.tolist(), score=float(np.abs(jump).mean())))
        result[edge] = dict(status='COMPLETE' if values else 'UNSUPPORTED', bins=values,
                            paired_bins=len(values), mean=float(np.mean([v['score'] for v in values])) if values else None)
    return result


def wheel_delta(candidate, baseline, rays):
    values = []
    for ray in rays:
        x,y = ray['pixel_xy']
        bw,cw = float(baseline[3][0,y,x]),float(candidate[3][0,y,x])
        values.append(dict(pixel_xy=[x,y],ordering=ray['ordering'],legacy_V44_assetW=ray['legacy_V44_assetW'],
                           baseline_assetW=bw,candidate_assetW=cw,assetW_delta=cw-bw,
                           joint_RGB_mean_abs_delta=float(np.abs(candidate[0][:,y,x]-baseline[0][:,y,x]).mean())))
    return dict(ray_count=len(values),pixels=values,
                max_assetW_abs_delta=max((abs(v['assetW_delta']) for v in values),default=None),
                max_assetW_drop=max((max(0.0,-v['assetW_delta']) for v in values),default=None),
                chart_first_new_drop_gt001=sum(v['ordering']=='chart_first' and v['assetW_delta']<-.01 for v in values),
                ground_first_new_exposure_gt001=sum(v['ordering']=='ground_first' and v['assetW_delta']>.01 for v in values),
                role='Original fixed71-ray diagnostic; legacy expected ordering is not hidden GT or opaque first hit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', required=True, help='baseline=PATH / AA=PATH / C=PATH')
    parser.add_argument('--obs-root', type=Path, required=True)
    parser.add_argument('--fixed-reference', type=Path, required=True)
    parser.add_argument('--world-mask-root', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--method-contract',type=Path,help='Required for a separately registered posterior method; does not change numeric gates')
    parser.add_argument('--wheel-registry',type=Path)
    parser.add_argument('--candidate-training-receipt',type=Path,help='Actual appearance-fit receipt, distinct from this zero-update CPU evaluation')
    parser.add_argument('--raws', nargs='+', type=int, default=[94, 105, 128, 137])
    parser.add_argument('--diagnostic-raws', nargs='*', type=int, default=[130])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Keep prior independent evidence immutable'
    variants = {key: Path(value) for key, value in (spec.split('=', 1) for spec in args.variant)}
    assert 'baseline' in variants and len(variants)>1 and set(variants)<= {'baseline','AA','C','D','E','F','H'}
    candidates = [name for name in ('AA','C','D','E','F','H') if name in variants]
    if any(name in variants for name in ('D','E','F','H')):
        assert args.method_contract and args.method_contract.is_file(), 'A new posterior method requires its own contract'
    contract = read(args.contract)
    assert args.raws == contract['pilot_raws'] or args.raws == list(range(89, 138))
    assert set(args.raws).isdisjoint(args.diagnostic_raws)
    inputs = {args.contract, args.fixed_reference, args.world_mask_root/'registry.json'}
    if args.method_contract:inputs.add(args.method_contract)
    if args.candidate_training_receipt:inputs.add(args.candidate_training_receipt)
    wheel = read(args.wheel_registry) if args.wheel_registry else None
    if args.wheel_registry:inputs.add(args.wheel_registry)
    registry = read(args.world_mask_root/'registry.json')
    assert sha(args.world_mask_root/'registry.json') == contract['road_domains']['world_registry_SHA256']
    registered = {row['raw']: row for row in registry['frames']}
    fixed = load(args.fixed_reference)
    assert fixed['raw_ids'].tolist() == list(range(89, 138))
    assert fixed['layer_names'].tolist()[3] == 'ground_beneath_a37_proxy'
    ground = fixed['depth_m'][:, 3] > 0
    primary_path = args.world_mask_root/'primary_raw128/fixed_masks_raw128.npz'
    inputs.add(primary_path)
    expected_primary = next(row['sha256'] for row in contract['source_references'] if row['relative'].endswith('/fixed_masks_raw128.npz'))
    assert sha(primary_path) == expected_primary
    primary = load(primary_path)
    primary_masks = dict(local_xyz=primary['local_xyz'], **{edge+'_'+side: primary['band_'+edge+'_'+side]
                        for edge in EDGES for side in ('minus','plus')})
    primary_registered = registration(primary_masks)
    assert int(primary['original_road'].sum()) == 1385
    assert len(primary_registered['near_x']['bins']) == 6
    assert sum(v['minus_count'] for v in primary_registered['near_x']['bins']) == 35
    assert sum(v['plus_count'] for v in primary_registered['near_x']['bins']) == 33
    records, gates, diagnostic, cached = [], [], [], {}
    gate_values = contract['gates']['for_each_AA_and_C_vs_latest_per_frame_state']

    def gate(name, value, threshold, **context):
        gates.append(dict(name=name, value=value, threshold=threshold, passed=None if value is None else value <= threshold, **context))

    for raw in args.raws+args.diagnostic_raws:
        obs_path, world_path = args.obs_root/f'{raw:03d}.npz', args.world_mask_root/f'{raw:03d}.npz'
        inputs.update([obs_path, world_path])
        assert sha(world_path) == registered[raw]['mask_sha256']
        obs, world = load(obs_path), load(world_path)
        gt = chw(obs['rgb'])
        if gt.max() > 1:
            gt = gt/255
        proxy = chw(obs['road_edit_rgb'])
        if proxy.max() > 1:
            proxy = proxy/255
        a5, a6, a37 = mask(obs['actor_labels']==1), mask(obs['visible'][0]), mask(obs['visible'][1])
        road, road_edit, protect = mask(obs['road']), mask(obs['road_edit']), mask(obs['protect'])
        nonroad = protect & ~a6 & ~a37 & ~road & ~road_edit
        geometry_guard = ~road & ~road_edit & (protect | a5 | a6 | a37)
        measured = mask(obs['depth_valid'][0])
        target_depth = np.asarray(obs['depth_m'][0]).squeeze()
        assert target_depth.shape == (320,480)
        domains = dict(a5_RGB=a5, a5_upper_RGB=upper(a5), a6_RGB=a6,
                       protected_nonroad_RGB=nonroad, legacy_protected_RGB=protect & ~a6 & ~a37,
                       original_road_RGB=road)
        for state in ('keep','delete'):
            render = {}
            for variant, directory in variants.items():
                path = directory/f'frames/{raw:03d}/{state}_float.npz'
                inputs.add(path)
                value = load(path)
                rgb, alpha, depth = chw(value['rgb']), hw(value['alpha']), hw(value['depth_m'])
                weights = np.asarray(value['group_alpha_T'])
                assert weights.ndim == 3 and weights.shape[1:] == (320,480) and np.isfinite(weights).all()
                assert weights.shape[0] == (7 if variant in ('baseline','E','F') else 8)
                assert np.abs(weights[5:7]).max() == 0, 'Only a18/a26 slots are unused; newroad is slot7'
                assert float(np.abs(weights.sum(0)-alpha).max()) < 2e-5
                metric = {name: stats(np.abs(rgb-gt).mean(0), domain) for name, domain in domains.items()}
                metric['a6_depth_m'] = stats(np.abs(depth-target_depth), measured)
                if state == 'keep':
                    metric['a37_RGB'] = stats(np.abs(rgb-gt).mean(0), a37)
                else:
                    metric['delete_proxy_road_RGB'] = stats(np.abs(rgb-proxy).mean(0), road_edit & mask(ground[raw-89]))
                row = dict(raw=raw,state=state,variant=variant,diagnostic_only=raw in args.diagnostic_raws,
                           metrics=metric,world_seams=seams(rgb,gt,world,registered[raw]['edges']),
                           native_RGB_range=dict(min=float(rgb.min()),max=float(rgb.max()),additional_upper_clamp_in_evaluator=False),
                           native_group_order=['asset','a37','oldroad','native_static','a5','a18','a26']+(['newroad'] if weights.shape[0]==8 else []),
                           oldroad_alpha_T=stats(weights[2],road),
                           newroad_alpha_T=stats(weights[7],road) if weights.shape[0]==8 else dict(status='ABSENT_NATIVE_SEVEN_GROUP_REPRESENTATION_NOT_MEASURED_EIGHTH_CHANNEL'))
                if raw == 128:
                    row['primary128_seams'] = seams(rgb,gt,primary_masks,primary_registered)
                    row['primary128_RGB'] = stats(np.abs(rgb-gt).mean(0),primary['original_road'])
                render[variant] = (rgb,alpha,depth,weights,row)
                (diagnostic if row['diagnostic_only'] else records).append(row)
                cached[(raw,state,variant)] = row
            base = render['baseline']
            for variant in candidates:
                current = render[variant]
                row = current[4]
                row['vs_latest'] = {}
                for name in domains:
                    b, c = base[4]['metrics'][name],row['metrics'][name]
                    delta = c['mean']-b['mean'] if c['mean'] is not None else None
                    row['vs_latest'][name] = dict(delta_MAE=delta,
                        RGB_mean_absolute_change=stats(np.abs(current[0]-base[0]).mean(0),domains[name])['mean'],
                        RGB_pixel_max_abs=stats(np.abs(current[0]-base[0]).max(0),domains[name])['max'])
                if state=='keep':
                    row['vs_latest']['a37_RGB']=dict(
                        delta_MAE=row['metrics']['a37_RGB']['mean']-base[4]['metrics']['a37_RGB']['mean'] if a37.any() else None,
                        RGB_mean_absolute_change=stats(np.abs(current[0]-base[0]).mean(0),a37)['mean'],
                        RGB_pixel_max_abs=stats(np.abs(current[0]-base[0]).max(0),a37)['max'])
                for metric_name,threshold_name in [('a5_RGB','a5_full_RGB_MAE_delta_max'),('a5_upper_RGB','a5_upper_RGB_MAE_delta_max'),
                    ('a6_RGB','a6_RGB_MAE_delta_max'),('protected_nonroad_RGB','protected_nonroad_RGB_MAE_delta_max'),
                    ('original_road_RGB','original_road_RGB_MAE_delta_max'),('a6_depth_m','a6_measured_depth_MAE_delta_max_m')]+([('a37_RGB','KEEP_a37_RGB_MAE_delta_max')] if state=='keep' else []):
                    b,c = base[4]['metrics'][metric_name],row['metrics'][metric_name]
                    delta = c['mean']-b['mean'] if c['mean'] is not None else None
                    if raw in args.raws:
                        gate(variant+':'+metric_name,delta,gate_values[threshold_name],raw=raw,state=state,pixels=c['pixels'])
                row['fixed_nonroad_geometry_guard'] = dict(pixels=int(geometry_guard.sum()),
                    alpha_pixel_max_abs=stats(np.abs(current[1]-base[1]),geometry_guard)['max'],
                    depth_pixel_max_abs_m=stats(np.abs(current[2]-base[2]),geometry_guard)['max'])
                if variant in ('E','F'):
                    row['existing_surface_colour_only_global_geometry']=dict(
                        alpha_pixel_max_abs=float(np.abs(current[1]-base[1]).max()),
                        depth_pixel_max_abs_m=float(np.abs(current[2]-base[2]).max()),
                        all_seven_group_alpha_pixel_max_abs=float(np.abs(current[3]-base[3]).max()),
                        role='Full-image geometry diagnostics in original7group ABI; no newroad slot or upper RGB clamp')
                if raw in args.raws:
                    for name, threshold_name in [('alpha_pixel_max_abs','fixed_nonroad_alpha_pixel_max_abs_delta'),
                                                 ('depth_pixel_max_abs_m','fixed_nonroad_depth_pixel_max_abs_delta_m')]:
                        gate(variant+':nonroad_guard:'+name,row['fixed_nonroad_geometry_guard'][name],gate_values[threshold_name],raw=raw,state=state)
                row['a6_new_holes'] = {str(t): int((a6 & (base[3][0]>=t) & (current[3][0]<t)).sum()) for t in (.9,.5)}
                if wheel and str(raw) in wheel['rays']:
                    row['fixed_wheel_rays_vs_latest']=wheel_delta(current,base,wheel['rays'][str(raw)])
                row['vehicle_pixel_diagnostics'] = dict(a6_RGB_maxabs=stats(np.abs(current[0]-base[0]).max(0),a6)['max'], a6_measured_depth_maxabs_m=stats(np.abs(current[2]-base[2]),measured)['max'], KEEP_a37_RGB_maxabs=stats(np.abs(current[0]-base[0]).max(0),a37)['max'] if state=='keep' else None)
                if raw in args.raws:
                    for threshold,count in row['a6_new_holes'].items():
                        gate(variant+':new_a6_holes_alpha'+threshold,count,0,raw=raw,state=state)
            if 'AA' in render and 'C' in render:
                aa,c = render['AA'],render['C']
                c[4]['vs_AA'] = dict(RGB_pixel_max_abs=float(np.abs(c[0]-aa[0]).max()),
                                    alpha_pixel_max_abs=float(np.abs(c[1]-aa[1]).max()),
                                    depth_pixel_max_abs_m=float(np.abs(c[2]-aa[2]).max()),
                                    oldroad_group2_alpha_pixel_max_abs=float(np.abs(c[3][2]-aa[3][2]).max()),
                                    newroad_group7_alpha_pixel_max_abs=float(np.abs(c[3][7]-aa[3][7]).max()),
                                    all_group_alpha_pixel_max_abs=float(np.abs(c[3]-aa[3]).max()))
                if wheel and str(raw) in wheel['rays']:
                    c[4]['fixed_wheel_rays_vs_AA']=wheel_delta(c,aa,wheel['rays'][str(raw)])
                if raw in args.raws:
                    for field, threshold in [('alpha_pixel_max_abs',1e-6),('depth_pixel_max_abs_m',1e-5),('all_group_alpha_pixel_max_abs',1e-6)]:
                        gate('C-vs-AA:'+field,c[4]['vs_AA'][field],threshold,raw=raw,state=state)
    pooled = []
    for state in ('keep','delete'):
        for variant in candidates:
            for edge in EDGES:
                series = {name:[v['score'] for raw in args.raws for v in cached[(raw,state,name)]['world_seams'][edge]['bins']] for name in ('baseline',variant)}
                assert len(series['baseline']) == len(series[variant])
                baseline,candidate = [float(np.mean(series[name])) if series[name] else None for name in ('baseline',variant)]
                delta = candidate-baseline if candidate is not None else None
                pooled.append(dict(state=state,variant=variant,edge=edge,bins=len(series[variant]),baseline=baseline,candidate=candidate,delta=delta))
                if edge != 'far_x':
                    gate(variant+':'+edge+':equal_bin_pooled',delta,1e-6,state=state,bins=len(series[variant]))
            if 128 in args.raws:
                baseline = cached[(128,state,'baseline')]['primary128_seams']['near_x']['mean']
                candidate = cached[(128,state,variant)]['primary128_seams']['near_x']['mean']
                gate(variant+':raw128:primary_near_no_regression',candidate-baseline,1e-6,state=state)
                gate(variant+':raw128:primary_near_absolute_reduction',candidate-baseline,-1e-4,state=state)
                gate(variant+':raw128:primary_near_relative_reduction',(candidate-baseline)/baseline if baseline>0 else None,-.05,state=state)
    manifest = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(inputs)]
    extra_wheel_gates=[]
    if 'D' in variants:
        for raw,count in [(128,36),(130,35)]:
            selected=[row for row in records+diagnostic if row['raw']==raw and row['state']=='delete' and row['variant']=='D']
            value=selected[0].get('fixed_wheel_rays_vs_latest') if selected else None
            complete=bool(value and value['ray_count']==count)
            drop=value['max_assetW_drop'] if complete else None
            extra_wheel_gates.append(dict(name='D:fixed_wheel71:no_new_assetW_drop',raw=raw,state='delete',
                expected_rays=count,actual_rays=value['ray_count'] if value else 0,value=drop,threshold=1e-6,
                passed=None if not complete else drop<=1e-6,
                role='Additional stricter root release rule from posterior V2; original4560 gates unchanged'))
    existing_surface_invariants=[]
    if 'E' in variants or 'F' in variants:
        for row in records+diagnostic:
            if row['variant'] not in ('E','F'):continue
            for field,value in row['existing_surface_colour_only_global_geometry'].items():
                if field=='role':continue
                existing_surface_invariants.append(dict(name=row['variant']+':global_frozen_geometry:'+field,raw=row['raw'],state=row['state'],
                    value=value,threshold=0.0,passed=value==0.0,
                    role='Registered appearance method requires baseline-exact global alpha/depth/groupW; separate from original4560 numeric gates'))
    result = dict(schema='INDEPENDENT_CPU_ROAD_INTEGRATION_FIXED_GATES_V1',
        status='NUMERIC_GATES_PASSED_ROOT_IMAGE_REVIEW_SEPARATE' if all(g['passed'] is not False for g in gates) else 'REGISTERED_NUMERIC_GATES_FAILED',
        contract_sha256=sha(args.contract),script_sha256=sha(Path(__file__)),raw_ids=args.raws,diagnostic_raw_ids=args.diagnostic_raws,
        posterior_method_contract_sha256=sha(args.method_contract) if args.method_contract else None,
        variants_evaluated=list(variants),all_three_variants_complete=set(variants)=={'baseline','AA','C'},
        records=records,diagnostic_records=diagnostic,pooled_world_seams=pooled,numeric_gates=gates,
        failed_gates=[g for g in gates if g['passed'] is False],unsupported_gates=[g for g in gates if g['passed'] is None],
        additional_root_wheel71_release_gates=extra_wheel_gates,
        additional_root_wheel71_complete=all(g['passed'] is not None for g in extra_wheel_gates) if extra_wheel_gates else None,
        E_registered_global_geometry_invariants=[g for g in existing_surface_invariants if g['name'].startswith('E:')],
        F_registered_global_geometry_invariants=[g for g in existing_surface_invariants if g['name'].startswith('F:')],
        appearance_registered_global_geometry_invariants=existing_surface_invariants,
        F_observed_RGB_movement_in_fixed_protection_domains=[dict(raw=row['raw'],state=row['state'],domain=name,
            pixels=row['metrics'][name]['pixels'],**value)
            for row in records+diagnostic if row['variant']=='F' for name,value in row['vs_latest'].items()
            if name in ('a5_RGB','a5_upper_RGB','a6_RGB','protected_nonroad_RGB','a37_RGB')],
        actual_candidate_training_receipt=read(args.candidate_training_receipt) if args.candidate_training_receipt else None,
        source_manifest=manifest,independent_CPU_only=True,CUDA_initialized=False,models_loaded=False,
        optimizer_updates=0,optimizer_updates_scope='Independent CPU evaluation only; NOT source candidate creation or appearance fit',
        candidate_training_accounting='Read actual candidate formal/smoke/restoration receipts; no source-training-zero claim',
        quality_accepted=False,human_verdict=None,
        limits=['Original fixed domains only; RGB MAE is not a pixel-exactness claim.',
                'Runtime asset/source/serialized tuple guards must be joined from their own receipts.',
                'World residual seam and original128 box remain separate; far unsupported is null.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],gates=len(gates),failed=len(result['failed_gates']),unsupported=len(result['unsupported_gates']))))


if __name__=='__main__':
    main()

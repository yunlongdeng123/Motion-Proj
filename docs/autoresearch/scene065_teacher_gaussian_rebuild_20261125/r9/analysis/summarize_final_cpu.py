"""Standalone final CPU summary; never imports training code or initializes CUDA."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np

DEFAULT_ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_background_coverage_fix_20261125_r9')
DEFAULT_R8=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_background_continuation_20261125_r8')
DEFAULT_R7=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_region_rebuild_20261125_r7')
DEFAULT_R5=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_gaussian_rebuild_20261125_r5')
DEFAULT_R6=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_visibility_rebuild_20261125_r6')
READOUT_STEPS=(2600,2616,2700,2800)
ROAD_CATEGORIES=('real_visible','teacher_pseudo','similar_reference')
BG_ID_MIN=10**12
PARAMETERS=('xyz','log_scales','raw_quaternion','opacity_logits','dc')
KINDS=('birth','clone','split','prune')

def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:return default

def pin(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return dict(path=str(path),sha256=h.hexdigest())

def array_file(path):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}

def balanced(values,groups):
    return float(np.mean([np.mean([values[x]for x in group])for group in groups]))

def fixed_summary(phase,point,cfg):
    rows=point['rows'];groups=cfg['view_groups'];rgb={r['raw']:0. for r in rows};categories={}
    for row in rows:
        for name,item in row['categories'].items():
            q=categories.setdefault(name,dict(pixels=0,weight_denominator=0.,error_numerator=0.,values={}))
            mse=item.get('weighted_MSE');den=float(item['weight_denominator']);q['pixels']+=int(item['pixels']);q['weight_denominator']+=den
            q['values'][row['raw']]=float(mse)if mse is not None else 0.
            if mse is not None:q['error_numerator']+=den*mse;rgb[row['raw']]+=cfg['category_weights'][name]*mse
    for q in categories.values():
        q['balanced_group_MSE']=balanced(q.pop('values'),groups)
        q['denominator_pooled_MSE']=q['error_numerator']/q['weight_denominator']if q['weight_denominator']else None
    result=dict(phase=phase,unique_step=int(point['step']),balanced_RGB_MSE_only=balanced(rgb,groups),RGB_categories=categories,rows=rows)
    for field,label in [('protection_RGB','protection_RGB'),('opacity_inside','coverage_inside'),('opacity_outside','coverage_outside'),('A20_visibility','A20_visibility')]:
        selected=[r for r in rows if field in r]
        if not selected:continue
        values={r['raw']:float(r[field]['MSE'])for r in selected};den=lambda r:float(r[field].get('weight_denominator',r[field].get('pixels',0)))
        total=sum(den(r)for r in selected)
        q=dict(fixed_denominator=total,balanced_group_MSE=balanced(values,groups)if len(selected)==len(rows)else None,
               denominator_pooled_MSE=sum(den(r)*r[field]['MSE']for r in selected)/total if total else None)
        if all(r[field].get('mean_alpha')is not None for r in selected):q['balanced_group_mean_alpha']=balanced({r['raw']:r[field]['mean_alpha']for r in selected},groups)
        result[label]=q
    result['balanced_fixed_total_without_prior']=balanced({r['raw']:r['fixed_objective_without_prior']for r in rows},groups)
    return result

def collect(root,run_name='training_v1'):
    run=root/run_name;bindings={};phases={};records=[];points=[]
    for name in ['zero','probe','train']:
        folder=run/name;receipt=read(folder/'receipt.json');b=read(folder/'bindings.json')or(receipt or{}).get('bindings',{})
        if b:bindings=b
        losses=read(folder/'losses.json',[]);records.extend((name,x)for x in losses)
        points.extend((name,x)for x in read(folder/'fixed_eval_curve.json',[]))
        native=read(folder/'native/native30k/native30k_load_receipt.json')
        phases[name]=dict(receipt=receipt,recorded_optimizer_rows=len(losses),native_parameter_freeze_receipt=native,
                          frozen_assets=b.get('frozen_assets'),losses_source=str(folder/'losses.json'))
    cfg=read(bindings['configuration']['path']);spec=bindings['cache_manifest'];cache=read(spec['path'])
    fixed=[fixed_summary(phase,point,cfg)for phase,point in points]
    for point in fixed:
        point['road_RGB_balanced_MSE_only']=sum(cfg['category_weights'][k]*point['RGB_categories'][k]['balanced_group_MSE']for k in ROAD_CATEGORIES)
        point['real_visible_background_RGB_balanced_MSE_only']=point['RGB_categories'].get('real_visible_background',{}).get('balanced_group_MSE')
        point['background_unique_age']=max(0,point['unique_step']-2000)
    return dict(root=root,run=run,bindings=bindings,cfg=cfg,cache=cache,phases=phases,records=records,fixed=fixed)

def canonical(rgb,cache):
    rgb=np.asarray(rgb,np.float32)
    if rgb.shape[0]!=3 and rgb.shape[-1]==3:rgb=rgb.transpose(2,0,1)
    x=np.clip(rgb,0,1)
    linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    return linear*cache['gain'][:,None,None]+cache['bias'][:,None,None]

def rgb_readout_row(raw,rgb,arrays,frame):
    pred=canonical(rgb,arrays);categories={}
    for name,item in frame['categories'].items():
        support=arrays[f'{name}__support'].astype(bool);weight=arrays[f'{name}__weight']
        den=float(item['final_weight_denominator']);error=np.square(pred-arrays[f'{name}__rgb']).mean(0)
        numerator=float(np.sum(error[support]*weight[support],dtype=np.float64))
        categories[name]=dict(pixels=int(support.sum()),weight_denominator=den,
            error_numerator=numerator,weighted_MSE=numerator/den if den else None)
    return dict(raw=int(raw),categories=categories)

def summarize_rgb_rows(rows,cfg):
    values={r['raw']:0. for r in rows};categories={}
    complete=set(values)==set(cfg['train_frames'])
    for row in rows:
        for name,item in row['categories'].items():
            mse=item['weighted_MSE'];q=categories.setdefault(name,dict(pixels=0,weight_denominator=0.,error_numerator=0.,values={}))
            q['pixels']+=item['pixels'];q['weight_denominator']+=item['weight_denominator'];q['error_numerator']+=item.get('error_numerator',float(mse or 0)*item['weight_denominator'])
            q['values'][row['raw']]=float(mse or 0);values[row['raw']]+=cfg['category_weights'][name]*float(mse or 0)
    for q in categories.values():
        q['balanced_group_MSE']=balanced(q.pop('values'),cfg['view_groups'])if complete else None
        q['denominator_pooled_MSE']=q['error_numerator']/q['weight_denominator']if q['weight_denominator']else None
    return dict(complete_eight_train_frames=complete,balanced_RGB_MSE_only=balanced(values,cfg['view_groups'])if complete else None,
        RGB_categories=categories,rows=rows)

def readout_mask_metrics(raw,joint,cache):
    rgb=np.clip(joint['rgb'],0,1);reference_error=np.square(rgb-cache['reference_RGB_LDR']).mean(0)
    local=np.squeeze(joint['local_effective_alpha']);actor=np.squeeze(joint['actor_alpha']);result={}
    for name in ['red','green','orange','protection','alpha_inside','alpha_outside','background_rebuild','background_geometry_scope','background_unknown']:
        if name not in cache:continue
        mask=cache[name].astype(bool);den=int(mask.sum())
        result[name]=dict(fixed_pixels=den,local_effective_alpha_mean=float(local[mask].mean())if den else None,
            local_effective_alpha_gt_0_5_fraction=float((local[mask]>.5).mean())if den else None,
            local_effective_alpha_gt_0_9_fraction=float((local[mask]>.9).mean())if den else None,
            retained_A20_source21_effective_alpha_mean=float(actor[mask].mean())if den else None,
            unweighted_reference_RGB_LDR_MSE_diagnostic=float(reference_error[mask].mean())if den else None,
            fixed_alpha_target_MSE=float(np.square(local-np.squeeze(cache['alpha_target']))[mask].mean())if den else None)
    if 'protection_RGB_weight' in cache:
        full=int(cache['protection'].sum());weight=cache['protection_RGB_weight']
        result['protection']['training_weighted_reference_MSE_with_full_denominator']=float(np.sum(reference_error*weight,dtype=np.float64))/full if full else None
        result['protection']['RGB_weight_sum']=float(weight.sum(dtype=np.float64))
        result['protection']['fullmask_diagnostic_is_separate_from_training_weighted_loss']=True
    ref=np.squeeze(cache['A20_alpha_reference']);den=float(ref.sum(dtype=np.float64))
    result['A20_visibility']=dict(reference_alpha_weight_denominator=den,
        one_sided_decrease_MSE=float(np.sum(ref*np.maximum(ref-actor,0)**2,dtype=np.float64))/den if den else None,
        reference_effective_alpha_sum=den,actual_effective_alpha_sum=float(actor.sum(dtype=np.float64)),
        alpha_identity='actual retained checkpoint source21; no semantic/source label mixing')
    return dict(raw=raw,domains=result)

def compare_fixed_RGB(control,current):
    """Re-score actual R9 pixels on unchanged R8 road and legal BG domains."""
    old={int(r['raw']):r for r in control['cache']['frame_rows']};new={int(r['raw']):r for r in current['cache']['frame_rows']}
    checks=[];photo_checks=[];legacy={s:[]for s in READOUT_STEPS};own={s:[]for s in READOUT_STEPS};sources={s:[]for s in READOUT_STEPS};mask_rows={s:[]for s in READOUT_STEPS};missing=[]
    for raw in sorted(new):
        a=array_file(old[raw]['cache']['path']);b=array_file(new[raw]['cache']['path'])
        assert pin(old[raw]['cache']['path'])['sha256']==old[raw]['cache']['sha256']
        assert pin(new[raw]['cache']['path'])['sha256']==new[raw]['cache']['sha256']
        bg=b['background_rebuild'].astype(bool)
        linear=(b['real_visible_background__rgb']-b['bias'][:,None,None])/b['gain'][:,None,None]
        expected=np.clip(np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055),0,1)
        item=new[raw]['categories']['real_visible_background']
        photo_checks.append(dict(raw=raw,original_BG_support_pixels=item['original_pixels'],fixed_BG_RGB_pixels=item['final_pixels'],
            fixed_BG_RGB_weight_denominator=item['final_weight_denominator'],background_rebuild_pixels=int(bg.sum()),
            full_orange_pixels=int(b['orange'].sum()),full_green_pixels=int(b['green'].sum()),source=item['source'],
            photo_target_vs_BG_protection_reference_max_abs=float(np.abs(expected[:,bg]-b['reference_RGB_LDR'][:,bg]).max())if bg.any()else None))
        unknown=b['background_unknown'].astype(bool)
        photo_checks[-1].update(background_geometry_scope_pixels=int(b['background_geometry_scope'].sum()),unknown_BG_pixels=int(unknown.sum()),
            unknown_BG_category_RGB_weight_sums={k:float(b[f'{k}__weight'][unknown].sum(dtype=np.float64))for k in new[raw]['categories']},
            unknown_BG_protection_RGB_weight_sum=float(b['protection_RGB_weight'][unknown].sum(dtype=np.float64)),
            full_protection_pixels=int(b['protection'].sum()),R8_full_protection_pixels=int(a['protection'].sum()),
            full_protection_mask_equal_to_R8=bool(np.array_equal(a['protection'],b['protection'])))
        for category in (*ROAD_CATEGORIES,'real_visible_background'):
            sk=f'{category}__support';wk=f'{category}__weight';rk=f'{category}__rgb';sa=a[sk].astype(bool);sb=b[sk].astype(bool)
            checks.append(dict(raw=raw,category=category,support_equal=bool(np.array_equal(sa,sb)),weight_equal=bool(np.array_equal(a[wk],b[wk])),
                target_RGB_equal=bool(np.array_equal(a[rk],b[rk])),r8_control_pixels=int(sa.sum()),r9_pixels=int(sb.sum()),
                canonical_gain_bias_equal=bool(np.array_equal(a['gain'],b['gain'])and np.array_equal(a['bias'],b['bias'])),
                r8_control_weight_denominator=float(old[raw]['categories'][category]['final_weight_denominator']),
                r9_weight_denominator=float(new[raw]['categories'][category]['final_weight_denominator'])))
        for step in READOUT_STEPS:
            phase='probe'if step==2616 else'train'
            run=control['run']if step==2600 else current['run']
            folder=run/phase/f'actual_step{step:06d}';path=folder/f'{raw:03d}'/'joint.npz'
            if not path.is_file():missing.append(dict(step=step,raw=raw,path=str(path)));continue
            with np.load(path,allow_pickle=False)as z:joint={k:z[k]for k in ['rgb','local_effective_alpha','actor_alpha']}
            rgb=joint['rgb']
            legacy[step].append(rgb_readout_row(raw,rgb,a,old[raw]));own[step].append(rgb_readout_row(raw,rgb,b,new[raw]))
            mask_rows[step].append(readout_mask_metrics(raw,joint,b))
            sources[step].append(str(path))
    baseline=next(x for x in control['fixed']if x['unique_step']==2600);new_points={x['unique_step']:x for x in current['fixed']};points=[]
    for step in READOUT_STEPS:
        before=baseline['balanced_RGB_MSE_only'];q=summarize_rgb_rows(legacy[step],control['cfg']);n=summarize_rgb_rows(own[step],current['cfg'])
        after=q['balanced_RGB_MSE_only'];saved=new_points.get(step,{}).get('balanced_RGB_MSE_only')
        common_road=sum(control['cfg']['category_weights'][k]*q['RGB_categories'][k]['balanced_group_MSE']for k in ROAD_CATEGORIES)if q['complete_eight_train_frames']else None
        common_BG=q['RGB_categories'].get('real_visible_background',{}).get('balanced_group_MSE')
        control_road=sum(control['cfg']['category_weights'][k]*baseline['RGB_categories'][k]['balanced_group_MSE']for k in ROAD_CATEGORIES)
        control_BG=baseline['RGB_categories']['real_visible_background']['balanced_group_MSE']
        mask_summary={}
        for domain in ['green','orange','protection','background_geometry_scope','background_unknown']:
            selected=[(r['raw'],r['domains'][domain])for r in mask_rows[step]if domain in r['domains']]
            den=sum(r['fixed_pixels']for _,r in selected);dq=dict(fixed_pixels=den)
            for key in ['local_effective_alpha_mean','local_effective_alpha_gt_0_5_fraction','local_effective_alpha_gt_0_9_fraction','fixed_alpha_target_MSE','unweighted_reference_RGB_LDR_MSE_diagnostic']:
                valid=[(raw,r)for raw,r in selected if r[key]is not None]
                dq[f'pooled_{key}']=sum(r['fixed_pixels']*r[key]for _,r in valid)/den if den else None
                dq[f'balanced_{key}']=balanced({raw:r[key]for raw,r in valid},current['cfg']['view_groups'])if len(valid)==8 else None
            mask_summary[domain]=dq
        points.append(dict(unique_step=step,background_unique_age=step-2000,R9_additional_age=step-2600,
            source_R8_control_step2600_four_class_RGB_balanced_MSE=before,R9_four_class_RGB_on_R8_support_balanced_MSE=after,
            R8_control_road_RGB_balanced_MSE=control_road,R9_common_road_RGB_balanced_MSE=common_road,
            R8_control_legal_BG_RGB_balanced_MSE=control_BG,R9_common_legal_BG_RGB_balanced_MSE=common_BG,
            relative_change=(after/before-1)if before and after is not None else None,
            r9_recomputed_four_classes_on_own_support=n,r9_saved_fixed_four_class_RGB_balanced_MSE=saved,
            CPU_vs_saved_own_RGB_abs_difference=abs(n['balanced_RGB_MSE_only']-saved)if step!=2600 and saved is not None and n['balanced_RGB_MSE_only']is not None else None,
            actual_2600_source_is_R8_before_context_policy_binding=step==2600,
            r9_on_R8_fixed_RGB_domains=q,fullmask_raw_diagnostic_summaries=mask_summary,
            fullmask_raw_diagnostics_on_R9_fixed_masks=mask_rows[step],source_actual_joint_NPZs=sources[step]))
    return dict(source_R8_control_step=2600,legal_RGB_support_weights_and_targets_unchanged=all(x['support_equal']and x['weight_equal']and x['target_RGB_equal']and x['canonical_gain_bias_equal']for x in checks),
        recomputed_R9_uses_exact_R8_RGB_domains=True,common_definition='Actual R8-2600/R9 joint RGB uses fixed R8 road and legal BG targets/support/weights/denominators, gain/bias and sampler groups. Road and BG components are reported separately.',
        per_frame_category_checks=checks,common_step_RGB=points,missing_actual_readouts=missing,
        real_photo_BG_fixed_denominators_and_reference_checks=photo_checks,
        all_four_common_readouts_complete=not missing,heldout_RGB_evaluated=False,
        mixed_total_comparable=False,coverage_targets_comparable=False,
        note='At2600 the raw image is the saved R8 control; its coverage and fullmask diagnostics are recomputed on the fixed R9 geometry/protection domains. R9 fixed evaluation at2600 is separately labelled after policy binding.')

def geometry_summary(data):
    root=data['root'];cfg=read(data['bindings']['runtime_contract']['path']);field_path=data['run']/'train/new_field_final.npz'
    if not field_path.exists():return None
    seed=array_file(cfg['teacher_seed']['path']);field=array_file(field_path);pose=np.asarray(read(cfg['pose']['path'])['world_from_patch'],np.float64)
    origin_local=np.asarray(seed.get('prior_xyz',seed['xyz'])[0],np.float64);normal_local=np.asarray(seed.get('normal_local',seed.get('prior_normal'))[0],np.float64)
    normal_local/=np.linalg.norm(normal_local);origin=origin_local@pose[:3,:3].T+pose[:3,3];normal=normal_local@pose[:3,:3].T;normal/=np.linalg.norm(normal)
    expected_world=field['xyz']@pose[:3,:3].T+pose[:3,3];world=field.get('world_xyz',expected_world);signed=(world-origin)@normal;absolute=np.abs(signed)
    def stats(values):
        if not len(values):return None
        q=np.quantile(values,[.5,.9,.99,1]);return dict(P50=float(q[0]),P90=float(q[1]),P99=float(q[2]),max=float(q[3]))
    high=field['opacity']>=.05
    scales=np.asarray(field['scales']);opacity=np.asarray(field['opacity']).reshape(-1)
    background=np.asarray(field['source_candidate_ids'],np.int64)>=BG_ID_MIN
    group_stats={}
    for name,mask in [('road',~background),('real_visible_background',background)]:
        group_stats[name]=dict(rows=int(mask.sum()),absolute_height_metres=stats(absolute[mask]),
            maximum_axis_scale_metres=stats(scales[mask].max(-1)),opacity=stats(opacity[mask]))
    return dict(final_field=pin(field_path),rows=len(world),height_reference='fixed original teacher seed plane, not split/clone prior_xyz',
                origin_world=origin.tolist(),normal_world=normal.tolist(),absolute_height_metres=stats(absolute),
                signed_height_metres=dict(min=float(signed.min()),median=float(np.median(signed)),max=float(signed.max())),
                opacity_ge_0_05_rows=int(high.sum()),opacity_ge_0_05_absolute_height_metres=stats(absolute[high]),
                world_xyz_vs_pose_max_abs_error_m=float(np.abs(world-expected_world).max()),road_height_sigma_m=cfg.get('road_height_sigma_m'),
                maximum_axis_scale_metres=stats(scales.max(-1)),opacity=stats(opacity),
                point_sources=group_stats,background_ID_rule=f'source_candidate_ids >= {BG_ID_MIN}',
                seed_absolute_height_metres=stats(np.abs((np.asarray(seed['xyz'],np.float64)-origin_local)@normal_local)),
                displacement_from_current_lineage_prior_m=stats(np.linalg.norm(np.asarray(field['xyz'])-np.asarray(field['prior_xyz']),axis=-1)))

def fragment_summary(data):
    runtime=read(data['bindings']['runtime_contract']['path']);spec=runtime.get('retained_fragments')
    if not spec:return dict(enabled=False)
    path=Path(spec['path']if isinstance(spec,dict)else spec);actual=pin(path);recipe=array_file(path)
    parents=np.asarray(recipe['parent_global_row_ids'],np.int64);unique=np.unique(parents);retire_spec=runtime['context_retirement'];retire=array_file(retire_spec['path'])
    def stats(x):
        if not len(x):return None
        return dict(zip(['min','P50','P90','P99','max'],map(float,np.quantile(x,[0,.5,.9,.99,1]))))
    checks={phase:dict(recipe_hash_in_frozen_assets=evidence.get('frozen_assets',{}).get('retained_fragments')==actual['sha256'],
        native_load_receipt=evidence['native_parameter_freeze_receipt'])for phase,evidence in data['phases'].items()if evidence.get('frozen_assets')}
    receipt=read(path.parent/'receipt.json')
    return dict(enabled=True,recipe=actual,contract_hash_matches=not isinstance(spec,dict)or actual['sha256']==spec['sha256'],
        frozen_children=len(parents),unique_original_parent_rows=len(unique),all_parents_retired=bool(np.isin(unique,retire['remove_global_row_ids']).all()),
        retired_original_rows=len(retire['remove_global_row_ids']),scale_factors=stats(np.asarray(recipe['scale_factors'])),
        opacity_factors=stats(np.asarray(recipe['opacity_factors'])),standardized_offset_norm=stats(np.linalg.norm(recipe['offsets'],axis=-1)),
        phase_frozen_asset_evidence=checks,preparation_receipt=receipt,children_in_optimizer=False,
        physical_child_near_m=.2,original_prefix_native_near_preserved=True,
        representation='offline fixed 3D Gaussian fragments; one native raster; no per-view image gate')

def execution_summary(data):
    records=data['records'];phases=data['phases'];train=phases['train']['receipt'];replay=read(data['run']/'train/step32_replay.json')
    ledger=(train or{}).get('execution_ledger',{});totals={k:0 for k in KINDS};events=[]
    groups={k:dict(updates=0,finite_grad_updates=0,positive_gradient_updates=0,positive_delta_updates=0,changed_values=0,gradient_nonzero_values=0,
        gradient_l2_sum=0.,delta_l2_sum=0.,gradient_l2_max=0.,delta_max_abs=0.)for k in PARAMETERS}
    initial=(phases['zero']['receipt']or{}).get('point_count',len(array_file(read(data['bindings']['runtime_contract']['path'])['teacher_seed']['path'])['xyz']))
    history=[dict(step=0,rows=initial)]
    for phase,x in records:
        record=x['record']
        for name,q in groups.items():
            grad=record['gradients_by_group'][name];delta=record['parameter_delta_by_group'][name];q['updates']+=1;q['finite_grad_updates']+=int(grad['finite'])
            q['positive_gradient_updates']+=int(grad['l2']>0);q['positive_delta_updates']+=int(delta['l2']>0);q['changed_values']+=delta['nonzero_count'];q['gradient_nonzero_values']+=grad['nonzero_count']
            q['gradient_l2_sum']+=grad['l2'];q['delta_l2_sum']+=delta['l2'];q['gradient_l2_max']=max(q['gradient_l2_max'],grad['l2']);q['delta_max_abs']=max(q['delta_max_abs'],delta['max_abs'])
        event=x.get('event')or record.get('topology')
        if event:
            e={k:event[k]for k in ['logical_step','rows_before','rows_after',*KINDS]};e['phase']=phase;events.append(e);history.append(dict(step=x['step'],rows=event['rows_after']))
            for k in KINDS:totals[k]+=event[k]
    unique=ledger.get('unique_updates',len(records));replay_count=ledger.get('engineering_replay_updates',int(bool(replay)));actual=ledger.get('actual_optimizer_updates',unique+replay_count)
    wiring=read(data['root']/'gpu_wiring_v1/receipt.json',{});wiring_count=wiring.get('engineering_optimizer_updates',0)
    receipts=[p['receipt']for p in phases.values()if p['receipt']]
    summary=dict(latest_saved_step=max((x['step']for _,x in records),default=0),formal_unique_updates=unique,formal_actual_optimizer_updates=actual,
        engineering_replay_updates=replay_count,engineering_wiring_optimizer_updates=wiring_count,recorded_physical_optimizer_updates=actual+wiring_count,
        optimizer_step_attempts=sum(p.get('optimizer_step_attempts',0)for p in receipts),
        failed_or_uncommitted_optimizer_updates=sum(p.get('failed_or_uncommitted_optimizer_updates',0)for p in receipts),
        unique_training_seconds=sum(x['active_seconds']for _,x in records),engineering_replay_seconds=sum(p.get('engineering_replay_seconds',0)for p in receipts),
        cache_seconds=sum(p.get('cache_seconds',0)for p in receipts),evaluation_seconds=sum(p.get('evaluation_seconds',0)for p in receipts),
        readout_seconds=sum(p.get('readout_seconds',0)for p in receipts),phase_wall_seconds=sum(p.get('wall_seconds',0)for p in receipts),
        topology_unique=ledger.get('topology_unique',totals),topology_replay=ledger.get('topology_replay'),
        topology_from_saved_unique_events=totals,topology_ledger_consistent=ledger.get('topology_unique',totals)==totals,
        count_consistent=actual==unique+replay_count,old_r5_or_r4_optimized_state_loaded=False,
        native_forward_and_backward_calls_by_phase={k:dict(forward=(p['receipt']or{}).get('actual_native_forward_calls'),backward=(p['receipt']or{}).get('actual_native_backward_calls'))for k,p in phases.items()})
    summary['recorded_unique_rows']=len(records);summary['unique_step_numbers_exact']=sorted(x['step']for _,x in records)==list(range(1,int(unique)+1))
    summary['group_update_statistics_scope']='Unique formal update records; replay is accounted separately by execution ledger and fresh-process replay evidence.'
    summary['memory_peak_bytes']=next((r.get('peak_GPU_bytes',r.get('peak_memory_allocated_bytes'))for r in reversed(receipts)if r.get('peak_GPU_bytes',r.get('peak_memory_allocated_bytes'))is not None),None)
    if summary['memory_peak_bytes']is None:summary['memory_note']='Peak allocation was not recorded in the phase receipts; CPU helper does not infer it from current device state.'
    if train and history[-1]['step']!=int(train['logical_step']):history.append(dict(step=int(train['logical_step']),rows=train['final_point_count']))
    return summary,groups,history,events,replay

def group_update_totals(records):
    groups={k:dict(updates=0,finite_grad_updates=0,positive_gradient_updates=0,positive_delta_updates=0,
        changed_values=0,gradient_nonzero_values=0,gradient_l2_sum=0.,delta_l2_sum=0.,gradient_l2_max=0.,delta_max_abs=0.)for k in PARAMETERS}
    for _,x in records:
        for name,q in groups.items():
            grad=x['record']['gradients_by_group'][name];delta=x['record']['parameter_delta_by_group'][name]
            q['updates']+=1;q['finite_grad_updates']+=int(grad['finite']);q['positive_gradient_updates']+=int(grad['l2']>0);q['positive_delta_updates']+=int(delta['l2']>0)
            q['changed_values']+=delta['nonzero_count'];q['gradient_nonzero_values']+=grad['nonzero_count'];q['gradient_l2_sum']+=grad['l2'];q['delta_l2_sum']+=delta['l2']
            q['gradient_l2_max']=max(q['gradient_l2_max'],grad['l2']);q['delta_max_abs']=max(q['delta_max_abs'],delta['max_abs'])
    return groups

def saved_topology_events(records):
    events=[];totals={k:0 for k in KINDS}
    for phase,x in records:
        event=x.get('event')or x['record'].get('topology')
        if event:
            e={k:event[k]for k in ['logical_step','rows_before','rows_after',*KINDS]};e['phase']=phase;events.append(e)
            for k in KINDS:totals[k]+=e[k]
    return events,totals

def continuation_execution_summary(source,data):
    transition=read(data['run']/'probe/transition.json');replay=read(data['run']/'probe/step2016_replay.json')
    old_receipt=source['phases']['train']['receipt'];final=data['phases']['train']['receipt']or{}
    old_ledger=old_receipt['execution_ledger'];ledger=final.get('execution_ledger',{})
    old_events,old_counts=saved_topology_events(source['records']);events,new_counts=saved_topology_events(data['records'])
    expected_topology={k:old_counts[k]+new_counts[k]for k in KINDS}
    old_replay=read(source['run']/'train/step32_replay.json')
    new_replay_event=next((x for x in events if x['logical_step']==2016),None)
    expected_replay_topology={k:old_ledger['topology_replay'][k]+(new_replay_event[k]if new_replay_event else 0)for k in KINDS}
    phase_records={'source_R7_2000':source['records'],'R8_probe_16':[(p,x)for p,x in data['records']if p=='probe'],
        'R8_train_584':[(p,x)for p,x in data['records']if p=='train']}
    group_stats={name:group_update_totals(rows)for name,rows in phase_records.items()}
    group_stats['cumulative_2600']=group_update_totals(source['records']+data['records'])
    source_steps=sorted(x['step']for _,x in source['records']);new_steps=sorted(x['step']for _,x in data['records'])
    unique=int(ledger.get('unique_updates',2000+len(new_steps)));actual=int(ledger.get('actual_optimizer_updates',old_ledger['actual_optimizer_updates']+len(new_steps)+int(bool(replay))))
    replay_count=int(ledger.get('engineering_replay_updates',old_ledger['engineering_replay_updates']+int(bool(replay))))
    added=int(transition['background_rows_added']);initial=int(transition['old_rows']);net=sum(new_counts[k]for k in ('birth','clone','split'))-new_counts['prune']
    history=[dict(step=2000,stage='R7_control_before_append',rows=initial),dict(step=2000,stage='R8_after_BG_append_before_optimizer',rows=initial+added)]
    history.extend(dict(step=e['logical_step'],stage='R8_density_event',rows=e['rows_after'])for e in events)
    if final:history.append(dict(step=final['logical_step'],stage='R8_final',rows=final['final_point_count']))
    old_active=sum(x['active_seconds']for _,x in source['records']);new_active=sum(x['active_seconds']for _,x in data['records'])
    summary=dict(road_cumulative_unique_updates=unique,background_birth_at_road_step=2000,background_family_unique_training_updates=len(new_steps),
        source_R7_unique_updates=len(source_steps),R8_probe_unique_updates=len(phase_records['R8_probe_16']),R8_train_unique_updates=len(phase_records['R8_train_584']),
        cumulative_actual_optimizer_updates=actual,cumulative_engineering_replay_updates=replay_count,
        R8_actual_optimizer_updates=actual-old_ledger['actual_optimizer_updates'],R8_additional_engineering_replay_updates=replay_count-old_ledger['engineering_replay_updates'],
        source_steps_exact=source_steps==list(range(1,2001)),R8_steps_exact=new_steps==list(range(2001,unique+1)),count_consistent=actual==unique+replay_count,
        cumulative_execution_ledger=ledger,source_R7_execution_ledger=old_ledger,
        topology_source_R7_unique=old_counts,topology_R8_unique=new_counts,topology_cumulative_unique_expected=expected_topology,
        topology_unique_ledger_consistent=ledger.get('topology_unique')==expected_topology,
        topology_cumulative_replay_expected=expected_replay_topology,topology_replay_ledger_consistent=ledger.get('topology_replay')==expected_replay_topology,
        BG_append_outside_optimizer=dict(step=2000,rows=added,optimizer_updates=0,old_road_rows=initial,rows_after=initial+added),
        final_point_count_from_row_arithmetic=initial+added+net,final_point_count_from_receipt=final.get('final_point_count'),
        point_count_consistent=not final or initial+added+net==final['final_point_count'],
        source_R7_training_seconds=old_active,R8_unique_training_seconds=new_active,cumulative_unique_training_seconds=old_active+new_active,
        R8_replay_engineering_seconds=(replay or{}).get('engineering_seconds'),
        source_R7_replay_engineering_seconds=old_receipt.get('engineering_replay_seconds'),
        gradient_delta_statistics_scope='Unique updates only; source R7 2000 + R8 probe16 + R8 train584. Both extra replay updates remain separate in the actual ledger.',
        previous_optimized_road_reinitialized=False,previous_road_optimizer_reset=False)
    return summary,group_stats,history,events,dict(source_R7_step32=old_replay,R8_step2016=replay),transition

def coverage_execution_summary(r7,r8,data):
    transition=read(data['run']/'probe/transition.json');certificate=read(data['run']/'probe/step2616_replay.json')
    diff=read(data['run']/'probe/replay_state_diff.json')
    parent=r8['phases']['train']['receipt'];final=data['phases']['train']['receipt']or{}
    old_ledger=parent['execution_ledger'];ledger=final.get('execution_ledger',{})
    old_records=r7['records']+r8['records'];all_records=old_records+data['records']
    _,old_counts=saved_topology_events(old_records);events,new_counts=saved_topology_events(data['records'])
    expected_topology={k:old_counts[k]+new_counts[k]for k in KINDS}
    replay_event=next((e for e in events if e['logical_step']==2616),None)
    stage_replays=int(certificate['engineering_replay_updates_this_stage'])
    expected_replay_topology={k:old_ledger['topology_replay'][k]+stage_replays*(replay_event[k]if replay_event else 0)for k in KINDS}
    stages={'source_R7_2000':r7['records'],'source_R8_600':r8['records'],
        'R9_probe_16':[(p,x)for p,x in data['records']if p=='probe'],'R9_train_184':[(p,x)for p,x in data['records']if p=='train'],
        'cumulative_2800':all_records}
    groups={name:group_update_totals(rows)for name,rows in stages.items()}
    unique=int(ledger.get('unique_updates',2600+len(data['records'])));actual=int(ledger.get('actual_optimizer_updates',2602+len(data['records'])+stage_replays))
    replay_count=int(ledger.get('engineering_replay_updates',2+stage_replays));added=int(transition['background_rows_added']);initial=int(transition['old_rows'])
    net=sum(new_counts[k]for k in ('birth','clone','split'))-new_counts['prune']
    history=[dict(step=2600,stage='R8_control',rows=initial),dict(step=2600,stage='R9_policy_binding_empty_append',rows=initial+added)]
    history.extend(dict(step=e['logical_step'],stage='R9_density_event',rows=e['rows_after'])for e in events)
    if final:history.append(dict(step=final['logical_step'],stage='R9_final',rows=final['final_point_count']))
    replay_evidence=dict(source_R7_step32=read(r7['run']/'train/step32_replay.json'),source_R8_step2016=read(r8['run']/'probe/step2016_replay.json'),
        R9_step2616_parameter_aware_certificate=certificate,R9_replay_state_diff=diff,
        R9_nextstep_bit_exact=False,R9_nextstep_original_raw_tolerance_passed=certificate['raw_log_scale_comparison_passed'],
        R9_physical_scale_equivalence_passed=certificate['physical_parameter_equivalence'],
        exact_unique_probe_state_used_for_continuation=certificate['continue_from_exact_unique_probe_state'],
        explanation='Pre-step state was restored exactly. Next-step replay uses the recorded physical-scale certificate; raw log_scales exceeded original4eps for2 values. Continue from the exact saved unique-probe expected2616 state, not the replay result.')
    summary=dict(road_cumulative_unique_updates=unique,background_birth_at_road_step=2000,background_family_unique_training_updates=len(r8['records'])+len(data['records']),
        source_R7_unique_updates=len(r7['records']),source_R8_unique_updates=len(r8['records']),R9_unique_updates=len(data['records']),
        R9_probe_unique_updates=len(stages['R9_probe_16']),R9_train_unique_updates=len(stages['R9_train_184']),
        cumulative_actual_optimizer_updates=actual,cumulative_engineering_replay_updates=replay_count,R9_actual_optimizer_updates=actual-old_ledger['actual_optimizer_updates'],
        R9_additional_engineering_replay_updates=stage_replays,failed_original_raw_tolerance_extra_updates=certificate['failed_raw_tolerance_replays'],
        source_directory_collision_retry_optimizer_updates=certificate['load_retry_optimizer_updates'],
        unique_step_numbers_exact=sorted(x['step']for _,x in all_records)==list(range(1,unique+1)),count_consistent=actual==unique+replay_count,
        cumulative_execution_ledger=ledger,source_R8_execution_ledger=old_ledger,topology_R9_unique=new_counts,
        topology_cumulative_unique_expected=expected_topology,topology_unique_ledger_consistent=ledger.get('topology_unique')==expected_topology,
        topology_cumulative_replay_expected=expected_replay_topology,topology_replay_ledger_consistent=ledger.get('topology_replay')==expected_replay_topology,
        append_outside_optimizer=dict(step=2600,rows=added,optimizer_updates=0),no_new_seed_or_candidate_rows=added==0,
        final_point_count_from_row_arithmetic=initial+added+net,final_point_count_from_receipt=final.get('final_point_count'),
        point_count_consistent=not final or initial+added+net==final['final_point_count'],
        unique_training_seconds_by_source={name:sum(x['active_seconds']for _,x in rows)for name,rows in [('R7',r7['records']),('R8',r8['records']),('R9',data['records'])]},
        unique_training_seconds_cumulative=sum(x['active_seconds']for _,x in all_records),
        gradient_delta_statistics_scope='All2800 unique records grouped by R7/R8/R9. Four extra optimizer replays remain separate; failed raw-tolerance replay is included in actual ledger.',
        optimized_parameters_reinitialized=False,Adam_reset=False)
    return summary,groups,history,events,replay_evidence,transition

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--r7-root',type=Path,default=DEFAULT_R7)
    p.add_argument('--r8-root',type=Path,default=DEFAULT_R8)
    p.add_argument('--run',default='training_v1');p.add_argument('--r7-run',default='training_v1');p.add_argument('--r8-run',default='training_v1');p.add_argument('--output',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    data=collect(a.root,a.run);final=data['phases']['train']['receipt']
    if not a.allow_partial and not(final and final.get('complete')and final.get('logical_step')==2800):p.error('Final complete cumulative2800/BG800-step receipt required; no polling is performed.')
    r7=collect(a.r7_root,a.r7_run);source=collect(a.r8_root,a.r8_run);comparison=compare_fixed_RGB(source,data)
    accounting,groups,history,events,replay,transition=coverage_execution_summary(r7,source,data)
    for point in data['fixed']:
        point['protection_RGB_training_weighted_fixed_denominator']=point.pop('protection_RGB',None)
        rows=[r for r in point['rows']if 'unknown_BG'in r];den=sum(r['unknown_BG']['pixels']for r in rows)
        point['unknown_BG_fixed_monitor']=dict(fixed_pixels=den,RGB_loss_weight=0,
            balanced_mean_local_alpha=balanced({r['raw']:r['unknown_BG']['mean_local_alpha']for r in rows},data['cfg']['view_groups'])if len(rows)==8 else None,
            denominator_pooled_mean_local_alpha=sum(r['unknown_BG']['pixels']*r['unknown_BG']['mean_local_alpha']for r in rows)/den if den else None,
            balanced_alpha_target_MSE=balanced({r['raw']:r['unknown_BG']['alpha_target_MSE']for r in rows},data['cfg']['view_groups'])if len(rows)==8 else None,
            denominator_pooled_alpha_target_MSE=sum(r['unknown_BG']['pixels']*r['unknown_BG']['alpha_target_MSE']for r in rows)/den if den else None,
            protection_RGB_weight_sum=sum(r['unknown_BG']['protection_RGB_weight_sum']for r in rows))
    runtime=read(data['bindings']['runtime_contract']['path']);background_spec=runtime['real_background_manifest'];background=read(background_spec['path'])
    background_hash_matches=pin(background_spec['path'])['sha256']==background_spec['sha256']
    assert background_hash_matches
    native_receipts=[q['native_parameter_freeze_receipt']for q in data['phases'].values()if q['native_parameter_freeze_receipt']]
    control=next(x for x in source['fixed']if x['unique_step']==2600)
    photo_rows=comparison['real_photo_BG_fixed_denominators_and_reference_checks']
    invariants=dict(legal_BG_fixed_RGB_pixels_total=sum(r['fixed_BG_RGB_pixels']for r in photo_rows),
        legal_BG_fixed_RGB_weight_denominator_total=sum(r['fixed_BG_RGB_weight_denominator']for r in photo_rows),
        raw110=next(r for r in photo_rows if r['raw']==110),unknown_BG_total_pixels=sum(r['unknown_BG_pixels']for r in photo_rows),
        all_unknown_BG_category_RGB_weights_zero=all(all(v==0 for v in r['unknown_BG_category_RGB_weight_sums'].values())for r in photo_rows),
        all_unknown_BG_protection_RGB_weights_zero=all(r['unknown_BG_protection_RGB_weight_sum']==0 for r in photo_rows),
        full_protection_masks_and_denominators_equal_R8=all(r['full_protection_mask_equal_to_R8']for r in photo_rows))
    old_runtime=read(source['bindings']['runtime_contract']['path'])
    invariants['teacher_seed_unchanged']=runtime['teacher_seed']==old_runtime['teacher_seed']
    invariants['candidate_pool_unchanged']=runtime['candidate_pool']==old_runtime['candidate_pool']
    invariants['real_photo_source_manifest_unchanged']=runtime['real_background_manifest']==old_runtime['real_background_manifest']
    result=dict(schema='scene065.teacher_background_coverage_fix.final_metrics.v1',updated_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,GPU_started=False,
        complete=bool(final and final.get('complete')and final.get('logical_step')==2800),bindings=data['bindings'],
        definitions=dict(RGB='strict fixed eight-frame category canonical-linear MSE, separately balanced across four sampler groups',
            mixed_total='fixed_objective_without_prior; includes protection/coverage/A20 terms, never labelled RGB MSE',
            road_cumulative='R7 2000 + R8 600 + R9 200 unique updates; no parameter or Adam reset at either transition',
            background='Real-photo BG family born at road2000, optimized through cumulative2800:800 unique family updates; R9 append is empty',
            r9_inside='effective local alpha target=1-frozen retained foreground on fixed road+fullBG geometry_scope; unknownBG has geometry/visibility supervision and zero direct RGB weight',
            A20_visibility='actual checkpoint source21 effective-alpha preservation; identity fixed by current source mapping',
            denominators='All four legal RGB domains/weights/denominators are checked against fixed R8 cache. Common-road and legal-BG RGB are separate components',
            protection='Full green/orange mask and denominator remain fixed; unknownBG per-pixel protection RGB weight0. Raw fullmask RGB differences are computed separately from actual joint arrays',
            fullmask_reference='Legal BG pixels reference real photos. UnknownBG reference values are diagnostic differences only, not RGB ground truth or supervision',
            replay='R9 nextstep is parameter-aware physical-scale equivalence, not bit-exact and not original raw-log tolerance; both physical replay updates remain in actual count',
            geometry='original seed global plane is a dimensional soft prior; no hard plane lock or permanent green carve',
            heldout='Eight non-training views are render-only; no heldout RGB ground-truth score is computed'),
        fixed_monitors=data['fixed'],cache_denominators=dict(source=data['bindings']['cache_manifest'],inventory=data['cache']['inventory'],frame_rows=data['cache']['frame_rows']),
        source_R8_control=dict(root=str(a.r8_root),fixed_step2600=control,geometry=geometry_summary(source),
            source_receipt=pin(a.r8_root/a.r8_run/'train/receipt.json'),curve_is_separate_from_R9=True),
        source_R7_control=dict(root=str(a.r7_root),source_metrics=pin(a.r7_root/'results/metrics.json')),
        common_R8_RGB_comparison=comparison,RGB_and_geometry_scope_invariants=invariants,accounting=accounting,five_parameter_group_updates_by_stage=groups,point_count_history=history,
        topology_events_R9_unique=events,fresh_process_replays=replay,transition=transition,geometry=geometry_summary(data),
        real_photo_background_provenance=dict(manifest=background_spec,manifest_hash_matches=background_hash_matches,
            source_photo_only=background['source_photo_only'],teacher_generated=background['teacher_generated'],
            independent_teacher_count=data['cfg']['independent_teacher_count'],structural_unknown_RGB_loss_weight=data['cfg']['structural_unknown_RGB_loss_weight'],
            background_seed=transition['background_seed'],source_training_frames=background['training_frames']),
        continuation_density_policy=runtime['densification'],
        frozen_fragments=fragment_summary(data),phase_evidence=data['phases'],
        source_native_freeze=dict(verified_by_native_load_receipts=bool(native_receipts)and all(q['original_native_parameters_frozen']for q in native_receipts),
            original_native_optimizer_created=any(q['optimizer_created']for q in native_receipts),
            native_source_parameter_changes_in_method=0,source='native load receipts plus per-step base.assert_fixed; checkpoint parameters are not optimizer groups'))
    out=a.output or a.root/'results/metrics.json';out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(out)
    latest=result['fixed_monitors'][-1]
    print(json.dumps(dict(output=str(out),complete=result['complete'],legal_RGB_domain_unchanged=comparison['legal_RGB_support_weights_and_targets_unchanged'],
        common_R8_RGB=[dict(step=x['unique_step'],road=x['R9_common_road_RGB_balanced_MSE'],legal_BG=x['R9_common_legal_BG_RGB_balanced_MSE'])for x in comparison['common_step_RGB']],
        real_photo_BG_curve=[dict(step=x['unique_step'],age=x['background_unique_age'],MSE=x['real_visible_background_RGB_balanced_MSE_only'])for x in data['fixed']],
        frozen_fragments=result['frozen_fragments']['frozen_children']if result['frozen_fragments']['enabled']else 0,
        final_four_class_RGB_balanced_MSE=latest['balanced_RGB_MSE_only'],final_road_RGB_balanced_MSE=latest['road_RGB_balanced_MSE_only'],
        final_A20_visibility=latest.get('A20_visibility'),geometry=result['geometry']['point_sources']if result['geometry']else None,
        accounting=accounting),ensure_ascii=False))

if __name__=='__main__':main()

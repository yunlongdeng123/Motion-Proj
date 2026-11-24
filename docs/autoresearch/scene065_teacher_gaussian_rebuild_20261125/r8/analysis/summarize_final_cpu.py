"""Standalone final CPU summary; never imports training code or initializes CUDA."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np

DEFAULT_ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_background_continuation_20261125_r8')
DEFAULT_R7=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_region_rebuild_20261125_r7')
DEFAULT_R5=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_gaussian_rebuild_20261125_r5')
DEFAULT_R6=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_visibility_rebuild_20261125_r6')
READOUT_STEPS=(2016,2100,2400,2600)
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
    for name in ['red','green','orange','protection','alpha_inside','alpha_outside','background_rebuild']:
        if name not in cache:continue
        mask=cache[name].astype(bool);den=int(mask.sum())
        result[name]=dict(fixed_pixels=den,local_effective_alpha_mean=float(local[mask].mean())if den else None,
            local_effective_alpha_gt_0_5_fraction=float((local[mask]>.5).mean())if den else None,
            local_effective_alpha_gt_0_9_fraction=float((local[mask]>.9).mean())if den else None,
            retained_A20_source21_effective_alpha_mean=float(actor[mask].mean())if den else None,
            frozen_context_reference_RGB_LDR_MSE=float(reference_error[mask].mean())if den else None)
    ref=np.squeeze(cache['A20_alpha_reference']);den=float(ref.sum(dtype=np.float64))
    result['A20_visibility']=dict(reference_alpha_weight_denominator=den,
        one_sided_decrease_MSE=float(np.sum(ref*np.maximum(ref-actor,0)**2,dtype=np.float64))/den if den else None,
        reference_effective_alpha_sum=den,actual_effective_alpha_sum=float(actor.sum(dtype=np.float64)),
        alpha_identity='actual retained checkpoint source21; no semantic/source label mixing')
    return dict(raw=raw,domains=result)

def compare_fixed_RGB(control,current):
    """Re-score actual R8 pixels on the original R7 fixed road domain."""
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
        for category in ROAD_CATEGORIES:
            sk=f'{category}__support';wk=f'{category}__weight';rk=f'{category}__rgb';sa=a[sk].astype(bool);sb=b[sk].astype(bool)
            checks.append(dict(raw=raw,category=category,support_equal=bool(np.array_equal(sa,sb)),weight_equal=bool(np.array_equal(a[wk],b[wk])),
                target_RGB_equal=bool(np.array_equal(a[rk],b[rk])),r7_control_pixels=int(sa.sum()),r8_pixels=int(sb.sum()),
                r7_control_weight_denominator=float(old[raw]['categories'][category]['final_weight_denominator']),
                r8_weight_denominator=float(new[raw]['categories'][category]['final_weight_denominator'])))
        for step in READOUT_STEPS:
            phase='probe'if step==2016 else'train'
            folder=current['run']/phase/f'actual_step{step:06d}';path=folder/f'{raw:03d}'/'joint.npz'
            if not path.is_file():missing.append(dict(step=step,raw=raw,path=str(path)));continue
            with np.load(path,allow_pickle=False)as z:joint={k:z[k]for k in ['rgb','local_effective_alpha','actor_alpha']}
            rgb=joint['rgb']
            legacy[step].append(rgb_readout_row(raw,rgb,a,old[raw]));own[step].append(rgb_readout_row(raw,rgb,b,new[raw]))
            mask_rows[step].append(readout_mask_metrics(raw,joint,b))
            sources[step].append(str(path))
    baseline=next(x for x in control['fixed']if x['unique_step']==2000);new_points={x['unique_step']:x for x in current['fixed']};points=[]
    for step in READOUT_STEPS:
        before=baseline['balanced_RGB_MSE_only'];q=summarize_rgb_rows(legacy[step],control['cfg']);n=summarize_rgb_rows(own[step],current['cfg'])
        after=q['balanced_RGB_MSE_only'];saved=new_points.get(step,{}).get('balanced_RGB_MSE_only')
        points.append(dict(unique_step=step,background_unique_age=step-2000,r7_control_step2000_road_RGB_balanced_MSE=before,r8_RGB_on_R7_road_support_balanced_MSE=after,
            relative_change=(after/before-1)if before and after is not None else None,
            r8_recomputed_four_classes_on_own_support=n,r8_saved_fixed_four_class_RGB_balanced_MSE=saved,
            CPU_vs_saved_own_RGB_abs_difference=abs(n['balanced_RGB_MSE_only']-saved)if saved is not None and n['balanced_RGB_MSE_only']is not None else None,
            r8_on_R7_fixed_road_domain=q,r8_fixed_mask_readout_metrics=mask_rows[step],source_actual_joint_NPZs=sources[step]))
    return dict(source_R7_control_step=2000,new_training_road_support_equals_R7=all(x['support_equal']and x['weight_equal']for x in checks),
        recomputed_R8_uses_exact_R7_road_domain=True,common_definition='Actual R8 joint RGB uses exact R7 road target RGB/support/weight/denominator, gain/bias, category weights and four sampler groups. Background is excluded from this common-road score.',
        per_frame_category_checks=checks,common_step_RGB=points,missing_actual_readouts=missing,
        real_photo_BG_fixed_denominators_and_reference_checks=photo_checks,
        all_four_common_readouts_complete=not missing,heldout_RGB_evaluated=False,
        mixed_total_comparable=False,coverage_targets_comparable=False,
        note='R7 2000 control and R8 four-class training curves stay separate. At 2000_after_append only fixed evaluation was saved, so no additional actual RGB render is inferred.')

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

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--r7-root',type=Path,default=DEFAULT_R7)
    p.add_argument('--run',default='training_v1');p.add_argument('--r7-run',default='training_v1');p.add_argument('--output',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    data=collect(a.root,a.run);final=data['phases']['train']['receipt']
    if not a.allow_partial and not(final and final.get('complete')and final.get('logical_step')==2600):p.error('Final complete cumulative2600/BG600-step receipt required; no polling is performed.')
    source=collect(a.r7_root,a.r7_run);comparison=compare_fixed_RGB(source,data)
    accounting,groups,history,events,replay,transition=continuation_execution_summary(source,data)
    runtime=read(data['bindings']['runtime_contract']['path']);background_spec=runtime['real_background_manifest'];background=read(background_spec['path'])
    background_hash_matches=pin(background_spec['path'])['sha256']==background_spec['sha256']
    assert background_hash_matches
    native_receipts=[q['native_parameter_freeze_receipt']for q in data['phases'].values()if q['native_parameter_freeze_receipt']]
    for phase in ['prepare','verify']:
        q=read(a.root/f'{phase}_runtime/native30k/native30k_load_receipt.json')
        if q:native_receipts.append(q)
    control=next(x for x in source['fixed']if x['unique_step']==2000)
    result=dict(schema='scene065.teacher_background_continuation.final_metrics.v1',updated_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,GPU_started=False,
        complete=bool(final and final.get('complete')and final.get('logical_step')==2600),bindings=data['bindings'],
        definitions=dict(RGB='strict fixed eight-frame category canonical-linear MSE, separately balanced across four sampler groups',
            mixed_total='fixed_objective_without_prior; includes protection/coverage/A20 terms, never labelled RGB MSE',
            road_cumulative='R7 2000 inherited optimizer updates plus R8 600; no road parameter or Adam reset at transition',
            background='Real-photo BG family is appended at road step2000 and trained for600 unique updates; later density births retain their own born_step',
            r8_inside='effective local alpha target=1-frozen retained foreground alpha on fixed road+authorized_BG inside; full complement outside target0',
            A20_visibility='actual checkpoint source21 effective-alpha preservation; identity fixed by current source mapping',
            denominators='R8 has four independently locked category supports; common-road comparison reuses exact R7 three-category masks, weights, denominators and canonical transforms',
            protection='Full green/orange RGB preservation; authorized damaged BG pixels in orange reference the legal real-photo target, other protection pixels retain context-only reference',
            geometry='original seed global plane is a dimensional soft prior; no hard plane lock or permanent green carve',
            heldout='Eight non-training views are render-only; no heldout RGB ground-truth score is computed'),
        fixed_monitors=data['fixed'],cache_denominators=dict(source=data['bindings']['cache_manifest'],inventory=data['cache']['inventory'],frame_rows=data['cache']['frame_rows']),
        source_R7_control=dict(root=str(a.r7_root),fixed_step2000=control,geometry=geometry_summary(source),
            source_metrics=pin(a.r7_root/'results/metrics.json'),curve_is_separate_from_R8=True),source_R7_fixed_monitors=source['fixed'],
        common_road_RGB_comparison=comparison,accounting=accounting,five_parameter_group_updates_by_stage=groups,point_count_history=history,
        topology_events_R8_unique=events,fresh_process_replays=replay,transition=transition,geometry=geometry_summary(data),
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
    print(json.dumps(dict(output=str(out),complete=result['complete'],new_training_road_support_equals_R7=comparison['new_training_road_support_equals_R7'],
        common_R7_road_RGB=[dict(step=x['unique_step'],R7_control=x['r7_control_step2000_road_RGB_balanced_MSE'],R8=x['r8_RGB_on_R7_road_support_balanced_MSE'])for x in comparison['common_step_RGB']],
        real_photo_BG_curve=[dict(step=x['unique_step'],age=x['background_unique_age'],MSE=x['real_visible_background_RGB_balanced_MSE_only'])for x in data['fixed']],
        frozen_fragments=result['frozen_fragments']['frozen_children']if result['frozen_fragments']['enabled']else 0,
        final_four_class_RGB_balanced_MSE=latest['balanced_RGB_MSE_only'],final_road_RGB_balanced_MSE=latest['road_RGB_balanced_MSE_only'],
        final_A20_visibility=latest.get('A20_visibility'),geometry=result['geometry']['point_sources']if result['geometry']else None,
        accounting=accounting),ensure_ascii=False))

if __name__=='__main__':main()

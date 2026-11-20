"""Frozen CPU endpoint evaluation; no Torch/native import, renderer or optimization.

Input is a JSON plan binding final actual native packets, fixed source masks and
the published metric implementation. Unavailable references remain explicit N/A.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import ndimage as ndi
from ideal_endpoint_metrics import metric_rgb,paint_metrics

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
B=Q/'gpu_20261119/native_capacity_20261120_r1';HIGH=Q/'gpu_20261119/quality_bench_20261120_r1/highres'
METRIC_SHA='86780162b5bb3ab87e455054c800fccdeaba12337557960870b1b4943e92a03c'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text('utf-8'))
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def load(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in(z.files if keys is None else keys)}

def metric_module(path,expected_sha=METRIC_SHA):
    if sha(path)!=expected_sha:raise ValueError('published metric source SHA differs; do not silently change evaluation')
    spec=importlib.util.spec_from_file_location('frozen_native_quality_bench',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def fixed_edge_scalars(report):
    edge=next(e for e in report['edges']['threshold_curves']if abs(e['strength_threshold']-2/255)<1e-12)
    curve=next(c for c in edge['curves']if c['tolerance_pixels']==1.)
    p,r=curve['precision'],curve['recall']
    f1=None if edge['reference_edges']==0 else (0. if edge['predicted_edges']==0 else (2*p*r/(p+r)if p+r else 0.))
    return dict(edge_precision=p,edge_recall=r,edge_F1=f1,edge_reference_n=edge['reference_edges'],edge_predicted_n=edge['predicted_edges'],
                edge_context_core_n=report['edges']['context_core_pixels'],edge_sigma_independent=True)

def validate_packet(packet,shape=(1280,1920)):
    for key in ['road_rgb','joint_rgb']:
        x=packet[key]
        if x.shape!=(3,*shape)or not np.issubdtype(x.dtype,np.floating)or not np.isfinite(x).all():raise ValueError('actual native CHW finite floating packet required: '+key)
    for key in ['road_alpha','joint_alpha']:
        x=packet[key]
        if x.shape!=shape or not np.issubdtype(x.dtype,np.floating)or not np.isfinite(x).all()or np.any((x<0)|(x>1+1e-5)):raise ValueError('actual native HW alpha required: '+key)
    if packet['K'].shape!=shape or packet['K'].dtype!=bool:raise ValueError('actual native bool K required')

def validate_endpoint(receipt,arm,protocol):
    if receipt.get('completed')is not True or receipt.get('arm')!=arm:raise ValueError('final endpoint receipt incomplete or wrong arm')
    cfg=protocol['arms'][arm];extra=receipt.get('extra')or{}
    if cfg['type']=='shared_field':
        if extra.get('logical_steps')!=cfg['steps']:raise ValueError('shared primary evaluation must use the registered final step')
    else:
        count=extra.get('accepted_steps');cap=cfg['max_accepted_steps']
        if arm=='G_ORACLE'and extra.get('continued'):cap+=2000
        if not isinstance(count,int)or count<0 or count>cap:raise ValueError('oracle endpoint exceeds its registered accepted-step cap')
    return dict(status='bound_completed_registered_endpoint',steps=extra.get('logical_steps',extra.get('accepted_steps')),
                measured_training_convergence=extra.get('measured_training_convergence'),strict_numeric_certificate=False)

def signed_sobel_error(pred,teacher,context,score):
    core=(ndi.minimum_filter(context.astype(np.uint8),size=3,mode='constant',cval=0)>0)&score
    n=int(core.sum())
    if not n:return dict(pixels=0,MAE=None,RMSE=None,status='N/A_no_strict_core',radius=1)
    e=np.where(context[None],np.asarray(pred,np.float64)-np.asarray(teacher,np.float64),0.)
    gx=np.stack([ndi.sobel(c,axis=1,mode='constant')/8 for c in e]);gy=np.stack([ndi.sobel(c,axis=0,mode='constant')/8 for c in e])
    v=np.stack([gx[:,core],gy[:,core]])
    return dict(pixels=n,MAE=float(np.abs(v).mean()),RMSE=float(np.sqrt((v*v).mean())),status='defined',radius=1,
                formula='fixed signed Sobel x/y on masked straight-RGB residual; no gradient-magnitude substitution')

def visibility_counts(domain,alpha,visibility):
    if visibility is None:return dict(status='N/A_no_bound_same_sort_visibility',causal_static_removal_claim=False)
    road=visibility.get('joint_road_alphaT',visibility.get('entire_road_weight'))
    static=visibility.get('static_alphaT')
    if road is None or static is None:return dict(status='N/A_missing_joint_road_or_static_alphaT',causal_static_removal_claim=False)
    if road.shape!=domain.shape or static.shape!=domain.shape:raise ValueError('visibility must match native camera exactly')
    high=domain&(alpha>=.95);hidden=high&(road<.05)
    return dict(status='defined_cached_frozen_same_sort',high_road_alpha_but_joint_road_below05=int(hidden.sum()),
        of_these_static_alphaT_ge50=int((hidden&(static>=.5)).sum()),high_road_alpha_pixels=int(high.sum()),
        joint_road_weight_mean=float(road[domain].mean())if domain.any()else None,
        static_weight_mean=float(static[domain].mean())if domain.any()else None,
        causal_static_removal_claim=False,interpretation='actual fixed-depth-order contributions; not a counterfactual static deletion')

def full_source_summary(packet,teacher,domain,canonical_hit,source_valid,visibility):
    a=packet['road_alpha']
    return dict(expected_pre_alpha=int(domain.sum()),canonical_hit=int((domain&canonical_hit).sum()),unmapped=int((domain&~canonical_hit).sum()),
        source_material_valid=int((domain&source_valid).sum()),coverage=[dict(alpha_threshold=t,pixels=int((domain&(a>=t)).sum()))for t in [.1,.5,.9,.95]],
        gap_below_01=int((domain&(a<.1)).sum()),thin_01_to_09=int((domain&(a>=.1)&(a<=.9)).sum()),
        opaque_road_P_vs_teacher=metric_rgb(packet['road_rgb'],teacher,domain)if teacher is not None else None,
        legacy_premult_P_vs_alphaT=metric_rgb(packet['road_rgb'],a[None]*teacher,domain)if teacher is not None else None,
        joint_J_vs_teacher=metric_rgb(packet['joint_rgb'],teacher,domain)if teacher is not None else None,
        visibility=visibility_counts(domain,a,visibility),
        denominator_policy='complete legal source domain; no alpha/mesh/source trimming. A chart-exterior count is descriptive, not automatic asset-failure.',
        caution='C/S error is pseudo-teacher consistency; zero-alpha premult residual is not evidence of recovered material.')

def evaluate_region(metric,packet,teacher,split,role,partition,source_receipt):
    context=split[f'{role}_{partition}_context'];score=split[f'{role}_{partition}'];a=split['fixed_alpha']
    straight=np.divide(packet['road_rgb'],a[None],out=np.zeros_like(packet['road_rgb']),where=a[None]>=.95)
    report=metric.evaluate_detail(straight.transpose(1,2,0),teacher.transpose(1,2,0),context_mask=context,score_mask=score,
        resolution=metric.ResolutionSpec(a.shape[1],a.shape[0],'native1920_current_pixels','srgb'),tile_layout=metric.TileLayout(128,128),
        teacher_role='real_observation'if role=='A'else'frozen_pseudo_teacher',color_space='stable_alpha_unpremult_srgb',
        input_conditioning=dict(alpha_min=.95,alpha_source='frozen common source split',partition=partition,
            held_label='prior-conditioned new-label exclusion; not blind',teacher_role=role))
    report['RGB_stable']=metric_rgb(straight,teacher,context&score)
    report['signed_Sobel_error']=signed_sobel_error(straight,teacher,context,score)
    report['prior_exposure']=dict(F0_center_source=source_receipt['F0'][partition],optimized_A200_L_preexposed=True,
        full_source_history_independence_not_proven=True,blind_test=False)
    report['world_blocks']=source_receipt[partition+'_blocks']
    report['interpretation_status']='N/A_domain_generalization_insufficient_world_blocks' if partition=='held'and report['world_blocks']<3 else'conditional_descriptive_not_true_blind'
    report['semantic_lane_geometry_truth_claim']=False
    return report

def ideal_report(packet,root,raw,highres,config_path,basis_path):
    saved=load(Path(root)/f'{raw:03d}.npz',['pixel_ids','ideal_available','road_fixed_alpha','road_eligible_weight','road_K','ideal_rgb','paint_fraction','road_target_P',
        'marking_w120_resolvable_sparse','marking_w120_line_corridor_sparse'])
    t=load(Path(highres)/f'targets/{raw:03d}.npz',['A_pre_alpha','C_pre_alpha','S_pre_alpha'])
    legal=t['A_pre_alpha']|t['C_pre_alpha']|t['S_pre_alpha'];a=saved['road_fixed_alpha'];we=saved['road_eligible_weight'];ids=saved['pixel_ids']
    qual=np.zeros(legal.size,bool);qual[ids]=saved['marking_w120_resolvable_sparse']&saved['marking_w120_line_corridor_sparse'];qual=qual.reshape(legal.shape)
    context=legal&saved['ideal_available']&(a>=.95)&(we>=.95)&~saved['road_K']
    mask=context&qual
    cfg=read(config_path);case=next(c for c in cfg['cases']if c['name']=='marking_w120')
    # Independent known world dash IDs; geometry cache only, never real teacher paint inference.
    geometry=load(Path(highres)/f'maps/{raw:03d}.npz',['surface_world_xyz'])
    helper=Path(config_path).parent.parent/'ideal_code/ideal_world_teacher.py'
    spec=importlib.util.spec_from_file_location('ideal_world_function_for_evaluation',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    u,_,_,_=m.curve_coordinates(geometry['surface_world_xyz'],cfg);dash=np.full(legal.size,np.iinfo(np.int64).min,np.int64)
    dash[ids]=np.rint((u-case['marking']['phase_m'])/case['marking']['period_m']).astype(np.int64);dash=dash.reshape(legal.shape)
    readonly=load(basis_path,['readonly_rgb'])['readonly_rgb']
    result=paint_metrics(packet['road_rgb'],readonly=readonly,weight=we,asphalt=saved['ideal_rgb'][0],paint_rgb=cfg['paint_srgb'],
        paint_fraction=saved['paint_fraction'][1],qualified_mask=mask,dash_ids=dash)
    result.update(full_source_pixels=int(legal.sum()),known_ideal_available=int((legal&saved['ideal_available']).sum()),
        We_zero_fullsource=int((legal&(we==0)).sum()),qualified_context_pixels=int(context.sum()),
        conditional_P_target_RGB=metric_rgb(packet['road_rgb'],saved['road_target_P'][1],context),
        target='P*=native readonly + eligibleWe * independent analytic ideal pixelAA; already premult, never alpha-multiplied twice',
        sampler='density8 same pixelbox, numeric qualification against density16 retained in source preparation',
        role='open known world pattern capacity control, no held generalization or real-method validation ranking',case='marking_w120')
    return result

def evaluate(plan,output):
    output=Path(output)
    if not plan.get('packets'):raise ValueError('nonempty bound endpoint packet list is required; an empty run is not complete')
    if output.exists():raise FileExistsError('preserve previous evaluation; fresh output directory required')
    protocol_path=Path(plan['protocol']);protocol=read(protocol_path)
    if sha(protocol_path)!=plan['protocol_sha256']:raise ValueError('frozen protocol differs')
    metric=metric_module(plan['metric'],plan.get('metric_sha256',METRIC_SHA));output.mkdir(parents=True)
    split_root=Path(plan['split_root']);highres=Path(plan['highres_root']);split_receipt=read(split_root/'receipt.json')
    source_rows={r['raw']:r for r in split_receipt['frames']};records=[];flat_rows=[];coverage_rows=[];K_rows=[];ideal_rows=[]
    sources={str(protocol_path):sha(protocol_path),plan['metric']:sha(plan['metric'])};began=time.perf_counter()
    for item in plan['packets']:
        path=Path(item['path']);arm=item['arm'];raw=int(item['raw']);packet=load(path);validate_packet(packet)
        sources[str(path)]=sha(path);row=dict(arm=arm,raw=raw,packet=str(path),packet_sha256=sources[str(path)],native_resolution=[1920,1280],sources=[])
        if item.get('endpoint_receipt'):
            ep=Path(item['endpoint_receipt']);row['endpoint_provenance']=validate_endpoint(read(ep),arm,protocol);sources[str(ep)]=sha(ep)
        else:row['endpoint_provenance']=dict(status='N/A_no_bound_endpoint_receipt',not_a_checkpoint_selection_rule=True)
        baseline=load(item['baseline'])if item.get('baseline')else None
        if baseline is not None:
            alpha0=baseline.get('road_alpha',baseline.get('alpha')if not item.get('baseline_is_joint_basis')else None)
            joint0=baseline.get('joint_rgb',baseline.get('baseline_rgb')if item.get('baseline_is_joint_basis')else None)
            joint_alpha0=baseline.get('joint_alpha',baseline.get('alpha')if item.get('baseline_is_joint_basis')else None)
            row['frozen_guard']=dict(road_alpha_exact=bool(np.array_equal(alpha0,packet['road_alpha']))if alpha0 is not None else None,
                joint_alpha_exact=bool(np.array_equal(joint_alpha0,packet['joint_alpha']))if joint_alpha0 is not None else None,
                K_pixels=int(packet['K'].sum()),joint_K_max=float(np.abs(packet['joint_rgb']-joint0)[:,packet['K']].max())if joint0 is not None and packet['K'].any()else None,
                baseline=str(item['baseline']),claim='actual packet comparison, not a screen-pixel copy')
            sources[str(item['baseline'])]=sha(item['baseline'])
        else:row['frozen_guard']=dict(status='N/A_no_baseline_packet',K_pixels=int(packet['K'].sum()))
        if raw in [86,110]:
            sp=split_root/f'{raw:03d}.npz';tp=highres/f'targets/{raw:03d}.npz';split=load(sp)
            if not np.array_equal(packet['road_alpha'],split['fixed_alpha']):raise ValueError('endpoint alpha differs from common metric support: '+str(path))
            if not np.array_equal(packet['K'],split['K']):raise ValueError('endpoint K/camera differs from source masks: '+str(path))
            row['frozen_guard']['common_source_road_alpha_exact']=True
            visibility=None
            if item.get('visibility_sidecar'):
                if item.get('frozen_visibility_reuse_confirmed')is not True:raise ValueError('cached visibility needs bound unchanged geometry/camera/order declaration')
                visibility=load(item['visibility_sidecar']);sources[item['visibility_sidecar']]=sha(item['visibility_sidecar'])
            elif any(k in packet for k in ['joint_road_alphaT','static_alphaT']):visibility=packet
            coverage=[]
            with np.load(tp,allow_pickle=False)as targets:
                for role in ['A','C','S']:
                    teacher=targets['T_'+role]if arm!=plan.get('ideal_arm','IDEAL_ORACLE')else None
                    coverage.append(dict(role=role,teacher_role=('visible_real_appearance'if role=='A'else'frozen_pseudo_teacher')if teacher is not None else'synthetic_control_not_ranked_against_real_targets',
                        **full_source_summary(packet,teacher,split[role+'_pre_alpha'],split['canonical_hit'],split['source_material_valid'],visibility)))
                    if arm!=plan.get('ideal_arm','IDEAL_ORACLE'):
                        for partition in ['seen','held']:
                            report=evaluate_region(metric,packet,teacher,split,role,partition,source_rows[raw]['domains'][role])
                            rel=f'{arm}/{raw:03d}_{role}_{partition}.json';dest=output/rel;dest.parent.mkdir(exist_ok=True)
                            write(dest,report)
                            row['sources'].append(dict(role=role,partition=partition,path=rel,RGB_stable=report['RGB_stable'],
                                signed_Sobel=report['signed_Sobel_error'],world_blocks=report['world_blocks'],interpretation=report['interpretation_status'],
                                HF=[dict(sigma=s['sigma_pixels'],core=s['core'],boundary_pixels=s['boundary_pixels'],flat=s['teacher_flat_spurious'],
                                    tile_macro=s['tile_aggregate'])for s in report['scales']],edges=fixed_edge_scalars(report)))
                            for s in report['scales']:
                                c=s['core'];macro=s['tile_aggregate']['macro']['HF_MAE']
                                flat=next(f for f in s['teacher_flat_spurious']if abs(f['teacher_HF_RMS_cut']-2/255)<1e-12)
                                flat_rows.append(dict(arm=arm,raw=raw,source=role,split=partition,scale=s['sigma_pixels'],
                                    teacher_role=report['teacher_role'],signal_space='stable_alpha_unpremult_srgb',
                                    score_n=report['fixed_score_pixels'],context_n=report['context_pixels'],stable_score_n=report['context_scored_pixels'],
                                    core_n=c['pixels'],boundary_n=s['boundary_pixels'],RGB_MAE=report['RGB_stable']['MAE'],RGB_RMSE=report['RGB_stable']['RMSE'],
                                    HF_cosine=c['HF_cosine'],HF_energy_ratio=c['HF_energy_ratio'],HF_relative_RMSE=c['HF_relative_RMSE'],
                                    HF_MAE=c['HF_MAE'],HF_RMSE=c['HF_RMSE'],HF_excess=c['excess_HF_energy'],
                                    flat_pred_RMS=flat['predicted_HF_RMS'],flat_ref_RMS=flat['teacher_HF_RMS'],flat_n=flat['pixels'],
                                    Sobel_MAE=report['signed_Sobel_error']['MAE'],Sobel_RMSE=report['signed_Sobel_error']['RMSE'],Sobel_core_n=report['signed_Sobel_error']['pixels'],
                                    HF_tile_macro_MAE=macro['mean'],HF_tile_p95_MAE=macro['p95'],nonempty_tiles=s['tile_aggregate']['nonempty_core_tiles'],
                                    world_blocks=report['world_blocks'],interpretation=report['interpretation_status'],prior_preexposed=True,blind_test=False,
                                    detail_file=rel,**fixed_edge_scalars(report)))
            row['full_source_coverage_and_joint']=coverage
            coverage_rows.extend(dict(arm=arm,raw=raw,source=c['role'],**{k:v for k,v in c.items()if k!='role'})for c in coverage)
            sources[str(sp)]=sha(sp);sources[str(tp)]=sha(tp)
            if arm==plan.get('ideal_arm','IDEAL_ORACLE'):
                basis=item.get('ideal_road_basis',str(highres/f'native_r1/{raw:03d}/road_basis.npz'))
                cfg=plan['ideal_config']
                row['ideal']=ideal_report(packet,plan['ideal_root'],raw,highres,cfg,basis)
                ideal_rows.append(dict(arm=arm,raw=raw,**row['ideal']))
                for p in [Path(plan['ideal_root'])/f'{raw:03d}.npz',Path(cfg),Path(basis)]:sources[str(p)]=sha(p)
        else:row['diagnostic_reference']=dict(status='N/A_no_hidden_road_target',teacher_metrics_computed=False,blind_test=False)
        K_rows.append(dict(arm=arm,raw=raw,**row['frozen_guard']))
        records.append(row);print(json.dumps(dict(arm=arm,raw=raw,completed=True,elapsed_seconds=time.perf_counter()-began)),flush=True)
    expected_arms=plan.get('expected_arms',['G_RGB','G_DETAIL','G_ORACLE','IDEAL_ORACLE']);expected_frames=plan.get('expected_frames',[86,110,134,179,198])
    actual={(r['arm'],r['raw'])for r in records};missing=[dict(arm=a,raw=r)for a in expected_arms for r in expected_frames if(a,r)not in actual]
    result=dict(schema='native_GS_capacity_fixed_endpoint_evaluation.v1',complete=True,complete_bound_plan_only=True,final_endpoint_set_complete=not missing,missing_registered_packets=missing,
        rows=flat_rows,packets=records,coverage_rows=coverage_rows,K_rows=K_rows,ideal_rows=ideal_rows,source_SHA256=sources,
        seconds=time.perf_counter()-began,code_SHA256=sha(__file__),frozen_protocol_sha256=plan['protocol_sha256'],
        endpoint_selection='registered final checkpoints only; evaluation never selects training steps or hyperparameters',
        prior_exposure='A0/A200/routing already exposed; held is no-new-label-gradient diagnostic, not blind or effective new-texture generalization',
        oracle='independent box-ci is same ordinaryGS color relaxation for the registered seen data objective; numeric FW evidence is not a universal perceptual certificate',
        marking='real complete-lane geometry GT unavailable; synthetic paint is known only for IDEAL_ORACLE. Source-verified realA paint ROI remains local appearance evidence.',
        no_overall_quality_score=True,human_verdict=None,production_promoted=False,GPU_executed=False,native_calls=0,optimizer_updates=0)
    write(output/'results.json',result)
    if flat_rows:
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig')as f:
            writer=csv.DictWriter(f,fieldnames=list(flat_rows[0]));writer.writeheader();writer.writerows(flat_rows)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=evaluate(read(a.plan),a.output);print(json.dumps(dict(complete=True,packets=len(result['packets']),output=str(a.output),seconds=result['seconds'])))

if __name__=='__main__':main()

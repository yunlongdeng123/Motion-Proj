"""门2已完成数据的独立CPU重算；不启动native或优化，不改case。"""
from pathlib import Path
import argparse,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np

FRAMES=(86,110,134,179,198)
CASES=('constant','lowfreq','marking_w120','marking_w240','stress_highfreq')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if keys is None or k in keys}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def weighted_f(pred,target,q):
    q=np.asarray(q,np.float64)
    if q.ndim!=2 or q.sum()<=0:raise ValueError('invalid fixed q')
    residual=np.asarray(pred,np.float64)-np.asarray(target,np.float64)
    return float(.5*(np.square(residual)*q[None]).sum()/(3*q.sum()))


def range_floor(target,lower,upper,q):
    delta=np.maximum(np.maximum(np.asarray(lower,np.float64)-target,target-np.asarray(upper,np.float64)),0)
    return float(.5*(np.square(delta)*q[None]).sum()/(3*np.asarray(q,np.float64).sum()))


def fw_numeric(x,gradient,f,safety=1e-5):
    x=np.asarray(x,np.float64);gradient=np.asarray(gradient,np.float64)
    gap=max(0.,float((gradient*x-np.minimum(gradient,0)).sum()))
    return dict(FW_gap_numeric_proxy=gap,lower_numeric_proxy=max(0.,float(f)-gap),L_safe_numeric_proxy=max(0.,float(f)-gap-safety),strict_numeric_bound_established=False,negative_certificate_allowed=False)


def level_status(rmse,level,detail_pass,stress=False):
    rgb=bool(rmse<=level)
    return dict(RGB_pass=rgb,detail_pass=bool(detail_pass),joint_feasible_witness=bool(rgb and detail_pass and not stress),
        status='diagnostic_stress_only' if stress else 'feasible_solution_witness' if rgb and detail_pass else 'optimization_or_numeric_certificate_inconclusive',
        negative_capacity_claim=False)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('independent gate2 audit requires CPU only')
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-campaign',type=Path,required=True);parser.add_argument('--cpu-campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();g,c=args.gpu_campaign,args.cpu_campaign;root=c/'ideal_targets_r2';out=args.output
    if out.exists():raise FileExistsError('preserve previous independent report: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(1)
    def checkpoint(p):return torch.load(p,map_location='cpu',weights_only=False)
    protocol=read(c/'protocol_cpu_prepared.json')['check2'];checks={};case_rows=[];reload_count=0;scope_iterations={};max_metric_error=0.;max_floor_error=0.
    geometry={raw:load(root/'geometry'/f'{raw:03d}.npz') for raw in FRAMES};scopes=load(root/'color_scopes.npz')
    masks={raw:geometry[raw]['fixed_evaluation_domain']&geometry[raw]['canonical_mesh_hit_valid']&(geometry[raw]['fixed_alpha']>.9) for raw in FRAMES}
    q={raw:masks[raw].astype(np.float64) for raw in FRAMES};denominator=3*sum(float(v.sum()) for v in q.values())
    full=sum(int(z['fixed_evaluation_domain'].sum()) for z in geometry.values());exact_count=sum(int((z['fixed_evaluation_domain']&z['canonical_mesh_hit_valid']).sum()) for z in geometry.values())
    checks['fixed_37206_111618_38416_38198_218']=sum(int(m.sum()) for m in masks.values())==37206 and denominator==111618 and full==38416 and exact_count==38198 and full-exact_count==218
    levels=protocol['error_envelope']['registered_RMSE_levels'];checks['three_registered_levels_unchanged']=np.allclose(levels,np.array([1,2,4])/255,rtol=0,atol=1e-16) and np.allclose(protocol['error_envelope']['objective_tolerances'],.5*np.square(levels),rtol=0,atol=1e-16)
    endpoint=g/'check2/endpoint_replay_r1';epreceipt=read(endpoint/'receipt.json')
    checks['endpoint_replay_only20_forwards_zero_updates']=epreceipt['native_forward_calls']==20 and epreceipt['optimizer_updates']==0 and epreceipt['solver_iterations']==0 and not epreceipt['original_results_modified'] and epreceipt['source_binding_verified']
    endpoints={scope:{raw:load(endpoint/scope/f'{raw:03d}.npz') for raw in FRAMES} for scope in ['primary','relaxed']}
    baseasset=c.parent/'road_asset_20261116_r1/phase2/formal_r1/asset';base_road=load(baseasset/'road.npz')
    for scope in ['primary','relaxed']:
        folder=g/'check2'/f'native_{scope}_r1';done=read(folder/'completed.json');binding=read(folder/'binding.json');pre=read(folder/'operator_preflight.json')
        checks[scope+'_five_fixed_cases_completed']=[x['case'] for x in done['cases']]==list(CASES) and done['completed'] and not done['extra_case_automatically_started']
        checks[scope+'_preflight_spotcheck_only_no_strict_certificate']=pre['passed'] and not pre['strict_error_bound_established'] and not pre['negative_certificate_allowed'] and not done['negative_certificate_allowed']
        expected_world=scopes['primary_world'] if scope=='primary' else scopes['conditional_relaxed_world'];active=scopes['active_world_rows'];expected_active=expected_world[active]
        checks[scope+'_endpoint_helper_and_runner_bound']=epreceipt['source_operator_helper_sha256']==binding['helper_sha256'] and epreceipt['original_runner_source_sha256']==binding['runner_sha256']
        for raw in FRAMES:
            e=endpoints[scope][raw]
            checks[f'{scope}_{raw}_endpoint_fixed_Dq_alpha_scope']=exact(e['D_color'],masks[raw]) and np.array_equal(e['q'],q[raw]) and exact(e['fixed_alpha'],geometry[raw]['fixed_alpha']) and exact(e['eligible_world'],expected_world) and exact(e['active_world_rows'],active)
            checks[f'{scope}_{raw}_pointwise_box_order']=np.isfinite(e['fixed_b']).all() and np.isfinite(e['box_upper']).all() and float((e['fixed_b']-e['box_upper']).max())<=2e-6
        iterations=0
        for case in CASES:
            casepath=folder/case;record=read(casepath/'evaluation/evaluation.json');completion=read(casepath/'completed.json');trace=read(casepath/'trace.json');saved=checkpoint(casepath/'checkpoint_final.pt');initial=checkpoint(casepath/'checkpoint000000.pt')
            state=saved['state'];x=state['x'].numpy();gradient=state['g'].numpy();iteration=int(state['iteration']);iterations+=iteration
            checks[f'{scope}_{case}_caps_and_actual500']=iteration==completion['accepted_iterations']==500 and len(trace)==completion['solver_attempts']==500 and all(z['accepted'] for z in trace) and [z['accepted_iteration'] for z in trace]==list(range(1,501)) and completion['solver_elapsed_seconds']<=300 and completion['time_overrun_seconds']==0 and completion['work_cap']==protocol['work_cap']
            checks[f'{scope}_{case}_same_start_fixed_rows_box']=saved['case']==case and saved['scope']==scope and exact(saved['world_rows'],active) and exact(saved['eligible'],expected_active) and initial['state']['iteration']==0 and np.array_equal(initial['state']['x'].numpy(),base_road['reference_srgb'][active[expected_active]]) and np.isfinite(x).all() and bool(((x>=0)&(x<=1)).all())
            checks[f'{scope}_{case}_no_negative_or_promotion']=not record['negative_capacity_claim'] and not record['certificate']['strict_numeric_bound_established'] and not record['certificate']['negative_certificate_allowed'] and record['certificate']['certified_lower_bound'] is None and not completion['negative_certificate_allowed'] and not completion['numeric_bound_established'] and not completion['production_promoted']
            sse=0.;floor_sse=0.;frame_rows=[];pred_parts=[];gain_num=gain_den=0.;full_floor_sse=0.
            for raw in FRAMES:
                exported=load(casepath/'evaluation'/f'{raw:03d}.npz');target=load(root/case/f'{raw:03d}.npz');geo=geometry[raw];mask=masks[raw];e=endpoints[scope][raw]
                T=target['target_rgb_srgb'];a=geo['fixed_alpha'];P=exported['road_only_rgb'];soft=(T*a[None]).astype(np.float32)
                checks[f'{scope}_{case}_{raw}_independent_GT_Dcolor_reference_exact']=exact(exported['target_rgb_srgb'],T) and exact(exported['D_color'],mask) and exact(exported['fixed_alpha'],a) and bool((target['q'][mask]==1).all())
                residual=P.astype(np.float64)-soft.astype(np.float64);sse+=float(np.square(residual)[:,mask].sum());pred_parts.append(P[:,mask].T)
                floor=range_floor(soft,e['fixed_b'],e['box_upper'],q[raw]);floor_sse+=2*floor*3*float(q[raw].sum())
                D=geo['fixed_evaluation_domain'];opaque_violation=np.maximum(T.astype(np.float64)-a[None].astype(np.float64),0)
                opaque_floor=float(.5*np.square(opaque_violation)[:,D].mean());full_floor_sse+=float(np.square(opaque_violation)[:,D].sum())
                reported=next(v for v in record['frames'] if v['raw']==raw)
                max_metric_error=max(max_metric_error,abs(opaque_floor-reported['opaque_coverage_range_floor']))
                rmse=float(np.sqrt(np.square(residual)[:,mask].mean()));mae=float(np.abs(residual)[:,mask].mean())
                max_metric_error=max(max_metric_error,abs(rmse-reported['soft_metrics']['D_color']['RMSE']),abs(mae-reported['soft_metrics']['D_color']['MAE']))
                frame_rows.append(dict(raw=raw,D_full=int(D.sum()),D_exact=int((D&geo['canonical_mesh_hit_valid']).sum()),D_color=int(mask.sum()),q_sum=float(q[raw].sum()),
                    RGB_RMSE=rmse,RGB_MAE=mae,guarded_box_floor_numeric_proxy=floor,opaque_coverage_floor_Dfull=opaque_floor,
                    gap=int((D&(a<.05)).sum()),thin=int((D&(a>=.05)&(a<=.9)).sum()),support=int((D&(a>.9)).sum())))
                if case=='lowfreq':
                    true=T[:,mask].T.astype(np.float64);alpha=a[mask].astype(np.float64)[:,None];pred=P[:,mask].T.astype(np.float64);mu=true.mean(0);signal=alpha*(true-mu)
                    gain_num+=float((signal*(pred-alpha*mu)).sum());gain_den+=float(np.square(signal).sum())
            f=.5*sse/denominator;rmse=float(np.sqrt(sse/denominator));floor=.5*floor_sse/denominator
            max_metric_error=max(max_metric_error,abs(f-record['certificate']['f']),abs(rmse-record['certificate']['RGB_RMSE']))
            max_floor_error=max(max_floor_error,abs(floor-record['certificate']['guarded_box_range_floor_numeric_proxy']))
            checks[f'{scope}_{case}_final_prediction_checkpoint_consistent']=float(np.abs(np.concatenate(pred_parts)-state['p'].numpy()).max())<=1e-6 and abs(f-float(state['f']))<1e-10
            proxies=fw_numeric(x,gradient,f)
            checks[f'{scope}_{case}_FW_and_safe_proxy_recomputed']=all(abs(proxies[key]-record['certificate'][key])<1e-10 for key in ['FW_gap_numeric_proxy','lower_numeric_proxy','L_safe_numeric_proxy'])
            detail=record['detail'];detail_rows=[]
            if case=='constant':detail_pass=True;detail_kind='not_applicable_constant'
            elif case=='lowfreq':
                gain=gain_num/gain_den;detail_pass=gain>=protocol['error_envelope']['detail_contrast_retention'];detail_kind='independently_recomputed_modulation_gain'
                checks[scope+'_lowfreq_modulation_independently_matches']=abs(gain-detail['modulation_gain'])<1e-12 and detail_pass==detail['passed']
            else:
                for row in detail['frames']:
                    passed=(row['edge_F1']>=protocol['error_envelope']['resolvable_line_edge_F1'] and row['line_center_error_max_px'] is not None and row['line_center_error_max_px']<=protocol['error_envelope']['line_center_error_px'] and row['missing_predicted_fragments']==0 and row['contrast_retention'] is not None and row['contrast_retention']>=protocol['error_envelope']['detail_contrast_retention'])
                    detail_rows.append(dict(raw=row['raw'],independently_rethresholded_pass=passed,registered_metrics=row))
                detail_pass=all(z['independently_rethresholded_pass'] for z in detail_rows);detail_kind='original_registered_detail_statistics_thresholds_independently_checked'
                if case!='stress_highfreq':checks[f'{scope}_{case}_detail_registered_threshold_logic']=detail_pass==detail['passed'] and [z['raw'] for z in detail_rows]==list(FRAMES)
            statuses={f'{n}_over_255':dict(RGB_RMSE_level=float(level),objective_tolerance=.5*float(level)**2,**level_status(rmse,float(level),detail_pass,case=='stress_highfreq')) for n,level in zip([1,2,4],levels)}
            reload=read(casepath/'public_reload.json');reload_count+=len(reload['rows'])
            checks[f'{scope}_{case}_five_public_native_reload']=reload['public_loader'] and reload['GPU_executed'] and [r['raw'] for r in reload['rows']]==list(FRAMES) and all(max(r['RGB_max'],r['alpha_max'])<=1e-6 for r in reload['rows'])
            if scope=='primary':
                K=read(casepath/'joint_K.json');checks[f'{scope}_{case}_K77266_primary_only']=K['all_exact'] and K['protected_pixel_frames']==77266 and all(row['RGB_max']==0 and row['changed_pixels']==0 for row in K['rows']) and completion['K_exact'] is True
            else:checks[f'{scope}_{case}_relaxed_has_no_K_claim']=completion['K_exact'] is None
            asset=load(casepath/'asset/road.npz');checks[f'{scope}_{case}_fixed_geometry_opacity_scale_topology']=asset.keys()==base_road.keys() and all(exact(value,base_road[key]) for key,value in asset.items() if key!='reference_srgb')
            checks[f'{scope}_{case}_only_scope_RGB_changes']=exact(asset['reference_srgb'][~expected_world],base_road['reference_srgb'][~expected_world]) and exact(asset['reference_srgb'][active[expected_active]],x)
            case_rows.append(dict(scope=scope,case=case,eligible_rows=int(expected_active.sum()),q_sum=37206,RGB_denominator=int(denominator),f=f,RGB_RMSE=rmse,
                FW_numeric=proxies,guarded_box_floor_numeric_proxy=floor,original_logged_guarded_floor=record['certificate']['guarded_box_range_floor_numeric_proxy'],guarded_floor_independently_recomputed=True,
                opaque_coverage_floor_full38416=.5*full_floor_sse/(3*full),detail_kind=detail_kind,detail_pass=bool(detail_pass),detail_rows=detail_rows,
                lowfreq_modulation_gain=gain_num/gain_den if case=='lowfreq' else None,registered_error_levels=statuses,frames=frame_rows,
                iterations=iteration,solver_seconds=completion['solver_elapsed_seconds'],public_reload=reload,
                K_condition='native joint77266 exact0' if scope=='primary' else 'not applicable: relaxed virtual road-only oracle, no K or production claim',strict_negative_certificate_allowed=False))
            del saved,initial,asset
        scope_iterations[scope]=iterations
    checks['2500_primary_plus2500_conditional_LS']=scope_iterations==dict(primary=2500,relaxed=2500)
    checks['all_50_formal_case_views_reloaded']=reload_count==50
    checks['independent_RGB_F_opaque_metrics_match']=max_metric_error<1e-10
    checks['independent_endpoint_range_floors_match_original']=max_floor_error<1e-12
    smoke=read(g/'check2/smoke_r1/lifecycle/completed.json');actual_smoke=sum(int(a['accepted']) for a in smoke['actions'])
    checks['3_engineering_LS_distinct_from_formal']=actual_smoke==smoke['extra_solver_updates']==3 and smoke['formal_solver_updates']==0 and smoke['restore_tensor_state_exact'] and smoke['replay_pass']
    prior=read(g/'qa/real_pair_independent_acceptance.json');checks['400plus8_GD_prior_budget_retained']=prior['summary']['formal_updates']==400 and prior['summary']['engineering_updates']==8
    checks={key:bool(value) for key,value in checks.items()}
    witnesses={level:[dict(scope=r['scope'],case=r['case']) for r in case_rows if r['registered_error_levels'][level]['joint_feasible_witness']] for level in ['1_over_255','2_over_255','4_over_255']}
    result=dict(schema='ideal_capacity_independent_CPU_acceptance.v1',completed=True,CPU_only=True,new_GPU_operations=0,new_solver_or_optimizer_updates=0,case_data_modified=False,
        status='INDEPENDENT_NUMERICS_VERIFIED' if all(checks.values()) else 'AUDIT_CHECK_FAILURE',checks=checks,check_count=len(checks),failed_checks=[k for k,v in checks.items() if not v],
        fixed_domain=dict(D_full=38416,D_exact=38198,fallback=218,D_color=37206,q_sum=37206,RGB_denominator=111618,q='binary pre-frozen D_color, identical in f/gradient/FW/floor'),
        cases=case_rows,witnesses_by_registered_level=witnesses,all_negative_capacity_claims_false=True,strict_negative_certificate_allowed=False,stress_is_not_failure=True,
        scope_conditions=dict(primary='guarded existing112058-row color oracle with exact joint K',relaxed='conditional existing151096-active-row virtual color-only oracle; no K/production claim'),
        work_ledger=dict(primary_LS_iterations=scope_iterations['primary'],conditional_relaxed_LS_iterations=scope_iterations['relaxed'],engineering_LS_accepted_updates=actual_smoke,
            prior_pair_formal_GD_updates=400,prior_pair_engineering_GD_updates=8,endpoint_evidence_native_forward_calls=epreceipt['native_forward_calls'],endpoint_optimizer_updates=0,endpoint_solver_iterations=0),
        public_reload_formal_views=reload_count,engineering_smoke_extra_reload_views=len(smoke['public_reload']['rows']),
        recompute_errors=dict(max_RGB_f_opaque_metric_abs=max_metric_error,max_guarded_floor_vs_original_abs=max_floor_error),
        endpoint_gap_closed=dict(original_b_upper_not_persisted=True,root_authorized_readonly_replay=True,replayed_arrays_available=True,case_results_untouched=True,receipt=str(endpoint/'receipt.json')),
        evidence_boundaries=dict(RGB_f_FW_floor='independently recomputed from saved final predictions/GT/x/g and replayed pointwise b/upper',detail='constant has no detail gate; lowfreq gain independently recomputed; paint uses originally registered per-view detail statistics with independent threshold logic, not new edge/centroid code',
            negative='native FP32/atomic adjoint global conservative objective+box-gradient error bound remains unavailable; pointwise floor and FW values are numeric proxies, not strict negative certificates',
            K_reload='existing actual native execution receipts and immutable road fields audited; no new GPU replay by this independent CPU audit'),human_verdict=None,visual_verdict=None,production_promoted=False)
    write(out,result);print(json.dumps(dict(status=result['status'],checks=result['check_count'],failed=result['failed_checks'],witnesses=witnesses,work_ledger=result['work_ledger'],recompute_errors=result['recompute_errors'],output=str(out)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

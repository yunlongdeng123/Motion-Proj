"""Separate all-qualified BUILD-fit evaluation. CPU only; original QUERY intact."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
from scipy import ndimage as ndi

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
B=Q/'gpu_20261119/native_capacity_20261120_r1';BUILD=B/'all_teacher_build_20261120_r1'
local=Path(__file__).resolve().parent.parent/'final_eval_code'
sys.path.insert(0,str(local if local.exists()else B/'qa/final_eval_code'))
import eval_native_endpoints as E

LABEL='BUILD_fit_all_qualified'

def union_source(split,role):
    score=split[role+'_seen']|split[role+'_held']
    context=split[role+'_seen_context']|split[role+'_held_context']
    if np.any(score&(split[role+'_unmapped']|split['K']))or np.any(context&~score):raise ValueError('BUILD union includes unmapped/K or invalid context')
    if np.any(context&(split['fixed_alpha']<.95)):raise ValueError('unstable BUILD alpha')
    # Core is intentionally NOT the union of original per-split cores.
    return score,context

def combine_prior(a,b):
    n=a['pixels']+b['pixels'];means={}
    for k in a['mean_support_mass']:
        means[k]=((a['mean_support_mass'][k]or 0.)*a['pixels']+(b['mean_support_mass'][k]or 0.)*b['pixels'])/n if n else None
    return dict(pixels=n,measurement='center-UV fourcorner support, not physicalSigma attribution',
        original_source_weight_sum={k:a['original_source_weight_sum'][k]+b['original_source_weight_sum'][k]for k in a['original_source_weight_sum']},
        mean_support_mass=means,pixels_with_any_seed=a['pixels_with_any_seed']+b['pixels_with_any_seed'],
        pixels_with_any_teacher_H_permission=a['pixels_with_any_teacher_H_permission']+b['pixels_with_any_teacher_H_permission'],
        prior_exposure='preexposed prior plus all qualified labels intentionally used in BUILD; no validation claim')

def build_region(metric,packet,teacher,split,role,receipt):
    score,context=union_source(split,role);adapted=dict(split)
    adapted[f'{role}_{LABEL}']=score;adapted[f'{role}_{LABEL}_context']=context
    source=dict(F0={LABEL:combine_prior(receipt['F0']['seen'],receipt['F0']['held'])})
    source[LABEL+'_blocks']=receipt['seen_blocks']+receipt['held_blocks']
    result=E.evaluate_region(metric,packet,teacher,adapted,role,LABEL,source)
    result['interpretation_status']='training_fit_only_all_qualified_labels_not_validation'
    result['input_conditioning'].update(partition=LABEL,held_label='former QUERY held labels are now BUILD supervision',all_qualified_labels_fit=True,blind_test=False)
    result['data_role']=LABEL;result['former_held_now_training']=True
    return result

def flat_rows(report,arm,raw,source,path):
    result=[]
    for s in report['scales']:
        c=s['core'];macro=s['tile_aggregate']['macro']['HF_MAE'];f=next(x for x in s['teacher_flat_spurious']if abs(x['teacher_HF_RMS_cut']-2/255)<1e-12)
        result.append(dict(arm=arm,raw=raw,source=source,split=LABEL,scale=s['sigma_pixels'],teacher_role=report['teacher_role'],
            signal_space='stable_alpha_unpremult_srgb',score_n=report['fixed_score_pixels'],context_n=report['context_pixels'],stable_score_n=report['context_scored_pixels'],
            core_n=c['pixels'],boundary_n=s['boundary_pixels'],RGB_MAE=report['RGB_stable']['MAE'],RGB_RMSE=report['RGB_stable']['RMSE'],
            HF_cosine=c['HF_cosine'],HF_energy_ratio=c['HF_energy_ratio'],HF_relative_RMSE=c['HF_relative_RMSE'],HF_MAE=c['HF_MAE'],HF_RMSE=c['HF_RMSE'],HF_excess=c['excess_HF_energy'],
            flat_pred_RMS=f['predicted_HF_RMS'],flat_ref_RMS=f['teacher_HF_RMS'],flat_n=f['pixels'],
            Sobel_MAE=report['signed_Sobel_error']['MAE'],Sobel_RMSE=report['signed_Sobel_error']['RMSE'],Sobel_core_n=report['signed_Sobel_error']['pixels'],
            HF_tile_macro_MAE=macro['mean'],HF_tile_p95_MAE=macro['p95'],nonempty_tiles=s['tile_aggregate']['nonempty_core_tiles'],
            world_blocks=report['world_blocks'],interpretation=report['interpretation_status'],prior_preexposed=True,blind_test=False,former_held_now_training=True,detail_file=path,
            **E.fixed_edge_scalars(report)))
    return result

def verify_initial_ci(new,old):
    keys=['ci0','active_world_rows','eligible_active'];a=E.load(new,keys);b=E.load(old,keys)
    if not np.isfinite(a['ci0']).all()or np.any((a['ci0']<0)|(a['ci0']>1)):raise ValueError('BUILD initialci must be finite in[0,1]')
    exact={k:a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes()for k in keys}
    if not all(exact.values()):raise ValueError('BUILD initialci differs; old F0 packets cannot be reused')
    return dict(exact=exact,new_path=str(new),new_SHA256=E.sha(new),original_path=str(old),original_SHA256=E.sha(old),F0_native_packet_reuse_allowed=True)

def validate_build_endpoint(receipt,arm,protocol):
    if receipt.get('completed')is not True or receipt.get('arm')!=arm:raise ValueError('BUILD endpoint incomplete/wrongarm')
    extra=receipt.get('extra')or{};physical=receipt.get('physical')or{}
    if arm=='G_BUILD_FIELD':
        count=extra.get('logical_steps',receipt.get('logical_steps'))
        if count!=protocol['arms'][arm]['total_steps']:raise ValueError('BUILD FIELD not at fixed1200 endpoint')
        return dict(status='bound_fixed_BUILD_FIELD_endpoint',steps=count)
    if arm=='G_BUILD_NUMERIC':
        fg=extra.get('FGcount',physical.get('FG'));accepted=extra.get('accepted_steps',physical.get('accepted'))
        if not isinstance(fg,int)or fg<0 or fg>protocol['arms'][arm]['maxFG']or not isinstance(accepted,int)or accepted<0 or accepted>fg:raise ValueError('BUILD numeric cap/counters differ')
        point=extra.get('export_point',receipt.get('export_point',''))
        if 'last accepted'not in point or 'NEW_X'not in point:raise ValueError('BUILD numeric must export lastaccepted NEW_X, not pending/besttrial')
        return dict(status='bound_lastaccepted_BUILD_numeric',FGcount=fg,accepted_steps=accepted,strict_negative_certificate=False)
    raise ValueError('unknown BUILD arm')

def readonly_summary(P,T,alpha,mask,basis):
    if basis is None:return dict(status='N/A_no_bound_immutable_readonly_basis')
    b=basis['readonly_rgb'].astype(np.float64);w=basis['eligible_weight'].astype(np.float64);a=alpha.astype(np.float64)
    target=T.astype(np.float64)*a[None];lo=b;hi=b+w[None]
    residual=target-np.minimum(np.maximum(target,lo),hi);n=int(mask.sum())
    if not n:return dict(status='N/A_no_qualified_pixels',pixels=0)
    f=residual[:,mask]/a[mask][None];e=(P.astype(np.float64)[:,mask]-target[:,mask])/a[mask][None]
    floor=float((f*f).mean());actual=float((e*e).mean());ratio=w[mask]/a[mask]
    return dict(status='numeric_RGB_only_pointwise_proxy',pixels=n,We_over_alpha_mean=float(ratio.mean()),
        We_over_alpha_below={str(v):int((ratio<v).sum())for v in [.95,.5,.05]},
        straight_RGB_MAE_lower_proxy=float(np.abs(f).mean()),straight_RGB_MSE_lower_proxy=floor,actual_straight_RGB_MSE=actual,actual_over_floor=actual/floor if floor>0 else None,
        HF_bound_claim=False,strict_global_certificate=False,teacher_role='per-source frozen appearance target; C/S pseudo',
        no_semantic_readonly_error_claim=True)

def evaluate(plan,out):
    out=Path(out)
    if out.exists():raise FileExistsError('new BUILD evaluation directory required')
    if not plan.get('packets'):raise ValueError('no BUILD final packets bound')
    protocol=E.read(plan['protocol'])
    if E.sha(plan['protocol'])!=plan['protocol_sha256']:raise ValueError('BUILD protocol changed')
    initial=verify_initial_ci(plan['initial_ci_new'],plan['initial_ci_original'])
    metric=E.metric_module(plan['metric'],plan.get('metric_sha256',E.METRIC_SHA));split_root=Path(plan['split_root']);high=Path(plan['highres_root'])
    sr={r['raw']:r for r in E.read(split_root/'receipt.json')['frames']};out.mkdir(parents=True)
    rows=[];packets=[];coverage=[];guards=[];readonly=[];sources={str(plan['protocol']):E.sha(plan['protocol']),str(plan['metric']):E.sha(plan['metric'])};start=time.perf_counter()
    for item in plan['packets']:
        arm=item['arm'];raw=int(item['raw']);p=E.load(item['path']);E.validate_packet(p);baseline=E.load(item['baseline'])
        guard=dict(arm=arm,raw=raw,K_pixels=int(p['K'].sum()),road_alpha_exact=np.array_equal(p['road_alpha'],baseline['road_alpha']),joint_alpha_exact=np.array_equal(p['joint_alpha'],baseline['joint_alpha']),
            joint_K_max=float(np.abs(p['joint_rgb']-baseline['joint_rgb'])[:,p['K']].max())if p['K'].any()else None)
        guard={k:bool(v)if isinstance(v,np.bool_)else v for k,v in guard.items()};guards.append(guard)
        ep=dict(status='F0_reused_after_exact_initialci_check')if arm=='F0'else validate_build_endpoint(E.read(item['endpoint_receipt']),arm,protocol)
        packet=dict(arm=arm,raw=raw,path=item['path'],endpoint_provenance=ep)
        if raw in [86,110]:
            sp=split_root/f'{raw:03d}.npz';tp=high/f'targets/{raw:03d}.npz';split=E.load(sp)
            if not np.array_equal(p['road_alpha'],split['fixed_alpha'])or not np.array_equal(p['K'],split['K']):raise ValueError('BUILD native camera/alpha/K mismatch')
            vis=None;roadbasis=None
            if item.get('visibility_sidecar'):
                if item.get('frozen_visibility_reuse_confirmed')is not True:raise ValueError('unbound BUILD visibility')
                vis=E.load(item['visibility_sidecar']);sources[item['visibility_sidecar']]=E.sha(item['visibility_sidecar'])
            if item.get('readonly_road_basis'):
                if item.get('readonly_basis_reuse_confirmed')is not True:raise ValueError('unbound readonly source')
                roadbasis=E.load(item['readonly_road_basis'],['readonly_rgb','eligible_weight','alpha'])
                if not np.array_equal(roadbasis['alpha'],p['road_alpha']):raise ValueError('readonly basis alpha mismatch')
                sources[item['readonly_road_basis']]=E.sha(item['readonly_road_basis'])
            with np.load(tp,allow_pickle=False)as t:
                for role in ['A','C','S']:
                    T=t['T_'+role];report=build_region(metric,p,T,split,role,sr[raw]['domains'][role]);rel=f'{arm}/{raw:03d}_{role}_{LABEL}.json';dest=out/rel;dest.parent.mkdir(exist_ok=True);E.write(dest,report)
                    rows+=flat_rows(report,arm,raw,role,rel)
                    coverage.append(dict(arm=arm,raw=raw,source=role,**E.full_source_summary(p,T,split[role+'_pre_alpha'],split['canonical_hit'],split['source_material_valid'],vis)))
                    _,ctx=union_source(split,role);readonly.append(dict(arm=arm,raw=raw,source=role,split=LABEL,**readonly_summary(p['road_rgb'],T,p['road_alpha'],ctx,roadbasis)))
            for path in [sp,tp]:sources[str(path)]=E.sha(path)
        else:packet['reference']=dict(status='N/A_no_hidden_road_GT',teacher_metrics_computed=False)
        for key in ['path','baseline','endpoint_receipt']:
            if item.get(key):sources[item[key]]=E.sha(item[key])
        packets.append(packet);print(json.dumps(dict(arm=arm,raw=raw,complete=True,seconds=time.perf_counter()-start)),flush=True)
    expected=plan.get('expected_arms',['F0','G_BUILD_FIELD','G_BUILD_NUMERIC']);frames=plan.get('expected_frames',[86,110,134,179,198]);actual={(r['arm'],r['raw'])for r in packets};missing=[dict(arm=a,raw=f)for a in expected for f in frames if(a,f)not in actual]
    result=dict(schema='native_GS_all_teacher_BUILD_fit_evaluation.v1',complete=True,final_endpoint_set_complete=not missing,missing_registered_packets=missing,
        data_role=LABEL,rows=rows,packets=packets,coverage_rows=coverage,K_rows=guards,readonly_rows=readonly,initial_ci_reuse=initial,source_SHA256=sources,code_SHA256=E.sha(__file__),seconds=time.perf_counter()-start,
        interpretation='All formerly held qualified source labels are now BUILD_fit. No validation, blind/generalization or new method leaderboard claim.',
        causal_boundary='RGB-vsDETAIL loss conclusion remains the original QUERY matched experiment. BUILD is fixed-method asset refit; FIELD-seeded NUMERIC is extra numerical work, not samecompute comparison.',
        normalization='NEW all-context F0 training normalizers; QUERY and BUILD training objective values are not directly comparable. Raw quality vectors here share this BUILD evaluation domain.',
        original_QUERY_results_modified=False,GPU_executed=False,native_calls=0,optimizer_updates=0,human_verdict=None,production_promoted=False)
    E.write(out/'results.json',result)
    with(out/'metrics.csv').open('w',newline='',encoding='utf-8-sig')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=evaluate(E.read(a.plan),a.output);print(json.dumps(dict(complete=True,packets=len(d['packets']),rows=len(d['rows']),output=str(a.output))))

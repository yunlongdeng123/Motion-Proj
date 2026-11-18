"""只读共享世界材质结果：固定源域、频率掩码、alpha/K；默认仅准备manifest。"""
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
P=C/'road_multiscale_material_20261118_r1'
N=C/'road_activation_20261118_r1'
G=C/'road_capacity_gpu_20261118_r1'
FRAMES=(86,110,134,179,198)


def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (keys if keys else z.files)}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_phase_table(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=['stage','stage_role','raw','view_role','A_real_premult_MAE','C_pseudo_premult_MAE','S_pseudo_premult_MAE',
        'old2222_premult_MAE','new909_premult_MAE','full3187_premult_MAE','remaining56_premult_MAE',
        'stripe_vertical_E','stripe_horizontal_E','D_pixels','D_gap','D_thin','D_supported','alpha_max_change','joint_K_max']
    with path.open('w',encoding='utf-8',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for name,stage in result['stages'].items():
            for row in stage['rows']:
                value=dict(stage=name,stage_role=stage['role'],raw=row['raw'],view_role=row['role'],
                    stripe_vertical_E=row['stripe']['vertical_E'],stripe_horizontal_E=row['stripe']['horizontal_E'],
                    alpha_max_change=row['alpha_max_change'],joint_K_max=row['joint_K_max'])
                value.update({'D_'+key:row['original_D'][key]for key in('pixels','gap','thin','supported')})
                for label,region in [('A_real','A'),('C_pseudo','C'),('S_pseudo','S'),('old2222','old2222'),
                                     ('new909','new909'),('full3187','full3187'),('remaining56','remaining56')]:
                    value[label+'_premult_MAE']=row.get('material',{}).get(region,{}).get('premult',{}).get('MAE')
                writer.writerow(value)


def default_manifest():
    return dict(schema='world_material_CPU_result_manifest.v1',
        coverage_baseline='coverage_parent',historical_I0='historical_I0',L_control='L800',ABC='ABC800',
        stripe_reference='filtered_A0',stripe_I0_comparator='matched_I0',fit_reference='matched_I0',A_guard_reference='filtered_A0',
        frames=list(FRAMES),targets=str(P/'stage0_native_r1'),
        fixed_original_D=str(C/'road_capacity_cpu_20261117_r1/ideal_targets_r2/constant/{raw:03d}.npz'),
        source_scope=str(P/'method_scope_r1/scope.npz'),
        frozen_frequency=str(G/'check3/frozen_frequency_masks.npz'),
        fixed_material_masks=str(N/'qa/material_masks.npz'),
        stages=[dict(name='coverage_parent',role='fixed repaired coverage, zero optimization',
            array_template=str(N/'B32_beam95_native_r1/{raw:03d}_native.npz'),
            keys=dict(P='road_rgb',alpha='road_alpha',J='joint_rgb')),
            dict(name='historical_I0',role='historical olderS2222 training, same repaired coverage',
            array_template=str(N/'composed_I0_native_r1/{raw:03d}_native.npz'),
            keys=dict(P='road_rgb',alpha='road_alpha',J='joint_rgb'))],
        pending_stages=['center_A0','filtered_A0','A200','B500','matched_I0','L800','ABC800','A0_midpoint','TRELLIS2'],
        append_stage_schema=dict(name='L800',role='matched final',array_template='absolute/{raw:03d}_native.npz',
            keys=dict(P='road_rgb',alpha='road_alpha',J='joint_rgb'),field_asset='absolute/asset',reload_receipt='absolute/reload.json'),
        optional_common_visibility='same-geometry current N parent actual branch-alphaT arrays only; olderA1 is not current state',
        thresholds=dict(K_max=0,alpha_max_change=0,reload_float=1e-6,
            stripe_excess_fraction=.5,fit_MAE_tolerance=1/255,A_initial_guard_tolerance=1/255,
            quality_threshold_policy='root frozen before formal:ABC vs matchedI0 A/C/S <=1code; A vs filteredA0<=1code; diagnostic positive stripe excess <=half; no automatic visual pass'),
        source_roles=dict(A='real visible',C='frozen hole pseudo',S='frozen target-shadow pseudo'),
        evaluation_only=True,GPU_executed_by_auditor=False)


def packet(stage,raw):
    path=Path(stage['array_template'].format(raw=raw))
    z=load(path);key=stage['keys']
    result=dict(path=str(path),P=z[key['P']],alpha=z[key['alpha']].squeeze())
    if key.get('J') in z:result['J']=z[key['J']]
    for name in ('P','J'):
        if name in result and result[name].shape!=(3,320,480):raise ValueError('expected3x320x480:'+name)
    if result['alpha'].shape!=(320,480):raise ValueError('alpha形状错误')
    if not all(np.isfinite(v).all() for v in result.values() if isinstance(v,np.ndarray)):
        raise ValueError('native float含非finite')
    return result


def region_stats(z,T,mask):
    a=z['alpha'][mask].astype(np.float64);P0=z['P'][:,mask].astype(np.float64);T=T[:,mask].astype(np.float64)
    n=len(a)
    if not n:return dict(pixel_frames=0)
    def metric(r):return dict(MAE=float(abs(r).mean()),RMSE=float(np.sqrt((r*r).mean())),SSE=float((r*r).sum()))
    return dict(pixel_frames=n,RGB_denominator=3*n,opaque=metric(P0-T),premult=metric(P0-a[None]*T),
        joint=metric(z['J'][:,mask].astype(np.float64)-T) if 'J'in z else None,
        alpha=dict(mean=float(a.mean()),gap=int((a<.05).sum()),thin=int(((a>=.05)&(a<=.9)).sum()),
                   supported=int((a>.9).sum()),exact_zero=int((a==0).sum())),
        all_color_supported=bool((a>.9).all()),
        comparability='premult only compares material directly when coverage is fixed; unqualified/unsupported residual is not an advantage')


def coverage_stats(alpha,mask):
    a=alpha[mask]
    return dict(pixels=int(mask.sum()),gap=int((a<.05).sum()),thin=int(((a>=.05)&(a<=.9)).sum()),
                supported=int((a>.9).sum()),exact_zero=int((a==0).sum()),mean=float(a.mean()))


def pooled(stats):
    n=sum(x['pixel_frames'] for x in stats)
    out=dict(pixel_frames=n,RGB_denominator=3*n)
    for metric in ('opaque','premult','joint'):
        if any(x.get(metric)is None for x in stats):out[metric]=None;continue
        out[metric]=dict(MAE=sum(x['pixel_frames']*x[metric]['MAE'] for x in stats)/n,
                         RMSE=math.sqrt(sum(x[metric]['SSE'] for x in stats)/(3*n)))
    out['alpha']=dict(mean=sum(x['pixel_frames']*x['alpha']['mean'] for x in stats)/n,
        **{key:sum(x['alpha'][key] for x in stats) for key in ('gap','thin','supported','exact_zero')})
    return out


def evaluate(manifest):
    m=read(manifest);stages={s['name']:s for s in m['stages']}
    baseline={raw:packet(stages[m['coverage_baseline']],raw) for raw in FRAMES}
    frequency=load(m['frozen_frequency']);fixed=load(m['fixed_material_masks']);K=load(G/'check1/native_r1/K_masks.npz')
    targets={raw:load(Path(m['targets'])/f'{raw:03d}_targets.npz') for raw in (86,110)}
    source_counts={k:sum(int(t[k].sum())for t in targets.values())for k in ('A','C','S','fullS')}
    if source_counts!=dict(A=16904,C=10830,S=3131,fullS=3187):raise ValueError('冻结A/C/S/fullS分母不同')
    original_D={raw:load(m['fixed_original_D'].format(raw=raw),['mask'])['mask'] for raw in FRAMES}
    if sum(int(mask.sum())for mask in original_D.values())!=38416:raise ValueError('原D38416分母不同')
    common_masks={'old2222':'old_photo_S','new909':'current_new_support_909','full3187':'full_S','remaining56':'current_remaining_56'}
    for label,expected in [('old2222',2222),('new909',909),('full3187',3187),('remaining56',56)]:
        if sum(int(fixed[f'{raw:03d}_{common_masks[label]}'].sum())for raw in(86,110))!=expected:
            raise ValueError('冻结material层分母不同:'+label)
    for raw in(86,110):
        old,new,full,remaining=[fixed[f'{raw:03d}_{common_masks[k]}'] for k in('old2222','new909','full3187','remaining56')]
        if np.any(old&new)or np.any((old|new)&remaining)or not np.array_equal(old|new,targets[raw]['S'])or not np.array_equal(full,old|new|remaining):
            raise ValueError('冻结old/new/remaining层不构成固定fullS分区')
    output={};all_checks={};missing=[]
    for raw,t in targets.items():
        for a,b in [('A','C'),('A','S'),('C','S')]:
            all_checks[f'sources:{raw}:{a}_{b}_disjoint']=not bool((t[a]&t[b]).any())
        for key in('A','C','S','fullS'):
            all_checks[f'sources:{raw}:{key}_excludes_K']=not bool((t[key]&K[f'{raw:03d}']).any())
    for name,stage in stages.items():
        rows=[];material={}
        for raw in FRAMES:
            try:z=packet(stage,raw)
            except FileNotFoundError:missing.append(name+':'+str(raw));continue
            alpha=z['alpha'];previous=baseline[raw]
            maxalpha=float(abs(alpha-previous['alpha']).max())
            k=K[f'{raw:03d}'];Kmax=float(abs(z['J'][:,k]-previous['J'][:,k]).max())if'J'in z else None
            all_checks[f'{name}:{raw}:alpha_exact']=np.array_equal(alpha,previous['alpha'])
            if Kmax is None:missing.append(name+':'+str(raw)+':jointRGB')
            else:all_checks[f'{name}:{raw}:K_exact']=Kmax==0
            V=frequency[f'{raw:03d}_vertical'];H=frequency[f'{raw:03d}_horizontal'];D=frequency[f'{raw:03d}_domain']
            F=z['P'].astype(np.float64)
            ev=float(abs(F[:,2:]-2*F[:,1:-1]+F[:,:-2])[:,V].mean())
            eh=float(abs(F[:,:,2:]-2*F[:,:,1:-1]+F[:,:,:-2])[:,H].mean())
            row=dict(raw=raw,role='train-source'if raw in(86,110)else'diagnostic, not blind',source=z['path'],
                     alpha_max_change=maxalpha,K_pixels=int(k.sum()),joint_K_max=Kmax,
                     original_D=coverage_stats(alpha,original_D[raw]),
                     stripe=dict(vertical_E=ev,horizontal_E=eh,domain_pixels=int(D.sum()),
                                 RGB_delta_energy=float(((F-previous['P'])[:,D]**2).mean())))
            if raw in targets:
                t=targets[raw];region={key:(t[key],t['target_'+key])for key in('A','C','S')}
                region.update({key:(fixed[f'{raw:03d}_{mask}'],t['target_S'])for key,mask in common_masks.items()})
                row['material']={key:region_stats(z,T,mask)for key,(mask,T)in region.items()}
                for key,value in row['material'].items():material.setdefault(key,[]).append(value)
            rows.append(row)
        output[name]=dict(role=stage['role'],rows=rows,pooled={k:pooled(v)for k,v in material.items()},
                          field_asset=stage.get('field_asset'),reload_receipt=stage.get('reload_receipt'))
        if stage.get('reload_receipt'):
            path=Path(stage['reload_receipt'])
            if not path.exists():missing.append(name+':reload_receipt')
            else:output[name]['reload']=read(path)
    stripes=[];fit=[];L_deltas=[];A_guard=[]
    def complete_rows(names,raws):
        return all(name in output and set(raws).issubset({r['raw']for r in output[name]['rows']})for name in names)
    if complete_rows([m.get('ABC'),m.get('stripe_I0_comparator'),m.get('stripe_reference')],(134,179,198)):
        byraw={name:{r['raw']:r for r in data['rows']}for name,data in output.items()}
        for raw in(134,179,198):
            base=byraw[m['stripe_reference']][raw]['stripe']['vertical_E'];old=byraw[m['stripe_I0_comparator']][raw]['stripe']['vertical_E']
            new=byraw[m['ABC']][raw]['stripe']['vertical_E'];excess=old-base;fraction=m['thresholds'].get('stripe_excess_fraction')
            stripes.append(dict(raw=raw,filtered_A0_baseline=base,matched_I0=old,ABC=new,I0_positive_excess=excess,
                excess_reduction=1-(new-base)/excess if excess>0 else None,
                applicable=excess>0,registered_limit_fraction=fraction,
                passed=(new-base)<=fraction*excess if excess>0 and fraction is not None else None))
    if complete_rows([m.get('ABC'),m.get('L_control')],(86,110)):
        for raw in(86,110):
            L=next(r for r in output[m['L_control']]['rows']if r['raw']==raw)['material']
            A=next(r for r in output[m['ABC']]['rows']if r['raw']==raw)['material']
            for region in('A','C','S','old2222','new909','full3187','remaining56'):
                delta=A[region]['premult']['MAE']-L[region]['premult']['MAE']
                L_deltas.append(dict(raw=raw,region=region,ABC_minus_L_premult_MAE=delta,role='matched strategy report, not substitute for matchedI0 gate'))
    if complete_rows([m.get('ABC'),m.get('fit_reference')],(86,110)):
        for raw in(86,110):
            ref=next(r for r in output[m['fit_reference']]['rows']if r['raw']==raw)['material']
            new=next(r for r in output[m['ABC']]['rows']if r['raw']==raw)['material']
            for region in('A','C','S'):
                delta=new[region]['premult']['MAE']-ref[region]['premult']['MAE'];tol=m['thresholds']['fit_MAE_tolerance']
                fit.append(dict(raw=raw,region=region,ABC_minus_matchedI0_premult_MAE=delta,registered_tolerance=tol,passed=delta<=tol))
    if complete_rows([m.get('ABC'),m.get('A_guard_reference')],(86,110)):
        for raw in(86,110):
            ref=next(r for r in output[m['A_guard_reference']]['rows']if r['raw']==raw)['material']['A']
            new=next(r for r in output[m['ABC']]['rows']if r['raw']==raw)['material']['A']
            delta=new['premult']['MAE']-ref['premult']['MAE'];tol=m['thresholds']['A_initial_guard_tolerance']
            A_guard.append(dict(raw=raw,ABC_minus_filteredA0_A_premult_MAE=delta,registered_tolerance=tol,passed=delta<=tol))
    result=dict(schema='shared_world_material_independent_CPU_metrics.v1',manifest=str(manifest),
        source_counts=source_counts,fixed_denominators=dict(S3131=3131,old2222=2222,new909=909,full3187=3187,remaining56=56,
            originalD=38416,frequency_domain=37404,K=77266),
        GPU_executed_by_auditor=False,native_calls=0,training_updates=0,
        normalization='pooled binary-mask pixel/channel means + per-view values; training stochastic views are balanced separately',
        teachers='A real visible; C/S pseudo, notGT; two train/three diagnostic views; inherited199BUILD not blind',
        stages=output,mechanical_checks=all_checks,missing=missing,
        mechanical_status='failed'if not all(all_checks.values())else'incomplete'if missing else'pass_float_packets',
        matched_I0_fit_gate=fit,real_A_initial_guard=A_guard,matched_L_strategy_deltas=L_deltas,diagnostic_stripe_comparison=stripes,
        detail_status='requires frozen observed/source-qualified proxies and actual-image review; never transplant ideal12cm GT to real road',
        actual_joint_alphaT=None,visibility_boundary='numeric branch values require current same-geometry native contribution export; road alpha is not joint contribution',
        authoritative_field_and_optimizer_audit='separate lifecycle/state receipts; do not infer from low image loss',
        human_verdict=None,visual_verdict=None,production_promoted=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,default=P/'qa/result_metrics.json');args=parser.parse_args()
    if args.manifest is None:
        path=P/'qa/result_manifest_template.json';write(path,default_manifest());print(path);return
    result=evaluate(args.manifest);write(args.output,result);write_phase_table(args.output.with_suffix('.csv'),result)
    print(json.dumps(dict(output=str(args.output),status=result['mechanical_status'],stages=list(result['stages']),missing=result['missing'])))


if __name__=='__main__':main()

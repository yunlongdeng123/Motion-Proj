"""Numerical pointwise RGB relaxation with frozen actual native eligible weights."""
from pathlib import Path
import json
import numpy as np
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
P=Q.parent/'road_multiscale_material_20261118_r1';R=Q/'gpu_20261119'
out=R/'pixel_color_range_audit_r1';out.mkdir(exist_ok=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
rows=[]
for raw in (86,110):
    b=load(R/'native_qualification_r2/basis'/f'{raw:03d}'/'road_basis.npz')
    t=load(P/'stage0_native_r1'/f'{raw:03d}_targets.npz')
    lower=b['baseline_rgb'].astype(np.float64)-b['eligible_rgb'].astype(np.float64)
    weight=b['eligible_weight'].astype(np.float64);upper=lower+weight[None]
    for domain in ('A','C','S','fullS'):
        m=t[domain];key='S' if domain=='fullS' else domain
        target=t['fixed_alpha'][None]*t['target_'+key]
        error=np.maximum(np.maximum(lower-target,target-upper),0)
        zero=weight==0
        methods={}
        for arm in ('M00','M10','M01','M11','T11'):
            d=load(R/'formal_r1'/arm/'C800/rendered'/f'{raw:03d}.npz')
            methods[arm]=float(np.abs(d['road_rgb']-target)[:,m].mean())
        rows.append(dict(raw=raw,domain=domain,pixels=int(m.sum()),eligible_weight_zero_pixels=int((zero&m).sum()),
            numerical_relaxed_min_MAE=float(error[:,m].mean()),
            error_budget_5e_6_subtracted_proxy=float(np.maximum(error-5e-6,0)[:,m].mean()),
            zero_weight_MAE_contribution_to_full_domain=float((error[:,m]*(zero[m][None])).sum()/(3*m.sum())),
            native_eligible_weight_mean=float(weight[m].mean()),actual_final_MAE=methods))
pooled={}
for key in ('A','C','S','fullS'):
    selected=[r for r in rows if r['domain']==key];den=sum(r['pixels'] for r in selected)
    pooled[key]=dict(pixels=den,eligible_weight_zero_pixels=sum(r['eligible_weight_zero_pixels'] for r in selected),
        numerical_relaxed_min_MAE=sum(r['pixels']*r['numerical_relaxed_min_MAE'] for r in selected)/den,
        zero_weight_MAE_contribution_to_full_domain=sum(r['pixels']*r['zero_weight_MAE_contribution_to_full_domain'] for r in selected)/den,
        actual_final_MAE={a:sum(r['pixels']*r['actual_final_MAE'][a] for r in selected)/den for a in ('M00','M10','M01','M11','T11')})
result=dict(schema='frozen_eligible_pointwise_RGB_range_relaxation.v1',rows=rows,pooled=pooled,
    equation='P=D+We*F, F in [0,1]^3; independent pixel RGB relaxes shared fields and Gaussian-average RGB constraints',
    use='Locate immutable native contribution/color-range floors; not a global 3DGS representation or convergence certificate.',
    floating_point='FP64 calculation over actual float32 exported basis; unverified global rounding bounds, 5e-6 subtraction is an engineering sensitivity diagnostic only.',
    original_denominators_unchanged=True,scene_optimization_updates=0,native_calls=0,GPU_calls=0)
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(pooled,indent=2))

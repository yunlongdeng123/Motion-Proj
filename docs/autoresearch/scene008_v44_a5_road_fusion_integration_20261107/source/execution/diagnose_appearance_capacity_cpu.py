"""Lipschitz预算下的行级可写质量谱，纯CPU诊断，不输出可载入选择器。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
E=ROOT/'scene008_v44_a5_road_fusion_integration_20261107_r1/E4_appearance_qualification'
OBS=ROOT/'scene008_instances_20261105_r1/observations/frames'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def capacity(primary,inside,outside,budget):
    eligible=inside>0
    free=eligible&(outside==0)&(primary>0)
    positive=np.flatnonzero(eligible&(outside>0)&(primary>0))
    ratio=primary[positive]/outside[positive]
    order=np.lexsort((positive,-ratio));ids=positive[order]
    cost=outside[ids];value=primary[ids];cum=np.cumsum(cost)
    full=int(np.searchsorted(cum,budget,side='right'))
    used=float(cum[full-1]) if full else 0.
    fractional=(budget-used)/cost[full] if full<len(ids) else 0.
    fractional=min(max(float(fractional),0.),1.)
    free_value=float(primary[free].sum())
    objective=free_value+float(value[:full].sum())+(float(value[full])*fractional if full<len(ids) else 0.)
    remaining=budget;greedy=free_value;chosen=[]
    for index,c,v in zip(ids,cost,value):
        if c<=remaining:
            remaining-=float(c);greedy+=float(v);chosen.append(int(index))
    qs=[0,.1,.5,.9,.95,.99,1]
    return dict(free_primary_mass=free_value,free_primary_contributing_rows=int(free.sum()),
        positive_outside_primary_contributing_rows=len(ids),
        positive_outside_ratio_quantiles={str(q):float(np.quantile(ratio,q)) for q in qs} if len(ratio) else {},
        continuous_amplitude_LP_optimal_primary_mass=objective,
        continuous_amplitude_LP_full_positive_rows=full,
        fractional_next_row=int(ids[full]) if full<len(ids) and fractional>0 else None,
        fractional_next_amplitude=fractional,
        whole_row_greedy_feasible_primary_mass=greedy,whole_row_greedy_positive_rows=len(chosen),
        whole_row_greedy_outside_cost=budget-remaining,
        primary_mean_RGB_change_Lipschitz_upper_bound=objective/68.,
        top_primary_to_outside_rows=[dict(global_branch_row=int(i),primary_mass=float(primary[i]),
            outside_sum=float(outside[i]),ratio=float(primary[i]/outside[i])) for i in ids[:30]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    q=json.loads((E/'qualification_manifest.json').read_text());n=q['branch_rows']['native_static']
    paths={};counts=[]
    for raw in q['calibration_raw_ids']:
        p=OBS/f'{raw:03d}.npz';paths[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            a5=z['actor_labels']==1;a6=z['visible'][0].astype(bool);a37=z['visible'][1].astype(bool)
            protected=z['protect'].astype(bool)&~a6&~a37&~z['road'].astype(bool)&~z['road_edit'].astype(bool)
            y,_=np.nonzero(a5);upper=a5&(np.indices(a5.shape)[0]<=y.min()+.25*(y.max()-y.min())) if len(y) else a5
        counts.append(dict(raw=raw,a5_upper=int(upper.sum()),a6_RGB=int(a6.sum()),
                           a37_KEEP_RGB=int(a37.sum()),protected_nonroad=int(protected.sum())))
    minimum=min(v for c in counts for k,v in c.items() if k!='raw' and v>0)
    # y_r=||delta_r||_1/3 ∈[0,1]。Σ outside*y <= Nmin*1e-6。
    normalized_budget=minimum*1.e-6
    with np.load(E/q['selection_file'],allow_pickle=False) as z:
        inside=np.r_[z['static_inside_mass'],z['road_inside_mass']]
        outside=np.r_[z['static_outside_mass'],z['road_outside_mass']]
    primary={}
    for state in ['keep','delete']:
        p=E/'point_mass'/f'128_{state}.npz';paths[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z:primary[state]=z['mass'][:,5].astype(np.float64)
    results={state:capacity(p,inside,outside,normalized_budget) for state,p in primary.items()}
    results['shared_average_operator_leverage']=capacity((primary['keep']+primary['delete'])/2,
        inside,outside,normalized_budget)
    for result in results.values():
        for row in result['top_primary_to_outside_rows']:
            i=row.pop('global_branch_row');row['branch']='native_static' if i<n else 'oldroad';row['local_row_id']=i if i<n else i-n
    report=dict(status='CPU_DIAGNOSTIC_ONLY_LIPSCHITZ_CAPACITY_SPECTRUM',
        input_hashes=paths,qualification_manifest_sha256=sha(E/'qualification_manifest.json'),
        protection_original_pixel_counts=counts,N_min=minimum,per_domain_MAE_regression_bound=1.e-6,
        full_L1_budget=3*normalized_budget,normalized_y_budget=normalized_budget,
        y_definition='y_r=||deltaRGB_r||_1/3 in[0,1] under deltaRGB channel bounds[-1,1]',
        proof=['Geometry/opacity/masks/depth-sort/environment are fixed.',
            'Selected-row clamp_min(c+delta,0) is1-Lipschitz and original c>=0, so each row color L1 change<=||delta||1.',
            'Every measured original protection domain is a subset of its corresponding registered guard.',
            'outside_sum sums alphaT across union guards and all8 views/states, so each domain RGB MAE regression<=sum(outside_sum*||delta||1)/(3*Ndomain).',
            'Budget<=3*Nmin*1e-6 thus conservatively bounds each measured original protection-domain MAE regression by1e-6.'],
        optimization_role='Exact continuous-amplitude fractional-knapsack LP for operator leverage; whole-row greedy is feasible lower bound, not claimed integer optimum.',
        results=results,static_or_road_geometry_changes=0,fit_steps=0,GPU_forwards=0,
        not_a_registered_new_method=True,no_selection_sidecar_created=True,
        limitations=['Operator leverage is not achievable target-error reduction; row color directions and all target domains can conflict.',
            'Continuous y values do not define a chosen fitted delta; no new rows or thresholds are adopted here.',
            'Bound covers only calibrated four BUILD views/state protection sets, not unseen49-frame claims.',
            'Actual floating-point/replay checks remain required even for a mathematically conservative bound.'])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(N_min=minimum,full_L1_budget=3*normalized_budget,
        capacities={s:{k:v for k,v in d.items() if k in ['free_primary_mass','continuous_amplitude_LP_optimal_primary_mass','whole_row_greedy_feasible_primary_mass','positive_outside_primary_contributing_rows','primary_mean_RGB_change_Lipschitz_upper_bound']} for s,d in results.items()}),indent=2))


if __name__=='__main__':main()

"""Apply the previously frozen gates; never choose a rule per masked region."""
import json,gzip,math,hashlib
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent.parent
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def get(d,path):
    for key in path.split('.'):d=d[key]
    return d
def main():
    contract=read(RUN/'evaluation_contract.json');limits=contract['gates'];pseudo=read(RUN/'pseudo_evaluation/summary.json')
    real=read(RUN/'real_evaluation/summary.json')
    with gzip.open(RUN/'real_evaluation/metrics.json.gz','rt') as f:frames=json.load(f)
    methods=contract['methods'];cases=pseudo['cases'];results={};base=[next(m for m in c['methods'] if m['method']=='median') for c in cases]
    paths=['solo.a6','scene.keep.a6','scene.delete.a6','scene.keep.preservation.a37',
       'scene.keep.preservation.road_visible','scene.keep.preservation.outside',
       'scene.delete.preservation.road_visible','scene.delete.preservation.outside','scene.delete.preservation.road_reobserved']
    informative=all(c['status']=='INFORMATIVE' for c in contract['cases'])
    for name in methods:
        current=[next(m for m in c['methods'] if m['method']==name) for c in cases]
        gains=[b['HELD_spatial_withheld']['mean']-c['HELD_spatial_withheld']['mean'] for b,c in zip(base,current)]
        baseline=float(np.mean([b['HELD_spatial_withheld']['mean'] for b in base]));value=float(np.mean([c['HELD_spatial_withheld']['mean'] for c in current]));gain=baseline-value
        mixed=float(np.mean([c['HELD_mixed']['mean']-b['HELD_mixed']['mean'] for b,c in zip(base,current)]))
        checks=dict(informative=informative,macro_absolute_gain=gain>=limits['absolute_HELD_macro_improvement_min'],
            macro_relative_gain=gain/baseline>=limits['relative_HELD_macro_improvement_min'],
            no_case_worsening=max(-g for g in gains)<=limits['maximum_case_worsening'],
            sufficient_cases_improve=sum(g>=limits['case_improvement_min'] for g in gains)>=limits['case_improvements_at_least'],
            mixed_guard=mixed<=limits['HELD_mixed_macro_worsening_max'])
        guards=[]
        for p in paths:
            # Fixed pooled masks; counts do not depend on candidate colours.
            summary=real['methods'][name]['all49'];error=summary[p+'.rgb_error']['mean'];before=summary[p+'.baseline_rgb_error']['mean'];change=summary[p+'.parent_rgb_change']['mean']
            samples=[(r['raw'],get(r,p)) for r in frames[name]]
            worst_change=max(samples,key=lambda item:item[1]['parent_rgb_change']['mean'] or 0)
            worst_error=max(samples,key=lambda item:(item[1]['rgb_error']['mean'] or 0)-(item[1]['baseline_rgb_error']['mean'] or 0))
            maximum_change=worst_change[1]['parent_rgb_change']['mean'] or 0
            maximum_error=(worst_error[1]['rgb_error']['mean'] or 0)-(worst_error[1]['baseline_rgb_error']['mean'] or 0)
            passed=(change or 0)<=limits['real_mean_RGB_parent_change_max'] and maximum_change<=limits['real_mean_RGB_parent_change_max'] and \
                (error-before)<=limits['real_pooled_and_each_frame_RGB_MAE_increase_max'] and maximum_error<=limits['real_pooled_and_each_frame_RGB_MAE_increase_max']
            guards.append(dict(domain=p,pooled_parent_RGB_change=change,pooled_RGB_MAE_increase=error-before,
                worst_frame_parent_RGB_change=maximum_change,worst_change_raw=worst_change[0],
                worst_frame_RGB_MAE_increase=maximum_error,worst_error_raw=worst_error[0],
                parent_change_max=summary[p+'.parent_rgb_change']['max'],above002=summary[p+'.parent_rgb_change']['above_002'],pass_guard=passed))
        preserve=all(g['pass_guard'] for g in guards)
        results[name]=dict(HELD_macro_MAE=value,baseline_HELD_macro_MAE=baseline,absolute_gain=gain,relative_gain=gain/baseline,
            case_gains=dict(zip([c['case'] for c in cases],gains)),HELD_mixed_macro_worsening=mixed,
            quality_checks=checks,quality_pass=all(checks.values()),preservation_pass=preserve,preservation_domains=guards,
            eligible_to_replace_reference=all(checks.values()) and preserve and name!='median')
    eligible=[name for name in methods if results[name]['eligible_to_replace_reference']]
    chosen=min(eligible,key=lambda name:results[name]['HELD_macro_MAE']) if eligible else 'median'
    decision=dict(status='COMPLETED_FIXED_GATE_DECISION',evaluation_contract_sha256=hashlib.sha256((RUN/'evaluation_contract.json').read_bytes()).hexdigest(),
        methods=results,chosen_global_method=chosen,new_scene_adopted=bool(eligible),
        current_M2_retained=not bool(eligible),parameter_search_performed=False,
        alpha_depth_groups_bitwise_unchanged=all(real[k] for k in ('alpha_bitwise_same','depth_bitwise_same','group_contributions_bitwise_same')),
        interpretation='Preservation and visibility are distinct from recovering true missing appearance. Failed colour continuation is not evidence about hidden PVD shape accuracy.')
    write(RUN/'decision.json',decision);print(json.dumps(dict(chosen=chosen,new_adoption=bool(eligible),quality={m:r['quality_pass'] for m,r in results.items()},preservation={m:r['preservation_pass'] for m,r in results.items()})))
if __name__=='__main__':main()

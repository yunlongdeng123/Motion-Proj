import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from audit_worldsim_v74_h2_p15 import write,stats
p=Path(sys.argv[1])
read=lambda f:json.loads((p/f).read_text())
trans=read('ray_transitions.json');teacher=read('teacher_error_rows.json');drifts=read('drifts.json');eps=read('episodes.json');cf=read('normal_tangent_counterfactual.json');local=read('local_control_rows.json');paired=read('paired_failures.json');controls=read('control_rows.json')
lookup={(r['model'],r['case'],r['step'],r['ray']):r for r in trans}
out={}
for m in ['A','C2']:
    out[m]={}
    for fam in ['all','multilayer','grazing_thin','shared_support','missing_support']:
        sel=lambda rr:[r for r in rr if r['model']==m and (fam=='all' or r['family']==fam)]
        failures=[r for r in sel(trans) if r['category']=='hit_early'];stable=[r for r in sel(trans) if r['category']=='hit_hit'];item={}
        for label,rows in [('failure',failures),('stable',stable)]:
            dec=[r['decomposition'] for r in rows if r['decomposition']]
            item[label]=dict(n=len(rows),metrics={k:stats([abs(d[k]) if k.endswith('_m') else d[k] for d in dec]) for k in ['normal_m','tangent_m','center_m','condition_gain','normal_depth_m','tangent_depth_m','rotation_depth_m','rotation_deg']})
        td=[r for r in sel(teacher) if lookup[r['model'],r['case'],r['step'],r['ray']]['category']=='hit_early']
        item['teacher_at_failure']=dict(n=len(td),teacher_hit=sum(r['teacher_label']==0 for r in td),same_winner=sum(r['teacher_winner']==r['learner_winner'] for r in td),
            metrics={k:stats([abs(r['decomposition'][k]) if k.endswith('_m') else r['decomposition'][k] for r in td if r['decomposition']]) for k in ['normal_m','tangent_m','center_m','condition_gain','normal_depth_m','tangent_depth_m','rotation_depth_m','rotation_deg']})
        for variant in ['normal_only','tangent_only']:
            rr=[r for r in sel(cf) if r['variant']==variant];item[variant]={k:sum(r['label']==i for r in rr) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])}
        de=sel(drifts);ee=[r for r in sel(eps) if r['from_hit']]
        item['drift']=dict(n=len(de),one_step=sum(r['hit_run_steps']==1 for r in de),multi_step=sum(r['hit_run_steps']>1 for r in de),last_exceeds_half_net=sum(r['last_share_of_total'] is not None and r['last_share_of_total']>.5 for r in de),
            metrics={k:stats([r[k] for r in de]) for k in ['hit_run_steps','total_depth_m','last_depth_m','earlier_advance_m','last_share_of_total','normal_sum_m','rotation_sum_m','switch_sum_m']},
            recovered=sum(r['recovered'] for r in ee),right_censored=sum(r['right_censored'] for r in ee),recoveries=[dict(case=r['case'],ray=r['ray'],start=r['start_step'],hit=r['first_hit_step']) for r in ee if r['recovered']])
        item['local_control']={}
        for mode in ['local_build','local_fit_diagnostic']:
            rr=[r for r in sel(local) if r['mode']==mode];item['local_control'][mode]=dict(states=len(rr),original_failure_rays=sum(r['p15_failure_rays'] for r in rr),
                failure_labels={k:sum(r['p15_failure_labels'][str(i)] for r in rr) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])},
                all_state_raw_hit_to_nonhit=sum(r['raw_hit_to_nonhit'] for r in rr),all_state_raw_early_to_hit=sum(r['raw_early_to_hit'] for r in rr))
        item['paired_closed']={mode:{k:sum(r['labels'][mode]==i for r in sel(paired)) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])} for mode in ['closed_build','closed_fit_diagnostic']}
        out[m][fam]=item
write(p/'interpretation_summary.json',out)
# 射线条件尺度与硬 first-depth error 的描述性相关；不将任务内射线视为独立检验。
cor=[]
for m in ['A','C2']:
    for fam in ['all','grazing_thin']:
        rows=[r for r in teacher if r['model']==m and (fam=='all' or r['family']==fam) and r['decomposition'] and r['teacher_has_learner_patch_hit'] and r['first_depth_error_m'] is not None]
        y=np.array([abs(r['first_depth_error_m']) for r in rows]);rec=dict(model=m,family=fam,n=len(rows),correlation={})
        for key in ['center_m','normal_m','normal_depth_m','tangent_m']:
            x=np.array([abs(r['decomposition'][key]) for r in rows]);rec['correlation'][key]=float(spearmanr(x,y).statistic) if len(np.unique(x))>1 else None
        cor.append(rec)
write(p/'ray_metric_correlations.json',cor)
print(json.dumps({m:{f:{k:v for k,v in out[m][f].items() if k in ['normal_only','tangent_only','local_control','paired_closed']} for f in ['all','grazing_thin']} for m in out},indent=2))
print('CONTROL MACRO')
cs=read('control_summary.json')
for mode in cs:
    for m in cs[mode]:
        for fam in ['all','grazing_thin']:
            d=cs[mode][m][fam];print(mode,m,fam,json.dumps(d))
print('DRIFT')
for m in out:print(m,json.dumps(out[m]['grazing_thin']['drift']))
print('TEACHER ERROR')
for m in out:print(m,json.dumps(out[m]['grazing_thin']['teacher_at_failure']))
print('CORRELATIONS',json.dumps(cor))

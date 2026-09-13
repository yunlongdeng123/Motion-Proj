"""固定控制后的残差定位；复用同一算法，不调参数。"""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from audit_worldsim_v74_h2_p16 import *

p=Path(sys.argv[1]);start=time.monotonic();torch.set_grad_enabled(False);torch.set_num_threads(4)
names=json.loads((p/'manifest.json').read_text())['cases'];rows=[];newfail=[];recoveries=[];sameparent=[];failures=json.loads((p/'episodes.json').read_text())
for model in MODELS:
    states=[surface(p/'closed_build'/model/n/'step-8.npz') for n in names]
    bo=[load(p/'inputs'/n/'build.npz') for n in names];qo=[load(p/'inputs'/n/'supervision.npz') for n in names]
    full=observations([joined(b,q) for b,q in zip(bo,qo)])
    corrected,log=normal_control(pack(states),full);sol=unpack(corrected)
    for j,name in enumerate(names):
        s=states[j];build,obs=bo[j],qo[j];dest=p/'same_parent_fit_diagnostic'/model/name;dest.mkdir(parents=True,exist_ok=True);sol[j].save(dest/'surface.npz');write(dest/'solver.json',split_logs(log,j))
        tr=trace(s,obs);inf=describe(s,obs,tr);q=tr['first'];y=obs['observed_first_range_m'];pos=positive(obs);lab=labels(q,y)
        rt=load(p/'raw'/model/name/'query_chain-8.npz');rl=labels(rt['first'],y);ct=trace(sol[j],obs);cl=labels(ct['first'],y);bt=trace(s,build);bi=describe(s,build,bt)
        bpos=positive(build);by=build['observed_first_range_m'];bfinite=np.isfinite(bt['first']);bearly=bfinite&(bt['first']<by-.2)
        hist=[surface(p/'closed_build'/model/name/f'step-{t}.npz') for t in range(9)]
        for ray in np.flatnonzero((pos&(lab!=0)) | (np.isfinite(q)&(q<y-.2))):
            k=int(inf['winner'][ray]);winner=None
            if k>=0:
                own=bi['winner']==k;bp=own&bpos;bf=own&bearly
                winner=dict(patch=k,birth_step=next(t for t,st in enumerate(hist) if len(st.center)>k),build_owns_positive=int(bp.sum()),build_owns_prefix_violation=int(bf.sum()),
                    build_any_positive_intersections=int((bi['hit'][:,k]&bpos).sum()),build_own_positive_rms_m=float(np.sqrt(np.mean((bt['first'][bp]-by[bp])**2))) if bp.any() else None,
                    boundary_m=float(inf['boundary'][ray,k]),incidence=float(inf['incidence'][ray,k]))
            rows.append(dict(model=model,case=name,family=name.rsplit('-',1)[0],ray=int(ray),positive=bool(pos[ray]),ambiguous=bool(obs['ambiguous_owner'][ray]),
                raw_label=int(rl[ray]),raw_first_m=rt['first'][ray],raw_intrusion_m=max(float(y[ray]-rt['first'][ray]-.2),0) if np.isfinite(rt['first'][ray]) else 0,
                label=int(lab[ray]),first_m=q[ray],observed_m=y[ray],intrusion_m=max(float(y[ray]-q[ray]-.2),0) if np.isfinite(q[ray]) else 0,
                same_parent_fit_label=int(cl[ray]),same_parent_fit_intrusion_m=max(float(y[ray]-ct['first'][ray]-.2),0) if np.isfinite(ct['first'][ray]) else 0,winner=winner))
        sameparent.append(dict(model=model,case=name,before=metrics(s,obs),after=metrics(sol[j],obs)))
        _,infos,labs=assess_states(hist,obs);rr,es,ds=audit_raw(model,name,hist,obs,infos,labs)
        for r in rr:
            if r['category']=='hit_early':newfail.append(r)
        # 原始 EARLY→HIT 恢复路径的真正几何通道。
        raw=[surface(p/'raw'/model/name/f'step-{t}.npz') for t in range(9)];rawtr=[load(p/'raw'/model/name/f'query_chain-{t}.npz') for t in range(9)];ri=[describe(st,obs,tr) for st,tr in zip(raw,rawtr)]
        for ep in failures:
            if ep['model']!=model or ep['case']!=name or not ep['from_hit'] or not ep['recovered']:continue
            ray=ep['ray'];inc=[]
            for t in range(ep['start_step']+1,ep['first_hit_step']+1):
                k0=int(ri[t-1]['winner'][ray]);k1=int(ri[t]['winner'][ray]);same=k0>=0 and k0==k1
                inc.append(dict(step=t,same_owner=same,winner_before=k0,winner_after=k1,depth_m=ri[t]['first'][ray]-ri[t-1]['first'][ray],decomposition=plane_change(raw[t-1],raw[t],obs,ray,k0) if same else None))
            recoveries.append(dict(model=model,case=name,family=ep['family'],ray=ray,start=ep['start_step'],recovery=ep['first_hit_step'],increments=inc))
write(p/'residual_rays.json',rows);write(p/'closed_new_transitions.json',newfail);write(p/'recovery_channels.json',recoveries);write(p/'same_parent_fit_rows.json',sameparent)
summary={}
for model in MODELS:
    rr=[r for r in rows if r['model']==model];neg=[r for r in rr if not r['positive'] and r['intrusion_m']>0];pos=[r for r in rr if r['positive']]
    recovery=[r for r in recoveries if r['model']==model and r['family']=='grazing_thin']
    summary[model]=dict(final_bad_positive=len(pos),positive_labels={k:sum(r['label']==i for r in pos) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])},
        final_negative_intrusions=len(neg),negative_intrusion_sum_m=sum(r['intrusion_m'] for r in neg),negative_intrusion_max_m=max([r['intrusion_m'] for r in neg],default=0),
        negative_owners_without_build_positive=sum(r['winner']['build_owns_positive']==0 for r in neg),negative_owners_without_build_violation=sum(r['winner']['build_owns_prefix_violation']==0 for r in neg),
        same_parent_fit_bad_positive_labels={k:sum(r['same_parent_fit_label']==i for r in pos) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])},
        same_parent_fit_still_negative_intrusions=sum(r['same_parent_fit_intrusion_m']>0 for r in neg),
        thin_recoveries=len(recovery),thin_recoveries_with_owner_switch=sum(any(not t['same_owner'] for t in r['increments']) for r in recovery),
        new_hit_early_transitions=[r for r in newfail if r['model']==model])
write(p/'residual_summary.json',dict(results=summary,wall_s=time.monotonic()-start,source_note='ordinary plane tangent degeneracy: Open3D Colored ICP; no added color/prior or new solver'))
print(json.dumps(clean(summary),indent=2))
print('NEGATIVE DETAILS',json.dumps(clean([r for r in rows if not r['positive'] and r['intrusion_m']>0]),indent=2))
print('SAME PARENT MACRO')
for m in MODELS:
    for fam in ['all','grazing_thin']:
        rr=[r for r in sameparent if r['model']==m and (fam=='all' or r['case'].startswith(fam))]
        print(m,fam,{side:{k:float(np.mean([r[side][k] for r in rr])) for k in ['hit','early','miss','recall_02','free_m']} for side in ['before','after']})

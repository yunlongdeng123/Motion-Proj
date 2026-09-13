import sys,json
from pathlib import Path
from audit_worldsim_v74_h2_p16 import *
p=Path(sys.argv[1]);eps=json.loads((p/'episodes.json').read_text());rows=[]
for e in eps:
    if not (e['family']=='grazing_thin' and e['from_hit'] and e['recovered']):continue
    m,n,r,t=e['model'],e['case'],e['ray'],e['first_hit_step'];obs=load(p/'inputs'/n/'supervision.npz')
    a=surface(p/'raw'/m/n/f'step-{t-1}.npz');b=surface(p/'raw'/m/n/f'step-{t}.npz');at=trace(a,obs);bt=trace(b,obs);ai=describe(a,obs,at);bi=describe(b,obs,bt);k=int(ai['winner'][r]);new=int(bi['winner'][r]);y=float(obs['observed_first_range_m'][r])
    count=len(a.center);dc=b.center[:count]-a.center;ns=a.rotation[:,:,2];normal=(dc*ns).sum(-1)[:,None]*ns
    variants={}
    st=a.copy();st.center+=normal;variants['normal_only']=st
    st=a.copy();st.center+=dc-normal;variants['tangent_only']=st
    st=a.copy();st.radius=b.radius[:count].copy();variants['radius_only']=st
    st=a.copy();st.center+=dc-normal;st.radius=b.radius[:count].copy();variants['tangent_radius']=st
    results={}
    for typ,st in variants.items():
        tr=trace(st,obs);inf=describe(st,obs,tr);results[typ]=dict(label=int(labels(tr['first'],obs['observed_first_range_m'])[r]),winner=int(inf['winner'][r]),old_intersects=bool(inf['hit'][r,k]))
    rows.append(dict(model=m,case=n,ray=r,recovery_step=t,winner_before=k,winner_after=new,old_after_intersects=bool(bi['hit'][r,k]),old_after_plane_residual_m=float(bi['plane'][r,k]-y),
        old_boundary_before_m=float(ai['boundary'][r,k]),old_boundary_after_m=float(bi['boundary'][r,k]),decomposition=plane_change(a,b,obs,r,k),counterfactuals=results))
write(p/'recovery_detail.json',rows)
summary={}
for m in MODELS:
    rr=[r for r in rows if r['model']==m]
    summary[m]=dict(n=len(rr),owner_switch=sum(r['winner_before']!=r['winner_after'] for r in rr),old_support_exit=sum(not r['old_after_intersects'] for r in rr),
                   old_plane_still_early=sum(r['old_after_plane_residual_m']<-.2 for r in rr),
                   counterfactual_hit={v:sum(r['counterfactuals'][v]['label']==0 for r in rr) for v in ['normal_only','tangent_only','radius_only','tangent_radius']})
write(p/'recovery_detail_summary.json',summary);print(json.dumps(summary,indent=2))

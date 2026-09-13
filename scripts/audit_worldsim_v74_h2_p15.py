"""存量轨迹取证；不训练、不修改模型或推理规则。"""
import argparse, json, sys, time, os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.surface_state import Surface
from motion_proj.worldsim_v74_h2.first_event_trace import trace
from motion_proj.worldsim_v74_h2.gpu_geometry import pack, unpack, observations, geometry, candidate_cost
from motion_proj.worldsim_v74_h2.witness_events import birth_pool, proposals, choose

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def surface(p):
    z=load(p);return Surface(z['center'],z['rotation'],z['radius'])
def labels(q,y):
    return np.where(~np.isfinite(q),3,np.where(q<y-.2,1,np.where(np.abs(q-y)<=.2,0,2)))

def describe(s,obs,tr):
    n=len(tr['first']);k=len(s.center);off=tr['offsets'];winner=np.full(n,-1,int);second=np.full(n,np.inf)
    nonempty=off[1:]>off[:-1];winner[nonempty]=tr['patch'][off[:-1][nonempty]]
    has2=off[1:]-off[:-1]>1;second[has2]=tr['t'][off[:-1][has2]+1]
    o=obs['origins_actor_m'];d=obs['directions_actor'];normal=s.rotation[:,:,2]
    den=d@normal.T;safe=np.where(den!=0,den,1)
    t=np.einsum('npc,pc->np',s.center[None]-o[:,None],normal)/safe
    point=o[:,None]+t[:,:,None]*d[:,None]
    uv=np.einsum('npc,pcj->npj',point-s.center[None],s.rotation[:,:,:2])
    a=np.arange(8)*np.pi/4;ring=s.radius[:,:,None]*np.stack([np.cos(a),np.sin(a)],-1)
    edge=np.roll(ring,-1,axis=1)-ring;diff=uv[:,:,None]-ring[None]
    u=np.clip(np.sum(diff*edge[None],-1)/np.sum(edge**2,-1)[None],0,1)
    boundary=np.linalg.norm(diff-u[:,:,:,None]*edge[None],axis=-1).min(-1)
    hit=np.zeros((n,k),bool);hit[tr['ray'],tr['patch']]=True
    valid=(den!=0)&(t>0)&np.isfinite(t)
    signed=np.where(hit,-boundary,boundary)
    depth=np.full((n,k),np.inf);depth[tr['ray'],tr['patch']]=tr['t']
    return dict(first=tr['first'],winner=winner,second=second,depth=depth,plane=t,incidence=np.abs(den),boundary=boundary,signed=signed,valid=valid,hit=hit)

def snapshot(info,ray,patch):
    if patch<0 or patch>=info['plane'].shape[1]:return None
    return dict(patch=int(patch),plane_t_m=info['plane'][ray,patch],valid_plane=info['valid'][ray,patch],intersects=info['hit'][ray,patch],boundary_m=info['boundary'][ray,patch],signed_boundary_m=info['signed'][ray,patch],incidence=info['incidence'][ray,patch])

def transition_row(model,name,step,ray,prev,nxt,y,is_first,after_label):
    old=int(prev['winner'][ray]);new=int(nxt['winner'][ray]);pcount=prev['plane'].shape[1]
    kind='same_owner_depth' if old==new else ('new_patch_owner' if new>=pcount else 'existing_patch_owner_switch')
    # 有效旧交点链的第二交点；单交点时 margin 缺失，不填零。
    gap=prev['second'][ray]-prev['first'][ray]
    return dict(model=model,case=name,family=name.rsplit('-',1)[0],step=step,ray=int(ray),first_failure=bool(is_first),after_label=int(after_label),kind=kind,
                winner_before=old,winner_after=new,observed_m=float(y[ray]),before_m=prev['first'][ray],after_m=nxt['first'][ray],first_hit_margin_m=gap,
                old_before=snapshot(prev,ray,old),new_before=snapshot(prev,ray,new),new_after=snapshot(nxt,ray,new),old_after=snapshot(nxt,ray,old),
                hit_threshold_slack_m=prev['first'][ray]-(y[ray]-.2))

def stats(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(n=len(a),min=float(a.min()) if len(a) else None,q10=float(np.quantile(a,.1)) if len(a) else None,median=float(np.median(a)) if len(a) else None,q90=float(np.quantile(a,.9)) if len(a) else None,max=float(a.max()) if len(a) else None)
def margins(rows):
    specs=[('first_hit_margin_m',lambda r:r['first_hit_margin_m'],[.001,.01,.05]),
           ('old_boundary_m',lambda r:r['old_before']['boundary_m'],[.001,.005,.01]),
           ('old_incidence',lambda r:r['old_before']['incidence'],[.01,.05,.1]),
           ('incoming_pre_boundary_m',lambda r:r['new_before']['boundary_m'] if r['new_before'] else None,[.001,.005,.01]),
           ('incoming_pre_incidence',lambda r:r['new_before']['incidence'] if r['new_before'] else None,[.01,.05,.1]),
           ('after_boundary_m',lambda r:r['new_after']['boundary_m'],[.001,.005,.01]),
           ('after_incidence',lambda r:r['new_after']['incidence'],[.01,.05,.1])]
    out={}
    for key,fn,cuts in specs:
        a=[fn(r) for r in rows];finite=[v for v in a if v is not None and np.isfinite(v)]
        out[key]={**stats(a),'missing':len(a)-len(finite),'counts_le':{str(c):sum(v<=c for v in finite) for c in cuts}}
    return out

def interpolate(start,end,alphas):
    t=np.asarray(alphas);rel=Rotation.from_matrix(end.rotation@np.swapaxes(start.rotation,1,2)).as_rotvec()
    rot=Rotation.from_rotvec((t[:,None,None]*rel[None]).reshape(-1,3)).as_matrix().reshape(len(t),len(rel),3,3)@start.rotation[None]
    c=start.center[None]+t[:,None,None]*(end.center-start.center)[None]
    rho=np.exp(np.log(start.radius)[None]+t[:,None,None]*(np.log(end.radius)-np.log(start.radius))[None])
    return [Surface(c[i],rot[i],rho[i]) for i in range(len(t))]

def query_many(states,obs):
    results=[];who=[]
    with torch.no_grad():
        for lo in range(0,len(states),128):
            group=states[lo:lo+128];g=geometry(pack(group),observations([obs]*len(group)))
            q=g['first'].cpu().numpy();win=g['winner'].cpu().numpy();win[~np.isfinite(q)]=-1
            results.append(q);who.append(win)
    return np.concatenate(results),np.concatenate(who)

def delta_sizes(start,end,alpha,patch):
    if alpha is None:return None
    now=interpolate(start,end,[alpha])[0];v0=start.mesh()[0].reshape(-1,9,3);v1=now.mesh()[0].reshape(-1,9,3)
    d=np.linalg.norm(v1-v0,axis=-1);rot=Rotation.from_matrix(now.rotation@np.swapaxes(start.rotation,1,2)).magnitude()*180/np.pi
    return dict(max_center_mm=float(np.linalg.norm(now.center-start.center,axis=1).max()*1000),max_outer_vertex_mm=float(d[:,1:].max()*1000),max_rotation_deg=float(rot.max()),
                winner_center_mm=float(np.linalg.norm(now.center[patch]-start.center[patch])*1000) if patch>=0 else None,
                winner_outer_vertex_mm=float(d[patch,1:].max()*1000) if patch>=0 else None,winner_rotation_deg=float(rot[patch]) if patch>=0 else None)

def scan(start,end,obs,rays,dest,scope):
    alphas=np.unique(np.r_[np.linspace(0,1,1001),.0001,.0003]);states=interpolate(start,end,alphas)
    q,win=query_many(states,obs);lab=labels(q,obs['observed_first_range_m'][None])
    np.savez_compressed(dest,alpha=alphas,first_m=q,winner=win,label=lab,rays=rays)
    out=[]
    for ray in rays:
        ray=int(ray);record=dict(ray=ray,scope=scope,reference_label=int(lab[0,ray]),learner_label=int(lab[-1,ray]),reference_winner=int(win[0,ray]),learner_winner=int(win[-1,ray]),
                               grid_owner_changes=int(np.count_nonzero(win[1:,ray]!=win[:-1,ray])),grid_label_changes=int(np.count_nonzero(lab[1:,ray]!=lab[:-1,ray])))
        for kind in ['owner','early']:
            # 首次相对起始首面的归属变化，或首次 EARLY。
            flag=(win[:,ray]!=win[0,ray]) if kind=='owner' else lab[:,ray]==1
            indices=np.flatnonzero(flag)
            if not len(indices):record[kind]=None;continue
            i=int(indices[0]);lo=float(alphas[max(i-1,0)]);hi=float(alphas[i])
            if i:
                for _ in range(16):
                    mid=(lo+hi)/2;mq,mw=query_many(interpolate(start,end,[mid]),obs)
                    on=(mw[0,ray]!=win[0,ray]) if kind=='owner' else labels(mq[0],obs['observed_first_range_m'])[ray]==1
                    if on:hi=mid
                    else:lo=mid
            # 极窄转移附近以原 CPU 三角形查询核对 bracket，避免平面 fan 数值差影响归因。
            cpu=[]
            for st in interpolate(start,end,[lo,hi]):
                ct=trace(st,obs);cf=ct['first'][ray];idx=ct['offsets'][ray];cw=int(ct['patch'][idx]) if np.isfinite(cf) else -1
                cpu.append(dict(first_m=cf,winner=cw,label=int(labels(np.array([cf]),obs['observed_first_range_m'][ray:ray+1])[0])))
            record[kind]=dict(alpha_lo=lo,alpha_hi=hi,already_at_reference=(i==0),cpu_bracket=cpu,displacement=delta_sizes(start,end,hi,int(win[-1,ray])))
        record['endpoint_error']=delta_sizes(start,end,1.,int(win[-1,ray]));out.append(record)
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();torch.set_num_threads(4);out=args.output;out.mkdir(parents=True,exist_ok=False)
    models={'A':'20260912__A-dagger12-r1','C2':'20260912__C2-dagger12-r1'}
    teacher=args.source/'20260912__fit-teacher-r3'/'fit';names=sorted(p.name for p in (args.source/models['A']).iterdir() if p.is_dir())
    manifest=dict(task='WS-V74-H2-P15-01',run='switching-margin-r1',source=str(args.source),output=str(out),cases=names,models=models,training=False,epsilon_m=.2,role='exposed FIT diagnostics',alpha_grid='1001 uniform plus .0001/.0003; first observed bracket bisected 16 times',failure_ledger_refs=['V74-H2-F03','V74-H2-F05','V74-H2-F08'])
    write(out/'manifest.json',manifest);failures=[];stable=[];caserows=[];data={};gpu_errors=[];interpolations=[];references=[]
    for model,run in models.items():
        for name in names:
            folder=args.source/run/name;obs=load(teacher/name/'supervision.npz');build=load(teacher/name/'build.npz');states=[surface(folder/f'step-{i}.npz') for i in range(9)]
            trs=[load(folder/f'query_chain-{i}.npz') for i in range(9)];infos=[describe(s,obs,tr) for s,tr in zip(states,trs)]
            labels_all=np.stack([labels(t['first'],obs['observed_first_range_m']) for t in trs]);positive=obs['positive_actor']&~obs['ambiguous_owner'];seen=set();nf=0
            q,w=query_many(states,obs)
            for i,(gg,cc) in enumerate(zip(q,trs)):
                common=np.isfinite(gg)&np.isfinite(cc['first']);refwin=infos[i]['winner']
                gpu_errors.append(dict(model=model,case=name,step=i,miss_disagreement=int(np.count_nonzero(np.isfinite(gg)!=np.isfinite(cc['first']))),owner_disagreement=int(np.count_nonzero(w[i]!=refwin)),label_disagreement=int(np.count_nonzero(labels(gg,obs['observed_first_range_m'])!=labels_all[i])),max_depth_difference_m=float(np.max(np.abs(gg[common]-cc['first'][common]))) if common.any() else 0.))
            for step in range(1,9):
                for ray in np.flatnonzero(positive&(labels_all[step-1]==0)&(labels_all[step]==1)):
                    row=transition_row(model,name,step,ray,infos[step-1],infos[step],obs['observed_first_range_m'],ray not in seen,1);failures.append(row);seen.add(ray);nf+=1
                for ray in np.flatnonzero(positive&(labels_all[step-1]==0)&(labels_all[step]==0)):
                    stable.append(transition_row(model,name,step,ray,infos[step-1],infos[step],obs['observed_first_range_m'],False,0))
            case=dict(model=model,case=name,positive_rays=int(positive.sum()),all_hit_early_transitions=nf,first_hit_early_rays=len(seen),final_early_rays=int(np.count_nonzero(positive&(labels_all[-1]==1))),
                      labels_by_step=[{k:int(np.count_nonzero(positive&(x==i))) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])} for x in labels_all])
            caserows.append(case);data[model,name]=(states,obs,build,infos,labels_all)
    write(out/'transitions.json',failures);write(out/'stable_hit_transitions.json',stable);write(out/'case_counts.json',caserows);write(out/'saved_geometry_comparison.json',gpu_errors)
    print(json.dumps({'stage':'extracted','failures':len(failures),'first_rays':sum(r['first_failure'] for r in failures),'stable':len(stable),'wall_s':time.monotonic()-start}),flush=True)
    # 仅在出现首次退化的父状态重算原有限候选教师；不引入新的几何提议器。
    groups=sorted(set((r['model'],r['case'],r['step']) for r in failures if r['first_failure']))
    for model,name,step in groups:
        states,obs,build,infos,labs=data[model,name];parent=states[step-1];learner=states[step]
        rays=np.array([r['ray'] for r in failures if r['first_failure'] and (r['model'],r['case'],r['step'])==(model,name,step)])
        dest=out/model/name/f'step-{step}';dest.mkdir(parents=True);r=pack([parent]);bo=observations([build]);su=observations([obs]);full={k:torch.cat([bo[k],su[k]],1) for k in bo}
        with torch.no_grad():
            birth,_=birth_pool([build]);pool=proposals(r,bo,birth,step-1);costs=candidate_cost(pool['states'],full).masked_fill(~pool['valid'],float('inf'));best=int(costs.argmin(-1).item());teach=unpack(choose(pool,torch.tensor([best],device='cuda')))[0]
        logs=torch.load(args.source/models[model]/name/'events.pt',map_location='cpu',weights_only=False);le=int(logs[step-1]['event'])
        tk=int(pool['kind'][best]);lk=int(pool['kind'][le]);ti=int(pool['index'][best]);li=int(pool['index'][le]);same_topology=(tk==0 and lk==0) or (tk==lk and ti==li)
        assert same_topology==(len(teach.center)==len(learner.center) and (tk==0 or (tk==lk and ti==li)))
        parent.save(dest/'parent.npz');learner.save(dest/'learner.npz');teach.save(dest/'teacher.npz')
        ttrace=trace(teach,obs);tlab=labels(ttrace['first'],obs['observed_first_range_m']);tinfo=describe(teach,obs,ttrace)
        np.savez_compressed(dest/'teacher_query.npz',**ttrace)
        # 拓扑不同：只用教师的共同父片替代 learner 父片，新片固定为 learner 新片。
        ref=teach
        scope='matched_teacher_to_learner'
        if not same_topology:
            ref=learner.copy();np0=len(parent.center);ref.center[:np0]=teach.center[:np0];ref.rotation[:np0]=teach.rotation[:np0];ref.radius[:np0]=teach.radius[:np0];scope='conditional_existing_patches_learner_topology'
        ref.save(dest/'continuous_reference.npz')
        candidate_names=pool['names'];rec=dict(model=model,case=name,step=step,rays=rays,teacher_event=best,learner_event=le,teacher_event_name=candidate_names[best],learner_event_name=candidate_names[le],teacher_kind=tk,learner_kind=lk,teacher_index=ti,learner_index=li,
                   parent_patches=len(parent.center),teacher_patches=len(teach.center),learner_patches=len(learner.center),same_topology=same_topology,scope=scope,
                   teacher_cost=float(costs[0,best]),keep_cost=float(costs[0,0]),raw_delta_mse_active=float(np.mean((logs[step-1]['delta'][:len(parent.center)].numpy()-pool['delta'][0,best,:len(parent.center)].cpu().numpy())**2)),
                   teacher_labels={str(ray):int(tlab[ray]) for ray in rays},teacher_first_margin={str(ray):float(tinfo['second'][ray]-tinfo['first'][ray]) for ray in rays},
                   teacher_winner_margin={str(ray):snapshot(tinfo,ray,int(tinfo['winner'][ray])) for ray in rays},
                   learner_candidate_scores=logs[step-1]['score'].numpy(),teacher_candidate_cost=costs[0].cpu().numpy(),valid=pool['valid'][0].cpu().numpy())
        references.append(rec);write(dest/'reference.json',rec)
        scans=scan(ref,learner,obs,rays,dest/'teacher_learner_scan.npz',scope)
        for x in scans:x.update(model=model,case=name,step=step,teacher_label=int(tlab[x['ray']]),same_topology=same_topology);interpolations.append(x)
        write(dest/'teacher_learner_crossings.json',scans)
        if len(parent.center)==len(learner.center):
            actual=scan(parent,learner,obs,rays,dest/'actual_update_scan.npz','actual_parent_to_learner')
            write(dest/'actual_update_crossings.json',actual)
        print(json.dumps({'stage':'scanned','model':model,'case':name,'step':step,'rays':len(rays),'same_topology':same_topology,'wall_s':time.monotonic()-start}),flush=True)
    write(out/'references.json',references);write(out/'interpolations.json',interpolations)
    summary={}
    for model in models:
        summary[model]={}
        for family in ['all','multilayer','shared_support','missing_support','grazing_thin']:
            rows=[r for r in failures if r['model']==model and (family=='all' or r['family']==family)];first=[r for r in rows if r['first_failure']];sr=[r for r in stable if r['model']==model and (family=='all' or r['family']==family)]
            counts={kind:sum(r['kind']==kind for r in first) for kind in ['same_owner_depth','existing_patch_owner_switch','new_patch_owner']}
            paired=[]
            for r in first:
                other='C2' if model=='A' else 'A';ol=data[other,r['case']][4];paired.append(int(ol[r['step'],r['ray']]))
                r['paired_other_label_after']=int(ol[r['step'],r['ray']]);r['paired_other_label_before']=int(ol[r['step']-1,r['ray']])
            summary[model][family]=dict(first_ray_count=len(first),all_transition_count=len(rows),stable_hit_transition_count=len(sr),first_failure_types=counts,first_failure_margins=margins(first),stable_margins=margins(sr),
                                      paired_other_labels={k:paired.count(i) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])})
    write(out/'transitions.json',failures)
    ip=[]
    for model in models:
        rr=[r for r in interpolations if r['model']==model];eligible=[r for r in rr if r['same_topology'] and r['teacher_label']==0 and r['reference_label']==0]
        ip.append(dict(model=model,first_failure_rays=len(rr),matched_topology=sum(r['same_topology'] for r in rr),matched_teacher_hit_rays=len(eligible),teacher_labels={k:sum(r['teacher_label']==i for r in rr) for i,k in enumerate(['HIT','EARLY','LATE','MISS'])},
                       eligible_owner_switch_rays=sum(r['owner'] is not None for r in eligible),eligible_early_alpha=stats([r['early']['alpha_hi'] if r['early'] else None for r in eligible]),
                       eligible_owner_alpha=stats([r['owner']['alpha_hi'] if r['owner'] else None for r in eligible]),eligible_endpoint_outer_vertex_mm=stats([r['endpoint_error']['max_outer_vertex_mm'] for r in eligible])))
    completion=dict(status='DONE',task=manifest['task'],run=manifest['run'],wall_s=time.monotonic()-start,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,cases=caserows,summary=summary,interpolation_summary=ip,
                    saved_geometry_comparison={'states':len(gpu_errors),'miss_disagreement':sum(r['miss_disagreement'] for r in gpu_errors),'owner_disagreement':sum(r['owner_disagreement'] for r in gpu_errors),'label_disagreement':sum(r['label_disagreement'] for r in gpu_errors),'max_depth_difference_m':max(r['max_depth_difference_m'] for r in gpu_errors)},training=False)
    write(out/'summary.json',completion);print(json.dumps(clean({k:v for k,v in completion.items() if k not in ['summary','cases']})),flush=True)
if __name__=='__main__':main()

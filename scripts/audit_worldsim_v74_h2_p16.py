"""P1.6 存量状态病因诊断与普通物理最小二乘；冻结所有网络。"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from audit_worldsim_v74_h2_p15 import load,surface,describe,labels,write,stats,clean
from motion_proj.worldsim_v74_h2.surface_state import Surface
from motion_proj.worldsim_v74_h2.first_event_trace import trace,metrics
from motion_proj.worldsim_v74_h2.synthetic import surface_quality
from motion_proj.worldsim_v74_h2.gpu_geometry import pack,unpack,observations,geometry,features,candidate_cost
from motion_proj.worldsim_v74_h2.witness_events import birth_pool,proposals,choose,learned_step
from motion_proj.worldsim_v74_h2.witness_dynamics import WitnessDynamics

MODELS={'A':'20260912__A-dagger12-r1','C2':'20260912__C2-dagger12-r1'}
FAMILIES=['multilayer','shared_support','missing_support','grazing_thin']
METRICS=['hit','early','miss','late','any_correct','free_intrusion_rate','free_m','recall_02','distance_m','area_m2','faces']

def physical_metric(s,obs,g=None):
    g=geometry(s,obs) if g is None else g
    finite=torch.isfinite(g['first']);r=torch.where(finite,g['first']-obs['y'],torch.zeros_like(obs['y']))
    pos=obs['positive'];free=torch.where(finite,(-r-.2).clamp_min(0),torch.zeros_like(r))
    return (torch.where(finite,r.square(),torch.full_like(r,4.))*pos).sum(-1)/pos.sum(-1).clamp_min(1)+free.square().mean(-1)

@torch.no_grad()
def normal_control(s,obs):
    """普通对角 Gauss–Newton + 有界位移 + 固定预算 Armijo。无逐射线门槛。"""
    s={k:v.clone() for k,v in s.items()};history=[]
    for it in range(8):
        g=geometry(s,obs);e=physical_metric(s,obs,g);finite=torch.isfinite(g['first']);win=g['winner']
        den=g['den'].gather(-1,win[...,None]).squeeze(-1)
        jac=torch.where(finite,1/torch.where(den!=0,den,torch.ones_like(den)),torch.zeros_like(den))
        r=torch.where(finite,g['first']-obs['y'],torch.zeros_like(obs['y']))
        wpos=obs['positive']*finite/obs['positive'].sum(-1,keepdim=True).clamp_min(1)
        wfree=(finite&(r<-.2))/r.shape[1]
        gr=torch.zeros_like(s['active'],dtype=s['c'].dtype);h=gr.clone()
        gr.scatter_add_(1,win,jac*(wpos*r+wfree*(r+.2)))
        h.scatter_add_(1,win,jac.square()*(wpos+wfree))
        step=torch.where(h>0,-gr/torch.where(h>0,h,torch.ones_like(h)),torch.zeros_like(h)).clamp(-.75,.75)
        step*=s['active'];slope=2*(gr*step).sum(-1)
        accepted=torch.zeros_like(e,dtype=torch.bool);alpha=torch.zeros_like(e);after=e.clone()
        moved=s['c'].clone();tries=torch.zeros_like(e,dtype=torch.long)
        for j in range(8):
            a=2.**-j;trial={**s,'c':s['c']+a*step[...,None]*s['rot'][...,:,2]}
            en=physical_metric(trial,obs);take=(~accepted)&(en<=e+1e-4*a*slope)
            moved[take]=trial['c'][take];after[take]=en[take];alpha[take]=a;tries[~accepted]+=1;accepted|=take
            if accepted.all():break
        history.append(dict(iteration=it,before=e.cpu().numpy(),after=after.cpu().numpy(),alpha=alpha.cpu().numpy(),normal_step_m=step.cpu().numpy(),line_search_trials=tries.cpu().numpy()))
        s={**s,'c':moved}
    return s,history

def split_logs(log,j):return [{k:(v[j] if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in log]
def joined(a,b):return {k:np.concatenate([a[k],b[k]],0) for k in a}
def positive(o):return o['positive_actor']&~o['ambiguous_owner']

def plane_change(a,b,o,ray,k):
    """相同片的精确根差分解；不把无限平面当有限面命中。"""
    n0=a.rotation[k,:,2];n1=b.rotation[k,:,2];dc=b.center[k]-a.center[k]
    d=o['directions_actor'][ray];origin=o['origins_actor_m'][ray];den0=n0@d;den1=n1@d
    if den0==0 or den1==0:return None
    t0=n0@(a.center[k]-origin)/den0;t1=n1@(b.center[k]-origin)/den1
    normal=float(n0@dc);tangent=dc-normal*n0
    cn=float(n1@(normal*n0)/den1);ct=float(n1@tangent/den1)
    rot=float((n1-n0)@(a.center[k]-origin-t0*d)/den1)
    return dict(normal_m=normal,tangent_m=float(np.linalg.norm(tangent)),center_m=float(np.linalg.norm(dc)),incidence_before=float(abs(den0)),incidence_after=float(abs(den1)),
                signed_den_before=float(den0),condition_gain=float(1/abs(den0)),normal_depth_m=cn,tangent_depth_m=ct,rotation_depth_m=rot,plane_depth_m=float(t1-t0),
                fixed_normal_prediction_m=normal/den0,identity_error_m=float(abs(cn+ct+rot-(t1-t0))),rotation_deg=float(Rotation.from_matrix(b.rotation[k]@a.rotation[k].T).magnitude()*180/np.pi),
                log_radius_rmse=float(np.sqrt(np.mean(np.log(b.radius[k]/a.radius[k])**2))))

def assess_states(states,obs):
    trs=[trace(s,obs) for s in states];infos=[describe(s,obs,tr) for s,tr in zip(states,trs)]
    labs=np.stack([labels(t['first'],obs['observed_first_range_m']) for t in trs])
    return trs,infos,labs

def summarize_rows(rows):
    out={}
    for model in MODELS:
        out[model]={}
        for fam in ['all']+FAMILIES:
            rr=[r for r in rows if r['model']==model and (fam=='all' or r['family']==fam)]
            out[model][fam]={}
            for category in ['hit_early','hit_hit','other']:
                sel=[r for r in rr if r['category']==category]
                keys=['normal_m','tangent_m','center_m','condition_gain','normal_depth_m','tangent_depth_m','rotation_depth_m','identity_error_m','rotation_deg']
                out[model][fam][category]=dict(n=len(sel),same_owner=sum(r['same_owner'] for r in sel),stats={k:stats([r['decomposition'][k] for r in sel if r['decomposition']]) for k in keys})
    return out

def audit_raw(model,name,states,obs,infos,labs):
    rows=[];episodes=[];drifts=[];y=obs['observed_first_range_m'];pos=positive(obs)
    for t in range(1,9):
        for ray in np.flatnonzero(pos):
            k0=int(infos[t-1]['winner'][ray]);k1=int(infos[t]['winner'][ray]);same=k0>=0 and k0==k1
            dec=plane_change(states[t-1],states[t],obs,ray,k1) if 0<=k1<len(states[t-1].center) else None
            q0=float(infos[t-1]['first'][ray]);q1=float(infos[t]['first'][ray])
            category='hit_early' if labs[t-1,ray]==0 and labs[t,ray]==1 else ('hit_hit' if labs[t-1,ray]==0 and labs[t,ray]==0 else 'other')
            rows.append(dict(model=model,case=name,family=name.rsplit('-',1)[0],step=t,ray=int(ray),category=category,same_owner=same,winner_before=k0,winner_after=k1,
                             label_before=int(labs[t-1,ray]),label_after=int(labs[t,ray]),residual_before_m=q0-y[ray],residual_after_m=q1-y[ray],actual_depth_change_m=q1-q0 if np.isfinite(q0+q1) else None,decomposition=dec))
    for ray in np.flatnonzero(pos):
        seq=labs[:,ray]
        for t in range(9):
            if seq[t]!=1 or (t and seq[t-1]==1):continue
            leave=next((j for j in range(t+1,9) if seq[j]!=1),None)
            recover=next((j for j in range(t+1,9) if seq[j]==0),None)
            ep=dict(model=model,case=name,family=name.rsplit('-',1)[0],ray=int(ray),start_step=t,from_hit=t>0 and seq[t-1]==0,initial_early=t==0,
                    first_non_early_step=leave,first_hit_step=recover,recovered=recover is not None,right_censored=recover is None,
                    observed_early_states=(leave if leave is not None else 9)-t,steps_to_hit=recover-t if recover is not None else None,final_label=int(seq[-1]),
                    residual_by_step_m=[i['first'][ray]-y[ray] for i in infos],winner_by_step=[int(i['winner'][ray]) for i in infos])
            episodes.append(ep)
            if not ep['from_hit']:continue
            start=t-1
            while start>0 and seq[start-1]==0:start-=1
            increments=[]
            for j in range(start+1,t+1):
                k=int(infos[j-1]['winner'][ray]);same=k>=0 and k==int(infos[j]['winner'][ray])
                dec=plane_change(states[j-1],states[j],obs,ray,k) if same else None
                increments.append(dict(step=j,same_owner=same,depth_m=infos[j]['first'][ray]-infos[j-1]['first'][ray],decomposition=dec))
            inc=np.array([z['depth_m'] for z in increments]);total=float(inc.sum());last=float(inc[-1])
            drifts.append(dict(model=model,case=name,family=ep['family'],ray=int(ray),failure_step=t,hit_run_start=start,hit_run_steps=t-start,
                               initial_residual_m=infos[start]['first'][ray]-y[ray],failure_residual_m=infos[t]['first'][ray]-y[ray],total_depth_m=total,last_depth_m=last,
                               last_share_of_total=abs(last)/abs(total) if total!=0 else None,earlier_advance_m=float(-inc[:-1].sum()),
                               negative_steps=int((inc<0).sum()),positive_steps=int((inc>0).sum()),cancellation_ratio=float(abs(total)/np.abs(inc).sum()) if np.abs(inc).sum()>0 else None,
                               normal_sum_m=sum(z['decomposition']['normal_depth_m'] for z in increments if z['decomposition']),
                               tangent_sum_m=sum(z['decomposition']['tangent_depth_m'] for z in increments if z['decomposition']),
                               rotation_sum_m=sum(z['decomposition']['rotation_depth_m'] for z in increments if z['decomposition']),
                               switch_sum_m=sum(z['depth_m'] for z in increments if not z['same_owner']),increments=increments))
    return rows,episodes,drifts

def episode_summary(episodes,drifts):
    out={}
    for model in MODELS:
        out[model]={}
        for fam in ['all']+FAMILIES:
            es=[r for r in episodes if r['model']==model and (fam=='all' or r['family']==fam)];fromhit=[r for r in es if r['from_hit']]
            ds=[r for r in drifts if r['model']==model and (fam=='all' or r['family']==fam)]
            out[model][fam]=dict(all_early_episodes=len(es),initial_early=sum(r['initial_early'] for r in es),from_hit=len(fromhit),from_hit_recovered=sum(r['recovered'] for r in fromhit),
                                from_hit_right_censored=sum(r['right_censored'] for r in fromhit),recovery_steps=stats([r['steps_to_hit'] for r in fromhit]),
                                drift={k:stats([r[k] for r in ds]) for k in ['hit_run_steps','initial_residual_m','total_depth_m','last_depth_m','last_share_of_total','earlier_advance_m','normal_sum_m','switch_sum_m']})
    return out

def aggregate_control(rows):
    out={}
    for mode in sorted(set(r['mode'] for r in rows)):
        out[mode]={}
        for model in MODELS:
            out[mode][model]={}
            for fam in ['all']+FAMILIES:
                rr=[r for r in rows if r['mode']==mode and r['model']==model and (fam=='all' or r['family']==fam)]
                out[mode][model][fam]=dict(cases=len(rr),final={k:float(np.mean([r['query'][k] for r in rr])) if rr else None for k in METRICS},
                    hit_early_transitions=sum(r['hit_early_transitions'] for r in rr),any_post_step_early_positive_observations=sum(r['post_step_early_observations'] for r in rr),
                    final_early_rays=sum(r['final_early_rays'] for r in rr),final_hit_rays=sum(r['final_hit_rays'] for r in rr),positive_rays=sum(r['query']['positive_rays'] for r in rr))
    return out

def state_record(mode,model,name,hist,obs,build,truth,dest):
    dest.mkdir(parents=True,exist_ok=True);tr,info,lab=assess_states(hist,obs);trajectory=[]
    for step,s in enumerate(hist):
        s.save(dest/f'step-{step}.npz');np.savez_compressed(dest/f'query_chain-{step}.npz',**tr[step]);trajectory.append(metrics(s,obs))
    pos=positive(obs)
    row=dict(mode=mode,model=model,case=name,family=name.rsplit('-',1)[0],trajectory=trajectory,query=trajectory[-1],build=metrics(hist[-1],build),**surface_quality(hist[-1],truth),
             hit_early_transitions=int(((lab[:-1]==0)&(lab[1:]==1)&pos[None]).sum()),post_step_early_observations=int(((lab[1:]==1)&pos[None]).sum()),
             final_early_rays=int(((lab[-1]==1)&pos).sum()),final_hit_rays=int(((lab[-1]==0)&pos).sum()))
    write(dest/'results.json',row);return row,lab

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--p15',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.monotonic();out=a.output;out.mkdir(parents=True,exist_ok=False)
    data=a.source/'20260912__fit-teacher-r3'/'fit';names=sorted(p.name for p in (a.source/MODELS['A']).iterdir() if p.is_dir())
    builds={n:load(data/n/'build.npz') for n in names};obs={n:load(data/n/'supervision.npz') for n in names};truth={n:surface(data/n/'truth.npz') for n in names}
    write(out/'manifest.json',dict(task='WS-V74-H2-P16-01',run=out.name,status='RUNNING',training=False,source=str(a.source),p15=str(a.p15),models=MODELS,cases=names,
                                  role='exposed 12 FIT tasks; BUILD-only control; FIT evaluation; full FIT control diagnostic only',iterations=8,backtracking_trials=8,armijo=1e-4,epsilon_m=.2,
                                  failure_ledger_refs=['V74-H2-F08','V74-H2-F09']))
    for n in names:
        (out/'inputs'/n).mkdir(parents=True)
        for f in ['initial.npz','truth.npz','build.npz','supervision.npz']:shutil.copy2(data/n/f,out/'inputs'/n/f)
    raw={};allrows=[];episodes=[];drifts=[];parents=[];controls=[];labels_saved={};local=[];teacher_rows=[];teacher_parents=[]
    for model,run in MODELS.items():
        for name in names:
            folder=a.source/run/name;states=[surface(folder/f'step-{i}.npz') for i in range(9)];trs=[load(folder/f'query_chain-{i}.npz') for i in range(9)]
            infos=[describe(s,obs[name],tr) for s,tr in zip(states,trs)];labs=np.stack([labels(t['first'],obs[name]['observed_first_range_m']) for t in trs]);raw[model,name]=(states,infos,labs)
            rr,es,ds=audit_raw(model,name,states,obs[name],infos,labs);allrows.extend(rr);episodes.extend(es);drifts.extend(ds)
            row,lab=state_record('raw',model,name,states,obs[name],builds[name],truth[name],out/'raw'/model/name);controls.append(row);labels_saved['raw',model,name]=lab
            parents.extend((model,name,t) for t in range(1,9))
    write(out/'ray_transitions.json',allrows);write(out/'episodes.json',episodes);write(out/'drifts.json',drifts)
    write(out/'decomposition_summary.json',summarize_rows(allrows));write(out/'recovery_summary.json',episode_summary(episodes,drifts))
    print(json.dumps(dict(stage='raw_diagnosis',transitions=len(allrows),episodes=len(episodes),wall_s=time.monotonic()-start)),flush=True)
    # 所有原始父状态上同池重标注，防止只挑失败点制造误差结论。
    for lo in range(0,len(parents),12):
        group=parents[lo:lo+12];ss=[raw[m,n][0][t-1] for m,n,t in group];bo=[builds[n] for m,n,t in group];su=[obs[n] for m,n,t in group]
        s=pack(ss);b=observations(bo);full=observations([joined(x,y) for x,y in zip(bo,su)]);birth,_=birth_pool(bo)
        # 各任务 step 不同；保留原步号的候选几何与排序。
        teachers=[];meta=[]
        for j,(m,n,t) in enumerate(group):
            pool=proposals({k:v[j:j+1] for k,v in s.items()},{k:v[j:j+1] for k,v in b.items()},{k:v[j:j+1] for k,v in birth.items()},t-1)
            costs=candidate_cost(pool['states'],{k:v[j:j+1] for k,v in full.items()}).masked_fill(~pool['valid'],float('inf'));idx=costs.argmin(-1)
            ts=unpack(choose(pool,idx))[0];teachers.append(ts);meta.append(dict(event=int(idx.item()),name=pool['names'][int(idx.item())],kind=int(pool['kind'][idx].item()),cost=float(costs.min())))
        for j,(m,n,t) in enumerate(group):
            teacher=teachers[j];learner=raw[m,n][0][t];parent=raw[m,n][0][t-1];info=raw[m,n][1][t]
            dest=out/'teacher'/m/n/f'step-{t}';dest.mkdir(parents=True);teacher.save(dest/'surface.npz')
            tr=trace(teacher,obs[n]);ti=describe(teacher,obs[n],tr);tl=labels(tr['first'],obs[n]['observed_first_range_m']);np.savez_compressed(dest/'query.npz',**tr)
            teacher_parents.append(dict(model=m,case=n,step=t,parent_patches=len(parent.center),learner_patches=len(learner.center),teacher_patches=len(teacher.center),**meta[j]))
            for ray in np.flatnonzero(positive(obs[n])):
                k=int(info['winner'][ray]);dec=plane_change(teacher,learner,obs[n],ray,k) if 0<=k<min(len(parent.center),len(teacher.center)) else None
                good=dec is not None and ti['hit'][ray,k]
                teacher_rows.append(dict(model=m,case=n,family=n.rsplit('-',1)[0],step=t,ray=int(ray),learner_label=int(raw[m,n][2][t,ray]),teacher_label=int(tl[ray]),
                    learner_winner=k,teacher_winner=int(ti['winner'][ray]),teacher_has_learner_patch_hit=bool(good),decomposition=dec,
                    first_depth_error_m=info['first'][ray]-ti['first'][ray] if np.isfinite(info['first'][ray]+ti['first'][ray]) else None,
                    teacher_patch_to_first_gap_m=ti['plane'][ray,k]-ti['first'][ray] if dec is not None and np.isfinite(ti['first'][ray]) else None))
    write(out/'teacher_error_rows.json',teacher_rows);write(out/'teacher_parents.json',teacher_parents)
    # 在 P15 70 条转移上分离实际动作的法向与切向，所有其他参数保持父状态。
    transitions=json.loads((a.p15/'transitions.json').read_text());cf=[]
    for m,n,t in sorted(set((r['model'],r['case'],r['step']) for r in transitions)):
        parent,learner=raw[m,n][0][t-1:t+1];count=len(parent.center);dc=learner.center[:count]-parent.center;norm=parent.rotation[:,:,2];dn=(dc*norm).sum(-1)[:,None]*norm
        variants={'normal_only':dn,'tangent_only':dc-dn}
        for typ,delta in variants.items():
            st=parent.copy();st.center+=delta;tr=trace(st,obs[n]);info=describe(st,obs[n],tr);lab=labels(tr['first'],obs[n]['observed_first_range_m']);dest=out/'counterfactual'/m/n/f'step-{t}';dest.mkdir(parents=True,exist_ok=True);st.save(dest/f'{typ}.npz')
            for r in transitions:
                if (r['model'],r['case'],r['step'])==(m,n,t):
                    ray=r['ray'];cf.append(dict(model=m,case=n,family=r['family'],step=t,ray=ray,variant=typ,label=int(lab[ray]),first_m=tr['first'][ray],winner=int(info['winner'][ray])))
    write(out/'normal_tangent_counterfactual.json',cf)
    print(json.dumps(dict(stage='teacher_and_components',teacher_parents=len(teacher_parents),wall_s=time.monotonic()-start)),flush=True)
    # 同一普通控制用于所有 192 后状态；BUILD 主控制与监督信息诊断严格分目录。
    for mode in ['local_build','local_fit_diagnostic']:
        for lo in range(0,len(parents),12):
            group=parents[lo:lo+12];ss=[raw[m,n][0][t] for m,n,t in group]
            inputs=[builds[n] if mode=='local_build' else joined(builds[n],obs[n]) for m,n,t in group]
            corrected,log=normal_control(pack(ss),observations(inputs));sol=unpack(corrected)
            for j,(m,n,t) in enumerate(group):
                dest=out/mode/m/n/f'step-{t}';dest.mkdir(parents=True);sol[j].save(dest/'surface.npz');write(dest/'solver.json',split_logs(log,j));tr=trace(sol[j],obs[n]);np.savez_compressed(dest/'query.npz',**tr)
                lab=labels(tr['first'],obs[n]['observed_first_range_m']);base=raw[m,n][2][t];pos=positive(obs[n]);firstfail=[r['ray'] for r in transitions if (r['model'],r['case'],r['step'])==(m,n,t)]
                local.append(dict(mode=mode,model=m,case=n,family=n.rsplit('-',1)[0],step=t,query=metrics(sol[j],obs[n]),
                    p15_failure_rays=len(firstfail),p15_failure_labels={str(k):int((lab[firstfail]==k).sum()) for k in range(4)},
                    raw_hit_to_nonhit=int((pos&(base==0)&(lab!=0)).sum()),raw_early_to_hit=int((pos&(base==1)&(lab==0)).sum())))
    write(out/'local_control_rows.json',local)
    # 真闭环：模型每步读取已校正的自身状态，不把原动作序列当反馈策略。
    first_proposal_comparison=[]
    for model in MODELS:
        ckpath=a.source/'20260912__dagger12-w32-r1'/f'{model}_best.pt';ck=torch.load(ckpath,map_location='cuda',weights_only=False)
        net=WitnessDynamics(ck['width'],ck['ordered']).cuda();net.load_state_dict(ck['model']);net.eval();(out/'checkpoints').mkdir(exist_ok=True);shutil.copy2(ckpath,out/'checkpoints'/ckpath.name)
        for mode in ['closed_build','closed_fit_diagnostic']:
            # 监督诊断仍然让网络只读 BUILD；只有普通控制额外读取 FIT。
            bo=[builds[n] for n in names];b=observations(bo);inp=b if mode=='closed_build' else observations([joined(builds[n],obs[n]) for n in names]);s=pack([raw[model,n][0][0] for n in names]);birth,_=birth_pool(bo);hidden=None;hist=[unpack(s)];logs=[]
            for t in range(8):
                pool=proposals(s,b,birth,t);f={**features(s,b),'evidence':pool['evidence'],'candidate':pool['desc'],'all_delta':pool['delta'].float()}
                delta,score,hidden=net(f,hidden);event=score.masked_fill(~pool['valid'],1e4).argmin(-1);proposal=learned_step(s,delta,pool,event)
                pre=unpack(proposal)
                if t==0:
                    for j,n in enumerate(names):
                        target=raw[model,n][0][1];first_proposal_comparison.append(dict(model=model,mode=mode,case=n,same_count=len(target.center)==len(pre[j].center),max_center_difference_m=float(np.max(np.abs(target.center-pre[j].center))) if len(target.center)==len(pre[j].center) else None))
                s,log=normal_control(proposal,inp);hist.append(unpack(s))
                for j,n in enumerate(names):
                    dest=out/mode/model/n;dest.mkdir(parents=True,exist_ok=True);pre[j].save(dest/f'proposal-{t+1}.npz');write(dest/f'solver-{t+1}.json',split_logs(log,j))
                logs.append(dict(delta=delta.cpu(),event=event.cpu(),score=score.cpu(),hidden=hidden.cpu()))
            for j,n in enumerate(names):
                dest=out/mode/model/n;row,lab=state_record(mode,model,n,[x[j] for x in hist],obs[n],builds[n],truth[n],dest);controls.append(row);labels_saved[mode,model,n]=lab
                torch.save([{k:v[j] for k,v in log.items()} for log in logs],dest/'events.pt')
            print(json.dumps(dict(stage=mode,model=model,wall_s=time.monotonic()-start)),flush=True)
    write(out/'control_rows.json',controls);write(out/'control_summary.json',aggregate_control(controls));write(out/'first_proposal_comparison.json',first_proposal_comparison)
    # 对原失败 ray/step 配对查看闭环修正，连同新生退化一起报告。
    paired=[]
    for r in transitions:
        m,n,t,ray=r['model'],r['case'],r['step'],r['ray']
        paired.append(dict(model=m,case=n,family=r['family'],step=t,ray=ray,labels={mode:int(labels_saved[mode,m,n][t,ray]) for mode in ['raw','closed_build','closed_fit_diagnostic']}))
    write(out/'paired_failures.json',paired)
    # 单次科学实现核对：有限面旋转/平移的解析恒等式与保存的硬求交。
    same=[r for r in allrows if r['same_owner'] and r['decomposition'] and r['actual_depth_change_m'] is not None]
    err=max(abs(r['actual_depth_change_m']-r['decomposition']['plane_depth_m']) for r in same)
    write(out/'completion.json',dict(status='DONE',training=False,wall_s=time.monotonic()-start,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
        raw_ray_transitions=len(allrows),teacher_parents=len(teacher_parents),teacher_ray_rows=len(teacher_rows),local_control_states=len(local),closed_rollouts=48,
        same_owner_finite_transitions=len(same),max_analytic_hard_difference_m=err,normal_control='BUILD deployment arm; BUILD+FIT diagnostic arm',failure_ledger_delta='pending interpretation'))
    print(json.dumps(clean(json.loads((out/'completion.json').read_text()))),flush=True)

if __name__=='__main__':main()

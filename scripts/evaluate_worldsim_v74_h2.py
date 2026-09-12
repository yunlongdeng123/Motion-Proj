"""GPU共享表面八步闭环；独立CPU三角查询记账，不在推理中读监督。"""
import argparse,json,sys,time,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.surface_state import Surface
from motion_proj.worldsim_v74_h2.synthetic import surface_quality
from motion_proj.worldsim_v74_h2.first_event_trace import trace,metrics
from motion_proj.worldsim_v74_h2.gpu_geometry import pack,unpack,observations,geometry,features,candidate_cost
from motion_proj.worldsim_v74_h2.witness_events import birth_pool,proposals,learned_step
from motion_proj.worldsim_v74_h2.witness_dynamics import WitnessDynamics

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_surface(p):
    with np.load(p) as z:return Surface(z['center'],z['rotation'],z['radius'])
def load_obs(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=['fit','fit_val','mechanism'],default='mechanism');p.add_argument('--per-family',type=int,default=20);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    ck=torch.load(a.model,map_location='cuda',weights_only=False);model=WitnessDynamics(ck['width'],ck['ordered']).cuda();model.load_state_dict(ck['model']);model.eval()
    folders=[p for p in sorted((a.data/a.role).iterdir()) if p.is_dir() and int(p.name.rsplit('-',1)[-1])<a.per_family];rows=[]
    write(a.output/'manifest.json',{'task':'WS-V74-H2-GPU-ROLLOUT-01','model':str(a.model),'role':a.role,'cases':len(folders),'steps':8,
         'forward_access':'BUILD plus current surface; no QUERY/supervision values','candidate_geometry_limit':64,'human_verdict':None,'failure_ledger_delta':'pending'})
    for lo in range(0,len(folders),8):
        group=folders[lo:lo+8];initial=[load_surface(p/'initial.npz') for p in group];builds=[load_obs(p/'build.npz') for p in group]
        obs=observations(builds);s=pack(initial);birth,counts=birth_pool(builds);hidden=None;hist=[unpack(s)];logs=[]
        with torch.no_grad():
            for step in range(8):
                pool=proposals(s,obs,birth,step);f={**features(s,obs),'evidence':pool['evidence'],'candidate':pool['desc'],'all_delta':pool['delta'].float()}
                delta,score,hidden=model(f,hidden);event=score.masked_fill(~pool['valid'],1e4).argmin(-1)
                before=geometry(s,obs);actual=candidate_cost(pool['states'],obs)
                logs.append({'event':event.cpu(),'delta':delta.cpu(),'score':score.cpu(),'candidate_cost_build':actual.cpu(),
                             'full_chain_t':before['depth'].cpu(),'witness_evidence':pool['evidence'].cpu(),'valid':pool['valid'].cpu()})
                s=learned_step(s,delta,pool,event);hist.append(unpack(s))
        for j,folder in enumerate(group):
            dest=a.output/folder.name;dest.mkdir();query=load_obs(folder/('query.npz' if a.role=='mechanism' else 'supervision.npz'));truth=load_surface(folder/'truth.npz')
            trajectory=[];first_deg=None;prev=metrics(initial[j],query)
            for step,states in enumerate(hist):
                state=states[j];state.save(dest/f'step-{step}.npz');q=metrics(state,query);trajectory.append(q)
                if step and first_deg is None and (q['hit']<prev['hit']-.005 or q['early']>prev['early']+.005):first_deg=step
                prev=q
                np.savez_compressed(dest/f'query_chain-{step}.npz',**trace(state,query))
            final=hist[-1][j];torch.save([{k:v[j] for k,v in log.items()} for log in logs],dest/'events.pt')
            row={'case':folder.name,'family':folder.name.rsplit('-',1)[0],'build':metrics(final,builds[j]),'query':trajectory[-1],
                 'initial_query':trajectory[0],'trajectory':trajectory,'first_degradation':first_deg,**surface_quality(final,truth)}
            rows.append(row);write(dest/'results.json',row)
        print(json.dumps({'completed':lo+len(group),'total':len(folders),'wall_s':time.monotonic()-start}),flush=True)
    write(a.output/'results.json',rows)
    summary=[]
    for f in ['multilayer','shared_support','missing_support','grazing_thin']:
        rr=[r for r in rows if r['family']==f];summary.append({'family':f,'cases':len(rr),
            'query':{k:float(np.mean([r['query'][k] for r in rr])) for k in ['hit','early','miss','recall_02','any_correct','free_m']}})
    write(a.output/'summary.json',{'rows':summary,'wall_s':time.monotonic()-start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'status':'done','human_verdict':None})
if __name__=='__main__':main()

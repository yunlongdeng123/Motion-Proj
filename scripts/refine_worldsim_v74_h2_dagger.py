"""一次明确的优化修正：在学习器访问的共享几何状态上重标注同池硬事件。"""
import argparse,sys,json,time,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v74_h2.witness_dynamics import WitnessDynamics,learning_loss
from motion_proj.worldsim_v74_h2.gpu_geometry import pack,observations,features,candidate_cost,cost
from motion_proj.worldsim_v74_h2.witness_events import birth_pool,proposals,learned_step
from evaluate_worldsim_v74_h2 import load_surface,load_obs
from train_worldsim_v74_h2_surface import load,FIELDS,assess,batch_step

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def data_rows(root,role,per):
    out=[]
    for p in sorted((root/role).iterdir()):
        if p.is_dir() and int(p.name.rsplit('-',1)[-1])<per:
            out.append({'name':p.name,'initial':load_surface(p/'initial.npz'),'build':load_obs(p/'build.npz'),'supervision':load_obs(p/'supervision.npz')})
    return out

def visit(model,rows,collect=False):
    batches=[];costs=[]
    model.eval()
    with torch.no_grad():
        for lo in range(0,len(rows),16):
            group=rows[lo:lo+16];s=pack([r['initial'] for r in group]);obs=observations([r['build'] for r in group]);sup=observations([r['supervision'] for r in group]);full={k:torch.cat([obs[k],sup[k]],1) for k in obs}
            birth,_=birth_pool([r['build'] for r in group]);hidden=None;sequence=[];history=[]
            for step in range(8):
                pool=proposals(s,obs,birth,step);f={**features(s,obs),'evidence':pool['evidence'],'candidate':pool['desc'],'all_delta':pool['delta'].float(),'valid':pool['valid']}
                delta,score,hidden=model(f,hidden);event=score.masked_fill(~pool['valid'],1e4).argmin(-1)
                if collect:
                    target=candidate_cost(pool['states'],full).masked_fill(~pool['valid'],float('inf'));best=target.argmin(-1)
                    relative=target-target[:,:1];relative=torch.where(torch.isfinite(relative),relative,torch.zeros_like(relative))
                    f.update(target_score=relative.float(),target_event=best,target_delta=pool['delta'][torch.arange(len(group),device='cuda'),best].float())
                    sequence.append({k:v.cpu() for k,v in f.items()});history.append({k:v.cpu() for k,v in s.items()})
                s=learned_step(s,delta,pool,event)
            costs.extend(cost(s,full).cpu().tolist())
            if collect:batches.append({'offset':lo,'cases':[r['name'] for r in group],'sequence':sequence,'history':history})
    return batches,float(np.mean(costs))

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--initial-models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--memorize12',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();per=3 if a.memorize12 else 20
    rows=data_rows(a.teacher,'fit',per);validation=rows if a.memorize12 else data_rows(a.teacher,'fit_val',5)
    expert=load(a.teacher/'teacher.pt',3 if a.memorize12 else None)
    manifest={'task':'WS-V74-H2-GPU-OPTIMIZATION-01','correction':'DAgger-style learner-state dataset aggregation, two fixed rounds, 200 epochs each; same network/features/event pool/loss',
        'source':'https://proceedings.mlr.press/v15/ross11a.html','initial_models':str(a.initial_models),'memorization_only':a.memorize12,
        'selection':'mean hard closed-loop FIT validation objective' if not a.memorize12 else 'mean hard closed-loop 12-task training objective',
        'failure_ledger_refs':['V74-H2-F08'],'failure_ledger_delta':'pending','human_verdict':None,'seed':7411,'batch_episodes':32}
    write(a.output/'manifest.json',manifest);summaries=[]
    for name in ['A','C2']:
        torch.manual_seed(7411);ck=torch.load(a.initial_models/f'{name}_best.pt',map_location='cuda',weights_only=False)
        model=WitnessDynamics(ck['width'],ck['ordered']).cuda();model.load_state_dict(ck['model']);opt=torch.optim.Adam(model.parameters(),lr=.001)
        data={k:v for k,v in expert.items()};curve=[];_,best=visit(model,validation);initial_cost=best
        torch.save(ck,a.output/f'{name}_best.pt')
        for round_id in range(2):
            visits,onpolicy=visit(model,rows,True);path=a.output/f'{name}_round-{round_id}_visited.pt';torch.save(visits,path)
            extra=load(path);data={k:torch.cat([data[k],extra[k]],0) for k in FIELDS};n=len(data['node'])
            for epoch in range(200):
                model.train();ids=torch.randperm(n,device='cuda')
                for chunk in ids.split(32):
                    opt.zero_grad();hidden=None;loss=0
                    for step in range(8):
                        f=batch_step(data,chunk,step);delta,score,hidden=model(f,hidden);value,_=learning_loss(delta,score,f);loss+=value/8
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                if epoch%50==0 or epoch==199:
                    model.eval();tr=assess(model,data);_,v=visit(model,validation)
                    row={'model':name,'round':round_id,'epoch':epoch,'aggregated_fit_scenes':n,'train':tr,'closed_loop_selection_cost':v,'wall_s':time.monotonic()-start}
                    curve.append(row);print(json.dumps(row),flush=True);write(a.output/f'{name}_curve.json',curve)
                    payload={'model':model.state_dict(),'width':ck['width'],'ordered':ck['ordered'],'epoch':epoch,'round':round_id,'config':manifest}
                    if v<best:best=v;torch.save(payload,a.output/f'{name}_best.pt')
                    if epoch==199:torch.save(payload,a.output/f'{name}_round-{round_id}_last.pt')
            # 下一轮访问当前最佳闭环策略；不读取机制集或真实DEV。
            model.load_state_dict(torch.load(a.output/f'{name}_best.pt',map_location='cuda',weights_only=False)['model'])
        summaries.append({'model':name,'initial_closed_loop_cost':initial_cost,'best_closed_loop_cost':best,'width':ck['width'],'parameters':sum(x.numel() for x in model.parameters())})
        write(a.output/'summary.json',{'models':summaries,'wall_s':time.monotonic()-start,'status':'done' if name=='C2' else 'running','human_verdict':None})
if __name__=='__main__':main()

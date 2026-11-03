"""仅拟合持久车辆RGB；不更新几何、道路、时间、opacity或原模型。

delete中的原a6严格可见像素仍有合法原RGB；原a37像素/未知下部不作RGB真值。
无编辑保留原路径是一项明确工程约束，不记作重建算法改进。
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse,json,time,random,copy,shutil
import torch,numpy as np
from vehicle_runtime import VehicleExperiment
from editor_common import save

RUN=Path(__file__).resolve().parent.parent

def write(path,obj):
    p=Path(path);temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,p)

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    dest=RUN/args.output;dest.mkdir(exist_ok=False)
    for name in ['fit_vehicle.py','vehicle_runtime.py']:shutil.copy2(RUN/'source'/name,dest/name)
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0)
    cfg=dict(actor_subset='all',legacy_replacement='native',colour_source='observed_initial',road_policy='delete_only',
             completion_state='delete_only',replacement_corridor=True,corridor_replace_whole_a6=True,corridor_road_protect_m=.20,
             protected_static_file=str(RUN/'inputs/retained_static_protection.npz'))
    e=VehicleExperiment(cfg);actor=e.old.actor
    actor.requires_grad_(False);actor.colour_logit.requires_grad_(True)
    opt=torch.optim.Adam([actor.colour_logit],lr=.035,eps=1e-15)
    raws=[r for r in range(89,138) if r not in (101,113,125)]
    geometry={k:v.detach().cpu().clone() for k,v in actor.state_dict().items() if k!='colour_logit'}
    initial_colour=actor.initial_colour.detach().clone();known=actor.colour_known
    startstep=0;beg=time.monotonic()
    def state(step):
        return dict(step=step,actor=actor.state_dict(),optimizer=opt.state_dict(),sampler=sampler.getstate(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            parent_identity=e.n.before,config=cfg,trainable='canonical RGB only',fixed='geometry/normals/scales/opacity/road/parent/time/PVD')
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert ck['config']==cfg and ck['parent_identity']==e.n.before
        actor.load_state_dict(ck['actor'],strict=True);opt.load_state_dict(ck['optimizer']);sampler.setstate(ck['sampler'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        startstep=ck['step'];assert same(ck,state(startstep))
        write(dest/'restore.json',dict(exact_full_state=True,step=startstep,optimizer_and_rng=True))
    def evaluate(step):
        folder=dest/('step%06d'%step);folder.mkdir(exist_ok=True);metrics=[]
        with torch.no_grad():
            for raw in (105,113,130):
                out=e.render(raw,False);v=e.old.view(raw);vis=torch.as_tensor(v['visible_evidence_mask'],device='cuda',dtype=torch.bool)
                labels=torch.as_tensor(v['actor_labels'],device='cuda');unc=torch.as_tensor(v['unknown_mask'],device='cuda',dtype=torch.bool)
                save(folder/f'{raw}_delete.png',out['rgb']);save(folder/f'{raw}_lower_alpha.png',out['component_alpha_T'][5:6].repeat(3,1,1))
                metrics.append(dict(raw=raw,a6_strict_visible_MAE=float((out['rgb']-e.old.truth(raw)).abs()[:,vis].mean()),
                    unknown_actor_alpha_T=float(out['actor_alpha'][0,unc].mean()),unknown_lower_alpha_T=float(out['component_alpha_T'][5,unc].mean())))
        write(folder/'metrics.json',metrics);return metrics
    write(dest/'contract.json',dict(config=cfg,steps=args.steps,resume=str(args.resume) if args.resume else None,smoke=args.smoke,
         optimizer_raws=raws,all_BUILD=True,independent_test=False,hidden_RGB_supervision=False,new_prior=False,new_road=False,
         noedit_policy='Native original path by explicit delete-conditioned completion design; not an always-enabled reconstructed-scene improvement',
         alpha_policy='all opacity and geometry fixed; inherited parent untouched; new RGB loss does not update alpha or remove parent opacity supervision',
         loss='delete strict a6 visible RGB L1 +0.002 known-colour MSE; no target a37/unknown RGB treated as GT',human_verdict=None))
    evaluate(startstep)
    steps_done=startstep
    with (dest/'training.jsonl').open('x') as log:
        for step in range(startstep+1,args.steps+1):
            if (RUN/'STOP_REQUESTED').exists():break
            raw=sampler.choice(raws);out=e.render(raw,False)
            mask=torch.as_tensor(e.old.view(raw)['visible_evidence_mask'],device='cuda',dtype=torch.bool)
            photo=(out['rgb']-e.old.truth(raw)).abs()[:,mask].mean()
            prior=(actor.colors()[known]-initial_colour[known]).square().mean()
            loss=photo+.002*prior;assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();assert actor.colour_logit.grad is not None and torch.isfinite(actor.colour_logit.grad).all()
            opt.step();steps_done=step
            rec=dict(step=step,raw=raw,loss=float(loss.detach()),photo=float(photo.detach()),seconds=time.monotonic()-beg)
            log.write(json.dumps(rec)+'\n');log.flush()
            if step==startstep+1 or step%100==0:print(json.dumps(rec),flush=True)
            if step in (32,100,500,1000) or step==args.steps:
                torch.save(state(step),dest/f'checkpoint{step:06d}.pt')
            if step==args.steps:evaluate(step)
    if steps_done<args.steps:torch.save(state(steps_done),dest/'checkpoint_stop.pt')
    fixed=all(torch.equal(v,actor.state_dict()[k].detach().cpu()) for k,v in geometry.items())
    assert fixed and e.unchanged()
    write(dest/'receipt.json',dict(step=steps_done,updates=steps_done-startstep,complete=steps_done==args.steps,
        seconds=time.monotonic()-beg,geometry_fixed=fixed,parent_unchanged=e.unchanged(),quality_accepted=False,human_verdict=None))
    print(json.dumps(dict(stage='COMPLETE',step=steps_done,seconds=time.monotonic()-beg,geometry_fixed=fixed)),flush=True)

if __name__=='__main__':main()

"""Inherited 1000-step static-opacity handoff with frozen road and vehicle surfaces."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time,random,argparse
import numpy as np
from editable_scene import ROOT,Scene
def linear(x):
    import torch
    x=x.clamp(0,1);return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def main():
    import torch
    from utils.v4_road_render import raster
    p=argparse.ArgumentParser();p.add_argument('--output',default='road_gates');p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/a.output;out.mkdir(exist_ok=False)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0);scene=Scene(with_road=True);N=len(scene.sids);score=torch.zeros(N,device='cuda');guard=torch.zeros_like(score);frames=[];records=[];start=time.monotonic()
    for raw in scene.raws:
        d=np.load(ROOT/'world_road/views'/f'{raw:03d}.npz');fit=(d['fit']>.05)&~d['marking'];protection=d['other']|d['marking']
        if not fit.any():continue
        values,ids,cam=scene.rows(raw);features=values['features'].detach().clone().requires_grad_(True);values=dict(values,features=features,mask=values['mask']&(ids!=10));package=raster(cam,scene.pipe,0,values,torch.zeros(3,device='cuda'));mass=package['extra'][4];mask=torch.tensor(fit,device='cuda');other=torch.tensor(protection,device='cuda');s=torch.autograd.grad(mass[mask].sum(),features,retain_graph=True)[0][:N,6];g=torch.autograd.grad(mass[other].sum(),features)[0][:N,6];score=torch.maximum(score,s);guard=torch.maximum(guard,g);frames.append(raw)
        del values,features,package,s,g
    eligible=(score>1e-4)&~(guard>20*score)&(scene.sids==0);indices=torch.nonzero(eligible).flatten();np.savez_compressed(out/'contributors.npz',ids=indices.cpu().numpy(),fit_score=score[eligible].cpu().numpy(),guard_score=guard[eligible].cpu().numpy());
    if not len(indices):
        (out/'receipt.json').write_text(json.dumps(dict(status='NO_ELIGIBLE_STATIC_OPACITY_ROWS',steps=0,only_gate_trainable=True)));return
    gate=torch.nn.Parameter(torch.full((len(indices),),2.,device='cuda'));opt=torch.optim.Adam([gate],lr=.06,eps=1e-15);scene.gate_ids=indices;startstep=0
    def state(step):return dict(step=step,gate=gate.detach().clone(),optimizer=opt.state_dict(),sampler=sampler.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    if a.resume:
        z=torch.load(str(a.resume),map_location='cuda',weights_only=False);gate.data.copy_(z['gate']);opt.load_state_dict(z['optimizer']);sampler.setstate(z['sampler']);torch.set_rng_state(z['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in z['cuda_rng']]);np.random.set_state(z['numpy_rng']);random.setstate(z['python_rng']);startstep=z['step']
        sys_path=ROOT.parent/'scene008_native_road1k_20261105_r1/source'
        import sys;sys.path.insert(0,str(sys_path));from compare_state import differences
        diff=differences(z,state(startstep));assert not diff,diff;(out/'restore.json').write_text(json.dumps(dict(exact_full_state=True,differences=diff,step=startstep)))
    for step in range(startstep+1,a.steps+1):
        raw=sampler.choice(frames);d=np.load(ROOT/'world_road/views'/f'{raw:03d}.npz');scene.gates=gate.sigmoid();opt.zero_grad(set_to_none=True);pack=scene.render(raw,True);pred=linear(pack['rgb']);fit=torch.tensor(d['fit'],device='cuda');keep=torch.tensor(d['keep'],device='cuda');target=torch.tensor(d['reference_linear'],device='cuda');base=torch.tensor(np.load(ROOT/'semantic_all_eval'/f'{raw:03d}/delete.npz')['rgb'],device='cuda')
        photo=((pred-target).abs()*fit).sum()/(3*fit.sum().clamp_min(1));preserve=((pack['rgb']-base).abs()*keep).sum()/(3*keep.sum().clamp_min(1));m=torch.tensor(d['marking'],device='cuda');o=torch.tensor(d['other'],device='cuda');lane=(pack['rgb']-base).abs()[:,m].mean() if m.any() else photo*0;vehicle=(pack['rgb']-base).abs()[:,o].mean() if o.any() else photo*0;left=torch.tensor(~d['inside']&~d['other'],device='cuda');left_keep=(pack['rgb']-base).abs()[:,left].mean() if left.any() else photo*0;loss=photo+2*preserve+4*lane+3*vehicle+2*left_keep+.0005*(1-gate.sigmoid()).mean();assert torch.isfinite(loss);loss.backward();assert torch.isfinite(gate.grad).all();opt.step()
        rec=dict(step=step,raw=raw,loss=float(loss),road=float(photo),preserve=float(preserve),vehicle=float(vehicle),seconds=time.monotonic()-start);records.append(rec)
        if step%100==0 or step==startstep+1:print(json.dumps(rec),flush=True)
        if step in [32,500,1000] or step==a.steps:torch.save(state(step),out/f'checkpoint{step:06d}.pt')
    assert scene.unchanged();(out/'training.json').write_text(json.dumps(records));(out/'receipt.json').write_text(json.dumps(dict(status='PASS_FROZEN_STATIC_GATE_FIT',steps=a.steps,updates=a.steps-startstep,selected_static_rows=len(indices),fit_frames=frames,parent_unchanged=True,only_gate_trainable=True,road_colour_geometry_fixed=True,holdout_used=False,quality_accepted=None,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main()

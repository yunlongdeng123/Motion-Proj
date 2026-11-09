"""Inherited frozen-surface, 1000-step static-opacity handoff, shared across scenes.

Only explicitly selected static opacity multipliers are optimized. Native model
parameters, ownership, surfaces and road colour remain unchanged. This fitting
entry is separate from the read-only V44Scene.render query API.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np


def _differences(a,b,path=''):
    import torch
    if torch.is_tensor(a):return [] if torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu()) else [path]
    if isinstance(a,np.ndarray):return [] if np.array_equal(a,b) else [path]
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path]
        return [x for k in a for x in _differences(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):return [path]
        return [x for i,(u,v) in enumerate(zip(a,b)) for x in _differences(u,v,path+'/'+str(i))]
    return [] if a==b else [path]


def fit(manifest,views,baseline,output,delete_tracks,steps=1000,resume=None):
    import torch
    from .v44_factory import from_manifest
    scene,m=from_manifest(manifest);deleted=m.actor_ids(delete_tracks)
    if not deleted or m.static_gates is not None:raise ValueError('Explicit deletion and an ungated scene are required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);views=Path(views);baseline=Path(baseline)
    # The file layout is an explicit dataset adapter, not parsing image names.
    entries=json.loads((views/'manifest.json').read_text())['frames']
    view_by_key={r['key']:views/r['view'] for r in entries}
    base_by_key={r['key']:baseline/r['baseline'] for r in entries}
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
        return h.hexdigest()
    source_binding=dict(scene_manifest=digest(manifest),views_manifest=digest(views/'manifest.json'),
        delete_actor_ids=list(deleted),algorithm_sha256=digest(__file__),
        sources=[dict(key=r['key'],view=digest(view_by_key[r['key']]),baseline=digest(base_by_key[r['key']])) for r in entries])
    torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0)
    N=len(scene.static.get_xyz);device=scene.static.get_xyz.device
    score=torch.zeros(N,device=device);guard=torch.zeros_like(score);frames=[];records=[];started=time.monotonic()
    ops=scene._bindings()
    for frame in m.frames:
        if frame.key not in view_by_key:raise ValueError('Missing declared road-view data')
        d=np.load(view_by_key[frame.key]);fit_mask=(d['fit']>.05)&~d['marking'];protect=d['other']|d['marking']
        if not fit_mask.any():continue
        values,owners,cam,_=scene._rows(frame.key,(),(),False)
        features=values['features'].detach().clone();features[:,8]=0;features[:N,8]=1;features.requires_grad_(True)
        mask=values['mask']&~torch.isin(owners,owners.new_tensor(deleted))
        pack=ops.raster(cam,scene.pipe,0,dict(values,features=features,mask=mask),features.new_zeros(3))
        mass=pack['extra'][6]
        s=torch.autograd.grad(mass[torch.as_tensor(fit_mask,device=device)].sum(),features,retain_graph=True)[0][:N,8]
        g=torch.autograd.grad(mass[torch.as_tensor(protect,device=device)].sum(),features)[0][:N,8]
        score=torch.maximum(score,s);guard=torch.maximum(guard,g);frames.append(frame.key)
        del pack,values,features,s,g
    eligible=(score>1e-4)&~(guard>20*score)&(scene._owners_s<=0)
    indices=torch.nonzero(eligible).flatten()
    np.savez_compressed(out/'contributors.npz',ids=indices.cpu().numpy(),fit_score=score[eligible].cpu().numpy(),guard_score=guard[eligible].cpu().numpy())
    if not len(indices):
        (out/'receipt.json').write_text(json.dumps(dict(status='NO_ELIGIBLE_STATIC_OPACITY_ROWS',steps=0)))
        return
    gate=torch.nn.Parameter(torch.full((len(indices),),2.,device=device));opt=torch.optim.Adam([gate],lr=.06,eps=1e-15);startstep=0
    def state(step):return dict(step=step,gate=gate.detach().clone(),contributors=indices.detach().clone(),
        fit_frames=frames,source_binding=source_binding,optimizer=opt.state_dict(),sampler=sampler.getstate(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    if resume is not None:
        z=torch.load(resume,map_location=device,weights_only=False)
        if (not torch.equal(z['contributors'],indices) or z['fit_frames']!=frames
                or z['source_binding']!=source_binding):raise RuntimeError('Gate restore changes contributor identity or source binding')
        gate.data.copy_(z['gate']);opt.load_state_dict(z['optimizer']);sampler.setstate(z['sampler']);torch.set_rng_state(z['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in z['cuda_rng']]);np.random.set_state(z['numpy_rng']);random.setstate(z['python_rng']);startstep=z['step']
        diff=_differences(z,state(startstep))
        if diff:raise RuntimeError('Gate full-state restore mismatch: '+str(diff))
        (out/'restore.json').write_text(json.dumps(dict(exact_full_state=True,differences=diff,step=startstep)))
    def linear(x):
        x=x.clamp(0,1);return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    for step in range(startstep+1,steps+1):
        key=sampler.choice(frames);d=np.load(view_by_key[key]);opt.zero_grad(set_to_none=True)
        values,owners,cam,_=scene._rows(key,(),(),False)
        opacity=values['opacities'].index_copy(0,indices,values['opacities'][indices]*gate.sigmoid()[:,None])
        mask=values['mask']&~torch.isin(owners,owners.new_tensor(deleted))
        pack=ops.raster(cam,scene.pipe,0,dict(values,opacities=opacity,mask=mask),opacity.new_zeros(3))
        rgb=pack['rgb']+(1-pack['alpha'])*scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        pred=linear(rgb);weight=torch.as_tensor(d['fit'],device=device);keep=torch.as_tensor(d['keep'],device=device);target=torch.as_tensor(d['reference_linear'],device=device)
        base=torch.as_tensor(np.load(base_by_key[key])['rgb'],device=device)
        photo=((pred-target).abs()*weight).sum()/(3*weight.sum().clamp_min(1));preserve=((rgb-base).abs()*keep).sum()/(3*keep.sum().clamp_min(1))
        marking=torch.as_tensor(d['marking'],device=device);other=torch.as_tensor(d['other'],device=device);left=torch.as_tensor(~d['inside']&~d['other'],device=device)
        lane=(rgb-base).abs()[:,marking].mean() if marking.any() else photo*0
        vehicle=(rgb-base).abs()[:,other].mean() if other.any() else photo*0
        left_keep=(rgb-base).abs()[:,left].mean() if left.any() else photo*0
        loss=photo+2*preserve+4*lane+3*vehicle+2*left_keep+.0005*(1-gate.sigmoid()).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite gate loss')
        loss.backward()
        if not torch.isfinite(gate.grad).all():raise RuntimeError('Nonfinite gate gradient')
        opt.step();record=dict(step=step,key=key,loss=float(loss),road=float(photo),preserve=float(preserve),vehicle=float(vehicle),seconds=time.monotonic()-started);records.append(record)
        if step%100==0 or step==startstep+1:print(json.dumps(record),flush=True)
        if step in [32,500,1000] or step==steps:torch.save(state(step),out/f'checkpoint{step:06d}.pt')
    scene.assert_persistent_unchanged()
    (out/'training.json').write_text(json.dumps(records))
    (out/'receipt.json').write_text(json.dumps(dict(status='PASS_FROZEN_STATIC_GATE_FIT',steps=steps,updates=steps-startstep,selected_static_rows=len(indices),fit_frames=frames,parent_unchanged=True,only_gate_trainable=True,road_colour_geometry_fixed=True,delete_tracks=delete_tracks,seconds=time.monotonic()-started),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','views','baseline','output']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--delete-track-id',action='append',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume',type=Path)
    a=p.parse_args();fit(a.manifest,a.views,a.baseline,a.output,a.delete_track_id,a.steps,a.resume)

if __name__=='__main__':main()

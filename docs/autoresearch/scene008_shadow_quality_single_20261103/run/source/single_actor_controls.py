"""Fresh single-actor R4 controls; same original B, 199 evidence and 1k recipe."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
import r4_runtime as r4
from editor_common import Native,prepare_rasterization,raster,cpu

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.with_name('shadow_a5_a37_199_20261103_r4')

def prepare(actor,run):
    dest=run/'inputs';dest.mkdir(parents=True,exist_ok=False);(dest/'views').mkdir()
    r=dict(np.load(OLD/'inputs/road_reference.npz'));original=r['active'].copy();domain=r['domain_a%d'%actor].copy()
    r['active']&=domain;r['domain']=domain;np.savez_compressed(dest/'road_reference.npz',**r)
    for raw in range(89,138):
        d=dict(np.load(OLD/'inputs/views'/('%03d.npz'%raw)));uid=d['unit_id'];safe=np.maximum(uid,0)
        supported=(uid>=0)&~d['protect_mask']&~d['other_vehicle_guard']
        repair=supported&domain[safe]&r['active'][safe]
        # The parent joint coherent-domain mask remains the geometry authority.
        coherent=(uid>=0)&domain[safe]&~d['protect_mask']
        d['repair_mask']=repair;d['protect_mask']=~coherent|d['other_vehicle_guard'];d['fit_weight']=r['fit_weight'][safe]*repair
        d['inferred']&=repair;d['weak_measured']&=repair
        np.savez_compressed(dest/'views'/('%03d.npz'%raw),**d)
    n=Native();N=len(n.static.get_xyz);score=torch.zeros(N,device='cuda')
    for raw in range(89,138):
        c=n.camera(raw);d=np.load(dest/'views'/('%03d.npz'%raw));mask=torch.as_tensor(d['repair_mask'],device='cuda')
        kw=prepare_rasterization(c,n.static,n.args);kw['means2D']=torch.zeros_like(kw['means3D']);marker=torch.zeros((N,1),device='cuda',requires_grad=True)
        kw['features']=torch.cat([kw['features'],marker],1);out=raster(c,n.args,n.static.active_sh_degree,kw,n.bg)
        out['extra'][0,mask].sum().backward();score=torch.maximum(score,marker.grad[:,0].detach());del kw,out,marker
    ids=np.flatnonzero(cpu(score)>1e-4);np.savez_compressed(dest/'contributors.npz',ids=ids)
    r4.write(dest/'receipt.json',dict(actor=actor,source_frames=199,active_cells=int(r['active'].sum()),joint_active_cells=int(original.sum()),candidate_rows=len(ids),
        cell_m=.08,tangent_sigma_m=.052,normal_sigma_m=.015,road_colour_and_opacity_fixed=True,opacity=.99005,updates=0,parent_unchanged=n.unchanged(),
        deletion_set=[5,26,18,6,37],note='Only road repair target varies; native static delete5 baseline and original other-vehicle guard are identical.'))

def train(a,run):
    r4.RUN=run;r4.INPUT=run/'inputs';dest=run/a.output;dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0);e=r4.Experiment();opt=torch.optim.Adam(e.repair.groups(),eps=1e-15);startstep=0
    contract=dict(actor=a.actor,formal_steps=1000,source_frames=199,source_parent='native v41 B30k',gate_init=2.,lr=.06,Adam_eps=1e-15,
        loss='same R4: road L1 + .25 outside L1 +1 other-vehicle L1 +.002 mean gate',
        cell_m=.08,normal_sigma_m=.015,fixed_colour=True,fixed_road_opacity=.99005,original_time_preserved=True,
        differences_from_joint='single actor swept road domain, its coherent preserve mask and exact contributor set; no other hyperparameter changes')
    def state(step):return dict(step=step,repair=e.repair.state_dict(),optimizer=opt.state_dict(),sampler=sampler.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract,parent_identity=e.n.before)
    if a.resume:
        ck=torch.load(run/a.resume,map_location='cuda',weights_only=False);assert ck['parent_identity']==e.n.before;e.repair.load_state_dict(ck['repair']);opt.load_state_dict(ck['optimizer']);sampler.setstate(ck['sampler'])
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);startstep=ck['step']
        assert all(torch.equal(v,ck['repair'][k]) for k,v in e.repair.state_dict().items());r4.write(dest/'restore.json',dict(step=startstep,parameters_exact=True,optimizer_RNG_sampler_restored=True))
    if a.evaluate:e.evaluate(dest,range(89,138),'all49');e.checks(dest);return
    r4.write(dest/'contract.json',contract);e.checks(dest);fit=[r for r in range(89,138) if r not in [101,113,125]];shadow=[r for r in fit if e.view(r)['shadow_proxy'].sum()>=20]
    start=time.time()
    with (dest/('training_from%d.jsonl'%startstep)).open('x') as log:
        for step in range(startstep+1,a.steps+1):
            raw=sampler.choice(shadow if step%2 else fit);opt.zero_grad(set_to_none=True);loss,record=e.loss(raw);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(e.repair.gate_raw.grad).all();opt.step()
            record.update(step=step,raw=raw,loss=float(loss),seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved());log.write(json.dumps(record)+'\n');log.flush()
            if step==startstep+1 or step%100==0:print(json.dumps(record),flush=True)
            if step in [32,100,500,1000] or step==a.steps:
                torch.save(state(step),dest/('checkpoint%06d.pt'%step));e.evaluate(dest,[101,105,113],'step%06d'%step)
    r4.write(dest/'receipt.json',dict(step=a.steps,updates=a.steps-startstep,seconds=time.time()-start,parent_unchanged=e.unchanged(),quality_accepted=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=int,choices=[5,37],required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',default='formal');p.add_argument('--resume');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    run=ROOT/('single_a%d'%a.actor)
    if a.prepare:prepare(a.actor,run)
    else:train(a,run)
if __name__=='__main__':main()

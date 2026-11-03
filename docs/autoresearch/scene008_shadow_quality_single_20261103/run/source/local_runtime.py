"""Compact 3D road recovery with native detail preservation, full state resume."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
import r4_runtime as r4
from road_runtime import StableRoad
from editor_common import cpu,save

RUN=Path(__file__).resolve().parent.parent
OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
r4.RUN=RUN;r4.INPUT=RUN/'inputs'

class Repair(torch.nn.Module):
    def __init__(self,ref,meta):
        super().__init__();self.road=StableRoad(ref,meta)
        p=torch.tensor(ref['road_opacity'][self.road.ids,None],device='cuda').clamp(.00001,.99999)
        with torch.no_grad():self.road.opacity.copy_(torch.logit(p))
        self.road.dc.requires_grad_(False);self.road.opacity.requires_grad_(False)
        self.register_buffer('ids',torch.as_tensor(np.load(RUN/'inputs/contributors.npz')['ids'],device='cuda',dtype=torch.long))
        self.gate_raw=torch.nn.Parameter(torch.full((len(self.ids),),4.,device='cuda'))
        if RUN.name.endswith('_r6'):
            ck=torch.load(OLD/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False)
            assert torch.equal(ck['repair']['ids'],self.ids)
            with torch.no_grad():self.gate_raw.copy_(ck['repair']['gate_raw'])
    def groups(self):return [dict(params=[self.gate_raw],lr=.06)]
    def prior(self):return .001*(1-self.gate_raw.sigmoid()).mean()

r4.Repair=Repair

class Experiment(r4.Experiment):
    def loss(self,raw):
        d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=r4.linear(out['rgb'])
        target=torch.as_tensor(d['recovery_reference_linear'].transpose(2,0,1),device='cuda')
        fit=torch.as_tensor(d['fit_weight'],device='cuda');keep=torch.as_tensor(d['native_keep_weight'],device='cuda')
        road=((pred-target).abs()*fit[None]).sum()/(3*fit.sum().clamp_min(1))
        native=((out['rgb']-base).abs()*keep[None]).sum()/(3*keep.sum().clamp_min(1))
        marking=torch.as_tensor(d['marking_mask'],device='cuda')
        lane=(out['rgb']-base).abs()[:,marking].mean() if marking.any() else native*0
        guard=torch.as_tensor(d['other_vehicle_guard'],device='cuda')
        vehicle=(out['rgb']-base).abs()[:,guard].mean() if guard.any() else native*0
        total=road+2*native+2*lane+vehicle+self.repair.prior()
        return total,dict(road=float(road.detach()),native=float(native.detach()),marking=float(lane.detach()),other_vehicle=float(vehicle.detach()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--resume');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0)
    e=Experiment();opt=torch.optim.Adam(e.repair.groups(),eps=1e-15);startstep=0
    contract=dict(updates=a.steps,parent='native v41 B30k',source_frames=199,gate_init=4.,lr=.06,
        method='compact 3D real-evidence road envelope + native marking/clean-road protection + keep-native gate prior',
        road_time_stable=True,native_time_preserved=True,no_hard_image_route=True,metric_reference_unchanged=True)
    if RUN.name.endswith('_r6'):contract.update(gate_init='R4 addon step1000, optimizer fresh',warm_start_updates=1000)
    def state(step):
        return dict(step=step,repair=e.repair.state_dict(),optimizer=opt.state_dict(),sampler=sampler.getstate(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract,parent_identity=e.n.before)
    if a.resume:
        ck=torch.load(a.resume,map_location='cuda',weights_only=False)
        assert ck['parent_identity']==e.n.before
        e.repair.load_state_dict(ck['repair']);opt.load_state_dict(ck['optimizer']);sampler.setstate(ck['sampler'])
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        startstep=ck['step'];assert all(torch.equal(v,ck['repair'][k]) for k,v in e.repair.state_dict().items())
        r4.write(dest/'restore.json',dict(step=startstep,parameters_exact=True,optimizer_RNG_sampler_restored=True,parent_identity_exact=True))
    if a.evaluate:
        e.evaluate(dest,range(89,138),'all49');e.checks(dest);return
    r4.write(dest/'contract.json',contract);e.checks(dest)
    fit=[r for r in range(89,138) if r not in [101,113,125]];shadow=[r for r in fit if e.view(r)['shadow_proxy'].sum()>=20]
    start=time.time()
    with (dest/('training_from%d.jsonl'%startstep)).open('x') as log:
        for step in range(startstep+1,a.steps+1):
            raw=sampler.choice(shadow if step%2 else fit);opt.zero_grad(set_to_none=True);loss,record=e.loss(raw)
            assert torch.isfinite(loss);loss.backward();assert torch.isfinite(e.repair.gate_raw.grad).all()
            record['gradient_max']=float(e.repair.gate_raw.grad.abs().max());opt.step()
            record.update(step=step,raw=raw,loss=float(loss),seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n');log.flush()
            if step==startstep+1 or step%100==0:print(json.dumps(record),flush=True)
            if step in [32,100,500,1000] or step==a.steps:
                torch.save(state(step),dest/('checkpoint%06d.pt'%step));e.evaluate(dest,[101,105,113],'step%06d'%step)
    r4.write(dest/'receipt.json',dict(start_step=startstep,updates=a.steps-startstep,step=a.steps,seconds=time.time()-start,parent_unchanged=e.unchanged(),quality_accepted=False))

if __name__=='__main__':main()

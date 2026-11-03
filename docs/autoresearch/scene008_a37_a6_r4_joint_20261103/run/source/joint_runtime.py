"""Joint a37/a6 residual editing above an immutable R4 baseline."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,argparse,random,time,copy
from pathlib import Path
import numpy as np
import torch
import r4_runtime as r4
from editor_common import cpu,save,prepare_rasterization,road_kwargs,merge_rows,raster
from road_runtime import StableRoad

RUN=Path(__file__).resolve().parent.parent;OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
r4.RUN=OLD;r4.INPUT=OLD/'inputs'

class Joint(torch.nn.Module):
    def __init__(self,reference,meta):
        super().__init__();self.road=StableRoad(reference,meta);self.road.dc.requires_grad_(False);self.road.opacity.requires_grad_(False)
        self.register_buffer('ids',torch.tensor(np.load(RUN/'joint_contributors.npz')['ids'],device='cuda',dtype=torch.long))
        self.gate_raw=torch.nn.Parameter(torch.full((len(self.ids),),4.,device='cuda'))
    def prior(self):return .0005*(1-self.gate_raw.sigmoid()).mean()

class Experiment:
    def __init__(self):
        self.base_e=r4.Experiment();ck=torch.load(OLD/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False);self.base_e.repair.load_state_dict(ck['repair'])
        self.base_e.repair.requires_grad_(False);self.ref=dict(np.load(RUN/'inputs/extra_road_reference.npz'));self.joint=Joint(self.ref,self.base_e.meta)
        self.n=self.base_e.n;self.meta=self.base_e.meta;self.views={};self.bases={}
    def view(self,raw):
        if raw not in self.views:self.views[raw]=dict(np.load(RUN/'inputs/views'/('%03d.npz'%raw)))
        return self.views[raw]
    @torch.no_grad()
    def base(self,raw):
        if raw not in self.bases:self.bases[raw]=self.base_e.render(raw)['rgb'].detach()
        return self.bases[raw]
    def render(self,raw,enabled=True,camera=None):
        if not enabled:return self.base_e.render(raw,camera=camera)
        c=camera or self.base_e.camera(raw);kw=prepare_rasterization(c,self.n.static,self.n.args);kw['means2D']=torch.zeros_like(kw['means3D'])
        ids=self.base_e.repair.ids;kw['opacities']=kw['opacities'].index_copy(0,ids,kw['opacities'][ids]*self.base_e.repair.gate_raw.sigmoid()[:,None])
        ids=self.joint.ids;kw['opacities']=kw['opacities'].index_copy(0,ids,kw['opacities'][ids]*self.joint.gate_raw.sigmoid()[:,None])
        shn=kw['shs'].shape[1] if kw['shs'] is not None else 1;pre=kw['colors_precomp'] is not None
        kw['features']=torch.cat([kw['features'],torch.zeros((len(kw['means3D']),2),device='cuda')],1)
        for road,which in [(self.base_e.repair.road,0),(self.joint.road,1)]:
            rr=road_kwargs(c,road,self.n.args,shn,precomputed_color=pre);marker=torch.zeros((len(rr['means3D']),2),device='cuda');marker[:,which]=1
            rr['features']=torch.cat([rr['features'],marker],1);kw=merge_rows(kw,rr)
        out=raster(c,self.n.args,self.n.static.active_sh_degree,kw,self.n.bg);env=self.n.env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1);out['rgb']=out['rgb']+(1-out['alpha'])*env
        return out
    def unchanged(self):return self.base_e.unchanged()
    def loss(self,raw):
        d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=r4.linear(out['rgb']);target=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
        photo=[];rec={}
        for a in [6,37]:
            mask=torch.tensor(d['fit_a%d'%a],device='cuda');term=(pred-target).abs()[:,mask].mean() if mask.any() else pred.sum()*0
            photo.append(term);rec['road_a%d'%a]=float(term.detach())
        protect=torch.tensor(d['protect_mask'],device='cuda');keep=(out['rgb']-base).abs()[:,protect].mean();guard=torch.tensor(d['other_vehicle_guard'],device='cuda')
        other=(out['rgb']-base).abs()[:,guard].mean() if guard.any() else keep*0
        total=.5*(photo[0]+photo[1])+.5*keep+other+self.joint.prior();rec.update(keep_R4=float(keep.detach()),other_vehicle=float(other.detach()));return total,rec
    @torch.no_grad()
    def evaluate(self,dest,raws,tag):
        folder=Path(dest)/tag;folder.mkdir(parents=True,exist_ok=True);rows=[]
        for raw in raws:
            d=self.view(raw);old=self.base_e.view(raw);base=self.base(raw);out=self.render(raw);pred=r4.linear(out['rgb']);before=r4.linear(base);target=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
            rec=dict(raw=raw,actors={})
            for a in [6,37]:
                mask=torch.tensor(d['fit_a%d'%a],device='cuda');rec['actors'][str(a)]=dict(samples=int(mask.sum()),R4_MAE=float((before-target).abs()[:,mask].mean()) if mask.any() else None,
                    joint_MAE=float((pred-target).abs()[:,mask].mean()) if mask.any() else None,extra_road_alpha_T=float(out['extra'][1,mask].mean()) if mask.any() else None)
            prior_target=torch.tensor(old['reference_linear'].transpose(2,0,1),device='cuda');rec['original_shadow']={}
            for a in [5,37]:
                mask=torch.tensor(old['shadow_proxy']&old['domain_a%d'%a],device='cuda');rec['original_shadow'][str(a)]=dict(samples=int(mask.sum()),R4_MAE=float((before-prior_target).abs()[:,mask].mean()) if mask.any() else None,joint_MAE=float((pred-prior_target).abs()[:,mask].mean()) if mask.any() else None)
            protect=torch.tensor(d['protect_mask'],device='cuda');rec['outside_MAE']=float((out['rgb']-base).abs()[:,protect].mean());rows.append(rec)
            save(folder/('%03d_R4.png'%raw),base);save(folder/('%03d_joint.png'%raw),out['rgb']);save(folder/('%03d_extra_alpha_T.png'%raw),out['extra'][1:].expand(3,-1,-1))
        r4.write(folder/'metrics.json',dict(rows=rows,parent_unchanged=self.unchanged(),all199_BUILD=True,reference_is_proxy=True,independent_test=False));return rows
    @torch.no_grad()
    def checks(self,dest):
        base=self.base(105);bypass=self.render(105,False)['rgb'];diff=float((base-bypass).abs().max());assert diff==0
        c=self.base_e.camera(105);roads=[];native=[]
        for raw in [89,105,137]:
            camera=copy.copy(c);camera.timestamp=self.base_e.camera(raw).timestamp
            rr=road_kwargs(camera,self.joint.road,self.n.args,precomputed_color=True);roads.append(raster(camera,self.n.args,0,rr,self.n.bg)['rgb']);native.append(self.base_e.base(105,camera))
        rec=dict(bypass_R4_maxabs=diff,added_road_time_maxabs=[float((x-roads[0]).abs().max()) for x in roads],native_time_MAE=[float((x-native[0]).abs().mean()) for x in native],parent_unchanged=self.unchanged(),base_R4_parameters_unchanged=all(not p.requires_grad for p in self.base_e.repair.parameters()),hard_image_route=False)
        assert max(rec['added_road_time_maxabs'])==0;r4.write(Path(dest)/'invariants.json',rec)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume');p.add_argument('--evaluate',action='store_true');a=p.parse_args();dest=RUN/a.output;dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0);e=Experiment();opt=torch.optim.Adam([e.joint.gate_raw],lr=.06,eps=1e-15);step0=0
    contract=dict(joint_actors=[37,6],base_R4='frozen checkpoint001000',formal_steps=1000,source_frames=199,lr=.06,gate_init=4.,loss='.5 a6 L1 +.5 a37 L1 +.5 preserve R4 outside +1 other vehicle +.0005 keep-native gate prior',only_additional_static_opacity_gate_trainable=True,new_road_fixed=True,original_B_and_R4_readonly=True)
    def state(step):return dict(step=step,joint=e.joint.state_dict(),optimizer=opt.state_dict(),sampler=sampler.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract,parent_identity=e.n.before)
    if a.resume:
        ck=torch.load(RUN/a.resume,map_location='cuda',weights_only=False);assert ck['parent_identity']==e.n.before;e.joint.load_state_dict(ck['joint']);opt.load_state_dict(ck['optimizer']);sampler.setstate(ck['sampler']);torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);step0=ck['step']
        assert all(torch.equal(v,ck['joint'][k]) for k,v in e.joint.state_dict().items());r4.write(dest/'restore.json',dict(step=step0,parameters_exact=True,optimizer_RNG_sampler_restored=True))
    e.checks(dest)
    if a.evaluate:e.evaluate(dest,range(89,138),'all49');return
    r4.write(dest/'contract.json',contract);fit=[r for r in range(89,138) if r not in [101,113,125]];start=time.time()
    with (dest/('training_from%d.jsonl'%step0)).open('x') as log:
        for step in range(step0+1,a.steps+1):
            raw=sampler.choice(fit);opt.zero_grad(set_to_none=True);loss,rec=e.loss(raw);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(e.joint.gate_raw.grad).all();opt.step()
            rec.update(step=step,raw=raw,loss=float(loss),seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved());log.write(json.dumps(rec)+'\n');log.flush()
            if step==step0+1 or step%100==0:print(json.dumps(rec),flush=True)
            if step in [32,100,500,1000] or step==a.steps:
                torch.save(state(step),dest/('checkpoint%06d.pt'%step));e.evaluate(dest,[101,105,113],'step%06d'%step)
    r4.write(dest/'receipt.json',dict(step=a.steps,updates=a.steps-step0,seconds=time.time()-start,parent_unchanged=e.unchanged(),quality_accepted=False,human_verdict=None))
if __name__=='__main__':main()

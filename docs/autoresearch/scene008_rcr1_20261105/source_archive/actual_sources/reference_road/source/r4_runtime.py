"""Stable road recovery plus all-view local opacity repair; parent retains B(t)."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from editor_common import Native,cpu,save,prepare_rasterization,road_kwargs,merge_rows,raster
from road_runtime import StableRoad

RUN=Path(__file__).resolve().parent.parent
INPUT=RUN/'inputs'
META=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))

def linear(x):
    x=x.clamp_min(0)
    return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

class Repair(torch.nn.Module):
    def __init__(self,ref,meta):
        super().__init__();self.road=StableRoad(ref,meta)
        self.register_buffer('ids',torch.as_tensor(np.load(INPUT/'contributors.npz')['ids'],device='cuda',dtype=torch.long))
        self.road.dc.requires_grad_(False);self.road.opacity.requires_grad_(False)
        self.gate_raw=torch.nn.Parameter(torch.full((len(self.ids),),2.,device='cuda'))
    def groups(self):
        return [dict(params=[self.gate_raw],lr=.06)]
    def prior(self):
        return .002*self.gate_raw.sigmoid().mean()

class Experiment:
    def __init__(self):
        self.n=Native();self.ref=dict(np.load(INPUT/'road_reference.npz'));self.meta=json.loads(META.read_text())
        self.repair=Repair(self.ref,self.meta);self.cameras={};self.views={};self.bases={}
        self.versions={k:getattr(self.n.static,k)._version for k in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity']}
    def camera(self,raw):
        if raw not in self.cameras:self.cameras[raw]=self.n.camera(raw)
        return self.cameras[raw]
    def view(self,raw):
        if raw not in self.views:self.views[raw]=dict(np.load(INPUT/'views'/('%03d.npz'%raw)))
        return self.views[raw]
    @torch.no_grad()
    def base(self,raw,camera=None):
        if camera is not None:return self.n.render(camera)['rgb'].detach()
        if raw not in self.bases:self.bases[raw]=self.n.render(self.camera(raw))['rgb'].detach()
        return self.bases[raw]
    def render(self,raw,enabled=True,camera=None):
        c=camera or self.camera(raw);kw=prepare_rasterization(c,self.n.static,self.n.args)
        kw['means2D']=torch.zeros_like(kw['means3D'])
        if enabled:
            # Only scale the current native opacity. In particular, never force
            # another frame's native mask, mean, or time envelope into this view.
            ids=self.repair.ids
            kw['opacities']=kw['opacities'].index_copy(0,ids,kw['opacities'][ids]*self.repair.gate_raw.sigmoid()[:,None])
            n=kw['shs'].shape[1] if kw['shs'] is not None else 1
            rr=road_kwargs(c,self.repair.road,self.n.args,n,precomputed_color=kw['colors_precomp'] is not None)
            kw['features']=torch.cat([kw['features'],torch.zeros_like(kw['features'][:,:1])],1)
            rr['features']=torch.cat([rr['features'],torch.ones_like(rr['features'][:,:1])],1)
            kw=merge_rows(kw,rr)
        else:kw['features']=torch.cat([kw['features'],torch.zeros_like(kw['features'][:,:1])],1)
        out=raster(c,self.n.args,self.n.static.active_sh_degree,kw,self.n.bg)
        env=self.n.env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        out['rgb']=out['rgb']+(1-out['alpha'])*env
        return out
    def unchanged(self):return self.n.unchanged() and all(getattr(self.n.static,k)._version==v for k,v in self.versions.items())
    def loss(self,raw):
        d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=linear(out['rgb'])
        target=torch.as_tensor(d['recovery_reference_linear'].transpose(2,0,1),device='cuda')
        weight=torch.as_tensor(d['fit_weight'],device='cuda')
        road=((pred-target).abs()*weight[None]).sum()/(3*weight.sum())
        protect=torch.as_tensor(d['protect_mask'],device='cuda')
        outside=(out['rgb']-base).abs()[:,protect].mean()
        guard=torch.as_tensor(d['other_vehicle_guard'],device='cuda')
        vehicle=(out['rgb']-base).abs()[:,guard].mean() if bool(guard.any()) else outside*0
        total=road+.25*outside+vehicle+self.repair.prior()
        return total,dict(road=float(road.detach()),outside=float(outside.detach()),other_vehicle=float(vehicle.detach()))
    @torch.no_grad()
    def evaluate(self,dest,raws,tag):
        folder=Path(dest)/tag;folder.mkdir(parents=True,exist_ok=True);rows=[]
        for raw in raws:
            d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=linear(out['rgb']);before=linear(base)
            target=torch.as_tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
            actors={}
            for alias in ['5','37']:
                road=torch.as_tensor(d['valid']&d['domain_a'+alias],device='cuda')
                shadow=torch.as_tensor(d['shadow_proxy']&d['domain_a'+alias],device='cuda')
                def mae(image,mask):return float((image-target).abs()[:,mask].mean()) if bool(mask.any()) else None
                actors[alias]=dict(road_samples=int(road.sum()),shadow_samples=int(shadow.sum()),
                    road_MAE=mae(pred,road),baseline_road_MAE=mae(before,road),shadow_MAE=mae(pred,shadow),baseline_shadow_MAE=mae(before,shadow),
                    road_alpha_T_on_shadow=float(out['extra'][0,shadow].mean()) if bool(shadow.any()) else None)
            outside=torch.as_tensor(d['protect_mask'],device='cuda');guard=torch.as_tensor(d['other_vehicle_guard'],device='cuda')
            rows.append(dict(raw=raw,actors=actors,outside_MAE=float((out['rgb']-base).abs()[:,outside].mean()),
                outside_maxabs=float((out['rgb']-base).abs()[:,outside].max()),
                other_vehicle_MAE=float((out['rgb']-base).abs()[:,guard].mean()) if bool(guard.any()) else None,
                weak_measured_pixels=int(d['weak_measured'].sum()),inferred_pixels=int(d['inferred'].sum()),
                gate_mean=float(self.repair.gate_raw.sigmoid().mean())))
            save(folder/('%03d_repaired_delete5.png'%raw),out['rgb']);save(folder/('%03d_baseline_delete5.png'%raw),base)
            save(folder/('%03d_road_alpha_T.png'%raw),out['extra'].expand(3,-1,-1))
        write(folder/'metrics.json',dict(rows=rows,parent_unchanged=self.unchanged(),
            metrics_use_original_strong_reference_only=True,inferred_not_used_as_truth=True,reference_is_proxy=True))
        return rows
    @torch.no_grad()
    def checks(self,dest):
        c=self.camera(105);baseline=self.base(105);bypass=self.render(105,False)['rgb']
        bypassdiff=float((baseline-bypass).abs().max());assert bypassdiff<2e-6
        roads=[];natives=[]
        for raw in [89,105,137]:
            camera=copy.copy(c);camera.timestamp=self.camera(raw).timestamp
            rr=road_kwargs(camera,self.repair.road,self.n.args,precomputed_color=True)
            roads.append(raster(camera,self.n.args,0,rr,self.n.bg)['rgb']);natives.append(self.base(105,camera))
        receipt=dict(bypass_maxabs=bypassdiff,road_fixed_camera_time_maxabs=[float((x-roads[0]).abs().max()) for x in roads],
            native_fixed_camera_time_MAE=[float((x-natives[0]).abs().mean()) for x in natives],
            native_time_envelope_and_position_preserved=True,global_timestamp_override=False,hard_image_route=False,
            only_road_representation_has_no_time_degrees=True,parent_unchanged=self.unchanged())
        assert max(receipt['road_fixed_camera_time_maxabs'])==0;write(Path(dest)/'invariants.json',receipt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    dest=Path(a.output);dest.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);e=Experiment();opt=torch.optim.Adam(e.repair.groups(),eps=1e-15)
    sampler=random.Random(0)
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False);e.repair.load_state_dict(ck['repair']);opt.load_state_dict(ck['optimizer']);sampler.setstate(ck['sampler'])
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        assert all(torch.equal(v,ck['repair'][k]) for k,v in e.repair.state_dict().items())
        write(dest/'restore.json',dict(step=ck['step'],repair_tensors_exact=True,optimizer_and_all_RNG_restored=True,updates=0))
    if a.evaluate:
        e.evaluate(dest,range(89,138),'all49');e.checks(dest);return
    assert not a.checkpoint
    fit=[r for r in range(89,138) if r not in [101,113,125]]
    shadow=[r for r in fit if e.view(r)['shadow_proxy'].sum()>=20]
    contract=dict(parent='native v41 B 30k',updates=1000,source_frames=199,native_frames=49,
        method='fixed measured road recovery + all-view native contributor opacity gates; corrected deleted-actor guard',
        selection='exact all49 raster alpha*T, no semantic ownership claim',
        gate_init=2.,LR_opacity=.06,Adam_eps=1e-15,road_colour_and_opacity_fixed=True,
        only_road_branch_time_stable=True,native_time_state_preserved=True,independent_shadow_layer=False,hard_route=False,
        inferred_colours_explicit=True,metric_reference_unchanged=True,smoke_updates=0,
        loss='confidence weighted road L1 + .25 outside coherent domain L1 +1 corrected other-vehicle L1 + gate prior')
    write(dest/'contract.json',contract);e.checks(dest);e.evaluate(dest,[101,105,113],'initial');start=time.time()
    with (dest/'training.jsonl').open('x') as log:
        for step in range(1,1001):
            raw=sampler.choice(shadow if step%2 else fit);opt.zero_grad(set_to_none=True);loss,record=e.loss(raw)
            assert torch.isfinite(loss);loss.backward()
            gradients={name:float(x.grad.abs().max()) for name,x in e.repair.named_parameters() if x.requires_grad}
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in e.repair.parameters() if x.requires_grad);opt.step()
            record.update(step=step,raw=raw,loss=float(loss),gradients=gradients,seconds=time.time()-start,
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n');log.flush()
            if step==1 or step%100==0:print(json.dumps(record),flush=True)
            if step in [100,500,1000]:
                torch.save(dict(step=step,repair=e.repair.state_dict(),optimizer=opt.state_dict(),sampler=sampler.getstate(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract),dest/('checkpoint%06d.pt'%step))
                e.evaluate(dest,[101,105,113],'step%06d'%step)
    write(dest/'receipt.json',dict(updates=1000,seconds=time.time()-start,parent_unchanged=e.unchanged(),quality_accepted=False,human_verdict=None))

if __name__=='__main__':main()

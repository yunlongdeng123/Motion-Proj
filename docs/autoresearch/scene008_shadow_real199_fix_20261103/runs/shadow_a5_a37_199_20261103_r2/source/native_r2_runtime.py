"""Native B migration of the verified historical R2 road/opacity repair.

Only the local edit copy changes: road rows get stable position/alpha/colour;
dark residual contributors get stable-position opacity gates. Remaining B rows
retain their actual timestamp. No image hard-route, no independent shadow layer.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from editor_common import Native,cpu,save,prepare_rasterization,raster
from utils.sh_utils import eval_sh

RUN=Path(__file__).resolve().parent.parent
PREVIOUS=RUN.with_name('shadow_a5_a37_199_20261103')
INPUT=PREVIOUS/'inputs_final'
FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))

def linear(x):
    x=x.clamp_min(0)
    return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

class LocalRepair(torch.nn.Module):
    def __init__(self,native,selection):
        super().__init__();device='cuda'
        for name in ['receiver_ids','artifact_ids','receiver_xyz_anchor','artifact_xyz_anchor','artifact_base_opacity']:
            self.register_buffer(name,torch.as_tensor(selection[name],device=device,dtype=torch.long if name.endswith('_ids') else torch.float32))
        colour=torch.tensor(selection['receiver_rgb_anchor'],device=device).clamp(.01,.99)
        opacity=torch.tensor(selection['receiver_base_opacity'],device=device).clamp(.0001,.9999)
        self.color_raw=torch.nn.Parameter(torch.logit(colour));self.opacity_raw=torch.nn.Parameter(torch.logit(opacity))
        self.artifact_raw=torch.nn.Parameter(torch.full((len(self.artifact_ids),),6.,device=device))
        self.register_buffer('initial_opacity',opacity)
    def groups(self):
        return [dict(params=[self.color_raw],lr=.035,name='road_color'),
            dict(params=[self.opacity_raw,self.artifact_raw],lr=.06,name='local_opacity')]
    def regularization(self):
        return .002*self.artifact_raw.sigmoid().mean()+.001*(self.opacity_raw.sigmoid()-self.initial_opacity).abs().mean()
    def patch(self,kw,camera,colour):
        ri,ai=self.receiver_ids,self.artifact_ids
        kw['means3D']=kw['means3D'].index_copy(0,ri,self.receiver_xyz_anchor).index_copy(0,ai,self.artifact_xyz_anchor)
        kw['opacities']=kw['opacities'].index_copy(0,ri,self.opacity_raw.sigmoid()[:,None])
        kw['opacities']=kw['opacities'].index_copy(0,ai,self.artifact_base_opacity[:,None]*self.artifact_raw.sigmoid()[:,None])
        kw['mask']=kw['mask'].clone();kw['mask'][ri]=True;kw['mask'][ai]=True
        kw['colors_precomp']=colour.index_copy(0,ri,self.color_raw.sigmoid());kw['shs']=None
        # Re-enabled rows need corrected alpha/depth feature values as well as
        # a true prefilter bit; otherwise the environment would be added twice.
        ids=torch.cat([ri,ai]);xyz=kw['means3D'][ids]
        depth=torch.cat([xyz,torch.ones_like(xyz[:,:1])],1)@camera.world_view_transform[:,2:3]
        features=kw['features'].index_copy(0,ids,torch.cat([depth,torch.ones_like(depth)],1))
        indicator=torch.zeros_like(features[:,:1]);indicator[ri]=1
        kw['features']=torch.cat([features,indicator],1)
        return kw

class Experiment:
    def __init__(self):
        self.n=Native();self.cameras={};self.views={};self.bases={};self.guards={}
        self.repair=LocalRepair(self.n,dict(np.load(RUN/'selection/selection.npz')))
        self.authorized=np.zeros((320,480),bool);self.authorized[205:315,20:205]=True;self.authorized[235:320,195:340]=True
        self.versions={k:getattr(self.n.static,k)._version for k in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity']}
    def camera(self,raw):
        if raw not in self.cameras:self.cameras[raw]=self.n.camera(raw)
        return self.cameras[raw]
    def view(self,raw):
        if raw not in self.views:self.views[raw]=dict(np.load(INPUT/'views_current'/('%03d.npz'%raw)))
        return self.views[raw]
    def base(self,raw,camera=None):
        if camera is not None:return self.n.render(camera)['rgb'].detach()
        if raw not in self.bases:
            with torch.no_grad():self.bases[raw]=self.n.render(self.camera(raw))['rgb'].detach()
        return self.bases[raw]
    def render(self,raw,enabled=True,camera=None):
        camera=camera or self.camera(raw);kw=prepare_rasterization(camera,self.n.static,self.n.args)
        dirs=torch.nn.functional.normalize(kw['means3D']-camera.camera_center,dim=1)
        colour=kw['colors_precomp'] if kw['colors_precomp'] is not None else (eval_sh(self.n.static.active_sh_degree,kw['shs'].transpose(1,2),dirs)+.5).clamp_min(0)
        if enabled:kw=self.repair.patch(kw,camera,colour)
        else:kw['features']=torch.cat([kw['features'],torch.zeros_like(kw['features'][:,:1])],1)
        kw['means2D']=torch.zeros_like(kw['means3D'])
        out=raster(camera,self.n.args,self.n.static.active_sh_degree,kw,self.n.bg)
        environment=self.n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        out['rgb']=out['rgb']+(1-out['alpha'])*environment
        out['road_contribution']=out['extra']
        return out
    def unchanged(self):return self.n.unchanged() and all(getattr(self.n.static,k)._version==v for k,v in self.versions.items())
    def loss(self,raw):
        d=self.view(raw);out=self.render(raw);base=self.base(raw)
        mask=torch.tensor(d['valid']&self.authorized,device='cuda')
        target=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
        pred=linear(out['rgb']);road=(pred-target).abs()[:,mask].mean()
        gt=torch.tensor(d['observed'].transpose(2,0,1),device='cuda');truth=linear(gt)
        if raw not in self.guards:self.guards[raw]=np.load(FULL/'frames'/('%03d.npz'%raw))['guard']
        clean=mask&~torch.tensor(self.guards[raw],device='cuda')&((truth-target).abs().amax(0)<.045)
        observed=(pred-truth).abs()[:,clean].mean() if bool(clean.any()) else road*0
        protect=torch.tensor(~self.authorized|d['other_vehicle_guard'],device='cuda')
        anchor=(out['rgb']-base).abs()[:,protect].mean()
        total=road+.20*observed+.25*anchor+self.repair.regularization()
        return total,dict(road=float(road.detach()),clean_observed=float(observed.detach()),outside_anchor=float(anchor.detach()))
    @torch.no_grad()
    def evaluate(self,dest,raws,tag):
        folder=Path(dest)/tag;folder.mkdir(parents=True,exist_ok=True);rows=[]
        for raw in raws:
            d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=linear(out['rgb']);before=linear(base)
            target=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
            actors={}
            for alias in ['5','37']:
                road=torch.tensor(d['valid']&d['domain_a'+alias],device='cuda')
                shadow=torch.tensor(d['shadow_proxy']&d['domain_a'+alias],device='cuda')
                def mae(image,mask):return float((image-target).abs()[:,mask].mean()) if bool(mask.any()) else None
                actors[alias]=dict(road_samples=int(road.sum()),shadow_samples=int(shadow.sum()),
                    road_MAE=mae(pred,road),baseline_road_MAE=mae(before,road),shadow_MAE=mae(pred,shadow),baseline_shadow_MAE=mae(before,shadow),
                    actual_receiver_contribution=float(out['road_contribution'][0,shadow].mean()) if bool(shadow.any()) else None)
            outside=torch.tensor(~self.authorized,device='cuda')
            row=dict(raw=raw,actors=actors,outside_MAE=float((out['rgb']-base).abs()[:,outside].mean()),
                outside_maxabs=float((out['rgb']-base).abs()[:,outside].max()),artifact_opacity_mean=float(self.repair.artifact_raw.sigmoid().mean()))
            rows.append(row);save(folder/('%03d_repaired_delete5.png'%raw),out['rgb']);save(folder/('%03d_baseline_delete5.png'%raw),base)
            save(folder/('%03d_receiver_alpha_T.png'%raw),out['road_contribution'].expand(3,-1,-1))
        write(folder/'metrics.json',dict(rows=rows,updates=0,reference_is_proxy=True,parent_unchanged=self.unchanged()))
        return rows
    @torch.no_grad()
    def checks(self,dest):
        c=self.camera(105);baseline=self.base(105);bypass=self.render(105,False)['rgb']
        assert float((baseline-bypass).abs().max())<2e-6
        arrays=[];native=[]
        for raw in [89,105,137]:
            camera=copy.copy(c);camera.timestamp=self.camera(raw).timestamp
            kw=prepare_rasterization(camera,self.n.static,self.n.args)
            dirs=torch.nn.functional.normalize(kw['means3D']-camera.camera_center,dim=1)
            col=(eval_sh(self.n.static.active_sh_degree,kw['shs'].transpose(1,2),dirs)+.5).clamp_min(0) if kw['colors_precomp'] is None else kw['colors_precomp']
            kw=self.repair.patch(kw,camera,col);ids=self.repair.receiver_ids
            subset={k:(v[ids] if v is not None else None) for k,v in kw.items()}
            subset['means2D']=torch.zeros_like(subset['means3D']);rr=raster(camera,self.n.args,0,subset,self.n.bg)
            arrays.append(rr['rgb']);native.append(self.base(105,camera))
        receipt=dict(bypass_maxabs=float((baseline-bypass).abs().max()),
            receiver_fixed_camera_time_maxabs=[float((x-arrays[0]).abs().max()) for x in arrays],
            original_static_time_MAE=[float((x-native[0]).abs().mean()) for x in native],
            global_static_timestamp_override=False,hard_image_route=False,artifact_opacity_optimization=True)
        assert max(receipt['receiver_fixed_camera_time_maxabs'])==0
        write(Path(dest)/'invariants.json',receipt)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--output',required=True)
    a=parser.parse_args();dest=Path(a.output);dest.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);e=Experiment();optimizer=torch.optim.Adam(e.repair.groups())
    fit=[r for r in range(89,138) if r not in [101,113,125]]
    shadow=[r for r in fit if int(e.view(r)['shadow_proxy'].sum())>=20];sampler=random.Random(0)
    assert shadow
    e.checks(dest);e.evaluate(dest,[101,105,113],'initial')
    contract=dict(formal_updates=a.steps,source='Verified historical R2 two-way road/opacity repair migrated to native B',
        parent='v41 B30k',evidence_frames=199,only_repair_rows_time_stable=True,original_static_time_preserved=True,
        hard_image_route=False,independent_shadow_layer=False,learning_rates=dict(colour=.035,road_and_artifact_opacity=.06),
        loss='qualified road reference L1 + .20 clean observed L1 + .25 outside anchor L1 + historical opacity regularizers',
        sampler='historical alternating shadow-available / full fit frames, seed0; three checkpoint evaluation views omitted from sampling',
        no_heldout_claim=True,receiver_rows=len(e.repair.receiver_ids),artifact_rows=len(e.repair.artifact_ids))
    write(dest/'contract.json',contract);start=time.time()
    with (dest/'training.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            raw=sampler.choice(shadow if step%2 else fit);optimizer.zero_grad(set_to_none=True)
            loss,record=e.loss(raw);assert torch.isfinite(loss);loss.backward()
            gradients={name:float(p.grad.abs().max()) if p.grad is not None else None for name,p in e.repair.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.repair.parameters())
            optimizer.step();record.update(step=step,raw=raw,loss=float(loss),seconds=time.time()-start,gradients=gradients,
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n');log.flush()
            if step==1 or step%100==0:print(json.dumps(record),flush=True)
            if step in [100,500,a.steps]:
                torch.save(dict(step=step,repair=e.repair.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.getstate(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract),dest/('checkpoint%06d.pt'%step))
                e.evaluate(dest,[101,105,113],'step%06d'%step)
        e.evaluate(dest,range(89,138),'all49');e.checks(dest)
    assert e.unchanged();write(dest/'receipt.json',dict(updates=a.steps,seconds=time.time()-start,parent_unchanged=True,
        smoke_updates=0,legacy_replay_updates=0,quality_accepted=False,human_verdict=None))

if __name__=='__main__':main()

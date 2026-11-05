"""冻结单a37 B10k，只优化静态残留透明度；细道路颜色与几何固定。"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,random,time,copy
import numpy as np
import torch
from scipy.ndimage import binary_dilation,binary_erosion
import r4_runtime as r4
from editor_common import cpu,save,prepare_rasterization,raster,merge_rows
from native_binding import Native
r4.Native=Native
from road_runtime import StableRoad
RUN=Path(__file__).resolve().parent.parent;r4.RUN=RUN;r4.INPUT=RUN/'inputs'
PARENT=RUN/'native_delete'
ALIASES=[5,26,18,6]

class Repair(torch.nn.Module):
    def __init__(self,ref,meta):
        super().__init__();self.road=StableRoad(ref,meta)
        with torch.no_grad():self.road.opacity.copy_(torch.logit(torch.as_tensor(ref['road_opacity'][self.road.ids,None],device='cuda').clamp(.00001,.99999)))
        self.road.requires_grad_(False)
        self.register_buffer('ids',torch.as_tensor(np.load(RUN/'inputs/contributors.npz')['ids'],device='cuda',dtype=torch.long))
        self.gate_raw=torch.nn.Parameter(torch.full((len(self.ids),),2.,device='cuda'))
r4.Repair=Repair

class Experiment(r4.Experiment):
    def __init__(self):super().__init__();self.settings=json.loads((RUN/'settings.json').read_text())
    def render(self,raw,enabled=True,camera=None,keep=False):
        c=camera or self.camera(raw)
        if not enabled:
            return self.n.render(c,dynamic=keep)
        return self.n.render(c,road=self.repair.road,ids=self.repair.ids,
                             gates=self.repair.gate_raw.sigmoid(),dynamic=keep)
    def loss(self,raw):
        d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=r4.linear(out['rgb'])
        target=torch.as_tensor(d['recovery_reference_linear'].transpose(2,0,1),device='cuda')
        fit=torch.as_tensor(d['fit_weight'],device='cuda');keep=torch.as_tensor(d['native_keep_weight'],device='cuda')
        photo=((pred-target).abs()*fit[None]).sum()/(3*fit.sum().clamp_min(1))
        preserve=((out['rgb']-base).abs()*keep[None]).sum()/(3*keep.sum().clamp_min(1))
        marking=torch.as_tensor(d['marking_mask'],device='cuda');other=torch.as_tensor(d['other_vehicle_guard'],device='cuda')
        lane=(out['rgb']-base).abs()[:,marking].mean() if marking.any() else photo*0
        vehicle=(out['rgb']-base).abs()[:,other].mean() if other.any() else photo*0
        left=torch.as_tensor(d['valid']&d['domain_a5']&~d['domain_a37']&(d['actor_labels']==0),device='cuda')
        left_keep=(out['rgb']-base).abs()[:,left].mean() if left.any() else photo*0
        cfg=self.settings['loss_weights'];prior=cfg['keep_native_gate_prior']*(1-self.repair.gate_raw.sigmoid()).mean()
        total=cfg['road']*photo+cfg['native']*preserve+cfg['yellow']*lane+cfg['other_vehicle']*vehicle+cfg['left_road']*left_keep+prior
        return total,dict(road=float(photo.detach()),preserve=float(preserve.detach()),yellow=float(lane.detach()),other=float(vehicle.detach()))
    @torch.no_grad()
    def evaluate(self,dest,raws,tag):
        folder=Path(dest)/tag;folder.mkdir(parents=True,exist_ok=True);rows=[];replay=[]
        for raw in raws:
            d=self.view(raw);out=self.render(raw);base=self.base(raw);pred=r4.linear(out['rgb']);before=r4.linear(base)
            ref=torch.tensor(d['recovery_reference_linear'].transpose(2,0,1),device='cuda');original=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
            rec=dict(raw=raw,regions={},retained={})
            masks={'fit':d['repair_mask']&~d['inferred_fine'],'inferred':d['inferred_fine'],'new_dark':d['new_shadow_proxy']&d['repair_mask']&~d['inferred_fine'],'original_shadow':d['shadow_proxy']&d['domain_a37'],
                'yellow':d['marking_mask'],'outside':d['protect_mask'],'left_road':d['valid']&d['domain_a5']&~d['domain_a37']&(d['actor_labels']==0),
                'seam':binary_dilation(d['repair_mask'],iterations=2)&~binary_erosion(d['repair_mask'],iterations=2)&~d['other_vehicle_guard']}
            for name,m in masks.items():
                mask=torch.as_tensor(m,device='cuda');target=original if name=='original_shadow' else ref
                rec['regions'][name]=dict(samples=int(mask.sum()),before_MAE=float((before-target).abs()[:,mask].mean()) if mask.any() else None,
                    after_MAE=float((pred-target).abs()[:,mask].mean()) if mask.any() else None,native_srgb_change=float((out['rgb']-base).abs()[:,mask].mean()) if mask.any() else None,
                    road_alpha_T=float(out['extra'][0,mask].mean()) if mask.any() else None)
            for label,actor in enumerate(ALIASES,1):
                mask=torch.tensor(d['actor_labels']==label,device='cuda');rec['retained'][str(actor)]=dict(samples=int(mask.sum()),native_srgb_change=float((out['rgb']-base).abs()[:,mask].mean()) if mask.any() else None)
            reference=np.asarray(__import__('PIL').Image.open(PARENT/f'{raw:03d}/delete_a37.png'))
            rounded=np.uint8(np.floor(np.clip(cpu(base).transpose(1,2,0),0,1)*255+.5));replay.append(int(np.abs(reference.astype(int)-rounded).max()))
            camera=self.camera(raw)
            kept=self.render(raw,keep=True)
            keep_base=self.n.render(camera,dynamic=True)
            assert out['extra'][2].abs().max().item()==0
            assert (kept['extra'].sum(0)-kept['alpha'][0]).max().item()<2e-6
            rec['no_edit_abs_change']=float((kept['rgb']-keep_base['rgb']).abs().mean())
            save(folder/f'{raw:03d}_noedit_parent.png',keep_base['rgb'])
            save(folder/f'{raw:03d}_noedit_road.png',kept['rgb'])
            np.savez_compressed(folder/f'{raw:03d}_layers.npz',
                                before=cpu(self.n.render(camera)['extra'][1:2]).astype(np.float16),
                                after=cpu(out['extra'][:2]).astype(np.float16),
                                keep=cpu(kept['extra']).astype(np.float16))
            rows.append(rec);save(folder/f'{raw:03d}_baseline.png',base);save(folder/f'{raw:03d}_repaired.png',out['rgb']);save(folder/f'{raw:03d}_road_alpha.png',out['extra'].expand(3,-1,-1));save(folder/f'{raw:03d}_reference.png',ref)
        assert max(replay)<=1,replay
        r4.write(folder/'metrics.json',dict(rows=rows,parent_unchanged=self.unchanged(),parent_png_max_uint8_difference=max(replay),source_frames=199,reference_is_BUILD_proxy=True,
            dynamic_target=[37],retained_aliases=ALIASES,road_and_gates_active_in_noedit=True,hard_image_route=False,quality_accepted=False,human_verdict=None))
        return rows

def select():
    n=Native();score=torch.zeros(len(n.static.get_xyz),device='cuda');guard=torch.zeros_like(score);counts=[]
    for raw in range(89,138):
        d=np.load(RUN/'inputs/views'/f'{raw:03d}.npz');kw=prepare_rasterization(n.camera(raw),n.static,n.args);kw['means2D']=torch.zeros_like(kw['means3D'])
        marker=torch.zeros((len(score),2),device='cuda',requires_grad=True);kw['features']=torch.cat([kw['features'],marker],1)
        dy=n.actor_kwargs(n.camera(raw),delete_a37=True)
        dy['features']=torch.cat((dy['features'],dy['features'].new_zeros((len(n.owners),2))),1)
        out=raster(n.camera(raw),n.args,n.static.active_sh_degree,merge_rows(kw,dy),n.bg)
        a=torch.as_tensor((d['fit_weight']>.05)&~d['marking_mask'],device='cuda');b=torch.as_tensor(d['other_vehicle_guard']|d['marking_mask'],device='cuda')
        (out['extra'][0,a].sum()+out['extra'][1,b].sum()).backward()
        score=torch.maximum(score,marker.grad[:,0]);guard=torch.maximum(guard,marker.grad[:,1]);counts.append(int(a.sum()))
        del kw,out,marker
    eligible=(score>1e-4)&~(guard>20*score);ids=np.flatnonzero(cpu(eligible))
    np.savez_compressed(RUN/'inputs/contributors.npz',ids=ids,fit_score=cpu(score[eligible]),guard_score=cpu(guard[eligible]))
    r4.write(RUN/'selection.json',dict(candidate_rows=len(ids),static_rows=len(score),protected_dominant_rows=int(((score>1e-4)&~eligible).sum()),fit_pixel_counts=counts,
        policy='all49 exact joint-delete alpha*T fit/vehicle-paint scores retaining a6; freeze rows whose protected score>20x fit score; not semantic ownership GT',parent_unchanged=n.unchanged(),updates=0))
    print((RUN/'selection.json').read_text(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');p.add_argument('--output',default='formal');p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume');p.add_argument('--evaluate',action='store_true');p.add_argument('--rgb-off',action='store_true');a=p.parse_args()
    if a.select:select();return
    dest=RUN/a.output;dest.mkdir(parents=True,exist_ok=False);torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0)
    e=Experiment();opt=torch.optim.Adam([e.repair.gate_raw],lr=e.settings['learning_rate'],eps=1e-15);startstep=0
    contract=dict(parent='frozen native ownership 10K',target=[37],settings=e.settings,formal_steps=1000,source_frames=199,only_static_opacity_gate_trainable=True,
        road_colour_geometry_opacity_fixed=True,native_time_preserved=True,road_time_stable=True)
    def state(step):return dict(step=step,repair=e.repair.state_dict(),optimizer=opt.state_dict(),sampler=sampler.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract,parent_identity=e.n.before)
    if a.resume:
        ck=torch.load(RUN/a.resume,map_location='cuda',weights_only=False);assert ck['parent_identity']==e.n.before;e.repair.load_state_dict(ck['repair']);opt.load_state_dict(ck['optimizer']);sampler.setstate(ck['sampler'])
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);startstep=ck['step']
        from compare_state import differences
        errors=differences(ck,state(startstep));r4.write(dest/'restore.json',dict(step=startstep,exact_full_state=not errors,differences=errors,updates=0));assert not errors,errors
    if a.rgb_off:
        assert a.evaluate
        with torch.no_grad():
            value=torch.tensor(e.ref['uncalibrated_linear'][e.repair.road.ids],device='cuda')
            colour=torch.where(value<=.0031308,value*12.92,1.055*value.clamp_min(1e-10)**(1/2.4)-.055)
            e.repair.road.dc.copy_(((colour-.5)/.28209479177387814)[:,None])
        r4.write(dest/'rgb_control.json',dict(updates=0,only_native_RGB_calibration_disabled=True,gates_from='formal/checkpoint001000.pt'))
    e.checks(dest)
    if a.evaluate:e.evaluate(dest,range(89,138),'all49');return
    r4.write(dest/'contract.json',contract);e.evaluate(dest,[101,105,113],'initial');fit=[r for r in range(89,138) if r not in [101,113,125] and e.view(r)['fit_weight'].sum()>0]
    start=time.time()
    with (dest/'training.jsonl').open('x') as log:
        for step in range(startstep+1,a.steps+1):
            raw=sampler.choice(fit);opt.zero_grad(set_to_none=True);loss,rec=e.loss(raw);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(e.repair.gate_raw.grad).all();opt.step()
            rec.update(step=step,raw=raw,loss=float(loss),seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());log.write(json.dumps(rec)+'\n');log.flush()
            if step==startstep+1 or step%100==0:print(json.dumps(rec),flush=True)
            if step in [32,100,500,1000] or step==a.steps:
                torch.save(state(step),dest/f'checkpoint{step:06d}.pt');e.evaluate(dest,[101,105,113],f'step{step:06d}')
    r4.write(dest/'receipt.json',dict(step=a.steps,updates=a.steps-startstep,seconds=time.time()-start,parent_unchanged=e.unchanged(),quality_accepted=False,human_verdict=None))
if __name__=='__main__':main()

"""Restore scene/sky photometry while freezing the complete current M2 edit.

Only shared static DC residuals, the existing 11x9 road colour field, and
directional-environment RGB can change. No geometry/opacity/actor changes.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,copy,json,random,sys,time
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,binary_erosion

ROOT=Path(__file__).resolve().parent.parent
C=ROOT.parent
ASSETS=C/'scene008_a6_asset_controls_20261105'
sys.path.insert(0,str(ASSETS/'source'))
from scene_runtime import Runtime,world_basis,raster
from sh_asset import SHAsset
from utils.sh_utils import eval_sh
from utils.hybrid30k_losses import photometric_rgb_loss
from utils.loss_utils import ssim

HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(n for n in range(89,138) if n not in HELD)
PILOTS=(105,113,130)
BOXES={'left_road':(0,205,215,320),'right_curb':(315,177,480,290),
       'building':(365,85,480,185),'trees':(15,65,160,180)}

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cpu(x):return x.detach().cpu().numpy()
def blue(x):return x[2]-.5*(x[0]+x[1])
def image(x):return Image.fromarray(np.uint8(np.clip(cpu(x).transpose(1,2,0),0,1)*255+.5))
def metric(x,ref,mask):
    z=(x-ref)[:,mask].double();n=int(mask.sum())
    return dict(count=n,mae=float(z.abs().mean()) if n else None,
                blue=float(blue(z).mean()) if n else None,
                absolute_blue=float(blue(z).abs().mean()) if n else None)
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class Sampler:
    def __init__(self):self.rng=random.Random(0);self.stack=[]
    def next(self):
        if not self.stack:
            self.stack=[(raw,delete) for raw in BUILD for delete in (False,True)];self.rng.shuffle(self.stack)
        return self.stack.pop()
    def state(self):return dict(rng=self.rng.getstate(),stack=self.stack.copy())
    def load(self,s):self.rng.setstate(s['rng']);self.stack=s['stack'].copy()

class Colour(torch.nn.Module):
    def __init__(self,rt):
        super().__init__();self.rt=rt
        self.static_dc=torch.nn.Parameter(torch.zeros((len(rt.n.static.get_xyz),3),device='cuda'))
        self.road_world=torch.nn.Parameter(torch.zeros((99,3),device='cuda'))
        self.env=copy.deepcopy(rt.n.env).requires_grad_(True)
        self.register_buffer('initial_env',rt.n.env.base.detach().clone())
        inverse=np.linalg.inv(np.asarray(rt.exp.meta['world_to_model'])@np.asarray(rt.exp.meta['coordinate_anchor']))
        xyz=cpu(rt.exp.repair.road.xyz);xy=(xyz@inverse[:3,:3].T+inverse[:3,3])[:,:2]
        self.register_buffer('road_basis',torch.tensor(world_basis(xy,rt.meta['road_knots']),device='cuda'))
        self.asset=SHAsset.from_geometry(ASSETS/'m2/complete_a6.npz').cuda().requires_grad_(False)
        self.asset_versions={k:(id(v),v._version) for k,v in vars(self.asset).items() if torch.is_tensor(v)}

    def draw(self,raw,delete,baseline=False):
        d=self.rt.raw(raw);scene=self.rt.scene(raw,self.asset,delete=delete)
        if baseline:return self.rt.render_rows(raw,scene)
        kw=scene['rows']
        with torch.no_grad():
            if kw['colors_precomp'] is None:
                directions=kw['means3D']-d['cam'].camera_center
                directions=directions/directions.norm(dim=1,keepdim=True)
                colors=(eval_sh(self.rt.n.static.active_sh_degree,kw['shs'].transpose(1,2),directions)+.5).clamp_min(0)
            else:colors=kw['colors_precomp'].detach()
        delta=colors.new_zeros(colors.shape)
        delta[:len(self.static_dc)]=1.5*torch.tanh(self.static_dc)
        road=kw['features'][:,4]>0
        delta[road]=self.road_basis@(.09*torch.tanh(self.road_world))
        kw=dict(kw,shs=None,colors_precomp=(colors+delta).clamp_min(0))
        out=raster(d['cam'],self.rt.pipe,self.rt.n.static.active_sh_degree,kw,self.rt.n.bg)
        env=self.env(d['cam'].get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        out['rgb']=out['rgb']+(1-out['alpha'])*env
        out['component_alpha_T']=out['extra'][:4]
        return out

    def unchanged_geometry(self):
        return self.rt.modelunchanged() and self.asset.assert_geometry_unchanged() and self.asset_versions=={
            k:(id(v),v._version) for k,v in vars(self.asset).items() if torch.is_tensor(v)}

class Experiment:
    def __init__(self):
        torch.set_num_threads(1);random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        self.rt=Runtime();self.model=Colour(self.rt);self.sampler=Sampler();self.step=0;self.cache={}
        self.opt=torch.optim.Adam([
            dict(params=[self.model.static_dc],lr=.0025),
            dict(params=[self.model.road_world],lr=.0025),
            dict(params=list(self.model.env.parameters()),lr=.001)],eps=1e-15)
        self.contract=dict(task='scene008_blue_fix_staticcap15_20261105_r1',phase='frozen_M2_appearance_continuation',
            initial_native_updates=10000,extra_formal_update_limit=10000,BUILD=BUILD,HELD=HELD,
            frozen=['all Gaussian geometry, opacity, temporal parameters, ordering','all M2 a6 material and geometry','a37 material','road geometry and opacity','saved static gates'],
            learned=['static shared RGB residual <=1.5/channel','existing continuous11x9 road RGB residual <=0.09/channel','directional environment RGB'],
            loss=dict(composite_RGB=1,sky_RGB=5,M2_actor_preservation=20,static_residual_prior=.001,road_field_prior=.01,environment_prior=.001),
            photometric='Inherited L1/SSIM .8/.2, masked at fixed observed domains; no per-frame colour parameters',
            domain='40 same-scene BUILD frames; keep uses original observed background excluding actors/shadow; delete also uses existing reobserved road reference',
            optimizer=dict(name='Adam',static_latent_lr=.0025,road_latent_lr=.0025,environment_lr=.001,eps=1e-15),
            rationale='Old global RGB and configured sky RGB terms were absent from the isolated-ownership training objective.',
            visibility_invariance='Fixed geometry/opacity; verify full alpha, depth and all four alpha*T groups against M2 at every evaluation.',
            no_gpu_shutdown=True,no_subagent=True,failure_ledger_refs=['V82-F01'],human_verdict=None)

    @torch.no_grad()
    def evidence(self,raw,delete):
        key=(raw,delete)
        if key in self.cache:return self.cache[key]
        d=self.rt.raw(raw);baseline=self.model.draw(raw,delete,baseline=True)
        actors=d['visible_reviewed'].any(0).bool() | (d['actor_labels']>0)
        actors=torch.nn.functional.max_pool2d(actors[None,None].float(),5,1,2)[0,0].bool()
        target=d['rgb'].clone();fit=~actors
        with np.load(C/'scene008_native_road1k_20261105_r1/inputs/views'/f'{raw:03d}.npz') as view:
            shadows=torch.as_tensor(view['shadow_proxy']|view['new_shadow_proxy'],device='cuda')
        fit&=~shadows
        if delete:
            road=d['road_edit'].bool() & (baseline['component_alpha_T'][0]<.01)
            road_rgb=d['road_edit_rgb'].permute(2,0,1).float()
            if road_rgb.max()>1:road_rgb=road_rgb/255.
            target[:,road]=road_rgb[:,road];fit|=road
            guard=(baseline['component_alpha_T'][0]>.01) | d['a6_visible'].bool()
        else:guard=actors
        sky_path=Path(self.rt.n.config['source_path'])/'sky/sky_0'/f'0008{raw:03d}.png'
        sky=np.asarray(Image.open(sky_path).resize((480,320),Image.Resampling.NEAREST))>0
        if sky.ndim==3:sky=sky.any(2)
        sky=torch.as_tensor(sky,device='cuda') & fit
        evidence=dict(target=target,fit=fit,sky=sky,guard=guard,base=baseline['rgb'].detach())
        self.cache[key]=evidence
        return evidence

    def update(self):
        raw,delete=self.sampler.next();e=self.evidence(raw,delete)
        self.opt.zero_grad(set_to_none=True);p=self.model.draw(raw,delete)
        def photo(mask):
            return photometric_rgb_loss(torch.where(mask[None],p['rgb'],0),torch.where(mask[None],e['target'],0),ssim,.2)
        terms=dict(composite_RGB=photo(e['fit']),sky_RGB=5*photo(e['sky']),
                   preservation=20*((p['rgb']-e['base']).square()[:,e['guard']].mean() if e['guard'].any() else p['rgb'].sum()*0),
                   static_prior=.001*(1.5*torch.tanh(self.model.static_dc)).square().mean(),
                   road_prior=.01*(.09*torch.tanh(self.model.road_world)).square().mean(),
                   env_prior=.001*(self.model.env.base-self.model.initial_env).square().mean())
        total=sum(terms.values());assert torch.isfinite(total)
        total.backward()
        if self.step%50==0:
            for v in self.model.parameters():
                if v.grad is not None:assert torch.isfinite(v.grad).all()
        self.opt.step();self.step+=1
        return dict(step=self.step,raw=raw,delete=delete,loss=float(total.detach()),
                    terms={k:float(v.detach()) for k,v in terms.items()},allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved())

    def state(self):
        return dict(step=self.step,model=self.model.state_dict(),optimizer=self.opt.state_dict(),sampler=self.sampler.state(),
                    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
                    contract=self.contract,parent=self.rt.meta)

    def save(self,path):
        temp=Path(str(path)+'.tmp');torch.save(self.state(),temp);temp.replace(path)

    def load(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['contract']==self.contract and saved['parent']==self.rt.meta
        # Frozen asset buffers retain version counters used by the renderer's
        # invariant checks. Verify their exact contents without rewriting them.
        current_model=self.model.state_dict()
        mutable={'static_dc','road_world','env.base'}
        assert set(saved['model'])==set(current_model)
        for key in set(current_model)-mutable:
            assert equal(saved['model'][key],current_model[key]),f'Frozen dependency differs: {key}'
        loaded=self.model.load_state_dict({key:saved['model'][key] for key in mutable},strict=False)
        assert set(loaded.missing_keys)==set(current_model)-mutable and not loaded.unexpected_keys
        self.opt.load_state_dict(saved['optimizer'])
        self.step=saved['step'];self.sampler.load(saved['sampler'])
        random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        current=self.state();assert equal(saved,current),'Full training state restore mismatch'

    @torch.no_grad()
    def evaluate(self,dest,raws):
        start=time.monotonic();dest=Path(dest);dest.mkdir(exist_ok=False);rows=[];yy,xx=np.indices((320,480));maxima=dict(alpha=0.,depth=0.,groups=0.)
        for raw in raws:
            d=self.rt.raw(raw);ref=d['rgb'];actors=cpu(d['actor_labels'])>0
            regions={name:torch.as_tensor((xx>=a)&(xx<c)&(yy>=b)&(yy<e)&~actors,device='cuda') for name,(a,b,c,e) in BOXES.items()}
            regions.update(a6=d['a6_visible'].bool(),observed_background=~d['visible_reviewed'].any(0).bool())
            panels=[]
            for delete in (False,True):
                before=self.model.draw(raw,delete,baseline=True);after=self.model.draw(raw,delete)
                for key,a,b in [('alpha',before['alpha'],after['alpha']),('depth',before['depth_sum'],after['depth_sum']),('groups',before['component_alpha_T'],after['component_alpha_T'])]:
                    maxima[key]=max(maxima[key],float((a-b).abs().max()))
                row=dict(raw=raw,delete=delete,split='HELD' if raw in HELD else 'BUILD',regions={})
                for name,m in regions.items():row['regions'][name]=dict(before=metric(before['rgb'],ref,m),after=metric(after['rgb'],ref,m))
                road=d['road_edit'].bool() if delete else d['road'].bool();road&=~d['visible_reviewed'][0].bool()
                road_rgb=d['road_edit_rgb'].permute(2,0,1).float() if delete else ref
                if road_rgb.max()>1:road_rgb=road_rgb/255.
                row['regions']['road']=dict(before=metric(before['rgb'],road_rgb,road),after=metric(after['rgb'],road_rgb,road))
                rows.append(row);panels.extend([before['rgb'],after['rgb']])
            canvas=Image.new('RGB',(1920,348),'white')
            for j,(p,label) in enumerate(zip(panels,[f'R1 baseline (M2) / step0 / raw{raw} / keep',f'StaticCap1.5 / step{self.step} / raw{raw} / keep',f'R1 baseline (M2) / step0 / raw{raw} / delete',f'StaticCap1.5 / step{self.step} / raw{raw} / delete'])):
                canvas.paste(image(p),(j*480,28));ImageDraw.Draw(canvas).text((j*480+8,7),label,fill='black')
            canvas.save(dest/f'{raw:03d}.jpg',quality=95,subsampling=0)
        assert max(maxima.values())<3e-6,maxima
        assert self.model.unchanged_geometry()
        pooled={}
        for state in (False,True):
            selected=[r for r in rows if r['delete']==state];key='delete' if state else 'keep';pooled[key]={}
            for region in regions.keys()|{'road'}:
                pooled[key][region]={}
                for side in ('before','after'):
                    vals=[r['regions'][region][side] for r in selected if r['regions'][region][side]['count']]
                    count=sum(v['count'] for v in vals)
                    pooled[key][region][side]={field:sum(v[field]*v['count'] for v in vals)/count if count else None for field in ('mae','blue','absolute_blue')}
                pooled[key][region]['max_frame_rgb_increase']=max((r['regions'][region]['after']['mae']-r['regions'][region]['before']['mae'] for r in selected if r['regions'][region]['before']['count']),default=0)
        checks=dict(geometry_visibility_invariant=max(maxima.values())<3e-6)
        for state,p in pooled.items():
            checks[state+'_a6_RGB']=p['a6']['after']['mae']<=p['a6']['before']['mae']+1e-4 and p['a6']['max_frame_rgb_increase']<=5e-4
            checks[state+'_background_RGB']=p['observed_background']['after']['mae']<=p['observed_background']['before']['mae']+1e-4
            checks[state+'_road_RGB']=p['road']['after']['mae']<=p['road']['before']['mae']+1e-4 and p['road']['max_frame_rgb_increase']<=.001
            for region in ('left_road','right_curb','building'):
                checks[state+'_'+region+'_blue']=abs(p[region]['after']['blue'])<=max(.002,abs(p[region]['before']['blue'])*.3)
        result=dict(step=self.step,raws=list(raws),rows=rows,pooled=pooled,maxima=maxima,checks=checks,
                    automatic_gate_passed=all(checks.values()),human_verdict=None,seconds=time.monotonic()-start,
                    frozen_geometry_and_M2_asset=True)
        write(dest/'metrics.json',result)
        print(json.dumps(dict(evaluation=str(dest),step=self.step,pooled=pooled,checks=checks,seconds=result['seconds'])),flush=True)
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','formal','resume_check'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--steps',type=int,default=1000);p.add_argument('--seconds',type=float,default=600)
    args=p.parse_args();dest=ROOT/args.output;dest.mkdir(exist_ok=False);start=time.monotonic();e=Experiment()
    write(dest/'contract.json',e.contract)
    if args.resume:e.load(Path(args.resume));write(dest/'restore_audit.json',dict(status='PASS_EXACT_MODEL_ADAM_SAMPLER_RNG_RESTORE',step=e.step))
    if e.step==0:e.evaluate(dest/'initial_pilots',PILOTS)
    last={};stop=e.step+2 if args.mode=='resume_check' else args.steps
    while (time.monotonic()-start<args.seconds) if args.mode=='smoke' else (e.step<stop):
        last=e.update()
        if e.step%100==0:
            last['seconds']=time.monotonic()-start
            with (dest/'progress.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
            write(dest/'progress.json',last);print(json.dumps(last),flush=True)
        if e.step%1000==0:
            if args.mode!='smoke' or e.step==1000:e.save(dest/f'checkpoint{e.step:06d}.pt')
            e.evaluate(dest/f'eval{e.step:06d}',PILOTS if args.mode=='smoke' else tuple(range(89,138)))
    e.save(dest/f'checkpoint{e.step:06d}.pt')
    result=e.evaluate(dest/f'final{e.step:06d}',PILOTS if args.mode=='resume_check' else tuple(range(89,138)))
    assert e.model.unchanged_geometry()
    write(dest/'returned.json',dict(status='PASS_COMPLETE_STATE_AND_EVALUATION',mode=args.mode,step=e.step,seconds=time.monotonic()-start,
          checkpoint=str(dest/f'checkpoint{e.step:06d}.pt'),formal_updates=e.step if args.mode=='formal' else 0,
          smoke_updates=e.step if args.mode=='smoke' else 0,automatic_gate_passed=result['automatic_gate_passed'],
          max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),human_verdict=None))
if __name__=='__main__':main()

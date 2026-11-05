"""Local, time-invariant 3D road repair over an unchanged B(t) renderer.

No shadow component, no parent update, no global timestamp substitution.
The local branch removes only thin road-near native rows before a joint depth
sort. Output routing is limited to measured road interior, with other vehicles
protected. Above-road native foreground remains in that depth sort.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import copy, json, random, time, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from editor_common import Native, cpu, save, prepare_rasterization, road_kwargs, merge_rows, raster

RUN=Path(__file__).resolve().parent.parent
INPUT=RUN/'inputs_final'
V3=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')

def linear(x):
    x=x.clamp(0,1)
    return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

class StableRoad(torch.nn.Module):
    def __init__(self,reference,meta):
        super().__init__(); self.meta=meta
        ids=np.flatnonzero(reference['active']);self.ids=ids
        tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
        self.register_buffer('xyz',tensor(reference['model_xyz'][ids]))
        self.register_buffer('scaling',tensor(reference['gaussian_sigma_m'][ids]*meta['meter_to_model']))
        normal=reference['normal_world'][ids]@np.asarray(meta['world_to_model'])[:3,:3].T/meta['meter_to_model']
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        q=np.c_[1+normal[:,2],-normal[:,1],normal[:,0],np.zeros(len(normal))]
        q/=np.linalg.norm(q,axis=1)[:,None]
        self.register_buffer('rotation',tensor(q))
        self.register_buffer('active',torch.ones(len(ids),dtype=torch.bool,device='cuda'))
        self.register_buffer('initial_dc',tensor((reference['reference_srgb'][ids]-.5)/.28209479177387814)[:,None])
        self.dc=torch.nn.Parameter(self.initial_dc.clone())
        self.opacity=torch.nn.Parameter(torch.full((len(ids),1),4.6,device='cuda'))
    def colors(self):return .5+.28209479177387814*self.dc[:,0]
    def output_mask(self):return self.active
    def prior(self):return .025*(self.dc-self.initial_dc).abs().mean()+.001*(self.opacity.sigmoid()-.99).abs().mean()

class Experiment:
    def __init__(self):
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable: do not start training')
        self.contract=json.loads((RUN/'contract.json').read_text())
        self.n=Native();self.reference=dict(np.load(INPUT/'road_reference_199.npz'))
        assert all(not value.requires_grad for parent in [self.n.static,self.n.dynamic,self.n.env]
            for value in vars(parent).values() if torch.is_tensor(value))
        self.meta=json.loads(V3.read_text()); self.road=StableRoad(self.reference,self.meta)
        self.scale=self.meta['meter_to_model']; self.inv=np.linalg.inv(np.asarray(self.meta['world_to_model'])@np.asarray(self.meta['coordinate_anchor']))
        self.view_cache={};self.cameras={};self.tree=cKDTree(self.reference['local_xyz'][:,:2])
        self.q=cpu(self.n.static.get_rotation);self.scales=cpu(self.n.static.get_scaling)/self.scale
        self.dark=cpu(.5+.28209479177387814*self.n.static.get_features[:,0]).max(1)<.05
        self.cache=RUN/'road_row_cache';self.cache.mkdir(exist_ok=True)
        # All parent tensors and the B renderer remain immutable.
        self.versions={name:getattr(self.n.static,name)._version for name in
            ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity']}
    def view(self,raw):
        if raw not in self.view_cache:self.view_cache[raw]=dict(np.load(INPUT/'views_current'/('%03d.npz'%raw)))
        return self.view_cache[raw]
    def camera(self,raw):
        if raw not in self.cameras:self.cameras[raw]=self.n.camera(raw)
        return self.cameras[raw]
    def ground_rows(self,camera,kwargs):
        path=self.cache/('%s_%.9f.npz'%(camera.image_name,camera.timestamp))
        if path.exists():return torch.tensor(np.load(path)['ground'],device='cuda')
        xyz=cpu(kwargs['means3D']);local=xyz@self.inv[:3,:3].T+self.inv[:3,3]
        dist,ids=self.tree.query(local[:,:2]);normal=self.reference['normal_local'][ids]
        normal_model=normal@np.linalg.inv(self.inv[:3,:3]).T/self.scale
        normal_model/=np.linalg.norm(normal_model,axis=1)[:,None]
        w,x,y,z=self.q.T
        # R^T n: sigma along measured road normal, in metres.
        nx,ny,nz=normal_model.T
        rotated=np.stack(((1-2*(y*y+z*z))*nx+2*(x*y+w*z)*ny+2*(x*z-w*y)*nz,
            2*(x*y-w*z)*nx+(1-2*(x*x+z*z))*ny+2*(y*z+w*x)*nz,
            2*(x*z+w*y)*nx+2*(y*z-w*x)*ny+(1-2*(x*x+y*y))*nz),1)
        sigma=np.linalg.norm(rotated*self.scales,axis=1)
        height=((local-self.reference['local_xyz'][ids])*normal).sum(1)
        thin=(np.abs(height)<.20)&(3*sigma<.30)
        thin|=self.dark&(np.abs(height)<.40)&(3*sigma<.50)
        ground=(dist<=.35)&self.reference['active'][ids]&thin
        np.savez_compressed(path,ground=ground)
        return torch.as_tensor(ground,device='cuda')
    def render(self,raw,camera=None):
        camera=camera or self.camera(raw);d=self.view(raw)
        # Current timestamp goes into both native baseline and repair branch.
        kwargs=prepare_rasterization(camera,self.n.static,self.n.args)
        kwargs['means2D']=torch.zeros_like(kwargs['means3D'])
        ground=self.ground_rows(camera,kwargs)
        kwargs['mask']=kwargs['mask']&~ground
        channels=kwargs['shs'].shape[1] if kwargs['shs'] is not None else 1
        rr=road_kwargs(camera,self.road,self.n.args,channels,precomputed_color=kwargs['colors_precomp'] is not None)
        kwargs['features']=torch.cat((kwargs['features'],torch.zeros_like(kwargs['features'][:,:1])),1)
        rr['features']=torch.cat((rr['features'],torch.ones_like(rr['features'][:,:1])),1)
        merged=merge_rows(kwargs,rr)
        out=raster(camera,self.n.args,self.n.static.active_sh_degree,merged,self.n.bg)
        env=self.n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        branch=out['rgb']+(1-out['alpha'])*env
        with torch.no_grad():base=self.n.render(camera,dynamic=False)['rgb']
        route=torch.tensor(d['route'],device='cuda')
        # Above-road static foreground remains depth-ordered with the road.
        result=torch.where(route[None],branch,base)
        return dict(rgb=result,base=base,branch=branch,contribution=out['extra'],route=route,
            suppressed_road_rows=int(ground.sum()))
    def unchanged(self):
        return self.n.unchanged() and all(getattr(self.n.static,k)._version==v for k,v in self.versions.items())
    @torch.no_grad()
    def evaluate(self,dest,tag,raws):
        rows=[];dest=Path(dest)/tag;dest.mkdir(parents=True,exist_ok=True)
        def mae(a,b,mask):return float((a-b).abs()[:,mask].mean()) if bool(mask.any()) else None
        for raw in raws:
            d=self.view(raw);o=self.render(raw);pred=linear(o['rgb']);base=linear(o['base'])
            target=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
            valid=torch.tensor(d['valid'],device='cuda');shadow=torch.tensor(d['shadow_proxy'],device='cuda')
            outside=~o['route'];outside_max=float((o['rgb']-o['base']).abs()[:,outside].max()) if bool(outside.any()) else 0.
            assert outside_max==0.,outside_max
            row=dict(raw=raw,outside_maxabs=outside_max,suppressed_road_rows=o['suppressed_road_rows'],actors={})
            for alias in [5,37]:
                domain=torch.tensor(d['domain_a%d'%alias],device='cuda');v=domain&valid;s=domain&shadow
                row['actors'][str(alias)]=dict(domain_pixels=int(domain.sum()),qualified_pixels=int(v.sum()),
                    route_pixels=int((domain&o['route']).sum()),shadow_proxy_pixels=int(s.sum()),
                    road_MAE=mae(pred,target,v),baseline_road_MAE=mae(base,target,v),
                    shadow_MAE=mae(pred,target,s),baseline_shadow_MAE=mae(base,target,s),
                    dark_residual_fraction=float(((target-pred).mean(0)[s]>.06).float().mean()) if bool(s.any()) else None,
                    baseline_dark_residual_fraction=float(((target-base).mean(0)[s]>.06).float().mean()) if bool(s.any()) else None,
                    actual_road_contribution_mean=float(o['contribution'][0,v].mean()) if bool(v.any()) else None)
            save(dest/('%03d_baseline_delete5.png'%raw),o['base']);save(dest/('%03d_repaired_delete5.png'%raw),o['rgb'])
            save(dest/('%03d_road_contribution.png'%raw),o['contribution'].expand(3,-1,-1))
            if raw in [101,105,113]:save(dest/('%03d_original_full.png'%raw),self.n.render(self.camera(raw),dynamic=True)['rgb'])
            rows.append(row)
        aggregates={}
        for alias in ['5','37']:
            aggregate={}
            for key,countkey in [('road_MAE','qualified_pixels'),('baseline_road_MAE','qualified_pixels'),
                ('shadow_MAE','shadow_proxy_pixels'),('baseline_shadow_MAE','shadow_proxy_pixels')]:
                chosen=[(r['raw'],r['actors'][alias][key],r['actors'][alias][countkey]) for r in rows if r['actors'][alias][key] is not None]
                denominator=sum(n for raw,value,n in chosen)
                aggregate[key]=dict(valid_frames=len(chosen),pixels=denominator,
                    frame_macro_mean=float(np.mean([value for raw,value,n in chosen])) if chosen else None,
                    sample_weighted_mean=sum(value*n for raw,value,n in chosen)/denominator if denominator else None,
                    worst_frame_raw=max(chosen,key=lambda x:x[1])[0] if chosen else None,
                    worst_frame_MAE=max((value for raw,value,n in chosen),default=None))
            aggregates[alias]=aggregate
        write(dest/'metrics.json',dict(rows=rows,aggregates=aggregates,reference_is_proxy=True,all_199_build_inputs=True,
            no_unseen_generalization_claim=True,original_B_unchanged=self.unchanged()))
        return rows
    @torch.no_grad()
    def temporal_audit(self,dest):
        raw=105;c=self.camera(raw);outputs=[];bases=[];road_only=[]
        for t in [89,105,137]:
            camera=copy.copy(c);camera.timestamp=self.camera(t).timestamp
            o=self.render(raw,camera);outputs.append(o['rgb']);bases.append(o['base'])
            rr=road_kwargs(camera,self.road,self.n.args,precomputed_color=True)
            road_only.append(raster(camera,self.n.args,0,rr,self.n.bg)['rgb'])
        outside=~torch.tensor(self.view(raw)['route'],device='cuda')
        report=dict(fixed_camera_raw=raw,timestamps_raw=[89,105,137],
            road_branch_RGB_maxabs_vs_first=[float((r-road_only[0]).abs().max()) for r in road_only],
            original_static_full_MAE_vs_first=[float((r-bases[0]).abs().mean()) for r in bases],
            final_vs_current_native_outside_maxabs=[float((a-b).abs()[:,outside].max()) for a,b in zip(outputs,bases)],
            global_static_timestamp_override=False,original_time_parameters_updated=False)
        assert max(report['road_branch_RGB_maxabs_vs_first'])==0
        assert max(report['final_vs_current_native_outside_maxabs'])==0
        write(Path(dest)/'temporal_audit.json',report)

def train(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    e=Experiment();opt=torch.optim.Adam([dict(params=[e.road.dc],lr=.0025),dict(params=[e.road.opacity],lr=.05)],eps=1e-15)
    sampler=torch.Generator().manual_seed(20261103);start=0
    def state(step):
        return dict(step=step,road=e.road.state_dict(),optimizer=opt.state_dict(),sampler=sampler.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),parent_identity=e.n.before,contract=e.contract,
            reference_sha256=e.contract['reference_sha256'])
    if args.resume:
        ck=torch.load(args.resume,map_location='cuda',weights_only=False)
        assert ck['parent_identity']==e.n.before and ck['contract']==e.contract
        e.road.load_state_dict(ck['road']);opt.load_state_dict(ck['optimizer']);sampler.set_state(ck['sampler'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
        np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);start=ck['step']
        exact=all(torch.equal(v,ck['road'][k]) for k,v in e.road.state_dict().items())
        assert exact
        write(dest/'restore_loaded_exact.json',dict(step=start,road_state_exact=True,optimizer_loaded=True,RNG_and_sampler_loaded=True))
    else:
        e.evaluate(dest,'initial',[101,105,113])
    started=time.time()
    with (dest/('loss_from%d.jsonl'%start)).open('x') as log:
        for step in range(start+1,args.steps+1):
            raw=89+int(torch.randint(49,(1,),generator=sampler));d=e.view(raw)
            opt.zero_grad(set_to_none=True);o=e.render(raw)
            mask=torch.tensor(d['valid']&d['route'],device='cuda')
            if not bool(mask.any()):raise RuntimeError('Empty qualified training road at raw%d'%raw)
            target=torch.tensor(d['reference_linear'].transpose(2,0,1),device='cuda')
            photo=(linear(o['rgb'])-target).abs()[:,mask].mean()
            coverage=(.90-o['contribution'][0,mask]).clamp_min(0).mean()
            loss=photo+.10*coverage+e.road.prior();assert torch.isfinite(loss)
            loss.backward()
            assert e.road.dc.grad is not None and e.road.opacity.grad is not None
            assert torch.isfinite(e.road.dc.grad).all() and torch.isfinite(e.road.opacity.grad).all()
            opt.step()
            record=dict(step=step,raw=raw,loss=float(loss),road_photo=float(photo),contribution_shortfall=float(coverage),
                seconds=time.time()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n');log.flush()
            if step%100==0 or step==start+1:print(json.dumps(record),flush=True)
            if step in [100,1000,1001] or step==args.steps:
                torch.save(state(step),dest/('checkpoint%06d.pth'%step))
                e.evaluate(dest,'step%06d'%step,[101,105,113])
            del o,loss,photo,coverage
    assert e.unchanged()
    write(dest/'receipt.json',dict(completed=True,start_step=start,step=args.steps,updates=args.steps-start,
        seconds=time.time()-started,parent_unchanged=True,trainable_parameters=['road.dc','road.opacity'],
        quality_accepted=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume');p.add_argument('--checkpoint')
    a=p.parse_args()
    if a.stage=='train':train(a)
    else:
        e=Experiment()
        if a.checkpoint:
            ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
            assert ck['contract']==e.contract;e.road.load_state_dict(ck['road'])
        e.evaluate(a.output,'all49',range(89,138));e.temporal_audit(a.output)

if __name__=='__main__':main()

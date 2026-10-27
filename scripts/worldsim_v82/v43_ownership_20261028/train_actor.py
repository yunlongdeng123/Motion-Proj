"""V43 measured actor-local rebuild: explicit instance + rigid tracked motion.

Frozen original static/environment; new target only. This arm tests structure,
not background completion or superiority of joint optimization. All original
checkpoints and inference assignments remain read-only.
"""
import argparse,json,os,sys,time,random,hashlib,signal,math,gc
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
ROOT=OLD.parent/'v43_ownership_20261028'
sys.path.insert(0,str(OLD/'code'))
from render_motion_suite import NativeMotion,prepare_rasterization,helpers,torch,cpu
from gpu_slot import exclusive
from segmentation import write

class Actor(torch.nn.Module):
    def __init__(self,data,constrained=True):
        super().__init__();self.constrained=constrained
        self.register_buffer('origin',torch.from_numpy(data['xyz']).cuda())
        self.register_buffer('origin_sigma',torch.from_numpy(data['sigma']).cuda())
        self.register_buffer('box_half',torch.from_numpy(data['box_half'].astype('float32')).cuda())
        self.xyz=torch.nn.Parameter(self.origin.clone())
        self.log_scale=torch.nn.Parameter(self.origin_sigma[:,None].repeat(1,3).log())
        self.dc=torch.nn.Parameter(torch.from_numpy(data['rgb']).cuda().float().sub(.5).div(.28209479177387814)[:,None,:])
        self.opacity=torch.nn.Parameter(torch.full((len(self.xyz),1),-1.38629436112,device='cuda'))
        self.quat=torch.nn.Parameter(torch.tensor([1.,0.,0.,0.],device='cuda').repeat(len(self.xyz),1))
    def kwargs(self,n,camera,frame,edit=None):
        R=np.array(frame['actor_rotation_world']);center=np.array(frame['actor_center_world'])
        if edit:
            E=np.array(edit.get('rotation',np.eye(3)));c=np.array(edit.get('center',center));R=E@R;center=c
        M=n.T[:3,:3]@R;offset=n.T[:3,:3]@center+n.T[:3,3]
        xyz=self.xyz@torch.tensor(M.T,dtype=torch.float32,device='cuda')+torch.tensor(offset,dtype=torch.float32,device='cuda')
        q=np.roll(Rotation.from_matrix(M/n.scale).as_quat(),1)
        base=torch.tensor(q,dtype=torch.float32,device='cuda');local=self.quat/self.quat.norm(dim=1,keepdim=True).clamp_min(1e-8)
        from render_motion_suite import quat_product
        rotation=quat_product(base,local)
        z=torch.cat((xyz,torch.ones_like(xyz[:,:1])),1)@camera.world_view_transform[:,2:3]
        return dict(means3D=xyz,means2D=torch.zeros_like(xyz),shs=torch.cat((self.dc,self.dc.new_zeros((len(xyz),n.static.get_features.shape[1]-1,3))),1),colors_precomp=None,features=torch.cat((z,torch.ones_like(z),torch.ones_like(z)),1),opacities=self.opacity.sigmoid(),scales=self.log_scale.exp()*n.scale,rotations=rotation,cov3D_precomp=None,mask=torch.ones(len(xyz),dtype=torch.bool,device='cuda'))
    @torch.no_grad()
    def project(self,opt):
        if not self.constrained:return
        before=self.xyz.clone();delta=self.xyz-self.origin
        self.xyz.copy_(self.origin+delta*(.25/delta.norm(dim=1,keepdim=True).clamp_min(.25)))
        envelope=torch.maximum(self.box_half+.1,self.origin.abs());self.xyz.copy_(torch.maximum(torch.minimum(self.xyz,envelope),-envelope))
        old=self.log_scale.clone();self.log_scale.clamp_(min=math.log(.01));self.log_scale.copy_(torch.minimum(self.log_scale,(self.origin_sigma[:,None]*1.5).log()))
        for param,mask in [(self.xyz,(before-self.xyz).abs()>1e-7),(self.log_scale,(old-self.log_scale).abs()>1e-7)]:
            for value in opt.state.get(param,{}).values():
                if torch.is_tensor(value) and value.shape==param.shape:value[mask]=0
        self.opacity.clamp_(-10,8)
    @torch.no_grad()
    def geometry(self):
        return dict(points=len(self.xyz),outside_exact_box=int((self.xyz.abs()>self.box_half).any(1).sum()),outside_preserved_initial_envelope=int((self.xyz.abs()>torch.maximum(self.box_half+.1,self.origin.abs())+1e-5).any(1).sum()),max_drift_m=float((self.xyz-self.origin).norm(dim=1).max()),max_sigma_m=float(self.log_scale.exp().max()),actor_local_rigid=True)

def render(n,actor,frame,i,static=True,edit=None):
    row=dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=frame['model_time']);camera=n.camera(row,{})
    kw=actor.kwargs(n,camera,frame,edit)
    if static:
        with torch.no_grad():s=prepare_rasterization(camera,n.static,n.args);s['means2D']=torch.zeros_like(s['means3D']);s['features']=torch.cat((s['features'],torch.zeros_like(s['features'][:,:1])),1)
        kw=helpers.merge_rows(s,kw)
    out=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg)
    if static:
        with torch.no_grad():env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        out['rgb']=out['rgb']+(1-out['alpha'])*env
    return out

def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def evaluate(out,n,actor,frames,observations,step):
    dest=out/'evaluation'/('%06d'%step);dest.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.no_grad():
        for i,f in enumerate(frames):
            result=render(n,actor,f,i);target=render(n,actor,f,i,static=False)
            gt,body,depth=observations[i];err=(result['rgb']-gt).square();inside=body[None].expand_as(err)
            row=dict(raw=f['raw'],psnr=-10*math.log10(max(float(err.mean()),1e-12)),target_psnr=-10*math.log10(max(float(err[inside].mean()),1e-12)),body_alpha_mae=float((target['alpha'][0]-body.float()).abs().mean()))
            rows.append(row)
            for name,t in [('rgb',result['rgb']),('actor',target['rgb'])]:
                (dest/name).mkdir(exist_ok=True);Image.fromarray(np.round(np.clip(cpu(t).transpose(1,2,0),0,1)*255).astype('uint8')).save(dest/name/('%03d.png'%f['raw']))
            if i in [9,24,39]:
                E=Rotation.from_euler('z',15,degrees=True).as_matrix();turned=render(n,actor,f,i,edit=dict(rotation=E));(dest/'turn15_stress').mkdir(exist_ok=True);Image.fromarray(np.round(np.clip(cpu(turned['rgb']).transpose(1,2,0),0,1)*255).astype('uint8')).save(dest/'turn15_stress'/('%03d.png'%f['raw']))
        assert n.unchanged()
    write(dest/'summary.json',dict(step=step,frames=rows,geometry=actor.geometry(),mean_psnr=float(np.mean([x['psnr'] for x in rows])),mean_target_psnr=float(np.mean([x['target_psnr'] for x in rows])),frozen_background_includes_unrepaired_vehicle_residual=True,unseen_sides_not_completed=True,quality_accepted=False,human_verdict=None))
    return str(dest/'summary.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--steps',type=int,default=10000);p.add_argument('--smoke-seconds',type=float,default=0);p.add_argument('--resume',type=Path);p.add_argument('--unconstrained',action='store_true');p.add_argument('--entry-only',action='store_true');args=p.parse_args()
    assert args.steps<=20000 or args.smoke_seconds
    auth=json.loads((ROOT/'authorization.json').read_text());assert auth['allow_v43_training'] and not (ROOT/'USER_STOP').exists()
    args.output.mkdir(parents=True,exist_ok=False);status=dict(context=args.context,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',formal_steps=args.steps,extra_smoke=bool(args.smoke_seconds),quality_accepted=False,human_verdict=None)
    def update(**kw):status.update(kw);write(args.output/'status.json',status)
    update();stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    with exclusive():
        torch.set_num_threads(1);random.seed(43);np.random.seed(43);torch.manual_seed(43);torch.cuda.manual_seed_all(43)
        started=time.monotonic();dataset=ROOT/'datasets'/args.context;manifest=json.loads((dataset/'manifest.json').read_text());frames=manifest['frames'];data=np.load(dataset/'seeds.npz');actor=Actor(data,not args.unconstrained)
        n=NativeMotion(OLD/'native_scene_training_v1'/args.context/'formal30k',30000)
        observations=[]
        for f in frames:
            d=np.load(dataset/('%03d.npz'%f['raw']));observations.append((torch.from_numpy(d['rgb'].transpose(2,0,1).astype(np.float32)/255).cuda(),torch.from_numpy(d['body']).cuda(),torch.from_numpy(d['depth']).cuda()))
        opt=torch.optim.Adam([dict(params=[actor.xyz],lr=2e-4),dict(params=[actor.log_scale],lr=2e-3),dict(params=[actor.dc],lr=2e-3),dict(params=[actor.opacity],lr=5e-3),dict(params=[actor.quat],lr=5e-4)],eps=1e-8)
        stack=[];first=0;restore_checks=None
        if args.resume:
            saved=torch.load(str(args.resume),map_location='cuda');actor.load_state_dict(saved['actor']);opt.load_state_dict(saved['optimizer']);stack=saved['sampler'];first=saved['step']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
            restore_checks=dict(actor=equal(actor.state_dict(),saved['actor']),Adam=equal(opt.state_dict(),saved['optimizer']),sampler=equal(stack,saved['sampler']),python_rng=equal(random.getstate(),saved['python_rng']),numpy_rng=equal(np.random.get_state(),saved['numpy_rng']),torch_rng=equal(torch.get_rng_state(),saved['torch_rng'].cpu()),CUDA_rng=equal(torch.cuda.get_rng_state_all(),[x.cpu() for x in saved['cuda_rng']]))
            write(args.output/'restore_verification.json',dict(checks=restore_checks,all_passed=all(restore_checks.values()),resumed_step=first));assert all(restore_checks.values())
        write(args.output/'contract.json',dict(user_budget=args.steps,initialization='all49 measured target returns, actor-local 6cm voxel dedup, original seed identities retained',method='rigid tracked actor-local field, R15-style observed anchors and growth budgets',background='frozen V42 static/env, no background completion in this arm',constraints=not args.unconstrained,loss_weights=dict(composite_RGB=1,body_RGB=5,body_alpha=.1,sparse_LiDAR_depth=.05,anchor=.01),GT_boxes_filter_points=False,actor_box_pose_used_for_geometry_envelope=True,source_dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),backend='none in structure arm; native render diagnostics first',joint_completion_claim=False))
        def save(step):
            torch.cuda.synchronize();path=args.output/('checkpoint%06d.pth'%step);tmp=path.with_suffix('.tmp')
            state=dict(step=step,actor=actor.state_dict(),optimizer=opt.state_dict(),sampler=list(stack),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),contract=json.loads((args.output/'contract.json').read_text()))
            torch.save(state,str(tmp));tmp.replace(path)
            # Immediate CPU readability plus shapes; full state equality is checked on real restore.
            checked=torch.load(str(path),map_location='cpu');assert checked['step']==step and equal(checked['actor'],actor.state_dict());del checked
            evaluation=evaluate(args.output,n,actor,frames,observations,step)
            write(args.output/'latest_complete_node.json',dict(step=step,checkpoint=str(path),checkpoint_bytes=path.stat().st_size,evaluation=evaluation,frames=49,geometry=actor.geometry()))
            return path
        update(phase='training',seed_points=len(actor.xyz),step=first)
        active_started=time.monotonic();log=(args.output/'steps.jsonl').open('a');max_steps=100000 if args.smoke_seconds else args.steps
        if args.entry_only:max_steps=first+2
        for step in range(first+1,max_steps+1):
            if not stack:stack=list(range(49));random.shuffle(stack)
            i=stack.pop();gt,body,depth=observations[i];f=frames[i];opt.zero_grad(set_to_none=True)
            composite=render(n,actor,f,i);isolated=render(n,actor,f,i,static=False)
            rgb=(composite['rgb']-gt).abs().mean();body_rgb=((isolated['rgb']-gt).abs()*body[None]).sum()/(body.sum()*3).clamp_min(1)
            alpha=(isolated['alpha'][0]-body.float()).abs().mean();valid=torch.isfinite(depth)&body&(isolated['alpha'][0].detach()>.1)
            expected=isolated['depth_sum'][0]/isolated['alpha'][0].clamp_min(1e-6)/n.scale
            lidar=(expected[valid]-depth[valid]).abs().clamp_max(10).mean() if bool(valid.any()) else expected.sum()*0
            anchor=((actor.xyz-actor.origin).square().sum(1)).mean()
            loss=rgb+5*body_rgb+.1*alpha+.05*lidar+.01*anchor
            assert torch.isfinite(loss),step;loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters()),step
            opt.step();actor.project(opt)
            record=dict(step=step,camera_index=i,loss=float(loss),composite_RGB=float(rgb),body_RGB=float(body_rgb),body_alpha=float(alpha),sparse_depth=float(lidar),anchor=float(anchor),wall_seconds=time.monotonic()-started)
            log.write(json.dumps(record)+'\n')
            if step%25==0 or step==first+1:log.flush();update(step=step,last_loss=record,wall_seconds=time.monotonic()-started);print('V43_TRAIN',args.context,step,round(float(loss),6),round(time.monotonic()-started,2),flush=True)
            boundary=(step%250==0 and args.smoke_seconds) or step in [1000,5000,10000,20000] or args.entry_only and step==max_steps
            requested=stop[0] or (ROOT/'USER_STOP').exists()
            if boundary or requested:
                save(step)
                if requested or args.smoke_seconds and time.monotonic()-active_started>=args.smoke_seconds:break
        log.close();assert n.unchanged()
        update(phase='smoke_complete' if args.smoke_seconds else 'entry_complete' if args.entry_only else 'complete',step=step,wall_seconds=time.monotonic()-started,since_first_step=time.monotonic()-active_started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),geometry=actor.geometry(),source_checkpoint_unchanged=True,stop_requested=bool(stop[0] or (ROOT/'USER_STOP').exists()))
if __name__=='__main__':main()

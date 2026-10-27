"""Exploratory target-scoped decomposition with measured background support.

Joint gate/color/background optimization vs a frozen hard split. Geometry and
all unrelated static rows are fixed. Sparse cross-time colors are pseudo targets,
not counterfactual GT. No default renderer or original checkpoint is overwritten.
"""
import argparse,json,os,time,random
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from train_actor import NativeMotion,ROOT,OLD,torch,cpu,prepare_rasterization,helpers,equal
from segmentation import write
from probe_target_unknown import CTX,save
from review_gpu_slot import slot
from v43_resource_control import require_start

class Decomposition(torch.nn.Module):
    def __init__(self,n,evidence,memory,arm):
        super().__init__();self.arm=arm
        candidate=evidence['target_candidate'];ids=np.flatnonzero(candidate);self.register_buffer('ids',torch.from_numpy(ids).cuda())
        self.gate=torch.nn.Parameter(torch.full((len(ids),1),2.,device='cuda'),requires_grad=arm=='joint')
        self.dc_delta=torch.nn.Parameter(torch.zeros((len(ids),1,3),device='cuda'),requires_grad=arm=='joint')
        xyz=torch.from_numpy(memory['xyz']).cuda();self.register_buffer('xyz',xyz@torch.tensor(n.T[:3,:3].T,dtype=torch.float32,device='cuda')+torch.tensor(n.T[:3,3],dtype=torch.float32,device='cuda'))
        self.register_buffer('scale',torch.from_numpy(memory['sigma']).cuda()[:,None].repeat(1,3)*n.scale)
        self.register_buffer('initial_rgb',torch.from_numpy(memory['rgb']).cuda())
        self.dc=torch.nn.Parameter((self.initial_rgb-.5)[:,None,:]/.28209479177387814)
        self.opacity=torch.nn.Parameter(torch.full((len(xyz),1),0.,device='cuda'))
        self.register_buffer('quat',torch.tensor([1.,0,0,0],device='cuda').repeat(len(xyz),1))
    def kwargs(self,n,cam,st,phi=None):
        kw=dict(st);kw['features']=st['features'][:,:2];op=st['opacities'].clone();factor=self.gate.sigmoid() if self.arm=='joint' else torch.zeros_like(self.gate)
        if phi is not None:factor=1-(1-factor)*phi[:,None]
        op[self.ids]=op[self.ids]*factor;kw['opacities']=op
        sh=st['shs'].clone();sh[self.ids,:1]=sh[self.ids,:1]+self.dc_delta*(phi[:,None,None] if phi is not None else 1);kw['shs']=sh
        z=torch.cat((self.xyz,torch.ones_like(self.xyz[:,:1])),1)@cam.world_view_transform[:,2:3]
        bg=dict(means3D=self.xyz,means2D=torch.zeros_like(self.xyz),shs=torch.cat((self.dc,self.dc.new_zeros((len(z),n.static.get_features.shape[1]-1,3))),1),colors_precomp=None,features=torch.cat((z,torch.ones_like(z)),1),opacities=self.opacity.sigmoid(),scales=self.scale,rotations=self.quat,cov3D_precomp=None,mask=torch.ones(len(z),dtype=torch.bool,device='cuda'))
        return helpers.merge_rows(kw,bg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['joint','frozen_split'],required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume',type=Path);p.add_argument('--managed-round',type=Path);p.add_argument('--memory',default='clean_memory');p.add_argument('--protection',choices=['global','regions'],default='global');p.add_argument('--time-conditioned',action='store_true');p.add_argument('--video-teacher',action='store_true');a=p.parse_args()
    require_start(CTX);a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();R=ROOT/'target_unknown_v2'
    def status(**d):write(a.output/'status.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],arm=a.arm,**d))
    status(phase='waiting_GPU',step=0)
    with slot(a.managed_round):
        torch.set_num_threads(1);random.seed(43);np.random.seed(43);torch.manual_seed(43);torch.cuda.manual_seed_all(43)
        n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000);ds=ROOT/'datasets'/CTX;frames=json.loads((ds/'manifest.json').read_text())['frames']
        ev=np.load(R/'evidence_probe/evidence.npz');memory=np.load(R/a.memory/'seeds.npz');model=Decomposition(n,ev,memory,a.arm)
        opt=torch.optim.Adam([dict(params=[model.dc,model.opacity],lr=.003),dict(params=[model.gate],lr=.01),dict(params=[model.dc_delta],lr=.003)],eps=1e-8)
        data=[];row_specs=[];phis=[];candidate_ids=cpu(model.ids)
        if a.time_conditioned:
            for f in frames:
                e=np.load(R/'evidence_probe'/('%03d.npz'%f['raw']));v=np.zeros(len(n.static._xyz),np.float32);w=e['weights'];v[e['ids']]=w[:,1]/np.maximum(w[:,9],1e-8);phis.append(v[candidate_ids])
            phis=np.stack(phis);phis=np.stack([np.median(phis[max(0,i-1):min(len(frames),i+2)],0) for i in range(len(frames))]);phis=torch.from_numpy(phis).cuda()
            if a.arm=='frozen_split':phis=(phis>=.75).float()
        for i,f in enumerate(frames):
            row=dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time'],raw=f['raw']);row_specs.append(row);cam=n.camera(row,{})
            ob=np.load(ds/('%03d.npz'%f['raw']));m=np.load(R/a.memory/('%03d.npz'%f['raw']));body=ob['body'];edit=binary_dilation(body,iterations=8)
            sem=np.load(ROOT/'segformer_b0_attempt3'/CTX/('%03d.npz'%f['raw']));other=np.isin(sem['labels'],np.arange(12,20))&~edit;near=binary_dilation(edit,iterations=12)&~edit
            data.append(dict(target=torch.from_numpy(m['rgb'].transpose(2,0,1)).cuda(),valid=torch.from_numpy(m['valid']&edit).cuda(),protected=torch.from_numpy(~edit).cuda(),near=torch.from_numpy(near).cuda(),other=torch.from_numpy(other).cuda(),body=torch.from_numpy(body).cuda()))
            teacher=ROOT/'actor_motion_v1'/CTX/'delete_target/propainter/model_png'/('%03d.png'%f['raw']);mask=teacher.parents[2]/'mask'/('%03d.png'%f['raw'])
            if a.video_teacher and teacher.exists():
                from PIL import Image
                assert mask.exists(),mask
                data[-1]['teacher']=torch.from_numpy(np.asarray(Image.open(teacher).convert('RGB')).copy().transpose(2,0,1).astype('float32')/255).cuda();data[-1]['teacher_valid']=torch.from_numpy((np.asarray(Image.open(mask))>0)&~(m['valid']&edit)).cuda()
        def query(i,base_only=False,camera_case=None):
            row=row_specs[i];cam=n.camera(row,camera_case or {})
            with torch.no_grad():
                st=prepare_rasterization(cam,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);st['features']=st['features'][:,:2];env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            r=helpers.raster(cam,n.args,n.static.active_sh_degree,st if base_only else model.kwargs(n,cam,st,phis[i] if a.time_conditioned else None),n.bg)
            return r['rgb']+(1-r['alpha'])*env,r
        with torch.no_grad():
            for i,d in enumerate(data):d['reference']=query(i,True)[0].detach()
        first=0;stack=[]
        def restore(path):
            nonlocal first,stack
            s=torch.load(str(path),map_location='cuda');assert s.get('time_conditioned',False)==a.time_conditioned and s.get('video_teacher',False)==a.video_teacher;model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);first=s['step'];stack=s['sampler'];random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']]);assert equal(model.state_dict(),s['model']) and equal(opt.state_dict(),s['optimizer']);assert equal(random.getstate(),s['python_rng']) and equal(np.random.get_state(),s['numpy_rng']) and equal(torch.get_rng_state(),s['torch_rng']) and equal(torch.cuda.get_rng_state_all(),s['cuda_rng'])
            with torch.no_grad():probe=query(36)[0];diff=float((probe-s['probe_rgb']).abs().max());assert diff==0,diff
            return s
        if a.resume:
            restore(a.resume);write(a.output/'restore_verification.json',dict(model_optimizer_exact=True,all_RNG_and_sampler_restored=True,actual_native_RGB_max_difference=0,step=first))
        def checkpoint(step):
            with torch.no_grad():probe=query(36)[0].detach()
            s=dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),sampler=stack,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),arm=a.arm,probe_rgb=probe,time_conditioned=a.time_conditioned,video_teacher=a.video_teacher)
            path=a.output/('checkpoint%06d.pth'%step);torch.save(s,path);loaded=torch.load(str(path),map_location='cuda');assert equal(s,loaded);return path
        train_indices=[i for i in range(len(frames)) if i%3!=1];loss_log=[]
        for step in range(first+1,a.steps+1):
            require_start(CTX)
            if not stack:stack=train_indices.copy();random.shuffle(stack)
            i=stack.pop();d=data[i];rgb,r=query(i);valid=d['valid'];protected=d['protected'];target=(rgb[:,valid]-d['target'][:,valid]).abs().mean() if bool(valid.any()) else rgb.sum()*0
            # Full exterior protection, plus source color preservation for new measured surfaces.
            delta=(rgb-d['reference']).abs().mean(0);protect=delta[protected].mean()
            if a.protection=='regions':
                for mask in [d['near'],d['other']]:
                    if bool(mask.any()):protect=protect+delta[mask].mean()
            color_prior=(model.dc[:,0]*.28209479177387814+.5-model.initial_rgb).square().mean()
            gate_prior=model.gate.sigmoid().mean() if a.arm=='joint' else rgb.sum()*0
            teacher_loss=(rgb[:,d['teacher_valid']]-d['teacher'][:,d['teacher_valid']]).abs().mean() if 'teacher' in d and bool(d['teacher_valid'].any()) else rgb.sum()*0
            loss=target+.25*teacher_loss+2*protect+.01*color_prior+.0001*gate_prior+.0001*model.dc_delta.square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None);opt.step()
            with torch.no_grad():model.opacity.clamp_(-10,8);model.gate.clamp_(-8,8);model.dc_delta.clamp_(-3,3)
            if step%100==0 or step==first+1:
                log=dict(step=step,loss=float(loss),target=float(target),protection=float(protect),mean_retention=float(model.gate.sigmoid().mean()) if a.arm=='joint' else 0.,elapsed=time.monotonic()-started);loss_log.append(log);status(phase='training',**log,cuda_current_reserved=torch.cuda.memory_reserved());print('DECOMPOSE',json.dumps(log),flush=True)
            if step==100 or step==a.steps or step%5000==0:checkpoint(step)
        metrics=[]
        with torch.no_grad():
            for i,d in enumerate(data):
                rgb,r=query(i);delta=(rgb-d['reference']).abs().mean(0);v=d['valid'];outside=d['protected'];raw=frames[i]['raw'];save(a.output/'native_rgb'/('%03d.png'%raw),rgb)
                metrics.append(dict(raw=raw,split='calibration_holdout' if i%3==1 else 'train',sparse_background_pixels=int(v.sum()),sparse_background_L1=float((rgb[:,v]-d['target'][:,v]).abs().mean()) if bool(v.any()) else None,protected_RGB_change=float(delta[outside].mean()),protected_pixels_changed_gt1byte=int((delta[outside]>1/255).sum())))
                if raw in [91,98,101,109,113,114]:np.savez_compressed(a.output/('%03d_maps.npz'%raw),alpha=cpu(r['alpha'][0]).astype('float16'),RGB_change=cpu(delta).astype('float16'))
            if a.time_conditioned:
                for i,f in enumerate(frames):
                    if f['raw'] in [98,113,114]:
                        for offset in [-.5,.5]:save(a.output/'novel_camera'/('raw%d_offset%+.1fm.png'%(f['raw'],offset)),query(i,camera_case=dict(camera_lateral_m=offset))[0])
        np.savez_compressed(a.output/'decomposition_rows.npz',static_ids=cpu(model.ids),background_retention=cpu(model.gate.sigmoid()) if a.arm=='joint' else np.zeros((len(model.ids),1),np.float32),static_dc_delta=cpu(model.dc_delta),background_xyz_model=cpu(model.xyz),background_sigma_model=cpu(model.scale),background_dc=cpu(model.dc),background_opacity=cpu(model.opacity.sigmoid()),**({'time_phi':cpu(phis)} if a.time_conditioned else {}))
        assert n.unchanged();write(a.output/'summary.json',dict(context=CTX,arm=a.arm,memory=a.memory,protection=a.protection,time_conditioned=a.time_conditioned,video_teacher=a.video_teacher,video_teacher_weight=.25 if a.video_teacher else 0,video_teacher_is_generated_not_measurement=True,teacher_frames=[frames[i]['raw'] for i,d in enumerate(data) if 'teacher' in d],step=a.steps,formal_new_updates=a.steps-first,training_log=loss_log,metrics=metrics,static_candidate_rows=len(model.ids),measured_background_seeds=len(model.xyz),static_geometry_unchanged=True,unrelated_rows_frozen=True,quality_accepted=False,human_verdict=None,source_checkpoint_unchanged=True,sparse_background_is_not_counterfactual_GT=True,no_claim_of_complete_vehicle_removal=True))
        status(phase='complete',step=a.steps,wall_seconds=time.monotonic()-started,cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()

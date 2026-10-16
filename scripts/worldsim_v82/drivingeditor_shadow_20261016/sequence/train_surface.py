"""Read-only parent + explicit road reconstruction + staged joint fine tuning."""
import os,sys,time,json,math,copy,random,argparse,traceback
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OLD=ROOT/'shadow_BC_20261016_v1';NEW=ROOT/'shadow_sequence_20261016_v2'
sys.path.insert(0,str(OLD/'implementation_deferred_v1'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from run_experiments import Experiment,write,save_image,now,cpu_state
from bc_model import edited_casters,srgb_to_linear,linear_to_srgb,sample_render
from surface_model import RoadSurface,GeometryShadow
from observation_guards import project_guard
from scipy.ndimage import binary_dilation
import renderer

class SurfaceExperiment(Experiment):
    def __init__(self,args):
        super().__init__(args)
        with np.load(NEW/'surface_guarded_data/surface.npz') as f:
            self.surface_data={k:torch.as_tensor(f[k],device='cuda') for k in f.files if f[k].dtype.kind not in ['U','S']}
        for k in ['xyz','reference','confidence','lo','basis','origin','plane','observed','uv','gains']:
            self.surface_data[k]=self.surface_data[k].float()
        self.surface_data['shape']=self.surface_data['shape'].long()
        self.ray_cache={};self.base_images={};self.original_instances_prepare=renderer.prepare_instances
        self.protection_boxes=json.loads((NEW/'surface_guarded_data/protection_boxes.json').read_text())
        self.gt=[];self.body=[]
        ev=ROOT/'rigid_post30k/scene0008_089_137/evaluation'
        for row in self.manifest['streams']['observed']['frames']:
            with np.load(ev/row['npz']) as f:
                self.gt.append(torch.as_tensor(f['gt'],device='cuda'))
                self.body.append(torch.as_tensor(f['observed_masks'],device='cuda'))
        self.gt=torch.stack(self.gt);self.body=torch.stack(self.body)
        self.gate_ids,self.actor_receipt=self.find_actor_contamination()
        self.actor_gate_raw=torch.nn.Parameter(torch.full((len(self.gate_ids),),6.,device='cuda'))
        self.actor_opacity_delta=torch.nn.Parameter(torch.zeros(len(self.instances._xyz),device='cuda'),requires_grad=False)
        self.actor_opacity_enabled=False
        with np.load(Path(self.spec['prepared'])/'instances.npz') as f:self.mask_known=torch.tensor(f['mask_known'],device='cuda')
        centers,sizes,owners,parts=self.make_parts()
        warm=torch.load(OLD/'runs/deferred_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        shadow=GeometryShadow(self.data['plane_model'].cpu().float(),self.data['world_scale'],centers,sizes,owners,warm['shadow_state']).cuda()
        with torch.no_grad():anchor={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in self.original_prepare(self.cameras[12],self.static,self.config).items()}
        self.surface=RoadSurface(self.surface_data,self.static,anchor,shadow,self.instances.num_instances).cuda()
        self.surface.orig_prepare=self.original_prepare
        renderer.prepare_instances=self.prepare_actor
        self.contract.update(schema='explicit_road_surface_v2',road_grid=str(NEW/'surface_guarded_data/surface.npz'),
            original_ground_removed='abs height<.20m and 3sigma_normal<.30m, or dark and absheight<.40m 3sigma<.50m',anchor_static_frame=self.frame_ids[12],
            spatial_routing='world-coordinate multi-view/LiDAR confidence, zero outside; all scene geometry retains joint depth sort',
            parent_static_time='parent foreground retains original time; new road positions/opacity/color have no timestamp input',
            unselected_vehicles='conservative projected 3D annotation protection at render time; excluded from road evidence',
            actor_adaptation='only geometrically outside OBB and persistently outside observed body mask; opacity gates, no intrinsic relighting',
            caster='five measured local Gaussian envelope boxes per Actor; bounded shape corrections',
            interpretation='explicit plane surface reconstruction, not a topology-preserving Gaussian repair',
            no_true_clean_GT=True,not_unseen_frames=True)
        write(self.out/'runtime_contract.json',self.contract);write(self.out/'actor_contamination.json',self.actor_receipt)
        write(self.out/'caster_parts.json',parts)
        if args.restore:
            state=torch.load(args.restore,map_location='cpu',weights_only=False);self.surface.restore(state['surface']);self.actor_gate_raw.data.copy_(state['actor_gate'].to('cuda'))
            if state.get('actor_opacity_enabled',False):
                self.actor_opacity_delta.data.copy_(state['actor_opacity_delta'].to('cuda'));self.actor_opacity_enabled=True;self.actor_opacity_delta.requires_grad_(True)
            if state.get('runtime_contract',{}).get('ownership_refinement'):
                self.contract.update(state['runtime_contract']);write(self.out/'runtime_contract.json',self.contract)
        override=getattr(args,'guard_overrides',None) or self.contract.get('guard_override')
        self.guard_override=bool(override)
        if override:
            override=Path(override)
            with np.load(override) as f:
                for k in f.files:self.surface_data[k]=torch.tensor(f[k],device='cuda')
            self.guard_proxies=None
            if override.with_name('foreground_proxies.npz').exists():
                with np.load(override.with_name('foreground_proxies.npz')) as f:self.guard_proxies=[torch.tensor(f['points_%02d'%i],device='cuda') for i in range(49)]
                self.guard_fallback=json.loads(override.with_name('fallback_boxes.json').read_text())
            self.contract.update(guard_override=str(override),unselected_vehicles='existing DEVA/SAM segments matched to unselected3Dboxes; unmatched conservative fallback; custom-view foreground proxy guards')
            write(self.out/'runtime_contract.json',self.contract)
        self.sampler=random.Random(20261016)

    @torch.no_grad()
    def find_actor_contamination(self):
        local=self.instances._xyz;ids=self.instances.point_ids;sizes=self.data['sizes_model'].float()
        dims=torch.median(sizes,dim=0).values[ids]
        geometric=(abs(local)-dims/2).amax(1)>.012 # 12 cm beyond annotated 3D body box.
        seen=torch.zeros(len(local),device='cuda');outside=seen.clone()
        for i,cam in enumerate(self.cameras):
            world=self.instances.evaluate(cam.timestamp)
            hom=torch.cat([world.xyz,torch.ones_like(world.xyz[:,:1])],1)@cam.full_proj_transform
            uv=(hom[:,:2]/hom[:,3:4]+1)*torch.tensor([480,320],device='cuda')/2-.5
            x=uv[:,0].round().long().clamp(0,479);y=uv[:,1].round().long().clamp(0,319)
            good=(hom[:,3]>0)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)
            body=F.max_pool2d(self.body[i].float()[:,None],7,1,3)[:,0].bool()
            hits=body[world.point_ids,y,x]
            seen.index_add_(0,world.row_ids,good.float());outside.index_add_(0,world.row_ids,(good&~hits).float())
        ratio=outside/seen.clamp_min(1)
        candidate=geometric&(seen>=6)&(ratio>.95)
        gate_ids=torch.nonzero(candidate).ravel()
        from rigid_model import quaternion_to_matrix
        rotation=quaternion_to_matrix(self.instances._rotation)
        normal_sigma=((rotation.transpose(1,2)@torch.tensor([0.,0.,1.],device='cuda'))*self.instances._scaling.exp()).norm(dim=1)/self.data['world_scale']
        bottom_height=(local[:,2]+dims[:,2]/2)/self.data['world_scale']
        flat_bottom=(bottom_height.abs()<.25)&(normal_sigma*3<.15)
        ambiguous=flat_bottom&(seen>=6)&(ratio>.70)&~candidate
        return gate_ids,{'total_actor_points':len(local),'geometrically_outside_box_12cm':int(geometric.sum()),
            'persistent_outside_observed_body_and_geometry_candidates':len(gate_ids),
            'criteria':'center >0.12m outside local OBB; >=6 projected observations; >95% outside 3px-dilated own body mask',
            'screen_center_not_full_visibility_certificate':True,'self_shadow_and_reflection_separation_done':False,
            'flat_bottom_points_with_3sigma_normal_under_15cm':int(flat_bottom.sum()),
            'additional_flat_bottom_projected_outside_body_over70pct':int(ambiguous.sum()),
            'additional_bottom_candidates_not_pruned_without_contribution_evidence':True,
            'ambiguous_bottom_counts_by_actor':[int((ids[ambiguous]==k).sum()) for k in range(self.instances.num_instances)],
            'candidate_counts_by_actor':[int((ids[gate_ids]==k).sum()) for k in range(self.instances.num_instances)]}

    @torch.no_grad()
    def make_parts(self):
        centers=[];sizes=[];owners=[];receipt=[]
        for k in range(self.instances.num_instances):
            ids=self.instances.point_ids==k;q=self.instances._xyz[ids];opacity=self.instances._opacity[ids].sigmoid().ravel()
            dims=torch.median(self.data['sizes_model'][:,k].float(),dim=0).values
            good=(abs(q)<dims/2+.01).all(1)&(opacity>.05);q=q[good]
            if len(q)<100:raise RuntimeError('Insufficient real Gaussian geometry for caster')
            axis=int(dims[:2].argmax());low,high=torch.quantile(q[:,axis],torch.tensor([.01,.99],device='cuda'))
            bounds=torch.linspace(low,high,6,device='cuda')
            for j in range(5):
                s=q[(q[:,axis]>=bounds[j])&(q[:,axis]<=bounds[j+1])]
                if len(s)<30:continue
                lo=torch.quantile(s,.025,dim=0);hi=torch.quantile(s,.975,dim=0)
                lo[axis]=bounds[j];hi[axis]=bounds[j+1]
                centers.append((lo+hi)/2);sizes.append((hi-lo).clamp_min(.005));owners.append(k)
                receipt.append({'actor':k,'section':j,'source_Gaussians':len(s),'center_model':centers[-1].tolist(),'size_model':sizes[-1].tolist()})
        return torch.stack(centers),torch.stack(sizes),torch.tensor(owners,device='cuda'),receipt

    def prepare_actor(self,*args,**kwargs):
        pkg,world=self.original_instances_prepare(*args,**kwargs)
        if len(self.gate_ids):
            gate=torch.ones(len(self.instances._xyz),device='cuda').index_copy(0,self.gate_ids,self.actor_gate_raw.sigmoid())
            pkg['opacities']=pkg['opacities']*gate[world.row_ids,None]
        if self.actor_opacity_enabled:
            odds=torch.exp(3*self.actor_opacity_delta[world.row_ids].tanh())[:,None]
            opacity=pkg['opacities'];pkg['opacities']=opacity*odds/(1-opacity+opacity*odds).clamp_min(1e-8)
        return pkg,world

    def road_pixels(self,cam):
        key=id(cam)
        if key in self.ray_cache:return self.ray_cache[key]
        yy,xx=torch.meshgrid(torch.arange(320,device='cuda'),torch.arange(480,device='cuda'),indexing='ij')
        ndc=torch.stack([(2*xx+1)/480-1,(2*yy+1)/320-1,torch.full_like(xx,.5,dtype=torch.float32),torch.ones_like(xx)],-1).float().reshape(-1,4)
        p=ndc@torch.linalg.inv(cam.full_proj_transform);p=p[:,:3]/p[:,3:4]
        origin=cam.camera_center;direction=p-origin;plane=self.surface_data['plane']
        denom=direction@plane[:3];distance=-(origin@plane[:3]+plane[3])/torch.where(denom.abs()>1e-8,denom,torch.ones_like(denom))
        valid=(distance>0)&(denom.abs()>1e-8);q=origin+distance[:,None]*direction
        conf=self.surface.confidence_at(q)*valid.float()
        if cam in self.cameras:self.ray_cache[key]=(q,conf)
        return q,conf

    def render(self,i,*,shaded=False,instances=True,edits=None,camera=None):
        if not hasattr(self,'surface'):return super().render(i,shaded=False,instances=instances,edits=edits,camera=camera)
        cam=camera or self.cameras[i]
        renderer.prepare_static=self.original_prepare
        key=(i,instances)
        if camera is None and edits is None and key in self.base_images:
            base=self.base_images[key]
        else:
            # Gates are applied consistently to both spatial-routing branches.
            base=renderer.render(cam,self.static,self.instances if instances else None,self.config,self.bg,env_map=self.env,
                edits=edits if instances else None,max_features=10,render_separate=False,is_training=False)
            base={k:v.detach() if torch.is_tensor(v) else v for k,v in base.items()}
            if camera is None and edits is None and not len(self.gate_ids):self.base_images[key]=base
        renderer.prepare_static=self.surface.patch
        pkg=renderer.render(cam,self.static,self.instances if instances else None,self.config,self.bg,env_map=self.env,
            edits=edits if instances else None,max_features=10,render_separate=False,is_training=False)
        renderer.prepare_static=self.original_prepare
        q,conf0=self.road_pixels(cam)
        if camera is None:guard=self.surface_data['protection'][i]
        else:
            camera_dict={'full_proj_transform':cam.full_proj_transform.detach().cpu().tolist()}
            if self.guard_override and self.guard_proxies is not None:
                points=self.guard_proxies[i];hom=torch.cat([points,torch.ones_like(points[:,:1])],1)@cam.full_proj_transform
                uv=(hom[:,:2]/hom[:,3:4]+1)*torch.tensor([480,320],device='cuda')/2-.5
                u=uv[:,0].round().long();v=uv[:,1].round().long();valid=(hom[:,3]>0)&(u>=0)&(u<480)&(v>=0)&(v<320)
                guard=torch.zeros(320,480,device='cuda');guard[v[valid],u[valid]]=1
                guard=F.max_pool2d(guard[None,None],5,1,2)[0,0].bool()
                fallback=torch.tensor(binary_dilation(project_guard(self.guard_fallback[i],camera_dict),iterations=2).copy(),device='cuda')
                guard|=fallback
            else:guard=torch.as_tensor(binary_dilation(project_guard(self.protection_boxes[i],camera_dict),iterations=2).copy(),device='cuda')
        conf=conf0*(~guard).reshape(-1).float();weight=conf.reshape(1,320,480)
        pkg['unshaded']=pkg['render']
        if shaded and instances:
            cast=edited_casters(self.instances,cam.timestamp,edits,self.data['sizes_model'][i])
            if bool(cast[3].any()):
                take=conf>0;att=torch.ones(len(q),3,device='cuda')
                values=self.surface.shadow.attenuation(q[take],*cast)
                att=att.index_copy(0,torch.nonzero(take).ravel(),values).T.reshape(3,320,480)
                alpha=pkg['feature'][:1];road_rgb=pkg['feature'][1:4]
                replacement=alpha*linear_to_srgb(srgb_to_linear((road_rgb/alpha.clamp_min(1e-7)).clamp(0,1))*att)
                pkg['render']=pkg['render']-road_rgb+torch.where(alpha>1e-7,replacement,road_rgb)
                pkg['road_shadow_attenuation']=att
        pkg['render']=weight*pkg['render']+(1-weight)*base['render']
        pkg['unshaded']=weight*pkg['unshaded']+(1-weight)*base['render']
        # Public visibility features describe the same routed scene as RGB.
        # The raw road feature channels remain an internal shading payload.
        for key in ['alpha','depth','dynamic_visible_alpha','static_visible_alpha','instance_visible_contribution']:
            pkg[key]=weight*pkg[key]+(1-weight)*base[key]
        pkg['visible_instance_alpha']=pkg['instance_visible_contribution']
        pkg['instance_visible_alpha']={name:pkg['instance_visible_contribution'][j:j+1] for j,name in enumerate(pkg['instance_ids'])}
        pkg['env_map']=weight*pkg['env_map']+(1-weight)*base['env_map']
        pkg['raw_render']=pkg['render']-pkg['env_map']
        pkg['reconstruction_feature_unrouted']=pkg['feature']
        pkg['feature']=weight*pkg['feature']
        pkg['road_confidence']=weight;pkg['road_visible_alpha']=pkg['feature'][:1]
        pkg['parent_baseline_with_same_actor_gates']=base['render']
        return pkg

    def choose(self,mask,maximum=4096):
        ids=torch.nonzero(mask).ravel()
        if len(ids)>maximum:ids=ids[torch.randperm(len(ids),device='cuda')[:maximum]]
        return ids

    def losses(self,i,stage):
        d=self.surface_data;valid=(d['confidence']>.9)&d['free'][i]
        ids=self.choose(valid)
        if len(ids)<100:raise RuntimeError('Insufficient valid observations')
        clean_pkg=self.render(i,instances=False)
        pred=srgb_to_linear(sample_render(clean_pkg['render'],d['uv'][i,ids]))
        road=abs(pred-d['reference'][ids]).mean()
        if stage=='background':return road+self.surface.prior(),{'road':float(road.detach())}
        joint=self.render(i,shaded=True)
        seen=srgb_to_linear(sample_render(joint['render'],d['uv'][i,ids]))
        sm=d['shadow'][i,ids];cm=d['clean'][i,ids]
        zero=seen.sum()*0
        shadow_loss=abs(seen[sm]-d['observed'][i,ids][sm]).mean() if sm.any() else zero
        clean_loss=abs(seen[cm]-d['observed'][i,ids][cm]).mean() if cm.any() else zero
        body=self.body[i].any(0)
        actor_rgb=abs(joint['render'][:,body]-self.gt[i,:,body]).mean()
        outside=(~F.max_pool2d(body.float()[None,None],7,1,3)[0,0].bool())
        leakage=joint['dynamic_visible_alpha'][0,outside].mean()
        gate_prior=.002*(1-self.actor_gate_raw.sigmoid()).mean() if len(self.gate_ids) else zero
        mask_loss=zero;actor_prior=zero
        if self.actor_opacity_enabled:
            target=self.body[i].float();known=self.mask_known[i,:,None,None].float()
            mask_loss=((joint['instance_visible_contribution']-target).abs()*known).sum()/(target*known).sum().clamp_min(1)
            # A road-shadow RGB residual must not be explained by opaque Actor
            # particles outside the observed body. Normalize each subset on its
            # own support, so a large body area cannot dilute this constraint.
            actor_alpha=sample_render(joint['dynamic_visible_alpha'],d['uv'][i,ids])[:,0]
            if sm.any():mask_loss=mask_loss+2*actor_alpha[sm].mean()
            if cm.any():mask_loss=mask_loss+actor_alpha[cm].mean()
            actor_prior=.002*self.actor_opacity_delta.tanh().square().mean()
        total=road+.8*shadow_loss+.3*clean_loss+.1*actor_rgb+.05*leakage+mask_loss+self.surface.prior()+self.surface.shadow.prior()+gate_prior+actor_prior
        return total,{'road':float(road.detach()),'shadow':float(shadow_loss.detach()),'clean':float(clean_loss.detach()),'actor_RGB':float(actor_rgb.detach()),'actor_leak':float(leakage.detach()),'actor_mask':float(mask_loss.detach())}

    @torch.no_grad()
    def metrics(self,frames=None):
        rows=[]
        for i in (frames if frames is not None else range(49)):
            d=self.surface_data;valid=(d['confidence']>.9)&d['free'][i]
            ids=torch.nonzero(valid).ravel();clean=self.render(i,instances=False);seen=self.render(i,shaded=True)
            p=srgb_to_linear(sample_render(clean['render'],d['uv'][i,ids]));j=srgb_to_linear(sample_render(seen['render'],d['uv'][i,ids]));sm=d['shadow'][i,ids];cm=d['clean'][i,ids]
            old=srgb_to_linear(sample_render(clean['render'],self.data['uv'][i]));a=self.data['accepted_reference']
            row={'frame':self.frame_ids[i],'surface_samples':len(ids),'clean_surface_MAE':float(abs(p-d['reference'][ids]).mean()),
                'shadow_surface_MAE':float(abs(j[sm]-d['observed'][i,ids][sm]).mean()) if sm.any() else None,
                'clean_observed_surface_MAE':float(abs(j[cm]-d['observed'][i,ids][cm]).mean()) if cm.any() else None,
                'shadow_samples':int(sm.sum()),'old_local_clean_MAE':float(abs(old[a]-self.data['reference_linear'][a]).mean()),
                'visible_road_alpha_mean':float(seen['road_visible_alpha'][seen['road_confidence']>.99].mean())}
            rows.append(row)
        return {'per_frame':rows,'mean_clean_surface_MAE':float(np.mean([r['clean_surface_MAE'] for r in rows])),
            'mean_shadow_surface_MAE':float(np.mean([r['shadow_surface_MAE'] for r in rows if r['shadow_surface_MAE'] is not None])),
            'mean_old_local_clean_MAE':float(np.mean([r['old_local_clean_MAE'] for r in rows]))}

    def save(self,name,step,optimizer=None):
        dest=self.out/(name+'.pt');data={'schema':'explicit_road_surface_joint_v2','parent_checkpoint':self.contract.get('parent_checkpoint',str(ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt')),
            'step':step,'phase':name,'surface':self.surface.state(),'actor_gate':self.actor_gate_raw.detach().cpu(),
            'gate_ids':self.gate_ids.cpu(),'parts':json.loads((self.out/'caster_parts.json').read_text()),
            'actor_opacity_enabled':self.actor_opacity_enabled,'actor_opacity_delta':self.actor_opacity_delta.detach().cpu(),
            'optimizer':optimizer.state_dict() if optimizer else None,'sampler_rng':self.sampler.getstate(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'runtime_contract':self.contract}
        temp=dest.with_suffix('.partial');torch.save(data,temp);os.replace(temp,dest);return dest

    def smoke(self):
        loss,_=self.losses(12,'joint');loss.backward()
        params={'road_color':self.surface.color_raw,'road_opacity':self.surface.opacity_raw,
            'light_azimuth':self.surface.shadow.azimuth,'light_elevation':self.surface.shadow.elevation_raw,
            'softness':self.surface.shadow.softness_raw,'direct_RGB':self.surface.shadow.direct_raw,
            'shape_scale':self.surface.shadow.part_scale_raw,'shape_shift':self.surface.shadow.part_shift_raw}
        if len(self.gate_ids):params['actor_outlier_opacity']=self.actor_gate_raw
        if self.actor_opacity_enabled:params['actor_opacity_adapter']=self.actor_opacity_delta
        result={k:{'finite':v.grad is not None and bool(torch.isfinite(v.grad).all()),'max':float(v.grad.abs().max()) if v.grad is not None else None} for k,v in params.items()}
        assert all(r['finite'] for r in result.values()),result
        assert result['road_color']['max']>0 and result['light_azimuth']['max']>0,result
        for v in params.values():v.grad=None
        write(self.out/'gradient_smoke.json',result)

    @torch.no_grad()
    def preview(self,name):
        path=self.out/'previews'/name;path.mkdir(parents=True,exist_ok=True)
        for i in [0,6,12,24,36,48]:
            for mode in ['observed','static','delete_target']:
                p=self.render(i,instances=mode!='static',shaded=mode!='static',edits={0:{'delete':True}} if mode=='delete_target' else None)
                save_image(path/(self.frame_ids[i]+'_'+mode+'.png'),p['render'])

    def train(self):
        self.smoke();initial=self.metrics([0,6,12,18,24,36,48]);write(self.out/'initial_metrics.json',initial);self.preview('initial')
        if self.args.probe:
            self.verify_sources();self.update(state='complete',phase='zero_update_probe_complete',initial=initial);return
        plan=getattr(self.args,'plan_override',None) or [('background',1500),('shadow',2000),('joint',4500)]
        shadow=self.surface.shadow
        groups=[{'params':[self.surface.color_raw],'lr':.015,'name':'road_color'},
            {'params':[self.surface.opacity_raw],'lr':.01,'name':'road_opacity'},
            {'params':[shadow.azimuth,shadow.elevation_raw,shadow.softness_raw,shadow.direct_raw],'lr':.004,'name':'light'},
            {'params':[shadow.part_scale_raw,shadow.part_shift_raw],'lr':.002,'name':'shape'}]
        if len(self.gate_ids):groups.append({'params':[self.actor_gate_raw],'lr':.01,'name':'actor_outliers'})
        if self.actor_opacity_enabled:groups.append({'params':[self.actor_opacity_delta],'lr':.02,'name':'actor_opacity'})
        optimizer=torch.optim.Adam(groups);globalstep=0;start=time.time()
        checks=[]
        for stage,steps in plan:
            for group in optimizer.param_groups:
                allowed={'background':['road_color','road_opacity'],'shadow':['light','shape'],'joint':['road_color','road_opacity','light','shape','actor_outliers','actor_opacity']}[stage]
                for p in group['params']:p.requires_grad_(group['name'] in allowed)
                if stage=='joint':group['lr']*=.25
            for step in range(1,steps+1):
                if step%50==1:
                    check=__import__('priority').inspect_priority()
                    if not check['allowed']:
                        self.save('yielded_priority',globalstep,optimizer);self.update(state='yielded_to_priority_work',priority=check);return
                eligible=[i for i in self.fit_indices if bool(self.surface_data['shadow'][i].any())] if stage=='shadow' else self.fit_indices
                i=eligible[self.sampler.randrange(len(eligible))]
                optimizer.zero_grad(set_to_none=True);loss,values=self.losses(i,stage)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward()
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise ValueError('nonfinite gradient')
                optimizer.step();globalstep+=1
                if step==1 or step%50==0:
                    elapsed=time.time()-start;row={'stage':stage,'stage_step':step,'stage_steps':steps,'global_step':globalstep,'loss':float(loss.detach()),'elapsed_seconds':elapsed,**values}
                    with (self.out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    self.update(state='running',phase=stage,**row);print(json.dumps(row),flush=True)
                if step%500==0:
                    self.save(stage+'_%06d'%step,globalstep,optimizer)
                    m=self.metrics([0,6,12,18,24,36,48]);checks.append({'stage':stage,'stage_step':step,'global_step':globalstep,**m});write(self.out/'validation_curve.json',checks)
            checkpoint=self.save(stage+'_final',globalstep,optimizer)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            before=self.render(12,shaded=True)['render'].detach();self.surface.color_raw.data.add_(.01)
            if self.actor_opacity_enabled:self.actor_opacity_delta.data.add_(.01)
            self.surface.restore(saved['surface'])
            if self.actor_opacity_enabled:self.actor_opacity_delta.data.copy_(saved['actor_opacity_delta'].to('cuda'))
            after=self.render(12,shaded=True)['render'].detach();assert torch.equal(before,after)
            optimizer.load_state_dict(saved['optimizer'])
            write(self.out/(stage+'_restore_check.json'),{'actual_saved_addon_loaded':True,'RGB_bitwise_equal':True,'Adam_restored':True,'sampler_present':saved['sampler_rng'] is not None})
            self.preview(stage);self.verify_sources()
        final=self.metrics();write(self.out/'full49_metrics.json',final)
        self.save('final',globalstep,optimizer);self.verify_sources()
        self.update(state='complete',phase='staged_joint_training_complete',training_updates=globalstep,elapsed_seconds=time.time()-start,metrics=final)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--probe',action='store_true');p.add_argument('--restore');args=p.parse_args();args.steps=8000;args.warmup=1;e=None
    try:e=SurfaceExperiment(args);e.train()
    except BaseException as error:
        if e:e.update(state='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if e:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
if __name__=='__main__':main()

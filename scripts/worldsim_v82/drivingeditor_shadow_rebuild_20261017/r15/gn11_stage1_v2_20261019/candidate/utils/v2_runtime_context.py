"""Native-training adapter. Requires a separately verified CPU input bundle."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import functools
from utils.v2_static_partition import (route_depth, observed_rgb_loss,
                                       vehicle_volume_opacity, independent_road_backward)
from utils.v2_road_objective import road_objective

ACTIVE_CONTEXT=None


class V2Context:
    def __init__(self, args, scene, environment):
        bundle=Path(args['v2_bundle'])
        self.root=bundle.parent
        self.meta=json.loads(bundle.read_text(encoding='utf-8'))
        if self.meta['status']!='CPU_INPUT_BUNDLE_VERIFIED':
            raise RuntimeError('real inputs not bound; synthetic tests cannot authorize training')
        if int(args.iterations)!=30000 or args.start_checkpoint:
            raise RuntimeError('only fresh first-stage30K authorized')
        if set(self.meta['actors'])!={6,37,5,26,18}:
            raise RuntimeError('wrong actor set')
        self.frames=self.meta['frames']
        if set(self.frames)!={f'0008{i:03d}' for i in range(89,138)}:
            raise RuntimeError('49 real frame contract failed')
        self.config=self.meta['config']
        self.cache={}
        # No label changes, no learned previous30K state, no new camera views.
        from utils.hybrid30k_runtime import camera_id
        camera_ids={Path(camera_id(c)).stem for c in scene.getTrainCameras()}
        if camera_ids!=set(self.frames): raise RuntimeError('actual camera bundle mismatch')
        for c in scene.getTrainCameras():
            fid=Path(camera_id(c)).stem
            # Renderer matrix is transposed; scale converts camera coordinates to m.
            actual=c.world_view_transform.detach().cpu().numpy().T
            expected=np.asarray(self.frames[fid]['world_to_camera'])
            model_from_world=np.asarray(self.meta['world_to_model'])
            derived=actual @ model_from_world
            scale=float(self.meta['meter_to_model'])
            derived[:3,:]/=scale
            if not np.allclose(derived,expected,atol=2e-4,rtol=2e-4):
                raise RuntimeError('world/model camera transform mismatch: '+fid)
        self.log=Path(args.model_path)/'v2_objectives.jsonl'
        self.gate_only=bool(args.get('v2_zero_update_gate',False))
        self.models=[scene.gaussians,scene.dynamic_gaussians,environment]
        self.optimizer_steps={}
        for name,model in zip(['static','dynamic','environment'],self.models):
            if model is None:continue
            self.optimizer_steps[name]=0
            original_step=model.optimizer.step
            def counted_step(*args,_name=name,_original=original_step,**kwargs):
                result=_original(*args,**kwargs)
                self.optimizer_steps[_name]+=1
                return result
            model.optimizer.step=functools.wraps(original_step)(counted_step)
        global ACTIVE_CONTEXT
        ACTIVE_CONTEXT=self
        self.gate_details={}
        self.output=Path(args.model_path)

    def select_gate_camera(self,scene,camera,time_shift):
        if not self.gate_only:return camera,time_shift
        return next(c for c in scene.getTrainCameras() if self.frame(c)[0]=='0008089'),None

    def gate_forward(self,scene,render,camera,args,background,environment):
        if not self.gate_only:return
        import copy
        static=scene.gaussians
        times=[min(float(c.timestamp) for c in scene.getTrainCameras()),
               max(float(c.timestamp) for c in scene.getTrainCameras())]
        images=[]
        with torch.no_grad():
            for t in times:
                if not torch.equal(static.get_xyz_SHM(t),static._xyz):raise RuntimeError('static positions vary with time')
                if not torch.equal(static.get_marginal_t(t),torch.ones_like(static._opacity)):raise RuntimeError('static visibility varies with time')
                cam=copy.copy(camera);cam.timestamp=t
                pkg=render(cam,static,scene.dynamic_gaussians,args,background,env_map=environment,
                           other=[static.get_scaling_t,static.get_inst_velocity],
                           dynamic_other=[scene.dynamic_gaussians.get_scaling_t.clamp_max(2),scene.dynamic_gaussians.get_inst_velocity],
                           time_shift=None,is_training=False)
                images.append(pkg['render_s'].detach().clone())
                del pkg
            difference=float((images[0]-images[1]).abs().max())
            if not np.isfinite(difference) or difference>1e-6:raise RuntimeError('actual fixed-camera static RGB is time-varying')
        self.gate_details.update(static_fixed_camera_max_RGB_delta=difference,
                                 static_times_tested=times,gate_raw_frame=89)

    def frame(self,camera):
        from utils.hybrid30k_runtime import camera_id
        fid=Path(camera_id(camera)).stem
        return fid,self.frames[fid]

    def partition_state(self,state,camera,pkg,gt,mask,depth,args,ssim_fn):
        from utils.paper_losses import sparse_inverse_depth_loss
        import kornia
        state=dict(state)
        state['global_rgb']=observed_rgb_loss(pkg['render'],gt,mask,ssim_fn,args.lambda_dssim)
        routed_depth=route_depth(depth,pkg['dynamic_depth'],pkg['dynamic_alpha'],mask)
        state['depth']=sparse_inverse_depth_loss(routed_depth,camera.pts_depth.to(depth.device))
        state['inverse_depth_smoothness']=kornia.losses.inverse_depth_smoothness_loss(
            (1/(routed_depth+1e-5))[None],gt[None])
        state['alpha']=torch.where(mask.bool(),pkg['dynamic_alpha'],pkg['alpha'])
        # The mixed velocity normalization must not send body supervision to static alpha.
        joint_v=pkg['feature'][1:]/pkg['alpha'].clamp_min(1e-5)
        dynamic_v=pkg['dynamic_feature'][1:]/pkg['dynamic_alpha'].clamp_min(1e-5)
        routed_v=torch.where(mask.bool(),dynamic_v,joint_v)
        state['velocity_smoothness']=kornia.losses.inverse_depth_smoothness_loss(routed_v[None],gt[None])
        # Static temporal penalties are inapplicable when the representation is 3D.
        return state

    def objectives(self,camera,pkg,static,iteration):
        fid,frame=self.frame(camera)
        def tensor(x): return torch.as_tensor(x,device=static._xyz.device,dtype=static._xyz.dtype)
        boxes=frame['selected_boxes_model']
        centers=tensor([b['center'] for b in boxes]).reshape(-1,3)
        axes=tensor([b['axes'] for b in boxes]).reshape(-1,3,3)
        half=tensor([b['half'] for b in boxes]).reshape(-1,3)
        plane=self.meta['ground_plane_model']
        scale=float(self.meta['meter_to_model'])
        cfg=self.config['training']
        ownership=vehicle_volume_opacity(static._xyz,static.get_opacity,centers,axes,half,
            tensor(plane[:3]),float(plane[3]),cfg['road_clearance_m']*scale,cfg['volume_edge_m']*scale)
        if fid not in self.cache:
            p=self.root/frame['road_target']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=frame['road_target_sha256']:
                raise RuntimeError('road target changed after CPU verification')
            with np.load(p,allow_pickle=False) as z: self.cache[fid]={k:z[k].copy() for k in z.files}
        target={k:tensor(v) for k,v in self.cache[fid].items()}
        loss_road,terms=road_objective(pkg['render_s'],pkg['static_depth'],pkg['static_alpha'],target,
                                    self.config['road']['objective_weights'])
        if iteration==1 or iteration%100==0:
            record=dict(step=iteration,frame=fid,ownership_volume_raw=float(ownership.detach()),
                        ownership_volume_weight=cfg['volume_exclusion_weight'],
                        road={k:float(v.detach()) for k,v in terms.items()},
                        road_weights=self.config['road']['objective_weights'],
                        road_total=float(loss_road.detach()),road_optimizer_steps=0,
                        road_updates_dynamic=False,road_updates_environment=False)
            with self.log.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        return cfg['volume_exclusion_weight']*ownership,loss_road

    def backward(self,ownership,road,static,iteration):
        independent_road_backward(ownership,road,
            [p for group in static.optimizer.param_groups for p in group['params']])
        if self.gate_only:
            # Fail-stop before densification/Adam/checkpoint. It cannot become training.
            counts=[]
            for model in self.models:
                if model is None:continue
                params=[p for g in model.optimizer.param_groups for p in g['params'] if p.grad is not None]
                if not params or not all(torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('GPU backward contract failed')
                counts.append(len(params))
            for field in ['_t','_scaling_t','_velocity']:
                p=getattr(static,field)
                if p.requires_grad or p.grad is not None:raise RuntimeError('static temporal parameter has a gradient')
            self.gate_details.update(finite_gradient_parameter_groups=counts,optimizer_updates=0,
                                     optimizer_step_counts=dict(self.optimizer_steps),
                                     static_temporal_gradients_absent=True)
            if any(self.optimizer_steps.values()):raise RuntimeError('zero-update gate executed an optimizer step')
            (self.output/'gpu_gate_details.json').write_text(json.dumps(self.gate_details,indent=2),encoding='utf-8')
            raise ZeroUpdateGateComplete('actual GPU forward/backward complete; optimizer updates=0')


class ZeroUpdateGateComplete(RuntimeError):
    pass

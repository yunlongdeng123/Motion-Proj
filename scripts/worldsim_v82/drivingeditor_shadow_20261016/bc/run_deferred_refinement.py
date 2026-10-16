"""Deferred shading of depth-sorted visible road contributions, not a black overlay."""
import argparse,json,traceback
import numpy as np
import torch
from bc_model import edited_casters,srgb_to_linear,linear_to_srgb
from run_plane_refinement import PlaneExperiment
from run_experiments import BASE,write,now
import renderer

class DeferredExperiment(PlaneExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.warm=torch.load(BASE/'runs/plane_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        self.ray_cache={}
        self.contract.update(shadow_application='deferred_depth_sorted_visible_road_alpha_T',
            warm_start_addon=str(BASE/'runs/plane_1000_v1/R4/addon_001000.pt'),
            formula='joint_unshaded - visible_road_RGB + road_alpha_T * encode(decode(visible_road_RGB/road_alpha_T)*world_shadow)',
            full_scene_geometry_sort_and_actor_occlusion_preserved=True,
            physical_linear_compositing_claim=False)
        write(self.out/'runtime_contract.json',self.contract)
    def road_pixels(self,cam):
        key=id(cam)
        if key in self.ray_cache:return self.ray_cache[key]
        h,w=int(cam.image_height),int(cam.image_width);device=self.bg.device
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        ndc=torch.stack([(2*xx+1)/w-1,(2*yy+1)/h-1,torch.full_like(xx,.5,dtype=torch.float32),torch.ones_like(xx)],-1).float().reshape(-1,4)
        p=ndc@torch.linalg.inv(cam.full_proj_transform)
        p=p[:,:3]/p[:,3:4];origin=cam.camera_center;direction=p-origin;plane=self.data['plane_model'].float()
        denom=direction@plane[:3];distance=-(origin@plane[:3]+plane[3])/torch.where(denom.abs()>1e-8,denom,torch.ones_like(denom))
        valid=(denom.abs()>1e-8)&(distance>0)
        xyz=origin+distance[:,None]*direction;xyz=torch.where(valid[:,None],xyz,torch.zeros_like(xyz))
        if cam in self.cameras:self.ray_cache[key]=(xyz,valid)
        return xyz,valid
    def render(self,i,*,shaded=False,instances=True,edits=None,camera=None):
        if not self.active or not shaded or self.active.shadow is None:
            return super().render(i,shaded=False,instances=instances,edits=edits,camera=camera)
        cam=camera or self.cameras[i];self.active.current_frame=i;self.active.shaded=False;self.active.edits=edits
        cast=edited_casters(self.instances,cam.timestamp,edits,self.data['sizes_model'][i])
        if not bool(cast[3].any()):return super().render(i,shaded=False,instances=instances,edits=edits,camera=camera)
        # Four feature channels: alpha*T, and the three unshaded RGB contributions.
        # Other objects stay in the same rasterization. 4+3 builtin+3 Actor IDs=10.
        feature=torch.zeros((len(self.static.get_xyz),4),device='cuda')
        ri=self.active.receiver_ids
        values=torch.cat([torch.ones((len(ri),1),device='cuda'),self.active.color_raw.sigmoid()],1)
        feature=feature.index_copy(0,ri,values)
        pkg=renderer.render(cam,self.static,self.instances if instances else None,self.config,self.bg,
            env_map=self.env,edits=edits if instances else None,other=feature,max_features=10,render_separate=False,is_training=False)
        alpha=pkg['feature'][:1];road_rgb=pkg['feature'][1:4]
        q,valid=self.road_pixels(cam)
        att=self.active.shadow.attenuation(q,*cast).T.reshape(3,cam.image_height,cam.image_width)
        att=torch.where(valid.reshape(1,cam.image_height,cam.image_width),att,torch.ones_like(att))
        average=(road_rgb/alpha.clamp_min(1e-7)).clamp(0,1)
        replacement=alpha*linear_to_srgb(srgb_to_linear(average)*att)
        replacement=torch.where(alpha>1e-7,replacement,road_rgb)
        pkg['render']=pkg['render']-road_rgb+replacement
        pkg['raw_render']=pkg['raw_render']-road_rgb+replacement
        pkg['deferred_road_visible_alpha']=alpha
        return pkg
    def run(self):
        self.gradient_smoke()
        if self.args.steps==0:
            (self.out/'R4').mkdir();self.activate('R4',self.shadow());metrics=self.evaluate('R4')
            self.verify_sources();self.update(state='complete',phase='zero_update_deferred_probe',training_updates=0,metrics=metrics)
            print(json.dumps(metrics));return
        metrics=self.train_arm('R4',{'state':self.warm['shadow_state']})
        previous=json.loads((BASE/'runs/plane_1000_v1/R4/evaluation/metrics.json').read_text())
        write(self.out/'comparison.json',{'plane_per_Gaussian':previous,'deferred_road':metrics,
            'note':'Same geometry, fixed parent and visibility sorting. Shading applied to full-scene visible road feature contribution.'})
        self.verify_sources();self.update(state='complete',phase='deferred_refinement_ready',metrics=metrics,parent_checkpoint_unchanged=True,source_parameters_unchanged=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args();a.warmup=1
    assert 0<=a.steps<=1000;e=None
    try:e=DeferredExperiment(a);e.run()
    except BaseException as error:
        if e:e.update(state='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if e:renderer.prepare_static=e.original_prepare;e.slot.close()
if __name__=='__main__':main()

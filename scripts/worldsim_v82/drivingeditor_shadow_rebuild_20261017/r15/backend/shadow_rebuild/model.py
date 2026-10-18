"""Legacy SH/EnvLight + shared rigid/road/shadow G-buffer adapter."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .model_r14 import SceneModel as ShadowBuffers
from .field import GaussianField
from .environment import EnvLight
from .tracing import SunAmbient
from .raster import render_buffers
from .math3d import normalize,srgb_to_linear

class SceneModel(ShadowBuffers):
    def __init__(self,seed,config):
        nn.Module.__init__(self);self.config=config;self.field=GaussianField(seed,config);self.light=SunAmbient(config);self.environment=EnvLight(config.env_map_res)
    def sky(self,camera):
        if self.config.fixture_small_environment and not self.environment.base.is_cuda:
            # Explicit constant-sky synthetic fixture only. Production must use
            # the native cube kernel; verified separately on actual GPU inputs.
            return srgb_to_linear(self.environment.base.mean((0,1,2))).expand(camera.height,camera.width,3)
        y,x=torch.meshgrid(torch.arange(camera.height,device=camera.K.device),torch.arange(camera.width,device=camera.K.device),indexing='ij')
        ray=normalize((torch.stack((x,y,torch.ones_like(x)),-1).to(camera.K.dtype)@torch.linalg.inv(camera.K).T)@camera.pose[:3,:3].T)
        return srgb_to_linear(self.environment(ray).clamp_min(0))
    def buffers(self,frame,actor_count,backend,edits=None,isolated=True):
        g=self.field.evaluate(frame.poses,frame.valid,frame.camera.center,edits)
        if self.config.shadow_integrator=='factor_fp64':
            from .factor_shadow import factors_for_geometry
            g.shadow_factors=factors_for_geometry(self,frame,g,edits)
        ambient=self.light.ambient_logit.sigmoid();cosine=(g['normal']*self.light.direction()).sum(-1).clamp_min(0)
        c0=g['albedo']*(ambient+(1-ambient)*cosine[:,None])
        g['appearance']=torch.where(g['road'][:,None],c0,g['appearance'])
        p=render_buffers(frame.camera,g,actor_count,backend,self.config.max_features,isolated,self.config.continuous_ownership and isolated)
        sky=self.sky(frame.camera);p['sky']=sky;p['canonical']=p['appearance']+(1-p['alpha'])[...,None]*sky
        return g,p
    def train_image(self,p,g):
        # New physical-shadow visibility sampled on a receiver grid. All RGB,
        # SH and cube details and photometric losses remain full resolution.
        h,w=p['alpha'].shape;s=getattr(self,'_shadow_stride',self.config.shadow_train_stride)
        # Include both image endpoints; interpolation must not map row304 to319.
        ys=torch.linspace(0,h-1,math.ceil(h/s)+1,device=p['alpha'].device).round().long()
        xs=torch.linspace(0,w-1,math.ceil(w/s)+1,device=p['alpha'].device).round().long()
        yy,xx=torch.meshgrid(ys,xs,indexing='ij');ids=(yy*w+xx).flatten()
        pos=p['road_position'].reshape(-1,3)[ids];normal=p['road_normal'].reshape(-1,3)[ids];active=p['road_alpha'].reshape(-1)[ids].detach()>1e-5
        vis=pos[:,0]*0+1
        if self.config.shadows and active.any():vis=vis.index_copy(0,active.nonzero().flatten(),self.light.visibility(pos[active],normal[active],g))
        v=F.interpolate(vis.reshape(1,1,*yy.shape),size=(h,w),mode='bilinear',align_corners=True)[0,0]
        ambient=self.light.ambient_logit.sigmoid();cosine=(p['road_normal']*self.light.direction()).sum(-1).clamp_min(0)
        irradiance=ambient+(1-ambient)*cosine[...,None]*(1-self.light.strength_logit.sigmoid()*(1-v[...,None]))
        c0irr=ambient+(1-ambient)*cosine[...,None]
        p['rho_image']=((c0irr-irradiance)/c0irr.clamp_min(1e-8)).clamp(0,1)
        return p['road_alpha'][...,None]*p['road_albedo']*irradiance+p['appearance']-p['road_appearance']+(1-p['alpha'])[...,None]*p['sky']
    def set_stage(self,stage):
        self.field.set_stage(stage)
        self._shadow_stride={'geometry':self.config.shadow_train_stride,'material':max(4,self.config.shadow_train_stride//2),'joint':max(2,self.config.shadow_train_stride//4)}[stage]
        for p in self.light.parameters():p.requires_grad_(True)
    def optimizer(self,stage):
        self.set_stage(stage);c=self.config
        rates={'xyz':c.position_lr*c.spatial_lr_scale,'features_dc':c.appearance_lr,'features_rest':c.appearance_lr/20,'opacity_logit':c.opacity_lr,'log_scales':c.shape_lr,'quaternion':.001};groups=[]
        for name,p in self.named_parameters():
            if not p.requires_grad:continue
            lr=.01 if name.startswith('environment.') else c.light_lr if name.startswith('light.') else rates.get(name.rsplit('.',1)[-1],c.appearance_lr)
            if stage=='material' and name=='field.opacity_logit':lr*=c.material_opacity_lr_scale
            if stage=='joint':lr*=c.joint_lr_scale
            groups.append(dict(params=[p],lr=lr,name=name))
        return torch.optim.Adam(groups,eps=1e-15)
    def schedule(self,opt,step):
        c=self.config;t=min(max(step/c.geometry_steps,0),1);lr=math.exp(math.log(c.position_lr)*(1-t)+math.log(c.position_lr_final)*t)*c.spatial_lr_scale
        for group in opt.param_groups:
            if group['name']=='field.xyz':group['lr']=lr*(c.joint_lr_scale if step>=c.geometry_steps+c.material_steps else 1.)
        self.field.active_sh_degree.fill_(min(3,step//c.sh_increase_interval))

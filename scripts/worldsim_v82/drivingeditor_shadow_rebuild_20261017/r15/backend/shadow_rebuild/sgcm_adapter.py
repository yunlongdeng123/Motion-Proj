"""Use actual inherited SGCMController and guard on the static subset.

Only coordinates/row ownership/optimizer API adapted; original Eq8 and guard retained.
No Actor is pruned by plane height, and no temporal model is introduced.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from .legacy.paper_sgcm import SGCMController,SGCMConfig
from .legacy.hybrid_sgcm_helpers import bind_sgcm_guard
from .topology import remap

class StaticAdapter:
    names={'_xyz':'xyz','_features_dc':'features_dc','_features_rest':'features_rest','_opacity':'opacity_logit','_scaling':'log_scales','_rotation':'quaternion','max_radii2D':'max_radii','xyz_gradient_accum':'screen_gradient_accum','denom':'screen_gradient_count'}
    T=.2
    scaling_inverse_activation=staticmethod(torch.log)
    def __init__(self,f,opt):self.f=f;self.opt=opt;self.reload()
    def reload(self):
        self.rows=(self.f.owner<0).nonzero().flatten();self._sgcm_lineage_ids=self.f.sgcm_lineage[self.rows].clone();self._sgcm_clone_blocked=self.f.sgcm_clone_blocked[self.rows].clone()
    def __getattr__(self,k):
        if k in self.names:return getattr(self.f,self.names[k])[self.rows]
        if k in ('_t','_scaling_t','t_gradient_accum'):return self.f.xyz.new_zeros(len(self.rows),1)
        if k=='_velocity':return self.f.xyz.new_zeros(len(self.rows),3)
        raise AttributeError(k)
    get_xyz=property(lambda s:s.f.xyz[s.rows])
    get_scaling=property(lambda s:s.f.scales[s.rows])
    get_t=property(lambda s:s.f.xyz.new_zeros(len(s.rows),1))
    get_scaling_t=property(lambda s:s.f.xyz.new_full((len(s.rows),1),1e6))
    get_opacity=property(lambda s:s.f.opacity_logit[s.rows].sigmoid())
    def densification_postfix(self,*children,sgcm_parent_indices):
        f=self.f;n=len(f.xyz);f.sgcm_clone_blocked[self.rows]=self._sgcm_clone_blocked
        parents=self.rows[sgcm_parent_indices];rowmap=torch.cat((torch.arange(n,device=f.xyz.device),parents));remap(f,rowmap,self.opt,n)
        for name,x in zip(['xyz','features_dc','features_rest','opacity_logit','log_scales','quaternion'],children[:6]):getattr(f,name)[n:]=x
        f.seed_xyz[n:]=f.xyz[n:];f.anchor_xyz[n:]=f.xyz[n:];f.anchor_log_scales[n:]=f.log_scales[n:];f.anchor_quaternion[n:]=f.quaternion[n:];f.anchor_opacity[n:]=f.opacity_logit[n:]
        f.screen_gradient_accum[n:]=0;f.screen_gradient_count[n:]=0;f.max_radii[n:]=0
        self.reload()
    def prune_points(self,remove):
        keep=torch.ones(len(self.f.xyz),dtype=torch.bool,device=self.f.xyz.device);keep[self.rows[remove]]=False;rows=keep.nonzero().flatten();remap(self.f,rows,self.opt,len(rows));self.reload()

def make_controller(dataset,config):
    from .legacy import hybrid_sgcm_helpers
    hybrid_sgcm_helpers.MIN_VIEW_DEPTH=.2/config.depth_normalization_scale
    v=json.loads(Path(config.sgcm_geometry_path).read_text());cfg=SGCMConfig(**v['config'])
    ctl=SGCMController(cfg,1.,np.array(v['plane_model']),np.array(v['model_to_anchor']),[np.array(x) for x in v['regions_anchor']],v)
    cameras=[]
    for i in dataset.train_indices:
        row=dataset.rows[i];from .raster import Camera
        c=Camera(row['intrinsic'],row['world_from_camera'],320,480);k=c.K;near=.01;far=1000.;p=torch.zeros(4,4);p[0,0]=2*k[0,0]/480;p[1,1]=2*k[1,1]/320;p[0,2]=2*(k[0,2]+.5)/480-1;p[1,2]=2*(k[1,2]+.5)/320-1;p[2,2]=far/(far-near);p[2,3]=-far*near/(far-near);p[3,2]=1
        cameras.append(SimpleNamespace(world_view_transform=c.view.T,full_proj_transform=(p@c.view).T,image_width=480,image_height=320,timestamp=float(i),image_name=row['frame_id']))
    # All static guard timestamps are inert: lifetime is constant1e6, v=0.
    ctl.r15_cameras=cameras
    return ctl

@torch.no_grad()
def refine(ctl,field,opt,package,step):
    ctl.config.max_points=field.config.max_gaussians-int((field.owner>=0).sum())
    proxy=StaticAdapter(field,opt);bind_sgcm_guard(ctl,ctl.r15_cameras,proxy,resuming=True)
    radii=torch.zeros(len(field.xyz),device=field.xyz.device)
    if package['screen_gradients']:radii[package['geometry_rows']]=package['screen_gradients'][0][1].float()
    return ctl.refine(proxy,radii[proxy.rows],step,(radii[proxy.rows]>0))

"""Legacy degree3 SH restored; rigid/evidence operations remain explicit."""
import torch
from torch import nn
from .field_r14 import GaussianField as EvidenceField
from .legacy.sh_utils import RGB2SH,eval_sh
from .math3d import srgb_to_linear,normalize,rotation_z

class GaussianField(EvidenceField):
    point_parameters=('xyz','log_scales','quaternion','opacity_logit','features_dc','features_rest','albedo_logit')
    def __init__(self,seed,config):
        super().__init__(seed,config)
        rgb=torch.as_tensor(seed['rgb'],dtype=torch.float32)
        del self.appearance_logit;del self.view_coeff
        self.features_dc=nn.Parameter(RGB2SH(rgb)[:,None].clone())
        self.features_rest=nn.Parameter(torch.zeros(len(rgb),15,3))
        self.register_buffer('view_coeff',torch.zeros(len(rgb),3,3))
        self.register_buffer('active_sh_degree',torch.tensor(0,dtype=torch.long))
        self.register_buffer('screen_gradient_accum',torch.zeros(len(rgb)))
        self.register_buffer('screen_gradient_count',torch.zeros(len(rgb)))
        self.register_buffer('max_radii',torch.zeros(len(rgb)))
        self.register_buffer('sgcm_lineage',torch.arange(len(rgb),dtype=torch.long))
        self.register_buffer('sgcm_clone_blocked',torch.zeros(len(rgb),dtype=torch.bool))
        if config.repair_actor_geometry:
            # Inherited unchanged through EVERY topology operation. Only an
            # independently certified owner transfer may rebase this coordinate.
            self.register_buffer('geometry_origin',self.xyz.detach().clone())
            self.register_buffer('geometry_origin_sigma',self.scales.detach().amax(-1))
            self.register_buffer('actor_seed_counts',torch.bincount(self.owner[self.owner>=0])[None])
            self.register_buffer('identity_initial_status',getattr(self,'identity_evidence_status',torch.full((len(rgb),),-1,dtype=torch.long)).clone())
        if config.box_half_sizes_path:
            import json
            from pathlib import Path
            half=json.loads(Path(config.box_half_sizes_path).read_text())['half_sizes_m']
            self.register_buffer('actor_box_half',torch.tensor(half,dtype=torch.float32))
    @torch.no_grad()
    def project_constraints(self,stage,optimizer=None):
        super().project_constraints(stage)
        if self.config.repair_actor_geometry:
            from .repair_geometry import project_actor_constraints
            project_actor_constraints(self,optimizer)
    def __getattr__(self,name):
        # Compatibility is read-only, computed from native SH DC; no extra RGB parameter.
        if name=='appearance_logit' and 'features_dc' in self.__dict__.get('_parameters',{}):
            return torch.logit(self.dc_linear().clamp(1e-5,1-1e-5))
        return super().__getattr__(name)
    def dc_linear(self):return srgb_to_linear((.28209479177387814*self.features_dc[:,0]+.5).clamp_min(0))
    def set_stage(self,stage):
        geom=stage=='geometry' or (stage=='joint' and not self.config.freeze_joint_geometry)
        for name in self.point_parameters:
            allowed=geom if name in ('xyz','log_scales','quaternion') else True
            if name=='opacity_logit':allowed=geom or (stage=='material' and self.config.train_opacity_stage2)
            getattr(self,name).requires_grad_(allowed)
    def evaluate(self,poses,valid,camera_center,edits=None):
        g=super().evaluate(poses,valid,camera_center,edits)
        direction=normalize(g['xyz']-camera_center);edits=edits or {}
        for a in range(len(poses)):
            m=g['owner']==a;r=poses[a,:3,:3];e=edits.get(a,edits.get(str(a),{}))
            if 'rotation_matrix' in e:r=torch.as_tensor(e['rotation_matrix'],device=r.device,dtype=r.dtype)@r
            if 'yaw' in e:r=rotation_z(e['yaw'],device=r.device,dtype=r.dtype)@r
            direction=torch.where(m[:,None],direction@r,direction)
        sh=torch.cat((self.features_dc,self.features_rest),1)[g['rows']].transpose(1,2)
        g['appearance']=srgb_to_linear((eval_sh(int(self.active_sh_degree),sh,direction)+.5).clamp_min(0))
        return g
    @torch.no_grad()
    def accumulate_screen_gradients(self,package):
        rows=package['geometry_rows']
        for screen,radii in package.get('screen_gradients',[]):
            if screen.grad is None:continue
            vis=radii>0;ids=rows[vis]
            self.screen_gradient_accum[ids]+=screen.grad[vis,:2].norm(dim=-1)
            self.screen_gradient_count[ids]+=1
            self.max_radii[ids]=torch.maximum(self.max_radii[ids],radii[vis].float())

"""Explicit time-invariant road receiver; shape-derived multi-box caster."""
import sys,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
sys.path.insert(0,str(ROOT/'shadow_BC_20261016_v1/implementation_deferred_v1'))
from bc_model import ActorShadow,hull_indices,srgb_to_linear,linear_to_srgb

class GeometryShadow(ActorShadow):
    def __init__(self,plane,scale,centers,sizes,owners,warm):
        super().__init__(plane,scale,'geometry_parts')
        self.load_state_dict(warm)
        self.register_buffer('part_center',centers)
        self.register_buffer('part_size',sizes)
        self.register_buffer('part_owner',owners)
        self.part_scale_raw=nn.Parameter(torch.zeros_like(sizes))
        self.part_shift_raw=nn.Parameter(torch.zeros_like(centers))
    def mask(self,q_model,positions,rotations,sizes_model,valid):
        q=q_model/self.scale;visibility=torch.ones(len(q),device=q.device)
        # Actual trained Gaussian body envelopes, five longitudinal sections per
        # vehicle. This is a coarse geometric occluder, not an exact mesh shadow.
        for j in range(len(self.part_owner)):
            k=int(self.part_owner[j])
            if not bool(valid[k]):continue
            center=(self.part_center[j]+.08*self.scale*self.part_shift_raw[j].tanh())/self.scale
            size=self.part_size[j]/self.scale*(1+.12*self.part_scale_raw[j].tanh())
            local=self.signs*size+center
            vertices=local@rotations[k].T+positions[k]/self.scale
            height=(vertices@self.normal+self.plane_d_m).clamp_min(0)
            projected=vertices-height[:,None]*self.light/(self.normal@self.light).clamp_min(.10)
            hull=projected@self.basis;hull=hull[hull_indices(hull)]
            query=q@self.basis;edge=torch.roll(hull,-1,0)-hull;rel=query[:,None]-hull[None]
            inward=(edge[None,:,0]*rel[:,:,1]-edge[None,:,1]*rel[:,:,0])/edge.norm(dim=1).clamp_min(1e-8)
            distance=inward.min(1).values;mk=torch.sigmoid(distance/self.softness_m)
            mk=torch.where(distance < -5*self.softness_m,torch.zeros_like(mk),mk)
            # Adjacent sections use max coverage, so each box does not multiply
            # darkening again at its shared boundary. Union over all casters.
            visibility=torch.minimum(visibility,1-mk.clamp(0,1))
        return 1-visibility
    def prior(self):return .002*(self.part_scale_raw.tanh().square().mean()+self.part_shift_raw.tanh().square().mean())
    def describe(self):
        r=super().describe();r.update(parts=len(self.part_owner),source='trained Actor local Gaussian envelope, five longitudinal slices',max_size_change_fraction=.12,max_center_change_m=.08)
        return r

class RoadSurface(nn.Module):
    def __init__(self,data,static,anchor_state,shadow,actor_count):
        super().__init__();self.shadow=shadow;self.static=static;self.actor_count=actor_count
        for k in ['xyz','reference','confidence','support','lo','shape','basis','origin','plane']:
            self.register_buffer(k,data[k])
        self.step=float(data['step']);self.scale=float(data['world_scale'])
        self.register_buffer('ids',torch.nonzero(self.support).ravel())
        self.color_raw=nn.Parameter(torch.logit(self.reference[self.ids].clamp(.001,.999)))
        self.opacity_raw=nn.Parameter(torch.tensor(4.,device='cuda'))
        self.register_buffer('freeze_static',torch.tensor(False,device='cuda'))
        self.anchor_state=anchor_state
        height=(anchor_state['means3D']@self.plane[:3]+self.plane[3])/self.scale
        # Plane-near and below-plane background is replaced only where the
        # world-coordinate confidence chart permits output routing. Above-road
        # foreground stays in the same full-scene depth sort.
        from rigid_model import quaternion_to_matrix
        rotation=quaternion_to_matrix(anchor_state['rotations'])
        normal_sigma=((rotation.transpose(1,2)@self.plane[:3])*anchor_state['scales']).norm(dim=1)/self.scale
        dark=anchor_state['colors_precomp'].amax(1)<.05
        ground=(height.abs()<.20)&(normal_sigma*3<.30)
        ground|=dark&(height.abs()<.40)&(normal_sigma*3<.50)
        self.register_buffer('ground',ground)
        from rigid_model import matrix_to_quaternion
        rot=torch.stack([self.basis[:,0],self.basis[:,1],self.plane[:3]],1)
        self.register_buffer('quaternion',matrix_to_quaternion(rot))
        self.orig_prepare=None
    def chart(self,q):
        xy=(q-self.origin)@self.basis/self.scale
        chart=(xy-self.lo)/self.step
        shape=self.shape.float()
        return 2*(chart+.5)/shape-1
    def confidence_at(self,q):
        grid=self.chart(q)[None,None]
        image=self.confidence.reshape(1,1,int(self.shape[1]),int(self.shape[0]))
        return F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0,0]
    def patch(self,camera,pc,pipe,scaling_modifier=1.,other=None,mask=None):
        if bool(self.freeze_static):
            from renderer import evaluate_sh_rgb
            a=dict(self.anchor_state)
            a['colors_precomp']=evaluate_sh_rgb(pc.get_features,F.normalize(a['means3D']-camera.camera_center,dim=-1),int(pc.active_sh_degree))
        else:a=self.orig_prepare(camera,pc,pipe,scaling_modifier,other,mask)
        ids=self.ids;points=self.xyz[ids];rgb=linear_to_srgb(self.color_raw.sigmoid())
        old_mask=a['mask']&~self.ground
        count=len(points);old=len(a['means3D'])
        result={k:v for k,v in a.items()}
        result['means3D']=torch.cat([a['means3D'],points])
        result['colors_precomp']=torch.cat([a['colors_precomp'],rgb])
        result['opacities']=torch.cat([a['opacities'],self.opacity_raw.sigmoid().expand(count,1)])
        scales=torch.tensor([self.step*.68*self.scale,self.step*.68*self.scale,.0015*self.scale],device='cuda').expand(count,3)
        result['scales']=torch.cat([a['scales'],scales]);result['rotations']=torch.cat([a['rotations'],self.quaternion.expand(count,4)])
        result['mask']=torch.cat([old_mask,torch.ones(count,device='cuda',dtype=torch.bool)])
        # 4 road radiance feature channels + builtin depth/alpha/union +3IDs=10.
        feature=torch.cat([torch.ones(count,1,device='cuda'),rgb],1)
        result['other']=torch.cat([torch.zeros(old,4,device='cuda'),feature])
        return result
    def state(self):return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('static.')}
    def restore(self,state):
        own=self.state_dict()
        with torch.no_grad():
            for k,v in state.items():own[k].copy_(v.to(own[k]))
    def prior(self):
        # Small appearance tether prevents shadow and albedo exchanging roles.
        return .025*abs(self.color_raw.sigmoid()-self.reference[self.ids]).mean()

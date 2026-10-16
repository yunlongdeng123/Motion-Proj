"""Local-road color/opacity adapter and Actor-conditioned directional shadow.

All source checkpoint tensors remain frozen. New state is saved in a separate,
versioned addon. Scope: the evidenced local raw101 road patch, not the full scene.
"""
import itertools,math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def srgb_to_linear(x):
    x=x.clamp_min(0)
    return torch.where(x<=.04045,x/12.92,((x+.055)/1.055).pow(2.4))
def linear_to_srgb(x):
    x=x.clamp_min(0)
    return torch.where(x<=.0031308,12.92*x,1.055*x.clamp_min(1e-10).pow(1/2.4)-.055)
def hull_indices(points):
    values=points.detach().cpu().numpy();ids=sorted(range(len(values)),key=lambda k:tuple(values[k]))
    def cross(i,j,k):
        a=values[j]-values[i];b=values[k]-values[i];return a[0]*b[1]-a[1]*b[0]
    def half(seq):
        result=[]
        for i in seq:
            while len(result)>=2 and cross(result[-2],result[-1],i)<=1e-10:result.pop()
            result.append(i)
        return result
    result=half(ids)[:-1]+half(ids[::-1])[:-1]
    if len(result)<3:raise ValueError('Degenerate projected caster')
    return result

class ActorShadow(nn.Module):
    def __init__(self,plane,scale,kind='projection',azimuth=0.):
        super().__init__();p=torch.as_tensor(plane,dtype=torch.float32);p=p/torch.linalg.norm(p[:3])
        self.register_buffer('normal',p[:3]);self.register_buffer('plane_d_m',p[3]/float(scale))
        seed=torch.eye(3)[int(torch.argmin(abs(p[:3])))];u=torch.linalg.cross(p[:3],seed);u=u/u.norm();v=torch.linalg.cross(p[:3],u)
        self.register_buffer('basis',torch.stack([u,v],1));self.register_buffer('signs',torch.tensor(list(itertools.product([-.5,.5],repeat=3)),dtype=torch.float32))
        self.azimuth=nn.Parameter(torch.tensor(float(azimuth)));self.elevation_raw=nn.Parameter(torch.tensor(0.))
        self.softness_raw=nn.Parameter(torch.tensor(-2.4));self.direct_raw=nn.Parameter(torch.full((3,),1.6))
        self.kind=kind;self.scale=float(scale)
    @property
    def light(self):
        elevation=(.15+1.30*self.elevation_raw.sigmoid())
        tangent=self.basis[:,0]*self.azimuth.cos()+self.basis[:,1]*self.azimuth.sin()
        return self.normal*elevation.sin()+tangent*elevation.cos()
    @property
    def softness_m(self):return .025+F.softplus(self.softness_raw).clamp_max(.7)
    def mask(self,receiver_model,positions,rotations,sizes_model,valid):
        q=receiver_model/self.scale;vis=torch.ones(len(q),device=q.device,dtype=q.dtype)
        for k in range(len(positions)):
            if not bool(valid[k]):continue
            center=positions[k]/self.scale;size=sizes_model[k]/self.scale;rot=rotations[k]
            if self.kind=='contact':
                local=(q-center)@rot
                radius=((local[:,:2].abs()/(size[:2]/2)).pow(4).sum(1)+1e-8).pow(.25)
                mk=torch.exp(-radius*(.6+F.softplus(self.softness_raw)))
            else:
                vertices=(self.signs*size)@rot.T+center
                height=(vertices@self.normal+self.plane_d_m).clamp_min(0)
                projected=vertices-height[:,None]*self.light/(self.normal@self.light).clamp_min(.10)
                hull=projected@self.basis;hull=hull[hull_indices(hull)]
                query=q@self.basis;edge=torch.roll(hull,-1,0)-hull
                rel=query[:,None]-hull[None]
                inward=(edge[None,:,0]*rel[:,:,1]-edge[None,:,1]*rel[:,:,0])/edge.norm(dim=1).clamp_min(1e-8)
                distance=inward.min(1).values
                mk=torch.sigmoid(distance/self.softness_m)
                mk=torch.where(distance < -5*self.softness_m,torch.zeros_like(mk),mk)
            vis=vis*(1-mk.clamp(0,1))
        return 1-vis
    def attenuation(self,receiver_model,positions,rotations,sizes_model,valid):
        union=self.mask(receiver_model,positions,rotations,sizes_model,valid)
        return 1-union[:,None]*self.direct_raw.sigmoid()
    def describe(self):
        return {'kind':self.kind,'world_receiver_to_light':self.light.detach().cpu().tolist(),
                'softness_m':float(self.softness_m.detach()),'direct_fraction_rgb':self.direct_raw.sigmoid().detach().cpu().tolist()}

def edited_casters(instances,timestamp,edits,sizes):
    from rigid_model import quaternion_multiply,quaternion_to_matrix
    pos,quat,valid=instances.trajectory_at(timestamp)
    pos=pos+instances.edit_translation;quat=quaternion_multiply(instances.edit_rotation,quat);valid=valid&~instances.deleted
    if edits:
        pos=pos.clone();quat=quat.clone();valid=valid.clone()
        for track,s in edits.items():
            k=instances._instance_index(track)
            if s.get('delete',False):valid[k]=False
            if 'translation' in s:pos[k]=pos[k]+torch.as_tensor(s['translation'],device=pos.device,dtype=pos.dtype)
            if 'quaternion' in s or 'yaw_radians' in s:
                q=instances._edit_quaternion(s.get('quaternion'),s.get('yaw_radians'))
                quat[k]=quaternion_multiply(q,quat[k])
            if 'replacement' in s:raise ValueError('Replacement caster dimensions need explicit source mapping; outside first pilot')
    return pos,quaternion_to_matrix(quat),sizes,valid

class LocalRoad(nn.Module):
    def __init__(self,selection,static,instances,data,arm,shadow=None):
        super().__init__();device=static.get_xyz.device;self.arm=arm;self.instances=instances;self.static=static;self.shadow=shadow
        self.enabled=True;self.shaded=False;self.edits=None;self.current_frame=12
        for name in ['receiver_ids','artifact_ids','receiver_xyz_anchor','artifact_xyz_anchor']:
            self.register_buffer(name,torch.as_tensor(selection[name],device=device,dtype=torch.long if name.endswith('ids') else torch.float32))
        color=torch.as_tensor(selection['receiver_rgb_anchor'],device=device).clamp(.01,.99)
        self.color_raw=nn.Parameter(torch.logit(color))
        opacity=torch.as_tensor(selection['receiver_base_opacity'],device=device).clamp(.0001,.9999)
        self.opacity_raw=nn.Parameter(torch.logit(opacity))
        self.artifact_raw=nn.Parameter(torch.full((len(self.artifact_ids),),6.,device=device))
        self.register_buffer('initial_opacity',opacity)
        self.register_buffer('sizes',data['sizes_model'])
        self.register_buffer('times',data['times'].float())
        self.scale=float(data['world_scale']);self.original_prepare=None
    @property
    def temporal_fixed(self):return self.arm in ('R2','R3','R4')
    def patch(self,viewpoint_camera,pc,pipe,scaling_modifier=1.,other=None,mask=None):
        result=self.original_prepare(viewpoint_camera,pc,pipe,scaling_modifier,other,mask)
        if not self.enabled:return result
        ri,ai=self.receiver_ids,self.artifact_ids
        opacity=self.opacity_raw.sigmoid()[:,None]
        if self.temporal_fixed:
            result['means3D']=result['means3D'].index_copy(0,ri,self.receiver_xyz_anchor).index_copy(0,ai,self.artifact_xyz_anchor)
            result['opacities']=result['opacities'].index_copy(0,ri,opacity)
            # Removing time gating means both opacity envelope AND visibility prefilter.
            result['mask']=result['mask'].clone();result['mask'][ri]=True;result['mask'][ai]=True
            result['opacities']=result['opacities'].index_copy(0,ai,pc.get_opacity[ai]*self.artifact_raw.sigmoid()[:,None])
        else:
            marginal=pc.get_marginal_t(viewpoint_camera.timestamp)[ri]
            result['opacities']=result['opacities'].index_copy(0,ri,opacity*marginal)
            result['opacities']=result['opacities'].index_copy(0,ai,result['opacities'][ai]*self.artifact_raw.sigmoid()[:,None])
        color=self.color_raw.sigmoid()
        if self.shaded and self.shadow is not None:
            state=edited_casters(self.instances,viewpoint_camera.timestamp,self.edits,self.sizes[self.current_frame])
            attenuation=self.shadow.attenuation(self.receiver_xyz_anchor,*state)
            color=linear_to_srgb(srgb_to_linear(color)*attenuation)
        result['colors_precomp']=result['colors_precomp'].index_copy(0,ri,color)
        return result
    def regularization(self):
        # Encourage unsupported black occluders to disappear; preserve receiving-road alpha.
        return .002*self.artifact_raw.sigmoid().mean()+.001*(self.opacity_raw.sigmoid()-self.initial_opacity).abs().mean()
    def addon_state(self):
        # Do not serialize the registered parent/Actor modules; they remain immutable.
        names=['color_raw','opacity_raw','artifact_raw','receiver_ids','artifact_ids','receiver_xyz_anchor','artifact_xyz_anchor']
        return {name:getattr(self,name).detach().cpu().clone() for name in names}
    def load_addon_state(self,state):
        with torch.no_grad():
            for name,value in state.items():
                current=getattr(self,name)
                if current.shape!=value.shape:raise ValueError('Addon topology mismatch')
                current.copy_(value.to(current))
    def trainable_groups(self):
        return [{'params':[self.color_raw],'lr':.035,'name':'road_color'},
                {'params':[self.opacity_raw,self.artifact_raw],'lr':.06,'name':'local_opacity'}]

def sample_render(image,uv):
    h,w=image.shape[-2:]
    grid=torch.stack(((2*uv[:,0]+1)/w-1,(2*uv[:,1]+1)/h-1),-1)[None,None]
    return F.grid_sample(image[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,0].T

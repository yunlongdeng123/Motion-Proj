"""Parent-relative 2D Mip amplitude control, newroad only; no CUDA changes.

Primary formula: https://github.com/autonomousvision/mip-splatting/blob/main/
submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu (L65-112,244).
This control uses the standard determinant ratio and rejects degenerate inputs;
it is not full Mip-Splatting and does not preserve composited alpha or visibility.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
V44=ROOT.parent/'scene008_v44_rgb_asset_20261107_r1'


def rotation_matrix(q):
    """Standard column rotation; native GLM computes the equivalent R S² Rᵀ."""
    import torch
    w,x,y,z=q.unbind(1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),1).reshape(-1,3,3)


def model_covariance(rows,normalize_quaternion=False):
    import torch
    packed=rows.get('cov3D_precomp')
    if packed is not None:
        p=packed.double()
        return torch.stack((p[:,0],p[:,1],p[:,2],p[:,1],p[:,3],p[:,4],p[:,2],p[:,4],p[:,5]),1).reshape(-1,3,3)
    q=rows['rotations'].double()
    if normalize_quaternion:q=q/q.norm(dim=1,keepdim=True)
    r=rotation_matrix(q);scale=rows['scales'].double()
    return (r*scale.square()[:,None,:])@r.transpose(1,2)


def projected_raw_covariance(camera,rows,neg_fov=True,normalize_quaternion=False):
    import torch
    xyz=rows['means3D'].double();view=camera.world_view_transform.to(device=xyz.device,dtype=torch.float64)
    point=xyz@view[:3,:3]+view[3,:3];z=point[:,2]
    if bool((z==0).any()):raise ValueError('Zero camera depth in relative-Mip projection')
    tx=math.tan(-.5) if neg_fov else math.tan(float(camera.FoVx)*.5)
    ty=math.tan(-.5) if neg_fov else math.tan(float(camera.FoVy)*.5)
    fx=float(camera.image_width)/(2*tx);fy=float(camera.image_height)/(2*ty)
    # Deliberately preserve native min(lim,max(-lim,...)), even when lim<0.
    cx=torch.minimum(z.new_tensor(1.3*tx),torch.maximum(z.new_tensor(-1.3*tx),point[:,0]/z))*z
    cy=torch.minimum(z.new_tensor(1.3*ty),torch.maximum(z.new_tensor(-1.3*ty),point[:,1]/z))*z
    j=xyz.new_zeros((len(z),2,3));j[:,0,0]=fx/z;j[:,1,1]=fy/z
    j[:,0,2]=-fx*cx/z.square();j[:,1,2]=-fy*cy/z.square()
    cov=model_covariance(rows,normalize_quaternion)
    # WV is row-vector world->camera; column covariance uses WV[:3,:3].T.
    camera_cov=view[:3,:3].T[None]@cov@view[:3,:3][None]
    return j@camera_cov@j.transpose(1,2)


def filter_factor(cov,kernel=.3):
    import torch
    a,b,c=cov[:,0,0],cov[:,0,1],cov[:,1,1]
    determinant=a*c-b*b;filtered=(a+kernel)*(c+kernel)-b*b
    if not bool(torch.isfinite(cov).all()) or bool(((determinant<=0)|(filtered<=0)).any()):
        raise ValueError('Degenerate 2D covariance: relative-Mip refuses silent epsilon/clipping')
    factor=(determinant/filtered).sqrt()
    if not bool(torch.isfinite(factor).all()) or bool((factor<=0).any()):raise ValueError('Invalid parent Mip denominator')
    return factor


class RelativeMip:
    policy_id='PARENT_RELATIVE_MIP_NEWROAD_ONLY'
    def __init__(self,parent_asset_path=ROOT/'engineer/inputs/new_road_canonical_candidate.npz'):
        self.parent_asset_path=Path(parent_asset_path)
        self.parent_asset_sha256=hashlib.sha256(self.parent_asset_path.read_bytes()).hexdigest()
        with np.load(parent_asset_path,allow_pickle=False) as z:
            self.parent_xyz=z['model_xyz'].astype(np.float32)
            scale=float(z['meter_to_model']);self.parent_scales=(z['gaussian_sigma_m']*scale).astype(np.float32)
            normal=z['normal_world']@z['world_to_model'][:3,:3].T/scale
            normal/=np.linalg.norm(normal,axis=1)[:,None]
        q=np.c_[1+normal[:,2],-normal[:,1],normal[:,0],np.zeros(len(normal))];q/=np.linalg.norm(q,axis=1)[:,None]
        self.parent_rotation=q.astype(np.float32)
        contract=json.loads((V44/'contract/input_contract.json').read_text())
        self.cfg=json.loads((Path(contract['paths']['native_parent'])/'effective_config.json').read_text())
        self.neg_fov=bool(self.cfg['neg_fov']);self.normalize_quaternion=bool(self.cfg.get('compute_cov3D_python',False))
        self._geometry={};self._parent_factors={};self.records=[]
    def parent_geometry(self,device):
        import torch
        key=str(device)
        if key not in self._geometry:
            cv=lambda a:torch.as_tensor(a,device=device)
            self._geometry[key]=dict(means3D=cv(self.parent_xyz),scales=cv(self.parent_scales),rotations=cv(self.parent_rotation),cov3D_precomp=None)
        return self._geometry[key]
    def __call__(self,camera,rows,parent_indices):
        import torch
        with torch.no_grad():
            parent=self.parent_geometry(rows['means3D'].device)
            ids=parent_indices.long()
            if len(ids)==len(self.parent_xyz) and torch.equal(ids,torch.arange(len(ids),device=ids.device)) and rows.get('scales') is not None:
                if all(torch.equal(rows[k],parent[k]) for k in ('means3D','scales','rotations')):
                    self.records.append(dict(identity_n1_bypass=True,rows=len(ids)))
                    return rows['opacities']
            view=camera.world_view_transform
            # Content key survives camera-object/GPU-address reuse after rt._data.clear().
            key=(view.detach().cpu().contiguous().numpy().tobytes(),float(camera.FoVx),float(camera.FoVy),
                int(camera.image_width),int(camera.image_height),str(rows['means3D'].device))
            if key not in self._parent_factors:
                pcov=projected_raw_covariance(camera,parent,self.neg_fov,self.normalize_quaternion)
                self._parent_factors[key]=filter_factor(pcov)
                if len(self._parent_factors)>8:
                    first=next(iter(self._parent_factors))
                    if first!=key:del self._parent_factors[first]
            parent_f=self._parent_factors[key].index_select(0,ids)
            child_f=filter_factor(projected_raw_covariance(camera,rows,self.neg_fov,self.normalize_quaternion))
            ratio=child_f/parent_f;alpha=rows['opacities']*ratio.to(rows['opacities'].dtype)[:,None]
            if not bool(torch.isfinite(alpha).all()) or bool(((alpha<0)|(alpha>1)).any()):
                raise ValueError('Relative-Mip alpha outside[0,1]; not silently clipped')
            self.records.append(dict(identity_n1_bypass=False,rows=len(ids),
                ratio_min=float(ratio.min().cpu()),ratio_median=float(ratio.median().cpu()),ratio_max=float(ratio.max().cpu()),
                parent_f_min=float(parent_f.min().cpu()),child_f_min=float(child_f.min().cpu()),
                alpha_min=float(alpha.min().cpu()),alpha_max=float(alpha.max().cpu())))
            return alpha
    def receipt(self):
        return dict(policy=self.policy_id,formula='alpha_child=alpha_parent*f_child/f_parent; f=sqrt(detSigma/det(Sigma+.3I))',
            parent_asset_path=str(self.parent_asset_path),parent_asset_sha256=self.parent_asset_sha256,
            kernel_pixel2=.3,legacy_neg_fov=self.neg_fov,legacy_clamp_preserved=True,scaling_modifier=1.,
            cache_key='world_view16 content bytes + FoVx/FoVy/image W/H/device',
            covariance_units='model units with model-unit camera depth; converted exactly once in parent input',
            source='autonomousvision/mip-splatting forward.cu L65-112/244',
            numerical_policy='standard formula; reject degenerate covariance/parent denominator, no epsilon cutoff grid',
            no_covariance_modification=True,original_layers_untouched=True,coverage_preservation_claim=False,
            full_mip_splatting_claim=False,records=self.records)

"""CPU-testable edit-only loss contract and approximate Gaussian preview.

This module does not load or change a master checkpoint and does not launch a run.
The preview is NOT the native CUDA rasterizer and must not be used for final metrics.
"""
import numpy as np

def assert_independent(master_parameters,edit_parameters):
    master=list(master_parameters);edit=list(edit_parameters)
    if {x.untyped_storage().data_ptr() for x in master}&{x.untyped_storage().data_ptr() for x in edit}:
        raise ValueError('Master/edit storage alias: independent edit copy required')

def edit_teacher_loss(road_rgb,road_alpha,road_depth,deleted_rgb,road_contribution,
                      master_rgb,edited_full_rgb,target,selected_mask,core_mask,protect_mask):
    """HWC RGB, HW scalar tensors. Raw observations and generated targets stay distinct.

    road_rgb is the isolated road render over BLACK, not conditional/unpremultiplied
    color. This provides color/opacity gradients even when an opaque static ghost
    blocks the road in the deleted composite. The caller's selective edit-static
    gate still has to expose the road; the full master is never optimized here.
    No SSIM is applied to invented texture. Confidence is an absolute soft weight.
    """
    import torch
    # Real cross-view labels have priority. Never overwrite them with a smooth prior.
    if 'observed_reference_valid' not in target:
        raise ValueError('Explicit observed-reference validity is required to prevent pseudo-label replacement')
    use=selected_mask.bool()&(target['source_kind']==2)&~target['observed_reference_valid'].bool()
    core=use&core_mask.bool();boundary=use&~core
    w=target['loss_weight'].detach();teacher=target['pseudo_rgb'].detach();depth=target['pseudo_depth'].detach()
    def mean_regions(v):
        parts=[]
        for mask in [core,boundary]:
            if mask.any():parts.append((v*w*mask).sum()/mask.sum())
        return torch.stack(parts).mean() if parts else v.sum()*0
    direct=mean_regions((road_rgb-teacher).abs().mean(-1))
    composite=mean_regions((deleted_rgb-teacher).abs().mean(-1))
    depth_loss=mean_regions((road_depth-depth).abs()/depth.clamp_min(1))
    coverage=mean_regions((.8-road_alpha).clamp_min(0))
    visible=mean_regions((.25-road_contribution).clamp_min(0))
    protect=((edited_full_rgb-master_rgb.detach()).abs().mean(-1)*protect_mask).sum()/protect_mask.sum().clamp_min(1)
    total=direct+composite+.1*depth_loss+.1*coverage+.1*visible+protect
    return dict(total=total,direct_road=direct,deleted_composite=composite,depth=depth_loss,
                isolated_coverage=coverage,actual_contribution=visible,protected=protect)

def cpu_splat(cloud,frame,opacity=.65):
    """Reference projection: thin 3D Gaussians, Jacobian covariance, front-to-back.

    Float64 CPU, 3-sigma cutoff, .3px covariance floor, no CUDA-specific sorting /
    early termination. A geometric/coverage sanity preview, not native parity.
    """
    from cpu_plane_propagation import project,transform
    xyz=cloud['local_xyz'];n=cloud['normal_local'];n=n/np.linalg.norm(n,axis=1)[:,None]
    c=transform(xyz,frame['w2l']);uv,z=project(xyz,frame)
    sig=cloud['gaussian_sigma'];eye=np.eye(3)
    cov=sig[:,0,None,None]**2*eye+(sig[:,2]**2-sig[:,0]**2)[:,None,None]*n[:,:,None]*n[:,None,:]
    rot=frame['w2l'][:3,:3];cov=rot[None]@cov@rot.T[None]
    K=frame['K'];J=np.zeros((len(xyz),2,3));safe=np.maximum(z,.1)
    J[:,0,0]=K[0,0]/safe;J[:,0,2]=-K[0,0]*c[:,0]/safe**2
    J[:,1,1]=K[1,1]/safe;J[:,1,2]=-K[1,1]*c[:,1]/safe**2
    cov2=J@cov@J.transpose(0,2,1)+.3*np.eye(2)[None];inv=np.linalg.inv(cov2)
    rad=3*np.sqrt(np.linalg.eigvalsh(cov2)[:,1]);rgb=np.zeros((320,480,3));T=np.ones((320,480));depth=np.zeros((320,480))
    ok=(z>.1)&(uv[:,0]+rad>=0)&(uv[:,0]-rad<480)&(uv[:,1]+rad>=0)&(uv[:,1]-rad<320)
    for i in np.flatnonzero(ok)[np.argsort(z[ok],kind='stable')]:
        x,y=uv[i];r=rad[i];x0=max(0,int(np.floor(x-r)));x1=min(480,int(np.ceil(x+r))+1)
        y0=max(0,int(np.floor(y-r)));y1=min(320,int(np.ceil(y+r))+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];d=np.stack([xx-x,yy-y],-1)
        power=np.einsum('...i,ij,...j->...',d,inv[i],d);a=np.where(power<=9,opacity*np.exp(-.5*power),0)
        weight=T[y0:y1,x0:x1]*a
        rgb[y0:y1,x0:x1]+=weight[...,None]*cloud['rgb'][i]
        depth[y0:y1,x0:x1]+=weight*z[i];T[y0:y1,x0:x1]*=(1-a)
    return dict(rgb=rgb,alpha=1-T,depth=depth/np.maximum(1-T,1e-10))

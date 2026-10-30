"""Conditional finite-support depth/surface responsibility; never unique Gaussian GT."""
import json
import numpy as np
import torch
from pathlib import Path

def audit(rows, camera, transform, root, frame, policy, output):
    from utils.general_utils import build_rotation
    means=rows["means3D"]
    n=len(means)
    counts=torch.zeros((n,5),device=means.device,dtype=torch.int32)
    targets=torch.zeros((n,3),device=means.device)
    # 13 covariance sigma locations: center, +/- one and two std on each local axis.
    axes=build_rotation(rows["rotations"]) * rows["scales"][:,None,:]
    offsets=[torch.zeros_like(means)]
    for radius in (1.,2.):
        for axis in range(3):
            offsets.extend([axes[:,:,axis]*radius,-axes[:,:,axis]*radius])
    data=np.load(root/"original"/(frame+".npz"))
    depth=torch.tensor(data["depth_z_m"],device=means.device)
    semantic=torch.tensor(data["semantic_id"].astype(np.int32),device=means.device)
    surface=torch.tensor(data["surface_id"].astype(np.int32),device=means.device)
    h,w=depth.shape
    scale=float(np.linalg.norm(transform[:3,0]))
    tolerance=.15 if policy=="strict" else .35
    for offset in offsets:
        pts=means+offset
        cam=torch.cat([pts,torch.ones_like(pts[:,:1])],1)@camera.world_view_transform
        z=cam[:,2]
        u=torch.floor(camera.fx*cam[:,0]/z+camera.cx).long()
        v=torch.floor(camera.fy*cam[:,1]/z+camera.cy).long()
        inside=(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
        uu=u.clamp(0,w-1); vv=v.clamp(0,h-1)
        d=depth[vv,uu]; sem=semantic[vv,uu]
        observed=inside&(d>0)&(surface[vv,uu]>0)
        match=observed&((z/scale-d).abs()<=tolerance)
        vehicle=match&(sem==10); nonvehicle=match&(sem!=10)
        overlap=observed&(sem==10)
        counts[:,0]+=vehicle; counts[:,1]+=nonvehicle
        counts[:,2]+=overlap&~match; counts[:,3]+=observed; counts[:,4]+=overlap
        # First-hit surface point corresponding to accepted sample, reconstructed from calibrated depth.
        pcam=torch.stack(((uu+.5-camera.cx)/camera.fx*d*scale,(vv+.5-camera.cy)/camera.fy*d*scale,d*scale,torch.ones_like(d)),1)
        hit=pcam@torch.linalg.inv(camera.world_view_transform)
        targets+=hit[:,:3]*vehicle[:,None]
    vc,nc=counts[:,0],counts[:,1]
    ratio=vc.float()/(vc+nc).clamp_min(1)
    selected=(vc>= (2 if policy=="strict" else 1))&(ratio>=(.9 if policy=="strict" else .5))
    # Use model raster mask as visibility eligibility, without pretending opacity-weighted responsibility.
    selected=selected&rows["mask"].bool().reshape(-1)
    categories=torch.full((n,),3,device=means.device,dtype=torch.int32)
    categories[(counts[:,3]>0)&(vc+nc==0)]=2
    categories[(nc>0)&(vc==0)]=4
    categories[(vc>0)&(nc==0)]=0
    categories[(vc>0)&(nc>0)]=1
    target=targets/vc.clamp_min(1)[:,None]
    path=output/("selection_"+frame+".npz")
    np.savez_compressed(path,selected=selected.cpu().numpy(),static_index=np.arange(n),category=categories.cpu().numpy(),vehicle_support=vc.cpu().numpy(),nonvehicle_support=nc.cpu().numpy(),wrongdepth_vehicle_overlap=counts[:,2].cpu().numpy(),observed_support=counts[:,3].cpu().numpy(),vehicle_overlap=counts[:,4].cpu().numpy(),vehicle_surface_target_model=target.cpu().numpy(),means_model=means.cpu().numpy(),scales_model=rows["scales"].cpu().numpy(),rotations=rows["rotations"].cpu().numpy())
    labels=["vehicle_surface_supported","mixed_surface_support","wrongdepth_or_unmatched","unobserved","nonvehicle_surface_supported"]
    return selected,dict(frame=frame,n_static=n,selected=int(selected.sum()),categories={s:int((categories==i).sum()) for i,s in enumerate(labels)},selection=str(path),depth_tolerance_world_m=tolerance,sigma_samples=13,limitations="finite covariance sigma support is a diagnostic association, not composited alphaT or unique owner GT; behind-first-hit samples are unmatched not proven wrong owners")

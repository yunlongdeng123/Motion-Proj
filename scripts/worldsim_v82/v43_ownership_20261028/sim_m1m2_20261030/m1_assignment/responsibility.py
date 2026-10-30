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
    depth=torch.tensor(data["depth_z_m"],device=means.device,dtype=means.dtype)
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


def native_recall(static, dynamic, camera, pipe, degree, helpers, root, frame, output, transform):
    """Actual native alpha*T sum in independent body pixels; no near-depth gate."""
    geometry=np.load(root/"original"/(frame+".npz"))
    body=torch.tensor(geometry["semantic_id"]==10,device="cuda")
    background=torch.tensor((geometry["semantic_id"]!=10)&(geometry["surface_id"]>0),device="cuda")
    n=len(static["means3D"])
    combined=helpers.merge_rows(static,dynamic)
    with torch.enable_grad():
        features=combined["features"].detach().clone().requires_grad_(True)
        values=dict(combined,features=features)
        package=helpers.raster(camera,pipe,degree,values,torch.zeros(3,device="cuda"))
        body_grad=torch.autograd.grad((package["alpha"][0]*body).sum(),features,retain_graph=True)[0].detach()
        bg_grad=torch.autograd.grad((package["alpha"][0]*background).sum(),features,retain_graph=True)[0].detach()
        body_weight=body_grad[:n,1]; bg_weight=bg_grad[:n,1]
        checks=dict(body_featuregrad_sum=float(body_grad[:,1].sum()),body_raster_alpha_sum=float((package["alpha"][0]*body).sum()),background_featuregrad_sum=float(bg_grad[:,1].sum()),background_raster_alpha_sum=float((package["alpha"][0]*background).sum()),unused_depth_channel_gradient_max=float(torch.maximum(body_grad[:,0].abs().max(),bg_grad[:,0].abs().max())),dynamic_body_alphaT=float(body_grad[n:,1].sum()),unified_static_dynamic_depth_sort=True)
        checks["max_relative_conservation_error"]=max(abs(checks[k+"_featuregrad_sum"]-checks[k+"_raster_alpha_sum"])/max(1,checks[k+"_raster_alpha_sum"]) for k in ("body","background"))
        if checks["max_relative_conservation_error"]>2e-5 or checks["unused_depth_channel_gradient_max"]>1e-8: raise RuntimeError(checks)
        h,w=body.shape
        yy,xx=torch.meshgrid(torch.arange(h,device="cuda")+.5,torch.arange(w,device="cuda")+.5,indexing="ij")
        depth=torch.tensor(geometry["depth_z_m"],device="cuda",dtype=features.dtype)*float(np.linalg.norm(transform[:3,0]))
        cam_xyz=torch.stack(((xx-camera.cx)/camera.fx*depth,(yy-camera.cy)/camera.fy*depth,depth,torch.ones_like(depth)),-1)
        xyz=(cam_xyz@torch.linalg.inv(camera.world_view_transform))[...,:3]
        moments=[]
        for mask in (body,background):
            m=[]
            for axis in range(3):
                grad=torch.autograd.grad((package["alpha"][0]*mask*xyz[:,:,axis]).sum(),features,retain_graph=True)[0][:n,1].detach()
                m.append(grad)
            moments.append(torch.stack(m,1))
        vehicle_target=moments[0]/body_weight.clamp_min(1e-12)[:,None]
        background_target=moments[1]/bg_weight.clamp_min(1e-12)[:,None]
    selected=(body_weight>1e-4)&static["mask"].bool().reshape(-1)
    retained={k:(None if v is None else v[~selected]) for k,v in static.items()}
    after=helpers.raster(camera,pipe,degree,retained,torch.zeros(3,device="cuda"))
    after_body_alpha=float((after["alpha"][0]*body).sum())

    path=output/("selection_"+frame+".npz")
    np.savez_compressed(path,vehicle_surface_target_model=vehicle_target.cpu().numpy(),background_surface_target_model=background_target.cpu().numpy(),vehicle_target_valid=(body_weight>1e-8).cpu().numpy(),background_target_valid=(bg_weight>1e-8).cpu().numpy(),selected=selected.cpu().numpy(),static_index=np.arange(n),body_alphaT=body_weight.cpu().numpy(),background_alphaT=bg_weight.cpu().numpy(),means_model=static["means3D"].cpu().numpy(),scales_model=static["scales"].cpu().numpy(),rotations=static["rotations"].cpu().numpy())
    summary=dict(delete_after_same_body_alphaT_sum=after_body_alpha,delete_after_same_body_alphaT_scope="actual retained-static raster after deleting selected static plus all dynamic; reveals newly unoccluded contributions, not original-weight subtraction",moment_target="actual native alphaT weighted first-hit XYZ using original train view geometry only; not edited or heldout GT",feature_gradient_verification=checks,frame=frame,selected=int(selected.sum()),n_static=n,selected_with_background_conflict=int((selected&(bg_weight>1e-4)).sum()),body_contribution_pixel_sum=float(body_weight.sum()),selected_body_contribution_pixel_sum=float(body_weight[selected].sum()),selected_background_contribution_pixel_sum=float(bg_weight[selected].sum()),selection=str(path),selection_rule="native unified-scene sum(body pixels alpha*T)>1e-4; no depth gate; no background-conflict veto",persistent_identity_solved=False,unique_gaussian_gt=False)
    return selected,summary

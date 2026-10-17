"""Original hybrid full-image losses plus explicit ownership/road constraints."""
import json
from pathlib import Path
import torch
from .legacy.hybrid30k_losses import compose_hybrid_loss,photometric_rgb_loss
from .legacy.paper_losses import sky_loss,sparse_inverse_depth_loss,nps_loss,boundary_distance
from .legacy.loss_utils import ssim
from .legacy.rigid_losses import visible_instance_loss,local_box_boundary
from .losses import geometry_and_reference_terms,pixel_valid,instance_body_loss
from .competition import competition_terms
from .math3d import linear_to_srgb,srgb_to_linear

def smooth_inverse(depth,rgb):
    import kornia
    return kornia.losses.inverse_depth_smoothness_loss(depth.clamp_min(1e-5).reciprocal()[None,None],rgb.permute(2,0,1)[None])

def objective(model,frame,actor_count,backend,stage,generator,teacher=None,global_step=0):
    c=model.config;g,p=model.buffers(frame,actor_count,backend);image=model.train_image(p,g)
    target=linear_to_srgb(frame.rgb);photo=lambda x,y:photometric_rgb_loss(x.permute(2,0,1),y.permute(2,0,1),ssim,.2)
    union=(frame.masks&pixel_valid(frame)).any(-1).to(target.dtype)[...,None]
    # Known actors' local invalid pixels are omitted; unknown whole actors never
    # produce negative identity labels. RGB stays observed across the full image.
    valid=(pixel_valid(frame)|~frame.mask_known[None,None]).all(-1)[...,None].to(target.dtype)
    static=p['static_isolated_rgb']-p['static_isolated_road_rgb']*p['rho_image']+(1-p['static_isolated_alpha'])[...,None]*p['sky']
    dynamic=p['dynamic_isolated_rgb'];zero=image.sum()*0
    sky=torch.zeros_like(p['alpha']) if frame.sky is None else frame.sky.to(p['alpha'])
    alpha=p['alpha'];depth=p['depth'];far_world=900/c.depth_normalization_scale
    depth=(alpha/depth.clamp_min(1e-5)+(1-alpha)/far_world).clamp_min(1e-5).reciprocal()
    static_alpha=p['static_isolated_alpha'];features=target.permute(2,0,1)*0
    state=dict(iteration=global_step,global_rgb=photo(linear_to_srgb(image),target),dynamic_rgb=photo(linear_to_srgb(dynamic)*union,target*union),sky_rgb=photo(linear_to_srgb(static)*sky[...,None],target*sky[...,None]),static_rgb=photo(linear_to_srgb(static)*(1-union)*valid,target*(1-union)*valid),sky=sky_loss(static_alpha[None],sky[None]),depth=sparse_inverse_depth_loss(depth[None]*c.depth_normalization_scale,frame.depth[None]*c.depth_normalization_scale),inverse_depth_smoothness=smooth_inverse(depth,target),velocity_smoothness=zero,dynamic_alpha=p['visible_labels'].sum(-1)[None],dynamic_mask=union[...,0][None],static_t_map=2*(alpha-p['visible_labels'].sum(-1))[None]/alpha[None].clamp_min(1e-5),static_v_map=features,alpha=alpha[None],nps_features=features,nps_distances=boundary_distance(union[...,0][None]))
    policy=json.loads(Path(c.legacy_reference_config).read_text()) if c.legacy_reference_config else {}
    policy.update(hybrid_lambda_dynamic_opacity=0.,hybrid_lambda_dynamic_rgb=5.)
    hybrid=compose_hybrid_loss(state,policy,nps_fn=nps_loss)
    terms=geometry_and_reference_terms(model,p,frame,c,stage);terms.update(competition_terms(p,frame,c,global_step))
    # Depth is already supervised with the original formula/decay in hybrid.
    terms.pop('depth')
    terms['body_rgb']=instance_body_loss((linear_to_srgb(image)-target).abs(),frame)
    terms['body_teacher']=zero
    if teacher is not None and c.use_body_teacher:
        with torch.no_grad():_,tp=teacher.buffers(frame,actor_count,backend,isolated=False)
        terms['body_teacher']=instance_body_loss((p['canonical']-tp['canonical']).abs(),frame)
    from dataclasses import asdict
    weights=asdict(c.weights);total=hybrid.total+sum(value*weights[name] for name,value in terms.items())
    # Retain old100*equal-instance full-HW visible-identity MSE, with the
    # explicitly authorized pixel-valid extension. Added balanced terms above
    # protect small assets; they do not silently replace the baseline strength.
    pv=pixel_valid(frame)
    mask_loss=visible_instance_loss((p['visible_labels']*pv).permute(2,0,1),(frame.masks&pv).permute(2,0,1),frame.mask_known)
    boundary=zero;f=model.field;actor=f.owner>=0
    if hasattr(f,'actor_box_half') and actor.any():
        from types import SimpleNamespace
        adapter=SimpleNamespace(box_size=f.actor_box_half*2,pose_valid=None,local_xyz=f.xyz[actor],point_ids=f.owner[actor])
        boundary=local_box_boundary(adapter)*c.depth_normalization_scale**2
    total=total+c.legacy_visible_mask_weight*mask_loss+c.legacy_box_boundary_weight*boundary
    terms['legacy_visible_mask']=mask_loss;terms['legacy_box_boundary']=boundary
    terms.update({'legacy_'+k:value for k,value in hybrid.raw.items()});terms['hybrid_total']=hybrid.total
    return total,terms,p

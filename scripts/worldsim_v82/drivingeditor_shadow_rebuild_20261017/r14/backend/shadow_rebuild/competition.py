"""适配离散3DGS的表面竞争；不使用NeRF密度比值/整像素静动态熵。"""
import torch
from .losses import masked_mean,pixel_valid,instance_domain_mean


def competition_terms(package,frame,config,step):
    zero=package['alpha'].sum()*0
    if not config.continuous_ownership:
        return dict(ownership_occupancy=zero,ownership_render=zero)
    # 分支单独渲染避免前层T≈0藏住同表面重复；自身分支仍按深度alpha合成。
    sa=package['static_isolated_alpha'];da=package['dynamic_isolated_alpha']
    sd=package['static_isolated_depth'];dd=package['dynamic_isolated_depth']
    delta=(sd-dd).abs()
    # Gate has no direct gradient. Future geometry updates may still change the
    # gate; real depth supervision and seed/joint bounds remain necessary.
    match=(torch.exp(-.5*(delta/config.ownership_depth_tolerance_m).square())*
           (delta<3*config.ownership_depth_tolerance_m)*(sa>1e-4)*(da>1e-4)).detach()
    valid=pixel_valid(frame);body=frame.masks&valid
    certified=getattr(frame,'road_certified',None)
    road=torch.zeros_like(sa,dtype=torch.bool) if certified is None else certified&~body.any(-1)
    dynamic_visible=package['visible_labels'].sum(-1)
    static_visible=(package['alpha']-dynamic_visible).clamp_min(0)
    # Evidence chooses only the losing branch. Stop-gradient on the supported
    # branch protects tires/body and avoids the easy "erase both" solution.
    body_occ=(sa*da.detach()*match)[...,None].expand_as(body)
    body_render=(static_visible*dynamic_visible.detach()*match)[...,None].expand_as(body)
    occ=instance_domain_mean(body_occ,body,frame.mask_known)+masked_mean(da*sa.detach()*match,road)
    visible=instance_domain_mean(body_render,body,frame.mask_known)+masked_mean(dynamic_visible*static_visible.detach()*match,road)
    ramp=min(1.,max(0.,(step+1)/config.ownership_warmup_steps))
    return dict(ownership_occupancy=occ*ramp,ownership_render=visible*ramp)

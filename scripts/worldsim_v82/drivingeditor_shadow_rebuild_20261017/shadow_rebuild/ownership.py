"""证据驱动归属与跨边界分解。不可见/未知不是背景；无车底高度裁剪。"""
import math
import torch
from .math3d import quaternion_matrix


def classify_evidence(evidence, frames, owner, actors, min_frames=3):
    """Columns: own body, other known body, measured road, full-scene visibility.

    Only repeated POSITIVE road evidence can nominate a road reassignment. Mixed
    body/road support nominates splitting, never wholesale deletion/reassignment.
    Output candidates require a separate measured world-surface anchor check.
    """
    own,other,road,visible=evidence.unbind(-1)
    semantic=(own+other+road).clamp_min(1e-9)
    selected=torch.zeros_like(owner,dtype=torch.bool)
    for actor in actors:selected|=owner==actor
    body_ratio=own/semantic;road_ratio=road/semantic
    repeated_road=(frames[:,2]>=min_frames)&(road>=.5)
    repeated_body=(frames[:,0]>=min_frames)&(own>=.5)
    road_candidate=selected&repeated_road&(road_ratio>.8)&(body_ratio<.05)&(~repeated_body)
    mixed=selected&repeated_body&(frames[:,2]>=min_frames)&(road>=.2)
    unknown=selected&(~road_candidate)&(~mixed)&((semantic<.5)|(visible<.5))
    return dict(road_candidate=road_candidate,mixed=mixed,unknown=unknown)


def split_moments(cut):
    """Two truncated 1D normal moments; mixture preserves mass/mean/covariance.

    cut is signed in parent standard deviations, selected from actual projected
    mask intersection. Both children survive. This differs from SAGD's retained
    in-mask piece and discarded exterior. Bounds avoid negligible children.
    """
    if not bool(torch.isfinite(cut).all()) or bool((cut.abs()>1.5).any()):
        raise ValueError('boundary cut must be finite within +/-1.5 sigma')
    density=torch.exp(-.5*cut.square())/math.sqrt(2*math.pi)
    left=.5*(1+torch.erf(cut/math.sqrt(2)));right=1-left
    ml=-density/left;mr=density/right
    vl=(1-cut*density/left-ml.square()).clamp_min(1e-6)
    vr=(1+cut*density/right-mr.square()).clamp_min(1e-6)
    return torch.stack((left,right),-1),torch.stack((ml,mr),-1),torch.stack((vl,vr),-1)


@torch.no_grad()
def split_boundary(field, indices, axes, cuts, optimizer):
    """Keep both children, preserve unaffected Adam rows, reset changed rows.

    Existing parent rows become left children; right children are appended.
    Geometry anchors follow the new children. Identity is deliberately unchanged
    until independent multi-frame contribution + measured surface re-audit.
    """
    n=len(field.xyz);m=len(indices);device=field.xyz.device
    if not m:return dict(parents=0,children=0)
    if len(torch.unique(indices))!=m or bool((indices<0).any()) or bool((indices>=n).any()):raise ValueError('unique valid parent rows required')
    if axes.shape!=(m,) or cuts.shape!=(m,) or bool(((axes<0)|(axes>2)).any()):raise ValueError('invalid local split axes/cuts')
    fraction,offset,variance=split_moments(cuts)
    rowmap=torch.cat((torch.arange(n,device=device),indices))
    replacements={}
    for name in field.point_parameters:
        old=getattr(field,name);new=torch.nn.Parameter(old[rowmap].clone(),requires_grad=old.requires_grad)
        setattr(field,name,new);replacements[old]=new
    for name,value in list(field.named_buffers()):
        if value.ndim and len(value)==n:setattr(field,name,value[rowmap].clone())
    scales=field.scales[indices].clone();basis=quaternion_matrix(field.quaternion[indices])
    batch=torch.arange(m,device=device);direction=basis[batch,:,axes]
    means=field.xyz[indices].clone();sigma=scales[batch,axes]
    alpha=field.opacity_logit[indices,0].sigmoid();tau=-torch.log1p(-alpha)
    for side,rows in enumerate((indices,torch.arange(n,n+m,device=device))):
        field.xyz[rows]=means+direction*(sigma*offset[:,side])[:,None]
        child_scales=scales.clone();child_scales[batch,axes]*=variance[:,side].sqrt()
        field.log_scales[rows]=child_scales.log()
        # Density integral/mass preserved before explicit bounded-opacity projection.
        child_tau=tau*fraction[:,side]/variance[:,side].sqrt()
        field.opacity_logit[rows,0]=torch.logit((-torch.expm1(-child_tau)).clamp(1e-5,.999))
        for anchor,value in [('seed_xyz',field.xyz),('anchor_xyz',field.xyz),('anchor_log_scales',field.log_scales),
                             ('anchor_quaternion',field.quaternion),('anchor_opacity',field.opacity_logit)]:getattr(field,anchor)[rows]=value[rows]
    affected=torch.cat((indices,torch.arange(n,n+m,device=device)))
    for group in optimizer.param_groups:
        for j,old in enumerate(group['params']):
            if old not in replacements:continue
            new=replacements[old];group['params'][j]=new;state=optimizer.state.pop(old,{})
            newstate={}
            for key,value in state.items():
                if torch.is_tensor(value) and value.ndim and len(value)==n:
                    value=torch.cat((value,torch.zeros_like(value[indices])),0);value[affected]=0
                newstate[key]=value
            optimizer.state[new]=newstate
    # Each child initially points to its same-identity sibling. Global kNN rebuilt after any reassignment.
    field.neighbor[indices]=torch.arange(n,n+m,device=device);field.neighbor[n:]=indices
    return dict(parents=m,children=2*m,both_children_retained=True,identity_changed=False,
                adam='unaffected rows preserved; changed parent and new child rows reset')


@torch.no_grad()
def reanchor_road(field,indices,measured_xyz,measured_quaternion,optimizer):
    """Explicit measured-world-surface reanchor, not freezing an arbitrary Actor pose.

    The caller must supply repeated positive road attribution AND actual repeated
    world LiDAR anchors with source frame/point provenance. No new gray color or
    synthetic point. Colors are subsequently re-fit from train RGB.
    """
    if not len(indices):return
    if measured_xyz.shape!=(len(indices),3) or measured_quaternion.shape!=(len(indices),4):raise ValueError('measured anchor shape mismatch')
    field.xyz[indices]=measured_xyz;field.quaternion[indices]=measured_quaternion
    field.owner[indices]=-1;field.road[indices]=True;field.view_coeff[indices]=0
    field.reference_confidence[indices]=0;field.reference_rgb[indices]=0
    for key,value in [('seed_xyz',field.xyz),('anchor_xyz',field.xyz),('anchor_log_scales',field.log_scales),
                      ('anchor_quaternion',field.quaternion),('anchor_opacity',field.opacity_logit)]:getattr(field,key)[indices]=value[indices]
    for name in field.point_parameters:
        for value in optimizer.state.get(getattr(field,name),{}).values():
            if torch.is_tensor(value) and value.ndim and len(value)==len(field.xyz):value[indices]=0

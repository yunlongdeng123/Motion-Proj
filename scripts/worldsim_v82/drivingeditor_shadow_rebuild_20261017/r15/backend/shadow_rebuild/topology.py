"""Spatial 3DGS clone/split/prune with preserved rigid identity and source rows.

Adapted from existing GaussianModel, including its original 0.8*N split scale.
Temporal splitting is excluded because this representation is truly static/rigid.
"""
import torch
from torch import nn
from .math3d import quaternion_matrix

def generic_due(config,step):
    return bool(config.densify_interval and step>config.densify_from and step<config.densify_until and step%config.densify_interval==0)

@torch.no_grad()
def remap(field,rowmap,optimizer,new_start):
    n=len(field.xyz);replacements={}
    for name in field.point_parameters:
        old=getattr(field,name);new=nn.Parameter(old[rowmap].clone(),requires_grad=old.requires_grad);setattr(field,name,new);replacements[old]=new
    inverse=torch.full((n,),-1,device=rowmap.device,dtype=torch.long);inverse[rowmap[:new_start]]=torch.arange(new_start,device=rowmap.device)
    for name,b in list(field.named_buffers()):
        if b.ndim and len(b)==n:
            out=b[rowmap].clone()
            if name=='neighbor':
                out=inverse[out];bad=out<0;out[bad]=torch.arange(len(out),device=out.device)[bad]
            setattr(field,name,out)
    for group in optimizer.param_groups:
        for j,old in enumerate(group['params']):
            if old not in replacements:continue
            new=replacements[old];group['params'][j]=new;state=optimizer.state.pop(old,{})
            if state:
                optimizer.state[new]={k:torch.cat((x[rowmap[:new_start]],torch.zeros_like(x[rowmap[new_start:]])),0) if torch.is_tensor(x) and x.ndim and len(x)==n else x for k,x in state.items()}

@torch.no_grad()
def densify_prune(field,optimizer,step):
    c=field.config;n=len(field.xyz);score=field.screen_gradient_accum/field.screen_gradient_count.clamp_min(1)
    selected=score>=c.topology_grad_threshold
    if c.repair_actor_geometry:
        from .repair_geometry import select_actor_growth
        selected=select_actor_growth(field,selected,score)
    ids=selected.nonzero().flatten();budget=max(0,c.max_gaussians-n)
    if len(ids)>budget:
        ids=ids[score[ids].topk(budget).indices];selected.zero_();selected[ids]=True
    split=selected&(field.scales.max(-1).values>c.topology_percent_dense*c.spatial_lr_scale);clone=selected&~split
    split_ids=split.nonzero().flatten();clone_ids=clone.nonzero().flatten()
    prune=field.opacity_logit.sigmoid()[:,0]<c.prune_opacity_threshold
    if step>c.opacity_reset_interval:prune|=(field.max_radii>20)|(field.scales.max(-1).values>.1*c.spatial_lr_scale)
    prune&=~selected
    remain=(~split&~prune).nonzero().flatten();parents=torch.cat((clone_ids,split_ids.repeat_interleave(2)))
    if not len(remain) and not len(parents):raise ValueError('topology would erase all geometry')
    rowmap=torch.cat((remain,parents));xyz=field.xyz.detach().clone();scales=field.scales.detach().clone();q=quaternion_matrix(field.quaternion).detach()
    remap(field,rowmap,optimizer,len(remain));first=len(remain)+len(clone_ids)
    if len(split_ids):
        sigma=scales[split_ids].repeat_interleave(2,0);offset=torch.randn_like(sigma)*sigma
        offset=(q[split_ids].repeat_interleave(2,0)@offset[...,None]).squeeze(-1)
        field.xyz[first:]=xyz[split_ids].repeat_interleave(2,0)+offset;field.log_scales[first:]=(sigma/(.8*2)).log()
        field.seed_xyz[first:]=field.xyz[first:];field.anchor_xyz[first:]=field.xyz[first:]
    if c.repair_actor_geometry:
        from .repair_geometry import project_actor_constraints
        project_actor_constraints(field,optimizer)
    field.screen_gradient_accum.zero_();field.screen_gradient_count.zero_();field.max_radii.zero_()
    return dict(before=n,after=len(field.xyz),cloned=len(clone_ids),split_parents=len(split_ids),pruned=int(prune.sum()),capacity=c.max_gaussians)

@torch.no_grad()
def reset_opacity(field,optimizer):
    field.opacity_logit.copy_(torch.logit(field.opacity_logit.sigmoid().clamp_max(.01)))
    for value in optimizer.state.get(field.opacity_logit,{}).values():
        if torch.is_tensor(value) and value.ndim:value.zero_()

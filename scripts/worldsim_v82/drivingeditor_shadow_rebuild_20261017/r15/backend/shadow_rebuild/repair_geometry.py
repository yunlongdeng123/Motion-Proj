"""Actor-local geometry budgets. No deletion, opacity change or inferred owner."""
import math
import torch


@torch.no_grad()
def project_actor_constraints(field,optimizer=None):
    rows=(field.owner>=0).nonzero().flatten()
    if not len(rows):return
    if not hasattr(field,'actor_box_half'):raise ValueError('actual Actor box dimensions required')
    c=field.config;before=field.xyz[rows].clone();origin=field.geometry_origin[rows]
    delta=before-origin
    projected=origin+delta*(c.actor_drift_bound_m/delta.norm(dim=-1,keepdim=True).clamp_min(c.actor_drift_bound_m))
    # Every initial seed remains feasible, including initially outside seeds.
    # This limits NEW outward growth, and does not label/crop initial fragments.
    limit=torch.maximum(field.actor_box_half[field.owner[rows]]+c.actor_box_margin_m,origin.abs())
    projected=torch.maximum(torch.minimum(projected,limit),-limit)
    changed=(projected-before).abs()>1e-7
    field.xyz[rows]=projected
    previous_scale=field.log_scales[rows].clone()
    maximum=(field.geometry_origin_sigma[rows]*c.actor_scale_growth_limit).log()[:,None]
    field.log_scales[rows]=torch.minimum(previous_scale,maximum)
    # Remove only clipped coordinate moments: an outward Adam impulse must not
    # repeatedly push a constrained point. Unaffected rows/components stay exact.
    if optimizer is not None:
        for param,mask in [(field.xyz,changed),(field.log_scales,previous_scale>maximum)]:
            for value in optimizer.state.get(param,{}).values():
                if torch.is_tensor(value) and value.shape==param.shape:
                    part=value[rows];part[mask]=0;value[rows]=part


@torch.no_grad()
def select_actor_growth(field,selected,score):
    result=selected.clone();c=field.config
    if c.repair_evidence_audit:
        selected=selected&((field.owner<0)|(field.identity_evidence_status==1))
        result=selected.clone()
    for actor in field.owner[field.owner>=0].unique().tolist():
        owned=field.owner==actor;count=int(owned.sum())
        cap=math.floor(float(field.actor_seed_counts[0,actor])*c.actor_max_growth)
        # clone and split both add exactly ONE net point per selected parent.
        budget=min(math.floor(count*c.densify_fraction),max(0,cap-count))
        ids=(selected&owned).nonzero().flatten()
        result[ids]=False
        if budget and len(ids):result[ids[score[ids].topk(min(budget,len(ids))).indices]]=True
    return result


@torch.no_grad()
def geometry_receipt(field):
    records=[]
    for actor in [5,6,18,26,37]:
        m=field.owner==actor
        if not bool(m.any()):continue
        xyz=field.xyz[m];origin=field.geometry_origin[m];half=field.actor_box_half[actor]
        limit=torch.maximum(half+field.config.actor_box_margin_m,origin.abs())
        displacement=(xyz-origin).norm(dim=-1)
        records.append(dict(actor=actor,points=int(m.sum()),outside_exact_box=int((xyz.abs()>half).any(-1).sum()),
            outside_allowed_envelope=int((xyz.abs()>limit+1e-5).any(-1).sum()),
            max_origin_displacement_m=float(displacement.max()),
            drift_violations=int((displacement>field.config.actor_drift_bound_m+1e-5).sum()),
            unknown=int((field.identity_evidence_status[m]==0).sum()) if hasattr(field,'identity_evidence_status') else None))
    return records

"""R15候选：局部因子解析积分、空间块粗筛、五方向一次提交。

已验证的 FP64 因子积分；通过显式 shadow_integrator 配置接入模型。粗筛是10sigma
保守AABB，误差必须对无粗筛FP64参考验收；无caster删减或独立影子代理。
"""
from dataclasses import dataclass,field
import math
import types
import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class Factors:
    means: torch.Tensor
    rotation: torch.Tensor
    scales: torch.Tensor
    opacity: torch.Tensor
    owners: torch.Tensor
    cache: dict=field(default_factory=dict)


def _spread(v):
    v=v&1023
    v=(v|(v<<16))&0x030000FF
    v=(v|(v<<8))&0x0300F00F
    v=(v|(v<<4))&0x030C30C3
    return (v|(v<<2))&0x09249249


@torch.no_grad()
def block_plan(factors,block_size=256,sigma=10.):
    key=('plan',block_size,float(sigma))
    if key in factors.cache:return factors.cache[key]
    means=factors.means.detach();parts=[];valids=[]
    for actor in factors.owners.unique().tolist():
        ids=(factors.owners==actor).nonzero().flatten()
        xyz=means[ids];lo=xyz.amin(0);span=(xyz.amax(0)-lo).clamp_min(1e-8)
        cell=((xyz-lo)/span*1023).long().clamp(0,1023)
        code=_spread(cell[:,0])|(_spread(cell[:,1])<<1)|(_spread(cell[:,2])<<2)
        ids=ids[torch.argsort(code,stable=True)]
        pad=(-len(ids))%block_size
        valid=torch.arange(len(ids)+pad,device=ids.device)<len(ids)
        if pad:ids=torch.cat((ids,ids[-1:].expand(pad)))
        parts.append(ids.reshape(-1,block_size));valids.append(valid.reshape(-1,block_size))
    if not parts:return None
    ids=torch.cat(parts);valid=torch.cat(valids)
    # 刚性因子对有微小非正交舍入的pose仍按实际A A^T计算边界。
    axes=factors.rotation.detach()*factors.scales.detach()[:,None,:]
    radius=sigma*axes.square().sum(-1).sqrt()
    lo=(means-radius)[ids].amin(1);hi=(means+radius)[ids].amax(1)
    result=(ids,valid,lo,hi)
    factors.cache[key]=result
    return result


def whitening(factors,dtype):
    key=('whitening',dtype)
    if key not in factors.cache:
        # 只解近似旋转的良态因子，不构造病态世界协方差的inverse。
        r=factors.rotation.to(dtype);s=factors.scales.to(dtype)
        eye=torch.eye(3,device=r.device,dtype=dtype).expand_as(r)
        w=torch.linalg.solve(r,eye)/s[:,:,None]
        alpha=factors.opacity.to(dtype).flatten().clamp(1e-6,1-1e-6)
        amplitude=-torch.log1p(-alpha)/(math.sqrt(2*math.pi)*s.amin(-1))
        factors.cache[key]=(factors.means.to(dtype),w,amplitude)
    return factors.cache[key]


def pair_sum(origins,directions,means,w,amplitude,ray_ids,point_ids,valid):
    delta=origins[ray_ids,None,:]-means[point_ids]
    ww=w[point_ids]
    # 按主轴逐分量算白化坐标，避免einsum/bmm为极小3x3重复调度。
    y=(ww*delta[:,:,None,:]).sum(-1)
    v=(ww*directions[ray_ids,None,None,:]).sum(-1)
    speed=v.square().sum(-1).sqrt();u=v/speed[...,None]
    center=-(y*u).sum(-1)
    a=y[...,1]*u[...,2]-y[...,2]*u[...,1]
    b=y[...,2]*u[...,0]-y[...,0]*u[...,2]
    c=y[...,0]*u[...,1]-y[...,1]*u[...,0]
    transverse=a.square()+b.square()+c.square()
    tail=.5*torch.erfc(-center/math.sqrt(2))
    tau=amplitude[point_ids]*torch.exp(-.5*transverse)*(math.sqrt(2*math.pi)/speed)*tail
    return (tau*valid).sum(-1)


def transmittance(origins,directions,factors,block_size=256,pair_batch=256,
                  dtype=torch.float64,recompute=True,sigma=10.,stats=None):
    plan=block_plan(factors,block_size,sigma)
    if plan is None:return origins[:,0]*0+1
    ids,valid,lo,hi=plan
    with torch.no_grad():
        o=origins.detach().double();d=directions.detach().double()
        parallel=d.abs()<1e-10;safe=torch.where(parallel,torch.ones_like(d),d)
        t1=(lo[None]-o[:,None])/safe[:,None]
        t2=(hi[None]-o[:,None])/safe[:,None]
        near=torch.minimum(t1,t2).masked_fill(parallel[:,None],-torch.inf).amax(-1).clamp_min(0)
        far=torch.maximum(t1,t2).masked_fill(parallel[:,None],torch.inf).amin(-1)
        invalid=(parallel[:,None]&((o[:,None]<lo[None])|(o[:,None]>hi[None]))).any(-1)
        hits=((far>=near)&~invalid).nonzero()
    m,w,amp=whitening(factors,dtype)
    o=origins.to(dtype);d=directions.to(dtype)
    tau=o[:,0]*0
    if stats is not None:stats.update(blocks=len(ids),ray_block_hits=len(hits),
        pair_integral_upper_bound=len(hits)*block_size,checkpoint_calls=0)
    for start in range(0,len(hits),pair_batch):
        take=hits[start:start+pair_batch];rays=take[:,0];blocks=take[:,1]
        pi=ids[blocks];mask=valid[blocks]
        args=(o,d,m,w,amp,rays,pi,mask)
        if recompute and torch.is_grad_enabled() and any(x.requires_grad for x in args):
            value=checkpoint(pair_sum,*args,use_reentrant=False)
            if stats is not None:stats['checkpoint_calls']+=1
        else:value=pair_sum(*args)
        tau=tau.index_add(0,rays,value)
    # 与原模型的tau上限一致；未遮挡极限保持1。
    return torch.exp(-tau.clamp_max(80)).to(origins.dtype)


def factors_for_geometry(model,frame,g,edits=None):
    from shadow_rebuild.math3d import quaternion_matrix,rotation_z
    use=g['owner']>=0;rows=g['rows'][use];owners=g['owner'][use]
    r=frame.poses[owners,:3,:3].double()
    edits=edits or {}
    if edits:
        r=r.clone()
        for actor,edit in edits.items():
            actor=int(actor);mask=owners==actor
            if 'rotation_matrix' in edit:r[mask]=torch.as_tensor(edit['rotation_matrix'],device=r.device,dtype=r.dtype)@r[mask]
            if 'yaw' in edit:r[mask]=rotation_z(edit['yaw'],device=r.device,dtype=r.dtype)@r[mask]
    basis=r@quaternion_matrix(model.field.quaternion[rows].double())
    scales=model.field.log_scales[rows].double().exp().clamp(model.config.scale_min_m,model.config.scale_max_m)
    return Factors(g['xyz'][use],basis,scales,g['opacity'][use,0],owners)

"""批量GPU有限三角扇；完整片×射线关系，不截断后继链。"""
import torch
import numpy as np
from .surface_state import Surface

def pack(surfaces,capacity=16,device='cuda',dtype=torch.float64):
    b=len(surfaces);c=torch.zeros(b,capacity,3,device=device,dtype=dtype)
    rot=torch.eye(3,device=device,dtype=dtype).expand(b,capacity,3,3).clone()
    rho=torch.full((b,capacity,8),.15,device=device,dtype=dtype);active=torch.zeros(b,capacity,device=device,dtype=torch.bool)
    for i,s in enumerate(surfaces):
        k=len(s.center)
        if k>capacity:raise ValueError('padding capacity cannot discard active surfaces')
        c[i,:k]=torch.as_tensor(s.center,device=device);rot[i,:k]=torch.as_tensor(s.rotation,device=device)
        rho[i,:k]=torch.as_tensor(s.radius,device=device);active[i,:k]=True
    return {'c':c,'rot':rot,'rho':rho,'active':active}

def unpack(s):
    out=[]
    for c,r,rho,a in zip(*(s[k].detach().cpu().numpy() for k in ['c','rot','rho','active'])):
        out.append(Surface(c[a].copy(),r[a].copy(),rho[a].copy()))
    return out

def observations(items,device='cuda',dtype=torch.float64):
    names={'o':'origins_actor_m','d':'directions_actor','y':'observed_first_range_m','positive':'positive_actor','ambiguous':'ambiguous_owner'}
    result={k:torch.as_tensor(np.stack([o[n] for o in items]),device=device,dtype=torch.bool if k in ['positive','ambiguous'] else dtype) for k,n in names.items()}
    result['positive'] &= ~result['ambiguous']
    return result

def geometry(s,obs):
    c,r,rho=s['c'],s['rot'],s['rho'];o,d=obs['o'],obs['d'];n=r[:,:,:,2]
    den=torch.einsum('brc,bkc->brk',d,n);safe=torch.where(den!=0,den,torch.ones_like(den))
    t=torch.einsum('brkc,bkc->brk',c[:,None]-o[:,:,None],n)/safe
    rel=o[:,:,None]+t[:,:,:,None]*d[:,:,None]-c[:,None]
    uv=torch.einsum('brkc,bkcj->brkj',rel,r[:,:,:,:2])
    angle=torch.arange(8,device=c.device,dtype=c.dtype)*torch.pi/4
    ring=rho[:,:,:,None]*torch.stack([angle.cos(),angle.sin()],-1)
    a=ring[:,None];b=ring.roll(-1,2)[:,None];q=uv[:,:,:,None]
    cross=a[...,0]*b[...,1]-a[...,1]*b[...,0]
    u=(q[...,0]*b[...,1]-q[...,1]*b[...,0])/cross
    v=(a[...,0]*q[...,1]-a[...,1]*q[...,0])/cross
    inside=((u>=0)&(v>=0)&(u+v<=1)).any(-1)
    valid=(den!=0)&(t>0)&torch.isfinite(t)&s['active'][:,None]
    hit=inside&valid
    depth=torch.where(hit,t,torch.full_like(t,float('inf')));first,win=depth.min(-1)
    phi=torch.remainder(torch.atan2(uv[...,1],uv[...,0]),2*torch.pi)
    sector=(phi/(torch.pi/4)).floor().long()%8;local=phi-sector*torch.pi/4
    rr=rho[:,None].expand(-1,o.shape[1],-1,-1)
    r0=rr.gather(-1,sector[...,None]).squeeze(-1);r1=rr.gather(-1,((sector+1)%8)[...,None]).squeeze(-1)
    boundary=r0*r1*2**-.5/(r1*torch.sin(torch.pi/4-local)+r0*torch.sin(local)).clamp_min(1e-30)
    sd=uv.norm(dim=-1)-boundary
    return {'t':t,'uv':uv,'den':den,'valid':valid,'hit':hit,'depth':depth,'first':first,'winner':win,'sector':sector,'sd':sd}

def cost(s,obs,g=None):
    g=geometry(s,obs) if g is None else g;first=g['first'];finite=torch.isfinite(first);y=obs['y'];positive=obs['positive']
    safe=torch.where(finite,first,y);reg=(safe-y).abs().clamp(max=5)
    first_loss=torch.where(finite,reg,torch.full_like(reg,2.))
    first_loss=(first_loss*positive).sum(-1)/positive.sum(-1).clamp_min(1)
    free=torch.where(finite,(y-.2-safe).clamp(min=0,max=5),torch.zeros_like(y)).mean(-1)
    # 到支撑平面的法向距离与片外径向距离是共同训练proxy；最终几何指标用实际三角面距离。
    normal=((g['t']-y[:,:,None])*g['den']).abs()
    support=torch.sqrt(normal.square()+g['sd'].clamp_min(0).square())
    support=torch.where(s['active'][:,None],support,torch.full_like(support,5.)).min(-1).values.clamp(max=5)
    support=(support*positive).sum(-1)/positive.sum(-1).clamp_min(1)
    return first_loss+free+support

def rotation(delta):
    theta=delta.norm(dim=-1,keepdim=True);x,y,z=delta.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*delta.shape[:-1],3,3)
    a=torch.sinc(theta/torch.pi)[...,None];b=.5*torch.sinc(theta/(2*torch.pi)).square()[...,None]
    return torch.eye(3,device=delta.device,dtype=delta.dtype)+a*skew+b*(skew@skew)

def continuous(s,delta):
    delta=delta.to(s['c'].dtype)*s['active'][...,None]
    return {'c':s['c']+delta[...,:3].clamp(-.75,.75),
            'rot':rotation(delta[...,3:6].clamp(-.3,.3))@s['rot'],
            'rho':(s['rho']*delta[...,6:].clamp(-1.6,.8).exp()).clamp(1e-4,10.),'active':s['active'].clone()}

def index_state(s,index):return {k:v[index] for k,v in s.items()}

def features(s,obs,g=None,winner_only=False):
    g=geometry(s,obs) if g is None else g;b,r,k=g['t'].shape
    o=obs['o'][:,:,None].expand(-1,-1,k,-1)/10;d=obs['d'][:,:,None].expand(-1,-1,k,-1);y=obs['y'][:,:,None].expand(-1,-1,k)
    t=torch.where(torch.isfinite(g['t']),g['t'],torch.zeros_like(g['t']))
    edge=torch.cat([o,d,(y/10)[...,None],torch.asinh(t/10)[...,None],torch.asinh(t-y)[...,None],
                    torch.asinh(g['uv']),torch.asinh(g['sd'])[...,None],obs['positive'][:,:,None,None].expand(-1,-1,k,-1),
                    g['hit'][...,None],g['valid'][...,None]],-1).float()
    valid=s['active'][:,None].expand(-1,r,-1)
    if winner_only:valid=valid&g['hit']&(torch.arange(k,device=t.device)[None,None]==g['winner'][...,None])
    edge=edge*valid[...,None]
    order=torch.argsort(torch.where(g['valid'],t,torch.full_like(t,float('inf'))),dim=-1,stable=True).flip(-1)
    node=torch.cat([s['c']/3,s['rot'][...,:,2],s['rho'].log()],-1).float()
    return {'node':node,'edge':edge,'order':order,'active':s['active'],'edge_valid':valid}

def candidate_cost(states,obs,chunk=128):
    b,e=states['c'].shape[:2];flat={k:v.flatten(0,1) for k,v in states.items()};expanded={k:v[:,None].expand(b,e,*v.shape[1:]).flatten(0,1) for k,v in obs.items()}
    return torch.cat([cost({k:v[i:i+chunk] for k,v in flat.items()},{k:v[i:i+chunk] for k,v in expanded.items()}) for i in range(0,b*e,chunk)]).reshape(b,e)

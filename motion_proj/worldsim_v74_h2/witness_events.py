"""共享见证事件池；普通控制获得相同候选，联合更新必须重新求交。"""
import torch
import numpy as np
from .surface_state import initialize,Surface
from .gpu_geometry import geometry,continuous,pack

def birth_pool(observations,device='cuda'):
    candidates=[];counts=[]
    for obs in observations:
        pos=obs['positive_actor']&~obs['ambiguous_owner'];points=obs['points_actor_m'][pos]
        raw=initialize(obs,count=min(4,len(points)),width=.4);valid=[];evidence=[]
        for k,c in enumerate(raw.center):
            local=np.linalg.norm(points-c,axis=1)<.8
            plane=np.abs((points-c)@raw.rotation[k,:,2])<.2
            support=local&plane
            # 多个共面正见证授权出生；不由单条MISS直接生片。
            if support.sum()>=3:
                valid.append(k);evidence.append(int(support.sum()))
        candidates.append(Surface(raw.center[valid],raw.rotation[valid],raw.radius[valid]));counts.append(evidence)
    return pack(candidates,4,device),counts

def witness_delta(s,obs,g):
    y=obs['y'][:,:,None];positive=obs['positive'][:,:,None]
    normal_res=(g['t']-y)*g['den'];close=(normal_res.abs()<.4)&(g['sd']<.8)&positive&g['valid']
    # 表面的持续正支撑与前缀侵入分开聚合；不把first-hit责任预测成标签头。
    count=close.sum(1);move=-(normal_res*close).sum(1)/count.clamp_min(1)
    placement=move[...,None]*s['rot'][...,:,2];placement=placement.clamp(-.5,.5)
    prefix=g['valid']&(g['t']<y-.2)
    uvnorm=g['uv'].norm(dim=-1)
    target=[];pos_counts=[];neg_counts=[]
    for sector in range(8):
        adjacent=(g['sector']==sector)|(((g['sector']+1)%8)==sector)
        support=close&adjacent;negative=prefix&adjacent
        lower=torch.where(support,uvnorm*1.15,torch.zeros_like(uvnorm)).amax(1)
        upper=torch.where(negative,uvnorm*.9,torch.full_like(uvnorm,float('inf'))).amin(1)
        current=s['rho'][...,sector];supported=support.any(1)
        desired=torch.where(supported,torch.maximum(current,lower),current)
        # 冲突片保留局部正支持下界；分裂由同池候选提供。不是逐射线不许变差的gate。
        desired=torch.maximum(lower,torch.minimum(desired,upper)).clamp_min(1e-4)
        target.append((desired/current).log().clamp(-1.6,.8));pos_counts.append(support.sum(1));neg_counts.append(negative.sum(1))
    delta=torch.cat([placement,torch.zeros_like(placement),torch.stack(target,-1)],-1)*s['active'][...,None]
    evidence=torch.stack([torch.stack(pos_counts,-1),torch.stack(neg_counts,-1)],-1)
    return delta,evidence

def proposals(s,obs,birth,step,budget=64,ordinary=False):
    b,p=s['active'].shape;device=s['c'].device;dtype=s['c'].dtype
    g=geometry(s,obs);joint,evidence=witness_delta(s,obs,g)
    zero=torch.zeros(b,p,14,device=device,dtype=dtype)
    entries=[]
    def add(delta,kind=0,index=0,name='continuous'):
        entries.append((delta,kind,index,name))
    add(zero.clone(),name='keep')
    if ordinary:
        # 通用成对块坐标步：没有witness支撑投影；对任意前三个活动片组合公平提供。
        for ka,kb in [(0,1),(0,2),(1,2)]:
            for scale in [.2,.6]:
                x=zero.clone();x[:,[ka,kb],6:]=np.log(scale);add(x,name='ordinary_pair_scale')
    else:
        for factor in [.5,1.]:
            add(joint*factor,name='witness_joint')
            x=joint.clone();x[...,6:]=0;add(x*factor,name='witness_placement')
            x=joint.clone();x[...,:6]=0;add(x*factor,name='witness_extent')
    # 普通块更新也可同时改变所有片，不能只给A跨层联合修复。
    for scale in [.2,.6,1.4]:
        x=zero.clone();x[...,6:]=np.log(scale);add(x,name='all_uniform_scale')
    # 活动片按完整冲突/支撑量轮换。P1最多11活动片，所有片连续目标始终保留。
    priority=((evidence[...,1].sum(-1)+evidence[...,0].sum(-1))*s['active']).argsort(-1,descending=True)
    slot=s['active'].sum(-1).clamp(max=p-1);available=(s['active'].sum(-1)<p)&(s['active'].sum(-1)<512)
    for k in range(4):add(zero.clone(),kind=1,index=k,name='cluster_birth')
    for rank in range(min(p,4)):
        ix=priority[:,rank];rows=torch.arange(b,device=device)
        x=zero.clone()
        if ordinary:x[rows,ix,6+(step+rank)%8]=np.log(.2)
        else:x[rows,ix]=joint[rows,ix]
        add(x,name='ordinary_radius' if ordinary else 'one_witness_update')
        add(zero.clone(),kind=2,index=rank,name='split_supported_conflict')
        for scale in [.2,.6,1.4]:
            x=zero.clone();x[rows,ix,6:]=np.log(scale);add(x,name='one_uniform_scale')
        for axis in range(3):
            for sign in [-1,1]:
                x=zero.clone();x[rows,ix,axis]=sign*.3*(.7**(step//2));add(x,name='ordinary_placement')
        # 交替边界坐标覆盖8半径；同池8步内不固定剥夺某些扇区。
        sector=(step+rank*2)%8
        for scale in [.5,1.5]:
            x=zero.clone();x[rows,ix,6+sector]=np.log(scale);add(x,name='ordinary_radius')
        for sign in [-1,1]:
            x=zero.clone();x[rows,ix,3:6]=s['rot'][rows,ix,:,step%2]*.1*sign;add(x,name='ordinary_rotation')
    entries=entries[:budget]
    while len(entries)<budget:add(zero.clone(),name='unused_keep')
    delta=torch.stack([e[0] for e in entries],1)
    delta*=s['active'][:,None,:,None]
    flat={k:v[:,None].expand(b,budget,*v.shape[1:]).flatten(0,1) for k,v in s.items()}
    new=continuous(flat,delta.flatten(0,1));new={k:v.reshape(b,budget,*v.shape[1:]) for k,v in new.items()}
    valid=torch.ones(b,budget,device=device,dtype=torch.bool)
    types=torch.tensor([e[1] for e in entries],device=device);ids=torch.tensor([e[2] for e in entries],device=device)
    rows=torch.arange(b,device=device)
    for e,(_,kind,index,name) in enumerate(entries):
        if kind==1:
            ok=available&birth['active'][:,index];valid[:,e]=ok
            for key in ['c','rot','rho']:new[key][rows,e,slot]=birth[key][:,index]
            new['active'][rows,e,slot]=ok
        elif kind==2:
            ix=priority[:,index];ok=available&s['active'][rows,ix];valid[:,e]=ok
            shift=s['rot'][rows,ix,:,step%2]*s['rho'][rows,ix].mean(-1)[:,None]*.35
            new['c'][rows,e,ix]-=shift;new['rho'][rows,e,ix]*=.75
            new['c'][rows,e,slot]=s['c'][rows,ix]+shift;new['rot'][rows,e,slot]=s['rot'][rows,ix];new['rho'][rows,e,slot]=s['rho'][rows,ix]*.75
            new['active'][rows,e,slot]=ok
            delta[rows,e,ix,:3]=-shift;delta[rows,e,ix,6:]=np.log(.75)
    fresh_c=new['c'][rows[:,None],torch.arange(budget,device=device)[None],slot[:,None]]
    fresh_n=new['rot'][rows[:,None],torch.arange(budget,device=device)[None],slot[:,None],:,2]
    fresh_r=new['rho'][rows[:,None],torch.arange(budget,device=device)[None],slot[:,None]]
    fresh=torch.cat([fresh_c/3,fresh_n,fresh_r.mean(-1,keepdim=True),fresh_r.amin(-1,keepdim=True),fresh_r.amax(-1,keepdim=True)],-1)
    fresh*=types[None,:,None]>0
    desc=torch.cat([delta.float().mean(2),delta.float().abs().amax(2),
                    torch.nn.functional.one_hot(types,3)[None].expand(b,-1,-1).float(),fresh.float()],-1)
    return {'states':new,'delta':delta,'valid':valid,'kind':types,'index':ids,'desc':desc,
            'evidence':evidence,'names':[e[3] for e in entries],'priority':priority,'slot':slot}

def choose(pool,index):
    rows=torch.arange(len(index),device=index.device)
    return {k:v[rows,index] for k,v in pool['states'].items()}

def learned_step(s,delta,pool,index):
    result=continuous(s,delta);rows=torch.arange(len(index),device=index.device)
    topology=pool['kind'][index]>0
    if topology.any():
        # 新片本身用共同端点/分裂候选；既有片始终使用网络的连续参数输出。
        rr=rows[topology];ee=index[topology];ss=pool['slot'][topology]
        for k in result:result[k][rr,ss]=pool['states'][k][rr,ee,ss]
    return result

"""DCS：真实有限八边形、LP 对偶需求、连续几何提议、整数选择；与 A/B 无依赖。"""
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from scipy.spatial import cKDTree
import torch
from torch import nn
from .common import local_frames,farthest_ids,save_mesh,write_json,ray_state


class PricingNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.point=nn.Sequential(nn.Linear(11,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(135,64),nn.SiLU(),nn.Linear(64,7))
    def forward(self,tokens,global_features):
        h=self.point(tokens);x=self.head(torch.cat([h.max(1).values,h.mean(1),global_features],1))
        return torch.tanh(x)*torch.tensor([1.,1.,.4,.6,.6,.7,.7],device=x.device)


def geometry_context(build):
    p=build['support_points_actor_m'];owned=np.flatnonzero(build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool))
    # P0 保留的 support 点与正观测点顺序相同；不读取 QUERY 建立几何框。
    origins=build['origins_actor_m'][owned]
    frame,values=local_frames(p,p,origins if len(origins)==len(p) else None)
    return {'points':p,'owned':owned,'frame':frame,'tree':cKDTree(p),'values':values}


def patches_from_parameters(context,anchors,parameters,width):
    anchors=np.asarray(anchors,int);parameters=np.asarray(parameters).reshape(-1,7)
    base=context['frame'][anchors];center=context['points'][anchors]+np.einsum('nck,nk->nc',base,parameters[:,:3]*width)
    normal=base[:,:,2]+base[:,:,0]*parameters[:,3,None]+base[:,:,1]*parameters[:,4,None]
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
    tangent=base[:,:,0]-(base[:,:,0]*normal).sum(1)[:,None]*normal
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-10)
    second=np.cross(normal,tangent);scale=width*np.exp(parameters[:,5:7])
    return {'center':center,'u':tangent,'v':second,'normal':normal,'scale':scale}


def concat_patches(a,b):return {k:np.concatenate([a[k],b[k]],axis=0) for k in a}
def subset_patches(a,ids):return {k:v[ids] for k,v in a.items()}


def export_patches(patches,selected):
    selected=np.asarray(selected,int);p=subset_patches(patches,selected);n=len(selected)
    if not n:return np.empty((0,3)),np.empty((0,3),int)
    angle=np.arange(8)*np.pi/4
    ring=p['center'][:,None]+p['u'][:,None]*np.cos(angle)[None,:,None]*p['scale'][:,None,0:1]+p['v'][:,None]*np.sin(angle)[None,:,None]*p['scale'][:,None,1:2]
    verts=np.concatenate([p['center'][:,None],ring],axis=1).reshape(-1,3)
    base=np.array([[0,i+1,(i+1)%8+1] for i in range(8)])
    faces=(base[None]+np.arange(n)[:,None,None]*9).reshape(-1,3)
    return verts,faces


@torch.no_grad()
def patch_incidence(patches,build,epsilon):
    """精确凸八边形平面求交；包围矩形仅不参与返回计算。"""
    device='cuda';p={k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in patches.items()}
    owned=np.flatnonzero(build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool));mapping=np.full(len(build['positive_actor']),-1,int);mapping[owned]=np.arange(len(owned))
    er=[];ec=[];ar=[];ac=[];n=len(p['center'])
    angle=torch.arange(8,device=device)*np.pi/4+np.pi/8
    normals=torch.stack([angle.cos(),angle.sin()],1)
    for start in range(0,len(build['origins_actor_m']),256):
        o=torch.as_tensor(build['origins_actor_m'][start:start+256],dtype=torch.float32,device=device)
        d=torch.as_tensor(build['directions_actor'][start:start+256],dtype=torch.float32,device=device)
        den=d@p['normal'].T;distance=((p['center'][None]-o[:,None])*p['normal'][None]).sum(-1)/torch.where(den.abs()>1e-8,den,torch.ones_like(den))
        hitpoint=o[:,None]+distance[:,:,None]*d[:,None]-p['center'][None]
        uv=torch.stack([(hitpoint*p['u'][None]).sum(-1)/p['scale'][None,:,0],(hitpoint*p['v'][None]).sum(-1)/p['scale'][None,:,1]],-1)
        inside=((uv@normals.T).max(-1).values<=np.cos(np.pi/8)+1e-7)&(den.abs()>1e-8)&(distance>0)
        target=torch.as_tensor(build['observed_first_range_m'][start:start+len(o)],device=device)[:,None]
        ri,ci=torch.where(inside&(distance<target-epsilon));er.extend((ri.cpu().numpy()+start).tolist());ec.extend(ci.cpu().numpy().tolist())
        ri,ci=torch.where(inside&((distance-target).abs()<=epsilon));rr=ri.cpu().numpy()+start;cc=ci.cpu().numpy();take=mapping[rr]>=0
        ar.extend(mapping[rr[take]].tolist());ac.extend(cc[take].tolist())
    A=sp.csr_matrix((np.ones(len(ar)),(ar,ac)),shape=(len(owned),n));E=sp.csr_matrix((np.ones(len(er)),(er,ec)),shape=(len(build['origins_actor_m']),n))
    return A,E


def patch_cost(patches,width):
    area=2*np.sqrt(2)*patches['scale'].prod(1)
    return 1.+.02*area/(width*width)


def master(A,E,cost,face_budget,binary=False,seconds=30):
    j=A.shape[1];r=A.shape[0]
    # 完全透明连续混合从不作为输出；LP只取影子价格。真实自由段 nu=0。
    C=sp.vstack([sp.hstack([-A,-sp.eye(r)]),sp.hstack([E,sp.csr_matrix((E.shape[0],r))]),
                 sp.csr_matrix(np.r_[np.full(j,8.),np.zeros(r)][None])],format='csc')
    bounds=np.r_[np.full(r,-1.),np.zeros(E.shape[0]),face_budget]
    objective=np.r_[cost,np.full(r,100.)]
    if binary:
        res=milp(objective,integrality=np.r_[np.ones(j),np.zeros(r)],bounds=Bounds(np.zeros(j+r),np.ones(j+r)),
            constraints=LinearConstraint(C,np.full(len(bounds),-np.inf),bounds),options={'time_limit':seconds,'mip_rel_gap':1e-5})
        z=res.x[:j] if res.x is not None else np.zeros(j)
        return {'z':z,'objective':float(res.fun) if res.fun is not None else float(100*r),'status':int(res.status),
                'gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None,'s':res.x[j:] if res.x is not None else np.ones(r)}
    res=linprog(objective,A_ub=C,b_ub=bounds,bounds=[(0,1)]*(j+r),method='highs',options={'time_limit':seconds})
    if not res.success:raise RuntimeError('LP demand unavailable: '+res.message)
    dual=np.asarray(res.ineqlin.marginals)
    return {'z':res.x[:j],'objective':float(res.fun),'alpha':-dual[:r],'beta':-dual[r:r+E.shape[0]],'eta':float(-dual[-1]),'s':res.x[j:],'status':int(res.status)}


def pricing_features(build,context,anchors,demand,width,no_demand=False):
    anchors=np.asarray(anchors,int);points=context['points'];frame=context['frame'];k=min(16,len(points))
    distance,nn=context['tree'].query(points[anchors],k=k);nn=np.asarray(nn).reshape(len(anchors),k)
    local=points[nn]-points[anchors,None];base=frame[anchors]
    pos=np.einsum('nik,npi->npk',base,local)/max(width,1e-6)
    normal=np.einsum('nik,npi->npk',base,frame[nn,:,2])
    direction=build['directions_actor'][context['owned'][nn]]
    direction=np.einsum('nik,npi->npk',base,direction)
    alpha=demand['alpha'][nn]/100;beta=demand['beta'][context['owned'][nn]]/100
    tokens=np.concatenate([pos,normal,direction,alpha[:,:,None],beta[:,:,None]],axis=2)
    global_features=[]
    for a in anchors:
        center=points[a];o=build['origins_actor_m'];d=build['directions_actor'];dt=((center-o)*d).sum(1)
        near=o+dt[:,None]*d;offset=center-near;distance=np.linalg.norm(offset,axis=1)
        weight=demand['beta']*np.exp(-distance**2/max(width**2,1e-6))*(dt>0)*(dt<build['observed_first_range_m'])
        vec=(offset*weight[:,None]).sum(0)/max(weight.sum(),1.)/max(width,1e-6)
        vec=frame[a].T@vec
        global_features.append([demand['eta'],float(demand['alpha'].mean()/100) if len(points) else 0.,float(demand['s'].mean()),float(weight.max()/100) if len(weight) else 0.,*vec])
    global_features=np.asarray(global_features)
    if no_demand:tokens[:,:,-2:]=0;global_features[:]=0
    return tokens.astype(np.float32),global_features.astype(np.float32)


def load_network(path):
    state=torch.load(path,map_location='cpu',weights_only=False);net=PricingNetwork().cuda().eval();net.load_state_dict(state['state_dict']);return net


def reconstruct(build,params,variant,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();cfg=params['dcs'];width=params['carrier_half_width_m'];eps=params['epsilon_obs_m']
    if not len(build['support_points_actor_m']):return np.empty((0,3)),np.empty((0,3),int),{'status':'missing_build','wall_seconds':0.}
    context=geometry_context(build);p=context['points'];anchors=farthest_ids(p,cfg['initial_patches'])
    patches=patches_from_parameters(context,anchors,np.zeros((len(anchors),7)),width)
    A,E=patch_incidence(patches,build,eps);cost=patch_cost(patches,width);integer=master(A,E,cost,params['face_budget'],True)
    selected=np.flatnonzero(integer['z']>.5);v,f=export_patches(patches,selected);save_mesh(out/'initial.npz',v,f)
    learned=variant in ['DCS','C2_local_network','C3_no_demand']
    net=None
    if learned:
        key='full' if variant=='DCS' else 'no_demand'
        net=load_network(params['dcs_checkpoints'][key])
    events=[]
    for step in range(0 if variant=='C0_fixed' else cfg['pricing_rounds']):
        demand=master(A,E,cost,params['face_budget'])
        if time.monotonic()-start>=cfg['wall_seconds']:break
        if variant in ['DCS','C3_no_demand']:
            order=np.argsort(-demand['alpha'],kind='stable')
        else:
            covered=np.asarray(A@integer['z']).ravel();order=np.argsort(covered,kind='stable')
        # 每轮固定提议数；循环平移打破 LP 对偶退化造成的恒定同一锚点。
        # 在同需求值的整个队列中继续访问新锚点；不能每4轮回到固定32点浪费160次预算。
        anchors=np.roll(order,-step*cfg['proposals_per_round'])[:cfg['proposals_per_round']]
        if not len(anchors):break
        if learned:
            tokens,global_features=pricing_features(build,context,anchors,demand,width,no_demand=variant!='DCS')
            with torch.no_grad():parameters=net(torch.tensor(tokens,device='cuda'),torch.tensor(global_features,device='cuda')).cpu().numpy()
        else:
            parameters=np.zeros((len(anchors),7))
            # 普通局部高残差增密：以邻域点的有限二维范围决定半径，仍用相同八边形主问题。
            distances,nn=context['tree'].query(p[anchors],k=min(8,len(p)))
            distances=np.asarray(distances).reshape(len(anchors),-1)
            parameters[:,5:7]=np.clip(np.log(np.maximum(np.median(distances,axis=1),width/2)/width),-.7,.7)[:,None]
        proposals=patches_from_parameters(context,anchors,parameters,width)
        newA,newE=patch_incidence(proposals,build,eps);newcost=patch_cost(proposals,width)
        price=newcost-np.asarray(newA.T@demand['alpha']).ravel()+np.asarray(newE.T@demand['beta']).ravel()+8*demand['eta']
        eligible=np.flatnonzero(price<-1e-7)
        event={'step':step,'alpha_min_max':[float(demand['alpha'].min()),float(demand['alpha'].max())],
               'beta_nonzero':int((demand['beta']>1e-8).sum()),'eta':demand['eta'],'anchors':anchors.tolist(),
               'exact_reduced_cost':price.tolist(),'new_correct_support':np.asarray(newA.sum(0)).ravel().astype(int).tolist(),
               'new_early_support':np.asarray(newE.sum(0)).ravel().astype(int).tolist(),'eligible_columns':len(eligible),
               'lp_objective':demand['objective'],'integer_objective_before':integer['objective'],'lp_integer_gap':integer['objective']-demand['objective'],
               'reason':'no_negative_price','accepted_column_count':0}
        if len(eligible):
            trialA=sp.hstack([A,newA[:,eligible]],format='csr');trialE=sp.hstack([E,newE[:,eligible]],format='csr');trialcost=np.r_[cost,newcost[eligible]]
            trial=master(trialA,trialE,trialcost,params['face_budget'],True,seconds=max(1,min(30,cfg['wall_seconds']-(time.monotonic()-start))))
            if trial['objective']<integer['objective']-1e-6:
                patches=concat_patches(patches,subset_patches(proposals,eligible));A,E,cost=trialA,trialE,trialcost;integer=trial
                event['accepted_column_count']=len(eligible);event['reason']='true_integer_objective_improved'
            else:event['reason']='negative_LP_price_but_no_integer_improvement'
        selected=np.flatnonzero(integer['z']>.5);v,f=export_patches(patches,selected)
        actual=ray_state(v,f,build,eps)
        event.update({'integer_objective_after':integer['objective'],'integer_solver_status':integer['status'],'integer_solver_gap':integer['gap'],
            'candidate_count':len(cost),'selected_count':len(selected),'actual_build_early':int(actual['early'].sum()),
            'actual_build_hit':int(actual['hit'][context['owned']].sum())})
        save_mesh(out/f'event_{step:02d}.npz',v,f,**patches,selected=selected)
        np.savez_compressed(out/f'pricing_{step:02d}.npz',parameters=parameters,anchors=anchors,alpha=demand['alpha'],beta=demand['beta'],eta=demand['eta'],exact_price=price)
        events.append(event)
    selected=np.flatnonzero(integer['z']>.5);v,f=export_patches(patches,selected);save_mesh(out/'final.npz',v,f,**patches,selected=selected)
    actual=ray_state(v,f,build,eps);write_json(out/'events.json',events)
    record={'status':'complete','method':variant,'wall_seconds':time.monotonic()-start,'faces':len(f),'primitive_count':len(selected),'candidate_count':len(cost),
            'accepted_columns':sum(e['accepted_column_count'] for e in events),'integer_objective':integer['objective'],'integer_gap':integer['gap'],
            'unexplained_build_rays':int((~actual['hit'][context['owned']]).sum()),'build_early':int(actual['early'].sum()),
            'all_build_feasible':bool(actual['hit'][context['owned']].all() and not actual['early'].any()),'demand_network':str(params.get('dcs_checkpoints')) if learned else None}
    write_json(out/'reconstruction.json',record);return v,f,record

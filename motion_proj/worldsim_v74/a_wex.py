"""WEX：共享线性域、冲突相关联合责任配置；附同表示固定/贪心/MILP/软域控制。"""
import itertools, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, Bounds, LinearConstraint
import osqp
import torch
from .common import carriers, all_intersections, clip_domain, ray_state, save_mesh, write_json


def incidence(vertices,faces,build,epsilon):
    hits=all_intersections(vertices,faces,build['origins_actor_m'],build['directions_actor'])
    n=len(vertices);m=len(hits['ray'])
    mat=sp.csr_matrix((hits['bary'].ravel(),(np.repeat(np.arange(m),3),faces[hits['face']].ravel())),shape=(m,n))
    target=build['observed_first_range_m'][hits['ray']]
    free_ids=np.flatnonzero(hits['t']<target-epsilon)
    owned=build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool)
    good_ids=np.flatnonzero((np.abs(hits['t']-target)<=epsilon)&owned[hits['ray']])
    groups={int(r):good_ids[hits['ray'][good_ids]==r] for r in np.unique(hits['ray'][good_ids])}
    missing=np.setdiff1d(np.flatnonzero(owned),np.array(list(groups),int))
    return mat,free_ids,groups,hits,missing


def smooth_metric(faces,n):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    lap=sp.csr_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),n))
    return sp.eye(n,format='csc')+.1*(lap.T@lap)


class DomainProblem:
    def __init__(self,mat,free,groups,prior,metric,margin=.05,qp_max_iter=10000):
        self.mat=mat;self.free=np.asarray(free);self.groups=groups;self.rays=sorted(groups)
        self.prior=np.asarray(prior);self.metric=metric;self.n=mat.shape[1]
        self.margin=margin;self.qp_max_iter=qp_max_iter;self.calls=0

    def solve(self,y):
        n=self.n;k=len(y);m=self.margin
        selected=self.mat[np.asarray(y,dtype=int)]
        # free 是硬约束；显式正见证松弛用于记录不相容责任，不能冒充可行。
        A=sp.vstack([sp.hstack([self.mat[self.free],sp.csr_matrix((len(self.free),k))]),
                     sp.hstack([selected,sp.eye(k)]),sp.eye(n+k)],format='csc')
        lower=np.r_[np.full(len(self.free),-np.inf),np.full(k,m),np.full(n,-1.),np.zeros(k)]
        upper=np.r_[np.full(len(self.free),-m),np.full(k,np.inf),np.ones(n),np.full(k,2.)]
        P=sp.block_diag([self.metric,sp.eye(k)*1e-6],format='csc')
        q=np.r_[-self.prior,np.full(k,1000.)]
        solver=osqp.OSQP();solver.setup(P=sp.triu(P,format='csc'),q=q,A=A,l=lower,u=upper,
            verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=self.qp_max_iter,polishing=False)
        result=solver.solve();self.calls+=1
        if result.x is None or not np.all(np.isfinite(result.x)) or np.max(np.abs(result.x[:n]))>2:
            return {'x':np.full(n,-1.),'score':(k+1,1e6,1e6),'status':result.info.status,'slack':np.full(k,2.),'primal_residual':float(result.info.prim_res),'y':list(y)}
        x=result.x[:n];slack=np.maximum(m-selected@x,0)
        free_violation=np.maximum(self.mat[self.free]@x+m,0)
        bad=int((slack>1e-4).sum())+int((free_violation>1e-4).sum())
        cost=float(.5*x@self.metric@x-self.prior@x)
        return {'x':x,'score':(bad,float(slack.sum()+free_violation.sum()),cost),'status':result.info.status,
                'slack':slack,'primal_residual':float(result.info.prim_res),'y':list(y)}

    def search(self,variant,max_outer=100,beam=64,seconds=300):
        started=time.monotonic();y=[int(self.groups[r][0]) for r in self.rays];state=self.solve(y)
        trace=[self.event(0,state,[],'initial')]
        snapshots=[state['x'].copy()]
        for step in range(max_outer if variant!='A1_fixed' else 0):
            if state['score'][0]==0 or time.monotonic()-started>=seconds:break
            bad=np.flatnonzero(state['slack']>1e-4)
            nodes=np.unique(self.mat[np.asarray(y)[bad]].indices) if len(bad) else np.array([],int)
            # 正责任经共享 free 行影响另一条射线；遗漏这一跳会误把联合冲突当单射线问题。
            if len(nodes) and len(self.free):
                connected_free=self.free[np.asarray(self.mat[self.free][:,nodes].getnnz(axis=1)>0).ravel()]
                nodes=np.union1d(nodes,self.mat[connected_free].indices)
            related=[]
            for i,r in enumerate(self.rays):
                if len(self.groups[r])>1 and (i in bad or len(np.intersect1d(self.mat[self.groups[r]].indices,nodes))): related.append(i)
            related=sorted(related,key=lambda i:(-state['slack'][i],i))[:8]
            proposals=[]
            # 单责任控制与 WEX 共用相同子问题；WEX 增加同时改变两到四个关联责任。
            sizes=[1] if variant=='A2_greedy' else [2,3,4,1]
            for size in sizes:
                for subset in itertools.combinations(related,size):
                    choices=[[int(j) for j in self.groups[self.rays[i]] if j!=y[i]][:3] for i in subset]
                    for values in itertools.product(*choices):
                        proposal=y.copy()
                        for i,j in zip(subset,values):proposal[i]=j
                        proposals.append((proposal,list(subset)))
                        if len(proposals)>=beam:break
                    if len(proposals)>=beam:break
                if len(proposals)>=beam:break
            best=state;changed=[]
            for proposal,subset in proposals:
                if time.monotonic()-started>=seconds:break
                candidate=self.solve(proposal)
                if candidate['score']<best['score']:best=candidate;changed=subset
            if best is state:
                trace.append(self.event(step+1,state,[],'no_improving_configuration'));break
            state=best;y=state['y'];snapshots.append(state['x'].copy())
            trace.append(self.event(step+1,state,changed,'joint_exchange' if len(changed)>1 else 'single_exchange'))
        return state,trace,snapshots

    def event(self,step,state,changed,reason):
        return {'step':step,'violated_constraints':state['score'][0],'witness_slack_sum':state['score'][1],
                'domain_cost':state['score'][2],'primal_residual':state['primal_residual'],'solver_status':state['status'],
                'responsibility_changes':[self.rays[i] for i in changed],'joint_exchange_size':len(changed),
                'responsibility':state['y'],'conflict_rays':[self.rays[i] for i in np.flatnonzero(state['slack']>1e-4)],
                'conflict_is_irreducible_certificate':False,'reason':reason,'qp_calls':self.calls}

    def mixed_integer(self,seconds):
        # HiGHS 的标准混合约束控制。先最大化满足的见证数，再固定责任解同一 QP。
        n=self.n;k=len(self.rays);ids=np.concatenate([self.groups[r] for r in self.rays]) if k else np.empty(0,int)
        ng=len(ids);rowgroup=np.concatenate([np.full(len(self.groups[r]),i) for i,r in enumerate(self.rays)]) if k else np.empty(0,int)
        selector=sp.csr_matrix((np.ones(ng),(rowgroup,np.arange(ng))),shape=(k,ng))
        # x[-1,1]，故 1+m 足够关闭责任蕴含，不用任意巨大 M。
        A=sp.vstack([
            sp.hstack([self.mat[self.free],sp.csr_matrix((len(self.free),ng+k))]),
            sp.hstack([-self.mat[ids],sp.eye(ng)*(1+self.margin),sp.csr_matrix((ng,k))]),
            sp.hstack([sp.csr_matrix((k,n)),-selector,-sp.eye(k)])],format='csc')
        upper=np.r_[np.full(len(self.free),-self.margin),np.ones(ng),np.full(k,-1.)]
        c=np.r_[np.zeros(n),np.full(ng,1e-7),np.ones(k)]
        res=milp(c,integrality=np.r_[np.zeros(n),np.ones(ng+k)],bounds=Bounds(np.r_[np.full(n,-1.),np.zeros(ng+k)],np.ones(n+ng+k)),
            constraints=LinearConstraint(A,np.full(len(upper),-np.inf),upper),options={'time_limit':seconds,'mip_rel_gap':0.})
        y=[]
        for i,r in enumerate(self.rays):
            take=np.flatnonzero(rowgroup==i)
            j=take[np.argmax(res.x[n+take])] if res.x is not None and len(take) else take[0]
            y.append(int(ids[j]))
        state=self.solve(y)
        trace=[{**self.event(0,state,[],'HiGHS_MILP_then_same_QP'),'milp_status':int(res.status),
                'milp_objective':float(res.fun) if res.fun is not None else None,
                'milp_gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None}]
        return state,trace,[state['x'].copy()]

    def soft_domain(self,iterations=400):
        # PU 思想的共享域适配控制：正观测占域、前段空域，未观测节点只用共同先验。
        # 无官方 MNA 模型/训练数据；不将此命名为官方复现。
        x=torch.nn.Parameter(torch.tensor(self.prior,dtype=torch.float32,device='cuda'))
        matrix=self.mat.tocoo();idx=torch.tensor(np.vstack([matrix.row,matrix.col]),device='cuda')
        M=torch.sparse_coo_tensor(idx,torch.tensor(matrix.data,dtype=torch.float32,device='cuda'),matrix.shape).coalesce()
        optim=torch.optim.Adam([x],lr=.03);events=[]
        for i in range(iterations):
            value=torch.sparse.mm(M,x[:,None]).flatten()
            free=torch.relu(value[self.free]+self.margin).square().mean() if len(self.free) else x.sum()*0
            pos=torch.stack([torch.relu(self.margin-value[ids].max()).square() for ids in self.groups.values()]).mean() if self.groups else x.sum()*0
            loss=free+pos+.001*(x-1).square().mean();optim.zero_grad();loss.backward();optim.step()
            with torch.no_grad():x.clamp_(-1,1)
            if i in [0,iterations//2,iterations-1]:events.append({'step':i,'loss':float(loss),'reason':'PU-inspired soft ray constraints'})
        xx=x.detach().cpu().numpy();y=[int(ids[np.argmax(self.mat[ids]@xx)]) for ids in self.groups.values()]
        state={'x':xx,'y':y,'score':(0,0,0),'status':'soft_only_no_feasibility_claim','primal_residual':0.,'slack':np.array([])}
        return state,events,[xx.copy()]


def reconstruct(build,params,variant,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    cfg=params['wex'];eps=params['epsilon_obs_m']
    v,f,info=carriers(build,cfg['carriers'],cfg['grid'],params['carrier_half_width_m'])
    save_mesh(out/'initial.npz',v,f)
    if not len(f):return v,f,{'status':'missing_build','wall_seconds':time.monotonic()-start}
    initial=ray_state(v,f,build,eps)
    matrix,free,groups,hits,missing=incidence(v,f,build,eps)
    construction=[]
    # 无候选时只移动已有承载片，绝不调用 DCS 生成器。
    for step in range(cfg['construction_rounds']):
        if not len(missing):break
        count=min(8,len(missing),len(info['centers']))
        target_ids=missing[np.linspace(0,len(missing)-1,count,dtype=int)]
        points=build['points_actor_m'][target_ids]
        # 顺序使用最近承载片并限制每轮重复搬移，位移后完整重建候选。
        used=[]
        for p,r in zip(points,target_ids):
            order=np.argsort(np.linalg.norm(info['centers']-p,axis=1))
            j=next((int(j) for j in order if int(j) not in used),None)
            if j is None:continue
            used.append(j);delta=p-info['centers'][j]
            if np.linalg.norm(delta)>2*params['carrier_half_width_m']:continue
            sl=slice(j*cfg['grid']**2,(j+1)*cfg['grid']**2)
            v[sl]+=delta;info['centers'][j]=p
            construction.append({'round':step,'ray':int(r),'carrier':j,'translation_m':delta.tolist()})
        matrix,free,groups,hits,missing=incidence(v,f,build,eps)
    save_mesh(out/'carriers.npz',v,f)
    sp.save_npz(out/'candidate_domain_rows.npz',matrix)
    np.savez_compressed(out/'candidate_geometry.npz',**hits,free_ids=free,missing_candidate_rays=missing)
    problem=DomainProblem(matrix,free,groups,np.ones(len(v)),smooth_metric(f,len(v)),cfg['domain_margin'],cfg['qp_max_iter'])
    remaining=max(1.,cfg['wall_seconds']-(time.monotonic()-start))
    if variant=='A3_milp':state,trace,states=problem.mixed_integer(remaining)
    elif variant=='A0_soft':state,trace,states=problem.soft_domain()
    else:state,trace,states=problem.search(variant,cfg['max_outer_iterations'],cfg['beam_configurations'],remaining)
    vv,ff=clip_domain(v,f,state['x'])
    actual=ray_state(vv,ff,build,eps)
    owned=build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool)
    for i,x in enumerate(states):
        sv,sf=clip_domain(v,f,x);save_mesh(out/f'event_{i:03d}.npz',sv,sf,domain_values=x)
    save_mesh(out/'final.npz',vv,ff,domain_values=state['x'])
    write_json(out/'events.json',{'construction':construction,'domain_events':trace})
    record={'status':'complete','method':variant,'wall_seconds':time.monotonic()-start,'candidate_count':len(hits['ray']),
            'G_empty':len(missing),'F_count':len(free),'owned_build_rays':int(owned.sum()),'qp_calls':problem.calls,
            'build_witness_retained':int(actual['hit'][owned].sum()),'build_free_violations':int(actual['early'].sum()),
            'all_build_feasible':bool(actual['hit'][owned].all() and not actual['early'].any()),
            'initial_build_witness':int(initial['hit'][owned].sum()),'initial_free_violations':int(initial['early'].sum()),
            'domain_area_cannot_certify_query':True,'faces':len(ff),'solver_status':state['status']}
    write_json(out/'reconstruction.json',record)
    return vv,ff,record

"""RIF：共享 P1 四面体场、完整射线区间约束、保场延拓的局部细分。"""
import itertools,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import osqp
import torch
from .common import local_frames,all_intersections,ray_state,save_mesh,write_json


def regular_tets(lower,upper,n=7):
    vertices=np.stack(np.meshgrid(*[np.linspace(a,b,n) for a,b in zip(lower,upper)],indexing='ij'),-1).reshape(-1,3)
    ids=np.arange(n**3).reshape(n,n,n);tets=[]
    for base in itertools.product(range(n-1),repeat=3):
        for perm in itertools.permutations(range(3)):
            p=np.array(base);tet=[ids[tuple(p)]]
            for axis in perm:p=p.copy();p[axis]+=1;tet.append(ids[tuple(p)])
            tets.append(tet)
    return vertices,np.asarray(tets,int)


class TetField:
    def __init__(self,v,t):
        self.v=np.asarray(v,dtype=np.float64);self.t=np.asarray(t,dtype=int)
        xyz=self.v[self.t]
        # lambda(x)=D*x+e，全局坐标浮点64；相邻单元共享节点值。
        inv=np.linalg.inv(np.concatenate([xyz,np.ones((*xyz.shape[:2],1))],axis=2))
        self.D=np.transpose(inv[:,:3,:],(0,2,1));self.e=inv[:,3,:]

    @torch.no_grad()
    def point_rows(self,points):
        points=np.asarray(points,dtype=np.float64).reshape(-1,3);cells=np.full(len(points),-1,int);weights=np.zeros((len(points),4))
        D=torch.tensor(self.D,device='cuda');e=torch.tensor(self.e,device='cuda')
        for start in range(0,len(points),256):
            p=torch.tensor(points[start:start+256],device='cuda')
            lam=torch.einsum('tij,pj->pti',D,p)+e[None]
            inside=lam.min(-1).values>=-1e-8
            valid=inside.any(1);which=inside.to(torch.int8).argmax(1)
            cc=which.cpu().numpy();cc[~valid.cpu().numpy()]=-1;cells[start:start+len(p)]=cc
            weights[start:start+len(p)]=lam[torch.arange(len(p),device='cuda'),which].cpu().numpy()
        good=cells>=0;rr=np.repeat(np.flatnonzero(good),4)
        M=sp.csr_matrix((weights[good].ravel(),(rr,self.t[cells[good]].ravel())),shape=(len(points),len(self.v)))
        return M,cells

    @torch.no_grad()
    def intervals(self,origins,directions,ends):
        """lambda≥0 的四个仿射不等式求每个单元的精确进入/离开时间。"""
        D=torch.tensor(self.D,device='cuda');e=torch.tensor(self.e,device='cuda')
        ray=[];cell=[];lo=[];hi=[];weights=[]
        for start in range(0,len(origins),128):
            o=torch.as_tensor(np.asarray(origins[start:start+128],dtype=np.float64),device='cuda')
            d=torch.as_tensor(np.asarray(directions[start:start+128],dtype=np.float64),device='cuda')
            lam0=torch.einsum('tij,pj->pti',D,o)+e[None]
            slope=torch.einsum('tij,pj->pti',D,d)
            nonzero=slope.abs()>1e-12
            bound=-lam0/torch.where(nonzero,slope,torch.ones_like(slope))
            lower=torch.where(slope>1e-12,bound,torch.full_like(bound,-torch.inf)).max(-1).values.clamp_min(0)
            upper=torch.where(slope<-1e-12,bound,torch.full_like(bound,torch.inf)).min(-1).values
            upper=torch.minimum(upper,torch.as_tensor(ends[start:start+len(o)],dtype=torch.float64,device='cuda')[:,None])
            good=(upper-lower>1e-10)&~((~nonzero)&(lam0<-1e-8)).any(-1)
            ri,ti=torch.where(good)
            if len(ri):
                a=lower[ri,ti];b=upper[ri,ti]
                wa=lam0[ri,ti]+a[:,None]*slope[ri,ti];wb=lam0[ri,ti]+b[:,None]*slope[ri,ti]
                ray.append((ri+start).cpu().numpy());cell.append(ti.cpu().numpy())
                lo.append(a.cpu().numpy());hi.append(b.cpu().numpy());weights.append(torch.stack([wa,wb],1).cpu().numpy())
        if not ray:return sp.csr_matrix((0,len(self.v))),{'ray':np.empty(0,int),'cell':np.empty(0,int),'lo':np.empty(0),'hi':np.empty(0)}
        ray=np.concatenate(ray);cell=np.concatenate(cell);ww=np.concatenate(weights).reshape(-1,4)
        row=np.repeat(np.arange(len(ww)),4);cols=np.repeat(self.t[cell],2,axis=0).ravel()
        F=sp.csr_matrix((ww.ravel(),(row,cols)),shape=(len(ww),len(self.v)))
        return F,{'ray':ray,'cell':cell,'lo':np.concatenate(lo),'hi':np.concatenate(hi)}

    def refine(self,chosen,c):
        # 单元内部重心星形分裂，外侧三角面完全保留，因此相邻未细分单元也保持一致。
        chosen=np.unique(chosen);nv=self.v[self.t[chosen]].mean(1);nc=c[self.t[chosen]].mean(1)
        old=np.ones(len(self.t),bool);old[chosen]=False;children=[]
        for index,tid in enumerate(chosen):
            tet=self.t[tid];center=len(self.v)+index
            for omit in range(4):children.append([center,*np.delete(tet,omit)])
        vertices=np.vstack([self.v,nv]);tets=np.vstack([self.t[old],np.asarray(children,int)])
        return TetField(vertices,tets),np.r_[c,nc],{'parents':self.t[chosen],'parent_ids':chosen,'children':np.asarray(children,int),'new_node_parent_coefficients':np.full((len(chosen),4),.25)}

    def extract(self,c):
        # 同一仿射零平面截四面体；共享边缓存仅用于拓扑索引，不改变阈值或几何。
        verts=[];faces=[];edge_ids={};zero_tets=0
        for tet in self.t:
            values=c[tet]
            if np.max(np.abs(values))<1e-10:zero_tets+=1;continue
            if values.min()>0 or values.max()<0:continue
            polygon=[]
            for aa,bb in itertools.combinations(range(4),2):
                a=int(tet[aa]);b=int(tet[bb]);va=c[a];vb=c[b]
                if va==0:key=(a,a);position=self.v[a]
                elif vb==0:key=(b,b);position=self.v[b]
                elif (va>0)==(vb>0):continue
                else:
                    key=tuple(sorted([a,b]));ratio=va/(va-vb);position=(1-ratio)*self.v[a]+ratio*self.v[b]
                if key not in edge_ids:edge_ids[key]=len(verts);verts.append(position)
                index=edge_ids[key]
                if index not in polygon:polygon.append(index)
            if len(polygon)<3:continue
            xyz=np.asarray([verts[i] for i in polygon]);center=xyz.mean(0)
            normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);norm=np.linalg.norm(normal)
            if norm<1e-14:continue
            normal/=norm;u=xyz[0]-center;u/=max(np.linalg.norm(u),1e-12);w=np.cross(normal,u)
            order=np.argsort(np.arctan2((xyz-center)@w,(xyz-center)@u));polygon=[polygon[i] for i in order]
            for k in range(1,len(polygon)-1):faces.append([polygon[0],polygon[k],polygon[k+1]])
        return np.asarray(verts).reshape(-1,3),np.asarray(faces,int).reshape(-1,3),zero_tets


def prior_values(field,build):
    p=build['support_points_actor_m'];owned=build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool)
    origins=build['origins_actor_m'][owned]
    frame,_=local_frames(p,p,origins if len(origins)==len(p) else None)
    distance,ids=cKDTree(p).query(field.v,k=min(4,len(p)))
    distance=np.asarray(distance).reshape(len(field.v),-1);ids=np.asarray(ids).reshape(len(field.v),-1)
    signed=np.einsum('nki,nki->nk',field.v[:,None]-p[ids],frame[ids,:,2])
    weights=1/np.maximum(distance,.01)**2
    return (signed*weights).sum(1)/weights.sum(1)


def field_solve(field,c0,H,F,mu,soft=False,seconds=300):
    n=len(field.v);nr=H.shape[0];nf=F.shape[0]
    edges=np.unique(np.sort(field.t[:,list(itertools.combinations(range(4),2))].reshape(-1,2),axis=1),axis=0)
    L=sp.csr_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),n))
    P=sp.eye(n)+.02*(L.T@L)
    # 同一QP中 e=-Hc, s=max(mu-Fc,0) 可解析消元；仅改变数值求解变量，不改变目标/观测。
    # 避免为几十万区间端点各建一个额外变量；原始残差仍完整保存。
    H=H.tocsr();F=F.tocsr();P=P.tocsr()
    diagonal=np.asarray(P.diagonal()+1e4*H.power(2).sum(0).A1+1e4*F.power(2).sum(0).A1)
    scale=np.sqrt(np.maximum(diagonal,1.));begin=time.monotonic();latest=[c0.copy()]
    def objective(y):
        c=y/scale;hc=H@c;slack=np.maximum(mu-F@c,0);pc=P@c
        value=.5*c@pc-c0@c+.5e4*(hc@hc+slack@slack)
        grad=pc-c0+1e4*(H.T@hc-F.T@slack)
        latest[0]=c
        return float(value),np.asarray(grad/scale)
    def callback(y):
        if time.monotonic()-begin>seconds:raise TimeoutError('fixed FIT solver wall budget')
    try:
        res=minimize(objective,c0*scale,jac=True,method='L-BFGS-B',callback=callback,
            options={'maxiter':10000,'maxfun':20000,'maxcor':30,'ftol':1e-12,'gtol':1e-7,'maxls':40})
        c=res.x/scale;status=str(res.message);iterations=res.nit
    except TimeoutError:
        c=latest[0];status='time_limit';iterations=None
    value,grad=objective(c*scale)
    return c,{'solver_status':status,'solver_iterations':iterations,'primal_residual':0.,
              'eliminated_slack_variables':nr+nf,'solver_stationarity_scaled_inf':float(np.max(np.abs(grad))),
              'objective':value,'H_residual':np.asarray(H@c),'F_slack':np.maximum(mu-F@c,0)}


def reconstruct(build,params,variant,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();cfg=params['rif'];eps=params['epsilon_obs_m']
    p=build['support_points_actor_m']
    if not len(p):return np.empty((0,3)),np.empty((0,3),int),{'status':'missing_build','wall_seconds':0.}
    half=np.asarray(build['size_lwh_m'])/2+.2
    lower=np.minimum(-half,p.min(0)-.1);upper=np.maximum(half,p.max(0)+.1)
    field=TetField(*regular_tets(lower,upper,cfg['initial_grid']))
    c0=prior_values(field,build);c=c0.copy();v,f,_=field.extract(c);save_mesh(out/'initial.npz',v,f)
    owned=np.flatnonzero(build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool))
    o=build['origins_actor_m'];d=build['directions_actor'];observed=build['observed_first_range_m'];tau=observed[owned].astype(float).copy()
    events=[];rounds=cfg['refinement_rounds'] if variant in ['RIF','B2_regular'] else 0
    for step in range(rounds+1):
        H,rootcell=field.point_rows(o[owned]+tau[:,None]*d[owned])
        F,interval=field.intervals(o,d,np.maximum(observed-eps,0))
        if variant=='B0_sampled':
            # 每束固定 8 个 ROI 内前段点，与 exact 控制相同输入，不额外读取数据。
            ids=np.arange(len(interval['ray']))
            sample_rows=[]
            for q in np.linspace(0,1,8):
                # 以射线在整个 ROI 的进入/退出时间生成采样，而非每个 tet 采样。
                entry=np.full(len(o),np.inf);exit=np.full(len(o),-np.inf)
                np.minimum.at(entry,interval['ray'],interval['lo']);np.maximum.at(exit,interval['ray'],interval['hi'])
                take=np.flatnonzero(np.isfinite(entry));tt=entry[take]+q*(exit[take]-entry[take])
                rows,_=field.point_rows(o[take]+tt[:,None]*d[take]);sample_rows.append(rows)
            sampled=sp.vstack(sample_rows,format='csr');c,diag=field_solve(field,c0,H,sampled,cfg['free_margin'],soft=True,seconds=max(1,cfg['wall_seconds']-(time.monotonic()-start)))
            diag['F_slack']=np.maximum(cfg['free_margin']-F@c,0)
        else:c,diag=field_solve(field,c0,H,F,cfg['free_margin'],seconds=max(1,cfg['wall_seconds']-(time.monotonic()-start)))
        v,f,zero=field.extract(c);actual=ray_state(v,f,build,eps)
        save_mesh(out/f'event_{step:02d}.npz',v,f,field_vertices=field.v,tetrahedra=field.t,field_values=c,tau=tau)
        np.savez_compressed(out/f'constraints_{step:02d}.npz',root_cell=rootcell,root_residual=diag['H_residual'],free_slack=diag['F_slack'],**interval)
        event={k:value for k,value in diag.items() if np.isscalar(value)}
        event.update({'step':step,'nodes':len(field.v),'cells':len(field.t),'faces':len(f),'zero_tetra_count':zero,
            'ray_cell_crossings':len(interval['ray']),'exact_breakpoint_count':F.shape[0],
            'H_max_abs':float(np.max(np.abs(diag['H_residual']))) if len(owned) else None,
            'F_max_slack':float(np.max(diag['F_slack'])) if F.shape[0] else 0.,
            'actual_build_early':int(actual['early'].sum()),'actual_build_hit':int(actual['hit'][owned].sum()),
            'root_outside_domain':int((rootcell<0).sum()),'wall_seconds':time.monotonic()-start})
        events.append(event)
        if step==rounds or time.monotonic()-start>cfg['wall_seconds'] or len(field.v)>=cfg['max_nodes']:break
        count=min(128,len(field.t),cfg['max_nodes']-len(field.v))
        score=np.zeros(len(field.t))
        if variant=='RIF':
            valid=rootcell>=0
            np.add.at(score,rootcell[valid],np.abs(diag['H_residual'][valid])+eps*(~actual['hit'][owned][valid]))
            if len(interval['cell']):np.add.at(score,interval['cell'],diag['F_slack'].reshape(-1,2).max(1))
        else:
            # 普通场插值/几何误差细分，不读取射线残差；最终新增节点预算相同。
            centers=field.v[field.t].mean(1)
            _,nn=cKDTree(p).query(centers);score=np.abs(c[field.t].mean(1)-c0[field.t].mean(1))+.01/np.maximum(np.linalg.norm(centers-p[nn],axis=1),.01)
        chosen=np.argsort(-score,kind='stable')[:count]
        oldfield=field;oldc=c.copy();field,c,mapping=field.refine(chosen,c)
        # 保留同一个已满足场，随后只用新增自由度重解；新先验取同 BUILD 几何。
        check_rows,_=field.point_rows(oldfield.v[oldfield.t[chosen]].mean(1))
        error=float(np.max(np.abs(check_rows@c-oldc[oldfield.t[chosen]].mean(1)))) if count else 0.
        mapping['prolongation_error_m']=np.array(error);np.savez_compressed(out/f'refine_{step:02d}.npz',**mapping)
        event.update({'cells_refined':count,'refinement_reason':'ray_root_and_interval_conflict' if variant=='RIF' else 'ordinary_field_geometry_error','prolongation_root_error':error})
        c0=prior_values(field,build)
        if variant=='RIF':
            hits=all_intersections(v,f,o[owned],d[owned]);new=tau.copy()
            for i in range(len(owned)):
                take=np.flatnonzero((hits['ray']==i)&(np.abs(hits['t']-observed[owned[i]])<=eps))
                if len(take):new[i]=hits['t'][take[np.argmin(np.abs(hits['t'][take]-observed[owned[i]]))]]
            event['root_shift_max_m']=float(np.max(np.abs(new-tau))) if len(tau) else 0.;tau=new
    save_mesh(out/'final.npz',v,f,field_vertices=field.v,tetrahedra=field.t,field_values=c)
    write_json(out/'events.json',events)
    result={'status':'complete' if len(f)<=params['face_budget'] else 'face_budget_exceeded','method':variant,'faces':len(f),'nodes':len(field.v),
            'wall_seconds':time.monotonic()-start,'refinement_events':len(events)-1,'all_build_feasible':bool(actual['hit'][owned].all() and not actual['early'].any()),
            'zero_tetra_count':zero,'unexplained_build_rays':int((~actual['hit'][owned]).sum()),'build_early':int(actual['early'].sum()),
            'guarantee_claimed':False,'reason_no_guarantee':'explicit observation slack and numerical solver; hard mesh residuals reported'}
    write_json(out/'reconstruction.json',result);return v,f,result

"""CPU 高精度真实三角形查询；保留完整 CSR 链，无固定 K 截断。"""
import numpy as np
def triangle_intersections(vertices,faces,origins,directions,dtype=np.float64,chunk=128):
    """主轴置换/剪切边函数（PBRT 式），双面，闭边，严格 t>0；共面无孤立事件。"""
    v=np.asarray(vertices,dtype=dtype);o=np.asarray(origins,dtype=dtype);d=np.asarray(directions,dtype=dtype)
    rr=[];ff=[];tt=[];bb=[]
    if not len(faces):return {'ray':np.array([],int),'face':np.array([],int),'t':np.array([]),'bary':np.empty((0,3))}
    for lo in range(0,len(o),chunk):
        for axis in range(3):
            ids=np.flatnonzero(np.argmax(np.abs(d[lo:lo+chunk]),axis=1)==axis)+lo
            if not len(ids):continue
            x=(axis+1)%3;y=(x+1)%3
            tri=v[faces][None]-o[ids,None,None,:]
            sx=d[ids,x]/d[ids,axis];sy=d[ids,y]/d[ids,axis]
            px=tri[:,:,:,x]-sx[:,None,None]*tri[:,:,:,axis];py=tri[:,:,:,y]-sy[:,None,None]*tri[:,:,:,axis]
            e0=px[:,:,1]*py[:,:,2]-py[:,:,1]*px[:,:,2]
            e1=px[:,:,2]*py[:,:,0]-py[:,:,2]*px[:,:,0]
            e2=px[:,:,0]*py[:,:,1]-py[:,:,0]*px[:,:,1]
            det=e0+e1+e2;valid=((e0>=0)&(e1>=0)&(e2>=0))|((e0<=0)&(e1<=0)&(e2<=0))
            valid&=det!=0;safe=np.where(det!=0,det,1)
            dep=(e0*tri[:,:,0,axis]+e1*tri[:,:,1,axis]+e2*tri[:,:,2,axis])/safe/d[ids,axis,None]
            ri,fi=np.where(valid&(dep>0)&np.isfinite(dep))
            rr.extend(ids[ri]);ff.extend(fi);tt.extend(dep[ri,fi]);bb.extend(np.stack([e0[ri,fi],e1[ri,fi],e2[ri,fi]],1)/safe[ri,fi,None])
    return {'ray':np.asarray(rr,int),'face':np.asarray(ff,int),'t':np.asarray(tt,float),'bary':np.asarray(bb).reshape(-1,3)}
def trace_mesh(v,f,o,d,patch_ids=None,dtype=np.float64):
    hits=triangle_intersections(v,f,o,d,dtype=dtype);n=len(o)
    patch=(hits['face']//8 if patch_ids is None else np.asarray(patch_ids)[hits['face']]);order=np.lexsort((hits['face'],patch,hits['t'],hits['ray']))
    kept=[];last={}
    for i in order:
        key=(int(hits['ray'][i]),int(patch[i]));t=hits['t'][i]
        if key in last and abs(t-last[key])<=1e-10*max(1,abs(t)):continue
        last[key]=t;kept.append(i)
    k=np.asarray(kept,int);counts=np.bincount(hits['ray'][k],minlength=n)
    off=np.r_[0,counts.cumsum()];first=np.full(n,np.inf)
    active=np.flatnonzero(counts);first[active]=hits['t'][k[off[active]]]
    return {'offsets':off,'ray':hits['ray'][k],'patch':patch[k],'face':hits['face'][k],
            't':hits['t'][k],'bary':hits['bary'][k],'first':first}
def trace(s,obs,dtype=np.float64):
    v,f=s.mesh();return trace_mesh(v,f,obs['origins_actor_m'],obs['directions_actor'],dtype=dtype)
def boundary_neighborhood(s,obs):
    """完整射线×支撑平面关系；域外只作潜在进入消息，绝不作真实命中。"""
    o=obs['origins_actor_m'];d=obs['directions_actor'];normal=s.rotation[:,:,2]
    den=d@normal.T;safe=np.where(np.abs(den)>1e-12,den,1.)
    t=np.einsum('rkc,kc->rk',s.center[None]-o[:,None],normal)/safe
    rel=o[:,None]+t[:,:,None]*d[:,None]-s.center[None]
    uv=np.einsum('rkc,kcj->rkj',rel,s.rotation[:,:,:2]);angle=np.mod(np.arctan2(uv[:,:,1],uv[:,:,0]),2*np.pi)
    sector=np.floor(angle/(np.pi/4)).astype(int)%8;phi=angle-sector*np.pi/4
    r0=s.radius[np.arange(len(s.center))[None],sector];r1=s.radius[np.arange(len(s.center))[None],(sector+1)%8]
    radial=r0*r1*np.sin(np.pi/4)/(r1*np.sin(np.pi/4-phi)+r0*np.sin(phi))
    return {'plane_t':t,'uv':uv,'sector':sector,'signed_boundary_distance':np.linalg.norm(uv,axis=2)-radial,
            'valid_plane':(np.abs(den)>1e-12)&(t>0)}
def point_distance(v,f,points,chunk=64):
    if not len(points):return np.empty(0)
    if not len(f):return np.full(len(points),np.inf)
    tri=v[f];a=tri[:,0];ab=tri[:,1]-a;ac=tri[:,2]-a;normal=np.cross(ab,ac);nn=(normal**2).sum(1)
    out=[];aa=(ab*ab).sum(1);cc=(ac*ac).sum(1);bc=(ab*ac).sum(1);det=aa*cc-bc**2
    for lo in range(0,len(points),chunk):
        q=points[lo:lo+chunk,None]-a;qn=(q*normal).sum(2)/np.maximum(nn,1e-30)
        proj=q-qn[:,:,None]*normal;u=(proj*ab).sum(2);w=(proj*ac).sum(2)
        x=(cc*u-bc*w)/np.maximum(det,1e-30);y=(aa*w-bc*u)/np.maximum(det,1e-30)
        best=np.where((x>=0)&(y>=0)&(x+y<=1)&(nn>1e-24),qn**2*nn,np.inf)
        for j in range(3):
            edge=tri[:,(j+1)%3]-tri[:,j];delta=points[lo:lo+chunk,None]-tri[None,:,j]
            t=np.clip((delta*edge).sum(2)/np.maximum((edge**2).sum(1),1e-30),0,1)
            best=np.minimum(best,((delta-t[:,:,None]*edge)**2).sum(2))
        out.extend(np.sqrt(best.min(1)))
    return np.asarray(out)
def metrics(s,obs,epsilon=.2):
    tr=trace(s,obs);y=obs['observed_first_range_m'];positive=obs['positive_actor']&~obs['ambiguous_owner'];q=tr['first'];finite=np.isfinite(q)
    early=finite&(q<y-epsilon);hit=finite&(np.abs(q-y)<=epsilon);anygood=np.zeros(len(y),bool)
    if len(tr['t']):anygood[tr['ray'][np.abs(tr['t']-y[tr['ray']])<=epsilon]]=True
    v,f=s.mesh();p=obs['points_actor_m'][positive];dist=point_distance(v,f,p)
    avg=lambda x:float(np.mean(x[positive])) if positive.any() else None
    area=float(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).sum()/2) if len(f) else 0.
    return {'hit':avg(hit),'early':avg(early),'miss':avg(~finite),'late':avg(finite&~early&~hit),
        'any_correct':avg(anygood),'early_with_later_correct':avg(early&anygood),
        'free_intrusion_rate':float(early.mean()) if len(y) else None,
        'free_m':float(np.where(early,y-q-epsilon,0).mean()) if len(y) else None,
        'recall_02':float(np.mean(dist<=epsilon)) if len(p) else None,
        'distance_m':float(np.mean(np.minimum(dist,5.))) if len(p) else None,'distance_cap_m':5.,
        'faces':len(f),'area_m2':area,'positive_rays':int(positive.sum()),'rays':len(y),'empty':not len(f)}
def objective(s,obs):
    q=trace(s,obs)['first'];y=obs['observed_first_range_m'];p=obs['positive_actor']&~obs['ambiguous_owner'];finite=np.isfinite(q)
    residual=np.where(finite,np.minimum(np.abs(q-y),2.)/2.,1.)
    early=np.where(finite,np.maximum(y-q-.2,0),0)
    # 所有控制共享；这是训练/搜索目标，不替代任务联合指标。
    return float((residual[p].mean() if p.any() else 0)+early.mean())

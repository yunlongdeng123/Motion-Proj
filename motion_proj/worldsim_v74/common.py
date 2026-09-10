"""公共几何与记录；方法求解器不接收 QUERY 真值。"""
import json, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from motion_proj.worldsim_v73.surface_readout import closest_surface_points, first_return_metrics, direct_free_space_loss


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_mesh(path, vertices, faces, **extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,vertices_actor_m=np.asarray(vertices),faces=np.asarray(faces,dtype=np.int64),**extra)


def mesh_area(v,f):
    if not len(f): return 0.
    tri=np.asarray(v)[f]
    return float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2)


@torch.no_grad()
def all_intersections(v,f,o,d,device='cuda',ray_chunk=256,face_chunk=2048):
    """字面双面 Möller–Trumbore；保留所有面交点，不能按每片一个截断。"""
    vv=torch.as_tensor(v,dtype=torch.float32,device=device)
    ff=torch.as_tensor(f,dtype=torch.long,device=device)
    oo=torch.as_tensor(o,dtype=torch.float32,device=device)
    dd=torch.as_tensor(d,dtype=torch.float32,device=device)
    rows=[];cols=[];depth=[];bary=[]
    for rs in range(0,len(oo),ray_chunk):
        ro=oo[rs:rs+ray_chunk];rd=dd[rs:rs+ray_chunk]
        for fs in range(0,len(ff),face_chunk):
            tri=vv[ff[fs:fs+face_chunk]]
            e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
            pv=torch.cross(rd[:,None],e2[None],dim=-1)
            det=(e1[None]*pv).sum(-1)
            inv=torch.where(det.abs()>1e-8,det,torch.ones_like(det)).reciprocal()
            tv=ro[:,None]-tri[None,:,0]
            u=(tv*pv).sum(-1)*inv
            qv=torch.cross(tv,e1[None],dim=-1)
            w=(rd[:,None]*qv).sum(-1)*inv
            t=(e2[None]*qv).sum(-1)*inv
            good=(det.abs()>1e-8)&(u>=0)&(w>=0)&(u+w<=1)&(t>0)
            ri,fi=torch.where(good)
            if len(ri):
                rows.append((ri+rs).cpu().numpy());cols.append((fi+fs).cpu().numpy())
                depth.append(t[ri,fi].cpu().numpy());bary.append(torch.stack([1-u[ri,fi]-w[ri,fi],u[ri,fi],w[ri,fi]],1).cpu().numpy())
    if not rows: return {'ray':np.empty(0,int),'face':np.empty(0,int),'t':np.empty(0),'bary':np.empty((0,3))}
    return {'ray':np.concatenate(rows),'face':np.concatenate(cols),'t':np.concatenate(depth),'bary':np.concatenate(bary)}


def ray_state(v,f,b,epsilon=.2):
    hit=all_intersections(v,f,b['origins_actor_m'],b['directions_actor'])
    n=len(b['observed_first_range_m']); observed=np.asarray(b['observed_first_range_m'])
    first=np.full(n,np.inf);owner=np.full(n,-1,int); any_correct=np.zeros(n,bool)
    if len(hit['ray']):
        order=np.argsort(hit['t'],kind='stable')
        sorted_r=hit['ray'][order];_,first_ids=np.unique(sorted_r,return_index=True)
        chosen=order[first_ids];first[hit['ray'][chosen]]=hit['t'][chosen];owner[hit['ray'][chosen]]=hit['face'][chosen]
        correct=np.abs(hit['t']-observed[hit['ray']])<=epsilon
        any_correct[hit['ray'][correct]]=True
    early=np.isfinite(first)&(first<observed-epsilon)
    return {'first_range_m':first,'first_face':owner,'any_correct':any_correct,'early_with_later_correct':early&any_correct,
            'early':early,'hit':np.isfinite(first)&(np.abs(first-observed)<=epsilon),'miss':~np.isfinite(first)}


@torch.no_grad()
def evaluate(v,f,b,path=None):
    s=ray_state(v,f,b,.2)
    observed=torch.as_tensor(b['observed_first_range_m'],dtype=torch.float32)
    depth=torch.as_tensor(s['first_range_m'],dtype=torch.float32)
    positive=np.asarray(b['positive_actor'],dtype=bool)
    metrics=first_return_metrics(depth[positive],observed[positive])
    metrics.update({'all_near_box_rays':len(observed),'free_intrusion_rate':float(s['early'].mean()) if len(observed) else None,
                    'mean_free_intrusion_m':float(direct_free_space_loss(depth,observed)),
                    'any_correct_intersection':float(s['any_correct'][positive].mean()) if positive.any() else None,
                    'early_with_later_correct_support':float(s['early_with_later_correct'][positive].mean()) if positive.any() else None})
    p=torch.as_tensor(b['points_actor_m'][positive],dtype=torch.float32,device='cuda')
    if len(p) and len(f):
        q=closest_surface_points(torch.as_tensor(v,dtype=torch.float32,device='cuda'),torch.as_tensor(f,dtype=torch.long,device='cuda'),p,point_chunk=128,face_chunk=1024)
        dist=(q-p).norm(dim=1)
        metrics.update({'positive_surface_mean_m':float(dist.mean()),'positive_surface_recall_02':float((dist<=.2).float().mean())})
    else:
        metrics.update({'positive_surface_mean_m':None,'positive_surface_recall_02':0. if len(p) else None})
    metrics.update({'positive_points':len(p),'faces':len(f),'vertices':len(v),'area_m2':mesh_area(v,f)})
    if path: np.savez_compressed(path,**s,observed_first_range_m=np.asarray(observed),positive_actor=positive)
    return metrics


def farthest_ids(points,n):
    if not len(points): return np.empty(0,int)
    first=int(np.argmin(np.linalg.norm(points-points.mean(0),axis=1)))
    ids=[first];dist=np.linalg.norm(points-points[first],axis=1)**2
    for _ in range(1,min(n,len(points))):
        j=int(np.argmax(dist));ids.append(j)
        dist=np.minimum(dist,np.linalg.norm(points-points[j],axis=1)**2)
    return np.asarray(ids)


def local_frames(points,centers,sensor_origins=None):
    nn=cKDTree(points).query(centers,k=min(16,len(points)))[1]
    nn=np.asarray(nn).reshape(len(centers),-1);local=points[nn]
    delta=local-local.mean(1,keepdims=True)
    val,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta)/local.shape[1])
    normal=vec[:,:,0]
    invalid=(val[:,1]<1e-10) if len(points)>1 else np.ones(len(centers),bool)
    if sensor_origins is not None:
        sensor=np.asarray(sensor_origins)[nn[:,0]];toward=sensor-centers
        toward/=np.maximum(np.linalg.norm(toward,axis=1,keepdims=True),1e-9)
        normal[invalid]=toward[invalid]
        normal*=np.where((normal*toward).sum(1)<0,-1.,1.)[:,None]
    ref=np.tile([0.,0.,1.],(len(normal),1));ref[np.abs(normal[:,2])>.9]=[0.,1.,0.]
    tangent=np.cross(normal,ref);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    other=np.cross(normal,tangent)
    return np.stack([tangent,other,normal],axis=2),val


def carriers(build,count,grid,half_width):
    points=build['support_points_actor_m']
    if not len(points): return np.empty((0,3)),np.empty((0,3),int),{}
    ids=farthest_ids(points,count);centers=points[ids]
    owned=build['positive_actor'].astype(bool)&~build['ambiguous_owner'].astype(bool)
    origins=build['origins_actor_m'][owned]
    frames,_=local_frames(points,centers,origins if len(origins)==len(points) else None)
    uv=np.stack(np.meshgrid(np.linspace(-1,1,grid),np.linspace(-1,1,grid),indexing='ij'),-1).reshape(-1,2)
    verts=(centers[:,None]+np.einsum('nck,pk->npc',frames[:,:,:2],uv*half_width)).reshape(-1,3)
    base=[]
    for i in range(grid-1):
        for j in range(grid-1):
            a=i*grid+j;base.extend([[a,a+grid,a+grid+1],[a,a+grid+1,a+1]])
    faces=(np.asarray(base)[None]+np.arange(len(centers))[:,None,None]*grid*grid).reshape(-1,3)
    return verts,faces,{'centers':centers,'frames':frames,'point_ids':ids,'half_width_m':half_width}


def clip_domain(vertices,faces,x):
    """分片线性域的零线裁剪；跨共享边复用一个真实交点。"""
    out=list(np.asarray(vertices));new=[];edge_vertex={}
    for face in faces:
        poly=[]
        for a,b in zip(face,np.roll(face,-1)):
            if x[a]>=0: poly.append(int(a))
            if (x[a]>=0)!=(x[b]>=0):
                key=tuple(sorted((int(a),int(b))))
                if key not in edge_vertex:
                    t=float(x[a]/(x[a]-x[b]));edge_vertex[key]=len(out)
                    out.append((1-t)*vertices[a]+t*vertices[b])
                poly.append(edge_vertex[key])
        for k in range(1,len(poly)-1):new.append([poly[0],poly[k],poly[k+1]])
    f=np.asarray(new,dtype=np.int64).reshape(-1,3);v=np.asarray(out).reshape(-1,3)
    if len(f):
        tri=v[f];take=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>1e-12
        f=f[take];used,inv=np.unique(f,return_inverse=True);v=v[used];f=inv.reshape(-1,3)
    return v,f

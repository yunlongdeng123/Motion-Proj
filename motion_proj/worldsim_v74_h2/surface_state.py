"""真实八三角扇表面，位置、旋转、八个正半径共享跨射线。"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
@dataclass
class Surface:
    center: np.ndarray
    rotation: np.ndarray
    radius: np.ndarray
    def copy(self):return Surface(self.center.copy(),self.rotation.copy(),self.radius.copy())
    def mesh(self):
        if not len(self.center):return np.empty((0,3)),np.empty((0,3),int)
        angle=np.arange(8)*np.pi/4;uv=np.c_[np.cos(angle),np.sin(angle),np.zeros(8)]
        ring=self.center[:,None]+np.einsum('nij,nkj->nki',self.rotation,self.radius[:,:,None]*uv)
        v=np.concatenate([self.center[:,None],ring],axis=1).reshape(-1,3)
        base=np.array([[0,i+1,1+(i+1)%8] for i in range(8)])
        f=(base[None]+9*np.arange(len(self.center))[:,None,None]).reshape(-1,3)
        return v,f
    def save(self,path):
        v,f=self.mesh();np.savez_compressed(path,center=self.center,rotation=self.rotation,radius=self.radius,vertices_actor_m=v,faces=f)
    def transform(self,rotation,translation):
        return Surface(self.center@rotation.T+translation,rotation[None]@self.rotation,self.radius.copy())
def concatenate(*states):
    return Surface(*(np.concatenate([getattr(s,k) for s in states]) for k in ['center','rotation','radius']))
def empty():return Surface(np.empty((0,3)),np.empty((0,3,3)),np.empty((0,8)))
def square(center,half,rotation=None):
    # 方形的中边点/角点恰好位于固定八方向；八个三角形与原方形几何相同。
    r=np.array([1,2**.5]*4)*half
    return Surface(np.asarray(center,float).reshape(1,3),np.eye(3)[None] if rotation is None else np.asarray(rotation).reshape(1,3,3),r[None])
def initialize(obs,count=64,width=.15):
    positive=obs['positive_actor']&~obs['ambiguous_owner'];points=obs['points_actor_m'][positive]
    if not len(points):return empty()
    ids=[int(np.argmin(np.linalg.norm(points-points.mean(0),axis=1)))];dist=np.full(len(points),np.inf)
    for _ in range(min(count,len(points))-1):
        dist=np.minimum(dist,((points-points[ids[-1]])**2).sum(1));ids.append(int(dist.argmax()))
    c=points[ids];nn=np.asarray(cKDTree(points).query(c,k=min(16,len(points)))[1]).reshape(len(c),-1)
    local=points[nn];delta=local-local.mean(1,keepdims=True)
    val,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta)/local.shape[1]);normal=vec[:,:,0]
    toward=obs['origins_actor_m'][positive][np.asarray(ids)]-c;toward/=np.maximum(np.linalg.norm(toward,axis=1,keepdims=True),1e-12)
    normal[val[:,1]<1e-10]=toward[val[:,1]<1e-10];normal*=np.where((normal*toward).sum(1)<0,-1,1)[:,None]
    ref=np.tile([0.,0.,1.],(len(c),1));ref[np.abs(normal[:,2])>.9]=[0,1,0]
    u=np.cross(normal,ref);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12);v=np.cross(normal,u)
    return Surface(c,np.stack([u,v,normal],axis=2),np.full((len(c),8),width))
def apply(surface,event,max_patches=512):
    s=surface.copy()
    for edit in event:
        k=edit.get('patch',0);kind=edit['kind']
        if kind=='move':s.center[k]+=np.asarray(edit['delta'])
        elif kind=='rotate':s.rotation[k]=Rotation.from_rotvec(edit['rotvec']).as_matrix()@s.rotation[k]
        elif kind=='radius':s.radius[k,edit['sector']]*=edit['factor']
        elif kind=='scale':s.radius[k]*=edit['factor']
        elif kind=='birth' and len(s.center)<max_patches:
            s=concatenate(s,Surface(np.asarray(edit['center'])[None],np.asarray(edit['rotation'])[None],np.asarray(edit['radius'])[None]))
        elif kind=='split' and len(s.center)<max_patches:
            axis=s.rotation[k,:,edit['axis']];shift=.35*np.mean(s.radius[k])*axis
            twin=Surface((s.center[k]+shift)[None],s.rotation[k:k+1].copy(),s.radius[k:k+1]*.75)
            s.center[k]-=shift;s.radius[k]*=.75;s=concatenate(s,twin)
    s.radius=np.clip(s.radius,1e-4,10.)
    return s

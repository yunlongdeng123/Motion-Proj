"""实际世界XY三角面反查及逐face UV采样；空间桶只加速，不定义表面几何。"""
from pathlib import Path
import json
import numpy as np


class TriangleLocatorXY:
    def __init__(self,world_vertices,faces,*,face_priority=None):
        self.vertices=np.asarray(world_vertices,np.float64);self.faces=np.asarray(faces,np.int64)
        tri=self.vertices[self.faces,:2]
        self.origin=tri[:,0];self.edge=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)
        det=np.linalg.det(self.edge)
        if not np.isfinite(tri).all() or np.any(np.abs(det)<1e-16):raise ValueError('invalid or degenerate world XY triangle')
        self.inverse=np.linalg.inv(self.edge)
        self.priority=np.zeros(len(self.faces)) if face_priority is None else np.asarray(face_priority)
        # AABB覆盖完整bucket，不假设网格规则、仿射或候选半径。
        self.minimum=tri.min((0,1));self.maximum=tri.max((0,1))
        self.bin_size=float(np.median(np.sqrt(np.abs(det))))
        self.shape=np.maximum(np.floor((self.maximum-self.minimum)/self.bin_size).astype(np.int64)+1,1)
        lower=np.floor((tri.min(1)-self.minimum)/self.bin_size).astype(np.int64)
        upper=np.floor((tri.max(1)-self.minimum)/self.bin_size).astype(np.int64)
        buckets=[[] for _ in range(int(np.prod(self.shape)))]
        for fid,(lo,hi) in enumerate(zip(lower,upper)):
            for iy in range(int(lo[1]),int(hi[1])+1):
                for ix in range(int(lo[0]),int(hi[0])+1):buckets[iy*self.shape[0]+ix].append(fid)
        count=max(map(len,buckets),default=0)
        self.table=np.full((len(buckets),count),-1,np.int32)
        for i,b in enumerate(buckets):
            ordered=sorted(b,key=lambda f:(-self.priority[f],f));self.table[i,:len(b)]=ordered

    def locate(self,world_xy,*,chunk=4096,tolerance=1e-9):
        points=np.asarray(world_xy,np.float64);original_shape=points.shape[:-1];p=points.reshape(-1,points.shape[-1])[:,:2]
        fid=np.full(len(p),-1,np.int32);bary=np.full((len(p),3),np.nan,np.float64)
        finite=np.isfinite(p).all(1);inside=finite&(p>=self.minimum).all(1)&(p<=self.maximum).all(1)
        ids=np.flatnonzero(inside)
        for start in range(0,len(ids),chunk):
            index=ids[start:start+chunk];ij=np.floor((p[index]-self.minimum)/self.bin_size).astype(np.int64)
            ij=np.minimum(np.maximum(ij,0),self.shape-1);candidate=self.table[ij[:,1]*self.shape[0]+ij[:,0]]
            safe=np.maximum(candidate,0);delta=p[index,None,:]-self.origin[safe]
            st=np.einsum('nkij,nkj->nki',self.inverse[safe],delta)
            weights=np.concatenate([1-st.sum(-1,keepdims=True),st],axis=-1)
            ok=(candidate>=0)&(weights>=-tolerance).all(-1)&(weights<=1+tolerance).all(-1)
            any_ok=ok.any(1);chosen=np.argmax(ok,axis=1);ii=np.flatnonzero(any_ok)
            fid[index[ii]]=candidate[ii,chosen[ii]];bary[index[ii]]=weights[ii,chosen[ii]]
        valid=fid>=0;position=np.full((len(p),3),np.nan)
        position[valid]=np.einsum('ni,nij->nj',bary[valid],self.vertices[self.faces[fid[valid]]])
        error=np.full(len(p),np.nan);error[valid]=np.linalg.norm(position[valid,:2]-p[valid],axis=1)
        return dict(valid=valid.reshape(original_shape),fixed_face_id=fid.reshape(original_shape),
            barycentric=bary.reshape(*original_shape,3),world_xyz=position.reshape(*original_shape,3),worldXY_reconstruction_error=error.reshape(original_shape))


def sample_uv_rgba(texture,uv):
    """官方最终Trimesh UV为V向上；图像左上原点、归一化texel-center bilinear。"""
    image=np.asarray(texture,np.float64);h,w=image.shape[:2];uv=np.asarray(uv,np.float64)
    xy=np.c_[uv[:,0]*w-.5,(1-uv[:,1])*h-.5];xy=np.clip(xy,[0,0],[w-1,h-1]);lo=np.floor(xy).astype(np.int64);hi=np.minimum(lo+1,[w-1,h-1]);t=xy-lo;x,y=lo.T;xx,yy=hi.T;u,v=t.T
    return (1-u)[:,None]*(1-v)[:,None]*image[y,x]+u[:,None]*(1-v)[:,None]*image[y,xx]+(1-u)[:,None]*v[:,None]*image[yy,x]+u[:,None]*v[:,None]*image[yy,xx]


class WorldMeshUVMap:
    def __init__(self,world_vertices,faces,face_uv,texture,*,fixed_to_generated_face=None):
        self.vertices=np.asarray(world_vertices,np.float64);self.faces=np.asarray(faces,np.int64);self.face_uv=np.asarray(face_uv,np.float64)
        self.texture=np.asarray(texture,np.float64)
        area=np.abs(np.cross(self.face_uv[:,1]-self.face_uv[:,0],self.face_uv[:,2]-self.face_uv[:,0]))
        self.locator=TriangleLocatorXY(self.vertices,self.faces,face_priority=area)
        self.fixed_to_generated_face=np.arange(len(self.faces)) if fixed_to_generated_face is None else np.asarray(fixed_to_generated_face)

    def query(self,world_xy,*,chunk=4096):
        located=self.locator.locate(world_xy,chunk=chunk);valid=located['valid'];shape=valid.shape
        f=located['fixed_face_id'].ravel();b=located['barycentric'].reshape(-1,3);v=valid.ravel()
        uv=np.full((len(f),2),np.nan);rgba=np.full((len(f),4),np.nan);generated=np.full(len(f),-1,np.int32)
        uv[v]=np.einsum('ni,nij->nj',b[v],self.face_uv[f[v]])
        rgba[v]=sample_uv_rgba(self.texture,uv[v]);generated[v]=self.fixed_to_generated_face[f[v]]
        located.update(generated_face_id=generated.reshape(shape),generated_uv=uv.reshape(*shape,2),basecolor_rgba=rgba.reshape(*shape,4))
        return located

    @classmethod
    def load(cls,folder):
        folder=Path(folder)
        with np.load(folder/'mesh_uv_source.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        from PIL import Image
        texture=np.asarray(Image.open(folder/'basecolor_rgba.png').convert('RGBA'),np.float64)/255
        return cls(d['world_vertices'],d['fixed_faces'],d['generated_uv_by_fixed_face'],texture,fixed_to_generated_face=d['fixed_to_generated_face'])

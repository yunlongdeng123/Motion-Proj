"""固定认证道路三角网：米制XY查询、原三角采样与局部Jacobian。"""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

class FixedRoadSurface:
    def __init__(self, world, support, world_from_grid, target=None):
        self.world=np.asarray(world,np.float64);self.support=np.asarray(support,bool)
        self.matrix=np.asarray(world_from_grid,np.float64);self.inverse=np.linalg.inv(self.matrix[:2,:2])
        self.target=np.asarray(target,bool) if target is not None else np.zeros(self.support.shape,bool)
        self.cell=np.linalg.norm(self.matrix[:2,:2],axis=0)
        self.distance=ndi.distance_transform_edt(self.target,sampling=(self.cell[1],self.cell[0]))
    def grid(self,xy):
        xy=np.asarray(xy,np.float64);g=(xy-self.matrix[:2,3])@self.inverse.T
        h,w=self.support.shape
        near=(g[:,0]>-2)&(g[:,1]>-2)&(g[:,0]<w+1)&(g[:,1]<h+1)
        for _ in range(10):
            position,jac,_,_=self._evaluate(g)
            error=position[:,:2]-xy
            determinant=np.linalg.det(jac[:,:2,:]);valid=near&(abs(determinant)>1e-10)
            if not valid.any() or np.max(abs(error[valid]))<1e-10:break
            step=np.zeros_like(g);step[valid]=np.linalg.solve(jac[valid,:2,:],error[valid,:,None])[...,0]
            g-=np.clip(step,-2,2)
        rounded=np.rint(g);g=np.where(abs(g-rounded)<1e-8,rounded,g)
        return g
    def _evaluate(self,g):
        h,w=self.support.shape
        x=np.floor(g[:,0]).astype(np.int64);y=np.floor(g[:,1]).astype(np.int64)
        inside=np.isfinite(g).all(1)&(x>=0)&(y>=0)&(x<w-1)&(y<h-1)
        x=np.clip(x,0,w-2);y=np.clip(y,0,h-2);f=g-np.c_[x,y]
        a=self.world[y,x];b=self.world[y,x+1];c=self.world[y+1,x+1];d=self.world[y+1,x]
        k=f[:,0]>=f[:,1]
        position=np.where(k[:,None],a*(1-f[:,0,None])+b*(f[:,0]-f[:,1])[:,None]+c*f[:,1,None],
            a*(1-f[:,1,None])+c*f[:,0,None]+d*(f[:,1]-f[:,0])[:,None])
        jac_index=np.empty((len(g),3,2));jac_index[:,:,0]=np.where(k[:,None],b-a,c-d)
        jac_index[:,:,1]=np.where(k[:,None],c-b,d-a)
        good=np.where(k,self.support[y,x]&self.support[y,x+1]&self.support[y+1,x+1],
            self.support[y,x]&self.support[y+1,x+1]&self.support[y+1,x])
        return position,jac_index,good,inside
    def sample(self,uv):
        uv=np.asarray(uv,np.float64);g=self.grid(uv)
        position,jac_index,good,inside=self._evaluate(g)
        # 道路节点XY有亚毫米非仿射校正，必须反演实际三角XY，不能只用atlas仿射近似。
        determinant=np.linalg.det(jac_index[:,:2,:]);nonsingular=abs(determinant)>1e-10
        inverse=np.zeros((len(g),2,2));inverse[nonsingular]=np.linalg.inv(jac_index[nonsingular,:2,:])
        jac=jac_index@inverse
        normal=np.cross(jac[:,:,0],jac[:,:,1]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        normal*=np.where(normal[:,2:3]<0,-1,1)
        error=np.max(abs(position[:,:2]-uv),axis=1)
        return dict(position=position,jacobian=jac,normal=normal,valid=inside&good&nonsingular&(error<1e-8))
    def domain(self,xy):
        g=self.grid(xy)
        return ndi.map_coordinates(self.target.astype(np.float32),[g[:,1],g[:,0]],order=0,mode='constant',cval=0)>.5
    def blend_weight(self,xy,width):
        g=self.grid(xy);distance=ndi.map_coordinates(self.distance,[g[:,1],g[:,0]],order=1,mode='constant',cval=0)
        t=np.clip(distance/width,0,1);return t*t*(3-2*t)
    def save(self,path):np.savez_compressed(path,world=self.world,support=self.support,world_from_grid=self.matrix,target=self.target)
    @classmethod
    def load(cls,path):
        z=np.load(path);return cls(z['world'],z['support'],z['world_from_grid'],z['target'])

def optical_ownership(opacity,weight):
    """按光学厚度分配边界，不以opacity直接线性相乘。"""
    opacity=np.clip(np.asarray(opacity,np.float64),0,1)
    weight=np.clip(np.asarray(weight,np.float64),0,1)
    return -np.expm1(np.log1p(-np.minimum(opacity,1-1e-12))*weight)

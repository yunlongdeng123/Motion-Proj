"""FixedSurface with explicit cell size and near-range bounds for this addon.

Same four-supported-corner rule and ray/height solve as v41_dense_geometry.
The interpolated reference surface is not independent dense geometry truth.
"""
import sys
import numpy as np
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/tools')
from v41_dense_geometry import FixedSurface as OriginalSurface,transform

class FixedSurface(OriginalSurface):
    def __init__(self,meta,units):
        super().__init__(meta,units);self.cell=float(units.get('cell_m',.08))
    def height(self,xy):
        q=np.clip(np.nan_to_num(xy/self.cell-.5),-1e6,1e6);key=np.floor(q).astype(np.int64);fraction=q-key
        ids=np.stack([self.ids(key+offset) for offset in ((0,0),(1,0),(0,1),(1,1))],1);safe=np.maximum(ids,0)
        weights=np.stack(((1-fraction[:,0])*(1-fraction[:,1]),fraction[:,0]*(1-fraction[:,1]),
            (1-fraction[:,0])*fraction[:,1],fraction[:,0]*fraction[:,1]),1)
        heights=(self.local[safe,2]*weights).sum(1);valid=(ids>=0).all(1)&self.supported[safe].all(1)
        normal=self.normals[safe];valid&=((normal*normal[:,:1]).sum(2)>=.9).all(1)
        return heights,valid
    def camera(self,K,w2c):
        yy,xx=np.mgrid[:320,:480];pixels=np.c_[xx.ravel(),yy.ravel()];c2w=np.linalg.inv(w2c)
        rays=np.c_[pixels+.5,np.ones(len(pixels))]@np.linalg.inv(K).T
        rays=rays@c2w[:3,:3].T@self.inverse[:3,:3].T;origin=transform(c2w[None,:3,3],self.inverse)[0]
        denominator=rays[:,2]-rays[:,:2]@self.plane[:2];numerator=origin[:2]@self.plane[:2]+self.plane[2]-origin[2]
        distance=np.divide(numerator,denominator,out=np.full(len(rays),1e5),where=np.abs(denominator)>1e-7)
        initial=origin+distance[:,None]*rays
        potential=(distance>.1)&(distance<120)&(initial[:,0]>=.5)&(initial[:,0]<=90)&(np.abs(initial[:,1])<=8)
        for _ in range(8):
            point=origin+distance[:,None]*rays;height,support=self.height(point[:,:2])
            correction=np.divide(height-point[:,2],denominator,out=np.zeros_like(distance),where=(np.abs(denominator)>1e-7)&support&potential)
            distance+=np.clip(correction,-10,10)
        local=origin+distance[:,None]*rays;height,support=self.height(local[:,:2])
        unit=self.ids(np.floor(np.clip(local[:,:2]/self.cell,-1e6,1e6)).astype(np.int64))
        valid=potential&support&(np.abs(height-local[:,2])<.03)&(unit>=0)
        return dict(world=transform(local,self.anchor),depth=distance,potential=potential,supported=valid,
            observed=valid&self.observed[np.maximum(unit,0)],unit_id=unit,pixels=pixels)

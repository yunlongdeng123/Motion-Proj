"""官方AV2高程资产的普通查询与射线交点；不使用目标GT或未来观测。"""
import json
from pathlib import Path
import numpy as np


class RasterGround:
    def __init__(self, base):
        manifest=json.loads((Path(base)/'input_manifest.json').read_text())
        folder=Path(manifest['raw_path'])/'map'
        self.path=next(folder.glob('*_ground_height_surface____*.npy'))
        transform_path=next(folder.glob('*___img_Sim2_city.json'))
        transform=json.loads(transform_path.read_text())
        self.array=np.load(self.path).astype(np.float64)
        self.R=np.array(transform['R']).reshape(2,2); self.t=np.array(transform['t']); self.s=float(transform['s'])
        self.origin=np.array(manifest['city_origin'])
        self.provenance={'height_asset':str(self.path),'transform':str(transform_path),
                         'source':'https://github.com/argoverse/av2-api/blob/main/src/av2/map/map_api.py',
                         'coordinates':'array_xy = scale * (city_xy @ R.T + t); array[y,x]; local_z=city_z-origin_z',
                         'interpolation':'ordinary bilinear between stored raster samples; missing neighbors return NaN',
                         'raycast':'first above-to-below crossing on .15m samples to80m, then12 bisections; missing prefix rejects',
                         'role':'additional known metric ground-height map; no actor annotations'}

    def height(self, xy):
        xy=np.asarray(xy); shape=xy.shape[:-1]; city=xy.reshape(-1,2)+self.origin[:2]
        uv=(city@self.R.T+self.t)*self.s; q=np.floor(uv).astype(np.int64); frac=uv-q
        valid=(q[:,0]>=0)&(q[:,1]>=0)&(q[:,0]+1<self.array.shape[1])&(q[:,1]+1<self.array.shape[0])
        z=np.full(len(q),np.nan); x,y=q[valid].T; fx,fy=frac[valid].T
        z[valid]=(1-fy)*((1-fx)*self.array[y,x]+fx*self.array[y,x+1])+fy*((1-fx)*self.array[y+1,x]+fx*self.array[y+1,x+1])
        return (z-self.origin[2]).reshape(shape)

    def intersect(self, origin, direction):
        origin=np.asarray(origin); direction=np.asarray(direction)/np.linalg.norm(direction)
        distances=np.r_[np.arange(0.,80.,.15),80.]
        xyz=origin+distances[:,None]*direction
        residual=xyz[:,2]-self.height(xyz[:,:2])
        hit=np.flatnonzero((residual[:-1]>0)&(residual[1:]<=0))
        if not len(hit): return None
        i=int(hit[0])
        if not np.isfinite(residual[:i+2]).all(): return None
        lo,hi=float(distances[i]),float(distances[i+1])
        for _ in range(12):
            mid=(lo+hi)/2; point=origin+mid*direction
            if point[2]-self.height(point[:2])>0: lo=mid
            else: hi=mid
        point=origin+(lo+hi)/2*direction
        assert abs(point[2]-self.height(point[:2]))<.001
        return point

    def contact(self,u,v,camera,K):
        ray=np.array([(u-K[2])/K[0],(v-K[3])/K[1],1.])@camera[:3,:3].T
        return self.intersect(camera[:3,3],ray)

    def mesh_for_route(self,route_xy):
        """给官方GroundSnapper的局部网格；只用已提供路线确定计算范围。"""
        city=np.asarray(route_xy)[:,:2]+self.origin[:2]
        uv=(city@self.R.T+self.t)*self.s
        lower=np.maximum(np.floor(uv.min(0)-10*self.s).astype(int),0)
        upper=np.minimum(np.ceil(uv.max(0)+10*self.s).astype(int)+1,np.array(self.array.shape[::-1]))
        xx,yy=np.meshgrid(np.arange(lower[0],upper[0]),np.arange(lower[1],upper[1]))
        xy=(np.c_[xx.ravel(),yy.ravel()]/self.s-self.t)@self.R-self.origin[:2]
        z=self.array[yy,xx].ravel()-self.origin[2]; vertices=np.c_[xy,z]
        ids=np.arange(xx.size).reshape(xx.shape)
        a,b,c,d=ids[:-1,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,:-1].ravel(),ids[1:,1:].ravel()
        faces=np.concatenate([np.c_[a,b,c],np.c_[b,d,c]])
        faces=faces[np.isfinite(vertices[faces]).all(axis=(1,2))]
        assert len(faces)>0
        return vertices.astype(np.float32),faces.astype(np.int32)

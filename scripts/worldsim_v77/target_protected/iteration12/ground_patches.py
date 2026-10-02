"""由实测近地LiDAR三角片约束可见RGB采样，不外推无支持的地面。"""
import numpy as np
import cv2
from scipy.spatial import Delaunay,QhullError
from actor_state import project_world


def observed_ground_patches(observations, lidar_by_frame, excluded_masks, plane):
    points=[];colors=[];frames=[];pixels=[];stats=[]
    for i,(obs,xyz,excluded) in enumerate(zip(observations,lidar_by_frame,excluded_masks)):
        obs.validate();xyz=np.asarray(xyz)
        uv,z,good=project_world(xyz,obs);ids=np.flatnonzero(good)
        if len(ids):
            order=np.argsort(z[ids],kind='stable');ids=ids[order]
            _,first=np.unique(uv[ids],axis=0,return_index=True);ids=ids[first]
        eligible=np.zeros(obs.hole.shape,np.uint8);accepted=0
        if len(ids)>=3:
            try:triangles=Delaunay(uv[ids].astype(float)).simplices
            except QhullError:triangles=[]
            for tri in triangles:
                ix=ids[tri];p=xyz[ix];screen=uv[ix]
                world_edges=np.linalg.norm(p[:,None]-p[None,:],axis=-1)
                pixel_edges=np.linalg.norm(screen[:,None]-screen[None,:],axis=-1)
                if world_edges.max()>1.5 or pixel_edges.max()>32:continue
                if np.any(obs.hole[screen[:,1],screen[:,0]]) or np.any(excluded[screen[:,1],screen[:,0]]):continue
                cv2.fillConvexPoly(eligible,screen.astype('int32'),1);accepted+=1
        eligible=eligible.astype(bool)&~obs.hole&~excluded
        yy,xx=np.where(eligible);take=(xx%2==0)&(yy%2==0);yy,xx=yy[take],xx[take]
        rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(obs.K).T@obs.camera_to_world[:3,:3].T
        origin=obs.camera_to_world[:3,3];normal=np.r_[-plane[:2],1.]
        denominator=rays@normal
        distance=(plane[2]-origin@normal)/np.where(abs(denominator)>1e-8,denominator,1e-8)
        valid=(distance>.5)&(distance<100)&(abs(denominator)>1e-8)
        pp=origin+distance[valid,None]*rays[valid];yy,xx=yy[valid],xx[valid]
        points.append(pp);colors.append(obs.rgb[yy,xx]);frames.append(np.full(len(pp),i,np.int16));pixels.append(np.c_[xx,yy])
        stats.append({'frame':i,'observed_triangles':accepted,'legal_samples':len(pp),'world_edge_max_m':1.5,'pixel_edge_max':32,'outside_support_extrapolated':False})
    return {'world_xyz':np.concatenate(points).astype('float32'),'rgb':np.concatenate(colors),'source_frame':np.concatenate(frames),'source_pixel':np.concatenate(pixels),'positive_evidence_only':True,'confidence':.4},stats

"""CPU reference for experiment B1: directional OBB shadow on a road plane.

Original small reference, not an imported renderer or a trained model. Inputs are
metric world coordinates. It is intentionally independent of frame time and
camera. A future differentiable adapter must match these coordinate conventions.
"""
import itertools
import numpy as np

def plane_frame(plane):
    p=np.asarray(plane,dtype=float); p=p/np.linalg.norm(p[:3]); n=p[:3]
    seed=np.eye(3)[np.argmin(np.abs(n))]
    u=np.cross(n,seed);u/=np.linalg.norm(u);v=np.cross(n,u)
    return -p[3]*n,np.stack([u,v],axis=1),p

def obb_corners(world_from_actor,size_lwh_m):
    pose=np.asarray(world_from_actor,dtype=float); size=np.asarray(size_lwh_m,dtype=float)
    if pose.shape!=(4,4) or size.shape!=(3,) or not np.all(size>0): raise ValueError('Invalid metric OBB')
    if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-7): raise ValueError('Pose must be rigid')
    corners=np.array(list(itertools.product([-.5,.5],repeat=3)))*size
    return corners@pose[:3,:3].T+pose[:3,3]

def project_to_road(points,plane,receiver_to_light_world):
    _,_,p=plane_frame(plane)
    light=np.asarray(receiver_to_light_world,dtype=float);light/=np.linalg.norm(light)
    denominator=p[:3]@light
    if denominator<=.10: raise ValueError('Light must be above the road; grazing sun is outside this pilot')
    height=np.asarray(points)@p[:3]+p[3]
    if height.min()<-.10: raise ValueError('Caster below road: resolve plane / geometry before fitting light')
    return np.asarray(points)-height[:,None]*light/denominator

def hull2d(points):
    pts=sorted(set(map(tuple,np.asarray(points,dtype=float))))
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        h=[]
        for p in seq:
            while len(h)>=2 and cross(h[-2],h[-1],p)<=0:h.pop()
            h.append(p)
        return h
    h=half(pts)[:-1]+half(pts[::-1])[:-1]
    if len(h)<3:raise ValueError('Degenerate shadow footprint')
    return np.asarray(h)

def shadow_mask(receiver_xyz,world_from_actor,size_lwh_m,plane,receiver_to_light_world,softness_m=.10):
    if softness_m<=0:raise ValueError('Positive metric softness required')
    origin,basis,_=plane_frame(plane)
    corners=obb_corners(world_from_actor,size_lwh_m)
    shadow=project_to_road(corners,plane,receiver_to_light_world)
    hull=hull2d((shadow-origin)@basis)
    query=(np.asarray(receiver_xyz)-origin)@basis
    edge=np.roll(hull,-1,axis=0)-hull
    rel=query[:,None,:]-hull[None,:,:]
    inward=(edge[None,:,0]*rel[:,:,1]-edge[None,:,1]*rel[:,:,0])/np.linalg.norm(edge,axis=1)
    # Smooth convex half-plane footprint. Reference only; no fitted per-pixel residual.
    distance=inward.min(axis=1)
    mask=1/(1+np.exp(-np.clip(distance/softness_m,-40,40)))
    return np.where(distance < -5*softness_m,0,mask)

def apply_shadow(clean_road_rgb,actor_masks,direct_fraction_rgb=(.7,.7,.7)):
    clean=np.asarray(clean_road_rgb,dtype=float);fraction=np.asarray(direct_fraction_rgb,dtype=float)
    if np.any((fraction<0)|(fraction>1)):raise ValueError('Direct-light fraction outside [0,1]')
    visibility=np.ones(clean.shape[:-1])
    for mask in actor_masks:
        mask=np.asarray(mask,dtype=float)
        if mask.shape!=visibility.shape or np.any((mask<0)|(mask>1)):raise ValueError('Invalid actor mask')
        visibility*=1-mask
    # Shared direct light: overlapping actors cannot remove the same direct light twice.
    return clean*((1-fraction)+visibility[...,None]*fraction)

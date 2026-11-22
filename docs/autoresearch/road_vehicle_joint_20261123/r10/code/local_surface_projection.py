"""CPU perspective-correct projection of a certified nonplanar road patch.

Inputs use metric world XYZ and integer image sample centres. The camera K
uses boundary coordinates, as in the pinned native projection cache.
"""
from pathlib import Path
import argparse,json
import numpy as np


def raster_surface(world_xyz,uv_global,geometry_valid,world_to_camera,K,render_wh,cell_mask=None):
    xyz=np.asarray(world_xyz,np.float64);uv=np.asarray(uv_global,np.float64)
    valid=np.asarray(geometry_valid,bool);E=np.asarray(world_to_camera,np.float64);K=np.asarray(K,np.float64)
    h,w=valid.shape;W,H=map(int,render_wh)
    assert xyz.shape==(h,w,3) and uv.shape==(h,w,2) and min(h,w)>=2
    camera=xyz@E[:3,:3].T+E[:3,3];z=camera[...,2]
    q=camera@K.T;screen=q[...,:2]/np.where(z>1e-8,z,1)[...,None]-.5
    valid=valid&np.isfinite(xyz).all(-1)&np.isfinite(uv).all(-1)&(z>1e-8)
    ids=np.arange(h*w).reshape(h,w)
    corners=[ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]]
    tri=np.concatenate([np.stack([corners[0],corners[1],corners[3]],-1).reshape(-1,3),
                        np.stack([corners[0],corners[3],corners[2]],-1).reshape(-1,3)])
    tri_ok=valid.ravel()[tri].all(-1)
    if cell_mask is not None:
        cell_mask=np.asarray(cell_mask,bool);assert cell_mask.shape==(h-1,w-1)
        tri_ok&=np.tile(cell_mask.ravel(),2)
    depth=np.full((H,W),np.inf,np.float64);mapped=np.full((H,W,2),np.nan,np.float64)
    tri_index=np.full((H,W),-1,np.int32);weights=np.zeros((H,W,3),np.float32)
    flat_screen=screen.reshape(-1,2);flat_z=z.ravel();flat_uv=uv.reshape(-1,2)
    for index in np.flatnonzero(tri_ok):
        vertices=tri[index];p=flat_screen[vertices]
        lo=np.maximum(np.floor(p.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(p.max(0)).astype(int),[W-1,H-1])
        if (hi<lo).any():continue
        a,b,c=p;det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(det)<1e-12:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/det
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/det
        affine=np.stack([wa,wb,1-wa-wb],-1);inside=(affine>=-1e-9).all(-1)
        inv=affine/flat_z[vertices];den=inv.sum(-1);good=inside&(den>0)
        dz=np.divide(1.,den,out=np.full_like(den,np.inf),where=den>0)
        take=good&(dz<depth[yy,xx]);corrected=inv/np.where(den>0,den,1)[...,None]
        if not take.any():continue
        y=yy[take];x=xx[take];depth[y,x]=dz[take]
        mapped[y,x]=corrected[take]@flat_uv[vertices]
        tri_index[y,x]=index;weights[y,x]=corrected[take]
    supported=tri_index>=0
    return dict(atlas_uv_global=mapped,projection_valid=supported,camera_depth_m=np.where(supported,depth,0),
                triangle_index=tri_index,triangle_weights=weights,triangles_vertex_indices=tri)


def selfcheck():
    # Independent ray/analytic-plane intersection, with unequal vertex depths.
    yy,xx=np.mgrid[-1.:1.:9j,-1.:1.:9j];z=4.+.3*xx+.1*yy
    xyz=np.stack([xx,yy,z],-1);uv=np.stack([4*(xx+1),4*(yy+1)],-1)
    K=np.array([[100.,0,60.5],[0,105.,45.5],[0,0,1.]])
    out=raster_surface(xyz,uv,np.ones(xx.shape,bool),np.eye(4),K,[121,91])
    iy,ix=np.nonzero(out['projection_valid']);r=np.stack([(ix+.5-K[0,2])/K[0,0],(iy+.5-K[1,2])/K[1,1],np.ones(len(ix))],-1)
    expected_z=4./(1.-.3*r[:,0]-.1*r[:,1]);world=r*expected_z[:,None]
    expected_uv=4*(world[:,:2]+1)
    du=float(abs(out['atlas_uv_global'][iy,ix]-expected_uv).max())
    dz=float(abs(out['camera_depth_m'][iy,ix]-expected_z).max())
    assert len(ix)>100 and du<1e-10 and dz<1e-10
    # No active geometry means no fabricated support.
    empty=raster_surface(xyz,uv,np.zeros(xx.shape,bool),np.eye(4),K,[121,91])
    assert not empty['projection_valid'].any()
    cells=raster_surface(xyz,uv,np.ones(xx.shape,bool),np.eye(4),K,[121,91],cell_mask=np.ones((8,8),bool))
    assert np.array_equal(cells['projection_valid'],out['projection_valid']) and np.array_equal(cells['camera_depth_m'],out['camera_depth_m'])
    no_cells=raster_surface(xyz,uv,np.ones(xx.shape,bool),np.eye(4),K,[121,91],cell_mask=np.zeros((8,8),bool))
    assert not no_cells['projection_valid'].any()
    return dict(complete=True,analytic_plane_independent_ray_check=True,supported_pixels=len(ix),
                UV_max_abs_texels=du,depth_max_abs_m=dz,empty_geometry_has_zero_support=True,
                explicit_cell_scope_all_and_empty_checks=True,GPU_calls=0,Gaussian_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');p.add_argument('--receipt',type=Path)
    a=p.parse_args();assert a.selfcheck
    result=selfcheck()
    if a.receipt:a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

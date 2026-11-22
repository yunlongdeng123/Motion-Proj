"""Measure original real locks' XYZ/projection displacement; never alter locks."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from atlas_canvas_cpu import minimal_vertex_scope


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def stat(x):
    x=np.asarray(x);return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()),min=float(x.min())) if len(x) else dict(count=0)
def pinhole(X,E,K):
    c=X@E[:3,:3].T+E[:3,3];q=c@K.T;return q[:,:2]/np.maximum(q[:,2:],1e-8)-.5,c[:,2]
def brown(X,E,intr):
    c=X@E[:3,:3].T+E[:3,3];z=c[:,2];u,v=(c[:,:2]/np.maximum(z[:,None],1e-8)).T
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intr;r=u*u+v*v;rad=1+k1*r+k2*r*r+k3*r*r*r
    return np.c_[fx*(u*rad+2*p1*u*v+p2*(r+2*u*u))+cx-.5,fy*(v*rad+p1*(r+2*v*v)+2*p2*u*v)+cy-.5],z
def inside(uv,z,w,h):return (z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['r7','geometry','parent-field','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.time();assert not a.output.exists()
    with np.load(a.geometry) as g:world=g['world_xyz'];uv=g['uv_global'];vertex=minimal_vertex_scope(g['cell_asset_mask'])
    with np.load(a.parent_field) as f:locks=f['real_locks'];parent_rgb=f['atlas_linear'];H,W=locks.shape
    with np.load(a.r7/'shared_grid_geometry.npz') as old:oldworld=old['world_xyz'];oldvalid=old['geometry_valid']
    integer=np.rint(uv).astype(int);assert np.array_equal(integer[vertex],uv[vertex]);u,v=integer[...,0],integer[...,1]
    inbounds=(u>=0)&(v>=0)&(u<W)&(v<H);selected=np.zeros(vertex.shape,bool);selected[inbounds]=vertex[inbounds]&locks[v[inbounds],u[inbounds]]
    old=oldworld[v[selected],u[selected]];new=world[selected];assert np.isfinite(old).all() and np.isfinite(new).all()
    delta=new-old;source_rgb=parent_rgb[v[selected],u[selected]];original_count=int(locks.sum());global_uv=uv[selected]
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'locked_vertex_displacement.npz',global_uv=global_uv,old_R7_actual_worldXYZ=old,new_candidate_worldXYZ=new,delta_worldXYZ=delta,parent_locked_linear_RGB=source_rgb)
    B=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');views=[]
    for raw in [86,110,134,179,198]:
        vp=a.r7/f'projection_cache_v1/{raw:03d}/view_cache.npz'
        with np.load(vp) as z:E=z['world_to_camera'];K=z['camera_K']
        op=B/f'observations/{raw:03d}.json';o=read(op);old_uv,oz=pinhole(old,E,K);new_uv,nz=pinhole(new,E,K);common=inside(old_uv,oz,1152,768)&inside(new_uv,nz,1152,768)
        ob,_=brown(old,E,np.asarray(o['camera_intrinsic_Brown']));nb,_=brown(new,E,np.asarray(o['camera_intrinsic_Brown']));bw,bh=o['camera_size'];bc=inside(ob,oz,bw,bh)&inside(nb,nz,bw,bh)
        d=np.linalg.norm(new_uv-old_uv,axis=1);db=np.linalg.norm(nb-ob,axis=1)
        np.savez_compressed(a.output/f'{raw:03d}_projection_displacement.npz',global_uv=global_uv,old_pinhole_uv=old_uv,new_pinhole_uv=new_uv,pinhole_both_in_image=common,
            old_Brown1920_uv=ob,new_Brown1920_uv=nb,Brown_both_in_image=bc,pinhole_delta_px=d,Brown_delta_px=db)
        views.append(dict(raw=raw,all_covered_real_vertices=int(selected.sum()),pinhole1152_common_inframe=int(common.sum()),Brown1920_common_inframe=int(bc.sum()),
            pinhole1152_delta_px=stat(d[common]),Brown1920_delta_px=stat(db[bc]),pinhole_threshold_counts={str(t):int((common&(d>t)).sum()) for t in [1,2,5]},
            old_only_inframe=int((inside(old_uv,oz,1152,768)&~inside(new_uv,nz,1152,768)).sum()),new_only_inframe=int((~inside(old_uv,oz,1152,768)&inside(new_uv,nz,1152,768)).sum()),
            view_cache=pin(vp),observation=pin(op)))
    out=dict(schema='R10.parent_real_locks_new_geometry_displacement_CPU.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        parent_field=pin(a.parent_field),new_geometry=pin(a.geometry),old_actual_R7_geometry=pin(a.r7/'shared_grid_geometry.npz'),parent_real_locks_total=original_count,
        asset_vertex_scope=int(vertex.sum()),covered_parent_real_locked_vertices=int(selected.sum()),covered_lock_old_geometry_valid=int(oldvalid[v[selected],u[selected]].sum()),
        world_displacement_m=stat(np.linalg.norm(delta,axis=1)),XY_displacement_m=stat(np.linalg.norm(delta[:,:2],axis=1)),signed_Z_displacement_m=stat(delta[:,2]),absolute_Z_displacement_m=stat(abs(delta[:,2])),views=views,
        actual_new_and_old_worldXYZ_used=True,same_UV_not_assumed_same_XYZ=True,parent_RGB_and_lock_masks_not_modified=True,
        interpretation='Counts refer to real-locked atlas vertex addresses touched by actual asset cells. Per-view statistics preserve total covered locks and report only mutually in-image projections, with missing domains separate. Nonzero geometry displacement can change the source-photo correspondence; no lock is automatically unlocked/recoloured and Gaussian fitting is not used to conceal this input discrepancy.',
        source_requalification_or_reprojection='Requires explicit assessment of actual pixels/white-line seams and original observations; this diagnostic does not manufacture a fresh observation certificate.',source_accepted=False,geometry_accepted=False,seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()

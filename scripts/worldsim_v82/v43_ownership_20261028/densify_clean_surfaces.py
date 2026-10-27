"""Bounded interpolation of repeatedly measured near-horizontal surfaces.

No extrapolation: require all triangle edges<=1m, normal within18deg of up.
Colors/positions inside triangles are interpolated evidence, not new measurements.
The query z-buffer includes these surfaces only, so these targets are NOT GT.
"""
import json,time,argparse
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import binary_dilation,distance_transform_edt
from build_clean_background_memory import ROOT,CTX,project

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='clean_memory199_dense');args=p.parse_args()
    R=ROOT/'target_unknown_v2';S=R/'clean_memory199v3';D=R/args.output;D.mkdir(exist_ok=False);started=time.monotonic();a=np.load(S/'all_surfaces.npz');pos=a['xyz'];color=a['rgb'];tri=Delaunay(pos[:,:2]).simplices;t=pos[tri]
    edges=np.stack((t[:,1]-t[:,0],t[:,2]-t[:,0],t[:,2]-t[:,1]),1);normal=np.cross(edges[:,0],edges[:,1]);length=np.linalg.norm(edges,axis=2);nl=np.linalg.norm(normal,axis=1);ok=(length.max(1)<=1)&(nl>.001)&(np.abs(normal[:,2])/np.maximum(nl,1e-8)>.95);tri=tri[ok]
    ds=ROOT/'datasets'/CTX;frames=json.loads((ds/'manifest.json').read_text())['frames'];source=R/'fullclip_inputs';touched=np.zeros(len(tri),bool);records=[]
    for f in frames:
        raw=f['raw'];uv,z=project(pos,source,f);ob=np.load(ds/('%03d.npz'%raw));body=ob['body'];domain=binary_dilation(body,iterations=5);d0=np.load(S/('%03d.npz'%raw));image=d0['rgb'].copy();depth=d0['depth'].copy();valid=d0['valid'].copy();allq=uv[tri];allz=z[tri];lo=allq.min(1);hi=allq.max(1)
        # Cheap bounding-box intersection before pixel rasterization.
        yy,xx=np.where(domain);domainlo=np.array([xx.min(),yy.min()]);domainhi=np.array([xx.max(),yy.max()]);potential=(allz>.1).all(1)&(hi>=domainlo).all(1)&(lo<=domainhi).all(1)
        has=np.isfinite(ob['depth']);dist,near=distance_transform_edt(~has,return_indices=True);targetdepth=ob['depth'][tuple(near)]
        for j in np.flatnonzero(potential):
            pts=allq[j].astype(float);zz=allz[j];mn=np.maximum(lo[j],[0,0]);mx=np.minimum(hi[j],[479,319]);xs,ys=np.meshgrid(np.arange(mn[0],mx[0]+1),np.arange(mn[1],mx[1]+1));x=xs.ravel()+.5;y=ys.ravel()+.5;x0,y0=pts[0];x1,y1=pts[1];x2,y2=pts[2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            if abs(den)<1e-6:continue
            b0=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den;b1=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den;b2=1-b0-b1;b=np.stack((b0,b1,b2),1);px=xs.ravel();py=ys.ravel();inside=(b>=-1e-6).all(1)&domain[py,px]
            if not inside.any():continue
            b=b[inside]/zz[None];b/=b.sum(1,keepdims=True);px=px[inside];py=py[inside];znew=(b*zz[None]).sum(1);rgb=b@color[tri[j]]
            admissible=((dist[py,px]>5)|(znew>=targetdepth[py,px]-.3))&(znew<depth[py,px]);px=px[admissible];py=py[admissible]
            if len(px):depth[py,px]=znew[admissible];image[py,px]=rgb[admissible];valid[py,px]=True;touched[j]=True
        np.savez_compressed(D/('%03d.npz'%raw),rgb=image,depth=depth,valid=valid,body=body)
        records.append(dict(raw=raw,body_pixels=int(body.sum()),direct_valid=int((d0['valid']&body).sum()),interpolated_plus_direct_valid=int((valid&body).sum())));print('DENSE_SURFACE',raw,records[-1],flush=True)
    # Sample only touched measured triangles. Preserve direct-return seeds.
    base=np.load(S/'seeds.npz');xyz=[base['xyz']];rgb=[base['rgb']]
    for tr in tri[touched]:
        P=pos[tr];C=color[tr];n=max(1,int(np.ceil(np.linalg.norm(P[:,None]-P[None,:],axis=2).max()/.1)));b=np.array([[i/n,j/n,1-(i+j)/n] for i in range(n+1) for j in range(n+1-i)])
        xyz.append(b@P);rgb.append(b@C)
    xyz=np.concatenate(xyz);rgb=np.concatenate(rgb);_,keep=np.unique(np.floor((xyz-xyz.min(0))/.05).astype('int32'),axis=0,return_index=True);xyz=xyz[keep];rgb=rgb[keep]
    np.savez_compressed(D/'seeds.npz',xyz=xyz.astype('float32'),rgb=rgb.astype('float32'),sigma=np.full(len(xyz),.07,np.float32))
    (D/'summary.json').write_text(json.dumps(dict(schema='measured-surface-interpolation-v1',context=CTX,records=records,source_surface_voxels=len(pos),valid_triangles=len(tri),target_touched_triangles=int(touched.sum()),background_seeds=len(xyz),max_triangle_edge_m=1.,min_normal_up_cosine=.95,interpolated_not_measured=True,query_occlusion_only_interpolated_surfaces_not_fullscene=True,not_GT=True,wall_seconds=time.monotonic()-started),indent=2))

if __name__=='__main__':main()

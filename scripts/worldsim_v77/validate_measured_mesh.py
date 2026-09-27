"""以同相机恒等和独立射线/三角形求交检查r37纹理投影实现。"""
import sys
sys.path.insert(0,'/root/autodl-tmp/work')
import hybrid_measured_road_mesh_control as r
import numpy as np
from repair_common import dump
from pathlib import Path
im,mask,depth,mx,my,which,_=r.project_pair(5,5,r.p.road_roi()[0]);_,_,_,original,_=r.p.get(5);yy,xx=np.where(mask);identity_error=float(np.max(np.hypot(mx[mask]-xx,my[mask]-yy)));assert identity_error<1e-3;assert np.array_equal(im[mask],original[mask])
q,s=10,25;d=np.load(r.ROOT/'visible/f010/source025_map.npz');ys,xs=np.where(d['triangle']>=0);sel=np.linspace(0,len(ys)-1,min(100,len(ys))).astype(int);world,tri,sc,sk,_,_,_=r.source(s);_,qc,qk,_,_=r.p.get(q);errors=[]
for y,x in zip(ys[sel],xs[sel]):
 vertices=world[tri[d['triangle'][y,x]]];ray=qc[:3,:3]@np.linalg.solve(qk,np.array([x,y,1.]));origin=qc[:3,3];a,b,t=np.linalg.solve(np.column_stack([vertices[1]-vertices[0],vertices[2]-vertices[0],-ray]),origin-vertices[0]);point=origin+t*ray;uv,z=r.p.project(point[None],sc,sk);errors.append(float(np.linalg.norm(uv[0]-[d['mx'][y,x],d['my'][y,x]])))
assert max(errors)<1e-3
dump(r.ROOT/'renderer_validation.json',dict(identity_pixels=int(mask.sum()),identity_max_map_error_px=identity_error,identity_rgb_exact=True,independent_ray_triangle_samples=len(errors),max_source_reprojection_error_px=max(errors),scope='Projection implementation checks, not semantic visibility or cross-time camera calibration proof'))
print('MESH_PROJECTION_VALIDATED',identity_error,max(errors))

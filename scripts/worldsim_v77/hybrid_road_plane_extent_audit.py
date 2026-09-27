from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hybrid_road_plane_control as p
from geometry import transform
import numpy as np
from repair_common import read,dump
fit=read(p.ROOT/'plane_fit.json');n=np.array(fit['normal']);d=fit['offset'];roi,_=p.road_roi();rows=[]
for f in [0,5,10,20,30,40,60,90]:
 fr,c,k,im,ex=p.get(f);b=next((b for b in fr['all_boxes'] if b['actor_id']=='12'),None)
 pts=transform(np.fromfile(p.DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(p.DATA/'lidar_pose'/f'{f:03}.txt'));uv,z=p.project(pts,c,k);ij=np.rint(uv).astype(int);good=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);idx=np.flatnonzero(good);idx=idx[roi[ij[idx,1],ij[idx,0]]&~ex[ij[idx,1],ij[idx,0]]];res=pts[idx]@n+d
 row=dict(frame=f,camera=c[:3,3].tolist(),actor_center=np.array(b['pose'])[:3,3].tolist() if b else None,ground_candidates=len(idx),signed_ground_residual_quantiles=np.percentile(res,[10,50,90]).tolist(),candidate_z_camera_quantiles=np.percentile(pts[idx,2]-c[2,3],[10,50,90]).tolist());rows.append(row);print(row,flush=True)
dump(p.BASE/'r27/plane_extent_audit.json',dict(roles='post-control diagnostic of the processed source LiDAR/camera files, not additional raw sensor provenance verification',rows=rows,human_verdict=None))

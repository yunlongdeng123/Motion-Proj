from pathlib import Path
import sys,numpy as np
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from hybrid_road_height_prefix_control import local_ground
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');frames=read(T/'r5/camera_frames.json');rows=[]
for fr in frames:
    f=fr['frame'];b=next(b for b in fr['all_boxes'] if b['actor_id']=='12');p=np.array(b['pose'])[:3,3];bottom=p[2]-b['size_lwh'][2]/2;g,n,d=local_ground(f);z=-(n[0]*p[0]+n[1]*p[1]+d)/n[2];distance=float(np.linalg.norm(g[:,:2]-p[:2],axis=1).min());rows.append(dict(frame=f,gt_bottom=float(bottom),plane_ground=float(z),gt_bottom_minus_plane=float(bottom-z),nearest_plane_support_xy_m=distance,support_points=len(g)))
dump(T/'r5/ground_audit.json',dict(rows=rows,scope='Existing road ROI LiDAR plane; diagnose GT grounding, not unqualified new placement. Plane extrapolation depends on source proximity, original RGB review remains necessary.',human_verdict=None));print(rows)

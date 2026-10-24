"""Read-only native initialization provenance for traced static lineage roots."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v4/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/diagnostics_20261021/src')
from export_v1_ground import ply_points
from backend import Adapter,clean,CAMPAIGN
from export_inspection import measured_frame
import torch
torch.set_num_threads(1)
out=Path('/root/autodl-tmp/gsdiag_v41_ghost_20261024_r1')
run=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/arms/R/run30k')
cfg=json.loads((run/'effective_config.json').read_text())
import re
with (run.parent/'r_train.log').open() as f:initial_log=f.read(14000)
initial_count=int(re.search(r'Number of points at initialization\s*:\s*(\d+)',initial_log).group(1))
world,t=ply_points(run/'input.ply')
import open3d as o3d
native_pcd=o3d.io.read_point_cloud(str(Path(cfg['source_path'])/'points3d.ply'))
assert np.array_equal(world,np.asarray(native_pcd.points)), 'input.ply differs from exact loader point order'
assert np.array_equal(t,np.asarray(native_pcd.normals)[:,0].astype(np.float32))
dynamic_paths=sorted(Path(cfg['source_path']).glob('points3d_dynamic_*.ply'))
dyn=np.concatenate([ply_points(p)[0] for p in dynamic_paths])
native_dyn=np.concatenate([np.asarray(o3d.io.read_point_cloud(str(p)).points) for p in dynamic_paths])
assert np.array_equal(dyn,native_dyn)
distance=cKDTree(dyn).query(world,k=1,workers=1)[0];keep=distance>=.1
native_keep=cKDTree(native_dyn).query(np.asarray(native_pcd.points),k=1,eps=0,distance_upper_bound=.1,workers=1)[0]>=.1
assert np.array_equal(keep,native_keep), 'exclusion mask/order mismatch'
rows=np.flatnonzero(keep);world,t=world[keep],t[keep];distance=distance[keep]
assert cfg['including_dynamic'] and cfg['random_init_point']==0
assert len(world)==initial_count,(len(world),initial_count)
a=Adapter() # camera/world mapping independently checked by native export_v41_results
pay,step=torch.load(str(run/'chkpnt30000.pth'),map_location='cpu',weights_only=False,mmap=True)
lineage=pay[14]['sgcm_lineage_v1']['lineage_ids'].numpy();assert lineage.max()<len(world)
roots=sorted(set(x['source_lineage'] for r in json.loads((out/'ghost_summary.json').read_text())['queries'] if r['mode']=='deleted_five' for x in r['top'] if x['source_lineage'] is not None))
node_counts={}
for step in (10000,20000,30000):
 p,n=torch.load(str(run/f'chkpnt{step}.pth'),map_location='cpu',weights_only=False,mmap=True)
 assert n==step
 roots_n=p[14]['sgcm_lineage_v1']['lineage_ids'].numpy()
 node_counts[str(step)]={'total_static':len(roots_n),'root448440_descendants':int((roots_n==448440).sum()),'root420021_descendants':int((roots_n==420021).sum())}
results=[];anchor=np.linalg.inv(np.asarray(a.meta['coordinate_anchor']));readiness=json.loads(Path(a.meta['original_readiness']).read_text());plane=np.asarray(readiness['SGCM_geometry']['ground_fit']['plane_anchor'])
for root in roots:
 w=world[root];anchorpt=anchor[:3,:3]@w+anchor[:3,3];model=a.A[:3,:3]@w+a.A[:3,3]
 raw=int(round(float(t[root])/.02))+113
 rec=dict(root=int(root),input_row=int(rows[root]),initial_world_m=w,initial_model=model,initial_t=float(t[root]),inferred_source_raw=raw,nearest_initial_dynamic_distance_m=float(distance[root]),height_from_fitted_ground_m=float(abs(plane[:3]@anchorpt+plane[3])/np.linalg.norm(plane[:3])),descendants=int((lineage==root).sum()))
 if 89<=raw<=137:
  c=a.camera(raw);p=c['W']@np.r_[model,1];uv=c['K']@p[:3];uv=uv[:2]/uv[2]-.5;x,y=np.rint(uv).astype(int);rec.update(source_uv=uv,source_camera_z_m=float(p[2]/a.scale))
  if 0<=x<480 and 0<=y<320:
   with np.load(CAMPAIGN/f'final49/full_r2/0008{raw:03d}/render_float.npz') as z:rec['source_target_body_mask']=bool(z['body_mask'][0,y,x]);rec['source_rgb']=z['gt'][:,y,x]
   lidar=measured_frame(a,raw)['lidar_nearest_m'];yx=np.argwhere(lidar>0);ds=np.square(yx-np.array([y,x])).sum(1);best=np.argmin(ds);rec.update(nearest_lidar_pixel_distance=float(np.sqrt(ds[best])),nearest_lidar_z_m=float(lidar[tuple(yx[best])]))
 results.append(rec)
value=dict(status='NATIVE_INITIAL_COUNT_AND_LINEAGE_BOUND',roots=results,initial_static_count=len(world),native_open3d_point_order_verified=True,native_exclusion_mask_verified=True,node_counts=node_counts,source='actual R input.ply + original dynamic-neighbor <0.1m exclusion + saved SGCM lineage; no inferred optimization history',limitations='root lineage identifies common initialization; cloning/splitting/motion makes descendant location different. Mask projection alone does not prove foreground misassignment.',updates=0)
(out/'root_provenance.json').write_text(json.dumps(clean(value),indent=2))
print(json.dumps(clean(value),indent=2))

"""一次真实几何资产干预：右前方有LiDAR支撑的路面普通平面修复，附同面删除对照。"""
import json,sys,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from PIL import Image
B=Path('/root/autodl-tmp/external/worldsim_simimpact');sys.path[:0]=[str(B/'NAVSIM'),str(B/'HUGSIM'),str(B/'HUGSIM/sim')]
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Lidar
from hugsim.dataparser import parse_raw
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');L=I/'lidar_policy';C=L/'pi3x_geometry_oracle';C.mkdir(exist_ok=True)
reg={'task_id':'WS-SIM-PI3X-GEOMETRY-01','run_id':'20260915-r1','scene':'scene-0061',
 'role':'Actual saved triangle-asset intervention and rerendering; local ordinary plane with additional real LiDAR information. Same-time discovery, not a novel-pose closed loop or independent confirmation.',
 'selection':'Right-forward region chosen because six-view sensor oracle both recovered nominal contact and only-local error induced contact; twelve-view non-recovery retained.',
 'region':'0<x<=25, -12<=y<-3 m; original ego frame',
 'reference':'GT returns in region and -0.3<z<0.2; RANSAC plane 0.05m / 1000 iterations. Require |normal_z|>0.95 and >=100 supporting returns. No downstream tuning.',
 'repair':'Move mesh vertices vertically to fitted plane if within 0.5m plane and 0.35m XY of a measured return classified as plane support; exclude vertices in annotated actor boxes expanded 0.1m. Keep other vertices and triangle connectivity.',
 'deletion_control':'Remove all faces touching the identical selected vertices; no new plane geometry.',
 'coverage':'Restore only the originally missing rays identically in all conditions. Newly missing rays after geometry/deletion stay missing and are counted, so the deletion control is essential.',
 'budget':'Real + original residual / local-plane repair / same-face deletion for each of six and twelve inputs = 7 policy forwards and PDM executions. One fixed radius/plane protocol, no sweep.',
 'claim_boundary':'Ordinary oracle control, not a proposed learned method; nominal contact baseline has 1.6cm clearance. Require geometry readout and downstream recovery together.',
 'seed':20260915,'human_verdict':None}
if (C/'registration.json').exists():raise RuntimeError('Existing geometry audit; inspect before rerun')
(C/'registration.json').write_text(json.dumps(reg,indent=2))
ref=json.loads((L/'scene-0061_log_reference.json').read_text());(C/'scene-0061_log_reference.json').write_text(json.dumps(ref,indent=2))
views=json.loads((I/'inputs/scene-0061.json').read_text())['views'];real=np.load(L/'scans/scene-0061/real.npz');gt=real['points'];origin=real['origin'];dirs=real['directions']
def region(p):return (p[:,0]>0)&(p[:,0]<=25)&(p[:,1]>=-12)&(p[:,1]<-3)
ground_candidates=region(gt)&(gt[:,2]>-.3)&(gt[:,2]<.2)
pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt[ground_candidates]));o3d.utility.random.seed(20260915)
plane,inliers=pcd.segment_plane(distance_threshold=.05,ransac_n=3,num_iterations=1000);plane=np.array(plane)
if plane[2]<0:plane=-plane
assert abs(plane[2])>.95 and len(inliers)>=100
plane_z=lambda p:-(p[:,0]*plane[0]+p[:,1]*plane[1]+plane[3])/plane[2]
support=gt[ground_candidates][inliers];ground_flags=np.abs(gt[:,2]-plane_z(gt))<.05;tree=cKDTree(gt[:,:2])
rows=[];geometry_rows=[];torch.set_num_threads(4);torch.manual_seed(20260915)
c=TransfuserConfig();c.latent=False;agent=TransfuserAgent(c,1e-4,str(L/'assets/transfuser_seed_0.ckpt'));agent.initialize();agent.cpu().eval()
rgb={v['camera']:np.array(Image.open(v['image']).convert('RGB')) for v in views[:6] if v['camera'] in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}
data=parse_raw(({'rgb':rgb},{'ego_pos':[0,0,0],'ego_steer':0,'ego_velo':float(ref['initial_velocity_xy_mps'][0]),'accelerate':0}))['input']
data.ego_statuses[-1].ego_velocity=np.array(ref['initial_velocity_xy_mps'],np.float32);data.ego_statuses[-1].ego_acceleration=np.array(ref['initial_acceleration_xy_mps2'],np.float32)
def predict(points,condition,variant=None,changed=0):
 pc=np.zeros((6,len(points)),np.float32);pc[:3]=points.T;data.lidars[-1]=Lidar(pc);features={}
 for b in agent.get_feature_builders():features.update(b.compute_features(data))
 with torch.no_grad():p=agent({k:v.unsqueeze(0) for k,v in features.items()})['trajectory'][0].numpy()
 r={'scene':'scene-0061','condition':condition,'trajectory':p.tolist(),'protocol':'build_scale','points':len(points),'changed_returns':changed,'input_scope':'Saved triangle geometry changed, raycast again; original missing-ray restoration fixed'}
 if variant:r.update(method='pi3x',variant=variant)
 rows.append(r);(C/'policy_probe_summary.json').write_text(json.dumps(rows,indent=2));print('GEOMETRY_POLICY',condition,variant,len(points),flush=True)
predict(gt,'real')
rays=np.c_[np.broadcast_to(origin,gt.shape),dirs].astype(np.float32)
for variant in ['six','twelve']:
 s=np.load(L/'scans/scene-0061'/f'pi3x_{variant}_build_scale.npz');V=s['vertices'];F=s['faces'];old=s['first_range'];oldfinite=np.isfinite(old)
 pp=gt.copy();pp[oldfinite]=origin+dirs[oldfinite]*old[oldfinite,None];predict(pp,'full_restore_all_missing',variant)
 dist,idx=tree.query(V[:,:2],workers=2);mask=region(V)&(abs(V[:,2]-plane_z(V))<.5)&(dist<.35)&ground_flags[idx]
 for actor in ref['records'][0]['boxes']:
  b=np.array(actor['box']);h=b[6];rotation=np.array([[np.cos(h),-np.sin(h),0],[np.sin(h),np.cos(h),0],[0,0,1]])
  q=(V-b[:3])@rotation;inside=(abs(q)<np.array([b[4],b[3],b[5]])/2+.1).all(1);mask&=~inside
 fixed=V.copy();fixed[mask,2]=plane_z(V[mask]);touched=mask[F].any(1)
 np.savez_compressed(C/f'{variant}_local_plane_asset.npz',vertices=fixed,faces=F,changed_vertices=mask)
 for condition,verts,faces in [('local_plane_repair',fixed,F),('same_faces_deleted',V,F[~touched])]:
  scene=o3d.t.geometry.RaycastingScene(nthreads=2);scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(verts),o3d.core.Tensor(faces)))
  f=scene.cast_rays(o3d.core.Tensor(rays),nthreads=2)['t_hit'].numpy();finite=np.isfinite(f);kept=oldfinite&finite
  points=np.concatenate([origin+dirs[kept]*f[kept,None],gt[~oldfinite]])
  changed=oldfinite&finite&(abs(f-old)>.0001)
  rr={'variant':variant,'condition':condition,'vertices_changed':int(mask.sum()),'faces_touched':int(touched.sum()),'old_finite_rays':int(oldfinite.sum()),'new_missing_on_old_finite':int((oldfinite&~finite).sum()),'changed_ranges':int(changed.sum()),'changed_outside_region_GT':int((changed&~region(gt)).sum()),'height_shift_quantiles_m':np.quantile((fixed-V)[mask,2],[0,.1,.5,.9,1]).tolist(),
      'range_mae_before_m':float(abs(old[oldfinite]-real['ranges'][oldfinite]).mean()),'range_mae_after_on_retained_m':float(abs(f[kept]-real['ranges'][kept]).mean())}
  geometry_rows.append(rr);np.savez_compressed(C/f'{variant}_{condition}_readout.npz',first_range=f,points=points)
  predict(points,condition,variant,int(changed.sum()))
(C/'geometry_audit.json').write_text(json.dumps({'plane':plane.tolist(),'support_returns':len(support),'support_RMSE_m':float(np.sqrt(np.mean((support[:,2]-plane_z(support))**2))),'rows':geometry_rows},indent=2))
print('GEOMETRY_AUDIT_DONE',len(rows),flush=True)

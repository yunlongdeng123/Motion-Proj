"""Frozen10k CPU nominations for occluded asset contamination; never edits a Gaussian."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial import cKDTree
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r10/shadow_rebuild_v3'))
from shadow_rebuild.math3d import covariance
torch.set_num_threads(2);start=time.monotonic();out=r/'residual_source_mixtures_010000_r10';out.mkdir(exist_ok=False)
s=torch.load(r/'E6_visible_identity_s0/gate_010000.pt',map_location='cpu',weights_only=False)
if s['global_step']!=10000:raise ValueError('frozen checkpoint differs')
state=s['state'];xyz=state['field.xyz'].numpy();seed=state['field.seed_xyz'].numpy();owner=state['field.owner'].numpy()
cov=covariance(state['field.quaternion'],state['field.log_scales'].exp()).numpy();attribution=np.load(r/'commonR9_E6_010000/per_gaussian.npz');ev=attribution['evidence'];hits=attribution['frames'];records=[]
for actor in [5,6,12,18,25,26,37]:
 b=np.load(r/'initialization_source_audit_r7'/f'a{actor:02d}_actual_banks.npz');rows=np.flatnonzero(owner==actor)
 if not len(b['road']) or not len(b['body']):records.append(dict(actor=actor,reason='missing positive source class; no nomination'));continue
 rt=cKDTree(b['road']);bt=cKDTree(b['body']);rd,ri=rt.query(xyz[rows],workers=1);bd,bi=bt.query(xyz[rows],workers=1);precision=np.linalg.inv(cov[rows])
 dr=b['road'][ri]-xyz[rows];db=b['body'][bi]-xyz[rows];rm=np.einsum('ni,nij,nj->n',dr,precision,dr);bm=np.einsum('ni,nij,nj->n',db,precision,db)
 # Independent positive measurements of BOTH classes, not "no body" => road.
 candidates=np.flatnonzero((rm<9)&(bm<9));nominations=[];both_repeated=0
 for j in candidates:
  row=int(rows[j]);rp=rt.query_ball_point(b['road'][ri[j]],.06);bp=bt.query_ball_point(b['body'][bi[j]],.06)
  rf=np.unique(b['road_frame'][rp]);bf=np.unique(b['body_frame'][bp])
  if len(rf)<3 or len(bf)<3:continue
  both_repeated+=1;road=b['road'][rp];body=b['body'][bp]
  rc=road.mean(0);bc=body.mean(0);direction=bc-rc;distance=float(np.linalg.norm(direction))
  if distance<.025:continue
  direction/=distance;offset=float(((rc+bc)*.5)@direction)
  # Positive body side and negative road side must agree in >=90% actual samples.
  road_accuracy=float((road@direction<offset).mean());body_accuracy=float((body@direction>offset).mean())
  if min(road_accuracy,body_accuracy)<.9:continue
  sigma=float(np.sqrt(direction@cov[row]@direction));standard_offset=float((offset-xyz[row]@direction)/max(sigma,1e-8))
  if abs(standard_offset)>2:continue
  nominations.append(dict(row=row,road_source_frames=rf.tolist(),body_source_frames=bf.tolist(),normal=direction.tolist(),offset=offset,
   road_sample_agreement=road_accuracy,body_sample_agreement=body_accuracy,standard_offset=standard_offset,source_centroid_distance_m=distance,
   nearest_actual_road=dict(bank_row=int(ri[j]),frame_index=int(b['road_frame'][ri[j]]),lidar_row=int(b['road_lidar_row'][ri[j]]),canonical_xyz=b['road'][ri[j]].tolist()),
   nearest_actual_body=dict(bank_row=int(bi[j]),frame_index=int(b['body_frame'][bi[j]]),lidar_row=int(b['body_lidar_row'][bi[j]]),canonical_xyz=b['body'][bi[j]].tolist()),
   fullscene_own_body_mass=float(ev[row,0]),fullscene_other_body_mass=float(ev[row,1]),fullscene_road_mass=float(ev[row,2]),fullscene_visible_mass=float(ev[row,3]),
   source_row_rgb_body_frames=int(hits[row,0])))
 rec=dict(actor=actor,body_and_road_nearest_points_inside3sigma=len(candidates),both_classes_repeated3times=both_repeated,
  locally_separable_nominations=len(nominations),nominated_visible_road_mass=sum(x['fullscene_road_mass'] for x in nominations),
  nominated_low_visible_road_rows=sum(x['fullscene_road_mass']<.02 for x in nominations),nominations=nominations)
 records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='nominations'}),flush=True)
value=dict(global_step=10000,parameters_updated=False,training_updates=0,CPU_only=True,source='actual source banks with raw LiDAR row/frame provenance',
 domain='train46 only; R7 positive body and world-road measurement banks, current R10 3D covariance and exact R9 attribution',
 caveats=['nearest-point3sigma is a nomination, not an exact volume exclusion','R7 world-road bank is not guaranteed camera-visible; source diagnosis is3D, not a new pixel label','known body mask+LiDAR can still have boundary/visibility ambiguity','local separating plane is not proof that a retained child can safely be reanchored; require current multi-view/body volume review'],
 no_height_crop=True,invisible_or_unknown_is_not_background=True,actors=records,elapsed_seconds=time.monotonic()-start)
(out/'summary.json').write_text(json.dumps(value,indent=2));print(json.dumps(dict(status='COMPLETE_READ_ONLY',elapsed_seconds=value['elapsed_seconds'])),flush=True)

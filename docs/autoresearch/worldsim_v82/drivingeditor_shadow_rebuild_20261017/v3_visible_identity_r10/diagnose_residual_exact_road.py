"""Exact actual seed-road source nominations; global repeated world evidence, never a height cut."""
from pathlib import Path
import argparse,sys,json
import numpy as np
from scipy.spatial import cKDTree
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='E6_visible_identity_s0/gate_010000.pt');p.add_argument('--audit',default='commonR9_E6_010000');p.add_argument('--output',default='residual_exact_road_010000_r10.json');a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');sys.path.insert(0,str(r/'implementation_r10/shadow_rebuild_v3'))
from shadow_rebuild.math3d import covariance
torch.set_num_threads(2);target=r/a.output
if target.exists():raise FileExistsError('no overwrite')
s=torch.load(r/a.checkpoint,map_location='cpu',weights_only=False);state=s['state'];xyz=state['field.xyz'].numpy();seed=state['field.seed_xyz'].numpy();owner=state['field.owner'].numpy();cov=covariance(state['field.quaternion'],state['field.log_scales'].exp()).numpy()
audit=np.load(r/a.audit/'per_gaussian.npz')
if not np.array_equal(audit['owner'],owner):raise ValueError('audit identity/shape differs from checkpoint')
ev=audit['evidence'];hits=audit['frames'];atlas=np.load(r/'road_evidence_r5/actual_lidar_atlas.npz');stride=int(atlas['lidar_row'].max())+1;keys=atlas['frame_index'].astype(np.int64)*stride+atlas['lidar_row'];order=np.argsort(keys);keys=keys[order];results=[]
for actor in [5,6,12,18,25,26,37]:
 bank=np.load(r/'initialization_source_audit_r7'/f'a{actor:02d}_actual_banks.npz');rows=np.flatnonzero(owner==actor)
 if not len(bank['road']):results.append(dict(actor=actor,accepted_nominations=0,reason='no positive measured road bank'));continue
 rt=cKDTree(bank['road']);bt=cKDTree(bank['body']) if len(bank['body']) else None;distance,near=rt.query(xyz[rows]);sd,src=rt.query(seed[rows]);precision=np.linalg.inv(cov[rows]);delta=bank['road'][near]-xyz[rows];mahal=np.einsum('ni,nij,nj->n',delta,precision,delta)
 candidates=np.flatnonzero((sd<2e-5)&(distance<.06)&(mahal<9));accepted=[];protected_body=0;protected_rgb=0
 for j in candidates:
  row=int(rows[j])
  if (hits[row,0]>=3 and ev[row,0]>=.5) or (hits[row,1]>=3 and ev[row,1]>=.5):protected_rgb+=1;continue
  radius=4*np.sqrt(np.linalg.eigvalsh(cov[row]).max());neighbors=[] if bt is None else bt.query_ball_point(xyz[row],radius)
  if len(neighbors):
   dd=bank['body'][neighbors]-xyz[row]
   if np.einsum('ni,ij,nj->n',dd,precision[j],dd).min()<=16:protected_body+=1;continue
  b=int(src[j]);fi=int(bank['road_frame'][b]);li=int(bank['road_lidar_row'][b]);key=fi*stride+li;pos=np.searchsorted(keys,key)
  if pos>=len(keys) or keys[pos]!=key:raise ValueError('actual source provenance missing')
  ar=int(order[pos])
  if not bool(atlas['certified'][ar]) or int(atlas['support_frames'][ar])<3:raise ValueError('source world-road lacks repeated support')
  accepted.append(dict(row=row,owner=actor,source_frame_index=fi,source_lidar_row=li,source_atlas_row=ar,actual_world_anchor=atlas['xyz'][ar].tolist(),world_road_support_frames=int(atlas['support_frames'][ar]),
   canonical_seed_exact_distance_m=float(sd[j]),current_distance_to_actual_road_m=float(distance[j]),current_body_4sigma_clear=True,visible_body_mass=float(ev[row,0]),visible_road_mass=float(ev[row,2]),visible_total_mass=float(ev[row,3])))
 result=dict(actor=actor,exact_positive_road_source_candidates=len(candidates),body_volume_protected=protected_body,RGB_body_protected=protected_rgb,accepted_nominations=len(accepted),nominated_road_mass=sum(x['visible_road_mass'] for x in accepted),nominations=accepted);results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='nominations'}),flush=True)
target.write_text(json.dumps(dict(global_step=s['global_step'],parameters_updated=False,training_updates=0,actors=results,source_checkpoint=a.checkpoint,source_audit=a.audit,
 method='exact original canonical seed match2e-5m to actual world-road measurement already repeated>=3 world times; current covariance inside3sigma road, exact4sigma body and repeated RGB body protection',
 caveat='nominations only; no owner/geometry changes. Sparse body bank is incomplete, actual-source view/geometry must be reviewed before mutation; do not apply old row nominations to a later checkpoint',
 no_height_crop=True,invisible_is_not_background=True),indent=2))

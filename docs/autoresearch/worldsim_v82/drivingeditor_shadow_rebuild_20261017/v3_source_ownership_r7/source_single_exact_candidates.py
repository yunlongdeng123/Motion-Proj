"""Read-only: quantify actual exact road-source points excluded only by canonical3times."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r7/shadow_rebuild_v3'))
from shadow_rebuild.math3d import covariance
from shadow_rebuild.train import write
saved=torch.load(r/'E3_source_ownership_s0/latest.pt',map_location='cpu',weights_only=False);s=saved['state']
xyz=s['field.xyz'].numpy();seed=s['field.seed_xyz'].numpy();owner=s['field.owner'].numpy();cov=covariance(s['field.quaternion'],s['field.log_scales'].exp()).numpy()
original=np.load(r/'ownership_007000_r6/per_gaussian.npz');ev=np.zeros((len(owner),4));hits=np.zeros((len(owner),3),np.int16)
ev[:len(original['owner'])]=original['evidence'];hits[:len(original['owner'])]=original['frames'];result=[]
for actor in [5,6,12,18,25,26,37]:
 bank=np.load(r/'initialization_source_audit_r7'/f'a{actor:02d}_actual_banks.npz');rows=np.flatnonzero(owner==actor)
 if not len(bank['road']):continue
 rt=cKDTree(bank['road']);bt=cKDTree(bank['body']) if len(bank['body']) else None
 distance,near=rt.query(xyz[rows]);source_distance,source=rt.query(seed[rows]);precision=np.linalg.inv(cov[rows])
 delta=bank['road'][near]-xyz[rows];mahal=np.einsum('ni,nij,nj->n',delta,precision,delta)
 candidates=np.flatnonzero((source_distance<2e-5)&(distance<.06)&(mahal<9));accepted=[]
 for j in candidates:
  row=rows[j]
  if (hits[row,0]>=3 and ev[row,0]>=.5) or (hits[row,1]>=3 and ev[row,1]>=.5):continue
  radius=4*np.sqrt(np.linalg.eigvalsh(cov[row]).max());neighbors=[] if bt is None else bt.query_ball_point(xyz[row],radius)
  if len(neighbors):
   dd=bank['body'][neighbors]-xyz[row]
   if np.einsum('ni,ij,nj->n',dd,precision[j],dd).min()<=16:continue
  accepted.append(dict(row=int(row),source_bank_row=int(source[j]),source_distance_m=float(source_distance[j]),body_support=False))
 result.append(dict(actor=actor,exact_single_source_positive_candidates=len(candidates),safe_nominations=len(accepted),rows=accepted))
write(r/'exact_source_single_nominations_r7.json',dict(read_only=True,training_updates=0,world_road_support_already_atleast3=True,
       method='exact original seed canonical match2e-5m to an actual LiDAR road row already world-certified across>=3times; no new relabel performed',actors=result))
print(json.dumps([{k:v for k,v in a.items() if k!='rows'} for a in result]))

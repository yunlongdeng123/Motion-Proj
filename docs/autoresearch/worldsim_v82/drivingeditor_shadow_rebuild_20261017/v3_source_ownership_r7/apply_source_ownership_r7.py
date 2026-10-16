"""由实际初始化源点的道路正证据修正归属；精确检查体积内车体支持。"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial import cKDTree
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r7/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write,save
from shadow_rebuild.ownership import reanchor_road
from shadow_rebuild.math3d import covariance,logit,srgb_to_linear
from shadow_rebuild.references import refresh
torch.set_num_threads(2);start=time.monotonic();out=r/'E3_source_ownership_s0';out.mkdir(exist_ok=False)
parent=r/'E3_contour_split_s0/latest.pt';saved=torch.load(parent,map_location='cpu',weights_only=False)
if saved['global_step']!=7000:raise ValueError('frozen source changed')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate();d=Dataset(r/'data_r3_ownership','cpu',True)
model=SceneModel(seed_from_state(saved['state']),cfg);model.load_state_dict(saved['state']);opt=model.optimizer('geometry');opt.load_state_dict(saved['optimizer'])
field=model.field;xyz=field.xyz.detach().numpy();seed=field.seed_xyz.numpy();owners=field.owner.numpy();cov=covariance(field.quaternion,field.scales).detach().numpy()
old=np.load(r/'ownership_007000_r6/per_gaussian.npz');ev=np.zeros((len(xyz),4));nf=np.zeros((len(xyz),3),np.int16)
ev[:len(old['owner'])]=old['evidence'];nf[:len(old['owner'])]=old['frames']
atlas=np.load(r/'road_evidence_r5/actual_lidar_atlas.npz');stride=int(atlas['lidar_row'].max())+1
keys=atlas['frame_index'].astype(np.int64)*stride+atlas['lidar_row'];order=np.argsort(keys);sortedkeys=keys[order]
allrows=[];anchors=[];provenance=[];records=[]
for actor in [5,6,12,18,25,26,37]:
 bank=np.load(r/'initialization_source_audit_r7'/f'a{actor:02d}_actual_banks.npz');rows=np.flatnonzero(owners==actor)
 if not len(bank['road']):records.append(dict(actor=actor,accepted=0,reason='no positive measured road bank'));continue
 roadtree=cKDTree(bank['road']);bodytree=cKDTree(bank['body']) if len(bank['body']) else None
 distance,near=roadtree.query(xyz[rows],workers=1);delta=bank['road'][near]-xyz[rows];precision=np.linalg.inv(cov[rows]);mahal=np.einsum('ni,nij,nj->n',delta,precision,delta)
 candidates=np.flatnonzero((distance<.06)&(mahal<9))
 accepted=0;body_protected=0;rgb_protected=0;insufficient=0
 for j in candidates:
  row=int(rows[j]);neighborhood=roadtree.query_ball_point(seed[row],.06)
  times=np.unique(bank['road_frame'][neighborhood])
  if len(times)<3:insufficient+=1;continue
  # Exact 4sigma enclosing sphere query: nearest16 alone can miss an elongated Gaussian touching a tire.
  radius=4*float(field.scales[row].max().detach());neighbors=[] if bodytree is None else bodytree.query_ball_point(xyz[row],radius)
  body_mahal=float('inf')
  if len(neighbors):
   delta=bank['body'][neighbors]-xyz[row];body_mahal=float(np.einsum('ni,ij,nj->n',delta,precision[j],delta).min())
  if body_mahal<=16:body_protected+=1;continue
  if (nf[row,0]>=3 and ev[row,0]>=.5) or (nf[row,1]>=3 and ev[row,1]>=.5):rgb_protected+=1;continue
  b=int(near[j]);frame_index=int(bank['road_frame'][b]);lidar_row=int(bank['road_lidar_row'][b]);key=frame_index*stride+lidar_row
  pos=np.searchsorted(sortedkeys,key)
  if pos==len(sortedkeys) or sortedkeys[pos]!=key:raise ValueError('actual measurement provenance missing')
  ar=int(order[pos]);world=atlas['xyz'][ar]
  if not bool(atlas['certified'][ar]):raise ValueError('anchor lacks repeated world-road evidence')
  allrows.append(row);anchors.append(world);accepted+=1
  provenance.append(dict(row=row,previous_owner=actor,new_owner=-1,actual_world_xyz=world.tolist(),
    actual_source=dict(frame_id=d.rows[frame_index]['frame_id'],frame_index=frame_index,lidar_row=lidar_row,atlas_row=ar,world_support_frames=int(atlas['support_frames'][ar])),
    canonical_road_source_frames=times.tolist(),canonical_distance_m=float(distance[j]),road_mahal_squared=float(mahal[j]),
    exact_body_4sigma_support=False,fullscene_evidence=ev[row].tolist(),fullscene_body_frames=int(nf[row,0])))
 records.append(dict(actor=actor,positive_volume_candidates=len(candidates),accepted=accepted,actual_body_protected=body_protected,
                     repeated_RGB_body_protected=rgb_protected,insufficient_temporal_source=insufficient))
 print(json.dumps(records[-1]),flush=True)
rr=torch.tensor(allrows,dtype=torch.long);anchor=np.asarray(anchors,np.float32).reshape(-1,3)
source=np.flatnonzero(d.seed['road']);tree=cKDTree(d.seed['xyz'][source]);_,near=tree.query(anchor,workers=1)
reanchor_road(field,rr,torch.from_numpy(anchor),torch.from_numpy(d.seed['quaternion'][source[near]]),opt)
with torch.no_grad():
 if len(rr):field.log_scales[rr]=torch.from_numpy(d.seed['scales'][source[near]]).log();field.anchor_log_scales[rr]=field.log_scales[rr]
 for actor in np.unique(field.owner.numpy()):
  ids=np.flatnonzero(field.owner.numpy()==actor);points=field.xyz.detach().numpy()[ids];nn=cKDTree(points).query(points,k=min(2,len(ids)),workers=1)[1]
  field.neighbor[ids]=torch.as_tensor(ids[nn[:,1] if nn.ndim==2 else nn])
 # Actual observed road RGB, then independently repeated bright-reference initialization where available.
 frames={}
 for row,p in zip(rr,provenance):
  index=p['actual_source']['frame_index'];ar=p['actual_source']['atlas_row'];pixel=atlas['uv'][ar]
  if index not in frames:frames[index]=d.frame(index)
  color=frames[index].rgb[int(pixel[1]),int(pixel[0])];field.appearance_logit[row]=logit(color);field.albedo_logit[row]=logit(color)
reference=refresh(field,d)
with torch.no_grad():
 trusted=rr[field.reference_confidence[rr]>0]
 field.appearance_logit[trusted]=logit(srgb_to_linear(field.reference_rgb[trusted]));field.albedo_logit[trusted]=field.appearance_logit[trusted]
saved['state']=model.state_dict();saved['optimizer']=opt.state_dict();saved['supervision_lineage']=dict(kind='actual_source_positive_road_repair',parent_checkpoint=str(parent),previous=saved.get('supervision_lineage'),receipt='source_ownership.json')
save(out/'latest.pt',saved);write(out/'resolved_config.json',saved['config']);write(out/'input_contract.json',saved['input_contract'])
receipt=dict(global_step=7000,training_updates=0,parent_checkpoint=str(parent),reanchored=len(rr),actors=records,reference=reference,
             no_height_crop=True,invisible_is_not_background=True,
             method='actual positive repeated LiDAR-road source; exact4sigma body volume exclusion + repeated RGB-body protection; measured world anchor, not arbitrary pose freeze',
             unknown_without_road_positive='retained unchanged',provenance=provenance,elapsed_seconds=time.monotonic()-start)
write(out/'source_ownership.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='provenance'}),flush=True)

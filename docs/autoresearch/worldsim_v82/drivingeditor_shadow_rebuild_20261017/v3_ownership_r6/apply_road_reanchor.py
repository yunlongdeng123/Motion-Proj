"""Positive-evidence road ownership repair; explicit new dataset/checkpoint lineage."""
from pathlib import Path
import sys,json,copy,os,time,argparse
import numpy as np
import torch
from scipy.spatial import cKDTree
p=argparse.ArgumentParser();p.add_argument('--output',default='E2_ownership_s0');p.add_argument('--source-run',default='E2_split_s0');p.add_argument('--data-output',default='data_r3_ownership');p.add_argument('--skip-reanchor',action='store_true');p.add_argument('--raw-exterior',action='store_true');a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r6/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write,save
from shadow_rebuild.ownership import classify_evidence,reanchor_road
from shadow_rebuild.references import refresh
torch.set_num_threads(2);start=time.monotonic();out=r/a.output;out.mkdir(exist_ok=False)
parent=r/a.source_run/'latest.pt';saved=torch.load(parent,map_location='cpu',weights_only=False)
if saved['global_step']!=6000:raise ValueError('parent gate mismatch')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate()
d=Dataset(r/'data_r2','cpu',True);model=SceneModel(seed_from_state(saved['state']),cfg);model.load_state_dict(saved['state'])
optimizer=model.optimizer('geometry');optimizer.load_state_dict(saved['optimizer'])
audit=r/'ownership_split_006000_r5c';e=np.load(audit/'per_gaussian.npz');actors=[5,6,12,18,25,26,37]
labels=classify_evidence(torch.from_numpy(e['evidence']),torch.from_numpy(e['frames'].astype(np.int32)),model.field.owner,actors)
candidate=labels['road_candidate'].nonzero().flatten().numpy();owners=model.field.owner.numpy()[candidate]
local=model.field.xyz.detach().numpy()[candidate];scale=model.field.scales.detach().numpy()[candidate].max(-1)
atlas=np.load(r/'road_evidence_r5/actual_lidar_atlas.npz');cert=atlas['certified'];allxyz=atlas['xyz'];source=atlas['frame_index']
records=[[] for _ in candidate]
for index in d.train_indices:
 f=d.frame(index);frame_rows=np.flatnonzero(cert&(source==index))
 if not len(frame_rows):continue
 tree=cKDTree(allxyz[frame_rows]);v=np.load(audit/(f.frame_id+'.npz'))
 ev=np.zeros((len(model.field.xyz),4),np.float32);ev[v['rows']]=v['evidence'];ev=ev[candidate]
 for actor in actors:
  take=np.flatnonzero((owners==actor)&(ev[:,2]>.02))
  if not len(take) or not bool(f.valid[actor]):continue
  pose=f.poses[actor].numpy();world=local[take]@pose[:3,:3].T+pose[:3,3]
  dist,near=tree.query(world,workers=1)
  for j,pos in enumerate(take):
   if dist[j]>min(.30,2.5*scale[pos]+.05):continue
   row=int(frame_rows[near[j]])
   records[pos].append(dict(frame_id=f.frame_id,frame_index=index,atlas_row=row,lidar_row=int(atlas['lidar_row'][row]),
                            alphaT_road=float(ev[pos,2]),center_distance_m=float(dist[j]),support_frames=int(atlas['support_frames'][row])))
chosen=[];anchors=[];provenance=[]
for j,items in enumerate(records):
 if len(items)<3:continue
 best=max(items,key=lambda x:x['alphaT_road']);chosen.append(int(candidate[j]));anchors.append(allxyz[best['atlas_row']])
 provenance.append(dict(row=int(candidate[j]),previous_owner=int(owners[j]),new_owner=-1,actual_world_xyz=allxyz[best['atlas_row']].tolist(),
                        chosen_actual_measurement=best,positive_observation_frames=items))
if a.skip_reanchor:chosen=[];anchors=[];provenance=[]
rows=torch.tensor(chosen,dtype=torch.long);anchor=np.asarray(anchors,np.float32).reshape(-1,3)
roadrows=np.flatnonzero(d.seed['road']);tree=cKDTree(d.seed['xyz'][roadrows]);_,near=tree.query(anchor,workers=1)
quaternion=torch.from_numpy(d.seed['quaternion'][roadrows[near]])
reanchor_road(model.field,rows,torch.from_numpy(anchor),quaternion,optimizer)
# A moved surface uses an actual measured road orientation and measured seed's local surface shape.
with torch.no_grad():
 if len(rows):
  model.field.log_scales[rows]=torch.from_numpy(d.seed['scales'][roadrows[near]]).log()
  model.field.anchor_log_scales[rows]=model.field.log_scales[rows]
 # Identity-consistent neighbors; do not propagate a new label through old cross-owner kNN.
 for actor in np.unique(model.field.owner.numpy()):
  rr=np.flatnonzero(model.field.owner.numpy()==actor);points=model.field.xyz.detach().numpy()[rr]
  neighbor=cKDTree(points).query(points,k=min(2,len(rr)),workers=1)[1]
  model.field.neighbor[rr]=torch.as_tensor(rr[neighbor[:,1] if neighbor.ndim==2 else neighbor])
# New input version: original RGB/masks/poses unchanged; only add certified train road pixels.
data=r/a.data_output;data.mkdir(exist_ok=False);(data/'frames').mkdir()
for name in ['seed.npz','trajectory.npz']:os.link(r/'data_r2'/name,data/name)
manifest=copy.deepcopy(d.manifest)
for row in manifest['frames']:
 os.link(r/'data_r2'/row['rgb'],data/row['rgb'])
 with np.load(r/'data_r2'/row['supervision']) as old:values={k:old[k] for k in old.files}
 if row['split']=='train':values['road']=values['road']|np.load(r/'road_evidence_r5'/(row['frame_id']+'.npz'))['road']
 if (values['road']&values['masks'].any(-1)).any():raise ValueError('road/body overlap')
 np.savez_compressed(data/row['supervision'],**values)
manifest['ownership_road_evidence']=dict(version='r5',source=str(r/'road_evidence_r5'),train_only=True,
                                       actual_LiDAR_source_rows=True,reference='measured surface certificate, not clean RGB GT')
write(data/'manifest.json',manifest);updated=Dataset(data,'cpu',True);reference=refresh(model.field,updated)
cfg.outside_fullscene_visibility=not a.raw_exterior;model.config=cfg;model.field.config=cfg
saved['config']=__import__('dataclasses').asdict(cfg);saved['state']=model.state_dict();saved['optimizer']=optimizer.state_dict();saved['input_contract']=manifest
saved['supervision_lineage']=dict(kind='positive_evidence_ownership_repair',parent_checkpoint=str(parent),parent_global_step=6000,
                                previous=saved.get('supervision_lineage'),road_receipt='road_reanchor.json',
                                supervision_change='full image exterior uses full-scene visible alpha*T instead of isolated raw alpha; genuine occlusion does not imply background',
                                data_change='only certified train road pixels added, source provenance retained')
save(out/'latest.pt',saved)
receipt=dict(parent_checkpoint=str(parent),global_step=6000,training_updates=0,road_candidates=len(candidate),reanchored=len(chosen),
             mixed_retained=int(labels['mixed'].sum()),unknown_retained=int(labels['unknown'].sum()),
             no_height_crop=True,invisible_is_not_background=True,both_split_children_retained=True,
             anchor='actual measured world LiDAR point with repeated temporal surface support; not arbitrary Actor pose freeze',
             source_frames='train46 only',reanchor_enabled=not a.skip_reanchor,config_change=dict(outside_fullscene_visibility=[False,cfg.outside_fullscene_visibility]),
             reference=reference,provenance=provenance,elapsed_seconds=time.monotonic()-start)
write(out/'road_reanchor.json',receipt);write(out/'resolved_config.json',saved['config']);write(out/'input_contract.json',manifest)
print(json.dumps({k:v for k,v in receipt.items() if k!='provenance'}),flush=True)

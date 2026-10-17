"""Matched OBB vs semantic/independent-motion initialization, rebuilt from raw returns."""
from pathlib import Path
import sys,json,time,os,copy
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_rebuild_r14'
sys.path.insert(0,str(r/'implementation_r14/shadow_rebuild_v3'))
from shadow_rebuild.prepare import transform,project,box_pose,voxel_rows,local_geometry,ground_candidates
from shadow_rebuild.data import clean_reference,validate_seed
from shadow_rebuild.identity import evidence_probability
from shadow_rebuild.config import Config
from dataclasses import asdict
torch.set_num_threads(2);start=time.monotonic();m=json.loads((r/'data_r2/manifest.json').read_text());rows=m['frames'];k=len(m['actor_ids']);targets=[5,6,12,18,25,26,37]
tr=np.load(r/'data_r2/trajectory.npz');poses=tr['poses'];valid=tr['valid'];origin=np.array(m['original_world_origin_m']);scene=Path(m['source_scene'])
if not (root/'shared_masks_r14/summary.json').exists() or not (root/'independent_tracks/summary.json').exists():raise RuntimeError('complete real mask and independent tracks required')
if not (root/'shared_SAM3_root_review.json').exists():raise RuntimeError('Root must review actual native SAM3 output before dataset build')
A=root/'data_A_box';B=root/'data_B_semantic_motion'
A.mkdir(exist_ok=False);B.mkdir(exist_ok=False);(A/'frames').mkdir();(B/'frames').mkdir()
gathered={a:[] for a in range(-1,k)};banks={a:[] for a in targets};frames=[];static_frames=[]
def pack(xyz,rgb,frame,raw,world,owner):
 return dict(xyz=xyz.astype(np.float32),rgb=rgb.astype(np.float32),source_frame=np.full(len(xyz),frame,np.int64),source_raw_row=raw.astype(np.int64),source_world=world.astype(np.float32),source_candidate_owner=np.full(len(xyz),owner,np.int64))
def merge(values):return {key:np.concatenate([v[key] for v in values]) for key in values[0]}
for i,row in enumerate(rows):
 fid=row['frame_id'];d=np.load(root/'raw_measurements'/(fid+'.npz'));world=d['world'];raw=d['raw_row'];owner=d['candidate_owner'];ij=d['pixel'].astype(np.int64);z=d['camera_z'];K=np.array(row['intrinsic']);camera=np.array(row['world_from_camera'])
 sup=np.load(root/'shared_masks_r14'/(fid+'.npz'));mask=sup['masks'];known=sup['mask_known'];rgb=np.asarray(Image.open(r/'data_r2'/row['rgb']).convert('RGB'));build=d['BUILD'];train=row['split']=='train'
 # Depth and visibility for BUILD are recomputed from BUILD returns only. The
 # raw cache's ALL-return frontmost flag/color is diagnostic and not consumed.
 permitted=build if train else np.ones(len(raw),bool);ids=np.flatnonzero(permitted&d['in_image']);bins=ij[ids,1]*480+ij[ids,0];depth=np.full(320*480,np.inf);np.minimum.at(depth,bins,z[ids]);vis=ids[z[ids]<=depth[bins]+.1]
 colors=np.full((len(raw),3),.5,np.float32);colors[vis]=rgb[ij[vis,1],ij[vis,0]]/255.;depth=depth.reshape(320,480);depth[~np.isfinite(depth)]=0
 guard=mask.any(-1).copy()|sup['proposal_union']|sup['uncertainty'];ego=np.loadtxt(scene/'pose'/(fid+'.txt'));ego[:3,3]-=origin
 for label in json.loads((Path(m['annotations'])/(fid+'.json')).read_text())['laser_labels']:
  pose,size=box_pose(label,ego);corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*size;uv,_,good=project(transform(corners,pose),K,camera,depth.shape)
  if good.any():
   lo=np.maximum(uv[good].min(0)-2,0);hi=np.minimum(uv[good].max(0)+2,[479,319]);guard[lo[1]:hi[1]+1,lo[0]:hi[0]+1]=True
 frames.append(dict(row=row,rgb=rgb,masks=mask,mask_known=known,mask_valid=sup['mask_valid'],depth=depth.astype(np.float32),guard=guard))
 for output in [A,B]:os.link(r/'data_r2'/row['rgb'],output/row['rgb'])
 if train:
  for actor in range(-1,k):
   take=build&(owner==actor)
   if actor==-1:take&=d['in_image']&(z<80)
   ids0=np.flatnonzero(take)
   if not len(ids0):continue
   xyz=world[ids0] if actor==-1 else transform(world[ids0],np.linalg.inv(poses[i,actor]))
   entry=pack(xyz,colors[ids0],i,raw[ids0],world[ids0],actor);gathered[actor].append(entry)
   if actor==-1:static_frames.append((i,world[ids0]))
  for actor in targets:
   ids0=vis[(owner[vis]==actor)&bool(known[actor])];ids0=ids0[mask[ij[ids0,1],ij[ids0,0],actor]]
   if len(ids0):banks[actor].append(pack(transform(world[ids0],np.linalg.inv(poses[i,actor])),colors[ids0],i,raw[ids0],world[ids0],actor))
 print(json.dumps(dict(phase='raw_measurements',frame=i+1,total=49)),flush=True)
raw_fields={a:merge(v) for a,v in gathered.items() if v}
for v in raw_fields.values():
 v['identity_probability']=np.tile([.1,.1,.8],(len(v['xyz']),1)).astype(np.float32);v['identity_evidence_status']=np.zeros(len(v['xyz']),np.int64)
static=raw_fields[-1];static_ids=voxel_rows(static['xyz'],.12);static_seed=static['xyz'][static_ids];road=ground_candidates(static_seed);road_ids=np.flatnonzero(road);world_support=np.zeros(len(road_ids),np.int16)
for i,cloud in static_frames:world_support+=cKDTree(cloud).query(static_seed[road_ids],workers=1)[0]<.08
certified_ids=road_ids[world_support>=3];certified_world=static_seed[certified_ids];static_tree=cKDTree(certified_world)
if not len(certified_world):raise RuntimeError('no independent repeated actual world surface support')
np.savez_compressed(root/'actual_world_road_support.npz',xyz=certified_world,source_frame=static['source_frame'][static_ids[certified_ids]],source_raw_row=static['source_raw_row'][static_ids[certified_ids]],support_frames=world_support[world_support>=3])
body={a:merge(v) for a,v in banks.items() if v};body_trees={a:cKDTree(v['xyz']) for a,v in body.items()}
track=np.load(root/'independent_tracks/actual_tracks.npz');p=track['probability_static_actor_unknown'];moving={};stationary=[]
for actor in targets:
 take=(track['candidate_actor']==actor)&(p[:,1]>.7);xyz=track['world'][take];fi=track['frame_index'][take];values=[]
 for xyz3,fi3 in zip(xyz,fi):
  for point,index in zip(xyz3,fi3):values.append(transform(point[None],np.linalg.inv(poses[index,actor]))[0])
 if values:moving[actor]=cKDTree(np.asarray(values))
take=p[:,0]>.7
if take.any():stationary=cKDTree(track['world'][take].reshape(-1,3))
new_fields={a:[v] for a,v in raw_fields.items() if a not in targets};uncertain=[];classification=[]
for actor in targets:
 src=raw_fields[actor];xyz=src['xyz'];world=src['source_world'];tree=body_trees.get(actor);repeat=np.zeros(len(xyz),np.int16);near_color=None
 if tree is not None:
  distance,near=tree.query(xyz,k=min(32,len(body[actor]['xyz'])),workers=1)
  if distance.ndim==1:distance=distance[:,None];near=near[:,None]
  for j in range(len(xyz)):repeat[j]=len(np.unique(body[actor]['source_frame'][near[j,distance[j]<.08]]))
  near_color=body[actor]['rgb'][near[:,0]]
 road_positive=static_tree.query(world,workers=1)[0]<.06;motion_body=np.zeros(len(xyz),bool) if actor not in moving else moving[actor].query(xyz,workers=1)[0]<.1
 motion_static=np.zeros(len(xyz),bool) if not isinstance(stationary,cKDTree) else stationary.query(world,workers=1)[0]<.08
 keep_body=((repeat>=3)&~road_positive)|(motion_body&(repeat>=1));keep_static=(road_positive|motion_static)&~((repeat>=1)|motion_body);pending=~(keep_body|keep_static)
 probability=evidence_probability(repeat>=3,road_positive,motion_body,motion_static)
 for decision,take in [('Actor',keep_body),('static',keep_static),('unknown',pending)]:
  entry={key:value[take].copy() for key,value in src.items()};entry['identity_probability']=probability[take].astype(np.float32);entry['identity_evidence_status']=np.full(int(take.sum()),{'unknown':0,'Actor':1,'static':2}[decision],np.int64)
  if decision=='Actor':
   if near_color is not None:
    gray=np.max(np.abs(entry['rgb']-.5),axis=-1)<1e-6;entry['rgb'][gray]=near_color[take][gray]
   new_fields.setdefault(actor,[]).append(entry)
  elif decision=='static':entry['xyz']=entry['source_world'].copy();new_fields.setdefault(-1,[]).append(entry)
  else:uncertain.append(entry)
 rec=dict(actor=actor,raw_candidate_returns=len(xyz),Actor_kept=int(keep_body.sum()),world_static_reassigned=int(keep_static.sum()),uncertain_pool=int(pending.sum()),semantic_repeat3=int((repeat>=3).sum()),independent_Actor_motion_supported=int(motion_body.sum()),independent_static_motion_supported=int(motion_static.sum()))
 classification.append(rec);print(json.dumps(rec),flush=True)
 np.savez_compressed(root/f'source_candidate_evidence_a{actor:02d}.npz',**src,semantic_repeat=repeat,road_positive=road_positive,motion_body=motion_body,motion_static=motion_static)
 # Keep complete actual semantic measurement banks for staged identity re-evaluation.
 if actor in body:np.savez_compressed(root/f'body_bank_a{actor:02d}.npz',**body[actor])
if uncertain:np.savez_compressed(root/'uncertain_candidate_pool.npz',**merge(uncertain))
def seed_from_raw(source):
 parts=[];offset=0
 for actor in sorted(source):
  src=merge(source[actor]) if isinstance(source[actor],list) else source[actor]
  if not len(src['xyz']):continue
  chosen=voxel_rows(src['xyz'],.12 if actor==-1 else .04);g={key:value[chosen] for key,value in src.items()};xyz=g['xyz'];isroad=ground_candidates(xyz) if actor==-1 else np.zeros(len(xyz),bool);scale,q,neighbor=local_geometry(xyz,.12 if actor==-1 else .04,ground=actor==-1)
  if 'identity_probability' not in g:g['identity_probability']=np.tile([.1,.1,.8],(len(xyz),1)).astype(np.float32);g['identity_evidence_status']=np.zeros(len(xyz),np.int64)
  g.update(scales=scale,quaternion=q,neighbor=neighbor+offset,owner=np.full(len(xyz),actor,np.int64),road=isroad,observed_rgb=g['rgb'].copy(),reference_rgb=np.zeros_like(g['rgb']),reference_confidence=np.zeros(len(xyz),np.float32));parts.append(g);offset+=len(xyz)
 return {key:np.concatenate([g[key] for g in parts]) for key in parts[0]}
seedA=seed_from_raw(raw_fields);seedB=seed_from_raw(new_fields)
for frame in frames:
 row=frame['row'];K=np.array(row['intrinsic']);camera=np.array(row['world_from_camera']);depth=frame['depth'];uv,z,good=project(seedA['xyz'][seedA['road']],K,camera,depth.shape);ids=np.flatnonzero(good);yy,xx=uv[ids,1],uv[ids,0];ok=(depth[yy,xx]>0)&(np.abs(depth[yy,xx]-z[ids])<.2)&~frame['guard'][yy,xx];roadmask=np.zeros_like(frame['guard']);roadmask[yy[ok],xx[ok]]=True;roadmask=binary_dilation(roadmask,iterations=2)&~frame['guard']
 cert=np.zeros_like(roadmask)
 if row['split']=='train':
  uv,z,good=project(certified_world,K,camera,depth.shape);ids=np.flatnonzero(good);yy,xx=uv[ids,1],uv[ids,0];ok=(depth[yy,xx]>0)&(np.abs(depth[yy,xx]-z[ids])<.15)&~frame['guard'][yy,xx];cert[yy[ok],xx[ok]]=True
 roadmask|=cert
 for output in [A,B]:np.savez_compressed(output/row['supervision'],masks=frame['masks'],mask_known=frame['mask_known'],mask_valid=frame['mask_valid'],depth=depth,road=roadmask,road_certified=cert)
def finalize(output,seed,method):
 ids=np.flatnonzero(seed['road']);samples=np.zeros((len(ids),len(frames),3),np.float32);visible=np.zeros((len(ids),len(frames)),bool)
 for i,frame in enumerate(frames):
  if frame['row']['split']!='train':continue
  row=frame['row'];uv,z,good=project(seed['xyz'][ids],np.array(row['intrinsic']),np.array(row['world_from_camera']),frame['depth'].shape);jj=np.flatnonzero(good);yy,xx=uv[jj,1],uv[jj,0];ok=(frame['depth'][yy,xx]>0)&(np.abs(frame['depth'][yy,xx]-z[jj])<.2)&~frame['guard'][yy,xx];jj,yy,xx=jj[ok],yy[ok],xx[ok];samples[jj,i]=frame['rgb'][yy,xx]/255.;visible[jj,i]=True
 ref,conf=clean_reference(samples,visible);seed['reference_rgb'][ids]=ref;seed['reference_confidence'][ids]=conf;seed['rgb'][ids[conf>0]]=ref[conf>0]
 if not (conf>0).any():raise ValueError('no repeated bright road reference')
 validate_seed(seed,k)
 if np.any(seed['source_raw_row']%5==0) or any(rows[j]['split']!='train' for j in np.unique(seed['source_frame'])):raise ValueError('withheld measurement leaked into seed')
 np.savez_compressed(output/'seed.npz',**seed);os.link(r/'data_r2/trajectory.npz',output/'trajectory.npz')
 manifest={key:m[key] for key in ['schema','units','actor_ids','frames','source_scene','annotations','original_world_origin_m']};manifest.update(learned_checkpoint_used=False,reference_train_only=True,initialization=method,BUILD_raw_row_rule='rowmod5 !=0 andtrain46; raw101/113/125 never BUILD',shared_masks='ownership_controls_r11/shared_masks_r14; actual native SAM3 video train46 and isolated QUERY images, with explicit originalDEVAfallback',reference_valid_points=int((conf>0).sum()),excluded_unseeded_tracks=m['excluded_unseeded_tracks'],test_geometry_or_RGB_used_to_build_seed=False)
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return dict(gaussians=len(seed['xyz']),references=int((conf>0).sum()),Actor_gaussians=int((seed['owner']>=0).sum()))
resultA=finalize(A,seedA,'raw OBB candidate allocation, matched shared masks, no learned checkpoint');resultB=finalize(B,seedB,'raw SAM+actual depth+independent measured3D motion, uncertain pool retained outside rendering, no learned checkpoint')
cfg=Config(geometry_steps=15000,material_steps=7500,joint_steps=2500,instance_balanced_supervision=True,outside_fullscene_visibility=True,instance_balanced_visible_identity=True);cfg.weights.mask_outside=.5;cfg.weights.road_identity=.5;cfg.validate();(root/'common_config.json').write_text(json.dumps(asdict(cfg),indent=2))
summary=dict(status='RAW_MATCHED_DATA_BUILT_REQUIRES_BODY_ROAD_GATE',A=resultA,B=resultB,C='identical B data and initialization; staged correctable owner at1500',classification=classification,shared_input_changes='all3 same SAM3/DEVAfallback bundle, actual BUILD-only depth/reference, 20percent raw return QUERY split, fixed losses and steps',
 uncertain_pool_is_not_background=True,probabilities_are_evidence_scores_not_calibrated_ground_truth=True,independent_motion_source='7074 actual RGB-track/LiDAR trajectories; ambiguous stays unknown',no_height_crop=True,no_largest_cluster_only=True,seconds=time.monotonic()-start)
(root/'initialization_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

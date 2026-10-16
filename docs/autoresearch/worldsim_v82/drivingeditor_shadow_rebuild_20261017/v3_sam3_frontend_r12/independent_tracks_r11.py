"""RGB forward/backward point tracks lifted ONLY by actual BUILD LiDAR returns."""
from pathlib import Path
import json,time
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
cv2.setNumThreads(2)
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_controls_r11';out=root/'independent_tracks';out.mkdir(exist_ok=False)
if not (root/'raw_measurements/manifest.json').exists():raise RuntimeError('raw cache incomplete')
m=json.loads((r/'data_r2/manifest.json').read_text());trajectory=np.load(r/'data_r2/trajectory.npz');train=[i for i,x in enumerate(m['frames']) if x['split']=='train'];targets=[5,6,12,18,25,26,37];rng=np.random.default_rng(0);start=time.monotonic()
cache={};images={};masks={}
def frame(i):
 if i not in cache:
  row=m['frames'][i];p=np.load(root/'raw_measurements'/(row['frame_id']+'.npz'));d={k:p[k] for k in ['world','raw_row','uv','pixel','camera_z','in_image','candidate_owner','BUILD']}
  ids=np.flatnonzero(d['BUILD']&d['in_image']&(d['camera_z']<80));pixel=d['pixel'][ids].astype(np.int64);bins=pixel[:,1]*480+pixel[:,0];z=np.full(320*480,np.inf);np.minimum.at(z,bins,d['camera_z'][ids]);ids=ids[d['camera_z'][ids]<=z[bins]+.1]
  # Do NOT use raw cache ALL-return visibility; withheld returns cannot influence BUILD.
  d['ids']=ids;d['tree']=cKDTree(d['uv'][ids]);cache[i]=d
  images[i]=np.asarray(Image.open(r/'data_r2'/row['rgb']).convert('L'));q=np.load(root/'shared_sam_masks_r2'/(row['frame_id']+'.npz'));masks[i]=(q['masks'],q['mask_known'])
 return cache[i]
worlds=[];rawrows=[];frames=[];initialactors=[];residuals=[];probabilities=[];semantic=[];report=[]
anchors=sorted(set(list(range(0,len(train)-4,4))+[len(train)-5]))
for a in anchors:
 indices=[train[a],train[a+2],train[a+4]];data=[frame(i) for i in indices];source=data[0];ids=source['ids'][np.isin(source['candidate_owner'][source['ids']],targets)]
 if len(ids)>1500:ids=ids[rng.choice(len(ids),1500,replace=False)]
 points=source['uv'][ids].astype(np.float32).reshape(-1,1,2);current=points;valid=np.ones(len(ids),bool);matches=[ids];positions=[source['world'][ids]];source_owner=source['candidate_owner'][ids];semantic_rows=[]
 for j in range(2):
  nxt,status,error=cv2.calcOpticalFlowPyrLK(images[indices[j]],images[indices[j+1]],current,None,winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
  back,back_status,_=cv2.calcOpticalFlowPyrLK(images[indices[j+1]],images[indices[j]],nxt,None,winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
  fb=np.linalg.norm((back-current).reshape(-1,2),axis=-1);uv=nxt.reshape(-1,2);distance,near=data[j+1]['tree'].query(uv,workers=1);nearest=data[j+1]['ids'][near]
  valid&=(status[:,0]>0)&(back_status[:,0]>0)&(fb<.75)&(distance<1.25)&np.isfinite(uv).all(-1)&(error[:,0]<25)
  matches.append(nearest);positions.append(data[j+1]['world'][nearest]);current=nxt
 accepted=np.flatnonzero(valid);xyz=np.stack(positions,1)[accepted];rr=np.stack([data[j]['raw_row'][matches[j]] for j in range(3)],1)[accepted];owner=source_owner[accepted]
 if not len(accepted):report.append(dict(anchor=indices[0],queries=len(ids),accepted=0));continue
 local=np.zeros_like(xyz);sem=np.full((len(xyz),3),-1,np.int8)
 for actor in targets:
  take=owner==actor
  for j,index in enumerate(indices):
   pose=trajectory['poses'][index,actor];local[take,j]=(xyz[take,j]-pose[:3,3])@pose[:3,:3]
   px=data[j]['pixel'][matches[j][accepted[take]]].astype(np.int64);mask,known=masks[index]
   if known[actor]:sem[take,j]=mask[px[:,1],px[:,0],actor].astype(np.int8)
 world_res=np.median(np.linalg.norm(xyz-np.median(xyz,axis=1)[:,None],axis=-1),axis=1);actor_res=np.median(np.linalg.norm(local-np.median(local,axis=1)[:,None],axis=-1),axis=1)
 # Identifiability is assessed from the supplied poses, but neither 3D trajectory is generated with a model/pose warp.
 spans=np.zeros(len(xyz))
 for actor in targets:
  take=owner==actor;canonical=local[take].mean(1);pred=[]
  for index in indices:
   pose=trajectory['poses'][index,actor];pred.append(canonical@pose[:3,:3].T+pose[:3,3])
  pred=np.stack(pred,1);spans[take]=np.linalg.norm(pred[:,-1]-pred[:,0],axis=-1)
 depth=np.stack([data[j]['camera_z'][matches[j][accepted]] for j in range(3)],1).mean(1);noise=.04+.001*depth
 likelihood=np.exp(-.5*np.stack([world_res,actor_res],-1)**2/noise[:,None]**2);ratio=np.abs(np.log(np.maximum(likelihood[:,0],1e-12)/np.maximum(likelihood[:,1],1e-12)))
 decisive=(spans>np.maximum(.15,2*noise))&(np.minimum(world_res,actor_res)<noise)&(ratio>np.log(3))
 prob=np.full((len(xyz),3),[.1,.1,.8],np.float32);prob[decisive,:2]=likelihood[decisive]/likelihood[decisive].sum(-1,keepdims=True)*.95;prob[decisive,2]=.05
 worlds.append(xyz);rawrows.append(rr);frames.append(np.broadcast_to(indices,(len(xyz),3)).copy());initialactors.append(owner);residuals.append(np.stack([world_res,actor_res,spans,noise],-1));probabilities.append(prob);semantic.append(sem)
 rec=dict(anchor=indices[0],frames=indices,queries=len(ids),accepted=len(xyz),static_supported=int((prob[:,0]>.7).sum()),Actor_supported=int((prob[:,1]>.7).sum()),ambiguous=int((prob[:,2]>.5).sum()));report.append(rec);print(json.dumps(rec),flush=True)
if not worlds:raise RuntimeError('No independently tracked actual3D correspondences; do not fabricate motion evidence')
np.savez_compressed(out/'actual_tracks.npz',world=np.concatenate(worlds),frame_index=np.concatenate(frames),raw_lidar_row=np.concatenate(rawrows),candidate_actor=np.concatenate(initialactors),residual_world_actor_span_noise=np.concatenate(residuals),probability_static_actor_unknown=np.concatenate(probabilities),semantic_observation=np.concatenate(semantic))
summary=dict(status='INDEPENDENT_RGB_TRACK_AND_ACTUAL_LIDAR_COMPLETE',tracks=sum(len(x) for x in worlds),anchors=report,
 method='OpenCV pyramidal LK forward/back0.75px + actual BUILD-return match1.25px at3 observed train times; world and inverse-Actor residuals measured independently',
 no_Gaussian_motion_used=True,pose_only_evaluates_hypothesis_not_correspondence=True,no_inferred_depth_or_3D_filled_positions=True,withheld_returns_and_test_RGB_never_used=True,
 unknown_policy='insufficient pose separation or conflicting/noisy hypotheses remain unknown; optical motion alone is never car-body evidence',
 sigma_m='.04+.001*camera_depth',decision='pose surface displacement >max(.15,2sigma), min3D residual<sigma and likelihood ratio>3, elseunknown',
 point_identity_caveat='sparse LiDAR returns are surface samples near tracked RGB features, not exact repeatedly fired physical rays; raw frame/row provenance retained',seconds=time.monotonic()-start)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('INDEPENDENT_TRACKS_COMPLETE',flush=True)

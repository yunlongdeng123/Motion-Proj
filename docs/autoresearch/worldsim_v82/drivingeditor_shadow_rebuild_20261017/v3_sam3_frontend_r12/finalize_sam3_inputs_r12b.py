from pathlib import Path
import sys,json,time,numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';out=root/'shared_sam3_masks_r2';out.mkdir(exist_ok=False)
m=json.loads((R/'data_r2/manifest.json').read_text());rows=m['frames'];targets=[5,6,12,18,25,26,37];start=time.monotonic();train=[r for r in rows if r['split']=='train'];query=[r for r in rows if r['split']!='train']
raw=root/'shared_sam3_masks_r1/native_predictions'
assert len(list(raw.glob('*.npz')))==49
# Track identity is associated with GT actors using only the TRAIN-pass evidence.
# Each frame also requires independent same-frame raw-LiDAR support. Boxes provide
# candidate IDs only; a box by itself never creates a foreground mask.
def evidence(row):
 actual=np.load(R/'ownership_controls_r11/raw_measurements'/(row['frame_id']+'.npz'));old=np.load(R/'data_r2'/row['supervision']);pred=np.load(raw/(row['frame_id']+'.npz'));ij=actual['pixel'].astype(np.int64)
 permitted=actual['BUILD'] if row['split']=='train' else actual['QUERY'];ids=np.flatnonzero(permitted&actual['in_image']);bins=ij[ids,1]*480+ij[ids,0];depth=np.full(320*480,np.inf);np.minimum.at(depth,bins,actual['camera_z'][ids]);ids=ids[actual['camera_z'][ids]<=depth[bins]+.1]
 score=np.zeros((7,len(pred['masks'])),np.float64);details={}
 for ai,actor in enumerate(targets):
  body=old['masks'][...,actor];other=(old['masks']&old['mask_known'][None,None]).copy();other[...,actor]=False;other=other.any(-1)
  cand=ids[actual['candidate_owner'][ids]==actor];uv=ij[cand]
  for j,mask in enumerate(pred['masks']):
   hit=int(mask[uv[:,1],uv[:,0]].sum()) if len(uv) else 0;iou=int((body&mask).sum())/max(1,int((body|mask).sum()));foreign=int((mask&other).sum())/max(1,int(mask.sum()))
   # IoU is a cross-check, not a hard clipping domain; the original body labels
   # are separate predicted evidence, not hand-annotated ownership ground truth.
   score[ai,j]=2*hit/max(8,len(cand))+iou-2*foreign
   details[(ai,j)]=dict(actual_frontmost_candidate_hits=hit,actual_candidate_returns=len(cand),IoU_original_body=iou,overlap_other_original_body=foreign,prediction_score=float(pred['scores'][j]),native_tracker_id=int(pred['tracker_ids'][j]))
 return actual,old,pred,score,details
track_votes={}
for row in train:
 _,_,pred,score,details=evidence(row)
 if score.shape[1]:
  ii,jj=linear_sum_assignment(-score)
  for ai,j in zip(ii,jj):
   d=details[(int(ai),int(j))]
   if d['actual_frontmost_candidate_hits']>=8 and d['overlap_other_original_body']<.1 and score[ai,j]>.25:
    track_votes.setdefault(int(pred['tracker_ids'][j]),np.zeros(7))[ai]+=max(1,d['actual_frontmost_candidate_hits'])
mapping={track_id:int(np.argmax(votes)) for track_id,votes in track_votes.items() if votes.max()/max(1,votes.sum())>=.8}
records=[]
for i,row in enumerate(rows):
 _,old,pred,score,details=evidence(row);mask=old['masks'].copy();known=old['mask_known'].copy();assignment={}
 if score.shape[1]:
  adjusted=score.copy()
  if row['split']=='train':
   for j,tid in enumerate(pred['tracker_ids']):
    if int(tid) in mapping:
     for ai in range(7):
      if mapping[int(tid)]!=ai:adjusted[ai,j]=-100
  ii,jj=linear_sum_assignment(-adjusted);assignment={int(ai):int(j) for ai,j in zip(ii,jj) if adjusted[ai,j]>.25}
 meta=[];raw_selected=np.zeros((320,480,7),bool)
 for ai,actor in enumerate(targets):
  d=dict(actor=actor,raw_track_id=m['actor_ids'][actor],prediction_available=ai in assignment)
  accepted=False
  if ai in assignment:
   j=assignment[ai];candidate=pred['masks'][j];raw_selected[...,ai]=candidate;d.update(details[(ai,j)])
   accepted=d['actual_frontmost_candidate_hits']>=8 and d['overlap_other_original_body']<.1 and d['prediction_score']>=.5 and (not old['masks'][...,actor].any() or d['IoU_original_body']>=.45)
  if old['mask_known'][actor] and not old['masks'][...,actor].any():
   mask[...,actor]=False;known[actor]=True;d['used']='original_known_empty';d['positive_prediction_conflict']=bool(accepted)
  elif accepted:
   mask[...,actor]=candidate;known[actor]=True;d['used']='SAM3_native_video' if row['split']=='train' else 'SAM3_isolated_QUERY_image'
  elif old['mask_known'][actor]:d['used']='original_DEVA_fallback'
  else:mask[...,actor]=False;known[actor]=False;d['used']='unknown_excluded_from_mask_loss'
  d.update(accepted=bool(accepted),used_pixels=int(mask[...,actor].sum()));meta.append(d)
 # Native masks are exclusive; any conflict with a context original label is
 # explicitly unresolved, never hidden by mask clipping.
 # Reject conflicting proposals jointly. Restoring one original label can
 # reveal another conflict, so a one-pass fallback is insufficient. Original
 # labels are preserved and no overlapping pixel is silently turned background.
 for iteration in range(len(targets)+1):
  overlap=mask.sum(-1)>1
  if not overlap.any():break
  changed=False
  for ai,actor in enumerate(targets):
   if (mask[...,actor]&overlap).any() and meta[ai]['used'].startswith('SAM3_'):
    mask[...,actor]=old['masks'][...,actor];known[actor]=old['mask_known'][actor];meta[ai]['used']='original_label_after_unresolved_overlap';meta[ai]['overlap_fallback_iteration']=iteration+1;meta[ai]['used_pixels']=int(mask[...,actor].sum());changed=True
  if not changed:raise ValueError('Original visible labels themselves conflict; do not fabricate disjoint masks')
 assert not (mask.sum(-1)>1).any()
 np.savez_compressed(out/(row['frame_id']+'.npz'),SAM_prediction=raw_selected,masks=mask,mask_known=known)
 rgb=Image.open(R/'data_r2'/row['rgb']).convert('RGB');draw=ImageDraw.Draw(rgb)
 for ai,actor in enumerate(targets):
  body=mask[...,actor];edge=body.copy();edge[1:]=body[1:]&~body[:-1];yy,xx=np.where(edge)
  for x,y in zip(xx,yy):draw.point((int(x),int(y)),fill=(255,70,40))
  if body.any():yy,xx=np.where(body);draw.text((int(xx.mean()),int(yy.min())),str(actor),fill=(255,255,0))
 rgb.save(out/(row['frame_id']+'_overlay.png'));rec=dict(frame_id=row['frame_id'],split=row['split'],actors=meta);records.append(rec);(out/(row['frame_id']+'_record.json')).write_text(json.dumps(rec,indent=2));print(json.dumps({'phase':'shared_input','done':i+1,'used':[d['used'] for d in meta]}),flush=True)
summary=dict(status='REAL_SAM3_COMPLETE_REQUIRES_ROOT_REVIEW_BEFORE_TRAINING',model='SAM3_native_PT',source_commit='7f473cfd983d5eb43040ff608f66bff0d7bdcb53',text_prompt='vehicle',native_video_tracking_train46=True,QUERY_sessions_isolated=True,train_tracker_to_actor_index={str(k):targets[v] for k,v in mapping.items()},train_tracker_votes={str(k):v.tolist() for k,v in track_votes.items()},frames=records,all_three_controls_use_identical_bundle=True,original_DEVA_fallbacks_explicit=True,prediction_is_not_manual_GT=True,seconds=time.monotonic()-start,seconds_scope='CPU finalization only; native GPU predictions reused without inference',native_predictions_directory=str(raw),prior_assembly_failure_preserved=True,joint_original_label_fallback=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('SHARED_SAM3_COMPLETE',flush=True)

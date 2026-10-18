"""CPU-only full-frame native-proposal and raw-measurement coverage inventory.

Never declares segmentation ground truth, never changes original/R15 masks.
All annotations include stationary vehicles; actual projected LiDAR supports
association checks only. Full native pixels remain available for visual review.
"""
from pathlib import Path
import os,json,datetime,collections,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parent;ROOT=E.parent;R=ROOT/'shadow_rebuild_20261017_v3'
SRC=ROOT/'inputs/scene0008_089_137';DEST=E/'all_vehicle_coverage'
NATIVE=R/'ownership_controls_r12_sam3/shared_sam3_masks_r1/native_predictions'
def write(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 global DEST,NATIVE
 ap=argparse.ArgumentParser();ap.add_argument('--native-root',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--all49-training',action='store_true');args=ap.parse_args()
 if args.native_root:NATIVE=args.native_root
 if args.output:DEST=args.output
 assert not DEST.exists(),'existing coverage audit preserved; do not overwrite'
 DEST.mkdir();(DEST/'frames').mkdir();tracks={};frames=[];votes=collections.defaultdict(collections.Counter)
 old=json.loads((R/'backend_restore_r15/data/manifest.json').read_text())
 oldids=old['actor_ids'];frame_roles={r['frame_id']:r['split'] for r in old['frames']}
 for raw in range(89,138):
  fid=f'0008{raw:03d}';ann=json.loads((SRC/'annotations'/f'{fid}.json').read_text())
  labels=[a for a in ann['laser_labels'] if a['type']=='TYPE_VEHICLE']
  lidar=np.fromfile(SRC/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3].astype(np.float64)
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in s}
  cam=np.c_[lidar,np.ones(len(lidar))]@cal['Tr_velo_to_cam_0'].reshape(3,4).T
  keep=np.flatnonzero(np.isfinite(cam).all(1)&(cam[:,2]>.01));q=np.c_[cam[keep],np.ones(len(keep))]@cal['P0'].reshape(3,4).T
  uv=np.floor(q[:,:2]/q[:,2:3]*.25+.5).astype(int);inside=(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)
  keep=keep[inside];uv=uv[inside];depth=cam[keep,2];linear=uv[:,1]*480+uv[:,0]
  nearest=np.full(480*320,np.inf);np.minimum.at(nearest,linear,depth);front=depth<=nearest[linear]+.5
  keep=keep[front];uv=uv[front];xyz=lidar[keep]
  z=np.load(NATIVE/(fid+'.npz'));masks=z['masks']; tids=z['tracker_ids'];scores=z['scores'];union=masks.any(0)
  if masks.shape[1:]!=(320,480):
   masks=np.stack([np.asarray(Image.fromarray(m).resize((480,320),Image.Resampling.NEAREST)) for m in masks]);union=masks.any(0)
  ownership=np.full(len(xyz),-1,int);candidate=[]
  for i,a in enumerate(labels):
   b=a['box'];center=np.array([b['center_x'],b['center_y'],b['center_z']]);h=b['heading'];c,s=np.cos(h),np.sin(h)
   rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   hit=(((xyz-center)@rot).__abs__()<=np.array([b['length'],b['width'],b['height']])/2).all(1)
   # Ambiguous overlapping 3D boxes remain unassigned.
   ownership[hit&(ownership==-1)]=i
   candidate.append(hit)
   count=int(hit.sum());covered=int(union[uv[hit,1],uv[hit,0]].sum())
   speed=float(np.linalg.norm([a.get('metadata',{}).get('speed_x',0),a.get('metadata',{}).get('speed_y',0)]))
   tr=tracks.setdefault(a['id'],dict(track_id=a['id'],R15_actor=oldids.index(a['id']) if a['id'] in oldids else None,frames=0,front_LiDAR_points=0,covered_points=0,mask_hits=0,speeds=[],visible_frames=[]))
   tr['frames']+=1;tr['front_LiDAR_points']+=count;tr['covered_points']+=covered;tr['speeds'].append(speed)
   if count:tr['visible_frames'].append(dict(frame_id=fid,points=count,covered=covered))
  proposals=[]
  for j,(tid,score,mask) in enumerate(zip(tids,scores,masks)):
   phit=mask[uv[:,1],uv[:,0]];counts={a['id']:int((phit&hit).sum()) for a,hit in zip(labels,candidate)};counts={t:n for t,n in counts.items() if n}
   ordered=sorted(counts.items(),key=lambda x:-x[1]);best=ordered[0] if ordered else None
   key=('train:'+str(int(tid))) if args.all49_training or frame_roles[fid]=='train' else fid+':'+str(int(tid))
   votes[key].update(counts)
   if best:tracks[best[0]]['mask_hits']+=best[1]
   proposals.append(dict(native_index=j,tracker_id=int(tid),association_key=key,score=float(score),pixels=int(mask.sum()),raw_LiDAR_votes=ordered))
  rgb=np.asarray(Image.open(SRC/'image/image_0'/f'{fid}.png').resize((480,320),Image.Resampling.LANCZOS)).copy()
  overlay=rgb.copy()
  for j,mask in enumerate(masks):
   color=np.array([(int(tids[j])*71+30)%220+30,(int(tids[j])*43+70)%220+30,(int(tids[j])*103+20)%220+30])
   overlay[mask]=np.rint(.7*rgb[mask]+.3*color).astype(np.uint8)
  im=Image.fromarray(overlay);draw=ImageDraw.Draw(im)
  for p,mask in zip(proposals,masks):
   ys,xs=np.where(mask)
   if len(xs):draw.text((int(xs.mean()),max(0,int(ys.min())-9)),str(p['tracker_id']),fill='yellow',stroke_width=1,stroke_fill='black')
  panel=Image.new('RGB',(960,320));panel.paste(Image.fromarray(rgb),(0,0));panel.paste(im,(480,0));panel.save(DEST/'frames'/(fid+'.png'))
  frames.append(dict(frame_id=fid,native_proposals=proposals,vehicle_annotations=len(labels),union_pixels=int(union.sum()),overlap_pixels=int((masks.sum(0)>1).sum())))
  write(DEST/'progress.json',dict(frames=len(frames),total=49,GPU_used=False))
 for tr in tracks.values():
  tr['speed_median_m_s']=float(np.median(tr.pop('speeds')))
  tr['stationary_by_median_lt_0p5']=tr['speed_median_m_s']<.5
  tr['coverage_ratio_on_actual_front_LiDAR']=tr['covered_points']/tr['front_LiDAR_points'] if tr['front_LiDAR_points'] else None
 assoc={key:sorted(v.items(),key=lambda x:-x[1]) for key,v in votes.items()}
 report=dict(status='FULL49_CPU_COVERAGE_INVENTORY_REQUIRES_ASSOCIATION_AND_VISUAL_REVIEW',frames=frames,vehicles=list(tracks.values()),proposal_track_votes=assoc,
  target='all visible vehicles including stationary; full RGB retained',R15_training_masks_used=False,GPU_used=False,training_updates=0,
  source_native_prompt='vehicle',native_prediction_source=str(NATIVE),unknown_not_converted_to_background=True,
  readiness=False,human_verdict=None,all_vehicle_initialization_complete=False,
  caveat='Projected LiDAR is a sparse coverage probe, not proof of full pixel recall or clean-road ground truth.')
 write(DEST/'coverage.json',report)
 write(E/'all_actor_queue_status.json',dict(state='COVERAGE_INVENTORY_COMPLETE_PREPARING_FULL_IDENTITY_AND_INITIALIZATION',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),report=str(DEST/'coverage.json'),all_vehicle_training_updates=0,automatic_30k_authorized=True))
 print(json.dumps(dict(vehicles=len(tracks),with_front_LiDAR=sum(t['front_LiDAR_points']>0 for t in tracks.values()),stationary=sum(t['stationary_by_median_lt_0p5'] for t in tracks.values()),frames=49)))
if __name__=='__main__':main()

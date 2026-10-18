"""Native instance-interactive masks from full RGB and measurement prompts.

Unlike semantic exemplar prompting, this API returns masks for one prompted
instance. All outputs still require visibility, identity and real-seed review.
"""
from pathlib import Path
import os,sys,json,datetime,traceback,fcntl
E=Path(__file__).resolve().parent;OUT=E/'all_vehicle_instance_prompt49'
SRC=E.parent/'inputs/scene0008_089_137'
def write(p,x):
 t=p.with_suffix('.partial');t.write_text(json.dumps(x,indent=2));t.replace(p)
def status(s,**kw):write(E/'all_vehicle_instance_prompt_status.json',dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0,**kw))
def main():
 assert not OUT.exists(),'preserve partial outputs; do not blindly restart'
 lock=(E/'all_actor_secondary_gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 claim=E/'all_actor_gpu_claim.json';assert not claim.exists() or json.loads(claim.read_text()).get('active') is False
 sys.path.insert(0,str(E/'direct_implementation'));from priority import inspect_priority
 p=inspect_priority();assert not p['streetcrafter_processes'] and p['query_error'] is None
 for proc in p['other_compute_processes']:
  cmd=Path('/proc/'+str(proc['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert proc['pid']==6099 and 'r15_baseline_owner5_20261018' in cmd and int(proc['used_memory_mib'])<16000,proc
 write(claim,dict(active=True,pid=os.getpid(),kind='SAM3_INSTANCE_INPUT_PROMPTS_ONLY',allocator_cap_GiB=13,coexists_only_with_verified_primary_pid=6099,user_authorized_parallel=True,training_updates=0))
 try:
  import torch,numpy as np
  from PIL import Image
  torch.set_num_threads(2);torch.manual_seed(0);total=torch.cuda.get_device_properties(0).total_memory
  assert total>=31*1024**3;torch.cuda.set_per_process_memory_fraction(13*1024**3/total,0)
  sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3')
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  review=json.loads((E/'all_vehicle_association_review/review.json').read_text());byframe={f'0008{r:03d}':[] for r in range(89,138)}
  for v in review['vehicles']:
   for row in v['frames']:
    if row['box_intersects_image']:byframe[row['frame_id']].append(dict(row,track_id=v['raw_track_id'],actor=v['name']))
  OUT.mkdir();(OUT/'native_predictions').mkdir()
  write(OUT/'input_contract.json',dict(API='Sam3Image.predict_inst',box_prompts='projected real 3D annotations, clipped to full image extent only',positive_points='up to3 actual depth-visible LiDAR returns in that annotation; not initialization point filtering',image_source=str(SRC/'image/image_0'),all_full_RGB=True,frames=49,box_pixels_never_filled=True,segmentation_is_prediction_not_ground_truth=True,no_R15_inputs=True,training_updates=0))
  status('INSTANCE_PROMPT_MODEL_LOADING',frames=0)
  model=build_sam3_image_model(checkpoint_path='/root/autodl-tmp/models/idsplat/sam3/sam3.pt',bpe_path='/root/autodl-tmp/idsplat/asset_sources/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',load_from_HF=False,enable_inst_interactivity=True,compile=False)
  processor=Sam3Processor(model);records=[]
  with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   for fid,queries in byframe.items():
    ann={a['id']:a for a in json.loads((SRC/'annotations'/(fid+'.json')).read_text())['laser_labels']}
    xyz=np.fromfile(SRC/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3].astype(float)
    cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/(fid+'.txt')).read_text().splitlines() if ':' in s}
    cam=np.c_[xyz,np.ones(len(xyz))]@cal['Tr_velo_to_cam_0'].reshape(3,4).T
    q=np.c_[cam,np.ones(len(cam))]@cal['P0'].reshape(3,4).T;uv=q[:,:2]/np.maximum(q[:,2:3],.001)
    vis=(cam[:,2]>.01)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1280)
    idx=np.flatnonzero(vis);pix=np.floor(uv[idx]+.5).astype(int).clip([0,0],[1919,1279]);bins=(pix[:,1]//2)*960+pix[:,0]//2
    nearest=np.full(960*640,np.inf);np.minimum.at(nearest,bins,cam[idx,2]);idx=idx[cam[idx,2]<=nearest[bins]+.5]
    state=processor.set_image(Image.open(SRC/'image/image_0'/(fid+'.png')).convert('RGB'))
    masks=[];scores=[];query_indices=[];rows=[];offset=0
    for qi,row in enumerate(queries):
     b=ann[row['track_id']]['box'];h=b['heading'];c,s=np.cos(h),np.sin(h);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
     hit=((((xyz[idx]-[b['center_x'],b['center_y'],b['center_z']])@rot).__abs__())<=np.array([b['length'],b['width'],b['height']])/2).all(1)
     candidates=idx[hit];points=None;chosen=np.array([],int)
     if len(candidates):
      target=np.median(uv[candidates],axis=0);order=np.argsort(np.linalg.norm(uv[candidates]-target,axis=1));chosen=candidates[order[np.linspace(0,min(len(order)-1,10),min(3,len(order))).astype(int)]];points=uv[chosen].astype(np.float32)
     bb=np.array(row['projected_box'],np.float32);bb[[0,2]]=bb[[0,2]].clip(0,1919);bb[[1,3]]=bb[[1,3]].clip(0,1279)
     ms,sc,low=model.predict_inst(state,point_coords=points,point_labels=np.ones(len(points),np.int32) if points is not None else None,box=bb,multimask_output=True,normalize_coords=True)
     ms=ms.astype(bool);assert ms.shape[1:]==(1280,1920)
     masks.extend(ms);scores.extend(sc);query_indices.extend([qi]*len(ms))
     rows.append(dict(**row,prompt_box_xyxy=bb.tolist(),positive_point_source_indices=chosen.tolist(),positive_point_pixels=points.tolist() if points is not None else [],proposal_offset=offset,proposal_count=len(ms),predicted_mask_quality=sc.tolist(),visibility_confirmed=False))
     offset+=len(ms)
    np.savez_compressed(OUT/'native_predictions'/(fid+'.npz'),masks=np.stack(masks),scores=np.array(scores,np.float32),query_indices=np.array(query_indices,np.int32))
    write(OUT/'native_predictions'/(fid+'.json'),dict(frame_id=fid,queries=rows));del state,ms,low
    records.append(dict(frame_id=fid,prompts=len(rows),proposals=len(masks)));status('INSTANCE_PROMPTS_RUNNING',frames=len(records),total=49,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3);print(json.dumps(records[-1]),flush=True)
  write(OUT/'inference_receipt.json',dict(status='COMPLETE_NATIVE49_INSTANCE_MASKS_REVIEW_REQUIRED',frames=49,records=records,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,training_updates=0,human_verdict=None));status('INSTANCE_PROMPTS_COMPLETE_REVIEW_REQUIRED',frames=49)
 finally:
  write(claim,dict(active=False,pid=os.getpid(),released_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0));lock.close()
if __name__=='__main__':
 try:main()
 except BaseException:status('FAILED_INSTANCE_INPUT_PROMPTS_PRESERVED',error=traceback.format_exc());raise

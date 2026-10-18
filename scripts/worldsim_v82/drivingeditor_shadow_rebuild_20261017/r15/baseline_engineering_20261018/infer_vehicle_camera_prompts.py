"""Finite instance segmentation of all actual FRONT camera vehicle labels.

Full-resolution RGB remains unchanged. Actual camera-track rectangles prompt
SAM3; rectangle pixels are never used as body labels or initialization points.
"""
from pathlib import Path
import os,sys,json,datetime,traceback,fcntl
E=Path(__file__).resolve().parent;OUT=E/'all_vehicle_camera_prompt49';SRC=E.parent/'inputs/scene0008_089_137/image/image_0'
def write(p,x):
 t=p.with_suffix('.partial');t.write_text(json.dumps(x,indent=2));t.replace(p)
def status(s,**kw):write(E/'all_vehicle_camera_prompt_status.json',dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0,**kw))
def main():
 assert not OUT.exists(),'preserve partial outputs; inspect before resuming remaining input frames'
 lock=(E/'all_actor_secondary_gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 claim=E/'all_actor_gpu_claim.json';assert json.loads(claim.read_text()).get('active') is False
 sys.path.insert(0,str(E/'direct_implementation'));from priority import inspect_priority
 p=inspect_priority();assert not p['streetcrafter_processes'] and p['query_error'] is None
 for proc in p['other_compute_processes']:
  cmd=Path('/proc/'+str(proc['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert proc['pid']==6099 and 'r15_baseline_owner5_20261018' in cmd and int(proc['used_memory_mib'])<16000,proc
 write(claim,dict(active=True,pid=os.getpid(),kind='SAM3_FULL_CAMERA_ID_INPUT_PROMPTS_ONLY',allocator_cap_GiB=13,coexists_only_with_verified_primary_pid=6099,user_authorized_parallel=True,training_updates=0))
 try:
  import torch,numpy as np
  from PIL import Image
  torch.set_num_threads(2);torch.manual_seed(0);total=torch.cuda.get_device_properties(0).total_memory;assert total>=31*1024**3
  torch.cuda.set_per_process_memory_fraction(13*1024**3/total,0)
  sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3')
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  raw=json.loads((E/'full_segment_vehicle_measurements.json').read_text());frames=raw['camera_frames'];assert len(frames)==49
  camera_ids=sorted({q['raw_camera_track_id'] for f in frames for q in f['vehicle_camera_labels']});assert len(camera_ids)==33
  OUT.mkdir();(OUT/'native_predictions').mkdir()
  write(OUT/'input_contract.json',dict(API='Sam3Image.predict_inst',camera='FRONT',actual_camera_vehicle_track_ids=camera_ids,camera_vehicle_id_count=len(camera_ids),all_full_RGB=True,frames=49,prompts='actual Waymo camera-label image rectangles; image-instance ID, not LiDAR box ID',box_pixels_never_filled=True,segmentation_is_model_prediction=True,no_R15_inputs=True,training_updates=0))
  status('CAMERA_INSTANCE_MODEL_LOADING',frames=0)
  model=build_sam3_image_model(checkpoint_path='/root/autodl-tmp/models/idsplat/sam3/sam3.pt',bpe_path='/root/autodl-tmp/idsplat/asset_sources/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',load_from_HF=False,enable_inst_interactivity=True,compile=False)
  processor=Sam3Processor(model);records=[]
  with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   for frame in frames:
    fid=f"0008{frame['raw_frame']:03d}";state=processor.set_image(Image.open(SRC/(fid+'.png')).convert('RGB'));masks=[];scores=[];queries=[];offset=0
    for row in frame['vehicle_camera_labels']:
     x,y,w,h=row['box_xywh'];bb=np.array([x-w/2,y-h/2,x+w/2,y+h/2],np.float32);bb[[0,2]]=bb[[0,2]].clip(0,1919);bb[[1,3]]=bb[[1,3]].clip(0,1279)
     assert bb[2]>bb[0] and bb[3]>bb[1]
     ms,sc,low=model.predict_inst(state,box=bb,multimask_output=True,normalize_coords=True)
     assert ms.shape==(3,1280,1920)
     masks.extend(ms.astype(bool));scores.extend(sc);queries.append(dict(**row,camera_actor_index=camera_ids.index(row['raw_camera_track_id']),prompt_box_xyxy=bb.tolist(),proposal_offset=offset,proposal_count=3,predicted_mask_quality=sc.tolist(),visibility_and_identity_review_passed=False));offset+=3
    np.savez_compressed(OUT/'native_predictions'/(fid+'.npz'),masks=np.stack(masks),scores=np.array(scores,np.float32));write(OUT/'native_predictions'/(fid+'.json'),dict(frame_id=fid,queries=queries))
    del state,ms,low
    records.append(dict(frame_id=fid,prompts=len(queries),proposals=len(masks)));status('CAMERA_INSTANCE_PROMPTS_RUNNING',frames=len(records),total=49,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3);print(json.dumps(records[-1]),flush=True)
  write(OUT/'inference_receipt.json',dict(status='COMPLETE_CAMERA33_NATIVE49_REQUIRES_REVIEW_AND_REAL_INITIALIZATION',frames=49,camera_vehicle_tracks=33,records=records,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,training_updates=0,human_verdict=None));status('CAMERA33_PROMPTS_COMPLETE_REVIEW_REQUIRED',frames=49)
 finally:
  write(claim,dict(active=False,pid=os.getpid(),released_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0));lock.close()
if __name__=='__main__':
 try:main()
 except BaseException:status('FAILED_CAMERA_INPUT_PROMPTS_PRESERVED',error=traceback.format_exc());raise

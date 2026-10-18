"""Finite native SAM3 real-image segmentation for projected vehicle candidates.

Projected measurement rectangles are prompts, never filled masks. Save model
proposals before association/visibility review. No training and no input edits.
"""
from pathlib import Path
import os,sys,json,datetime,traceback,fcntl
E=Path(__file__).resolve().parent;OUT=E/'all_vehicle_prompted49'
SRC=E.parent/'inputs/scene0008_089_137/image/image_0'
def write(p,x):
 t=p.with_suffix('.partial');t.write_text(json.dumps(x,indent=2));t.replace(p)
def status(s,**kw):write(E/'all_vehicle_prompt_status.json',dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0,**kw))
def main():
 assert not OUT.exists(),'preserve partial prompts; inspect before arranging remaining frames'
 lock=(E/'all_actor_secondary_gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 claim=E/'all_actor_gpu_claim.json'
 assert not claim.exists() or json.loads(claim.read_text()).get('active') is False
 sys.path.insert(0,str(E/'direct_implementation'));from priority import inspect_priority
 priority=inspect_priority();assert not priority['streetcrafter_processes'] and priority['query_error'] is None
 for proc in priority['other_compute_processes']:
  cmd=Path('/proc/'+str(proc['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert proc['pid']==6099 and 'r15_baseline_owner5_20261018' in cmd,proc
  assert int(proc['used_memory_mib'])<16000,proc
 write(claim,dict(active=True,pid=os.getpid(),kind='SAM3_GEOMETRIC_INPUT_PROMPTS_ONLY',allocator_cap_GiB=13,coexists_only_with_verified_primary_pid=6099,user_authorized_parallel=True,training_updates=0))
 try:
  import torch,numpy as np
  from PIL import Image
  torch.set_num_threads(2);torch.manual_seed(0)
  total=torch.cuda.get_device_properties(0).total_memory;assert total>=31*1024**3
  torch.cuda.set_per_process_memory_fraction(13*1024**3/total,0)
  sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3')
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  review=json.loads((E/'all_vehicle_association_review/review.json').read_text())
  byframe={f'0008{r:03d}':[] for r in range(89,138)}
  for v in review['vehicles']:
   for row in v['frames']:
    if row['box_intersects_image']:byframe[row['frame_id']].append(dict(row,track_id=v['raw_track_id'],actor=v['name']))
  OUT.mkdir();(OUT/'native_predictions').mkdir()
  write(OUT/'input_contract.json',dict(prompt='vehicle + projected raw annotation box per candidate',all_full_RGB=True,frames=49,mask_from_model=True,box_pixels_never_filled=True,GT_boxes_prompt_segmentation_not_ground_truth=True,no_R15_inputs=True,max_proposals_per_prompt=3,confidence_threshold=.15,training_updates=0))
  model=build_sam3_image_model(checkpoint_path='/root/autodl-tmp/models/idsplat/sam3/sam3.pt',bpe_path='/root/autodl-tmp/idsplat/asset_sources/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',load_from_HF=False,enable_inst_interactivity=False,compile=False)
  processor=Sam3Processor(model,confidence_threshold=.15)
  records=[]
  with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   text_features=model.backbone.forward_text(['vehicle'],device='cuda')
   for fid,queries in byframe.items():
    state=processor.set_image(Image.open(SRC/(fid+'.png')).convert('RGB'))
    masks=[];scores=[];track_indices=[];candidate_records=[];offset=0
    for qi,row in enumerate(queries):
     bb=np.array(row['projected_box']);bb[[0,2]]=bb[[0,2]].clip(0,1919);bb[[1,3]]=bb[[1,3]].clip(0,1279)
     if bb[2]<=bb[0] or bb[3]<=bb[1]:continue
     box=[float((bb[0]+bb[2])/3840),float((bb[1]+bb[3])/2560),float((bb[2]-bb[0])/1920),float((bb[3]-bb[1])/1280)]
     processor.reset_all_prompts(state);state['backbone_out'].update(text_features)
     out=processor.add_geometric_prompt(box,True,state)
     ms=out['masks'][:,0].cpu().numpy();sc=out['scores'].float().cpu().numpy();pb=out['boxes'].float().cpu().numpy()
     order=np.argsort(-sc)[:3]
     for j in order:masks.append(ms[j]);scores.append(sc[j]);track_indices.append(qi)
     candidate_records.append(dict(**row,prompt_box_cxcywh=box,proposal_offset=offset,proposal_count=len(order),model_boxes=[pb[j].tolist() for j in order],model_scores=[float(sc[j]) for j in order]))
     offset+=len(order)
    a=np.stack(masks) if masks else np.zeros((0,1280,1920),bool)
    np.savez_compressed(OUT/'native_predictions'/(fid+'.npz'),masks=a,scores=np.array(scores,np.float32),query_indices=np.array(track_indices,np.int32))
    write(OUT/'native_predictions'/(fid+'.json'),dict(frame_id=fid,queries=candidate_records))
    records.append(dict(frame_id=fid,prompts=len(queries),proposals=len(masks)))
    del state,out,a,ms
    status('REAL_GEOMETRIC_PROMPTS_RUNNING',frames=len(records),total=49,prompts=sum(x['prompts'] for x in records),peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
    print(json.dumps(records[-1]),flush=True)
  receipt=dict(status='COMPLETE_NATIVE49_PROMPTS_REQUIRES_VISIBILITY_AND_INITIALIZATION_REVIEW',frames=49,records=records,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,training_updates=0,human_verdict=None)
  write(OUT/'inference_receipt.json',receipt);status('REAL_GEOMETRIC_PROMPTS_COMPLETE_REVIEW_REQUIRED',frames=49)
 finally:
  write(claim,dict(active=False,pid=os.getpid(),released_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0));lock.close()
if __name__=='__main__':
 try:main()
 except BaseException:status('FAILED_INPUT_PROMPTS_PRESERVED_NO_RESTART',error=traceback.format_exc());raise

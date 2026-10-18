"""Finite real full-resolution49 SAM3 vehicle proposals, isolated inputs/outputs.

Explicit user-approved secondary GPU admission; cap own PyTorch allocator at
14GiB to protect the existing five-ID GN11. Does not alter its global lock.
"""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,datetime,traceback
E=Path(__file__).resolve().parent;ROOT=E.parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
OUT=E/'all_vehicle_native49';source=ROOT/'inputs/scene0008_089_137/image/image_0'
def write(p,x):
 tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def status(s,**kw):write(E/'all_actor_queue_status.json',dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0,**kw))
def main():
 assert not OUT.exists(),'preserve partial inference; no automatic restart'
 lock=(E/'all_actor_secondary_gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 claim=E/'all_actor_gpu_claim.json'
 if claim.exists():assert json.loads(claim.read_text()).get('active') is False
 sys.path.insert(0,str(E/'direct_implementation'));from priority import inspect_priority
 p=inspect_priority();assert not p['streetcrafter_processes'] and p['query_error'] is None,p
 for proc in p['other_compute_processes']:
  cmd=Path('/proc/'+str(proc['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert proc['pid']==6099 and 'r15_baseline_owner5_20261018' in cmd,proc
  assert int(proc['used_memory_mib'])<16000,'current job requires too much memory; defer inference'
 write(claim,dict(active=True,pid=os.getpid(),kind='SAM3_INPUT_PREPARATION_ONLY',allocator_cap_GiB=14,coexists_only_with_verified_primary_pid=6099,user_authorized_parallel=True,training_updates=0))
 try:
  import torch,numpy as np
  torch.set_num_threads(2);torch.manual_seed(0)
  total=torch.cuda.get_device_properties(0).total_memory
  assert total>=31*1024**3
  torch.cuda.set_per_process_memory_fraction(14*1024**3/total,0)
  sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3')
  from sam3.model.sam3_video_predictor import Sam3VideoPredictor
  OUT.mkdir();raw=OUT/'native_predictions';raw.mkdir();folder=OUT/'all49_RGB';folder.mkdir()
  ids=[f'0008{raw:03d}' for raw in range(89,138)]
  for i,fid in enumerate(ids):os.symlink(source/(fid+'.png'),folder/f'{i:06d}.png')
  write(OUT/'input_contract.json',dict(frame_ids=ids,prompt='vehicle',images='original1920x1280, unchanged full RGB',training_scope='all49 observed frames, no heldout',mask_predictions_not_ground_truth=True,no_R15_model_or_tristate_inputs=True))
  status('FULL_RESOLUTION_ALL_VEHICLE_NATIVE_INFERENCE_RUNNING',frames=0,total=49)
  predictor=Sam3VideoPredictor(checkpoint_path='/root/autodl-tmp/models/idsplat/sam3/sam3.pt',bpe_path='/root/autodl-tmp/idsplat/asset_sources/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',strict_state_dict_loading=True,compile=False,async_loading_frames=False)
  seen=set()
  with torch.inference_mode():
   sid=predictor.handle_request({'type':'start_session','resource_path':str(folder),'offload_video_to_cpu':True,'offload_state_to_cpu':True})['session_id']
   predictor.handle_request({'type':'add_prompt','session_id':sid,'frame_index':0,'text':'vehicle','output_prob_thresh':.5})
   for r in predictor.handle_stream_request({'type':'propagate_in_video','session_id':sid,'propagation_direction':'forward','start_frame_index':0,'output_prob_thresh':.5}):
    idx=int(r['frame_index']);assert idx not in seen;seen.add(idx);v=r['outputs']
    masks=np.asarray(v['out_binary_masks'],bool);assert masks.shape[1:]==(1280,1920),masks.shape
    np.savez_compressed(raw/(ids[idx]+'.npz'),masks=masks,tracker_ids=np.asarray(v['out_obj_ids'],np.int64),scores=np.asarray(v['out_probs'],np.float32))
    status('FULL_RESOLUTION_ALL_VEHICLE_NATIVE_INFERENCE_RUNNING',frames=len(seen),total=49,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
    print(json.dumps(dict(frame=ids[idx],proposals=len(masks))),flush=True)
   predictor.handle_request({'type':'close_session','session_id':sid})
  predictor.shutdown();del predictor;torch.cuda.empty_cache()
  assert seen==set(range(49))
  write(OUT/'inference_receipt.json',dict(status='COMPLETE_NATIVE49_REQUIRES_COVERAGE_REVIEW',frames=49,training_updates=0,full_resolution=True,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,human_verdict=None))
  status('NATIVE49_COMPLETE_COVERAGE_AND_IDENTITY_REVIEW_NEXT',frames=49)
 finally:
  write(claim,dict(active=False,pid=os.getpid(),released_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0));lock.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  status('FAILED_ALL_VEHICLE_PREPARATION_PRESERVED_NO_RESTART',error=traceback.format_exc());raise

"""三秒来源的隔离SAM2暂存，不伪造来源AI准入、不裁GT框。"""
from pathlib import Path
import sys,os,time,fcntl,importlib.util
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P));sys.path.insert(0,str(P/'iteration9'))
import json,ast
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
from segment_sources import CHECKPOINT,CONFIG

import numpy as np,cv2,torch
from PIL import Image
# 只执行已有归一化检查函数原AST，避免给SAM2环境引入nuScenes依赖。
tree=ast.parse((P/'iteration5/planning.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalized_masks');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(P/'iteration5/planning.py'),'exec'))
O=T/'r16';ROOT=O/'factory'

def main():
 ready=read(O/'source_ready.json');assert ready['stage']=='ready_for_quarantined_SAM2' and ready['synthetic_admission']==0
 assert torch.cuda.is_available();torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42)
 lock=open(O/'segmentation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2');from sam2.build_sam import build_sam2_video_predictor
 predictor=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda');extension=importlib.util.find_spec('sam2._C') is not None
 sources=read(ROOT/'source_manifest.json')['clips'];jobs=[{'source_id':c['source_id'],'instance_token':a['instance_token'],'job_id':c['source_id'] if j==0 else c['source_id']+'_'+a['instance_token'][:8],'folder':'segmented' if j==0 else 'segmented_secondary','frame_filenames':[f['filename'] for f in c['frames']]} for c in sources for j,a in enumerate(c['actors'])]
 dump(O/'segmentation_queue.json',{'jobs':jobs,'role':'quarantined_only_no_source_AI_inherited','checkpoint':CHECKPOINT,'config':CONFIG,'frames':30,'human_verdict':None})
 sp=O/'segmentation_state.json';state=read(sp) if sp.exists() else {'completed':[]};state.update(stage='running_quarantined_masks',pid=os.getpid(),total_jobs=len(jobs),training_admission=0);dump(sp,state)
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for c in sources:
    sid=c['source_id'];rgb=ROOT/'SAM2_rgb'/sid;rgb.mkdir(parents=True,exist_ok=True)
    for f in c['frames']:
     dest=rgb/f'{f["frame"]:05}.jpg'
     if not dest.exists():
      with Image.open(ROOT/'rgb'/f['filename']) as im:im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS).save(dest,quality=96)
    assert {p.name for p in rgb.iterdir()}=={f'{i:05}.jpg' for i in range(30)}
    track=None
    for job in [j for j in jobs if j['source_id']==sid]:
     dest=ROOT/job['folder']/job['job_id'];dest.mkdir(parents=True,exist_ok=True);folder=dest/'sam2_raw';folder.mkdir(exist_ok=True)
     existing=next((r for r in state['completed'] if r['job_id']==job['job_id']),None)
     if existing and existing['frame_filenames']==job['frame_filenames'] and existing['checkpoint']==CHECKPOINT and existing['config']==CONFIG:
      for i in range(30):
       with Image.open(folder/f'{i:05}.png') as im:im.load();assert im.mode=='L' and im.size==(1024,576)
      continue
     start=time.time()
     if track is None:track=predictor.init_state(video_path=str(rgb),offload_video_to_cpu=True,offload_state_to_cpu=True)
     else:predictor.reset_state(track)
     actor=next(a for a in c['frames'][15]['actors'] if a['instance_token']==job['instance_token']);box=np.float32(actor['projection']['box_xyxy']);predictor.add_new_points_or_box(track,frame_idx=15,obj_id=1,box=box);raw={}
     for reverse in [False,True]:
      for fid,_,logits in predictor.propagate_in_video(track,start_frame_idx=15,reverse=reverse):raw[int(fid)]=(logits[0,0]>0).cpu().numpy()
     assert set(raw)==set(range(30))
     for i,m in raw.items():Image.fromarray(m.astype('uint8')*255).save(folder/f'{i:05}.png')
     stats=normalized_masks([raw[i] for i in range(30)],[next(a for a in f['actors'] if a['instance_token']==job['instance_token'])['projection']['box_xyxy'] for f in c['frames']]);eligible=all(s['pixels'] and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in stats)
     row=job|{'checkpoint':CHECKPOINT,'config':CONFIG,'prompt_frame':15,'frames':30,'normalization_stats':stats,'technical_temporal_mask_pass':bool(eligible),'source_AI_status':'not_inherited_pending_synthetic_review','synthetic_admission':0,'postprocess':'official defaults, no GT clipping or custom cleanup','compiled_extension_available':extension,'seconds':time.time()-start,'human_verdict':None}
     dump(dest/'mask_manifest.json',row);state['completed']=[r for r in state['completed'] if r['job_id']!=job['job_id']]+[row];dump(sp,state);print('SAM2',job['job_id'],bool(eligible),len(state['completed']),len(jobs),flush=True)
    if track is not None:predictor.reset_state(track);del track
    torch.cuda.empty_cache()
  state.update(stage='complete_quarantined_pending_synthetic_QA',technical_mask_pass=sum(r['technical_temporal_mask_pass'] for r in state['completed']),finished_unix=time.time(),training_admission=0);assert len(state['completed'])==len(jobs);dump(sp,state)
 except Exception as e:state.update(stage='failed',error=repr(e));dump(sp,state);raise
if __name__=='__main__':main()

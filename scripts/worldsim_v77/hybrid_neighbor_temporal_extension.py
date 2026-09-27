"""r22：冻结r18方法，检查f75–94新时间窗，不为新帧重调参数。"""
import os, sys, copy, time, datetime, signal
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_neighbor_condition import Engine, BASE, DE, SPEC, DATA, INST, box_for, tight, scene_frame, view_points, set_seed, load_model
from hybrid_common import read, dump, mask, write_mask
from interactive_gui import get_obj_im_cond
from torchvision import transforms
import torch, cv2, numpy as np
from PIL import Image

ROOT=BASE/'r22'
def build(start):
 e=Engine(False);e.im=[];e.masks=[];e.box=[];e.frames=[]
 for f in range(start,start+10):
  fr=scene_frame(SPEC['spec'],f,INST);box,k=box_for(fr,'52',3);e.camera_intrinsic=k;e.frames.append(fr);e.box.append(box)
  e.im.append(np.array(Image.open(DATA/'images'/f'{f:03}_3.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR)))
  m=cv2.resize(mask(BASE/'r3/scene_0255/condition'/f'{f-65:05}.png').astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0
  rect=tight(view_points(box.corners(),k,normalize=True)[:2].T,1.05)*.64;x0,y0,x1,y1=np.rint(rect).astype(int);m[y0:y1,x0:x1]=True;e.masks.append(m)
 p=e.payload(True);q=copy.deepcopy(p)
 idx=int(p[6][0]);source=list((BASE/'r17/sv3d').glob(f'{idx:02}_*.png'));assert len(source)==1,source
 src=np.array(Image.open(source[0]).convert('RGB'));m=(src.min(-1)<230).astype('uint8');contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);m[:]=0;cv2.drawContours(m,[max(contours,key=cv2.contourArea)],-1,1,-1)
 rect=tight(view_points(e.box[0].corners(),e.camera_intrinsic,normalize=True)[:2].T,1.05)
 obj=e.to_tensor(src)*2-1;mask3=torch.from_numpy(m.astype('float32'))[None].repeat(3,1,1)*2-1
 im,anchor=get_obj_im_cond(obj,mask3,rect,scale=e.objratio);resize=transforms.Resize((576,1024),antialias=True);im=resize(im);anchor=resize(anchor)>0;anchor&=torch.from_numpy(e.masks[0])[None];q[0][0][anchor]=im[anchor]
 assert torch.equal(p[0][1:],q[0][1:])
 for a,b in zip(p[1:],q[1:]):
  if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 change=(q[0][0]!=p[0][0]).any(0).numpy();assert change.any() and not (change&~e.masks[0]).any()
 e.payloads={'neighbor':q};e.arm='neighbor'
 return e,dict(start=start,end=start+9,first_frame_changed_pixels=int(change.sum()),later_context_unchanged=True,other_tensor_conditions_unchanged=True,saved_anchor_source=str(source[0]),azimuth=e.azimuth_audit)

def prepare():
 assert not ROOT.exists();ROOT.mkdir()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r22',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',target='25',keep_actor='52',camera=3,frames=list(range(75,95)),
  hypothesis='The user-preferred r18 appearance anchor should retain SUV identity beyond its initial ten frames, under the same fixed rule.',
  user_evidence='2026-09-28 user judges r18 > r15 and requests retaining/refining it before adding scene3.',
  inputs='Original f75–94 RGB, existing r3 masks, GT52 camera/box, original f130 reference, frozen r17 generated SV3D view selected by existing azimuth rule; no generated prior claimed factual.',
  fixed=dict(seed=42,steps=25,frames_per_window=10,fps=10,size=[1024,576],previous_segment_condition=False,reference='f130 CAM3 actor52',sv3d_anchor='saved r17 view selected by unchanged azimuth index',writeback='old r3 writeback kept; evaluate native separately'),
  scope='Two non-overlapping independent windows; compare joining boundary to saved r18 f65–74. No new scene-specific tuning, matting fit, seed sweep, model download, training, or new model family.',
  stop_rule='Stop on structural or temporal failure; no downstream Omega. Independent windows are not assumed continuous. Detailed body/wheel limitations stay visible.',
  resources=dict(gpu='single RTX3090',cpu_threads=4,timeout_per_window_seconds=600),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rows=[]
 for start in [75,85]:
  out=ROOT/f'f{start:03}';out.mkdir();e,v=build(start);rows.append(v)
  for sub in ['input','condition','mask','write']:(out/sub).mkdir()
  for i,f in enumerate(range(start,start+10)):
   Image.fromarray(e.im[i]).save(out/'input'/f'{i:05}.png');t=e.payloads['neighbor'][0][i];Image.fromarray(((t.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')).save(out/'condition'/f'{i:05}.png')
   write_mask(out/'mask'/f'{i:05}.png',e.masks[i]);wr=cv2.resize(mask(BASE/'r3/scene_0255/write'/f'{f-65:05}.png').astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0;write_mask(out/'write'/f'{i:05}.png',wr)
  dump(out/'condition_validation.json',v)
 dump(ROOT/'condition_validation.json',dict(windows=rows,human_verdict=None));print('R22_PREPARED',rows,flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists() and not (ROOT/'state.json').exists();os.chdir(DE);state=dict(state='loading',pid=os.getpid(),windows={},human_verdict=None,background_input_dir=None);dump(ROOT/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('window exceeded 600s')))
 try:
  model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True)
  for start in [75,85]:
   out=ROOT/f'f{start:03}';e,v=build(start);assert v==read(out/'condition_validation.json');e.model=model;e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();beg=time.time();state.update(state='running',current_start=start);dump(ROOT/'state.json',state);signal.alarm(600)
   try:e.predict(1,False,'Deletion')
   finally:signal.alarm(0)
   assert len(e.im_result)==10;(out/'native').mkdir();(out/'final').mkdir()
   for i,im in enumerate(e.im_result):
    w=np.array(Image.open(out/'write'/f'{i:05}.png'))>0;final=e.im[i].copy();final[w]=im[w];assert np.array_equal(final[~w],e.im[i][~w]);Image.fromarray(im).save(out/'native'/f'{i:05}.png');Image.fromarray(final).save(out/'final'/f'{i:05}.png')
   state['windows'][str(start)]=dict(frames=10,seconds=time.time()-beg,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30);dump(ROOT/'state.json',state);print('R22_WINDOW_COMPLETE',start,state['windows'][str(start)],flush=True);torch.cuda.empty_cache()
  state.update(state='complete');dump(ROOT/'state.json',state)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[-1]]()

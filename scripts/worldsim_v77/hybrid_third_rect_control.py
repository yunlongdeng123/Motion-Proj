"""r28：第三例固定写回区域，仅比较模型条件中的精确洞与外包矩形。"""
import os,sys,time,datetime,signal
from pathlib import Path
os.environ.setdefault('DRIVEEDITOR_SEQUENTIAL_CFG','1');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
import numpy as np,cv2
from PIL import Image
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r28';OLD=BASE.parent/'WS-V77-DELETE-REPAIR-20260927/r1/official_000'

def inputs():
 images=[];masks=[];writes=[]
 for f in range(10):
  im=np.array(Image.open(OLD/'rgb'/f'{f:05}.png').convert('RGB'));m=np.array(Image.open(OLD/'mask'/f'{f:05}.png'))>0;y,x=np.where(m);rect=np.zeros(m.shape,bool);rect[max(0,y.min()-8):min(576,y.max()+25),max(0,x.min()-8):min(1024,x.max()+9)]=True
  assert m.shape==(576,1024) and np.all(rect[m]);images.append(im);masks.append(rect);writes.append(m)
 return images,masks,writes

def prepare():
 assert not ROOT.exists();ROOT.mkdir();images,masks,writes=inputs()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r28',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',camera=0,frames=list(range(10)),
  question='Does separating a rectangular model hole from precise output writeback reduce vehicle-shaped residual on the third scene?',
  baseline='Saved DELETE-REPAIR/r1 official_000 precise/native frames0..9, same original RGB/checkpoint/seed42/25steps/1024x576, first independent10-frame window.',
  intervention='Only input RGB hole/model mask changes from precise contour to its bounding rectangle padded8px left/right/top and24px bottom. Final writeback remains EXACT old precise mask. No reference/object conditions.',
  origin='Rectangular model condition follows original DriveEditor deletion input structure; fixed modest margin reused from earlier0230 bounded control. No result-selected seeds or new model. Third scene was previously used for repair, not a held-out test.',
  evidence='r25 visible road can be reprojected locally; r26/r27 actual-hole measured road support zero. No unsupported ground or generated RGB introduced as evidence.',
  fixed=dict(seed=42,steps=25,frames=10,size=[1024,576],sequential_cfg=True,previous_segment_condition=False,decode_chunk=1,cpu_threads=4),
  resources=dict(gpu='RTX3090',timeout_s=600),stop_rule='One10-frame control; inspect native and exact-writeback. Stop this mask recipe if target regeneration or structural damage persists; preserve original baseline. Do not infer scene0255 actor52 preservation from this road-background task.',failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 for name in ['input','condition','model_mask','write_mask']:(ROOT/name).mkdir()
 rows=[]
 for i,(im,m,wr) in enumerate(zip(images,masks,writes)):
  for name,a in [('input',im),('condition',np.where(m[...,None],127,im).astype('uint8')),('model_mask',np.uint8(m)*255),('write_mask',np.uint8(wr)*255)]:Image.fromarray(a).save(ROOT/name/f'{i:05}.png')
  rows.append(dict(frame=i,old_write_pixels=int(wr.sum()),model_pixels=int(m.sum()),write_subset_model=bool(np.all(m[wr]))))
 dump(ROOT/'condition_validation.json',dict(rows=rows,writeback_identical_to_old=True,roles='condition PNG127 visualizes normalized tensor zero; not literal inference input scalar.',human_verdict=None));print('R28_PREPARED',rows,flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists() and not (ROOT/'state.json').exists()
 from repair_drive import Engine,set_seed,torch
 torch.set_num_threads(4);cv2.setNumThreads(4);state=dict(state='loading',pid=os.getpid(),human_verdict=None,background_input_dir=None);dump(ROOT/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('r28 exceeded600s')))
 try:
  e=Engine();e.im,e.masks,writes=inputs();e.im_result=[];e.previous_segment_last_frame=None;set_seed(42);torch.cuda.reset_peak_memory_stats();beg=time.time();state.update(state='running');dump(ROOT/'state.json',state);signal.alarm(600)
  try:e.predict(1,False,'Deletion')
  finally:signal.alarm(0)
  assert len(e.im_result)==10 and not e.used_previous_segment_condition
  for name in ['native','final']:(ROOT/name).mkdir()
  checks=[]
  for i,(im,native,wr) in enumerate(zip(e.im,e.im_result,writes)):
   result=im.copy();result[wr]=native[wr];assert np.array_equal(result[~wr],im[~wr]);Image.fromarray(native).save(ROOT/'native'/f'{i:05}.png');Image.fromarray(result).save(ROOT/'final'/f'{i:05}.png');checks.append(dict(frame=i,outside_write_changed=0))
  state.update(state='complete',seconds=time.time()-beg,frames=10,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,checks=checks);dump(ROOT/'state.json',state);print('R28_COMPLETE',state['seconds'],flush=True)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[-1]]()

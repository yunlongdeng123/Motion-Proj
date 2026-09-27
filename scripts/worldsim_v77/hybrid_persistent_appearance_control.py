"""r24：冻结r18，仅将对应SV3D外观锚点从首帧扩到各帧。"""
import os,sys,copy,time,datetime,signal
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_neighbor_condition import Engine,BASE,DE,set_seed,load_model,tight,view_points
from hybrid_projected_anchor import payload as initial_payload
from hybrid_object_positive_control import same,image_tensor
from hybrid_common import read,dump,write_mask
from interactive_gui import get_obj_im_cond
from torchvision import transforms
import torch,numpy as np,cv2
from PIL import Image
ROOT=BASE/'r24'

def build():
 e=Engine(False);p,*_=initial_payload(e);q=copy.deepcopy(p);resize=transforms.Resize((576,1024),antialias=True);rows=[]
 for i in range(10):
  idx=int(p[6][i]);files=list((BASE/'r17/sv3d').glob(f'{idx:02}_*.png'));assert len(files)==1
  if i:
   src=np.array(Image.open(files[0]).convert('RGB'));m=(src.min(-1)<230).astype('uint8');cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);m[:]=0;cv2.drawContours(m,[max(cs,key=cv2.contourArea)],-1,1,-1)
   rect=tight(view_points(e.box[i].corners(),e.camera_intrinsic,normalize=True)[:2].T,1.05);obj=e.to_tensor(src)*2-1;mask3=torch.from_numpy(m.astype('float32'))[None].repeat(3,1,1)*2-1;im,mask=get_obj_im_cond(obj,mask3,rect,scale=e.objratio);im=resize(im);mask=resize(mask)>0;mask&=torch.from_numpy(e.masks[i])[None];q[0][i][mask]=im[mask]
  baseline=image_tensor(p[0][i]);assert np.array_equal(baseline,np.array(Image.open(BASE/'r18/condition'/f'{i:05}.png')))
  changed=(q[0][i]!=p[0][i]).any(0).numpy();assert not (changed&~e.masks[i]).any();rows.append(dict(frame=65+i,changed_pixels=int(changed.sum()),source=str(files[0]),outside_model_mask_changed=0))
 assert rows[0]['changed_pixels']==0 and all(r['changed_pixels']>0 for r in rows[1:])
 assert all(same(p[k],q[k]) for k in range(1,12));e.payloads={'neighbor':q};e.arm='neighbor'
 return e,q,rows

def prepare():
 assert not ROOT.exists();ROOT.mkdir();e,q,rows=build()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r24',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',target='25',keep_actor='52',camera=3,frames=list(range(65,75)),
  hypothesis='A persistent query-oriented appearance condition may retain identity/details after the first frame, where r18/r23 only-first-frame conditions drift.',
  baseline='Exact saved r18 first-frame-anchored conditions; user prefers r18 over r15.',
  only_change='Add saved r17 selected SV3D view to context frames1–9 using the same official placement and original GT52 projection/ratio; frame0 and all eleven other payload components unchanged.',
  input_roles='SV3D images are generated priors, not real hidden RGB; same f130 real reference already used by r18, original GT camera/boxes. No future query RGB in added anchors.',
  limitations='Training used a keyframe paste; all-frame appearance is an input adaptation. Finite-view prior may pin wrong view or create billboard/texture repetition. Detailed fidelity and occluders must be inspected.',
  fixed=dict(seed=42,steps=25,size=[1024,576],frames=10,fps=10,previous_segment_condition=False,crop=False,sequential_cfg=True,decode_chunk=1),
  stop_rule='One ten-frame same-window control; inspect native vs r18 first, no expansion if identity or temporal structure worsens. No crop or writeback change folded into main comparison.',
  resources=dict(gpu='single RTX3090',cpu_threads=4,timeout_s=600),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 (ROOT/'condition').mkdir()
 for i in range(10):Image.fromarray(image_tensor(q[0][i])).save(ROOT/'condition'/f'{i:05}.png')
 dump(ROOT/'condition_validation.json',dict(first_frame_exact_r18=True,other_eleven_payloads_exact_r18=True,checks=rows,human_verdict=None));print('R24_PREPARED',rows,flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists() and not (ROOT/'state.json').exists();os.chdir(DE);state=dict(state='loading',pid=os.getpid(),human_verdict=None,background_input_dir=None);dump(ROOT/'state.json',state);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('r24 exceeded 600s')))
 try:
  e,q,rows=build();assert rows==read(ROOT/'condition_validation.json')['checks'];e.model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True);e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();beg=time.time();state.update(state='running');dump(ROOT/'state.json',state);signal.alarm(600)
  try:e.predict(1,False,'Deletion')
  finally:signal.alarm(0)
  assert len(e.im_result)==10;(ROOT/'native').mkdir();(ROOT/'final').mkdir()
  for i,im in enumerate(e.im_result):
   w=np.array(Image.open(BASE/'r15/write'/f'{i:05}.png'))>0;raw=np.array(Image.open(BASE/'r15/input'/f'{i:05}.png'));out=raw.copy();out[w]=im[w];assert np.array_equal(out[~w],raw[~w]);Image.fromarray(im).save(ROOT/'native'/f'{i:05}.png');Image.fromarray(out).save(ROOT/'final'/f'{i:05}.png')
  state.update(state='complete',frames=10,seconds=time.time()-beg,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,outside_write_changed=0);dump(ROOT/'state.json',state);print('R24_COMPLETE',state,flush=True)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[-1]]()

"""r18：只给r15 neighbor首帧加入对应SV3D视角外观锚点。

该锚点来自r17生成先验，不是观测背景，不修改保护写回以混淆归因。
"""
import os,sys,time,signal,datetime,copy
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_neighbor_condition import Engine,BASE,DE,set_seed,tight
from interactive_gui import get_obj_im_cond
from hybrid_common import dump,read,write_mask
from nuscenes.utils.geometry_utils import view_points
from torchvision import transforms
import numpy as np,torch,cv2
from PIL import Image
ROOT=BASE/'r18'

def payload(e):
 p=copy.deepcopy(e.payloads['neighbor']);src=np.array(Image.open(BASE/'r17/sv3d/12_315.png').convert('RGB'));m=(src.min(-1)<230).astype('uint8')
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);m[:]=0;cv2.drawContours(m,[max(contours,key=cv2.contourArea)],-1,1,-1)
 corners=view_points(e.box[0].corners(),e.camera_intrinsic,normalize=True)[:2].T;rect=tight(corners,1.05)
 obj=e.to_tensor(src)*2-1;mask3=torch.from_numpy(m.astype('float32'))[None].repeat(3,1,1)*2-1
 im,mask=get_obj_im_cond(obj,mask3,rect,scale=e.objratio);resize=transforms.Resize((576,1024),antialias=True);im=resize(im);mask=resize(mask)>0
 mask&=torch.from_numpy(e.masks[0])[None];p[0][0][mask]=im[mask]
 return p,src,m,mask.any(0).numpy()

def prepare():
 assert not ROOT.exists();assert read(BASE/'r17/replay_validation.json')['reproduced_video_exact'];ROOT.mkdir();e=Engine(False);p,src,m,anchor=payload(e)
 assert torch.equal(p[0][1:],e.payloads['neighbor'][0][1:]);changed=(p[0][0]!=e.payloads['neighbor'][0][0]).any(0).numpy();assert changed.any() and not (changed&~e.masks[0]).any()
 for name in ['condition','anchor']:(ROOT/name).mkdir()
 for i,t in enumerate(p[0]):Image.fromarray(((t.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')).save(ROOT/'condition'/f'{i:05}.png')
 Image.fromarray(src).save(ROOT/'source_sv3d.png');write_mask(ROOT/'source_silhouette.png',m);write_mask(ROOT/'anchor/00000.png',anchor)
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r18',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',target='25',preserve_actor='52',camera=3,frames=list(range(65,75)),
  hypothesis='A query-oriented SV3D appearance anchor in frame0 can preserve SUV body shape during video integration, where feature-only r15 failed.',
  source='r17 frozen DriveEditor SV3D view12 relative315deg; generated prior, not real observed evidence. Same r15 reference/GT/seed/window reused.',
  only_change='First frame masked context gets the saved view, positioned with existing official get_obj_im_cond and same GT52 fusion bbox/ratio. Original CLIP/SV3D/depth/pose/masks and later contexts unchanged.',
  silhouette='Largest filled contour of non-white generated object (minRGB<230), saved for review; no external model added.',
  fixed=dict(seed=42,steps=25,fps=10,size=[1024,576],sequential_cfg=True,decode_chunk=1,previous_segment_condition=False),
  resources=dict(gpu='one RTX3090',cpu_threads=4,timeout_seconds=600),stop_rule='One 10-frame repair only; reject if wrong identity/body or new unsupported changes remain. Do not enlarge failed run.',failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 dump(ROOT/'condition_validation.json',dict(changed_first_frame_pixels=int(changed.sum()),later_context_exact_equal=True,other_conditions='copied unchanged from r15 payload',all_changes_within_model_mask=True,source_kind='generated SV3D prior',human_verdict=None));print('R18_PREPARED',int(changed.sum()),flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists() and not (ROOT/'state.json').exists();os.chdir(DE);state=dict(state='loading',pid=os.getpid(),human_verdict=None);dump(ROOT/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('r18 exceeded 600s')))
 try:
  e=Engine(True);p,*_=payload(e);e.payloads['neighbor']=p;e.arm='neighbor';e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.time();signal.alarm(600);state.update(state='running');dump(ROOT/'state.json',state)
  e.predict(1,False,'Deletion');signal.alarm(0);(ROOT/'native').mkdir();(ROOT/'final').mkdir();rows=[]
  for i,im in enumerate(e.im_result):
   raw=np.array(Image.open(BASE/'r15/input'/f'{i:05}.png'));m=np.array(Image.open(BASE/'r15/write'/f'{i:05}.png'))>0;out=raw.copy();out[m]=im[m];assert np.array_equal(raw[~m],out[~m]);Image.fromarray(im).save(ROOT/'native'/f'{i:05}.png');Image.fromarray(out).save(ROOT/'final'/f'{i:05}.png');rows.append(dict(frame=65+i,outside_write_changed=0))
  assert len(rows)==10;state.update(state='complete',frames=10,seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,checks=rows);dump(ROOT/'state.json',state);print('R18_COMPLETE',state['seconds'],flush=True)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[-1]]()

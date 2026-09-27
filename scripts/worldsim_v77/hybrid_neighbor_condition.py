"""r15：同一DriveEditor checkpoint，空条件 vs 已知邻车参考/3D条件的有界控制。

这是background builder中补全保留对象的适配控制；不是官方原样deletion推理。
不把不同视角的参考crop硬贴到首帧，不改第三方源码和原GLB。
"""
import os,sys,time,datetime,signal,argparse
from pathlib import Path
os.environ.setdefault('DRIVEEDITOR_SEQUENTIAL_CFG','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera
from video_review import scene_frame
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from PIL import ImageDraw
DE=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor');sys.path.insert(0,str(DE))
from interactive_gui import GradioShow,load_model,set_seed,process_box,get_scale_by_obj_azimuth
BASE=ROOT.parent;R2=ROOT;ROOT=BASE/'r15';OUT_HW=(576,1024);torch.set_num_threads(4);cv2.setNumThreads(4)
SPEC=next(s for s in read(R2/'registration.json')['scenes'] if s['name']=='scene_0255');DATA=Path(SPEC['data']);INST=read(DATA/'instances/instances_info.json')

def box_for(fr,aid,cam):
 b=next(b for b in fr['all_boxes'] if b['actor_id']==aid);c,k=camera(fr,cam,(900,1600));p=np.linalg.inv(c)@np.array(b['pose']);l,w,h=b['size_lwh']
 return Box(p[:3,3],[w,l,h],Quaternion(matrix=p[:3,:3])),k

def tight(corners,scale=1.05):
 lo=corners.min(0);hi=corners.max(0);center=(lo+hi)/2;half=(hi-lo)*scale/2
 return np.array([max(0,center[0]-half[0]),max(0,center[1]-half[1]),min(1600,center[0]+half[0]),min(900,center[1]+half[1])])

class Engine(GradioShow):
 def __init__(self,load=False):
  self.out_size=OUT_HW;self.out_size_3d=(576,576);self.num_frames=10;self.num_frames_3d=21;self.device='cuda';self.ratio=.64
  self.to_tensor=transforms.ToTensor();self.transform_img=transforms.Compose([self.to_tensor,transforms.Resize(OUT_HW,antialias=True),transforms.Lambda(lambda x:x*2-1)])
  self.transform_mask=transforms.Compose([transforms.Resize([72,128],interpolation=InterpolationMode.NEAREST),transforms.Lambda(lambda x:x*2-1)])
  self.transform_depth=transforms.Compose([self.to_tensor,transforms.Lambda(lambda x:x/(256.*50.)),transforms.Lambda(lambda x:x*2-1),transforms.Resize(OUT_HW,antialias=True)])
  self.previous_segment_last_frame=None;self.used_previous_segment_condition=False;self.im_result=[];self.im=[];self.masks=[];self.box=[];self.frames=[]
  for i,f in enumerate(range(65,75)):
   fr=scene_frame(SPEC['spec'],f,INST);box,k=box_for(fr,'52',3);self.camera_intrinsic=k;self.frames.append(fr);self.box.append(box)
   im=np.array(Image.open(DATA/'images'/f'{f:03}_3.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));self.im.append(im)
   m=cv2.resize(mask(BASE/'r3/scene_0255/condition'/f'{i:05}.png').astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0
   corners=view_points(box.corners(),k,normalize=True)[:2].T;rect=tight(corners,1.05)*.64;x0,y0,x1,y1=np.rint(rect).astype(int);m[y0:y1,x0:x1]=True;self.masks.append(m)
  ref_frame=scene_frame(SPEC['spec'],130,INST);self.reference_box,self.reference_k=box_for(ref_frame,'52',3)
  reference=np.array(Image.open(DATA/'images/130_3.jpg').convert('RGB'));sm=np.array(Image.open(BASE/'r14/130_3/sam.png').convert('L').resize((1600,900),Image.Resampling.NEAREST))>127
  self.reference_rgb=reference;self.reference_mask=sm;white=np.where(sm[...,None],reference,255).astype('uint8');self.reference_white=white
  obj=self.to_tensor(white)*2-1;ms=torch.from_numpy(sm.astype('float32'))[None].repeat(3,1,1)*2-1
  _,self.ref_az=self._get_matrix(self.reference_box);self.obj_clip,_=self._get_obj_im(obj,ms,margin=32)
  self.obj3d,shape=self._get_obj_im(obj,ms,margin=20,out_size=576,scale_ratio=get_scale_by_obj_azimuth(self.ref_az));self.obj3d=self.obj3d[None];self.objheight=shape[0]
  rc=view_points(self.reference_box.corners(),self.reference_k,normalize=True)[:2].T;rf=tight(rc,1.05);y,x=np.where(sm);self.objratio=float((y.max()-y.min())/(rf[3]-rf[1]))
  self.arm='deletion';self.payloads={};self.payloads['deletion']=self.payload(False);self.payloads['neighbor']=self.payload(True)
  if load:self.model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True)

 def payload(self,neighbor):
  cond=[]
  for im,m in zip(self.im,self.masks):
   a=self.to_tensor(im.copy())*2-1;a[:,m]=0;cond.append(a)
  condition=torch.stack(cond);masks=self.transform_mask(torch.from_numpy(np.stack(self.masks).astype('float32'))[:,None])
  if not neighbor:return (condition,torch.ones(1,3,576,576),masks,torch.zeros_like(masks),torch.zeros(21,1),torch.zeros(21,1),torch.zeros(21,1),[{} for _ in cond],torch.ones(3,224,224),torch.ones(10,6,576,1024)*-1,torch.zeros(10),torch.zeros(10))
  fuse=[];positions=[];depth=[];el=[];az=[]
  for box,fr in zip(self.box,self.frames):
   k=np.array(fr['views'][3]['intrinsics']);corners=view_points(box.corners(),k,normalize=True)[:2].T;rect=tight(corners,1.05);x0,y0,x1,y1=np.rint(rect).astype(int);fm=np.zeros((900,1600),np.float32);fm[y0:y1,x0:x1]=1;fuse.append(torch.from_numpy(fm)[None])
   lo=corners.min(0);hi=corners.max(0);center=(lo+hi)/2
   positions.append(dict(yx=torch.tensor([[center[1]/900,center[0]/1600]],dtype=torch.float32),box_height=torch.tensor((rect[3]-rect[1])*.64,dtype=torch.float32),obj_height=torch.tensor(self.objheight)))
   # 使用官方六面深度绘制及官方深度归一化。
   dep=process_box(box.corners().T,corners.copy(),k);depth.append(torch.concat([self.transform_depth(x.astype('float32')) for x in dep]))
   e,a=self._get_matrix(box);el.append(e);az.append(a)
  samples=torch.tensor(np.deg2rad([3.,6.,9.,12.,16.,23.,30.,45.,90.,135.,225.,270.,315.,330.,337.,344.,348.,351.,354.,357.,0.]))
  # 原官方以首帧物体视角为reference；本控制用真实参考相机的姿态。
  diff=torch.remainder(torch.tensor(az)-self.ref_az,2*np.pi);indices=torch.argmin(torch.abs(diff[:,None]-samples),dim=1).to(torch.int32)
  self.azimuth_audit=dict(reference_frame=130,reference_camera=3,reference_azimuth_deg=float(np.rad2deg(self.ref_az)),query_azimuth_deg=np.rad2deg(az).tolist(),relative_deg=np.rad2deg(diff.numpy()).tolist(),selected_sv3d_deg=np.rad2deg(samples[indices].numpy()).tolist(),reference_ratio=self.objratio)
  return (condition,self.obj3d,masks,self.transform_mask(torch.stack(fuse)),torch.ones(21)*float(np.mean(el)),samples,indices,positions,self.obj_clip,torch.stack(depth),torch.ones(10),torch.ones(10)*self.objratio)

 def get_deletion(self):return self.payloads[self.arm]

def prepare():
 assert not ROOT.exists();ROOT.mkdir();e=Engine(False)
 cfg=dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r15',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',target='25',preserved_neighbor='52',query_frames=list(range(65,75)),camera=3,
  hypothesis='unconditioned deletion does not specify the occluded non-target car; fixed reference and 3D support may complete the correct retained actor',
  model='existing official trained DriveEditor checkpoint, frozen; no download or training',
  arms=dict(deletion='same mask/context with official neutral object conditions',neighbor='same mask/context + original actor52 SAM-matted reference + original GT 3D box/depth/SV3D view conditions'),
  adaptation='Use official condition interfaces and six-face depth; reference view angle from f130 CAM3 rather than query0; no different-view crop pasted into first frame. This is not untouched official deletion.',
  fixed=dict(frames=10,fps=10,size=[1024,576],seed=42,steps=25,sequential_cfg=True,decoding_t=1,previous_segment_condition=False,model_mask='r3 target rectangle union neighbor52 projection x1.05',write_mask='r3 write mask resized; preserve original protect/outside'),
  inputs='offline BUILD uses original later reference at f130; GT camera/box and manual development scene choice. No hidden ground truth.',
  stop_rule='one 10-frame window per arm; inspect native before writeback; no seed grid or further model families',resources=dict(gpu='one existing RTX3090',cpu_threads=4,timeout_per_arm_s=600),failure_ledger_refs=['V77-F02'],human_verdict=None)
 dump(ROOT/'registration.json',cfg);dump(ROOT/'azimuth_audit.json',e.azimuth_audit)
 for key in ['input','condition','write','masked']:(ROOT/key).mkdir()
 for i,(im,m) in enumerate(zip(e.im,e.masks)):
  Image.fromarray(im).save(ROOT/'input'/f'{i:05}.png');write_mask(ROOT/'condition'/f'{i:05}.png',m);Image.fromarray(np.where(m[...,None],0,im).astype('uint8')).save(ROOT/'masked'/f'{i:05}.png')
  wr=cv2.resize(mask(BASE/'r3/scene_0255/write'/f'{i:05}.png').astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0;write_mask(ROOT/'write'/f'{i:05}.png',wr)
 Image.fromarray(e.reference_rgb).save(ROOT/'reference_raw.jpg');Image.fromarray(e.reference_white).save(ROOT/'reference_white.png')
 for key,t in [('reference_clip',e.obj_clip),('reference_sv3d',e.obj3d[0])]:Image.fromarray(((t.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')).save(ROOT/f'{key}.png')
 a,b=e.payloads['deletion'],e.payloads['neighbor'];assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2])
 assert torch.count_nonzero(a[10])==0 and torch.all(b[10]==1) and len(b[7])==10
 assert all(torch.isfinite(t).all() for t in b if isinstance(t,torch.Tensor))
 assert b[9].shape==(10,6,576,1024) and b[6].min()>=0 and b[6].max()<21
 rows=[]
 for i,fr in enumerate(e.frames):
  box,k=box_for(fr,'52',3);corners=view_points(box.corners(),k,normalize=True)[:2].T;gt=next(x for x in fr['all_boxes'] if x['actor_id']=='52');c,_=camera(fr,3,(900,1600));expected=project_bbox(gt['pose'],gt['size_lwh'],c,k,(900,1600));actual=np.r_[corners.min(0),corners.max(0)];assert np.max(np.abs(actual-expected))<1e-5;rows.append(dict(frame=65+i,max_projection_error=float(np.max(np.abs(actual-expected)))))
 dump(ROOT/'condition_validation.json',dict(same_mask_and_context=True,finite=True,geometry_projection=rows,payload_shapes=[list(t.shape) if isinstance(t,torch.Tensor) else len(t) for t in b],human_verdict=None))
 print('NEIGHBOR CONDITION PREPARED',e.azimuth_audit,flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists();assert not (ROOT/'state.json').exists();os.chdir(DE);state=dict(state='loading',pid=os.getpid(),arms={},human_verdict=None);dump(ROOT/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('one arm exceeded 600s')))
 try:
  e=Engine(True)
  for arm in ['deletion','neighbor']:
   out=ROOT/arm;out.mkdir();(out/'native').mkdir();(out/'final').mkdir();state.update(state='running',current_arm=arm);dump(ROOT/'state.json',state);e.arm=arm;e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();started=time.time();signal.alarm(600)
   try:e.predict(1,False,'Deletion')
   finally:signal.alarm(0)
   assert len(e.im_result)==10;checks=[]
   for i,raw in enumerate(e.im_result):
    assert raw.shape==(576,1024,3);original=e.im[i];wr=np.array(Image.open(ROOT/'write'/f'{i:05}.png'))>0;result=original.copy();result[wr]=raw[wr];assert np.array_equal(result[~wr],original[~wr])
    Image.fromarray(raw).save(out/'native'/f'{i:05}.png');Image.fromarray(result).save(out/'final'/f'{i:05}.png');checks.append(dict(frame=65+i,outside_write_changed=0))
   record=dict(frames=10,seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,checks=checks);dump(out/'result.json',record);state['arms'][arm]=record;dump(ROOT/'state.json',state);print('ARM_COMPLETE',arm,record['seconds'],record['peak_allocated_gib'],flush=True);torch.cuda.empty_cache()
  state.update(state='complete');dump(ROOT/'state.json',state)
 except Exception as exc:
  state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':
 action=sys.argv[-1]
 if action=='prepare':prepare()
 elif action=='run':run()
 else:raise ValueError('prepare or run required')

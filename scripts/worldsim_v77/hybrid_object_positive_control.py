"""r16：已知可见actor52的官方条件正控制，只拆首帧外观锚点。

使用原始RGB作可见区域评价；这是oracle诊断，不是DELETE结果。
"""
import os,sys,time,signal,datetime,copy
from pathlib import Path
os.environ.setdefault('DRIVEEDITOR_SEQUENTIAL_CFG','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_neighbor_condition import box_for,SPEC,INST,DATA,DE,OUT_HW
sys.path.insert(0,str(DE))
import interactive_gui as official
from interactive_gui import GradioShow,load_model,set_seed
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import read,dump,write_mask
from video_review import scene_frame
from nuscenes.utils.geometry_utils import view_points
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r16'
torch.set_num_threads(4);cv2.setNumThreads(4)

def image_tensor(t):
 return ((t.detach().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')

class Engine(GradioShow):
 def __init__(self,load=False):
  self.out_size=OUT_HW;self.out_size_3d=(576,576);self.num_frames=10;self.num_frames_3d=21;self.device='cuda';self.ratio=.64
  self.to_tensor=transforms.ToTensor()
  self.transform_img=transforms.Compose([self.to_tensor,transforms.Resize(OUT_HW,antialias=True),transforms.Lambda(lambda x:x*2-1)])
  self.transform_img_resize=transforms.Resize(OUT_HW,antialias=True)
  self.transform_mask=transforms.Compose([transforms.Resize([72,128],interpolation=InterpolationMode.NEAREST),transforms.Lambda(lambda x:x*2-1)])
  self.transform_depth=transforms.Compose([self.to_tensor,transforms.Lambda(lambda x:x/(256.*50.)),transforms.Lambda(lambda x:x*2-1),transforms.Resize(OUT_HW,antialias=True)])
  self.previous_segment_last_frame=None;self.im_result=[];self.im=[];self.box=[];self.frames=[]
  for f in range(130,140):
   fr=scene_frame(SPEC['spec'],f,INST);box,k=box_for(fr,'52',3);self.frames.append(fr);self.box.append(box);self.camera_intrinsic=k
   self.im.append(np.array(Image.open(DATA/'images'/f'{f:03}_3.jpg').convert('RGB')))
  sm=np.array(Image.open(BASE/'r14/130_3/sam.png').convert('L').resize((1600,900),Image.Resampling.NEAREST))>127
  self.reference_mask=sm;self.reference_white=np.where(sm[...,None],self.im[0],255).astype('uint8')
  self.data=dict(im=self.reference_white,mask=(sm*255).astype('uint8'),category_name='vehicle.car',data=[dict(box=b) for b in self.box])
  self.payloads={};set_seed(42)
  # 原官方函数负责mask、首帧贴入、六面深度、CLIP、SV3D及其角度。
  self.payloads['anchor']=GradioShow.get_editing(self,'Repositioning')
  original=official.get_obj_im_cond
  def without_paste(*args,**kwargs):
   im,m=original(*args,**kwargs)
   return im,torch.zeros_like(m)
  try:
   official.get_obj_im_cond=without_paste;set_seed(42)
   self.payloads['no_anchor']=GradioShow.get_editing(self,'Repositioning')
  finally:official.get_obj_im_cond=original
  self.arm='anchor'
  if load:self.model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True)

 def get_deletion(self):return copy.deepcopy(self.payloads[self.arm])

def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def prepare():
 assert not ROOT.exists();ROOT.mkdir();e=Engine();a=e.payloads['anchor'];b=e.payloads['no_anchor']
 assert all(same(a[i],b[i]) for i in range(1,12));assert torch.equal(a[0][1:],b[0][1:]);assert not torch.equal(a[0][0],b[0][0])
 assert all(torch.isfinite(t).all() for t in a if isinstance(t,torch.Tensor));assert torch.all(a[10]==1)
 for name in ['input','anchor_condition','no_anchor_condition','model_mask','box_support']:(ROOT/name).mkdir()
 for i in range(10):
  im=image_tensor(e.transform_img(e.im[i]));Image.fromarray(im).save(ROOT/'input'/f'{i:05}.png')
  for arm in ['anchor','no_anchor']:Image.fromarray(image_tensor(e.payloads[arm][0][i])).save(ROOT/f'{arm}_condition'/f'{i:05}.png')
  m=cv2.resize((a[2][i,0].numpy()>0).astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0;write_mask(ROOT/'model_mask'/f'{i:05}.png',m)
  corners=view_points(e.box[i].corners(),e.camera_intrinsic,normalize=True)[:2].T*.64;lo=np.floor(corners.min(0)).astype(int);hi=np.ceil(corners.max(0)).astype(int);support=np.zeros((576,1024),bool);support[max(0,lo[1]):min(576,hi[1]),max(0,lo[0]):min(1024,hi[0])]=True;write_mask(ROOT/'box_support'/f'{i:05}.png',support)
 Image.fromarray(e.reference_white).save(ROOT/'reference_white.png');Image.fromarray(image_tensor(a[1][0])).save(ROOT/'reference_sv3d.png')
 diff=(a[0][0]!=b[0][0]).any(0).numpy();write_mask(ROOT/'anchor_difference.png',diff)
 audit=dict(all_other_conditions_exact_equal=True,frames_1_to_9_context_exact_equal=True,first_frame_changed_pixels=int(diff.sum()),sv3d_indices=a[6].tolist(),obj_ratio=a[11].tolist(),human_verdict=None)
 dump(ROOT/'condition_validation.json',audit)
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r16',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',actor='52',frames=list(range(130,140)),camera=3,
  question='Can the existing object branch reconstruct a known visible silver SUV with official same-view first-frame appearance, and what changes if only this keyframe paste is removed?',
  implementation='Official get_editing Repositioning at zero offset; exact same precomputed conditions in both arms except first-frame object paste. Existing serial CFG and offscreen paste-boundary patch remain; actor is on-screen.',
  arms=dict(anchor='official generated first-frame object paste',no_anchor='same crop/CLIP/SV3D/depth/pose/masks and RGB except no first-frame paste'),
  input_roles='First-frame original actor52 RGB and SAM plus GT box are oracle conditioning. Later original RGB is evaluation only inside masked region. Not independent test, not DELETE result.',
  fixed=dict(seed=42,steps=25,size=[1024,576],fps=10,num_frames=10,sequential_cfg=True,decode_chunk=1,previous_segment_condition=False,model_mask='official 1.3x projected box with saved seeded random center'),
  resources=dict(gpu='one RTX3090',cpu_threads=4,arm_timeout_seconds=600),stop_rule='One 10-frame positive control per arm; inspect identity/shape against known original; do not extend failed configuration.',
  failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 print('R16_PREPARED',audit,flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists();assert not (ROOT/'state.json').exists();os.chdir(DE)
 state=dict(state='loading',pid=os.getpid(),arms={},human_verdict=None);dump(ROOT/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('arm exceeded 600s')))
 try:
  e=Engine(True)
  for arm in ['anchor','no_anchor']:
   p=ROOT/arm;p.mkdir();(p/'native').mkdir();state.update(state='running',current_arm=arm);dump(ROOT/'state.json',state)
   e.arm=arm;e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.time();signal.alarm(600)
   try:e.predict(1,False,'Deletion')
   finally:signal.alarm(0)
   assert len(e.im_result)==10;rows=[]
   for i,im in enumerate(e.im_result):
    assert im.shape==(576,1024,3);Image.fromarray(im).save(p/'native'/f'{i:05}.png')
    gt=np.array(Image.open(ROOT/'input'/f'{i:05}.png'));m=np.array(Image.open(ROOT/'box_support'/f'{i:05}.png'))>0;err=np.abs(im.astype('float32')-gt).mean(-1)
    rows.append(dict(frame=130+i,box_pixels=int(m.sum()),box_mae_mean=float(err[m].mean()),box_mae_median=float(np.median(err[m])),box_mae_q95=float(np.percentile(err[m],95))))
   result=dict(frames=10,seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,oracle_metrics=rows,metric_limit='Projected box includes background and occluders; not exact surface truth. First-frame copy is a conditioning advantage; visual and later-frame review required.')
   dump(p/'result.json',result);state['arms'][arm]=result;dump(ROOT/'state.json',state);torch.cuda.empty_cache();print('R16_ARM_COMPLETE',arm,result['seconds'],flush=True)
  state.update(state='complete');dump(ROOT/'state.json',state)
 except Exception as exc:
  state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':
 {'prepare':prepare,'run':run}[sys.argv[-1]]()

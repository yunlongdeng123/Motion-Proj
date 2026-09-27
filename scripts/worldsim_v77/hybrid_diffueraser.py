"""保留官方网络/采样；显式PNG I/O适配、prior归档、保护与证据精确写回。"""
import os,gc,time,types,random,traceback
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
from hybrid_common import *
import torch
SOURCE=pathlib.Path('/root/autodl-tmp/third_party/worldsim_v77/DiffuEraser')
WEIGHTS=pathlib.Path('/root/autodl-tmp/models/worldsim_v77_diffueraser')
torch.set_num_threads(4);cv2.setNumThreads(4)
sys.path.insert(0,str(SOURCE));os.chdir(SOURCE)

def images(path,size=None):
 files=sorted(pathlib.Path(path).glob('*.png'));assert len(files)==30,(path,len(files))
 result=[Image.open(p).convert('RGB') for p in files]
 if size:result=[im.resize(size,Image.Resampling.BILINEAR) for im in result]
 return result
def exact_video(path,seconds,nframes,max_side):
 seq=images(path);assert seq[0].size==(W,H)
 return seq,10.,(W,H),int(np.ceil(len(seq)/nframes)),len(seq)
def exact_diffusion_mask(path,fps,count,size,dilation,frames):
 assert fps==10 and size==(W,H) and count==30 and dilation==0
 ms=[Image.fromarray(mask(pathlib.Path(path)/f'{i:05}.png').astype('uint8')*255) for i in range(count)]
 masked=[Image.fromarray(np.where(np.array(m)[...,None]>0,0,np.array(f)).astype('uint8')) for f,m in zip(frames,ms)]
 return ms,masked
def exact_prior_mask(path,count,size,flow_mask_dilates=0,mask_dilates=0):
 assert flow_mask_dilates==0 and mask_dilates==0 and count==30
 ms=[Image.open(pathlib.Path(path)/f'{i:05}.png').convert('L').resize(size,Image.Resampling.NEAREST) for i in range(count)]
 return ms,ms
def capture(seq,out,sub):
 dest=pathlib.Path(out).parent/sub;dest.mkdir(exist_ok=False)
 for i,im in enumerate(seq):Image.fromarray(np.asarray(im).astype('uint8')).save(dest/f'{i:05}.png')
def module(path,name,substitutions):
 code=path.read_text()
 for before,after in substitutions:
  assert code.count(before)==1,(path,before,code.count(before));code=code.replace(before,after)
 m=types.ModuleType(name);m.__file__=str(path);m.capture=capture
 exec(compile(code,str(path),'exec'),m.__dict__)
 return m

def main():
 assert torch.cuda.is_available()
 assert (ROOT/'evidence_summary.json').exists()
 assert not (ROOT/'generation_state.json').exists()
 scenes=read(ROOT/'registration.json')['scenes'];started=time.time()
 state=dict(state='running',pid=os.getpid(),started=time.time(),scenes={},seed=42,model='DiffuEraser official 2-Step',training=False,human_verdict=None)
 def save():dump(ROOT/'generation_state.json',state)
 save()
 # 首先固定全部帧的输入合同，避免模型运行期间更改mask。
 for s in scenes:
  out=ROOT/s['name']
  for i in range(30):
   m=load_masks(out,i);assert not (m['generate']&m['protect']).any();assert not (m['observed']&m['residual_generate']).any();assert np.array_equal(m['generate'],m['observed']|m['residual_generate']);assert np.array_equal(m['delete']&~m['observed'],m['residual_delete'])
   before=rgb(out/'rgb'/f'{i:05}.png');e=rgb(out/'evidence'/f'{i:05}.png');assert np.array_equal(before[~m['observed']],e[~m['observed']])
 dump(ROOT/'generation_input_preflight.json',dict(frames=60,mask_partition=True,evidence_outside_unchanged=True,canonical_png=True,model_mask_extra_dilation=0,prior_preserved=True))
 prior=module(SOURCE/'propainter/inference.py','v77_prior', [('        ##save composed video##','        capture(comp_frames, output_path, "prior_png")\n        ##save composed video##')])
 prior.read_frame_from_videos=lambda p,l:(images(p),10.,(W,H),'v77',30)
 prior.read_mask=exact_prior_mask
 try:
  p=prior.Propainter(str(WEIGHTS/'propainter'),device=torch.device('cuda'))
  for s in scenes:
   out=ROOT/s['name'];state['phase']='prior';state['scene']=s['name'];save()
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   t=time.time();p.forward(str(out/'evidence'),str(out/'residual_generate'),str(out/'prior_native.mp4'),video_length=3,mask_dilation=0,save_fps=10,resize_ratio=.6,neighbor_length=10,subvideo_length=50)
   state['scenes'][s['name']]=dict(prior_seconds=time.time()-t);save()
  del p;gc.collect();torch.cuda.empty_cache()
  de=module(SOURCE/'diffueraser/diffueraser.py','v77_diffueraser',[
   ('        ################ Compose ################','        capture(images, output_path, "native_png")\n        ################ Compose ################'),
   ('        default_fps = fps','        capture(comp_frames, output_path, "native_composite_png")\n        default_fps = fps')])
  de.read_video=exact_video;de.read_mask=exact_diffusion_mask
  de.read_priori=lambda p,fps,n,size:images(p,size)
  d=de.DiffuEraser(torch.device('cuda'),str(WEIGHTS/'stable-diffusion-v1-5'),str(WEIGHTS/'sd-vae-ft-mse'),str(WEIGHTS/'diffuEraser'),ckpt='2-Step')
  for s in scenes:
   out=ROOT/s['name'];state['phase']='diffusion';state['scene']=s['name'];save()
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
   t=time.time();d.forward(str(out/'evidence'),str(out/'residual_generate'),str(out/'prior_png'),str(out/'diffueraser_native.mp4'),max_img_size=960,video_length=3,mask_dilation_iter=0,seed=42,guidance_scale=0,blended=False)
   dest=out/'final';dest.mkdir(exist_ok=False)
   for i in range(30):
    m=load_masks(out,i);before=rgb(out/'rgb'/f'{i:05}.png');e=rgb(out/'evidence'/f'{i:05}.png');generated=rgb(out/'native_png'/f'{i:05}.png');final=compose_background(before,e,generated,m)
    assert np.array_equal(final[m['protect']],before[m['protect']]);assert np.array_equal(final[m['observed']],e[m['observed']]);assert np.array_equal(final[~m['generate']],before[~m['generate']])
    Image.fromarray(final).save(dest/f'{i:05}.png')
   state['scenes'][s['name']].update(diffusion_seconds=time.time()-t,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,output_frames=30);save()
   print('SCENE_COMPLETE',s['name'],state['scenes'][s['name']],flush=True)
  state.update(state='complete',elapsed_seconds=time.time()-started);save()
 except Exception as exc:
  state.update(state='failed_engineering',error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.time()-started);save();raise
if __name__=='__main__':main()

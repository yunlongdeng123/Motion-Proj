"""r4：只改变生成器所见车列上下文；最终操作区域完全沿用r3。"""
import sys,os,time,gc,random,datetime,shutil
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_diffueraser as io
from hybrid_common import *
import torch
R2=ROOT;R3=ROOT.parent/'r3';ROOT=ROOT.parent/'r4'
def prepare():
 assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
 cfg=read(R3/'registration.json');cfg.update(run_id='r4',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='parked vehicle-row context drives class repetition even with random initial hole; hide vehicle-row context from model while restoring all non-target pixels on writeback',fixed=['r3 output write masks','weights','seed42','PCM2Step','prior algorithm'],scene_context_rule={'scene_0230':'bounding rectangle of original detected vehicles whose pixel centroid lies in left half, union target condition','scene_0255':'bounding rectangle of original detected vehicle row, union target condition'},input_selection='assistant image-guided scene rule; not universal automatic system',fallback=None,human_verdict=None)
 dump(ROOT/'registration.json',cfg)
 for s in cfg['scenes']:
  old=R2/s['name'];r3=R3/s['name'];out=ROOT/s['name'];out.mkdir()
  for sub in ['condition','masked']:(out/sub).mkdir()
  for sub in ['input','write','observed']:(out/sub).symlink_to(r3/sub,target_is_directory=True)
  stats=[]
  for i,f in enumerate(s['source_frames']):
   union=mask(r3/'condition'/f'{i:05}.png');count=0
   with np.load(OLD/s['name']/'guard'/f'source_masks_{i:05}.npz') as z:
    for key in z.files:
     m=cv2.resize(z[key].astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0;yy,xx=np.where(m)
     if not len(xx):continue
     if s['name']=='scene_0230' and xx.mean()>W/2:continue
     union|=m;count+=1
   y,x=np.where(union);rect=np.zeros(HW,bool);b=[max(0,int(x.min())-8),max(0,int(y.min())-8),min(W,int(x.max())+9),min(H,int(y.max())+9)];rect[b[1]:b[3],b[0]:b[2]]=True;write_mask(out/'condition'/f'{i:05}.png',rect)
   orig=rgb(r3/'input'/f'{i:05}.png');Image.fromarray(np.where(rect[...,None],0,orig).astype('uint8')).save(out/'masked'/f'{i:05}.png');stats.append(dict(frame=f,context_box=b,source_instances=count,model_mask_pixels=int(rect.sum()),write_mask_pixels=int(mask(r3/'write'/f'{i:05}.png').sum())))
  dump(out/'context_stats.json',stats)
  sheet=Image.new('RGB',(960,536*4))
  for j,i in enumerate([0,7,15,29]):sheet.paste(Image.open(out/'masked'/f'{i:05}.png'),(0,536*j))
  sheet.resize((640,1429)).save(out/'context_review.jpg',quality=92)
 print('CONTEXT PREPARED',flush=True)
def run():
 assert not (ROOT/'state.json').exists();cfg=read(ROOT/'registration.json');state=dict(state='running',pid=os.getpid(),started=time.time(),scenes={},human_verdict=None);dump(ROOT/'state.json',state)
 pm=io.module(io.SOURCE/'propainter/inference.py','context_prior',[('        ##save composed video##','        capture(comp_frames, output_path, "prior_png")\n        ##save composed video##')]);pm.read_frame_from_videos=lambda p,l:(io.images(p),10.,(W,H),'context',30);pm.read_mask=io.exact_prior_mask
 p=pm.Propainter(str(io.WEIGHTS/'propainter'),device=torch.device('cuda'))
 for s in cfg['scenes']:
  out=ROOT/s['name'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  p.forward(str(out/'input'),str(out/'condition'),str(out/'prior_native.mp4'),video_length=3,mask_dilation=0,save_fps=10,resize_ratio=.6,neighbor_length=10,subvideo_length=50)
 del p;gc.collect();torch.cuda.empty_cache()
 dm=io.module(io.SOURCE/'diffueraser/diffueraser.py','context_diffusion',[('        ################ Compose ################','        capture(images, output_path, "native_png")\n        ################ Compose ################')]);dm.read_video=io.exact_video;dm.read_mask=io.exact_diffusion_mask;dm.read_priori=lambda p,fps,n,size:io.images(p,size)
 d=dm.DiffuEraser(torch.device('cuda'),str(io.WEIGHTS/'stable-diffusion-v1-5'),str(io.WEIGHTS/'sd-vae-ft-mse'),str(io.WEIGHTS/'diffuEraser'),ckpt='2-Step')
 for s in cfg['scenes']:
  out=ROOT/s['name'];old=R2/s['name'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  d.forward(str(out/'input'),str(out/'condition'),str(out/'prior_png'),str(out/'native.mp4'),max_img_size=960,video_length=3,mask_dilation_iter=0,seed=42,guidance_scale=0,blended=False)
  (out/'final').mkdir();checks=[]
  for i,f in enumerate(s['source_frames']):
   orig=rgb(old/'rgb'/f'{i:05}.png');wr=mask(out/'write'/f'{i:05}.png');obs=mask(out/'observed'/f'{i:05}.png');native=rgb(out/'native_png'/f'{i:05}.png');e=rgb(out/'input'/f'{i:05}.png');final=orig.copy();final[wr]=native[wr];final[obs]=e[obs];protect=mask(old/'protect'/f'{i:05}.png');assert np.array_equal(final[protect],orig[protect]);assert np.array_equal(final[~wr],orig[~wr]);Image.fromarray(final).save(out/'final'/f'{i:05}.png');checks.append(dict(frame=f,protect_changes=0,outside_changes=0))
  dump(out/'pixel_checks.json',checks)
  from PIL import ImageDraw
  sheet=Image.new('RGB',(420*4,236*4))
  for j,i in enumerate([0,7,15,29]):
   m=mask(old/'core'/f'{i:05}.png');y,x=np.where(m);cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W);left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
   for k,(name,path) in enumerate([('original',old/'rgb'),('r3 rect',R3/s['name']/'final'),('r4 row prior',out/'prior_png'),('r4 row context',out/'final')]):
    im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{name} f{s["source_frames"][i]}',fill='yellow');sheet.paste(im,(k*420,j*236))
  sheet.save(out/'quicklook.jpg',quality=94);state['scenes'][s['name']]=dict(frames=30,complete=True);dump(ROOT/'state.json',state);print('CONTEXT_COMPLETE',s['name'],flush=True)
 state.update(state='complete',elapsed_seconds=time.time()-state['started']);dump(ROOT/'state.json',state)
if __name__=='__main__':
 if sys.argv[-1]=='prepare':prepare()
 elif sys.argv[-1]=='run':run()
 else:raise ValueError('prepare/run required')

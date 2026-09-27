"""r5：r3局部条件 + r4无车prior；仅替换C→D桥，保存所有对照。"""
import sys,os,time,random,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_diffueraser as io
from hybrid_common import *
import torch
R2=ROOT;R3=ROOT.parent/'r3';R4=ROOT.parent/'r4';ROOT=ROOT.parent/'r5'
def prepare():
 assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
 cfg=read(R3/'registration.json');cfg.update(run_id='r5',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='r4 actor-free prior with r3 original local structure condition suppresses vehicle without broad-context structural drift',fixed=['r3 RGB','r3 mask','r3 writeback','weights','seed42','PCM2Step'],change='prior inside r3 condition from r4; outside condition exact original source',prior_role='synthetic proposal, not factual evidence',source_runs=['r3','r4'],fallback=None,human_verdict=None);dump(ROOT/'registration.json',cfg)
 for s in cfg['scenes']:
  base=R3/s['name'];old=R2/s['name'];out=ROOT/s['name'];out.mkdir();(out/'prior_png').mkdir()
  for key in ['input','condition','write','observed']:(out/key).symlink_to(base/key,target_is_directory=True)
  for i in range(30):
   original=rgb(base/'input'/f'{i:05}.png');p=rgb(R4/s['name']/'prior_png'/f'{i:05}.png');m=mask(base/'condition'/f'{i:05}.png');original[m]=p[m];Image.fromarray(original).save(out/'prior_png'/f'{i:05}.png')
 print('SWAP PREPARED',flush=True)
def run():
 assert not (ROOT/'state.json').exists();cfg=read(ROOT/'registration.json');state=dict(state='running',pid=os.getpid(),started=time.time(),scenes={},human_verdict=None);dump(ROOT/'state.json',state)
 mod=io.module(io.SOURCE/'diffueraser/diffueraser.py','prior_swap',[('        ################ Compose ################','        capture(images, output_path, "native_png")\n        ################ Compose ################')]);mod.read_video=io.exact_video;mod.read_mask=io.exact_diffusion_mask;mod.read_priori=lambda p,fps,n,size:io.images(p,size)
 model=mod.DiffuEraser(torch.device('cuda'),str(io.WEIGHTS/'stable-diffusion-v1-5'),str(io.WEIGHTS/'sd-vae-ft-mse'),str(io.WEIGHTS/'diffuEraser'),ckpt='2-Step')
 for s in cfg['scenes']:
  out=ROOT/s['name'];old=R2/s['name'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  model.forward(str(out/'input'),str(out/'condition'),str(out/'prior_png'),str(out/'native.mp4'),max_img_size=960,video_length=3,mask_dilation_iter=0,seed=42,guidance_scale=0,blended=False)
  (out/'final').mkdir();checks=[]
  for i,f in enumerate(s['source_frames']):
   orig=rgb(old/'rgb'/f'{i:05}.png');wr=mask(out/'write'/f'{i:05}.png');obs=mask(out/'observed'/f'{i:05}.png');native=rgb(out/'native_png'/f'{i:05}.png');e=rgb(out/'input'/f'{i:05}.png');final=orig.copy();final[wr]=native[wr];final[obs]=e[obs];protect=mask(old/'protect'/f'{i:05}.png');assert np.array_equal(final[protect],orig[protect]);assert np.array_equal(final[~wr],orig[~wr]);Image.fromarray(final).save(out/'final'/f'{i:05}.png');checks.append(dict(frame=f,protect_changes=0,outside_changes=0))
  dump(out/'pixel_checks.json',checks)
  from PIL import ImageDraw
  sheet=Image.new('RGB',(420*4,236*4))
  for j,i in enumerate([0,7,15,29]):
   m=mask(old/'core'/f'{i:05}.png');y,x=np.where(m);cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W);left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
   for k,(name,path) in enumerate([('original',old/'rgb'),('r3 rect',R3/s['name']/'final'),('r5 prior',out/'prior_png'),('r5 local+prior',out/'final')]):
    im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{name} f{s["source_frames"][i]}',fill='yellow');sheet.paste(im,(k*420,j*236))
  sheet.save(out/'quicklook.jpg',quality=94);state['scenes'][s['name']]=dict(frames=30,complete=True);dump(ROOT/'state.json',state);print('SWAP_COMPLETE',s['name'],flush=True)
 state.update(state='complete',elapsed_seconds=time.time()-state['started']);dump(ROOT/'state.json',state)
if __name__=='__main__':
 if sys.argv[-1]=='prepare':prepare()
 elif sys.argv[-1]=='run':run()
 else:raise ValueError('prepare/run required')

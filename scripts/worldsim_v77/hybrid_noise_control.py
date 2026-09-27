"""预登记fallback：只改变prior→diffusion桥，模型条件/seed不动。"""
import sys,os,time,random,gc
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_diffueraser as io
from hybrid_common import *
import torch
R2=ROOT;ROOT=ROOT.parent/'r3';DEST=ROOT/'noise_control';DEST.mkdir(exist_ok=False)
assert read(ROOT/'state.json')['state']=='complete'
dump(DEST/'registration.json',dict(arm='noise_control',registered_before_generation=True,parent='r3',change='replace initial diffusion latents by same seeded Gaussian noise inside full rectangular condition only; identical prior latents outside',fixed=['RGB','mask','model','seed42','PCM2Step','guidance0','protect writeback'],reason='prior probe shows unknown area filled with repeated vehicle content at Transformer stage',human_verdict=None))
current={}
def reset_unknown(latents,noise,maskpath):
 mm=np.stack([mask(Path(maskpath)/f'{i:05}.png') for i in range(30)])
 # 大mask条件在latent上的最大池，避免VAE边界卷积把prior车形带回洞边。
 m=torch.from_numpy(mm[:,None].astype('float32')).to(latents.device)
 m=torch.nn.functional.adaptive_max_pool2d(m,latents.shape[-2:])>0
 dump(Path(current['out'])/'latent_mask.json',dict(latent_pixels=int(m.sum()),latent_total=m.numel(),method='adaptive_max_pool of unchanged image condition',source='r3 same-seed noise tensor',human_verdict=None))
 return torch.where(m,noise,latents)
changes=[('        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)','        noisy_latents = reset_unknown(self.noise_scheduler.add_noise(latents, noise, timesteps), noise, validation_mask)'),('        ################ Compose ################','        capture(images, output_path, "native_png")\n        ################ Compose ################')]
mod=io.module(io.SOURCE/'diffueraser/diffueraser.py','noise_control',changes);mod.reset_unknown=reset_unknown;mod.read_video=io.exact_video;mod.read_mask=io.exact_diffusion_mask;mod.read_priori=lambda p,fps,n,size:io.images(p,size)
model=mod.DiffuEraser(torch.device('cuda'),str(io.WEIGHTS/'stable-diffusion-v1-5'),str(io.WEIGHTS/'sd-vae-ft-mse'),str(io.WEIGHTS/'diffuEraser'),ckpt='2-Step')
state=dict(state='running',pid=os.getpid(),started=time.time(),scenes={},human_verdict=None);dump(DEST/'state.json',state)
for s in read(ROOT/'registration.json')['scenes']:
 base=ROOT/s['name'];old=R2/s['name'];out=DEST/s['name'];out.mkdir();current['out']=str(out)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 model.forward(str(base/'input'),str(base/'condition'),str(base/'prior_png'),str(out/'native.mp4'),max_img_size=960,video_length=3,mask_dilation_iter=0,seed=42,guidance_scale=0,blended=False)
 (out/'final').mkdir();checks=[]
 for i,f in enumerate(s['source_frames']):
  orig=rgb(old/'rgb'/f'{i:05}.png');generated=rgb(out/'native_png'/f'{i:05}.png');wr=mask(base/'write'/f'{i:05}.png');obs=mask(base/'observed'/f'{i:05}.png');e=rgb(base/'input'/f'{i:05}.png');final=orig.copy();final[wr]=generated[wr];final[obs]=e[obs]
  p=mask(old/'protect'/f'{i:05}.png');assert np.array_equal(final[p],orig[p]);assert np.array_equal(final[~wr],orig[~wr]);Image.fromarray(final).save(out/'final'/f'{i:05}.png');checks.append(dict(frame=f,protect_changes=0,outside_changes=0))
 dump(out/'pixel_checks.json',checks)
 from PIL import ImageDraw
 sheet=Image.new('RGB',(420*4,236*4))
 for j,i in enumerate([0,7,15,29]):
  m=mask(old/'core'/f'{i:05}.png');y,x=np.where(m);cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W);left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
  for k,(name,path) in enumerate([('original',old/'rgb'),('r2',old/'final'),('r3 rectangle',base/'final'),('r3 noise',out/'final')]):
   im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{name} f{s["source_frames"][i]}',fill='yellow');sheet.paste(im,(k*420,j*236))
 sheet.save(out/'quicklook.jpg',quality=94);state['scenes'][s['name']]=dict(frames=30,complete=True);dump(DEST/'state.json',state);print('NOISE_COMPLETE',s['name'],flush=True)
state.update(state='complete',elapsed_seconds=time.time()-state['started']);dump(DEST/'state.json',state)

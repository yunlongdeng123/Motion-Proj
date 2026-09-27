"""r7只审计0255的C→D；不改模型/条件，保存VAE和每步latent解码。"""
import sys,os,time,random,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_diffueraser as io
from hybrid_common import *
import torch
R2=ROOT;R6=ROOT.parent/'r6';ROOT=ROOT.parent/'r7';NAME='scene_0255'
assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r7',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene=NAME,source_run='r6',role='instrumentation only, no new candidate',fixed=['RGB','mask','prior','seed42','PCM2Step','weights'],steps=['prior VAE decode','noisy initialization decode','step1 noisy latent decode','step2 final decode'],warning='intermediate noisy latent decoding is not a clean-image estimate',failure_ledger_refs=['V77-F02'],human_verdict=None))
state=dict(state='running',pid=os.getpid(),started=time.time(),human_verdict=None);dump(ROOT/'state.json',state)
out=ROOT/NAME;out.mkdir();src=R6/NAME; records=[]
def decode_stage(vae,latents,stage):
 dest=out/stage;dest.mkdir(exist_ok=False)
 with torch.no_grad():
  for i in range(len(latents)):
   x=vae.decode(latents[i:i+1]/vae.config.scaling_factor).sample
   im=(x.float()/2+.5).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
   Image.fromarray(np.rint(im*255).astype('uint8')).save(dest/f'{i:05}.png')
 records.append(dict(stage=stage,frames=len(latents),latent_shape=list(latents.shape)));dump(out/'stages.json',records)
def trace_initial(model,latents,noisy):
 decode_stage(model.vae,latents,'D0_vae');decode_stage(model.vae,noisy,'D0_noisy')
def step_callback(pipe,step,timestep,kwargs):
 decode_stage(pipe.vae,kwargs['latents'],f'D{step+1}_step')
 records[-1].update(timestep=int(timestep));dump(out/'stages.json',records)
 return kwargs
mod=io.module(io.SOURCE/'diffueraser/diffueraser.py','diffusion_probe',[
 ('        ################ Compose ################','        capture(images, output_path, "native_png")\n        ################ Compose ################'),
 ('        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) ','        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)\n        trace_initial(self, latents, noisy_latents)'),
 ('                latents=latents,','                latents=latents,\n                callback_on_step_end=step_callback,')])
mod.trace_initial=trace_initial;mod.step_callback=step_callback
mod.read_video=io.exact_video;mod.read_mask=io.exact_diffusion_mask;mod.read_priori=lambda p,fps,n,size:io.images(p,size)
model=mod.DiffuEraser(torch.device('cuda'),str(io.WEIGHTS/'stable-diffusion-v1-5'),str(io.WEIGHTS/'sd-vae-ft-mse'),str(io.WEIGHTS/'diffuEraser'),ckpt='2-Step')
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
model.forward(str(src/'input'),str(src/'condition'),str(src/'prior_png'),str(out/'native.mp4'),max_img_size=960,video_length=3,mask_dilation_iter=0,seed=42,guidance_scale=0,blended=False)
errors=[int(np.max(np.abs(rgb(out/'native_png'/f'{i:05}.png').astype('int16')-rgb(src/'native_png'/f'{i:05}.png').astype('int16')))) for i in range(30)]
dump(out/'reproduction.json',dict(per_frame_max_abs=errors,maximum=max(errors),exact=not any(errors)))
from PIL import ImageDraw
sheet=Image.new('RGB',(420*5,236*4))
for j,i in enumerate([0,7,15,29]):
 y,x=np.where(mask(R2/NAME/'core'/f'{i:05}.png'));cw=400;ch=round(cw*H/W);left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
 for k,(name,path) in enumerate([('C prior',src/'prior_png'),('D0 VAE',out/'D0_vae'),('D1 noisy state',out/'D1_step'),('D2 raw diffusion',out/'native_png'),('E final writeback',src/'final')]):
  im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{name} f{65+i}',fill='yellow');sheet.paste(im,(k*420,j*236))
sheet.save(out/'quicklook.jpg',quality=94)
state.update(state='complete',elapsed_seconds=time.time()-state['started'],reproduction_max_abs=max(errors));dump(ROOT/'state.json',state);print('DIFFUSION PROBE COMPLETE',max(errors),flush=True)

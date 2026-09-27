"""扩散子阶段、写回控制与全部帧接触表；可追溯中间检查点。"""
import sys,subprocess,shutil
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
import imageio_ffmpeg
from PIL import ImageDraw
R2=ROOT;BASE=ROOT.parent;DEST=BASE/'stage_review';name='scene_0255';out=DEST/name
v=read(DEST/'video_validation.json');assert v['total_videos']==42
streams={'r7_vae':BASE/'r7'/name/'D0_vae','r7_noisy':BASE/'r7'/name/'D1_step','r7_native':BASE/'r7'/name/'native_png','r8':BASE/'r8'/name/'final','r8_no_static':BASE/'r8'/name/'no_static','r8_mask':BASE/'r8'/name/'mask_overlay','r8_full':BASE/'r8'/name/'final','r9':BASE/'r9'/name/'final','r9_full':BASE/'r9'/name/'final'}
for key,path in streams.items():
 frames=out/'frames'/key;frames.mkdir(exist_ok=False);zoom=not key.endswith('_full')
 for i in range(30):
  im=Image.fromarray(rgb(path/f'{i:05}.png'))
  if zoom:
   y,x=np.where(mask(R2/name/'core'/f'{i:05}.png'));cw=400;ch=round(cw*H/W);l=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));t=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)));im=im.crop((l,t,l+cw,t+ch)).resize((W,H),Image.Resampling.BILINEAR)
  im.save(frames/f'{i:05}.png')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(frames/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{key}.mp4')],check=True)
 Image.open(frames/'00000.png').save(out/f'{key}_poster.jpg',quality=94)
 cap=cv2.VideoCapture(str(out/f'{key}.mp4'));n=0
 while True:
  ok,f=cap.read()
  if not ok:break
  assert f.shape[:2]==HW;n+=1
 cap.release();assert n==30
 v['videos'].append(dict(scene=name,video=key,decoded=n,fps=10,wh=[W,H],zoom=zoom,source=str(path)))
 print('EXTENDED',key,flush=True)
v.update(total_videos=len(v['videos']),decoded_frames=sum(r['decoded'] for r in v['videos']));dump(DEST/'video_validation.json',v)
for run in ['r7','r8','r9']:shutil.copy2(BASE/run/name/'quicklook.jpg',out/f'{run}_quicklook.jpg')
for scene,run,key in [('scene_0230','r6','r6'),('scene_0255','r6','r6'),('scene_0255','r9','r9')]:
 sheet=Image.new('RGB',(4*360,8*211),(10,15,24))
 for i in range(30):
  image=Image.open(DEST/scene/'frames'/key/f'{i:05}.png').resize((360,201));sheet.paste(image,((i%4)*360,(i//4)*211));ImageDraw.Draw(sheet).text(((i%4)*360+5,(i//4)*211+5),f'{scene} f{(18 if scene=="scene_0230" else 65)+i}',fill='yellow')
 sheet.save(DEST/scene/f'{run}_all30.jpg',quality=94)
summary=dict(task_id='WS-V77-HYBRID-BG-20260927',runs={},guard={},human_verdict=None,goal_complete=False,background_input_dir=None,failure_ledger_refs=['V77-F02'],failure_ledger_delta='updated V77-F02')
for run in ['r3','r4','r5','r6','r7','r8','r9']:
 cfg=read(BASE/run/'registration.json')
 if 'scenes' in cfg:cfg['scenes']=[{k:s[k] for k in ['name','actor','camera','source_frames','track_id','data'] if k in s} for s in cfg['scenes']]
 summary['runs'][run]=dict(registration=cfg,state=read(BASE/run/'state.json'))
summary['runs']['r3']['noise_control']=read(BASE/'r3/noise_control/state.json')
summary['prior_probe']=read(BASE/'r3/prior_probe_summary.json')
summary['diffusion_probe_reproduction']=read(BASE/'r7/scene_0255/reproduction.json')
for run in ['r4','r6']:summary['guard'][run]=read(BASE/run/'guard_summary.json')
summary['writeback']=dict(r8=read(BASE/'r8/scene_0255/pixel_checks.json'),r9=read(BASE/'r9/scene_0255/pixel_checks.json'))
summary['interpretation']='Model-context suppression reduces detected vehicle recurrence but changes scene structure. r7 reproduces r6 exactly: VAE retains prior layout, raw diffusion is more coherent than final writeback. Fence protection contains source-car pixels; r8 removes coarse blobs but thin bars retain dark contamination. r9 blending cannot fix geometry or amodal neighboring vehicles. No candidate admitted.'
dump(DEST/'summary.json',summary)

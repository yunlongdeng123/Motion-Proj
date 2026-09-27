"""分阶段控制的视频证据；保留原生帧，浏览用H264不是像素合同证据。"""
import sys, subprocess, shutil
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
import imageio_ffmpeg
from PIL import ImageDraw
R2=ROOT; BASE=ROOT.parent; DEST=BASE/'stage_review'; DEST.mkdir(exist_ok=True)
rows=[]
for s in read(R2/'registration.json')['scenes']:
 name=s['name']; out=DEST/name; out.mkdir(exist_ok=True); old=R2/name; r3=BASE/'r3'/name
 streams={
  'original':old/'rgb','r2':old/'final','r3':r3/'final',
  'r3_noise':BASE/'r3/noise_control'/name/'final',
  'r4':BASE/'r4'/name/'final','r5':BASE/'r5'/name/'final','r6':BASE/'r6'/name/'final',
  'B_masked':r3/'prior_probe/B_masked','C1_propagated':r3/'prior_probe/C1_propagated',
  'C1_remaining':r3/'prior_probe/C1_remaining','C2_transformer':r3/'prior_probe/C2_transformer',
  'D_native':r3/'native_png','r4_condition':BASE/'r4'/name/'masked',
  'r6_condition':BASE/'r6'/name/'masked','r6_prior':BASE/'r6'/name/'prior_png',
 }
 assert all(len(list(p.glob('*.png')))==30 for p in streams.values())
 streams['target']=old/'rgb';streams['original_zoom']=old/'rgb'
 for key,path in streams.items():
  frames=out/'frames'/key;frames.mkdir(parents=True,exist_ok=True)
  # 目标和结果给原画幅及相同逐帧放大框；stage只放大，原PNG另存run中。
  zoom=key not in ['original','target','r4_condition','r6_condition']
  for i,f in enumerate(s['source_frames']):
   im=Image.fromarray(rgb(path/f'{i:05}.png'))
   y,x=np.where(mask(old/'core'/f'{i:05}.png'))
   if key=='target':
    d=ImageDraw.Draw(im);d.rectangle([int(x.min()),int(y.min()),int(x.max()),int(y.max())],outline='yellow',width=3)
    d.text((10,10),f'{name} actor{s["actor"]} CAM{s["camera"]} f{f}',fill='yellow')
   if zoom:
    cw=480 if name=='scene_0230' else 400;ch=round(cw*H/W)
    left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
    im=im.crop((left,top,left+cw,top+ch)).resize((W,H),Image.Resampling.BILINEAR)
   im.save(frames/f'{i:05}.png')
  subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(frames/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{key}.mp4')],check=True)
  Image.open(frames/'00000.png').save(out/f'{key}_poster.jpg',quality=94)
  cap=cv2.VideoCapture(str(out/f'{key}.mp4'));assert cap.isOpened();assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01;n=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.shape[:2]==HW;n+=1
  cap.release();assert n==30
  rows.append(dict(scene=name,video=key,decoded=n,fps=10,wh=[W,H],zoom=zoom,source=str(path)))
  print('PACKAGED',name,key,flush=True)
 # 当前轮全画幅原生最终图也保留，避免仅展示crop藏问题。
 for key in ['r3','r4','r5','r6']:
  path=streams[key]
  subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(path/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{key}_full.mp4')],check=True)
  Image.open(path/'00000.png').save(out/f'{key}_full_poster.jpg',quality=94)
  cap=cv2.VideoCapture(str(out/f'{key}_full.mp4'));n=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.shape[:2]==HW;n+=1
  cap.release();assert n==30;rows.append(dict(scene=name,video=key+'_full',decoded=n,fps=10,wh=[W,H],zoom=False,source=str(path)))
 for run in ['r3','r4','r5','r6']:
  p=BASE/run/name/('quicklook_instances.jpg' if run=='r6' else 'quicklook.jpg')
  shutil.copy2(p,out/f'{run}_quicklook.jpg')
 shutil.copy2(r3/'prior_probe/stages.jpg',out/'stages.jpg')
dump(DEST/'video_validation.json',dict(videos=rows,total_videos=len(rows),decoded_frames=sum(r['decoded'] for r in rows),human_verdict=None))

"""r18对应视角锚点修复的实际10帧对照。"""
from pathlib import Path
import json,subprocess,shutil
import numpy as np,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');RUN=BASE/'r18';OUT=BASE/'neighbor_review/anchor-repair'
assert json.loads((RUN/'state.json').read_text())['state']=='complete';OUT.mkdir()
keys=['condition','native','final','zoom'];contact=Image.new('RGB',(1800,3300),(10,20,30))
for k in keys:(OUT/'frames'/k).mkdir(parents=True)
for i in range(10):
 for k in ['condition','native','final']:shutil.copy2(RUN/k/f'{i:05}.png',OUT/'frames'/k/f'{i:05}.png')
 box=(270,238,710,480);Image.open(RUN/'final'/f'{i:05}.png').crop(box).resize((1024,576)).save(OUT/'frames/zoom'/f'{i:05}.png')
 for j,(title,path) in enumerate([('r15 feature-only',BASE/'r15/neighbor/native'/f'{i:05}.png'),('r18 anchor native',RUN/'native'/f'{i:05}.png'),('r18 after writeback',RUN/'final'/f'{i:05}.png')]):
  im=Image.open(path).crop(box).resize((600,330));ImageDraw.Draw(im).text((7,7),f'{title} f{65+i}',fill='yellow');contact.paste(im,(j*600,i*330))
contact.save(OUT/'all10_comparison.jpg',quality=96);videos=[]
for k in keys:
 f=OUT/f'{k}.mp4';folder=OUT/'frames'/k
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(f)],check=True)
 v=cv2.VideoCapture(str(f));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape==(576,1024,3);n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==10 and fps==10
 Image.open(folder/'00000.png').save(OUT/f'{k}_poster.jpg',quality=96);videos.append(dict(file=f.name,decoded_frames=n,fps=fps,width=1024,height=576))
for name in ['state.json','registration.json','condition_validation.json','source_sv3d.png','source_silhouette.png']:shutil.copy2(RUN/name,OUT/name)
(OUT/'video_validation.json').write_text(json.dumps(dict(videos=videos,decoded_frames=40),indent=2)+'\n');print('R18_PACKAGE_COMPLETE',len(videos))

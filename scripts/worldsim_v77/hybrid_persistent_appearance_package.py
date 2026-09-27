"""r24单变量实际DELETE对照，保持旧写回以分离生成器变化。"""
from pathlib import Path
import json,shutil,subprocess
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');RUN=BASE/'r24';OUT=BASE/'crop_review/persistent'
read=lambda p:json.loads(p.read_text());assert read(RUN/'state.json')['state']=='complete';assert not OUT.exists();OUT.mkdir()
sources=dict(old_native=BASE/'r18/native',new_native=RUN/'native',old_final=BASE/'r18/final',new_final=RUN/'final',condition=RUN/'condition')
for key in sources:(OUT/'frames'/key).mkdir(parents=True)
contacts=[Image.new('RGB',(1800,1760),(10,20,30)) for _ in range(2)]
for i in range(10):
 for k,p in sources.items():shutil.copy2(p/f'{i:05}.png',OUT/'frames'/k/f'{i:05}.png')
 raw=np.array(Image.open(BASE/'r15/input'/f'{i:05}.png'));im=np.array(Image.open(RUN/'native'/f'{i:05}.png'));out=np.array(Image.open(RUN/'final'/f'{i:05}.png'));m=np.array(Image.open(BASE/'r15/write'/f'{i:05}.png'))>0
 assert np.array_equal(out[~m],raw[~m]) and np.array_equal(out[m],im[m])
 for j,key in enumerate(['old_native','new_native','new_final']):
  tile=Image.new('RGB',(600,352),(10,20,30));a=Image.open(sources[key]/f'{i:05}.png').crop((270,238,710,480)).resize((600,330));tile.paste(a,(0,22));ImageDraw.Draw(tile).text((5,4),f'f{65+i} | {key}',fill='white');contacts[i//5].paste(tile,(600*j,352*(i%5)))
for j,s in enumerate(contacts):s.save(OUT/f'all10_part{j+1}.jpg',quality=96)
videos=[]
for key in sources:
 folder=OUT/'frames'/key;f=OUT/f'{key}.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(f)],check=True)
 v=cv2.VideoCapture(str(f));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape==(576,1024,3);n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==10 and fps==10;Image.open(folder/'00000.png').save(OUT/f'{key}_poster.jpg',quality=96);videos.append(dict(file=f.name,decoded_frames=n,width=1024,height=576,fps=fps))
for name in ['registration.json','condition_validation.json','state.json']:shutil.copy2(RUN/name,OUT/name)
(OUT/'video_validation.json').write_text(json.dumps(dict(videos=videos,decoded_frames=50),indent=2)+'\n');print('R24_PACKAGE_COMPLETE')

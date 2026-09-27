"""把r16实际原生输出制作成一秒同步审核视频，不冒充DELETE。"""
import sys,subprocess,shutil,json
from pathlib import Path
import numpy as np,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');RUN=BASE/'r16';OUT=BASE/'neighbor_review/positive-control'
assert json.loads((RUN/'state.json').read_text())['state']=='complete';OUT.mkdir()
keys=['input','anchor_condition','no_anchor_condition','anchor','no_anchor','anchor_zoom','no_anchor_zoom']
for k in keys:(OUT/'frames'/k).mkdir(parents=True)
contact=Image.new('RGB',(1800,3200),(10,20,30))
for i,f in enumerate(range(130,140)):
 m=np.array(Image.open(RUN/'box_support'/f'{i:05}.png'))>0;y,x=np.where(m);cx=(x.min()+x.max())//2;cy=(y.min()+y.max())//2;left=max(0,min(1024-260,cx-130));top=max(0,min(576-148,cy-74));crop=(left,top,left+260,top+148)
 for k in ['input','anchor_condition','no_anchor_condition']:shutil.copy2(RUN/k/f'{i:05}.png',OUT/'frames'/k/f'{i:05}.png')
 for arm in ['anchor','no_anchor']:
  im=Image.open(RUN/arm/'native'/f'{i:05}.png');im.save(OUT/'frames'/arm/f'{i:05}.png');im.crop(crop).resize((1024,576)).save(OUT/'frames'/f'{arm}_zoom'/f'{i:05}.png')
 for j,k in enumerate(['input','anchor','no_anchor']):
  im=Image.open(OUT/'frames'/k/f'{i:05}.png').crop(crop).resize((600,320));ImageDraw.Draw(im).text((7,7),f'{k} f{f}',fill='yellow');contact.paste(im,(j*600,i*320))
contact.save(OUT/'all10_comparison.jpg',quality=96)
videos=[]
for k in keys:
 path=OUT/f'{k}.mp4';folder=OUT/'frames'/k
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
 v=cv2.VideoCapture(str(path));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape==(576,1024,3);n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==10 and fps==10
 Image.open(folder/'00000.png').save(OUT/f'{k}_poster.jpg',quality=96);videos.append(dict(file=path.name,decoded_frames=n,fps=fps,width=1024,height=576))
for name in ['state.json','registration.json','condition_validation.json','reference_white.png','reference_sv3d.png','anchor_difference.png']:shutil.copy2(RUN/name,OUT/name)
(OUT/'video_validation.json').write_text(json.dumps(dict(videos=videos,decoded_frames=70,scope='Known-visible reconstruction diagnostic; no hidden DELETE ground truth'),indent=2)+'\n')
print('R16_PACKAGE_COMPLETE',len(videos))

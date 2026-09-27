"""r22后续时间窗与原r18接成3秒，显式标出窗口边界。"""
from pathlib import Path
import shutil,json,subprocess
import cv2,imageio_ffmpeg,numpy as np
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');RUN=BASE/'r22';OUT=BASE/'fence_review/temporal'
read=lambda p:json.loads(p.read_text())
assert read(RUN/'state.json')['state']=='complete';assert not OUT.exists();OUT.mkdir()
keys=['original','native','final','native_zoom']
for k in keys:(OUT/'frames'/k).mkdir(parents=True)
rows=[];sheets=[Image.new('RGB',(1800,1760),(10,20,30)) for _ in range(4)]
selected=Image.new('RGB',(1800,330*9),(10,20,30));jselect=0
for i in range(30):
 f=65+i
 if i<10:
  paths=dict(original=BASE/'r15/input'/f'{i:05}.png',native=BASE/'r18/native'/f'{i:05}.png',final=BASE/'r18/final'/f'{i:05}.png');wpath=BASE/'r15/write'/f'{i:05}.png';window=65
 else:
  window=(f-65)//10*10+65;folder=RUN/f'f{window:03}';idx=f-window
  paths={k:folder/('input' if k=='original' else k)/f'{idx:05}.png' for k in keys[:3]};wpath=folder/'write'/f'{idx:05}.png'
 ims={k:np.array(Image.open(p).convert('RGB')) for k,p in paths.items()};w=np.array(Image.open(wpath))>0
 assert np.array_equal(ims['original'][~w],ims['final'][~w]);assert np.array_equal(ims['native'][w],ims['final'][w]);rows.append(dict(frame=f,window=window,outside_write_changed=0,inside_native_mismatch=0))
 for k in keys:
  im=Image.fromarray(ims['native' if k=='native_zoom' else k])
  if k=='native_zoom':im=im.crop((270,238,710,480)).resize((1024,576))
  im.save(OUT/'frames'/k/f'{i:05}.png')
 if i>=10:
  for j,k in enumerate(keys[:3]):
   tile=Image.new('RGB',(600,352),(10,20,30));tile.paste(Image.fromarray(ims[k]).crop((270,238,710,480)).resize((600,330)),(0,22));ImageDraw.Draw(tile).text((5,4),f'f{f} | {k} | window {window}',fill='white');sheets[(i-10)//5].paste(tile,(600*j,352*((i-10)%5)))
 if i in [0,4,9,10,14,19,20,24,29]:
  for j,k in enumerate(keys[:3]):
   im=Image.fromarray(ims[k]).crop((270,238,710,480)).resize((600,330));ImageDraw.Draw(im).text((5,5),f'f{f} | {k} | window {window}',fill='yellow');selected.paste(im,(600*j,330*jselect))
  jselect+=1
for j,s in enumerate(sheets):s.save(OUT/f'all20_part{j+1}.jpg',quality=96)
selected.save(OUT/'selected_frames.jpg',quality=96)
videos=[]
for k in keys:
 f=OUT/f'{k}.mp4';folder=OUT/'frames'/k
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(f)],check=True)
 v=cv2.VideoCapture(str(f));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape==(576,1024,3);n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==30 and fps==10
 Image.open(folder/'00000.png').save(OUT/f'{k}_poster.jpg',quality=96);videos.append(dict(file=f.name,decoded_frames=n,fps=fps,width=1024,height=576))
for n in ['registration.json','condition_validation.json','state.json']:shutil.copy2(RUN/n,OUT/n)
for name,value in [('video_validation.json',dict(videos=videos,decoded_frames=120)),('evidence_validation.json',dict(checks=rows,first10='actual saved r18',new20='actual r22 independent windows; old writeback kept',human_verdict=None,background_input_dir=None))]:(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print('R22_REVIEW_PACKAGED',len(videos))

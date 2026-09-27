"""保存第三场景的原视频、旧方案、新条件及可见路面控制。"""
from pathlib import Path
import json,shutil,subprocess
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');OLD=BASE.parent/'WS-V77-DELETE-REPAIR-20260927/r1/official_000';RUN=BASE/'r28';OUT=BASE/'road_review'
read=lambda p:json.loads(p.read_text());assert read(RUN/'state.json')['state']=='complete';assert not OUT.exists();OUT.mkdir()
keys=['original','target','old','new','native','condition','mask']
for k in keys:(OUT/'frames'/k).mkdir(parents=True)
contacts=[Image.new('RGB',(1600,5*250),(12,20,30)) for _ in range(2)]
checks=[]
for f in range(10):
 im=np.array(Image.open(RUN/'input'/f'{f:05}.png'));old=np.array(Image.open(OLD/'precise'/f'{f:05}.png'));new=np.array(Image.open(RUN/'final'/f'{f:05}.png'));native=np.array(Image.open(RUN/'native'/f'{f:05}.png'));wr=np.array(Image.open(RUN/'write_mask'/f'{f:05}.png'))>0;m=np.array(Image.open(RUN/'model_mask'/f'{f:05}.png'))>0
 assert np.array_equal(im,np.array(Image.open(OLD/'rgb'/f'{f:05}.png')));assert np.array_equal(wr,np.array(Image.open(OLD/'mask'/f'{f:05}.png'))>0);assert np.array_equal(new[~wr],im[~wr]);assert np.array_equal(new[wr],native[wr]);assert np.all(m[wr])
 target=Image.fromarray(im);draw=ImageDraw.Draw(target);y,x=np.where(wr);box=(int(x.min()-3),int(y.min()-3),int(x.max()+3),int(y.max()+3));draw.rectangle(box,outline='yellow',width=2);draw.text((box[0]-12,box[1]-15),'DELETE actor12',fill='yellow')
 mask=im.copy();mask[m]=(.4*mask[m]+.6*np.array([20,130,240])).astype('uint8');contours,_=cv2.findContours(wr.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(mask,contours,-1,(255,220,30),1)
 images=dict(original=Image.fromarray(im),target=target,old=Image.fromarray(old),new=Image.fromarray(new),native=Image.fromarray(native),condition=Image.open(RUN/'condition'/f'{f:05}.png'),mask=Image.fromarray(mask))
 for k,v in images.items():v.save(OUT/'frames'/k/f'{f:05}.png')
 left=int(np.clip((x.min()+x.max())/2-160,0,704));top=int(np.clip((y.min()+y.max())/2-90,0,396));crop=(left,top,left+320,top+180)
 for j,k in enumerate(['target','old','new','native']):
  tile=Image.new('RGB',(400,250),(12,20,30));tile.paste(images[k].crop(crop).resize((400,225)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} | {k}',fill='white');contacts[f//5].paste(tile,(400*j,250*(f%5)))
 checks.append(dict(frame=f,original_exact=True,old_write_mask_exact=True,outside_changed=0,inside_equals_native=True))
for i,im in enumerate(contacts):im.save(OUT/f'all10_part{i+1}.jpg',quality=96)
videos=[]
for k in keys:
 file=OUT/f'{k}.mp4';folder=OUT/'frames'/k;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],check=True)
 cap=cv2.VideoCapture(str(file));count=0
 while True:
  ok,arr=cap.read()
  if not ok:break
  assert arr.shape==(576,1024,3);count+=1
 fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==10 and fps==10;videos.append(dict(file=file.name,frames=count,fps=fps));Image.open(folder/'00000.png').save(OUT/f'{k}_poster.jpg',quality=96)
for r in ['r25','r26','r27','r28']:
 (OUT/r).mkdir()
 for name in ['registration.json','state.json','plane_fit.json','visible_validation.json','condition_validation.json','source_patch_metrics.json']:
  src=BASE/r/name
  if src.exists():shutil.copy2(src,OUT/r/name)
for f in [0,10,20]:shutil.copy2(BASE/'r25'/f'f{f:03}/comparison.jpg',OUT/'r25'/f'f{f:03}_comparison.jpg')
shutil.copy2(BASE/'r25/plane_points.jpg',OUT/'r25/plane_points.jpg');shutil.copy2(BASE/'r27/contact.jpg',OUT/'r27/contact.jpg');shutil.copy2(BASE/'r27/source_005_ground.jpg',OUT/'r27/source_005_ground.jpg')
dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
dump(OUT/'validation.json',dict(videos=videos,decoded_frames=sum(v['frames'] for v in videos),pixel_contract=checks,human_verdict=None));print('ROAD_REVIEW_PACKAGE_COMPLETE',len(videos),sum(v['frames'] for v in videos),flush=True)

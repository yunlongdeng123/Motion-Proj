"""r23可见正控制：相同显示尺度与回原图尺度均展示。"""
from pathlib import Path
import json,shutil,subprocess
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');RUN=BASE/'r23';OUT=BASE/'crop_review'
read=lambda p:json.loads(p.read_text());assert read(RUN/'state.json')['state']=='complete';assert not OUT.exists();OUT.mkdir();x,y,w,h=read(RUN/'registration.json')['roi_xywh']
keys=['original','old_full','new_full','original_crop','old_crop','new_crop','condition']
for k in keys:(OUT/'frames'/k).mkdir(parents=True)
contacts=[Image.new('RGB',(1800,1760),(10,20,30)) for _ in range(2)]
for i in range(10):
 raw=np.array(Image.open(RUN/'input_full'/f'{i:05}.png'));old=np.array(Image.open(BASE/'r16/anchor/native'/f'{i:05}.png'));mask=np.array(Image.open(BASE/'r16/model_mask'/f'{i:05}.png'))>0;old_full=raw.copy();old_full[mask]=old[mask];new=np.array(Image.open(RUN/'restored_full'/f'{i:05}.png'));assert np.array_equal(new[~mask],raw[~mask])
 old_crop=Image.fromarray(old).crop((x,y,x+w,y+h)).resize((1024,576),Image.Resampling.BILINEAR)
 images=dict(original=Image.fromarray(raw),old_full=Image.fromarray(old_full),new_full=Image.fromarray(new),original_crop=Image.open(RUN/'input_crop'/f'{i:05}.png'),old_crop=old_crop,new_crop=Image.open(RUN/'native_crop'/f'{i:05}.png'),condition=Image.open(RUN/'condition'/f'{i:05}.png'))
 for k,im in images.items():im.save(OUT/'frames'/k/f'{i:05}.png')
 # 使用相同GT框在三图中取同一对象局部，禁止按生成结果重选crop。
 b=np.array(Image.open(RUN/'box_crop'/f'{i:05}.png'))>0;yy,xx=np.where(b);cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2;cw=max(280,xx.max()-xx.min()+80);ch=cw*330/600;cb=(cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2)
 for j,k in enumerate(['original_crop','old_crop','new_crop']):
  tile=Image.new('RGB',(600,352),(10,20,30));tile.paste(images[k].crop(cb).resize((600,330)),(0,22));ImageDraw.Draw(tile).text((6,4),f'f{130+i} | {k}',fill='white');contacts[i//5].paste(tile,(600*j,352*(i%5)))
for j,s in enumerate(contacts):s.save(OUT/f'all10_part{j+1}.jpg',quality=96)
videos=[]
for k in keys:
 f=OUT/f'{k}.mp4';folder=OUT/'frames'/k;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(f)],check=True)
 v=cv2.VideoCapture(str(f));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape==(576,1024,3);n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==10 and fps==10
 Image.open(folder/'00000.png').save(OUT/f'{k}_poster.jpg',quality=96);videos.append(dict(file=f.name,decoded_frames=n,width=1024,height=576,fps=fps))
for name in ['registration.json','condition_validation.json','state.json']:shutil.copy2(RUN/name,OUT/name)
(OUT/'video_validation.json').write_text(json.dumps(dict(videos=videos,decoded_frames=70),indent=2)+'\n')
rows=read(RUN/'state.json')['oracle_metrics'][1:];summary=dict(excluded_condition_frame=130,frames=9,old_mean_mae=float(np.mean([r['old_mae'] for r in rows])),crop_mean_mae=float(np.mean([r['crop_mae'] for r in rows])),metric_limit='Same original-scale GT box intersect model mask; includes background/occluders, no true hidden-background evaluation',human_verdict=None)
(OUT/'metric_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('R23_REVIEW_PACKAGED',summary)

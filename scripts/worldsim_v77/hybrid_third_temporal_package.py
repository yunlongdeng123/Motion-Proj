"""将实际完成的长序列与首窗写回控制统一打包，不以计划帧数替代产物。"""
from pathlib import Path
import json,shutil,subprocess
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'temporal_review';OLD=BASE.parent/'WS-V77-DELETE-REPAIR-20260927/r1/official_000';read=lambda p:json.loads(p.read_text());assert not ROOT.exists();ROOT.mkdir()
run=BASE/'r31';frames=sorted(int(p.stem) for p in (run/'final').glob('*.png'));assert frames==list(range(len(frames)));assert len(frames)>=19
for p in (run/'windows').glob('*/state.json'):assert read(p)['state']=='complete_pending_visual_review'
sets={'temporal':{'frames':frames,'keys':['target','old','new','native','scope']},'extent':{'frames':list(range(10)),'keys':['target','r28','rect_hard','rect_feather','rect_feather_keep','scope']}}
videos=[]
for group,config in sets.items():
 out=ROOT/group;out.mkdir();sheets=[Image.new('RGB',(1600,1250),(12,20,30)) for _ in range((len(config['frames'])+4)//5)]
 for key in config['keys']:(out/'frames'/key).mkdir(parents=True)
 for f in config['frames']:
  original=np.array(Image.open(run/'input'/f'{f:05}.png'));write=np.array(Image.open(run/'write_mask'/f'{f:05}.png'))>0;model=np.array(Image.open(run/'model_mask'/f'{f:05}.png'))>0;protect=np.array(Image.open(run/'protect'/f'{f:05}.png'))>0
  target=Image.fromarray(original);dr=ImageDraw.Draw(target);y,x=np.where(write);dr.rectangle((x.min()-2,y.min()-2,x.max()+2,y.max()+2),outline='yellow',width=2);dr.text((x.min()-8,y.min()-15),'DELETE actor12',fill='yellow')
  scope=original.copy();scope[model]=(.4*scope[model]+.6*np.array([0,190,160])).astype('uint8');scope[write]=(.4*scope[write]+.6*np.array([255,220,0])).astype('uint8');scope[protect]=[70,120,255]
  if group=='temporal':
   images=dict(target=target,old=Image.open(OLD/'precise'/f'{f:05}.png'),new=Image.open(run/'final'/f'{f:05}.png'),native=Image.open(run/'native'/f'{f:05}.png'),scope=Image.fromarray(scope));contact_keys=['target','old','new','native']
  else:
   images=dict(target=target,r28=Image.open(BASE/'r28/final'/f'{f:05}.png'),scope=Image.open(BASE/'r30/scope'/f'{f:05}.png'))
   for key in ['rect_hard','rect_feather','rect_feather_keep']:images[key]=Image.open(BASE/'r30'/key/f'{f:05}.png')
   contact_keys=['target','r28','rect_feather','rect_feather_keep']
  for key,im in images.items():im.save(out/'frames'/key/f'{f:05}.png')
  left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));box=(left,top,left+384,top+216)
  for col,key in enumerate(contact_keys):
   tile=Image.new('RGB',(400,250),(12,20,30));tile.paste(images[key].crop(box).resize((400,225)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} {key}',fill='white');sheets[f//5].paste(tile,(col*400,(f%5)*250))
 for i,s in enumerate(sheets):s.save(out/f'all_part{i+1}.jpg',quality=96)
 for key in config['keys']:
  file=out/f'{key}.mp4';folder=out/'frames'/key;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],check=True)
  cap=cv2.VideoCapture(str(file));count=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape==(576,1024,3);count+=1
  cap.release();assert count==len(config['frames']);Image.open(folder/'00000.png').save(out/f'{key}_poster.jpg',quality=96);videos.append(dict(file=f'{group}/{key}.mp4',decoded_frames=count,width=1024,height=576,fps=10))
for r in ['r30','r31']:
 dest=ROOT/r;dest.mkdir()
 for name in ['registration.json','state.json','condition_validation.json','closeout.json']:
  p=BASE/r/name
  if p.exists():shutil.copy2(p,dest/name)
for p in (run/'windows').glob('*/state.json'):
 dest=ROOT/'r31/windows'/p.parent.name;dest.mkdir(parents=True);shutil.copy2(p,dest/p.name)
for name in ['all10_part1.jpg','all10_part2.jpg']:shutil.copy2(BASE/'r30'/name,ROOT/'extent'/name)
# 同时刻重叠，左为保存的前窗输出，右为后窗重新预测；不是相邻时刻差。
sheet=Image.new('RGB',(1200,260*3),(12,20,30));overlap=[]
for j,start in enumerate([9,18,27]):
 wd=run/'windows'/f'{start:05}'
 if not (wd/'00_final.png').exists():continue
 a=Image.open(run/'final'/f'{start:05}.png');b=Image.open(wd/'00_final.png');mask=np.array(Image.open(run/'model_mask'/f'{start:05}.png'))>0;y,x=np.where(mask);left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));crop=(left,top,left+384,top+216)
 for col,(name,im) in enumerate([('prior saved',a),('next window predicted',b)]):
  tile=Image.new('RGB',(600,260),(12,20,30));tile.paste(im.crop(crop).resize((400,225)),(100,25));ImageDraw.Draw(tile).text((8,5),f'f{start} {name}',fill='white');sheet.paste(tile,(col*600,j*260))
 overlap.append(read(wd/'state.json'))
sheet.save(ROOT/'overlaps.jpg',quality=96)
(ROOT/'video_validation.json').write_text(json.dumps(dict(videos=videos,decoded_frames=sum(v['decoded_frames'] for v in videos),temporal_frames=len(frames),human_verdict=None),indent=2)+'\n');print('TEMPORAL_REVIEW_PACKAGED',len(frames),len(videos),sum(v['decoded_frames'] for v in videos),flush=True)

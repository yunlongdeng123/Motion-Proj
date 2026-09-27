"""r14正控制图片与r15实际10帧生成视频，保留原生/写回对照。"""
import sys,subprocess,shutil
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera
from PIL import ImageDraw
import imageio_ffmpeg
BASE=ROOT.parent;R2=ROOT;RUN=BASE/'r15';OUT=BASE/'neighbor_review';assert read(RUN/'state.json')['state']=='complete';OUT.mkdir(exist_ok=False)
spec=next(s for s in read(R2/'registration.json')['scenes'] if s['name']=='scene_0255');frames=read(OLD/'scene_0255/frames.json');HW2=(576,1024)
def full(p):return np.array(Image.open(p).convert('RGB'))
def resized_mask(p):return np.array(Image.open(p).convert('L').resize((1024,576),Image.Resampling.NEAREST))>127
def encode(folder,path):
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
 v=cv2.VideoCapture(str(path));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape[:2]==HW2;n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==10 and fps==10
 Image.open(folder/'00000.png').save(path.with_name(path.stem+'_poster.jpg'),quality=96)
 return dict(file=path.name,decoded_frames=n,fps=fps,height=576,width=1024)
keys=['target','masked','deletion','neighbor','deletion_native','neighbor_native','deletion_zoom','neighbor_zoom']
for k in keys:(OUT/'frames'/k).mkdir(parents=True)
contact=Image.new('RGB',(640*3,360*10),(10,20,30))
for i,f in enumerate(range(65,75)):
 original=full(RUN/'input'/f'{i:05}.png');fr=frames[str(f)];c,k=camera(fr,3,HW2);target=next(b for b in fr['all_boxes'] if b['actor_id']=='25');neighbor=next(b for b in fr['all_boxes'] if b['actor_id']=='52')
 a=Image.fromarray(original);draw=ImageDraw.Draw(a)
 for b,color in [(target,'yellow'),(neighbor,'cyan')]:
  rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW2);draw.rectangle(rect,outline=color,width=2);draw.text((rect[0],rect[1]-12),('DELETE ' if b is target else 'KEEP ')+b['actor_id'],fill=color)
 a.save(OUT/'frames/target'/f'{i:05}.png');shutil.copy2(RUN/'masked'/f'{i:05}.png',OUT/'frames/masked'/f'{i:05}.png')
 core=resized_mask(R2/'scene_0255/core'/f'{i:05}.png');y,x=np.where(core);left=max(0,min(1024-430,int((x.min()+x.max())/2-215)));top=max(0,min(576-242,int((y.min()+y.max())/2-121)));box=(left,top,left+430,top+242)
 for arm in ['deletion','neighbor']:
  for kind,key in [('final',arm),('native',arm+'_native')]:shutil.copy2(RUN/arm/kind/f'{i:05}.png',OUT/'frames'/key/f'{i:05}.png')
  Image.open(RUN/arm/'final'/f'{i:05}.png').crop(box).resize((1024,576)).save(OUT/'frames'/f'{arm}_zoom'/f'{i:05}.png')
 for j,(title,im) in enumerate([('original',original),('neutral deletion',full(RUN/'deletion/final'/f'{i:05}.png')),('known-neighbor condition',full(RUN/'neighbor/final'/f'{i:05}.png'))]):
  p=Image.fromarray(im).crop(box).resize((640,360));ImageDraw.Draw(p).text((7,7),f'{title} f{f}',fill='yellow');contact.paste(p,(j*640,i*360))
contact.save(OUT/'all10_comparison.jpg',quality=96)
videos=[encode(OUT/'frames'/key,OUT/f'{key}.mp4') for key in keys]
for name in ['reference_raw.jpg','reference_white.png','reference_clip.png','reference_sv3d.png','condition_validation.json','azimuth_audit.json','registration.json','state.json']:shutil.copy2(RUN/name,OUT/name)
for name in ['source_mask_review.jpg','visible_control.jpg','visible_summary.json','source_index.json']:shutil.copy2(BASE/'r14'/name,OUT/name)
dump(OUT/'video_validation.json',dict(videos=videos,decoded_frames=sum(v['decoded_frames'] for v in videos),scope='all videos from actual source / masks / generated PNG, 10Hz 1s; no synthetic animation'))
print('NEIGHBOR REVIEW PACKAGE READY',len(videos),flush=True)

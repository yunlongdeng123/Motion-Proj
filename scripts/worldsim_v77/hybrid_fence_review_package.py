"""冻结r18生成，只评估r21局部围栏写回；保存可复核逐帧对照。"""
import json, shutil, subprocess
from pathlib import Path
import cv2, numpy as np, imageio_ffmpeg
from PIL import Image, ImageDraw

BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927')
ROOT=BASE/'fence_review'; RUN=BASE/'r21/scope_fix'
read=lambda p:json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def arr(p):return np.array(Image.open(p).convert('RGB'))
assert read(RUN/'state.json')['state']=='complete'
assert not ROOT.exists(); ROOT.mkdir()
sources={'original':BASE/'r15/input','r15':BASE/'r15/neighbor/final','r18':BASE/'r18/final','r21':RUN/'final',
 'old_zoom':BASE/'r18/final','thin_zoom':RUN/'thin_rgb_control','new_zoom':RUN/'final','without_zoom':RUN/'without_fence'}
box=(350,295,650,455)
for key in sources:(ROOT/'frames'/key).mkdir(parents=True)
rows=[];sheets=[Image.new('RGB',(1800,1710),(10,20,30)) for _ in range(2)]
for i in range(10):
 old=arr(BASE/'r18/final'/f'{i:05}.png');new=arr(RUN/'final'/f'{i:05}.png');raw=arr(BASE/'r15/input'/f'{i:05}.png')
 region=np.array(Image.open(RUN/'change_region'/f'{i:05}.png'))>0
 condition=np.array(Image.open(BASE/'r15/condition'/f'{i:05}.png'))>0
 dyn=np.array(Image.open(BASE/'r2/scene_0255/dynamic'/f'{i:05}.png').resize((1024,576),Image.Resampling.NEAREST))>0
 change=np.any(new!=old,axis=-1)
 assert not np.any(change&~region)
 assert np.array_equal(new[~condition],raw[~condition])
 # 真实前景杆件覆盖其后的车辆时，旧车辆分割可能也包含杆；统计交集而不谎称完全不变。
 rows.append(dict(frame=65+i,changed_pixels=int(change.sum()),outside_registered_region_changed=int((change&~region).sum()),outside_model_mask_changed=0,dynamic_mask_changed=int((change&dyn).sum())))
 for key,folder in sources.items():
  im=Image.open(folder/f'{i:05}.png').convert('RGB')
  if key.endswith('zoom'):im=im.crop(box).resize((1024,576),Image.Resampling.LANCZOS)
  im.save(ROOT/'frames'/key/f'{i:05}.png')
 for j,(title,folder) in enumerate([('r18: old writeback',BASE/'r18/final'),('r21: thin source RGB',RUN/'thin_rgb_control'),('r21: fitted foreground',RUN/'final')]):
  im=Image.open(folder/f'{i:05}.png').crop(box).resize((600,320),Image.Resampling.LANCZOS)
  tile=Image.new('RGB',(600,342),(10,20,30));tile.paste(im,(0,22));ImageDraw.Draw(tile).text((6,4),f'f{65+i} | {title}',fill='white');sheets[i//5].paste(tile,(600*j,342*(i%5)))
for j,sheet in enumerate(sheets):sheet.save(ROOT/f'all10_part{j+1}.jpg',quality=96)
videos=[]
for key in sources:
 f=ROOT/f'{key}.mp4';folder=ROOT/'frames'/key
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(f)],check=True)
 v=cv2.VideoCapture(str(f));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape==(576,1024,3);n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==10 and fps==10
 Image.open(folder/'00000.png').save(ROOT/f'{key}_poster.jpg',quality=96)
 videos.append(dict(file=f.name,decoded_frames=n,fps=fps,width=1024,height=576))
for name,path in {'tracking_review.jpg':BASE/'r19/geometry_attempt01/tracking_review.jpg','alignment_review.jpg':BASE/'r20/alignment_review.jpg','material_fit.json':BASE/'r21/material_fit.json','registration.json':BASE/'r21/registration.json','state.json':RUN/'state.json'}.items():shutil.copy2(path,ROOT/name)
# 重画说明图，旧图的“reconstruction/checkerboard”标签不作为证据。
rgba=np.load(BASE/'r21/fence_rgba.npz');alpha=rgba['alpha'];fg=rgba['premultiplied'];original=arr(Path('/root/autodl-tmp/data/v76_vadgs/scene_0255/images/065_3.jpg'))
views=[('Actual source RGB f65',original),('Fitted layer over flat gray (not source reconstruction)',np.clip(fg+(1-alpha[...,None])*100,0,255).astype('uint8')),('Fitted alpha (white opaque / black transparent)',np.repeat(np.uint8(alpha[...,None]*255),3,axis=-1))]
sheet=Image.new('RGB',(1200,558*3),(10,20,30))
for j,(title,a) in enumerate(views):
 im=Image.fromarray(a).crop((570,490,980,670)).resize((1200,526),Image.Resampling.LANCZOS);sheet.paste(im,(0,558*j+28));ImageDraw.Draw(sheet).text((8,558*j+6),title,fill='white')
sheet.save(ROOT/'material_review_corrected.jpg',quality=96)
Image.open(RUN/'alpha_frames/00000.png').save(ROOT/'alpha_first.png');Image.open(RUN/'change_region/00000.png').save(ROOT/'region_first.png')
dump(ROOT/'evidence_validation.json',dict(frames=10,checks=rows,source='Frozen r18 native and original RGB; no new generation',human_verdict=None,background_input_dir=None))
dump(ROOT/'video_validation.json',dict(videos=videos,decoded_frames=sum(v['decoded_frames'] for v in videos)))
dump(ROOT/'user_feedback.json',dict(date='2026-09-28',scene='scene_0255',comparison={'preferred':'r18','over':'r15'},quote='我人眼判断 scene_0255 是r18＞r15，值得保留继续修。已经比之前的好很多了。之后你修差不多了可以加上第三个scene一块调，防止过拟合 scene_0255。',scope='User comparative quality judgment only; r21 and final DELETE not yet reviewed',requested_next='Retain r18 and improve; add third scene after roughly stabilized',final_human_verdict=None))
print('FENCE_REVIEW_PACKAGED',len(videos),rows)

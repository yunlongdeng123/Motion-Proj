"""两场景完整视频与逐帧证据打包；所有视频从真实30帧生成。"""
import subprocess,shutil
import cv2
import imageio_ffmpeg
from PIL import ImageDraw
from hybrid_common import *
PACKAGE=ROOT/'review';PACKAGE.mkdir(exist_ok=True)
rows=[]
for s in read(ROOT/'registration.json')['scenes']:
 out=ROOT/s['name'];dest=PACKAGE/s['name'];dest.mkdir(exist_ok=True);framesdir=dest/'frames';framesdir.mkdir(exist_ok=True)
 streams=['original','target','baseline','prior','native','final','masks','evidence','evidence_map','final_zoom','baseline_zoom','original_zoom']
 for key in streams:(framesdir/key).mkdir(exist_ok=True)
 for i,f in enumerate(s['source_frames']):
  original=rgb(out/'rgb'/f'{i:05}.png');baseline=rgb(FULL/s['name']/f'cam{s["camera"]}/background/{f:05}.png');prior=rgb(out/'prior_png'/f'{i:05}.png');native=rgb(out/'native_png'/f'{i:05}.png');final=rgb(out/'final'/f'{i:05}.png');evidence=rgb(out/'evidence'/f'{i:05}.png');m=load_masks(out,i)
  target=Image.fromarray(original);d=ImageDraw.Draw(target);yy,xx=np.where(m['delete']);rect=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())];d.rectangle(rect,outline=(255,215,0),width=3);d.text((10,10),f'{s["name"]} | actor {s["actor"]} | CAM{s["camera"]} | source f{f} | DELETE',fill=(255,215,0))
  overlay=original.copy()
  for area,color in [(m['protect'],[0,220,230]),(m['generate'],[60,80,255]),(m['delete'],[255,210,0])]:overlay[area]=(.45*overlay[area]+.55*np.array(color)).astype('uint8')
  em=original.copy();em[m['residual_generate']]=(.35*em[m['residual_generate']]+.65*np.array([230,70,125])).astype('uint8');em[m['observed']]=[0,255,100]
  images={'original':original,'target':np.array(target),'baseline':baseline,'prior':prior,'native':native,'final':final,'masks':overlay,'evidence':evidence,'evidence_map':em}
  # 固定像素尺寸跟随目标中心，三视频同框；不改变各组相对尺寸。
  cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W);cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2;l=round(max(0,min(W-cw,cx-cw/2)));t=round(max(0,min(H-ch,cy-ch/2)))
  for key in ['original','baseline','final']:images[key+'_zoom']=np.array(Image.fromarray(images[key]).crop((l,t,l+cw,t+ch)).resize((W,H),Image.Resampling.BILINEAR))
  for key,im in images.items():Image.fromarray(im).save(framesdir/key/f'{i:05}.png')
 for key in streams:
  cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(framesdir/key/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/f'{key}.mp4')]
  subprocess.run(cmd,check=True)
  Image.open(framesdir/key/'00000.png').save(dest/f'{key}_poster.jpg',quality=93)
  # 解码实际帧，不能只相信容器中的帧数声明。
  cap=cv2.VideoCapture(str(dest/f'{key}.mp4'));assert cap.isOpened();count=0
  assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.shape[:2]==(H,W);count+=1
  cap.release();assert count==30
  rows.append(dict(scene=s['name'],video=key,frames=count,width=W,height=H,fps=10))
 for name in ['mask_stats.json','evidence_stats.json','temporal_pair_audit.json','quicklook.jpg','quicklook_zoom.jpg','mask_review.jpg']:
  shutil.copy2(out/name,dest/name)
 for name in ['fence_refinement.json']:
  if (out/name).exists():shutil.copy2(out/name,dest/name)
 print('PACKAGED',s['name'],flush=True)
for name in ['registration.json','execution_notes.json','generation_state.json','generation_input_preflight.json','evidence_summary.json','guard_summary.json','validation.json','background_admission.json']:
 shutil.copy2(ROOT/name,PACKAGE/name)
dump(PACKAGE/'video_validation.json',dict(videos=rows,total_videos=len(rows),decoded_frames=sum(r['frames'] for r in rows),human_verdict=None))

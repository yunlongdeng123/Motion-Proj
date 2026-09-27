"""r10–r13诊断审核包：新标注视频，旧生成视频注明来源；没有新生成结果。"""
import sys,shutil,subprocess,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera
from PIL import ImageDraw
import imageio_ffmpeg
BASE=ROOT.parent;R2=ROOT;OUT=BASE/'identity_review';OUT.mkdir(exist_ok=False)
cv2.setNumThreads(4)

def zbox(c,dc,b):
 p=np.array(b['pose']);r=p[:3,:3];o=(c[:3,3]-p[:3,3])@r;v=dc@c[:3,:3].T@r;half=np.array(b['size_lwh'])/2;parallel=np.abs(v)<1e-10;safe=np.where(parallel,1.,v)
 t0=(-half-o)/safe;t1=(half-o)/safe;lo=np.where(parallel,-np.inf,np.minimum(t0,t1)).max(-1);hi=np.where(parallel,np.inf,np.maximum(t0,t1)).min(-1)
 hit=(hi>=np.maximum(lo,.1))&~np.any(parallel&(np.abs(o)>half),axis=-1)
 return np.where(hit,np.maximum(lo,.1),np.inf)

def encode(folder,path):
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
 v=cv2.VideoCapture(str(path));n=0
 while True:
  ok,im=v.read()
  if not ok:break
  assert im.shape[:2]==HW;n+=1
 fps=v.get(cv2.CAP_PROP_FPS);v.release();assert n==30 and abs(fps-10)<.01
 Image.open(folder/'00000.png').save(path.with_name(path.stem+'_poster.jpg'),quality=94)
 return dict(path=str(path.relative_to(OUT)),decoded_frames=n,fps=fps,height=H,width=W)

videos=[];summaries=[]
for s in read(R2/'registration.json')['scenes']:
 name=s['name'];out=OUT/name;out.mkdir();frames=read(OLD/name/'frames.json');rows=[]
 keys=['source','support','r3','r6']
 for key in keys:(out/'frames'/key).mkdir(parents=True)
 for i,f in enumerate(s['source_frames']):
  fr=frames[str(f)];c,k=camera(fr,s['camera'],HW);yy,xx=np.mgrid[:H,:W];dc=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T
  core=mask(OLD/name/'sam'/f'core_{i:05}.png');orig=rgb(R2/name/'rgb'/f'{i:05}.png');target=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);zt=zbox(c,dc,target)
  best=np.full(HW,np.inf);ids=np.full(HW,-1,np.int32);boxes={}
  for b in fr['all_boxes']:
   if b['actor_id']==s['actor'] or not b['category'].startswith('vehicle.'):continue
   rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW)
   if rect is None:continue
   z=zbox(c,dc,b);hit=z<best;best[hit]=z[hit];ids[hit]=int(b['actor_id']);boxes[int(b['actor_id'])]=(b,rect)
  supported=core&np.isfinite(best)&(best>zt+.1);unknown=core&~supported
  area={int(a):int(((ids==a)&supported).sum()) for a in np.unique(ids[supported])};top=sorted(area,key=lambda a:-area[a])[:3]
  overlay=orig.copy();overlay[unknown]=(.2*orig[unknown]+.8*np.array([245,133,30])).astype('uint8');overlay[supported]=(.2*orig[supported]+.8*np.array([65,230,140])).astype('uint8')
  images={'source':orig,'support':overlay,'r3':rgb(BASE/'r3'/name/'final'/f'{i:05}.png'),'r6':rgb(BASE/'r6'/name/'final'/f'{i:05}.png')}
  for key,a in images.items():
   im=Image.fromarray(a);d=ImageDraw.Draw(im)
   if key in ['source','support']:
    rect=project_bbox(target['pose'],target['size_lwh'],c,k,HW);d.rectangle(rect,outline='yellow',width=2);d.text((rect[0],max(20,rect[1]-12)),f'DELETE actor{s["actor"]}',fill='yellow')
   for aid in top:
    b,rect=boxes[aid];d.rectangle(rect,outline='cyan',width=1);d.text((rect[0],rect[3]+3),f'keep {aid}',fill='cyan')
   d.rectangle((0,0,W,24),fill=(10,20,30));d.text((8,7),f'{name} f{f} | '+{'source':'ORIGINAL: target yellow / possible hidden neighbor cyan','support':'GT BOX RAYS: green neighbor support / orange unknown, not mesh truth','r3':'EXISTING r3 generated candidate; identity not verified','r6':'EXISTING r6 broad-context control; neighbor may be missing'}[key],fill='white')
   im.save(out/'frames'/key/f'{i:05}.png')
  rows.append(dict(frame=f,core_pixels=int(core.sum()),behind_vehicle_box_pixels=int(supported.sum()),support_fraction=float(supported.sum()/core.sum()),actors=area))
 for key in keys:videos.append(encode(out/'frames'/key,out/f'{key}.mp4'))
 dump(out/'ray_support.json',rows);summaries.append(dict(scene=name,frames=30,support_fraction_min=min(r['support_fraction'] for r in rows),support_fraction_max=max(r['support_fraction'] for r in rows)))
 for key in ['identity_audit.jpg','source_instances.jpg','occluded_actor_audit.jpg','factual_context.jpg']:
  shutil.copy2(BASE/'r10'/name/key,out/key)
refs=OUT/'scene_0255/actor_references';shutil.copytree(BASE/'r10/scene_0255/actor_references',refs)
for name in ['matte_probe.jpg','centerlines.jpg','matte_first.json','trimap_first.png','alpha_first.png']:
 shutil.copy2(BASE/'r11'/name,OUT/name)
for run in ['r12','r13']:
 dest=OUT/run;dest.mkdir();shutil.copy2(BASE/run/'summary.json',dest/'summary.json');shutil.copy2(BASE/run/'scene_0255/neighbor_evidence_probe.jpg',dest/'neighbor_evidence_probe.jpg')
for name in ['depth_attrition.jpg','depth_attrition.json']:shutil.copy2(BASE/'r12/scene_0255'/name,OUT/name)
shutil.copy2(BASE/'r13/scene_0255/calibration.json',OUT/'calibration.json')
dump(OUT/'video_validation.json',dict(videos=videos,decoded_frames=sum(v['decoded_frames'] for v in videos),scope='new annotations on preserved original/r3/r6 PNG; zero new generative inference'))
dump(OUT/'summary.json',dict(task_id='WS-V77-HYBRID-BG-20260927',runs=['r10','r11','r12','r13'],scenes=summaries,
 finding='actor-specific DELETE must preserve hidden non-target actors; vehicle detection alone cannot establish regeneration',
 local_scale_probe='5 source views admitted by heldout sparse depth; strict core evidence 47/26/11/1 pixels, insufficient for completed background',
 new_video_generation=0,new_annotation_videos=len(videos),human_verdict=None,background_input_dir=None,failure_ledger_refs=['V77-F02'],failure_ledger_delta='updated V77-F02'))
print('IDENTITY REVIEW PACKAGE READY',len(videos),flush=True)

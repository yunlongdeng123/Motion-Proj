"""r10：从原日志找actor25/52参考，几何可见候选仍须图像核对。"""
import sys,math
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera
from video_review import scene_frame
from PIL import ImageDraw,ImageOps
BASE=ROOT.parent;ROOT=BASE/'r10';NAME='scene_0255';spec=next(s for s in read(BASE/'r2/registration.json')['scenes'] if s['name']==NAME);data=Path(spec['data']);inst=read(data/'instances/instances_info.json');out=ROOT/NAME/'actor_references';out.mkdir(exist_ok=False)
def zbox(c,dc,b):
 p=np.array(b['pose']);r=p[:3,:3];o=(c[:3,3]-p[:3,3])@r;d=dc@c[:3,:3].T@r;half=np.array(b['size_lwh'])/2;parallel=np.abs(d)<1e-10;safe=np.where(parallel,1.,d);a=(-half-o)/safe;z=(half-o)/safe;near=np.where(parallel,-np.inf,np.minimum(a,z)).max(-1);far=np.where(parallel,np.inf,np.maximum(a,z)).min(-1);valid=(far>=np.maximum(near,.1))&~np.any(parallel&(np.abs(o)>half),axis=-1);return np.where(valid,np.maximum(near,.1),np.inf)
allrecords={}
for aid in ['25','52']:
 records=[];fa=inst[aid]['frame_annotations'];ids=fa['frame_idx'];sample=sorted(set(ids[::5]+[ids[-1]]))
 for f in sample:
  fr=scene_frame(spec['spec'],f,inst);target=next(b for b in fr['all_boxes'] if b['actor_id']==aid)
  for cam in range(6):
   c,k=camera(fr,cam,HW);rect=project_bbox(target['pose'],target['size_lwh'],c,k,HW)
   if rect is None:continue
   x1,y1,x2,y2=np.floor(rect).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(W-1,x2);y2=min(H-1,y2)
   if (x2-x1)*(y2-y1)<400:continue
   yy,xx=np.mgrid[y1:y2+1,x1:x2+1];dc=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T;zt=zbox(c,dc,target);other=np.full_like(zt,np.inf)
   for b in fr['all_boxes']:
    if b['actor_id']==aid:continue
    br=project_bbox(b['pose'],b['size_lwh'],c,k,HW)
    if br is None or br[2]<x1 or br[0]>x2 or br[3]<y1 or br[1]>y2:continue
    other=np.minimum(other,zbox(c,dc,b))
   area=np.isfinite(zt);visible=area&(zt<=other+.1);fraction=float(visible.sum()/max(1,area.sum()));p=np.array(target['pose']);local=(c[:3,3]-p[:3,3])@p[:3,:3];azimuth=math.degrees(math.atan2(local[1],local[0]));score=float(visible.sum()*fraction)
   records.append(dict(actor=aid,frame=f,camera=cam,box=rect,visible_box_fraction=fraction,projected_pixels=int(area.sum()),visible_box_pixels=int(visible.sum()),azimuth=azimuth,score=score))
 selected=[]
 for r in sorted(records,key=lambda r:-r['score']):
  if any(abs((r['azimuth']-p['azimuth']+180)%360-180)<12 for p in selected):continue
  selected.append(r)
  if len(selected)>=6:break
 # 最小角变化时仍保存最高可见参考，而不是编造多视图。
 for j,r in enumerate(selected):
  im=Image.fromarray(rgb(data/'images'/f'{r["frame"]:03}_{r["camera"]}.jpg'));box=r['box'];pad=12;box=[max(0,int(box[0])-pad),max(0,int(box[1])-pad),min(W,int(box[2])+pad),min(H,int(box[3])+pad)];crop=im.crop(box);crop.save(out/f'actor{aid}_{j}_crop.png');show=ImageOps.pad(crop,(640,360),color=(15,20,25));d=ImageDraw.Draw(show);d.text((8,8),f'actor{aid} f{r["frame"]} CAM{r["camera"]} | box visible {r["visible_box_fraction"]:.0%} | az {r["azimuth"]:.1f}',fill='yellow');show.save(out/f'actor{aid}_{j}.jpg',quality=96)
 allrecords[aid]=dict(all=records,selected=selected)
sheet=Image.new('RGB',(1280,360*6),(15,20,25))
for col,aid in enumerate(['25','52']):
 for j,_ in enumerate(allrecords[aid]['selected']):sheet.paste(Image.open(out/f'actor{aid}_{j}.jpg'),(col*640,j*360))
sheet.save(out/'identity_reference_atlas.jpg',quality=96);dump(out/'references.json',dict(actors=allrecords,scope='GT box visibility proposal; static structures not fully occlusion modeled; original RGB only',human_verdict=None));print('NEIGHBOR REFERENCE ATLAS COMPLETE', {k:len(v['selected']) for k,v in allrecords.items()})

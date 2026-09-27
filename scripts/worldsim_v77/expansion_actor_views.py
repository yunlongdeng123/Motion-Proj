from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from video_review import scene_frame
from geometry import project_bbox
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r5';spec=read(T/'r3/frames.json')[0];inst=read(Path(spec['root'])/'instances/instances_info.json');cand=[]
for f in inst['12']['frame_annotations']['frame_idx']:
    fr=scene_frame(spec,f,inst);b=next(b for b in fr['all_boxes'] if b['actor_id']=='12')
    for cid in range(6):
        c,k=camera(fr,cid);bb=project_bbox(b['pose'],b['size_lwh'],c,k,(576,1024))
        if bb and bb[0]>6 and bb[1]>6 and bb[2]<1018 and bb[3]<570:
            area=(bb[2]-bb[0])*(bb[3]-bb[1]);cand.append(dict(frame=f,camera=cid,box=bb,area=area,image=fr['views'][cid]['image']))
selected=[]
for r in sorted(cand,key=lambda r:-r['area']):
    if any(x['camera']==r['camera'] and abs(x['frame']-r['frame'])<10 for x in selected):continue
    selected.append(r)
    if len(selected)==8:break
sheet=Image.new('RGB',(1024,314*4),(25,30,40));d=ImageDraw.Draw(sheet)
for i,r in enumerate(selected):
    im=Image.open(r['image']).convert('RGB').resize((1024,576));bb=r['box'];box=[max(0,int(bb[0])-20),max(0,int(bb[1])-20),min(1024,int(bb[2])+20),min(576,int(bb[3])+20)];im=im.crop(box);im.thumbnail((512,288));x=i%2*512;y=i//2*314;sheet.paste(im,(x,y+26));d.text((x+4,y+5),f"{i}: target12 f{r['frame']} CAM{r['camera']} area{r['area']:.0f}",fill='white')
sheet.save(R/'view_candidates.jpg',quality=95);dump(R/'view_candidates.json',selected)

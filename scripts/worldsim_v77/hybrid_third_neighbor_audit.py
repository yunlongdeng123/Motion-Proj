import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hybrid_road_plane_control as p
from repair_common import hull_mask,dump
from geometry import project_bbox,transform
import numpy as np
from PIL import Image,ImageDraw
ROOT=p.BASE/'r28';rows=[]
for f in [0,5,9]:
 fr,c,k,im,ex=p.get(f);wr=np.array(Image.open(ROOT/'write_mask'/f'{f:05}.png'))>0;model=np.array(Image.open(ROOT/'model_mask'/f'{f:05}.png'))>0;viz=Image.fromarray(im);dr=ImageDraw.Draw(viz);records=[]
 for b in fr['all_boxes']:
  m=hull_mask(b,c,k,p.HW);overlap=int((m&model).sum())
  if overlap==0:continue
  box=project_bbox(b['pose'],b['size_lwh'],c,k,p.HW);depth=transform(np.array(b['pose'])[:3,3][None],np.linalg.inv(c))[0,2]
  records.append(dict(actor=b['actor_id'],category=b.get('category'),depth=float(depth),model_overlap=overlap,write_overlap=int((m&wr).sum()),bbox=box.tolist() if hasattr(box,'tolist') else box))
  if box is not None:
   color='yellow' if b['actor_id']=='12' else 'cyan';dr.rectangle(tuple(box),outline=color,width=2);dr.text((box[0],box[1]-12),str(b['actor_id']),fill=color)
 viz.save(ROOT/f'neighbors_f{f:03}.jpg',quality=96);rows.append(dict(frame=f,actors=records))
dump(ROOT/'neighbor_audit.json',rows);print(rows)

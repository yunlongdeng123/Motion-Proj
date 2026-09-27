from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,camera
from geometry import transform
R=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r3');reg=read(R/'proposal_registration.json');frames=read(R/'frames.json')
rows=[]
for pid,cams in [(6,[2,0]),(2,[5,4]),(7,[2,5])]:
    row=next(r for r in reg['candidates'] if r['proposal']==pid)
    s=np.array(row['size_lwh']);p=np.array(row['pose_world']);corn=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1]])*s/2;wp=transform(corn,p)
    for f in [0,9]:
        panel=Image.new('RGB',(1536,456),(20,30,40));d=ImageDraw.Draw(panel)
        for j,cid in enumerate(cams):
            fr=frames[f];c,k=camera(fr,cid);q=transform(wp,np.linalg.inv(c));uv=q@k.T;uv=uv[:,:2]/uv[:,2:];im=Image.open(fr['views'][cid]['image']).convert('RGB').resize((768,432));uv*=.75
            dd=ImageDraw.Draw(im);dd.polygon([tuple(v) for v in uv],outline='magenta',width=4);panel.paste(im,(j*768,24));d.text((j*768+4,4),f'proposal {pid} CAM{cid} f{f}: ground footprint',fill='white')
        rows.append(panel)
sheet=Image.new('RGB',(1536,len(rows)*456))
for i,p in enumerate(rows):sheet.paste(p,(0,i*456))
sheet.save(R/'footprints.jpg',quality=95)

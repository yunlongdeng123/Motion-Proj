"""真实隐藏邻车提示，禁止把任意车辆响应当再生；不影响生成规则。"""
from nine_common import *
from geometry import transform,project_bbox
from PIL import Image,ImageDraw
import numpy as np
reg=read(ROOT/'registration.json');cams={r['scene']:r['camera'] for r in read(ROOT/'display_registration.json')};results=[]
for s in reg['scenes']:
    c=cams[s['name']];base=ROOT/s['name'];stream=base/f'cam{c}';frames=read(base/'camera_frames.json');rows=[];sheet=Image.new('RGB',(1024,3*600),(20,30,40));draw=ImageDraw.Draw(sheet)
    for f,fr in enumerate(frames):
        cc,k=camera(fr,c);target=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);tz=transform(np.array(target['pose'])[None,:3,3],np.linalg.inv(cc))[0,2];mask=np.array(Image.open(stream/'write_mask'/f'{f:05}.png'))>0;neighbors=[]
        for b in fr['all_boxes']:
            if b['actor_id']==s['actor'] or not b.get('category','').startswith('vehicle.'):continue
            hull=hull_mask(b,cc,k,ground_extend=.4);hit=int((hull&mask).sum())
            if hit:
                z=transform(np.array(b['pose'])[None,:3,3],np.linalg.inv(cc))[0,2];neighbors.append(dict(actor=b['actor_id'],overlap_pixels=hit,write_fraction=hit/max(1,int(mask.sum())),center_behind=bool(z>tz),center_z=float(z),target_z=float(tz)))
        rows.append(dict(frame=f,write_pixels=int(mask.sum()),neighbors=neighbors))
        if f in [0,14,29]:
            im=Image.open(stream/'rgb'/f'{f:05}.png').convert('RGB');d=ImageDraw.Draw(im)
            for b,color in [(target,'yellow')]+[(next(b for b in fr['all_boxes'] if b['actor_id']==n['actor']),'cyan') for n in neighbors]:
                bb=project_bbox(b['pose'],b['size_lwh'],cc,k,(576,1024))
                if bb:d.rectangle(bb,outline=color,width=2);d.text((bb[0],max(0,bb[1]-15)),b['actor_id'],fill=color)
            y=[0,14,29].index(f)*600;sheet.paste(im,(0,y+24));draw.text((5,y+4),f"{s['name']} CAM{c} f{f}: yellow target / cyan otherGT overlaps write",fill='white')
    sheet.save(base/'neighbor_source.jpg',quality=93);row=dict(scene=s['name'],camera=c,frames=30,frames_with_other_gt_vehicle_write_overlap=sum(bool(r['neighbors']) for r in rows),rows=rows);results.append(row)
    print('NEIGHBORS',s['name'],row['frames_with_other_gt_vehicle_write_overlap'],flush=True)
dump(ROOT/'neighbor_audit.json',dict(scenes=results,scope='GT vehicle volumes extended0.4m downward; center-depth relation is not exact surface visibility. Presence may be true retained hidden neighbor; absence limited by GT recall. Emptywrite is input failure or invisibility, not background pass.',human_verdict=None))

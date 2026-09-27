"""疑似车辆再生仍须核对GT隐藏邻车；不能把所有车辆响应都判为幻觉。"""
from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera,hull_mask
from geometry import project_bbox,transform
R=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1');reg=read(R/'registration.json');allrows=[];tiles=[]
for s in reg['new_scenes']:
    if s['name']=='processed_756':continue
    c=s['primary_camera'];base=R/s['name']/f'cam{c}';frames=read(R/s['name']/'frames.json');rows=[]
    for f,fr in enumerate(frames):
        cc,k=camera(fr,c);target=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);tz=transform(np.array(target['pose'])[None,:3,3],np.linalg.inv(cc))[0,2];m=np.array(Image.open(base/'write_mask'/f'{f:05}.png'))>0;neighbors=[]
        for b in fr['all_boxes']:
            if b['actor_id']==s['actor'] or not b.get('category','').startswith('vehicle.'):continue
            h=hull_mask(b,cc,k,ground_extend=.4);hit=int((h&m).sum())
            if hit:
                z=transform(np.array(b['pose'])[None,:3,3],np.linalg.inv(cc))[0,2];neighbors.append(dict(actor=b['actor_id'],category=b.get('category'),write_overlap_pixels=hit,fraction=hit/max(1,int(m.sum())),center_depth=float(z),target_center_depth=float(tz),center_behind=bool(z>tz),bbox=project_bbox(b['pose'],b['size_lwh'],cc,k,(576,1024))))
        rows.append(dict(frame=f,neighbors=neighbors))
        if f in [0,14,29] and s['name'] in ['processed_425','processed_382','processed_191']:
            imgs=[]
            for kind,p in [('source',base/'rgb'/f'{f:05}.png'),('delete',base/'drive/composite'/f'{f:05}.png')]:
                im=Image.open(p).convert('RGB');d=ImageDraw.Draw(im)
                for b,color in [(target,'yellow')]+[(next(b for b in fr['all_boxes'] if b['actor_id']==n['actor']),'cyan') for n in neighbors]:
                    bb=project_bbox(b['pose'],b['size_lwh'],cc,k,(576,1024))
                    if bb:d.rectangle(bb,outline=color,width=2);d.text((bb[0],max(0,bb[1]-15)),b['actor_id'],fill=color)
                tile=Image.new('RGB',(768,456),(20,30,40));tile.paste(im.resize((768,432)),(0,24));ImageDraw.Draw(tile).text((4,4),f'{s["name"]} f{f} {kind} yellow target / cyan potential overlap',fill='white');imgs.append(tile)
            panel=Image.new('RGB',(1536,456));panel.paste(imgs[0],(0,0));panel.paste(imgs[1],(768,0));tiles.append(panel)
    allrows.append(dict(scene=s['name'],frames_with_any_gt_vehicle_write_overlap=sum(bool(r['neighbors']) for r in rows),rows=rows))
dump(R/'neighbor_audit.json',dict(scenes=allrows,scope='GT projected volumes enlarged .4m downward; center-depth order only, not exact surface visibility or completeness. Presence means cannot automatically call vehicle generation hallucination; absence is limited by GT recall.',human_verdict=None))
# 单一联系表，保持实际构造顺序191、425、382。
sheet=Image.new('RGB',(1536,len(tiles)*456))
for j,p in enumerate(tiles):sheet.paste(p,(0,456*j))
sheet.save(R/'neighbor_audit.jpg',quality=94);print('NEIGHBOR_AUDIT',[(r['scene'],r['frames_with_any_gt_vehicle_write_overlap'],[(v['frame'],[(n['actor'],round(n['fraction'],2),n['center_behind']) for n in v['neighbors']]) for v in r['rows'] if v['frame'] in [0,14,29]]) for r in allrows])

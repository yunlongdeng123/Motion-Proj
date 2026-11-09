from pathlib import Path
import json,importlib.util
import numpy as np
from PIL import Image,ImageDraw,ImageFont
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'waymo_val065_delete30k_20261109_r1';O=C/'waymo_val065_delete30k_diagnosis_20261109_r1'
load=lambda p:json.loads(p.read_text());B=Path(load(R/'world_road/receipt.json')['source_bank']);d=load(O/'cross_time.json')
spec=importlib.util.spec_from_file_location('road',C/'waymo_allframes_20261109_r1/public_api/drivingeditor_dynamic_edit/v44_road.py');road=importlib.util.module_from_spec(spec);spec.loader.exec_module(road)
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for rec in d['records']:
    e=rec['example'];query=rec['raw'];world=np.array(e['world_xyz'])[None];raws=[query,*e['reference_raws'][:3]]
    board=Image.new('RGB',(1920,594),'#0b111b');draw=ImageDraw.Draw(board)
    for i,raw in enumerate(raws):
        obs=load(B/'observations'/f'{raw:03d}.json');im=Image.open(B/obs['image_path']).convert('RGB').resize((480,320),Image.Resampling.BILINEAR)
        uv=e['pixel_xy'] if i==0 else road.project(world,obs)[0][0]/4
        px,py=map(float,uv);draw.text((i*480+8,8),('Query' if i==0 else 'Source candidate')+f' FRONT raw{raw}',font=f,fill='white')
        zoom=im.crop((round(px)-30,round(py)-25,round(px)+30,round(py)+25)).resize((240,200),Image.Resampling.NEAREST)
        dm=ImageDraw.Draw(im);dm.ellipse((px-5,py-5,px+5,py+5),outline='#ffda67',width=2);dm.line((px-10,py,px+10,py),fill='#ffda67');dm.line((px,py-10,px,py+10),fill='#ffda67')
        board.paste(im,(i*480,34));board.paste(zoom,(i*480+120,380));dz=ImageDraw.Draw(board);dz.line((i*480+230,480,i*480+250,480),fill='#ffda67',width=2);dz.line((i*480+240,470,i*480+240,490),fill='#ffda67',width=2)
    board.save(O/f'source_candidates{query}.png')

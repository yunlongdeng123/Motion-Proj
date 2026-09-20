"""在真实帧上显示点级支持，缺失比较不补零。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from audit_natural_observation import OUT,RUNS,ARMS,FRAMES

ref=json.loads((RUNS[42]/'evaluator_reference.json').read_text())
result=json.loads((OUT/'result.json').read_text())
labels=['Real RGB','GT condition','DVGT + known rays','Ordinary box fit','Extra target LiDAR']
for record in result['rows']:
    seed=record['seed']
    track=[np.load(OUT/'real/tracks.npz')]+[np.load(OUT/f'seed{seed}-{arm}'/'tracks.npz') for arm in ARMS]
    common=np.logical_and.reduce([r['valid'] for r in track])
    arrays={name:np.load(RUNS[seed]/name/'clean.npy',mmap_mode='r') for name in ARMS}
    sheet=Image.new('RGB',(1500,4*215+35),'#101b2b');draw=ImageDraw.Draw(sheet)
    for col,label in enumerate(labels):draw.text((col*300+5,7),label,fill='white')
    for i,f in enumerate(FRAMES[1:],1):
        cx,cy=ref['frames'][i]['match']['center'];crop=(round(cx-180),round(cy-108),round(cx+180),round(cy+108))
        row=record['frames'][i];y=35+(i-1)*215
        text=f'Seed {seed} | {f/30:.1f}s | common {int(common[i].sum())}/62 | '+('measurable' if row['admitted'] else 'insufficient common support')
        draw.text((6,y+3),text,fill='white')
        images=[Image.open(RUNS[42]/f'reference-{f:03d}.png')]+[Image.fromarray(arrays[name][f]) for name in ARMS]
        for col,im in enumerate(images):
            im=im.copy();d=ImageDraw.Draw(im)
            for j,(x,yj) in enumerate(track[col]['points'][i]):
                if not track[col]['valid'][i,j]:continue
                color='#00e5b5' if common[i,j] else '#ffe081'
                d.ellipse((x-2,yj-2,x+2,yj+2),fill=color)
            sheet.paste(im.crop(crop).resize((300,180)),(col*300,y+24))
    sheet.save(OUT/f'tracking-seed{seed}.jpg',quality=95)
print('Actual tracking overlays written; cyan=common, yellow=only individually retained.')

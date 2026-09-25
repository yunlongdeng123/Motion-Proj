import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
root=Path('/root/autodl-tmp/data/v76_vadgs')
audit=root/'sam_occlusion_audit'
selected=[('scene_0230','020_3',6),('scene_0230','020_3',21),('scene_0255','020_2',14),('scene_0255','020_2',23)]
canvas=Image.new('RGB',(1000,4*325),'white')
draw=ImageDraw.Draw(canvas)
for i,(scene,name,key) in enumerate(selected):
    view=next(x for x in json.loads((root/scene/'sam_prior_evidence/generation_report.json').read_text())['views'] if x['name']==name)
    obj=next(x for x in view['projected_objects'] if x['track_id']==key)
    box=np.array(obj['box_xyxy']).astype(int)
    x0,y0,x1,y1=box
    padding=65
    rgb=np.asarray(Image.open(root/scene/'images'/f'{name}.jpg').convert('RGB'))
    raw=np.asarray(Image.open(audit/f'{scene}_{name}_id{key}_raw.png'))>0
    overlay=rgb.copy()
    overlay[raw]=(.5*overlay[raw]+.5*np.array([255,0,0])).astype(np.uint8)
    cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,255,255),2)
    extent=(max(0,x0-padding),max(0,y0-padding),min(1600,x1+padding),min(900,y1+padding))
    for j,array in enumerate([rgb,overlay]):
        tile=Image.fromarray(array).crop(extent)
        tile.thumbnail((490,295))
        canvas.paste(tile,(j*500, i*325+28))
    draw.text((8,i*325+7),f'{scene} {name} pedestrian ID {key}; left RGB / right raw SAM and target box',fill='black')
canvas.save(audit/'identity_gate_crops.png')

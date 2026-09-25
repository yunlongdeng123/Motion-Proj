import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
root=Path('/root/autodl-tmp/data/v76_vadgs')
gate=root/'visible_detection_gate'
protocol=json.loads((gate/'protocol.json').read_text())
controls=protocol['positive_controls']+protocol['negative_controls']
views=json.loads((gate/'result.json').read_text())['views']
canvas=Image.new('RGB',(900,250*len(controls)),'white')
draw=ImageDraw.Draw(canvas)
for i,(scene,name,key) in enumerate(controls):
    v=next(v for v in views if v['scene']==scene and v['view']==name)
    a=next(a for a in v['associations'] if a['track_id']==key)
    rgb=np.asarray(Image.open(root/scene/'images'/f'{name}.jpg').convert('RGB'))
    overlay=rgb.copy()
    path=gate/'sam_r1'/scene/f'{name}_id{key}_raw.png'
    if path.is_file():
        mask=np.asarray(Image.open(path))>0
        overlay[mask]=(.5*overlay[mask]+.5*np.array([0,160,255])).astype(np.uint8)
    box=np.array(a['projected_box']).astype(int)
    x0,y0,x1,y1=box
    cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,255,0) if a['match'] else (255,0,0),2)
    margin=40
    extent=(max(0,x0-margin),max(0,y0-margin),min(1600,x1+margin),min(900,y1+margin))
    for j,array in enumerate((rgb,overlay)):
        tile=Image.fromarray(array).crop(extent)
        ratio=min(440/tile.width,215/tile.height)
        tile=tile.resize((int(tile.width*ratio),int(tile.height*ratio)))
        canvas.paste(tile,(j*450,i*250+30))
    draw.text((8,i*250+9),f'{scene} {name} ID {key}; '+('visible control' if i<6 else 'occlusion control')+f'; match={a["match"] is not None}',fill='black')
canvas.save(gate/'sam_r1/fixed_controls.png')

"""保存r30输入与几何输出的实际时序，不用二维补景代替几何结果。"""
from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from delete_full_query import Writer
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r2');OUT=ROOT/'review';assert not OUT.exists();OUT.mkdir();frames=read(ROOT/'frames.json');writers={k:Writer(OUT/f'{k}.mp4',(688,384)) for k in ['original','input_r30','geometry','hybrid']};sheet=Image.new('RGB',(1600,10*250),(20,30,40));draw=ImageDraw.Draw(sheet)
for f,fr in enumerate(frames):
    images={'original':Image.open(fr['views'][0]['image']).convert('RGB').resize((688,384)),'input_r30':Image.open(ROOT/f'f{f:03}/input_cam0.png').convert('RGB').resize((688,384)),'geometry':Image.open(ROOT/f'f{f:03}/geometry_cam0.png').convert('RGB'),'hybrid':Image.open(ROOT/f'f{f:03}/hybrid_cam0.png').convert('RGB')}
    for j,(name,im) in enumerate(images.items()):
        writers[name].write(np.array(im));sheet.paste(im.crop((340,145,560,269)).resize((400,225)),(j*400,f*250+25));draw.text((j*400+4,f*250+5),f'f{f} {name}',fill='white')
for w in writers.values():w.close()
sheet.save(OUT/'all10_crops.jpg',quality=95)
Image.open(ROOT/'f000/geometry_cam0.png').save(OUT/'geometry_poster.jpg');Image.open(ROOT/'f000/input_cam0.png').save(OUT/'input_poster.jpg')
dump(OUT/'provenance.json',dict(frames=10,fps=10,camera=0,scope='original / saved r30 input / pure Omega point render / same render with inputRGB fallback only at no-hit pixels. No actor put-back yet.',human_verdict=None))
print(OUT)

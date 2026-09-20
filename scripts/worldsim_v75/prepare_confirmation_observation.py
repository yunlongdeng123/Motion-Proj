"""只提取冻结来源的真实评价时刻，不读取重建或生成结果。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from prepare_argoverse import ROOT,CAMERA,crop_image

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-CONFIRM-SOURCES-01/20260920-r1')
screen=json.loads((OUT/'screen/screen_result.json').read_text())
rows=[r for r in screen['logs'] if r['selected_target'] is not None]
sheet=Image.new('RGB',(1600,len(rows)*244+28),'#101b2b');draw=ImageDraw.Draw(sheet)
for j,t in enumerate([0,.5,1,1.5,2]):draw.text((j*320+8,8),f'Real RGB | {t:.1f}s',fill='white')
for i,row in enumerate(rows):
    log=row['log_id'];path=OUT/'real'/log;assert not path.exists();path.mkdir(parents=True)
    raw=ROOT/log;files=sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'));times=np.array([int(f.stem) for f in files])
    intr=pd.read_feather(raw/'calibration/intrinsics.feather').set_index('sensor_name').loc[CAMERA]
    top=(intr.height_px-intr.width_px*704/1280)/2;crop=[0,top,float(intr.width_px),top+intr.width_px*704/1280]
    chosen=next(c for c in row['geometric_candidates'] if c['track']==row['selected_target'])
    records=[];y=28+i*244
    draw.text((5,y+3),f'{log[:8]} | target {row["selected_target"][:8]} | no output-based selection',fill='white')
    for j,frame in enumerate([0,15,30,45,60]):
        target=row['start_ns']+int(round(frame/30*1e9));idx=np.argmin(abs(times-target));assert abs(times[idx]-target)<=1000
        rgb=crop_image(files[idx],crop);rgb.save(path/f'reference-{frame:03d}.png')
        b=chosen['all_boxes'][j];center=(np.array(b[:2])+b[2:])/2
        window=[round(center[0]-180),round(center[1]-108),round(center[0]+180),round(center[1]+108)]
        im=rgb.copy();ImageDraw.Draw(im).rectangle(b,outline='yellow',width=2)
        sheet.paste(im.crop(window).resize((320,192)),(j*320,y+24))
        records.append({'frame':frame,'timestamp_ns':target,'source_image':str(files[idx]),'reference_bounds':b})
    (path/'reference.json').write_text(json.dumps({'log_id':log,'target':row['selected_target'],'frames':records,'human_verdict':None},indent=2)+'\n')
sheet.save(OUT/'real-targets.jpg',quality=95)
print(json.dumps({'status':'complete','real_logs':len(rows),'real_frames':len(rows)*5,'model_calls':0}))

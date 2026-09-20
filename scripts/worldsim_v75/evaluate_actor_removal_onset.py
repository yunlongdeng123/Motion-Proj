"""评价 frame0 删除机制诊断并与冻结的 frame5 删除配对。"""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from evaluate_actor_removal_generation import assign, vehicle_detections
from evaluate_localization import model, predict


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r3'
SOURCE = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01/20260921-r1'
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-ONSET-01/20260921-r1'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert json.loads((OUT/'result.json').read_text())['status'] == 'complete'
    assert not (OUT/'evaluation.json').exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    q = json.loads((QUAL/'result.json').read_text()); selected = q['selected']
    old = json.loads((SOURCE/'evaluation.json').read_text())
    base = Path(selected['base']); tr = np.load(base/'trajectory.npz')
    reference = json.loads(Path(selected['state_inputs']['reference']['source']).read_text())
    data = np.load(OUT/'generated.npy', mmap_mode='r'); detector = model(); began = time.monotonic(); rows=[]
    for frame in protocol['evaluation_frames']:
        detections = vehicle_detections(predict(detector, data[frame]))
        assignment = assign(reference, frame, tr['camera_world'], tr['K'], selected['actor'], selected['behind'], detections)
        rows.append({'frame':frame,'actor_present':assignment['actor'] is not None,
                     'behind_present':assignment['behind'] is not None,'assignment':assignment})
    post = [x for x in rows if x['frame'] >= 5]
    actor_count = sum(x['actor_present'] for x in post); behind_count = sum(x['behind_present'] for x in post)
    old_ref = next(x for x in old['summaries'] if x['arm']=='reference')['removed']
    if actor_count >= 6:
        interpretation = 'initial_image_anchor_supported'
    elif actor_count <= 2 and old_ref['actor_present'] >= 6:
        interpretation = 'autoregressive_history_lock_supported'
    else:
        interpretation = 'mixed_or_inconclusive'
    review_frames=[4,5,29,61,116]; variants=[
        ('reference / unedited', np.load(SOURCE/'reference-unedited/generated.npy',mmap_mode='r')),
        ('reference / removed at f=5', np.load(SOURCE/'reference-removed/generated.npy',mmap_mode='r')),
        ('reference / removed at f=0', data)]
    sheet=Image.new('RGB',(5*512,3*310),'#13202e');draw=ImageDraw.Draw(sheet)
    old_rows=old['rows']
    for ri,(name,array) in enumerate(variants):
        for ci,frame in enumerate(review_frames):
            image=Image.fromarray(np.asarray(array[frame])).copy();d=ImageDraw.Draw(image)
            if ri<2:
                variant='unedited' if ri==0 else 'removed'
                row=next(x for x in old_rows if x['arm']=='reference' and x['variant']==variant and x['frame']==frame)
            else: row=next(x for x in rows if x['frame']==frame)
            for key,color,label in [('projected_actor','#ff5148','A'),('projected_behind','#00e6cf','B')]:
                item=row['assignment'][key]
                if item: d.rectangle(item['bounds'],outline=color,width=3);d.text((item['bounds'][0],item['bounds'][1]-13),label,fill=color)
            for match in row['assignment']['matches']:
                if match['id'] in [selected['actor'],selected['behind']]:d.rectangle(match['detection']['box'],outline='#ffe86b',width=2)
            d.text((8,8),f'f={frame} A={int(row["actor_present"])} B={int(row["behind_present"])}',fill='white',stroke_width=2,stroke_fill='black')
            sheet.paste(image.resize((512,282)),(ci*512,ri*310+24))
        draw.text((5,ri*310+4),name,fill='white')
    sheet.save(OUT/'evaluation-review.jpg',quality=94)
    result={'status':'complete','rows':rows,'post_frame5_actor_present':actor_count,
            'post_frame5_behind_present':behind_count,'existing_frame5_removal':old_ref,
            'frozen_interpretation':interpretation,'new_detector_calls':len(rows),
            'new_world_model_calls':0,'wall_s':time.monotonic()-began,
            'cuda_initialized':torch.cuda.is_initialized(),'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

"""冻结门槛评价中远距离新来源的 reference editability。"""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from evaluate_actor_removal_generation import assign,vehicle_detections
from evaluate_localization import model,predict


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL=ROOT/'WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01/20260921-r1'
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not (OUT/'evaluation.json').exists();p=json.loads((OUT/'protocol.json').read_text());q=json.loads((QUAL/'result.json').read_text());assert json.loads((OUT/'queue_result.json').read_text())['status']=='complete'
    sel=q['selected'];base=Path(sel['base']);tr=np.load(base/'trajectory.npz');scene=json.loads(Path(sel['state_inputs']['reference']['source']).read_text())
    arrays={v:np.load(OUT/f'reference-{v}/generated.npy',mmap_mode='r') for v in ['unedited','removed']}
    c0=np.load(OUT/'reference-unedited/conditions.npy',mmap_mode='r');c1=np.load(OUT/'reference-removed/conditions.npy',mmap_mode='r')
    assert np.array_equal(c0[:5],c1[:5]) and np.array_equal(arrays['unedited'][:5],arrays['removed'][:5]) and not np.array_equal(c0[5:],c1[5:])
    detector=model();began=time.monotonic();rows=[]
    for variant in ['unedited','removed']:
        for frame in p['evaluation_frozen']['frames']:
            detections=vehicle_detections(predict(detector,arrays[variant][frame]))
            a=assign(scene,frame,tr['camera_world'],tr['K'],sel['actor'],sel['behind'],detections)
            rows.append({'variant':variant,'frame':frame,'actor_present':a['actor'] is not None,'behind_present':a['behind'] is not None,'assignment':a})
    post=[f for f in p['evaluation_frozen']['frames'] if f>=5];summary={}
    for variant in ['unedited','removed']:
        subset=[x for x in rows if x['variant']==variant and x['frame'] in post]
        summary[variant]={'actor_present':sum(x['actor_present'] for x in subset),'behind_present':sum(x['behind_present'] for x in subset),
                          'other_matches_mean':float(np.mean([x['assignment']['other_matches'] for x in subset]))}
    gate=summary['unedited']['actor_present']>=6 and summary['removed']['actor_present']<=2 and summary['removed']['behind_present']>=4 and summary['removed']['behind_present']>summary['unedited']['behind_present']
    review=[4,5,29,61,116];sheet=Image.new('RGB',(5*512,2*310),'#13202e');draw=ImageDraw.Draw(sheet)
    for ri,variant in enumerate(['unedited','removed']):
        for ci,frame in enumerate(review):
            image=Image.fromarray(np.asarray(arrays[variant][frame])).copy();d=ImageDraw.Draw(image);row=next(x for x in rows if x['variant']==variant and x['frame']==frame)
            for key,color,label in [('projected_actor','#ff5148','A'),('projected_behind','#00e6cf','B')]:
                item=row['assignment'][key]
                if item:d.rectangle(item['bounds'],outline=color,width=3);d.text((item['bounds'][0],item['bounds'][1]-13),label,fill=color)
            for match in row['assignment']['matches']:
                if match['id'] in [sel['actor'],sel['behind']]:d.rectangle(match['detection']['box'],outline='#ffe86b',width=2)
            d.text((8,8),f'{variant} f={frame} A={int(row["actor_present"])} B={int(row["behind_present"])}',fill='white',stroke_width=2,stroke_fill='black')
            sheet.paste(image.resize((512,282)),(ci*512,ri*310+24))
        draw.text((5,ri*310+4),f'reference / {variant}',fill='white')
    sheet.save(OUT/'evaluation-review.jpg',quality=94)
    result={'status':'complete','rows':rows,'summary':summary,'reference_gate_passed':bool(gate),'dvgt_admitted':bool(gate),
            'new_detector_calls':len(rows),'new_world_model_calls':0,'wall_s':time.monotonic()-began,'cuda_initialized':torch.cuda.is_initialized(),
            'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'evaluation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

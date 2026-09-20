"""评价未来减速轨迹是否在 reference 生成中可控。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from evaluate_localization import model,predict
from prepare_argoverse import project_track
from qualify_actor_removal import match_vehicle


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75');QUAL=ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01/20260921-r1'
SOURCE=ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1';OUT=ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01/20260921-r1'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def center(box):return (np.asarray(box[:2])+box[2:])/2


def main():
    global QUAL,SOURCE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',type=Path,default=QUAL)
    parser.add_argument('--source-generation',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();QUAL,SOURCE,OUT=args.qualification,args.source_generation,args.output
    assert json.loads((OUT/'queue_result.json').read_text())['status']=='complete' and not (OUT/'evaluation.json').exists()
    p=json.loads((OUT/'protocol.json').read_text());q=json.loads((QUAL/'result.json').read_text());s=q['selected'];base=Path(s['base']);tr=np.load(base/'trajectory.npz')
    original=json.loads(Path(s['state_inputs']['reference']['source']).read_text());edited=json.loads(Path(s['state_inputs']['reference']['edited']).read_text())
    ot=next(t for t in original['tracks'] if t['id']==s['actor'] and 0 in t['frames']);et=next(t for t in edited['tracks'] if (t['id'],t['segment'])==(ot['id'],ot['segment']))
    arrays={'unedited':np.load(SOURCE/'reference-unedited/generated.npy',mmap_mode='r'),'edited':np.load(OUT/'reference-edited/generated.npy',mmap_mode='r')}
    assert np.array_equal(arrays['unedited'][:5],arrays['edited'][:5]);detector=model();began=time.monotonic();rows=[]
    for variant in ['unedited','edited']:
        for frame in p['evaluation_frozen']['frames']:
            op=project_track(ot,frame,tr['camera_world'],tr['K']);ep=project_track(et,frame,tr['camera_world'],tr['K']);pred=predict(detector,arrays[variant][frame])
            om=match_vehicle(pred,op['bounds']);em=match_vehicle(pred,ep['bounds']);chosen=em if variant=='edited' else om
            det_center=None if chosen is None else center(chosen['box'])
            rows.append({'variant':variant,'frame':frame,'original_projection':op,'edited_projection':ep,'original_match':om,'edited_match':em,
              'detection_center':None if det_center is None else det_center.tolist(),
              'center_distance_to_original_px':None if det_center is None else float(np.linalg.norm(det_center-center(op['bounds']))),
              'center_distance_to_edited_px':None if det_center is None else float(np.linalg.norm(det_center-center(ep['bounds'])))})
    late=p['evaluation_frozen']['late_separable_frames'];u=[x for x in rows if x['variant']=='unedited' and x['frame'] in late];e=[x for x in rows if x['variant']=='edited' and x['frame'] in late]
    unedited_matches=sum(x['original_match'] is not None and x['original_match']['iou']>=.3 for x in u)
    edited_matches=sum(x['edited_match'] is not None and x['edited_match']['iou']>=.3 for x in e)
    edited_preferred=sum(x['edited_match'] is not None and x['edited_match']['iou']>=.3 and x['center_distance_to_original_px']-x['center_distance_to_edited_px']>=5 for x in e)
    gate=unedited_matches>=4 and edited_matches>=4 and edited_preferred>=4
    review=[4,5,45,61,85,109,116];sheet=Image.new('RGB',(7*366,2*250),'#13202e');draw=ImageDraw.Draw(sheet)
    for ri,variant in enumerate(['unedited','edited']):
        for ci,frame in enumerate(review):
            image=Image.fromarray(np.asarray(arrays[variant][frame])).copy();d=ImageDraw.Draw(image);row=next(x for x in rows if x['variant']==variant and x['frame']==frame)
            d.rectangle(row['original_projection']['bounds'],outline='#ff5148',width=3);d.rectangle(row['edited_projection']['bounds'],outline='#00e6cf',width=3)
            match=row['edited_match'] if variant=='edited' else row['original_match']
            if match and match['iou']>=.3:d.rectangle(match['box'],outline='#ffe86b',width=2)
            d.text((8,8),f'{variant} f={frame}',fill='white',stroke_width=2,stroke_fill='black');sheet.paste(image.resize((366,206)),(ci*366,ri*250+28))
        draw.text((5,ri*250+4),f'reference / {variant} | red=original cyan=slowed yellow=detection',fill='white')
    sheet.save(OUT/'evaluation-review.jpg',quality=94)
    result={'status':'complete','rows':rows,'late_separable_frames':late,'unedited_original_matches':unedited_matches,'edited_projection_matches':edited_matches,
      'edited_projection_preferred_by_5px':edited_preferred,'reference_gate_passed':bool(gate),'dvgt_admitted':bool(gate),'new_detector_calls':len(rows),
      'new_world_model_calls':0,'wall_s':time.monotonic()-began,'cuda_initialized':torch.cuda.is_initialized(),'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'evaluation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

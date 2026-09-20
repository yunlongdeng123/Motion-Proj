"""按生成前冻结的门槛评价对象移除、后车显露及状态差异。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch

from evaluate_localization import model, predict, iou
from prepare_argoverse import project_track


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-GENERATION-01/20260921-r1'
QUAL=ROOT/'WS-V75-ACTOR-REMOVAL-QUALIFY-01/20260921-r2'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def vehicle_detections(prediction):
    return [{'box':b.tolist(),'score':float(s),'class_id':int(l)} for b,s,l in zip(prediction['boxes'],prediction['scores'],prediction['labels'])
            if int(l) in [3,6,8] and float(s)>=.5]


def assign(scene, frame, camera, K, actor_id, behind_id, detections):
    projected=[]
    for track in scene['tracks']:
        if track['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE','WHEELED_RIDER','BUS','BOX_TRUCK','TRUCK','TRUCK_CAB','VEHICULAR_TRAILER','SCHOOL_BUS','ARTICULATED_BUS']:
            continue
        p=project_track(track,frame,camera,K)
        if p and p['fully_inside'] and p['large_enough']:
            projected.append({'id':track['id'],'bounds':p['bounds'],'depth_m':p['depth_m']})
    matches=[]
    if projected and detections:
        matrix=np.array([[iou(row['bounds'],d['box']) for d in detections] for row in projected])
        ti,di=linear_sum_assignment(-matrix)
        for a,b in zip(ti,di):
            if matrix[a,b]>=.3:matches.append({**projected[a],'detection':detections[b],'iou':float(matrix[a,b])})
    by={m['id']:m for m in matches}
    return {'actor':by.get(actor_id),'behind':by.get(behind_id),'other_matches':sum(m['id'] not in [actor_id,behind_id] for m in matches),
            'projected_actor':next((x for x in projected if x['id']==actor_id),None),
            'projected_behind':next((x for x in projected if x['id']==behind_id),None),
            'detections':detections,'matches':matches}


def outside_mae(one,two,a,b,padding=16):
    mask=np.ones(one.shape[:2],bool)
    boxes=[x['bounds'] for x in [a,b] if x]
    if boxes:
        x1=max(0,int(np.floor(min(x[0] for x in boxes)-padding)));y1=max(0,int(np.floor(min(x[1] for x in boxes)-padding)))
        x2=min(one.shape[1],int(np.ceil(max(x[2] for x in boxes)+padding)));y2=min(one.shape[0],int(np.ceil(max(x[3] for x in boxes)+padding)))
        mask[y1:y2,x1:x2]=False
    return float(np.abs(one.astype(np.float32)-two.astype(np.float32))[mask].mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--qualification',type=Path,default=QUAL);a=p.parse_args();out=a.output
    assert not (out/'evaluation.json').exists();protocol=json.loads((out/'protocol.json').read_text());queue=json.loads((out/'queue_result.json').read_text())
    assert queue['status']=='complete' and queue['completed_runs']==6
    q=json.loads((a.qualification/'result.json').read_text());actor_id=q['selected']['actor'];behind_id=q['selected']['behind'];base=Path(q['selected']['base'])
    tr=np.load(base/'trajectory.npz');frames=protocol['evaluation_frozen']['frames'];detector=model();started=time.monotonic();rows=[]
    arrays={};scenes={}
    for arm in protocol['arms']:
        source=Path(q['selected']['state_inputs'][arm]['source']);scenes[arm]=json.loads(source.read_text())
        for variant in protocol['variants']:
            run=out/f'{arm}-{variant}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete'
            arrays[arm,variant]=np.load(run/'generated.npy',mmap_mode='r')
        c0=np.load(out/f'{arm}-unedited/conditions.npy',mmap_mode='r');c1=np.load(out/f'{arm}-removed/conditions.npy',mmap_mode='r')
        assert np.array_equal(c0[:5],c1[:5]) and not np.array_equal(c0[5:],c1[5:])
        assert np.array_equal(arrays[arm,'unedited'][:5],arrays[arm,'removed'][:5])
    for arm in protocol['arms']:
        scene=scenes[arm]
        for variant in protocol['variants']:
            data=arrays[arm,variant]
            for frame in frames:
                detections=vehicle_detections(predict(detector,data[frame]))
                assignment=assign(scene,frame,tr['camera_world'],tr['K'],actor_id,behind_id,detections)
                rows.append({'arm':arm,'variant':variant,'frame':frame,'actor_present':assignment['actor'] is not None,
                             'behind_present':assignment['behind'] is not None,'assignment':assignment})
    paired=[]
    for arm in protocol['arms']:
        scene=scenes[arm]
        for frame in frames:
            one=np.asarray(arrays[arm,'unedited'][frame]);two=np.asarray(arrays[arm,'removed'][frame])
            actor=project_track(next(t for t in scene['tracks'] if t['id']==actor_id and 0 in t['frames']),frame,tr['camera_world'],tr['K'])
            behind=project_track(next(t for t in scene['tracks'] if t['id']==behind_id and 0 in t['frames']),frame,tr['camera_world'],tr['K'])
            paired.append({'arm':arm,'frame':frame,'full_frame_mae':float(np.abs(one.astype(np.float32)-two.astype(np.float32)).mean()),
                           'outside_edit_mae':outside_mae(one,two,actor,behind),'exactly_equal':bool(np.array_equal(one,two))})
    post=[f for f in frames if f>=5]
    summaries=[]
    for arm in protocol['arms']:
        s={'arm':arm}
        for variant in protocol['variants']:
            subset=[x for x in rows if x['arm']==arm and x['variant']==variant and x['frame'] in post]
            s[variant]={'actor_present':sum(x['actor_present'] for x in subset),'behind_present':sum(x['behind_present'] for x in subset),
                        'other_matches_mean':float(np.mean([x['assignment']['other_matches'] for x in subset]))}
        summaries.append(s)
    ref=next(x for x in summaries if x['arm']=='reference')
    reference_gate=(ref['unedited']['actor_present']>=6 and ref['removed']['actor_present']<=2 and ref['removed']['behind_present']>=4 and
                    ref['removed']['behind_present']>ref['unedited']['behind_present'])
    decisions={(r['arm'],r['variant'],r['frame'],entity):r[entity+'_present'] for r in rows for entity in ['actor','behind']}
    mismatches={arm:sum(decisions[arm,'removed',f,e]!=decisions['reference','removed',f,e] for f in post for e in ['actor','behind'])
                for arm in ['dvgt_metric','class_prior']}
    corrected=sum(decisions['dvgt_metric','removed',f,e]!=decisions['reference','removed',f,e] and
                  decisions['class_prior','removed',f,e]==decisions['reference','removed',f,e] for f in post for e in ['actor','behind'])
    confirmation = protocol['task_id'] == 'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01'
    class_actor_excess = (next(x for x in summaries if x['arm']=='class_prior')['removed']['actor_present']-
                          ref['removed']['actor_present'])
    state_effect_gate = ((mismatches['class_prior']>=3 and class_actor_excess>=2) if confirmation else
                         (mismatches['dvgt_metric']>=3 and corrected>=1))
    # 审阅图只展示预先指定的五个时刻，不参与门槛。
    review_frames=[4,5,29,61,116];sheet=Image.new('RGB',(5*512,6*310),'#13202e');draw=ImageDraw.Draw(sheet)
    for row,(arm,variant) in enumerate((a,v) for a in protocol['arms'] for v in protocol['variants']):
        for col,frame in enumerate(review_frames):
            image=Image.fromarray(np.asarray(arrays[arm,variant][frame])).copy();d=ImageDraw.Draw(image)
            r=next(x for x in rows if x['arm']==arm and x['variant']==variant and x['frame']==frame)
            for key,color,label in [('projected_actor','#ff5148','A'),('projected_behind','#00e6cf','B')]:
                item=r['assignment'][key]
                if item:d.rectangle(item['bounds'],outline=color,width=3);d.text((item['bounds'][0],item['bounds'][1]-13),label,fill=color)
            for m in r['assignment']['matches']:
                if m['id'] in [actor_id,behind_id]:d.rectangle(m['detection']['box'],outline='#ffe86b',width=2)
            d.text((8,8),f'{arm} / {variant} | f={frame} | A={int(r["actor_present"])} B={int(r["behind_present"])}',fill='white',stroke_width=2,stroke_fill='black')
            sheet.paste(image.resize((512,282)),(col*512,row*310+24))
        draw.text((5,row*310+4),f'{arm} / {variant}',fill='white')
    sheet.save(out/'evaluation-review.jpg',quality=93)
    result={'status':'complete','rows':rows,'paired_pixel_metrics':paired,'summaries':summaries,'reference_gate_passed':reference_gate,
            'state_mismatches_vs_reference_removed':mismatches,'class_prior_corrected_dvgt_decisions':corrected,'state_effect_gate_passed':state_effect_gate,
            'gate_kind':'independent_class_prior_editability_confirmation' if confirmation else 'development_dvgt_recovery_discovery',
            'class_prior_removed_actor_excess_vs_reference':class_actor_excess,
            'confirmation_gate_passed':bool(reference_gate and state_effect_gate) if confirmation else None,
            'feedback_admitted':False if confirmation else bool(reference_gate and state_effect_gate),
            'closed_loop_not_run_by_protocol':confirmation,
            'new_detector_calls':len(rows),'new_world_model_calls':0,
            'wall_s':time.monotonic()-started,'cuda_initialized':torch.cuda.is_initialized(),'human_verdict':None,'failure_ledger_delta':'none'}
    save(out/'evaluation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','paired_pixel_metrics']},ensure_ascii=False))


if __name__=='__main__':main()

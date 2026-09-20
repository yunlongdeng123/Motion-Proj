"""冻结并执行一个对象移除反事实的来源与输入资格检查。"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from evaluate_localization import model, predict, iou


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-QUALIFY-01/20260921-r1'
SAMPLES = [0, 30, 60, 90, 120, 150, 180, 234]
SUPPORT_SAMPLES = [150, 180, 234]
EVENT_FRAME = 5
CASES = [
    {
        'log': '02678d04-cc9f-3148-9f95-1ba66347dff9',
        'target': '94dede14-59da-4f09-b016-95f19596ac08',
        'base': ROOT/'WS-V75-BRAKING-TASKS-01/20260920-r1/cases/02678d04-cc9f-3148-9f95-1ba66347dff9/start-6.5/base',
        'states': {
            'reference': ROOT/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2/gt_clean/condition_scene.json',
            'dvgt_metric': ROOT/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2/dvgt_metric/condition_scene.json',
            'class_prior': ROOT/'WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2/02678d04-cc9f-3148-9f95-1ba66347dff9/dvgt_class_prior/condition_scene.json',
        },
        'conditioning': ROOT/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-r1/conditioning',
    },
    {
        'log': '24642607-2a51-384a-90a7-228067956d05',
        'target': '56569385-722e-4413-ba4d-068307bca2ab',
        'base': ROOT/'WS-V75-BRAKING-DEV2-01/20260920-r1/cases/24642607-2a51-384a-90a7-228067956d05/start-6.5/base',
        'states': {
            'reference': ROOT/'WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1/gt_clean/condition_scene.json',
            'dvgt_metric': ROOT/'WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1/dvgt_metric/condition_scene.json',
            'class_prior': ROOT/'WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2/24642607-2a51-384a-90a7-228067956d05/dvgt_class_prior/condition_scene.json',
        },
        'conditioning': ROOT/'WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1/conditioning',
    },
]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def area(box):
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])


def coverage(front, back):
    intersection = max(0., min(front[2], back[2])-max(front[0], back[0])) * max(0., min(front[3], back[3])-max(front[1], back[1]))
    return intersection/max(area(back), 1e-9)


def max_occlusion(tracks, target, frame):
    back = target['projections'][frame]
    if back is None:
        return None
    rows=[]
    for track in tracks:
        if track is target:
            continue
        front=track['projections'][frame]
        if front is not None and front['depth_m'] < back['depth_m']:
            rows.append({'track': track['id'], 'coverage': coverage(front['bounds'], back['bounds']), 'depth_m': front['depth_m']})
    return max(rows, key=lambda x:x['coverage'], default={'track': None, 'coverage': 0., 'depth_m': None})


def topology(case):
    tracks=json.loads((case['base']/'projections.json').read_text())
    actor=next(x for x in tracks if x['id']==case['target'])
    a0=actor['projections'][0]
    rows=[]
    for behind in tracks:
        if behind is actor or a0 is None or behind['projections'][0] is None:
            continue
        b0=behind['projections'][0]
        initial_coverage=coverage(a0['bounds'], b0['bounds'])
        support=[]
        for f in SUPPORT_SAMPLES:
            b=behind['projections'][f]
            occ=max_occlusion(tracks, behind, f)
            if b is not None and area(b['bounds']) >= 600 and b['depth_m'] <= 80 and occ['coverage'] <= .05:
                support.append({'frame': f, 'bounds': b['bounds'], 'area_px2': area(b['bounds']),
                                'depth_m': b['depth_m'], 'max_nearer_box_occlusion': occ})
        passed=(a0['depth_m']+5 <= b0['depth_m'] and area(b0['bounds']) >= 150 and initial_coverage >= .5 and len(support) >= 2)
        rows.append({'track': behind['id'], 'category': behind['category'], 'initial_bounds': b0['bounds'],
                     'initial_area_px2': area(b0['bounds']), 'initial_depth_m': b0['depth_m'],
                     'initial_actor_coverage': initial_coverage, 'support': support, 'topology_passed': passed})
    return actor, tracks, rows


def match_vehicle(prediction, bounds):
    rows=[]
    for box,score,label in zip(prediction['boxes'],prediction['scores'],prediction['labels']):
        if int(label) not in [3,6,8] or float(score) < .5:
            continue
        box=box.tolist();rows.append({'box':box,'score':float(score),'class_id':int(label),'iou':iou(box,bounds)})
    return max(rows,key=lambda x:x['iou'],default=None)


def target(scene, identity):
    return next(t for t in scene['tracks'] if t['id']==identity and 0 in t['frames'])


def state_inputs(case, selected, out):
    reference=json.loads(case['states']['reference'].read_text())
    result={};reference_non_target=[t for t in reference['tracks'] if t['id']!=case['target']]
    for arm,path in case['states'].items():
        scene=json.loads(path.read_text());t=target(scene,case['target'])
        assert [x for x in scene['tracks'] if x['id']!=case['target']] == reference_non_target
        assert scene['lines']==reference['lines'] and scene['crossings']==reference['crossings']
        edited=copy.deepcopy(scene);e=target(edited,case['target'])
        keep=[i for i,f in enumerate(e['frames']) if f<EVENT_FRAME]
        assert keep and max(e['frames'][i] for i in keep)==EVENT_FRAME-1
        for key in ['frames','centers','quaternions']:
            e[key]=[e[key][i] for i in keep]
        path_out=out/f'condition-{arm}-removed-after-{EVENT_FRAME:03d}.json';save(path_out,edited)
        result[arm]={'source':str(path),'edited':str(path_out),'target_dimensions_m':t['dimensions'],
                     'initial_center_world_m':t['centers'][t['frames'].index(0)],
                     'retained_target_frames':e['frames'],'other_tracks_exact_reference':True,
                     'map_exact_reference':True}
    for arm in ['dvgt_metric','class_prior']:
        assert result[arm]['initial_center_world_m'] != result['reference']['initial_center_world_m'] or result[arm]['target_dimensions_m'] != result['reference']['target_dimensions_m']
    conditioning=case['conditioning']
    assert json.loads((conditioning/'result.json').read_text())['status']=='complete'
    assert np.array_equal(np.asarray(Image.open(conditioning/'initial_rgb.png')), np.asarray(Image.open(case['base']/'initial_rgb.png')))
    assert (conditioning/'embeddings.pt').is_file()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();out=a.output
    assert not out.exists();out.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-QUALIFY-01','run_id':out.name,
              'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_revision':'ff62e1b2',
              'source_scope':'exactly two exposed approach tasks, sorted by log then behind-track id; select at most one',
              'topology_gate':{'frame0_actor_coverage_min':.5,'actor_depth_margin_min_m':5,'behind_frame0_area_min_px2':150,
                               'support_frames':SUPPORT_SAMPLES,'support_area_min_px2':600,'support_depth_max_m':80,
                               'support_max_nearer_box_occlusion':.05,'support_frames_required':2},
              'real_rgb_gate':{'detector':'torchvision FasterRCNN ResNet50 FPN v2 COCO_V1','vehicle_class_ids':[3,6,8],
                               'score_min':.5,'match_iou_min':.3,'initial_actor_match_required':True,
                               'later_behind_matches_required':2},
              'intervention':'declared instantaneous removal after generated frames0-4; target absent from frame5; not natural actor motion',
              'input_gate':'same real initial RGB/text embeddings; exactly three existing state estimates; non-target tracks/map unchanged; raster probe is a separate phase',
              'information_roles':'later real RGB and GT cuboids are evaluation-only; not supplied to reconstruction, condition or generator',
              'stop':'if topology, real support, initial consistency or raster gate fails, no world-model generation and no source replacement; no threshold/source/edit-time search',
              'new_reconstruction_calls':0,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(out/'protocol.json',protocol)
    topology_rows=[];passing=[]
    for case in sorted(CASES,key=lambda x:x['log']):
        actor,tracks,rows=topology(case);topology_rows.append({'log':case['log'],'target':case['target'],'candidates':rows})
        for row in sorted((x for x in rows if x['topology_passed']),key=lambda x:x['track']):
            passing.append((case,row,actor,tracks))
    if not passing:
        result={'status':'no_qualified_topology','topology':topology_rows,'generation_admitted':False,
                'new_detector_calls':0,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
        save(out/'result.json',result);return
    case,behind_row,actor,tracks=passing[0]
    behind=next(x for x in tracks if x['id']==behind_row['track'])
    detector=model();began=time.monotonic();detections=[]
    sheet=Image.new('RGB',(3*426,3*270),'#142130');draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(SAMPLES):
        image=Image.open(case['base']/f'reference-{f:03d}.png').convert('RGB');pred=predict(detector,image)
        ap=actor['projections'][f];bp=behind['projections'][f]
        am=None if ap is None else match_vehicle(pred,ap['bounds']);bm=None if bp is None else match_vehicle(pred,bp['bounds'])
        detections.append({'frame':f,'actor_projection':ap,'behind_projection':bp,'actor_match':am,'behind_match':bm})
        d=ImageDraw.Draw(image)
        for projection,color,label in [(ap,'#ff5148','A'),(bp,'#00e6cf','B')]:
            if projection:
                d.rectangle(projection['bounds'],outline=color,width=3);d.text((projection['bounds'][0],projection['bounds'][1]-14),label,fill=color)
        for match,color in [(am,'#ffd453'),(bm,'#7dff72')]:
            if match and match['iou']>=.3:d.rectangle(match['box'],outline=color,width=2)
        x=(i%3)*426;y=(i//3)*270;draw.text((x+5,y+4),f'f={f} A={None if am is None else am["iou"]:.3f} B={None if bm is None else bm["iou"]:.3f}' if am and bm else f'f={f} A={None if am is None else round(am["iou"],3)} B={None if bm is None else round(bm["iou"],3)}',fill='white')
        sheet.paste(image.resize((426,239)),(x,y+27))
    sheet.save(out/'real-support-review.jpg',quality=94)
    support_frames={x['frame'] for x in behind_row['support']}
    initial=next(x for x in detections if x['frame']==0)
    later=[x for x in detections if x['frame'] in support_frames and x['behind_match'] is not None and x['behind_match']['iou']>=.3]
    real_pass=initial['actor_match'] is not None and initial['actor_match']['iou']>=.3 and len(later)>=2
    states=state_inputs(case,behind_row,out) if real_pass else None
    result={'status':'passed_pending_raster' if real_pass else 'failed_real_support','selected':{'log':case['log'],'actor':case['target'],'behind':behind_row['track'],
            'topology':behind_row,'event_frame':EVENT_FRAME,'state_inputs':states,'conditioning':str(case['conditioning']),'base':str(case['base'])},
            'topology':topology_rows,'real_detections':detections,'real_gate_passed':real_pass,
            'later_support_detector_matches':len(later),'later_support_match_frames':[x['frame'] for x in later],
            'generation_admitted':False,'raster_probe_required':real_pass,'new_detector_calls':len(SAMPLES),
            'new_reconstruction_calls':0,'world_model_generation_calls':0,'detector_wall_s':time.monotonic()-began,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'cuda_initialized':torch.cuda.is_initialized(),
            'human_verdict':None,'failure_ledger_delta':'none'}
    save(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['topology','real_detections']},ensure_ascii=False))


if __name__=='__main__':main()

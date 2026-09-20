"""冻结八个未曝光日志，寻找一个独立对象移除拓扑，不读取模型输出。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw

from evaluate_localization import model,predict
from qualify_actor_removal import area,coverage,max_occlusion,match_vehicle


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-SOURCES-01/20260921-r1'
LOGS=['2e3f2ae7-9ab9-3aef-a3ce-a0a97a0cb1ab','335aabef-269e-3211-a99d-2c3a3a8f8475',
      '3bffdcff-c3a7-38b6-a0f2-64196d130958','42f92807-0c5e-3397-bd45-9d5303b4db2a',
      '47286726-5dd4-4e26-bd2d-5324f429e445','4e3fedbb-847c-3d5b-8a62-c9ff84550985',
      '51bbdd4d-3065-34ae-b369-b6e0444f34db','52071780-5758-3ed4-8835-0d64ecdc5575']
SAMPLES=[0,30,60,90,120,150,180,234];SUPPORT=[150,180,234]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def pairs(tracks):
    rows=[]
    for actor in sorted(tracks,key=lambda x:(x['id'],x['segment'])):
        a0=actor['projections'][0]
        valid=all(actor['projections'][f] is not None and actor['projections'][f]['fully_inside'] for f in [0,4,30,60,90,116])
        if actor['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE'] or not valid or a0 is None or area(a0['bounds'])<1500:
            continue
        for behind in sorted(tracks,key=lambda x:(x['id'],x['segment'])):
            b0=behind['projections'][0]
            if behind is actor or behind['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE'] or b0 is None:continue
            initial=coverage(a0['bounds'],b0['bounds']);support=[]
            for f in SUPPORT:
                b=behind['projections'][f];occ=max_occlusion(tracks,behind,f)
                if b is not None and area(b['bounds'])>=600 and b['depth_m']<=80 and occ['coverage']<=.05:
                    support.append({'frame':f,'bounds':b['bounds'],'area_px2':area(b['bounds']),'depth_m':b['depth_m'],'max_occlusion':occ})
            passed=a0['depth_m']+5<=b0['depth_m'] and area(b0['bounds'])>=150 and initial>=.5 and len(support)>=2
            if passed:rows.append({'actor':actor['id'],'actor_category':actor['category'],'behind':behind['id'],
                                   'behind_category':behind['category'],'initial_actor_bounds':a0['bounds'],
                                   'initial_behind_bounds':b0['bounds'],'actor_depth_m':a0['depth_m'],'behind_depth_m':b0['depth_m'],
                                   'initial_actor_coverage':initial,'support':support})
    return rows


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-CONFIRM-SOURCES-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'source_revision':'ff62e1b2','logs':LOGS,'source_selection':'first8 lexicographic locally complete AV2 val logs not referenced in prior docs/run JSON at freeze time',
              'role':'independent source screen after exposed development signal; no reconstruction or generation outputs used',
              'window':'one fixed 7.9s window starting at +6.5s per log','order':'log, actor UUID, behind UUID; first real-supported pair selected; at most one',
              'actor_gate':'regular/large vehicle; fully inside at frames0,4,30,60,90,116; frame0 projected area>=1500px2',
              'occlusion_gate':{'frame0_actor_coverage_min':.5,'depth_margin_min_m':5,'behind_area_min_px2':150,
                                'support_frames':SUPPORT,'support_area_min_px2':600,'support_depth_max_m':80,'support_max_occlusion':.05,'support_required':2},
              'real_gate':'same frozen COCO detector; score>=.5, vehicle class3/6/8, IoU>=.3; A matched at frame0 and B in at least2 topology-support frames',
              'stop':'at most eight logs and one fixed window each; stop at first qualified pair; no replacement, offsets, thresholds, seed or model output inspection',
              'new_reconstruction_calls':0,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);detector=None;rows=[];selected=None;calls=0
    for log in LOGS:
        base=OUT/'cases'/log/'base';base.parent.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).with_name('prepare_argoverse.py')),'--output',str(base),'--log-id',log,
                 '--start-offset-seconds','6.5','--task-id',protocol['task_id'],'--run-id',OUT.name]
        with (base.parent/'prepare.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        tracks=json.loads((base/'projections.json').read_text());candidates=pairs(tracks);record={'log':log,'base':str(base),'topology_candidates':candidates}
        for candidate in candidates:
            if detector is None:detector=model()
            actor=next(x for x in tracks if x['id']==candidate['actor']);behind=next(x for x in tracks if x['id']==candidate['behind'])
            measurements=[]
            for f in SAMPLES:
                pred=predict(detector,Image.open(base/f'reference-{f:03d}.png').convert('RGB'));calls+=1
                ap=actor['projections'][f];bp=behind['projections'][f]
                measurements.append({'frame':f,'actor_match':None if ap is None else match_vehicle(pred,ap['bounds']),
                                     'behind_match':None if bp is None else match_vehicle(pred,bp['bounds'])})
            support={x['frame'] for x in candidate['support']};later=[x for x in measurements if x['frame'] in support and x['behind_match'] and x['behind_match']['iou']>=.3]
            first=measurements[0];passed=bool(first['actor_match'] and first['actor_match']['iou']>=.3 and len(later)>=2)
            candidate.update(real_measurements=measurements,real_gate_passed=passed,later_support_match_frames=[x['frame'] for x in later])
            if passed:
                selected={'log':log,'base':str(base),**candidate};break
        rows.append(record);save(OUT/'progress.json',{'logs':rows,'selected':selected,'new_detector_calls':calls})
        print(json.dumps({'log':log,'topology_candidates':len(candidates),'selected':None if selected is None else [selected['actor'],selected['behind']]}),flush=True)
        if selected:break
    if selected:
        tracks=json.loads((Path(selected['base'])/'projections.json').read_text());actor=next(x for x in tracks if x['id']==selected['actor']);behind=next(x for x in tracks if x['id']==selected['behind'])
        sheet=Image.new('RGB',(4*640,2*378),'#13202e');draw=ImageDraw.Draw(sheet)
        for i,f in enumerate(SAMPLES):
            image=Image.open(Path(selected['base'])/f'reference-{f:03d}.png').convert('RGB');d=ImageDraw.Draw(image)
            for x,color,label in [(actor,'#ff5148','A'),(behind,'#00e6cf','B')]:
                p=x['projections'][f]
                if p:d.rectangle(p['bounds'],outline=color,width=3);d.text((p['bounds'][0],p['bounds'][1]-13),label,fill=color)
            m=selected['real_measurements'][i]
            for item in [m['actor_match'],m['behind_match']]:
                if item and item['iou']>=.3:d.rectangle(item['box'],outline='#ffe86b',width=2)
            x=(i%4)*640;y=(i//4)*378;draw.text((x+6,y+4),f'f={f}',fill='white');sheet.paste(image.resize((640,352)),(x,y+24))
        sheet.save(OUT/'selected-real-support.jpg',quality=94)
    result={'status':'qualified' if selected else 'no_qualified_source','screened_logs':len(rows),'logs':rows,'selected':selected,
            'new_detector_calls':calls,'new_reconstruction_calls':0,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='logs'},ensure_ascii=False))


if __name__=='__main__':main()

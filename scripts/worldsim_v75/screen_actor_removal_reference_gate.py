"""冻结新的八日志输入窗口，寻找中远距离、小视觉占比的 reference-editability 来源。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from evaluate_localization import model, predict
from qualify_actor_removal import area, coverage, max_occlusion, match_vehicle


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-G1-SCREEN-01/20260921-r1'
LOGS=['544a8102-0ef5-3044-921e-dc0544370376','5589de60-1727-3e3f-9423-33437fc5da4b',
      '5c0584a3-52a6-3029-b6ff-ca45a19d8aa6','5f8f4a26-59b1-3f70-bcab-b5e3e615d3bc',
      '65387aee-4490-38b9-8f4f-1fc43bd4ac06','6f128f23-ee40-3ea9-8c50-c9cdb9d3e8b6',
      '7039e410-b5ab-35aa-96bc-2c4b89d3c5e3','7606de8d-486c-4916-9cbb-002ee966f834']
SAMPLES=[0,30,60,90,120,150,180,234];SUPPORT=[150,180,234]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def candidates(tracks):
    rows=[]
    for actor in sorted(tracks,key=lambda x:(x['id'],x['segment'])):
        a0=actor['projections'][0]
        valid=all(actor['projections'][f] is not None and actor['projections'][f]['fully_inside'] for f in [0,4,30,60,90,116])
        if actor['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE'] or not valid or a0 is None:continue
        aa=area(a0['bounds'])
        if not (1500<=aa<=8000 and 35<=a0['depth_m']<=70):continue
        for behind in sorted(tracks,key=lambda x:(x['id'],x['segment'])):
            b0=behind['projections'][0]
            if behind is actor or behind['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE'] or b0 is None:continue
            initial=coverage(a0['bounds'],b0['bounds']);support=[]
            for f in SUPPORT:
                b=behind['projections'][f];occ=max_occlusion(tracks,behind,f)
                if b is not None and area(b['bounds'])>=600 and b['depth_m']<=80 and occ['coverage']<=.05:
                    support.append({'frame':f,'bounds':b['bounds'],'area_px2':area(b['bounds']),'depth_m':b['depth_m'],'max_occlusion':occ})
            if a0['depth_m']+5<=b0['depth_m'] and area(b0['bounds'])>=150 and initial>=.5 and len(support)>=2:
                rows.append({'actor':actor['id'],'behind':behind['id'],'actor_bounds':a0['bounds'],'actor_area_px2':aa,
                             'actor_depth_m':a0['depth_m'],'behind_bounds':b0['bounds'],'behind_depth_m':b0['depth_m'],
                             'initial_actor_coverage':initial,'support':support})
    return rows


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-G1-SCREEN-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'source_revision':'61bff442','logs':LOGS,
      'source_selection':'first8 lexicographic locally complete AV2 val logs absent from repository docs and worldsim_v75 run JSON at freeze time',
      'question':'Does lower initial visual salience define a reference-editable envelope for actor removal?',
      'window':'one fixed 7.9s window starting at +6.5s per log','order':'log, actor UUID, behind UUID; first real-supported pair; at most one',
      'actor_gate':'vehicle fully inside frames0,4,30,60,90,116; frame0 area1500..8000px2 and depth35..70m',
      'motivation':'input-only contrast to exposed editable source (~48m) and independently frozen non-editable source (~20m,18.2k px2); not selected by model output',
      'occlusion_gate':{'coverage_min':.5,'depth_margin_min_m':5,'behind_area_min_px2':150,'support_frames':SUPPORT,
                        'support_area_min_px2':600,'support_depth_max_m':80,'support_max_occlusion':.05,'support_required':2},
      'real_gate':'same frozen COCO detector; A frame0 and B in >=2 topology-support frames, score>=.5, IoU>=.3',
      'followup':'if qualified, run reference unedited/removed only; no DVGT until reference gate passes',
      'stop':'eight logs, one window each, first qualified pair; no offsets/replacements/thresholds; no reconstruction or generation in screen',
      'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);detector=None;records=[];selected=None;calls=0
    for log in LOGS:
        base=OUT/'cases'/log/'base';base.parent.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).with_name('prepare_argoverse.py')),'--output',str(base),'--log-id',log,
                 '--start-offset-seconds','6.5','--task-id',protocol['task_id'],'--run-id',OUT.name]
        with (base.parent/'prepare.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        tracks=json.loads((base/'projections.json').read_text());rows=candidates(tracks);record={'log':log,'base':str(base),'candidates':rows}
        for candidate in rows:
            if detector is None:detector=model()
            actor=next(x for x in tracks if x['id']==candidate['actor']);behind=next(x for x in tracks if x['id']==candidate['behind'])
            measurements=[]
            for f in SAMPLES:
                prediction=predict(detector,Image.open(base/f'reference-{f:03d}.png').convert('RGB'));calls+=1
                ap=actor['projections'][f];bp=behind['projections'][f]
                measurements.append({'frame':f,'actor_match':None if ap is None else match_vehicle(prediction,ap['bounds']),
                                     'behind_match':None if bp is None else match_vehicle(prediction,bp['bounds'])})
            support={x['frame'] for x in candidate['support']}
            later=[x for x in measurements if x['frame'] in support and x['behind_match'] and x['behind_match']['iou']>=.3]
            passed=bool(measurements[0]['actor_match'] and measurements[0]['actor_match']['iou']>=.3 and len(later)>=2)
            candidate.update(real_measurements=measurements,real_gate_passed=passed,later_support_match_frames=[x['frame'] for x in later])
            if passed:selected={'log':log,'base':str(base),**candidate};break
        records.append(record);save(OUT/'progress.json',{'logs':records,'selected':selected,'new_detector_calls':calls})
        print(json.dumps({'log':log,'candidates':len(rows),'selected':None if selected is None else [selected['actor'],selected['behind']]}),flush=True)
        if selected:break
    result={'status':'qualified' if selected else 'no_qualified_source','screened_logs':len(records),'logs':records,'selected':selected,
            'new_detector_calls':calls,'new_reconstruction_calls':0,'world_model_generation_calls':0,
            'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='logs'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

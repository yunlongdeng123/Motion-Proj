"""冻结下一批八日志，寻找远距离、小投影的 reference-editability 来源。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess,sys

from PIL import Image

from evaluate_localization import model,predict
from qualify_actor_removal import area,coverage,max_occlusion,match_vehicle


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75');OUT=ROOT/'WS-V75-ACTOR-REMOVAL-G1-FAR-SCREEN-01/20260921-r1'
LOGS=['77574006-881f-3bc8-bbb6-81d79cf02d83','78f7cb5c-9d51-34f0-b356-9b3d83263c75',
      '7a2c222d-addc-30b2-aac6-596cb65a22e3','7dbc2eac-5871-3480-b322-246e03d954d2',
      '858d739b-a0ba-35aa-bafc-4f7988bcad17','8749f79f-a30b-3c3f-8a44-dbfa682bbef1',
      '87ca3d9f-f317-3efb-b1cb-aaaf525227e5','88f47a10-87b4-3ea8-a0c7-a07d825b647d']
SAMPLES=[0,30,60,90,120,150,180,234];SUPPORT=[150,180,234]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def candidates(tracks):
    rows=[]
    for actor in sorted(tracks,key=lambda x:(x['id'],x['segment'])):
        a0=actor['projections'][0]
        valid=all(actor['projections'][f] is not None and actor['projections'][f]['fully_inside'] for f in [0,4,30,60,90,116])
        if actor['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE'] or not valid or a0 is None:continue
        aa=area(a0['bounds'])
        if not (1500<=aa<=4500 and 45<=a0['depth_m']<=80):continue
        for behind in sorted(tracks,key=lambda x:(x['id'],x['segment'])):
            b0=behind['projections'][0]
            if behind is actor or behind['category'] not in ['REGULAR_VEHICLE','LARGE_VEHICLE'] or b0 is None:continue
            support=[]
            for f in SUPPORT:
                b=behind['projections'][f];occ=max_occlusion(tracks,behind,f)
                if b is not None and area(b['bounds'])>=600 and b['depth_m']<=80 and occ['coverage']<=.05:
                    support.append({'frame':f,'bounds':b['bounds'],'area_px2':area(b['bounds']),'depth_m':b['depth_m'],'max_occlusion':occ})
            initial=coverage(a0['bounds'],b0['bounds'])
            if a0['depth_m']+5<=b0['depth_m'] and area(b0['bounds'])>=150 and initial>=.5 and len(support)>=2:
                rows.append({'actor':actor['id'],'behind':behind['id'],'actor_bounds':a0['bounds'],'actor_area_px2':aa,'actor_depth_m':a0['depth_m'],
                  'behind_bounds':b0['bounds'],'behind_depth_m':b0['depth_m'],'initial_actor_coverage':initial,'support':support})
    return rows


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-G1-FAR-SCREEN-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'source_revision':'39fe0194','logs':LOGS,
      'source_selection':'next8 lexicographic locally complete AV2 val logs absent from repository docs and worldsim_v75 run JSON at freeze time',
      'question':'Can a far, small initial actor pass the reference editability gate on a new source?',
      'window':'one fixed 7.9s window starting at +6.5s per log','order':'log, actor UUID, behind UUID; first real-supported pair; at most one',
      'actor_gate':'vehicle fully inside frames0,4,30,60,90,116; frame0 area1500..4500px2 and depth45..80m',
      'motivation':'input-only far/small stratum after the mid-distance G1 failure; not selected by model output',
      'occlusion_gate':{'coverage_min':.5,'depth_margin_min_m':5,'behind_area_min_px2':150,'support_frames':SUPPORT,'support_area_min_px2':600,'support_depth_max_m':80,'support_max_occlusion':.05,'support_required':2},
      'real_gate':'same COCO detector; A frame0 and B in >=2 topology-support frames, score>=.5, IoU>=.3',
      'followup':'reference unedited/removed only; no DVGT until G1 passes',
      'stop':'eight logs, one window each, first qualified pair; no offsets/replacements/thresholds; no reconstruction/generation in screen','human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);detector=None;records=[];selected=None;calls=0
    for log in LOGS:
        base=OUT/'cases'/log/'base';base.parent.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).with_name('prepare_argoverse.py')),'--output',str(base),'--log-id',log,'--start-offset-seconds','6.5','--task-id',protocol['task_id'],'--run-id',OUT.name]
        with (base.parent/'prepare.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        tracks=json.loads((base/'projections.json').read_text());rows=candidates(tracks);record={'log':log,'base':str(base),'candidates':rows}
        for c in rows:
            if detector is None:detector=model()
            actor=next(x for x in tracks if x['id']==c['actor']);behind=next(x for x in tracks if x['id']==c['behind']);measurements=[]
            for f in SAMPLES:
                pr=predict(detector,Image.open(base/f'reference-{f:03d}.png').convert('RGB'));calls+=1;ap=actor['projections'][f];bp=behind['projections'][f]
                measurements.append({'frame':f,'actor_match':None if ap is None else match_vehicle(pr,ap['bounds']),'behind_match':None if bp is None else match_vehicle(pr,bp['bounds'])})
            sf={x['frame'] for x in c['support']};later=[x for x in measurements if x['frame'] in sf and x['behind_match'] and x['behind_match']['iou']>=.3]
            passed=bool(measurements[0]['actor_match'] and measurements[0]['actor_match']['iou']>=.3 and len(later)>=2)
            c.update(real_measurements=measurements,real_gate_passed=passed,later_support_match_frames=[x['frame'] for x in later])
            if passed:selected={'log':log,'base':str(base),**c};break
        records.append(record);save(OUT/'progress.json',{'logs':records,'selected':selected,'new_detector_calls':calls});print(json.dumps({'log':log,'candidates':len(rows),'selected':None if selected is None else [selected['actor'],selected['behind']]}),flush=True)
        if selected:break
    result={'status':'qualified' if selected else 'no_qualified_source','screened_logs':len(records),'logs':records,'selected':selected,'new_detector_calls':calls,'new_reconstruction_calls':0,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='logs'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

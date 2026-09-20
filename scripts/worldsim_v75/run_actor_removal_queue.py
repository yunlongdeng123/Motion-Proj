"""冻结并顺序运行对象移除的六段固定相机比较。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL=ROOT/'WS-V75-ACTOR-REMOVAL-QUALIFY-01/20260921-r2'
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-GENERATION-01/20260921-r1'
TASKS=[(arm,variant) for arm in ['reference','dvgt_metric','class_prior'] for variant in ['unedited','removed']]
EVAL_FRAMES=[4,5,13,21,29,37,45,61,85,109,116]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists();q=json.loads((QUAL/'result.json').read_text());r=json.loads((QUAL/'raster_result.json').read_text())
    assert q['status']=='qualified' and q['generation_admitted'] and r['status']=='passed'
    OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-GENERATION-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'source_revision':'ff62e1b2','qualification':str(QUAL),'log':q['selected']['log'],'actor':q['selected']['actor'],'behind':q['selected']['behind'],
              'arms':['reference','dvgt_metric','class_prior'],'variants':['unedited','removed'],'tasks':[list(x) for x in TASKS],
              'frames':117,'blocks':15,'fps':30,'seed':42,'event_frame':5,
              'fixed':['logged camera trajectory','initial image embeddings and text','map and non-target actors','generator weights/configuration/random seed'],
              'change':'target actor is present through frame4 and absent from conditions at frame5 onward; unedited pair retains it',
              'information_roles':'reference state uses GT cuboid; DVGT readout has prior documented GT dimensions/yaw; class prior is ordinary control; later real RGB/GT behind actor are evaluation-only',
              'role':'fixed-camera sensor-path intervention, not closed-loop and not a pixel-ground-truth counterfactual',
              'evaluation_frozen':{'frames':EVAL_FRAMES,'detector':'same COCO FasterRCNN, score>=0.5, vehicle classes3/6/8, IoU>=0.3',
                  'reference_gate':'unedited actor presence at >=6 post-event samples; removed actor presence <=2; removed behind presence >=4 and strictly above unedited',
                  'state_effect_gate':'after reference gate, DVGT differs from reference on >=3 sampled actor/behind presence decisions; ordinary class prior moves at least one of those decisions toward reference',
                  'invariant_readout':'report matched non-target annotated vehicles and paired outside-edit pixel change; no pass claim from pixel similarity alone'},
              'stop':'exactly six runs; any OOM/error stops the queue without retry, lower resolution, new seed/source/event time or threshold change; no feedback unless frozen gates pass',
              'human_verdict':None,'failure_ledger_refs':['V75-F01','V74-H2-F20','V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);save(OUT/'queue_manifest.json',{'tasks':[{'arm':a,'variant':v} for a,v in TASKS],'human_verdict':None})
    result={'status':'running','runs':[],'human_verdict':None,'failure_ledger_delta':'none'};save(OUT/'queue_result.json',result);began=time.monotonic()
    script=Path(__file__).with_name('run_actor_removal_generation.py')
    for arm,variant in TASKS:
        command=[sys.executable,str(script),'--arm',arm,'--variant',variant,'--output',str(OUT),'--qualification',str(QUAL)]
        with (OUT/f'{arm}-{variant}.log').open('w') as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        path=OUT/f'{arm}-{variant}/result.json';child=json.loads(path.read_text()) if path.exists() else {'status':'missing_result'}
        result['runs'].append({'arm':arm,'variant':variant,'exit_code':process.returncode,'result':child});save(OUT/'queue_result.json',result)
        print(json.dumps({'arm':arm,'variant':variant,'exit_code':process.returncode,'status':child['status']}),flush=True)
        if process.returncode or child['status']!='complete':
            result.update(status='oom_stopped' if child['status']=='oom_stopped' else 'failed_stopped',wall_s=time.monotonic()-began)
            save(OUT/'queue_result.json',result);return
    result.update(status='complete',wall_s=time.monotonic()-began,completed_runs=len(result['runs']),world_model_sequences=6,generation_forwards=90,generated_frames=702)
    save(OUT/'queue_result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__=='__main__':main()

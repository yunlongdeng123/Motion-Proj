"""只运行新来源的 reference unedited/removed 两段 G1。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess,sys,time


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL=ROOT/'WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01/20260921-r1'
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1'
TASKS=[('reference','unedited'),('reference','removed')]
EVAL=[4,5,13,21,29,37,45,61,85,109,116]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists();q=json.loads((QUAL/'result.json').read_text());r=json.loads((QUAL/'raster_result.json').read_text());c=json.loads((QUAL/'conditioning/result.json').read_text())
    assert q['status']=='qualified' and r['status']=='passed' and c['status']=='complete';OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'source_revision':'61bff442','qualification':str(QUAL),'log':q['selected']['log'],'actor':q['selected']['actor'],'behind':q['selected']['behind'],
      'arms':['reference'],'variants':['unedited','removed'],'tasks':[list(x) for x in TASKS],'frames':117,'blocks':15,'fps':30,'seed':42,'event_frame':5,
      'role':'reference-state editability gate before any reconstruction call','fixed':['initial RGB/text','map/non-target actors','recorded camera','seed42','generator'],
      'evaluation_frozen':{'frames':EVAL,'reference_gate':'unedited A>=6; removed A<=2; removed B>=4 and >unedited B'},
      'followup':'only a passing reference gate admits one DVGT readout and remaining reconstruction arms',
      'stop':'two sequences exactly; OOM/error stops without retry/config/source/event/threshold change','human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);result={'status':'running','runs':[],'human_verdict':None,'failure_ledger_delta':'none'};save(OUT/'queue_result.json',result);began=time.monotonic()
    script=Path(__file__).with_name('run_actor_removal_generation.py')
    for arm,variant in TASKS:
        command=[sys.executable,str(script),'--arm',arm,'--variant',variant,'--output',str(OUT),'--qualification',str(QUAL)]
        with (OUT/f'{arm}-{variant}.log').open('w') as stream:child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        path=OUT/f'{arm}-{variant}/result.json';terminal=json.loads(path.read_text()) if path.exists() else {'status':'missing_result'}
        result['runs'].append({'arm':arm,'variant':variant,'exit_code':child.returncode,'result':terminal});save(OUT/'queue_result.json',result)
        print(json.dumps({'variant':variant,'exit_code':child.returncode,'status':terminal['status']}),flush=True)
        if child.returncode or terminal['status']!='complete':
            result.update(status='oom_stopped' if terminal['status']=='oom_stopped' else 'failed_stopped',wall_s=time.monotonic()-began);save(OUT/'queue_result.json',result);return
    result.update(status='complete',wall_s=time.monotonic()-began,completed_runs=2,world_model_sequences=2,generation_forwards=30,generated_frames=234)
    save(OUT/'queue_result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__=='__main__':main()

"""运行单一 reference slowdown 序列；unedited 复用同 seed 已有结果。"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess,sys,time


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75');QUAL=ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01/20260921-r1'
SOURCE=ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1';OUT=ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01/20260921-r1'
EVAL=[4,5,13,21,29,37,45,61,85,109,116];LATE=[45,61,85,109,116]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    global QUAL,SOURCE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',type=Path,default=QUAL)
    parser.add_argument('--source-generation',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();QUAL,SOURCE,OUT=args.qualification,args.source_generation,args.output
    assert not OUT.exists();q=json.loads((QUAL/'result.json').read_text());assert q['status']=='qualified' and json.loads((SOURCE/'reference-unedited/result.json').read_text())['status']=='complete';OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_revision':'7a5f22d2',
      'qualification':str(QUAL),'source_unedited':str(SOURCE/'reference-unedited'),'log':q['selected']['log'],'actor':q['selected']['actor'],
      'arms':['reference'],'variants':['edited'],'frames':117,'blocks':15,'fps':30,'seed':42,'event_frame':5,
      'fixed':['same initial RGB/text embeddings','same map/non-target actors/camera/generator/seed42','exact condition through frame4'],
      'evaluation_frozen':{'frames':EVAL,'late_separable_frames':LATE,'unedited_gate':'original projection matched at >=4/5 late frames',
        'edited_gate':'edited projection matched at >=4/5 late frames and matched detection center is >=5px closer to edited than original projection at >=4/5'},
      'followup':'passing G1 admits one DVGT readout; failure closes this source without reconstruction',
      'stop':'one new sequence exactly; no factor/event/source/seed/threshold search; OOM/error stops','human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);began=time.monotonic();script=Path(__file__).with_name('run_actor_removal_generation.py')
    command=[sys.executable,str(script),'--arm','reference','--variant','edited','--output',str(OUT),'--qualification',str(QUAL)]
    with (OUT/'reference-edited.log').open('w') as stream:child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    path=OUT/'reference-edited/result.json';terminal=json.loads(path.read_text()) if path.exists() else {'status':'missing_result'}
    result={'status':'complete' if child.returncode==0 and terminal['status']=='complete' else ('oom_stopped' if terminal['status']=='oom_stopped' else 'failed_stopped'),
      'run':{'exit_code':child.returncode,'result':terminal},'wall_s':time.monotonic()-began,'world_model_sequences':1 if terminal['status']=='complete' else 0,
      'generation_forwards':15 if terminal['status']=='complete' else 0,'generated_frames':117 if terminal['status']=='complete' else 0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'queue_result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='run'}),flush=True)


if __name__=='__main__':main()

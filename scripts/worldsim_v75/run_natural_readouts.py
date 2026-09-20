"""固定四个可见目标顺序执行，任一OOM立即停整个队列。"""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

P=Path('/root/autodl-tmp/motion_proj')
S=P/'scripts/worldsim_v75'
ROOT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-SOURCES-01/20260920-r1')
PY='/root/autodl-tmp/envs/worldsim-v75/bin/python'
DV='/root/autodl-tmp/envs/worldsim-v81/bin/python'
VISIBLE={
    '04994d08-156c-3018-9717-ba0e29be8153':'exclude: nearer parked SUV occludes left/front portion of selected silver car',
    '05fa5048-f355-3274-b565-c0ddc547b315':'pass: parked dark car, front and side visible',
    '0bae3b5e-417d-3b03-abaa-806b433233b8':'pass: white turning car, rear and side visible',
    '0c3bad78-9f1e-395d-a376-2eb7499229fd':'pass: parked white SUV, rear visible',
    '0fb7276f-ecb5-3e5b-87a8-cc74c709c715':'pass: silver sedan ahead, rear visible',
    '185d3943-dd15-397a-8b2e-69cd86628fb7':'exclude: blue foreground SUV occludes selected dark sedan left portion'}

def call(script,log,python=PY,args=()):
    with log.open('w') as output:
        result=subprocess.run([python,str(S/script),*map(str,args)],cwd=P,stdout=output,stderr=subprocess.STDOUT,
                              env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'})
    if result.returncode:
        # 只返回类型和日志位置；不吞掉OOM再进入下一模型。
        raise RuntimeError(f'{script} stopped rc={result.returncode}; log={log}')

def main():
    protocol_path=ROOT/'readout_protocol.json'
    assert not protocol_path.exists(),'拒绝重复执行本队列'
    protocol={'task_id':'WS-V75-NATURAL-SOURCES-01','run_id':'20260920-r1',
              'frozen_utc':datetime.now(timezone.utc).isoformat(),'visual_review':VISIBLE,
              'reviewer':'assistant inspected actual initial RGB contact sheets before model inference; not human verdict',
              'readout':'official DVGT-1 raw ego range + known calibrated rays; RGB detector and same GT dimensions/yaw; background LiDAR scale separately',
              'controls':'same frozen natural readout criteria; raw point orientation separately audited, not silently assumed known',
              'no_replacement_for_occluded_targets':True,'human_verdict':None}
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    screen=json.loads((ROOT/'screen_result.json').read_text());rows=[]
    result={'status':'running','pid':os.getpid(),'cases':rows,'human_verdict':None}
    began=time.monotonic()
    try:
        for row in screen['logs']:
            log=row['log_id']
            if not VISIBLE.get(log,'exclude').startswith('pass'):continue
            case=ROOT/'cases'/log;case.mkdir(parents=True)
            base=case/'base';out=case/'reconstruction'
            print(json.dumps({'stage':'prepare','log':log}),flush=True)
            call('prepare_argoverse.py',case/'prepare.log',args=['--output',base,'--log-id',log,'--start-offset-seconds','0.5',
                 '--task-id','WS-V75-NATURAL-SOURCES-01','--run-id','20260920-r1'])
            call('prepare_natural.py',case/'prepare_readout.log',args=['--base-dir',base,'--output',out,'--target',row['selected_target'],
                 '--task-id','WS-V75-NATURAL-SOURCES-01'])
            print(json.dumps({'stage':'inference','log':log}),flush=True)
            call('infer_natural.py',case/'inference.log',python=DV,args=['--run-dir',out])
            call('audit_natural_geometry.py',case/'audit.log',args=['--run-dir',out])
            call('readout_natural.py',case/'readout.log',args=['--run-dir',out,'--calibrated-rays'])
            read=json.loads((out/'readout_ray_control_result.json').read_text())
            record={'log_id':log,'target':row['selected_target'],'readout_dir':str(out),
                    'admitted':read['generation_admitted'],'stop_reasons':read['stop_reasons'],
                    'readouts':read['readouts'],'scale':read['metric_scale_control'],
                    'inference':json.loads((out/'inference_result.json').read_text())}
            rows.append(record);(ROOT/'readout_queue_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'stage':'readout_complete','log':log,'admitted':read['generation_admitted'],
                              'reasons':read['stop_reasons'],
                              'errors_m':{k:(v['center_error_m'] if v else None) for k,v in read['readouts'].items()}}),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-began
        (ROOT/'readout_queue_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()

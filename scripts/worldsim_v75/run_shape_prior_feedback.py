"""有限的两任务两导出接口反馈；协议全先冻结，任一错误或OOM停止整队列。"""
from datetime import datetime,timezone
from pathlib import Path
import copy,json,subprocess,sys,time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2'
AUDIT=ROOT/'WS-V75-SHAPE-PRIOR-AUDIT-01/20260920-r1'
ARMS=['dvgt_class_prior','dvgt_visible_extent']


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    audit=json.loads((AUDIT/'result.json').read_text());ap=json.loads((AUDIT/'protocol.json').read_text())
    assert audit['status']=='complete' and len(audit['cases'])==2
    queue=[]
    protocol={'task_id':'WS-V75-SHAPE-FEEDBACK-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'audit_protocol':str(AUDIT/'protocol.json'),'audit_result':str(AUDIT/'result.json'),
              'reason':'ordinary priors remove target GT shape; near-gap difference below4mm and direct progress difference below2mm, while exported width/height differ; test whether RGB feedback changes',
              'scope':'both original exposed tasks, same two ordinary adapters, seed42,117frames each; not a pure dimension-only causal intervention because exported centers also differ',
              'comparison':'two actual feedback arms and existing original-GT-shape DVGT arm; reference remains original physical scene; no new GT/old-task rerun',
              'recovery_boundary':'better projected-box fit is not automatically better shape or driving; report all action/progress/clearance tradeoffs; no method claim from a single favorable metric',
              'stop':'four runs exactly; any nonzero child/failed result/OOM stops the full queue; no retries/configuration reduction/seeds/horizon/source expansion',
              'human_verdict':None,'failure_ledger_refs':['V74-H2-F20','V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol)
    for source,case in zip(ap['sources'],audit['cases']):
        parent=Path(source['generated_run']);sp=json.loads((parent/'protocol.json').read_text())
        assert sp['source_log']==case['log_id'] and sp['target']==case['target']
        conditioning=Path(sp['conditioning_dir'])
        assert json.loads((conditioning/'result.json').read_text())['status']=='complete'
        assert (conditioning/'embeddings.pt').is_file() and (conditioning/'initial_rgb.png').is_file()
        base=Path(sp['base']);reference=json.loads((base/'scene.json').read_text())
        target=next(t for t in reference['tracks'] if t['id']==sp['target'] and 0 in t['frames'])
        original_centers=np.array(target['centers']);original_R=Rotation.from_quat(target['quaternions']).as_matrix()
        folder=OUT/case['log_id'];folder.mkdir()
        conditions={};inputs=[]
        for arm in ARMS:
            row=next(x for x in case['rows'] if x['arm']==arm)
            assert row['readout_numerically_supported'] and row['direct_control'] is not None
            condition=copy.deepcopy(reference)
            t=next(t for t in condition['tracks'] if (t['id'],t['segment'])==(target['id'],target['segment']))
            t['centers']=(original_centers-original_centers[0]+row['center_world']).tolist()
            t['dimensions']=row['dimensions_m']
            rotation=original_R@original_R[0].T@np.array(row['rotation_world'])
            t['quaternions']=Rotation.from_matrix(rotation).as_quat().tolist()
            path=folder/f'condition-{arm}.json';save(path,condition);conditions[arm]=str(path)
            inputs.append(row)
        gap_delta=inputs[1]['task_geometry']['unclipped_near_route_gap_m']-inputs[0]['task_geometry']['unclipped_near_route_gap_m']
        progress_delta=inputs[1]['direct_control']['progress_m']-inputs[0]['direct_control']['progress_m']
        assert abs(gap_delta)<.0041 and abs(progress_delta)<.002
        frozen={**protocol,'base':str(base),'target':sp['target'],'source_log':case['log_id'],
                'source_generated_run':str(parent),'source_real_run':sp['source_run'],'conditions':conditions,
                'conditioning_dir':str(conditioning),'frames':117,'seed':42,
                'audit_case_result':str(AUDIT/case['log_id']/'result.json'),
                'near_gap_difference_m':gap_delta,'direct_progress_difference_m':progress_delta,
                'extra_information':'ego/calibration/route/terrain; shared GT relative motion and other actors; no target GT initial shape in fitted state'}
        save(folder/'frozen_state_protocol.json',frozen)
        for arm in ARMS:queue.append({'source_log':case['log_id'],'arm':arm,'folder':str(folder),'frozen':str(folder/'frozen_state_protocol.json'),
                                     'source_real_run':sp['source_run'],'conditioning_dir':str(conditioning)})
    save(OUT/'queue_manifest.json',{'tasks':queue,'human_verdict':None})
    started=time.monotonic();result={'status':'running','runs':[],'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'queue_result.json',result)
    for task in queue:
        folder=Path(task['folder'])
        command=[sys.executable,str(Path(__file__).with_name('run_approach_closed_loop.py')),'--phase','generate','--arm',task['arm'],
                 '--run-dir',str(folder),'--source-run',task['source_real_run'],'--conditioning-dir',task['conditioning_dir'],
                 '--task-id',protocol['task_id'],'--frozen-state-protocol',task['frozen']]
        print(json.dumps({'stage':'start','log':task['source_log'],'arm':task['arm']}),flush=True)
        with (folder/f'{task["arm"]}.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        terminal=folder/task['arm']/'result.json'
        child=json.loads(terminal.read_text()) if terminal.exists() else {'status':'missing_result'}
        result['runs'].append({**task,'exit_code':process.returncode,'result':child})
        print(json.dumps({'stage':'finished','log':task['source_log'],'arm':task['arm'],'status':child['status']}),flush=True)
        if process.returncode or child['status']!='complete':
            result.update(status='oom_stopped' if child['status']=='oom_stopped' else 'failed_stopped',wall_s=time.monotonic()-started)
            save(OUT/'queue_result.json',result);return
        save(OUT/'queue_result.json',result)
    result.update(status='complete',wall_s=time.monotonic()-started);save(OUT/'queue_result.json',result)
    print(json.dumps({'status':result['status'],'runs':len(result['runs']),'wall_s':result['wall_s']}),flush=True)


if __name__=='__main__':main()

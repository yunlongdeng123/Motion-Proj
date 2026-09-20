"""一次seed43复核：两个GT先过原门控，再跑四个原样状态；不追加搜索。"""
import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
SOURCE=ROOT/'WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2'
OUT=ROOT/'WS-V75-SHAPE-CONFIRM-01/20260920-r1'
ARMS=['dvgt_class_prior','dvgt_visible_extent']


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    assert json.loads((SOURCE/'queue_result.json').read_text())['status']=='complete'
    protocol={'task_id':'WS-V75-SHAPE-CONFIRM-01','run_id':OUT.name,'source_discovery':str(SOURCE),
              'frozen_utc':datetime.now(timezone.utc).isoformat(),'seed':43,'frames':117,
              'scope':'one prespecified additional seed, both exposed development tasks; not independent-source confirmation',
              'inputs':'exact source ordinary condition JSON and cached initial embeddings; no new fitting, reconstruction, geometry change or policy tuning',
              'primary_log':'02678d04-cc9f-3148-9f95-1ba66347dff9',
              'primary_replication_rule':{'generated_progress_extent_minus_class_gt_m':1.0,
                  'mean_abs_action_error_extent_gt_class':True,'max_underbraking_extent_gt_class':True},
              'rule_interpretation':'chosen after seed42 discovery and frozen before any seed43 output; conjunction required for primary candidate, second case fully reported without selection',
              'recovery_rule':'report mean/median error, max under/overbraking, progress and clearance together; class prior cannot be called full recovery while any seed fails original absolute baseline gate',
              'baseline_gate':{'median_abs_action_error_at_own_ego_mps2_max':.5,'max_underbraking_mps2':2.,'max_overbraking_mps2':2.,'max_route_lateral_m':1.,'no_overlap':True},
              'stop':'at most6 runs; GT baseline/dense failure or engineering error stops queue; any OOM stops immediately without retry/downsize; if primary rule fails close this candidate, no additional seeds/threshold/horizon/source search',
              'human_verdict':None,'failure_ledger_refs':['V74-H2-F20','V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol)
    cases=[]
    for src in sorted(p for p in SOURCE.iterdir() if p.is_dir() and (p/'frozen_state_protocol.json').exists()):
        old=json.loads((src/'frozen_state_protocol.json').read_text())
        folder=OUT/old['source_log'];folder.mkdir()
        assert json.loads((Path(old['conditioning_dir'])/'result.json').read_text())['status']=='complete'
        for arm in ARMS:
            assert json.loads(Path(old['conditions'][arm]).read_text())==json.loads((src/arm/'condition_scene.json').read_text())
        frozen={**old,'task_id':protocol['task_id'],'run_id':OUT.name,'seed':43,
                'frozen_utc':protocol['frozen_utc'],'confirmation_protocol':str(OUT/'protocol.json'),
                'source_discovery_case':str(src),'conditions':{'gt_clean':str(Path(old['base'])/'scene.json'),**old['conditions']},
                'scope':protocol['scope'],'stop':protocol['stop']}
        save(folder/'frozen_state_protocol.json',frozen);cases.append((folder,frozen))
    assert len(cases)==2
    queue=[(f,p,'gt_clean') for f,p in cases]+[(f,p,a) for f,p in cases for a in ARMS]
    save(OUT/'queue_manifest.json',{'tasks':[{'folder':str(f),'arm':a} for f,p,a in queue],'human_verdict':None})
    began=time.monotonic();result={'status':'running','runs':[],'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'queue_result.json',result)
    for folder,p,arm in queue:
        command=[sys.executable,str(Path(__file__).with_name('run_approach_closed_loop.py')),'--phase','generate','--arm',arm,
                 '--run-dir',str(folder),'--source-run',p['source_real_run'],'--conditioning-dir',p['conditioning_dir'],
                 '--task-id',protocol['task_id'],'--frozen-state-protocol',str(folder/'frozen_state_protocol.json')]
        print(json.dumps({'stage':'start','log':p['source_log'],'arm':arm,'seed':43}),flush=True)
        with (folder/f'{arm}.log').open('w') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        path=folder/arm/'result.json';child=json.loads(path.read_text()) if path.exists() else {'status':'missing_result'}
        entry={'log':p['source_log'],'arm':arm,'source':str(folder/arm),'exit_code':run.returncode,'result':child}
        result['runs'].append(entry)
        if run.returncode or child['status']!='complete':
            result.update(status='oom_stopped' if child['status']=='oom_stopped' else 'failed_stopped',wall_s=time.monotonic()-began)
            save(OUT/'queue_result.json',result);return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        with (folder/f'{arm}-replay.log').open('w') as log:
            replay=subprocess.run([sys.executable,str(Path(__file__).with_name('replay_shape_confirmation.py')),'--folder',str(folder),'--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT)
        if replay.returncode:
            result.update(status='replay_failed_stopped',wall_s=time.monotonic()-began);save(OUT/'queue_result.json',result);return
        entry['assessment']=json.loads((folder/arm/'assessment.json').read_text())
        if arm=='gt_clean' and (not child['baseline_admitted'] or json.loads((folder/arm/'dense_reference_result.json').read_text())['status']!='passed'):
            result.update(status='baseline_failed_stopped',wall_s=time.monotonic()-began);save(OUT/'queue_result.json',result);return
        print(json.dumps({'stage':'finished','log':p['source_log'],'arm':arm,'assessment':entry['assessment']}),flush=True)
        save(OUT/'queue_result.json',result)
    result.update(status='complete',wall_s=time.monotonic()-began);save(OUT/'queue_result.json',result)
    print(json.dumps({'status':'complete','runs':6,'wall_s':result['wall_s']}),flush=True)


if __name__=='__main__':main()

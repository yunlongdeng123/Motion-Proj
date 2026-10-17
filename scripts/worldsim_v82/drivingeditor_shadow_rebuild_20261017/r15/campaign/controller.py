"""Finite restored-backend run, resumable chunks, explicit root visual reviews."""
from pathlib import Path
import os,json,time,subprocess,shutil,datetime
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';I=R/'implementation_r15_train_v3/shadow_rebuild_v3';PY='/root/autodl-tmp/envs/drivingeditor/bin/python';LOCK='/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock'
def state(status,**kw):
    v=dict(status=status,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);p=D/'pipeline_state.tmp';p.write_text(json.dumps(v,indent=2));p.replace(D/'pipeline_state.json')
def command(script,*args,gpu=False):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
    while gpu and subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():state('WAITING_OTHER_GPU_TASK',next_script=Path(script).name);time.sleep(30)
    cmd=[PY,str(script),*map(str,args)]
    if gpu:cmd=['flock',LOCK,*cmd]
    subprocess.run(cmd,cwd=R,env=env,check=True)
def audit(step,run,out):
    command(D/'ownership_audit_r14.py','--run',Path(run).relative_to(R),'--step',step,'--output',Path(out).relative_to(R),'--implementation','implementation_r15_train_v3','--data','backend_restore_r15/data','--all-identities','--certified-road',gpu=True)
# 人类明确暂停优先于旧连续训练授权；在任何自查、GPU 查询和子进程前退出。
pause_marker=D/'PAUSE_REQUESTED.json'
if pause_marker.exists():
    paused=json.loads(pause_marker.read_text())
    state('PAUSED_BY_USER',checkpoint=paused.get('checkpoint'),global_step=paused.get('global_step'),
          resume_training_authorized=False,shutdown_authorized=False)
    print('R15 remains paused by the latest human request; no worker launched.',flush=True)
    raise SystemExit(0)
try:
    proof=json.loads((D/'GPU_selfcheck.json').read_text());assert proof['status']=='ACTUAL_PRODUCTION_CAPACITY_GRADIENTS_AND_RESTORE_PASSED' and proof['steps']==0
    first=json.loads((D/'schedule_recovery_verified.json').read_text());assert first['actual_main_updates']==2 and first['Adam_moments_bitwise_equal']
    assert json.loads((D/'main/progress.json').read_text())['global_step']==2,'Only the verified schedule-repair checkpoint may be adopted'
    step=2;boundaries=list(range(1000,45001,1000))
    for end in boundaries:
        state('FULL_TRAINING',from_step=step,to_step=end,total=45000,implementation='implementation_r15_train_v3')
        args=['train','--data',D/'data','--config',D/'config.json','--output',D/'main','--execute','--allow-partial-masks','--max-updates',end-step]
        if step:args+=['--resume',D/'main/latest.pt']
        command(I/'run.py',*args,gpu=True);step=end
        assert json.loads((D/'main/progress.json').read_text())['global_step']==step
        if step in [3000,9000]:
            G=D/'gates'/f'{step:06d}';G.mkdir(parents=True,exist_ok=False);shutil.copy2(D/'main/latest.pt',D/'main'/f'pre_owner_{step:06d}.pt')
            state('ACTUAL_OWNER_REVIEW',step=step)
            audit(step,D/'main',G/'before');command(D/f'split_{step:06d}.py');audit(step,G/'split',G/'after_split');command(D/f'owner_{step:06d}.py')
        if step in [1000,3000,9000,30000,40000,45000]:
            state('DIAGNOSTIC_PREVIEW',step=step);command(D/'preview.py',gpu=True)
        if step in [3000,9000,30000,40000]:
            marker=D/f'root_review_{step:06d}.json';state('WAITING_ROOT_VISUAL_REVIEW',step=step,review=str(marker),shutdown=False)
            while not marker.exists():time.sleep(30)
            if json.loads(marker.read_text()).get('continue_registered_training') is not True:raise RuntimeError('Root review found an engineering issue; preserve checkpoint and diagnose')
    state('RENDERING_COMPLETE49_EDITS',training_updates=45000);command(D/'final_evaluate.py',gpu=True)
    state('WAITING_ROOT_COMPLETE49_REVIEW_AND_DELIVERY',training_updates=45000,auto_shutdown=False)
except BaseException as e:
    state('ENGINEERING_FAILURE_PRESERVE_AND_DIAGNOSE_NO_SHUTDOWN',error=repr(e));raise

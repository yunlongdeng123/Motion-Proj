"""One full finite experiment. Initialization diagnostics never trigger shutdown."""
from pathlib import Path
import os,json,subprocess,time,shutil,argparse,datetime
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'ownership_rebuild_r14';py='/root/autodl-tmp/envs/drivingeditor/bin/python';impl=R/'implementation_r14_final/shadow_rebuild_v3'
def state(status,**kw):
    value={'status':status,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw};tmp=D/'pipeline_state.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(D/'pipeline_state.json')
def command(script,*args,gpu=False,interpreter=py):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
    cmd=[interpreter,str(script),*map(str,args)]
    if gpu:
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():state('WAITING_OTHER_GPU_TASK',next_script=Path(script).name);time.sleep(30)
        cmd=['flock','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock',*cmd]
    subprocess.run(cmd,cwd=R,env=env,check=True)
def audit(step,run,out):
    command(D/'ownership_audit_r14.py','--run',str(Path(run).relative_to(R)),'--step',step,'--implementation','implementation_r14_final','--data','ownership_rebuild_r14/idsplat_frontend/data_IDSplat','--certified-road','--all-identities','--output',str(Path(out).relative_to(R)),gpu=True)
try:
    assert (D/'GPU_probe.json').exists(),'real ABI/throughput must pass before full training'
    run=D/'main'
    if run.exists():raise FileExistsError('Do not start duplicate full experiment; inspect actual controller/latest checkpoint')
    state('INDEPENDENT_DINOV3_RANSAC_NEW_POINTSETS_ZERO_SGD')
    command(D/'dinov3_registration_r14.py',gpu=True,interpreter=str(R/'ownership_controls_r12_sam3/runtime_sam3_dinov3/bin/python'))
    step=0
    # 1k chunks release the GPU slot for other work and give resumable checkpoints.
    boundaries=list(range(1000,13001,1000))+list(range(14000,20001,1000))+[21000,22000,22400]
    for end in boundaries:
        state('FULL_TRAINING',from_step=step,to_step=end,planned_total=22400)
        args=['train','--data',D/'idsplat_frontend/data_IDSplat','--config',D/'config.json','--output',run,'--execute','--allow-partial-masks','--max-updates',end-step]
        if step:args+=['--resume',run/'latest.pt']
        command(impl/'run.py',*args,gpu=True);step=end
        actual=json.loads((run/'progress.json').read_text())
        if actual['global_step']!=step:raise RuntimeError('actual update count differs from registered chunk')
        if step in [3000,9000]:
            key=f'{step:06d}';G=D/'gates'/key;G.mkdir(parents=True,exist_ok=False);shutil.copyfile(run/'latest.pt',run/f'pre_owner_{key}.pt')
            state('ACTUAL_OWNER_REVIEW',step=step,phase='fresh full-scene evidence, then3Dsplit/reaudit/actual owner')
            audit(step,run,G/'before');command(D/f'split_{key}.py');audit(step,G/'split',G/'after_split');command(D/f'owner_{key}.py')
        if step in [3000,9000,13000,20000,22400]:
            state('DIAGNOSTIC_PREVIEW_NOT_STOP_GATE',step=step);command(D/'preview.py',gpu=True)
    # Re-enter completed checkpoint only to emit TRAINING_FINISHED receipt; no updates.
    command(impl/'run.py','train','--data',D/'idsplat_frontend/data_IDSplat','--config',D/'config.json','--output',run,'--execute','--allow-partial-masks','--resume',run/'latest.pt',gpu=True)
    state('RENDERING_COMPLETE49_EDITS',steps=22400)
    command(D/'final_evaluate.py',gpu=True)
    state('WAITING_ROOT_FULL_TRAINING_AND49FRAME_REVIEW',training_updates=22400,auto_shutdown=False,required='root review, complete local report+video/source delivery and GitHub push before all-idle poweroff')
except BaseException as e:
    state('ENGINEERING_FAILURE_PRESERVE_AND_DIAGNOSE_NO_SHUTDOWN',error=repr(e));raise

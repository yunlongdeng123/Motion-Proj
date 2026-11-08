"""Own the smoke lifecycle and fresh 10K native phase through its read-only evaluation."""
from pathlib import Path
import json,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text())
def status(stage,**kw):(ROOT/'native_pipeline_status.json').write_text(json.dumps(dict(stage=stage,unix=time.time(),controller_pid=os.getpid(),**kw),indent=2))
def run(name,*args):
    log=ROOT/(name+'.log')
    with log.open('x') as f:
        process=subprocess.Popen([sys.executable,'-B',*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
        status(name,pid=process.pid,log=str(log));code=process.wait()
    if code:raise RuntimeError(f'{name} failed with {code}; inspect {log}')
def main():
    smoke=ROOT/'smoke_r5';status('WAIT_SMOKE_COMPLETE')
    while not (smoke/'returned.json').exists():time.sleep(5)
    done=read(smoke/'returned.json');step=done['iteration'];assert done['elapsed_seconds']>=600
    for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']:assert (smoke/f'{prefix}{step}.pth').is_file()
    run('smoke_eval_r2',ROOT/'source/evaluate_scene.py','--model',smoke,'--iteration',step,'--output',ROOT/'smoke_eval_r2')
    evaluation=read(ROOT/'smoke_eval_r2/summary.json');assert evaluation['status']=='PASS_READ_ONLY_KEEP_DELETE_KEEP_ENGINEERING'
    assert all(row['target_delete_alpha_max']<1e-6 and row['keep_again_maxabs']==0 for row in evaluation['rows'])
    assert any(row['target_keep_alpha_max']>.01 for row in evaluation['rows'])
    run('smoke_restore',ROOT/'source/train_scene.py','--mode','resume2step','--checkpoint',smoke/f'chkpnt{step}.pth','--output',ROOT/'smoke_restore')
    restore=read(ROOT/'smoke_restore/restore_audit.json');entry=read(ROOT/'smoke_restore/resume_entry.json')
    assert restore['status']=='PASS_EXACT_FULL_RESTORE' and not restore['differences']
    assert entry['completed_extra_updates']==2
    gate=dict(status='PASS_GPU_SMOKE_SAVE_EVALUATE_EXACT_RESTORE_NEXT_ENTRY',iteration=step,engineering_updates=step+2,formal_updates=0,elapsed_seconds=done['elapsed_seconds'],actual_views=158,heldout=40,exact_restore=True)
    (ROOT/'smoke_gate.json').write_text(json.dumps(gate,indent=2))
    run('formal10k',ROOT/'source/train_scene.py','--mode','formal','--stop-iteration','10000','--output',ROOT/'formal10k')
    final=read(ROOT/'formal10k/returned.json');assert final['iteration']==10000
    run('formal10k_eval',ROOT/'source/evaluate_scene.py','--model',ROOT/'formal10k','--iteration','10000','--output',ROOT/'formal10k_eval','--all-build')
    status('NATIVE_10K_AND_BUILD_DELETE_EVAL_COMPLETE',formal_updates=10000,engineering_updates=step+2,shutdown=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('FAILED',error=repr(e));raise

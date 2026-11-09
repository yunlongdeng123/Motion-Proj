"""单任务有界队列：真机600秒生命周期smoke后fresh10K；禁止电源操作。"""
from pathlib import Path
import json,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1')
def read(p):return json.loads(p.read_text())
def status(stage,**kw):
    (PREP/'native_pipeline_status.json').write_text(json.dumps(dict(stage=stage,unix=time.time(),controller_pid=os.getpid(),automatic_shutdown=False,**kw),indent=2))
def run(name,*args):
    with (ROOT/(name+'.log')).open('x') as f:
        p=subprocess.Popen([sys.executable,'-B',*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
        status(name,pid=p.pid);code=p.wait()
    if code:raise RuntimeError(f'{name} failed code={code}')
def main():
    ROOT.mkdir(exist_ok=True)
    status('WAIT_PREPARED_INPUTS')
    deadline=time.monotonic()+2400
    while not (ROOT/'preparation_receipt.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('input preparation timeout')
        if 'Traceback' in (PREP/'gpu_inputs.log').read_text():raise RuntimeError('input preparation failed')
        time.sleep(3)
    for name in ['continue_native.py']:
        # 本文件已从PREP复制到新的模型source目录；不使用旧场景模型。
        assert (ROOT/'source'/name).exists()
    run('smoke_r1',ROOT/'source/train_scene.py','--mode','smoke','--smoke-seconds','600','--smoke-policy','long540','--output',ROOT/'smoke_r1')
    done=read(ROOT/'smoke_r1/returned.json');step=done['iteration'];assert done['elapsed_seconds']>=600
    for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']:assert (ROOT/'smoke_r1'/f'{prefix}{step}.pth').is_file()
    run('smoke_eval',ROOT/'source/evaluate_native.py','--model',ROOT/'smoke_r1','--iteration',step,'--output',ROOT/'smoke_eval','--selection',PREP/'selected_case.json')
    ev=read(ROOT/'smoke_eval/summary.json');assert ev['status']=='PASS_READ_ONLY_KEEP_DELETE_KEEP_ENGINEERING'
    assert all(x['target_delete_alpha_max']<1e-6 and x['keep_again_maxabs']==0 for x in ev['rows'])
    assert any(x['target_keep_alpha_max']>.01 for x in ev['rows'])
    run('smoke_v1',ROOT/'source/render_v1.py','--engine',ROOT/'engine','--model',ROOT/'smoke_r1','--iteration',step,'--output',ROOT/'smoke_v1','--raws','30','60','150','198','--compare-saved-dir',ROOT/'smoke_eval')
    run('smoke_restore',ROOT/'source/train_scene.py','--mode','resume2step','--checkpoint',ROOT/'smoke_r1'/f'chkpnt{step}.pth','--output',ROOT/'smoke_restore')
    restore=read(ROOT/'smoke_restore/restore_audit.json');entry=read(ROOT/'smoke_restore/resume_entry.json')
    assert restore['status']=='PASS_EXACT_FULL_RESTORE' and restore['differences']==[] and entry['completed_extra_updates']==2
    gate=dict(status='PASS_GPU_SMOKE_SAVE_EVALUATE_EXACT_RESTORE_NEXT_ENTRY',iteration=step,engineering_updates=step+2,formal_updates=0,elapsed_seconds=done['elapsed_seconds'],actual_views=159,heldout=40,exact_restore=True,original_v1_render_parity=True,automatic_shutdown=False)
    (ROOT/'smoke_gate.json').write_text(json.dumps(gate,indent=2))
    run('formal10k',ROOT/'source/train_scene.py','--mode','formal','--stop-iteration','10000','--output',ROOT/'formal10k')
    assert read(ROOT/'formal10k/returned.json')['iteration']==10000
    run('formal10k_eval',ROOT/'source/evaluate_native.py','--model',ROOT/'formal10k','--iteration',10000,'--output',ROOT/'formal10k_eval','--selection',PREP/'selected_case.json','--all-build')
    run('formal10k_v1',ROOT/'source/render_v1.py','--engine',ROOT/'engine','--model',ROOT/'formal10k','--output',ROOT/'formal10k_v1','--raws',*read(PREP/'inputs/receipt.json')['build_raws'],'--compare-saved-dir',ROOT/'formal10k_eval')
    status('NATIVE_10K_BUILD_EXPORTS_COMPLETE_PENDING_EDITED_SCENE',formal_updates=10000,engineering_updates=step+2)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('RECOVERABLE_FAILURE',error=repr(e));raise

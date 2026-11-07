"""Real smoke -> fresh5K -> immediate a5 preview -> exact5K-to10K; no shutdown."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(stage,**extra):write(ROOT/'queue_status.json',dict(stage=stage,controller_pid=os.getpid(),updated_unix=time.time(),shutdown=False,**extra))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def verify_inputs(label):
    budget=json.loads((ROOT/'active_budget.json').read_text())
    assert budget['single_new_formal_updates']==0 and budget['dual_fresh_stop']==10000
    assert budget['dual_checkpoint_eval_nodes']==[5000,10000] and budget['learning_rate_horizon']==30000
    receipt=json.loads((ROOT/'preparation_receipt.json').read_text())
    assert sha(ROOT/'config.json')==receipt['config_sha256'] and sha(ROOT/'identity_contract.json')==receipt['identity_contract_sha256']
    checks={p:dict(expected=row['sha256'],actual=sha(p)) for p,row in receipt['source_files_before'].items()}
    assert all(row['expected']==row['actual'] for row in checks.values()),checks
    write(ROOT/(label+'_input_readonly_audit.json'),dict(status='PASS_ACTIVE_BUDGET_IDENTITY_CONFIG_SOURCE_READ_ONLY',files=checks))
def run(label,*args):
    free=shutil.disk_usage(ROOT).free;assert free>8*1024**3
    with (ROOT/(label+'.log')).open('x') as log:
        process=subprocess.Popen([PY,*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,pid=process.pid,log=str(ROOT/(label+'.log')),free_disk_bytes=free)
        assert process.wait()==0,(label,process.returncode)
def a5(label,model,step):
    run(label,ROOT/'source/evaluate_a5.py','--model',model,'--iteration',step,'--output',ROOT/label,
        '--dynamic-aliases','6','37','--delete-name','delete_a6_a37')
    s=json.loads((ROOT/label/'summary.json').read_text())
    assert s['status']=='PASS_FRESH_NATIVE49_A5_EVALUATION' and s['iteration']==step and s['frame_count']==49
    assert s['fixed_denominator']['pixels']==121128 and s['optimization_updates']==0 and s['dynamic_aliases']==[6,37]
    assert s['replay']['complete_all49_keep_delete_replay'] and s['replay']['max_uint8_difference']<=1
    assert json.loads((ROOT/label/'source_integrity.json').read_text())['status']=='PASS_UNCHANGED'
    return s
def main():
    assert ROOT.name=='scene008_a5_old_dual30k_20261106_r1'
    verify_inputs('before_smoke')
    if (ROOT/'smoke_only_launch.json').is_file():
        pid=json.loads((ROOT/'smoke_only_launch.json').read_text())['pid']
        while not (ROOT/'smoke/returned.json').is_file():
            os.kill(pid,0);status('WAITING_ALREADY_ACTIVE_SMOKE_ONLY',pid=pid);time.sleep(2)
    else:
        run('smoke',ROOT/'source/train_dual.py','--mode','smoke','--output',ROOT/'smoke','--smoke-policy','adaptive500','--smoke-seconds','540')
    smoke=json.loads((ROOT/'smoke/returned.json').read_text());step=smoke['iteration']
    run('smoke_eval',ROOT/'source/evaluate_dual_layers.py','--model',ROOT/'smoke','--iteration',step,'--output',ROOT/'smoke_eval','--quick')
    run('smoke_resume2step',ROOT/'source/train_dual.py','--mode','resume2step','--checkpoint',ROOT/'smoke'/f'chkpnt{step}.pth','--output',ROOT/'smoke_resume2step')
    restored=json.loads((ROOT/'smoke_resume2step/restore_audit.json').read_text())
    entry=json.loads((ROOT/'smoke_resume2step/resume_entry.json').read_text())
    replay=json.loads((ROOT/'smoke_eval/summary.json').read_text())
    time_policy=json.loads((ROOT/'smoke/smoke_time_policy.json').read_text())
    assert (step>=500 or time_policy['long_smoke_required']) and (not time_policy['long_smoke_required'] or smoke['elapsed_seconds']>=540)
    assert restored['status']=='PASS_EXACT_FULL_RESTORE' and not restored['differences']
    assert entry['completed_extra_updates']==2 and replay['native_png_replay_max_uint8_difference']<=1
    a5('smoke_a5_evaluation',ROOT/'smoke',step)
    verify_inputs('after_smoke')
    write(ROOT/'smoke_summary.json',dict(status='PASS_REAL_GPU_SMOKE_SAVE_EVAL_FULL_RESTORE_A5_NEXT_ENTRY',
        smoke_iterations=step,resume_extra_updates=2,smoke_train_wall_seconds=smoke['elapsed_seconds'],
        max_allocated_bytes=smoke['max_allocated_bytes'],max_reserved_bytes=smoke['max_reserved_bytes'],
        formal_updates=0,formal_fresh=True,formal_total_budget=10000,learning_rate_horizon=30000,time_policy=time_policy,shutdown=False))
    run('formal5k',ROOT/'source/train_dual.py','--mode','formal','--stop-iteration','5000','--output',ROOT/'formal5k')
    first=json.loads((ROOT/'formal5k/returned.json').read_text())
    assert first['iteration']==5000 and first['source_iteration']==0 and first['actual_additional_iterations']==5000
    preview=a5('a5_evaluation_5k',ROOT/'formal5k',5000)
    anchors={str(raw):dict(full=str(ROOT/f'a5_evaluation_5k/anchors/{raw:03d}_full.png'),crop=str(ROOT/f'a5_evaluation_5k/anchors/{raw:03d}_crop.png')) for raw in [89,97,105,109]}
    write(ROOT/'preview_5k_ready.json',dict(status='READY_DUAL_FRESH_5K_ZERO_UPDATE_A5_PREVIEW',iteration=5000,
        summary=str(ROOT/'a5_evaluation_5k/summary.json'),anchors=anchors,all49=preview['all49'],
        fixed_denominator=121128,formal_total_budget=10000,next='exact full-state5000 ->10000',shutdown=False))
    status('FORMAL5K_A5_PREVIEW_READY_CONTINUING_EXACTLY_TO10K',preview=str(ROOT/'preview_5k_ready.json'))
    run('formal10k',ROOT/'source/train_dual.py','--mode','extend','--checkpoint',ROOT/'formal5k/chkpnt5000.pth','--stop-iteration','10000','--output',ROOT/'formal10k')
    last=json.loads((ROOT/'formal10k/returned.json').read_text())
    audit=json.loads((ROOT/'formal10k/restore_audit.json').read_text())
    assert last['iteration']==10000 and last['source_iteration']==5000 and last['actual_additional_iterations']==5000
    assert audit['status']=='PASS_EXACT_FULL_RESTORE' and not audit['differences']
    run('evaluation',ROOT/'source/evaluate_dual_layers.py','--model',ROOT/'formal10k','--iteration','10000','--output',ROOT/'evaluation')
    a5('a5_evaluation_10k',ROOT/'formal10k',10000)
    verify_inputs('after_formal_and_eval')
    status('COMPLETE_DUAL_FRESH5K_EXACT_RESUME10K_AND_49FRAME_EVALUATIONS',formal_total_iterations=10000,
        phases=[5000,10000],smoke_iterations=step,resume_extra_updates=2,quality_accepted=False,human_verdict=None)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('FAILED',error=repr(exc));raise

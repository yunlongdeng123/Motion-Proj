from pathlib import Path
import json,os,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
def write(p,v):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def status(stage,**kw):write(ROOT/'queue_status.json',dict(stage=stage,controller_pid=os.getpid(),unix=time.time(),shutdown=False,reset_card=False,**kw))
def run(label,*args):
    with (ROOT/(label+'.log')).open('x') as log:
        proc=subprocess.Popen([PY,'-B',*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,child_pid=proc.pid,log=str(ROOT/(label+'.log')))
        code=proc.wait()
        if code:raise RuntimeError((label,code))
def main():
    c=json.loads((ROOT/'longrun_contract.json').read_text())
    source=Path(c['source_checkpoint']);total=0
    for step in [15000,20000,30000]:
        model=ROOT/f'continuation{step//1000}k'
        run(f'train{step//1000}k',ROOT/'source/train_longrun.py','--mode','continue','--checkpoint',source,'--output',model,'--stop-iteration',step)
        returned=json.loads((model/'returned.json').read_text());audit=json.loads((model/'restore_audit.json').read_text())
        assert audit['status']=='PASS_EXACT_FULL_RESTORE' and audit['differences']==[]
        assert returned['iteration']==step and returned['source_iteration']==int(source.stem.replace('chkpnt',''))
        total+=returned['actual_additional_iterations']
        label=f'a5_evaluation_{step//1000}k'
        run(label,ROOT/'source/evaluate_a5.py','--model',model,'--iteration',step,'--output',ROOT/label,
            '--engine',ENGINE,'--dynamic-aliases','37','--delete-name','delete_a37')
        summary=json.loads((ROOT/label/'summary.json').read_text())
        assert summary['status']=='PASS_FRESH_NATIVE49_A5_EVALUATION' and summary['frame_count']==49
        assert summary['fixed_denominator']['pixels']==121128 and summary['optimization_updates']==0
        assert summary['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256'
        assert summary['replay']['complete_all49_keep_delete_replay'] and summary['replay']['max_uint8_difference']<=1
        write(ROOT/f'node{step}_ready.json',dict(status='PASS_FULL_NODE_AND_FRESH49',iteration=step,
            new_updates_cumulative=total,prior_smoke_updates_reused_without_recount=1977,summary=str(ROOT/label/'summary.json'),all49=summary['all49']))
        source=model/f'chkpnt{step}.pth'
    assert total==18023
    final={}
    for path,record in c['protected_source_files'].items():
        p=Path(path);now=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        assert now==record,path;final[path]=now
    write(ROOT/'source_integrity_after.json',dict(status='PASS_UNCHANGED_BYTES_MTIME_SHA256',files=final))
    write(ROOT/'continuation_accounting.json',dict(status='COMPLETE_NATIVE_SINGLE_30000',new_updates=total,
        source_iteration=11977,final_iteration=30000,prior_original_model_updates=10000,
        prior_engineering1977_reused=True,old_engineering_ledger6599_unchanged=True,
        replay_updates=0,extra_smoke_updates=0,evaluations_optimizer_updates=0,shutdown=False,reset_card=False))
    status('COMPLETE_NATIVE_SINGLE_30000',new_updates=total)
if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED',error=repr(error));raise

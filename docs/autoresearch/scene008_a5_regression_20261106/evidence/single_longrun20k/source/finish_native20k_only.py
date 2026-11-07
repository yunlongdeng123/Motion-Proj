"""Finish the already-running20K trainer and evaluate; never launches training."""
from pathlib import Path
import json,hashlib,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
def read(p):return json.loads(p.read_text())
def write(p,v):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def status(stage,**kw):write(ROOT/'queue_status.json',dict(stage=stage,controller_pid=os.getpid(),unix=time.time(),shutdown=False,reset_card=False,training_launches_allowed=False,**kw))
def run(label,*args):
    with (ROOT/(label+'.log')).open('x') as log:
        proc=subprocess.Popen([PY,'-B',*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,child_pid=proc.pid,log=str(ROOT/(label+'.log')))
        assert proc.wait()==0,(label,proc.returncode)
def main():
    scope=read(ROOT/'scope_change_stop20k.json')
    assert scope['controller_state_after_signal']=='EXITED' and scope['preserved_running_trainer_pid']==6416
    assert not (ROOT/'continuation30k').exists()
    write(ROOT/'active_scope.json',dict(status='USER_CHANGED_MAINLINE_NATIVE_BASELINE_STOPS_AT20K',
        final_iteration=20000,new_updates_cap=8023,thirty_k_executed=False,thirty_k_conclusion=None,
        training_from_this_controller=False,prior_contract_is_historical_plan=True,shutdown=False,reset_card=False))
    status('WAIT_EXISTING_TRAINER20K_BOUNDARY',existing_trainer_pid=6416)
    model=ROOT/'continuation20k'
    while not (model/'returned.json').exists():
        proc=Path('/proc/6416')
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':raise RuntimeError('20K trainer exited before a complete boundary')
        time.sleep(2)
    ret=read(model/'returned.json');audit=read(model/'restore_audit.json')
    assert ret['status']=='PASS_COMPLETE_BOUNDARY' and ret['iteration']==20000 and ret['actual_additional_iterations']==5000
    assert audit['status']=='PASS_EXACT_FULL_RESTORE' and audit['differences']==[]
    run('a5_evaluation_20k',ROOT/'source/evaluate_a5.py','--model',model,'--iteration','20000',
        '--output',ROOT/'a5_evaluation_20k','--engine',ENGINE,'--dynamic-aliases','37','--delete-name','delete_a37')
    summary=read(ROOT/'a5_evaluation_20k/summary.json')
    assert summary['status']=='PASS_FRESH_NATIVE49_A5_EVALUATION' and summary['frame_count']==49
    assert summary['fixed_denominator']['pixels']==121128 and summary['optimization_updates']==0
    assert summary['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256'
    assert summary['replay']['complete_all49_keep_delete_replay'] and summary['replay']['max_uint8_difference']<=1
    run('upper_regions_20k',ROOT/'source/aggregate_fixed_a5_regions.py','--evaluation-root',ROOT/'a5_evaluation_20k',
        '--output',ROOT/'a5_evaluation_20k/upper_regions.json')
    total=sum(read(ROOT/f'continuation{tag}k/returned.json')['actual_additional_iterations'] for tag in [15,20])
    assert total==8023
    write(ROOT/'node20000_ready.json',dict(status='PASS_FULL_NODE_AND_FRESH49',iteration=20000,new_updates_cumulative=total,
        prior_smoke_updates_reused_without_recount=1977,summary=str(ROOT/'a5_evaluation_20k/summary.json'),all49=summary['all49']))
    c=read(ROOT/'longrun_contract.json');final={}
    for path,record in c['protected_source_files'].items():
        p=Path(path);now=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        assert now==record,path;final[path]=now
    write(ROOT/'source_integrity_after.json',dict(status='PASS_UNCHANGED_BYTES_MTIME_SHA256',files=final))
    assert not (ROOT/'continuation30k').exists()
    write(ROOT/'continuation_accounting.json',dict(status='COMPLETE_NATIVE_SINGLE_20000_SCOPE_CAPPED',new_updates=total,
        source_iteration=11977,final_iteration=20000,prior_original_model_updates=10000,prior_engineering1977_reused=True,
        old_engineering_ledger6599_unchanged=True,new_engineering_updates=0,replay_updates=0,evaluations_optimizer_updates=0,
        thirty_k_executed=False,thirty_k_conclusion=None,shutdown=False,reset_card=False))
    status('COMPLETE_NATIVE_SINGLE_20000_SCOPE_CAPPED',new_updates=total,thirty_k_executed=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED20K_CLOSEOUT',error=repr(error));raise

"""No new single formal updates; evaluate10K, then dual smoke/fresh5K->10K."""
from pathlib import Path
import hashlib,json,math,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
DUAL=ROOT.parent/'scene008_a5_old_dual30k_20261106_r1'
FULL5=ROOT.parent/'r15_gn11_stage1_v41_20261023/arms/B/run30k'
PY=sys.executable
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(stage,**extra):write(ROOT/'short_queue_status.json',dict(stage=stage,controller_pid=os.getpid(),updated_unix=time.time(),shutdown=False,**extra))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def source_audit(label):
    contract=json.loads((ROOT/'continuation_contract.json').read_text());rows={}
    for name,row in contract['source_artifacts'].items():rows[name]=dict(expected=row['sha256'],actual=sha(row['path']))
    for name,expected in contract['source_code_sha256'].items():rows[name]=dict(expected=expected,actual=sha(name))
    assert all(r['expected']==r['actual'] for r in rows.values()),rows
    write(ROOT/(label+'_source_audit.json'),dict(status='PASS_SOURCE_CHECKPOINT_AND_ENGINE_UNCHANGED',files=rows))
def audit_gate(directory):
    audit=json.loads((directory/'independent_preflight_audit.json').read_text())
    assert str(audit['status']).startswith('PASS') and not audit['unresolved_mandatory_source_defects']
    assert audit['current_source_snapshot_stable_during_audit']
    for path,row in audit['source_files'].items():
        if 'sha256' in row:assert sha(path)==row['sha256'],('source changed since audit',path)
def run(label,*args):
    assert shutil.disk_usage(ROOT).free>8*1024**3
    with (ROOT/(label+'.log')).open('x') as log:
        process=subprocess.Popen([PY,*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,pid=process.pid,log=str(ROOT/(label+'.log')))
        assert process.wait()==0,(label,process.returncode)
def a5(label,model,step,aliases,delete_name):
    if label.startswith('reference_a5_single10k') and (ROOT/(label+'_launch.json')).exists():
        pid=json.loads((ROOT/(label+'_launch.json')).read_text())['pid']
        while True:
            if (ROOT/label/'failure.json').is_file():raise RuntimeError((ROOT/label/'failure.json').read_text())
            if (ROOT/label/'summary.json').is_file() and json.loads((ROOT/label/'summary.json').read_text()).get('source_integrity')=='PASS_UNCHANGED_BYTES_MTIME_SHA256':break
            os.kill(pid,0);status('WAITING_ACTIVE_SINGLE10K_REFERENCE',pid=pid);time.sleep(2)
    else:
        run(label,ROOT/'source/evaluate_a5.py','--model',model,'--iteration',step,'--output',ROOT/label,
            '--dynamic-aliases',*aliases,'--delete-name',delete_name)
    s=json.loads((ROOT/label/'summary.json').read_text())
    assert s['status']=='PASS_FRESH_NATIVE49_A5_EVALUATION' and s['frame_count']==49 and s['iteration']==step
    assert s['fixed_denominator']['pixels']==121128 and s['optimization_updates']==0 and s['dynamic_aliases']==aliases
    assert s['replay']['complete_all49_keep_delete_replay'] and s['replay']['max_uint8_difference']<=1
    assert json.loads((ROOT/label/'source_integrity.json').read_text())['status']=='PASS_UNCHANGED'
def main():
    budget=json.loads((ROOT/'active_budget.json').read_text())
    assert budget['single_new_formal_updates']==0 and budget['dual_fresh_stop']==10000
    assert sha(ROOT/'active_budget.json')==sha(DUAL/'active_budget.json')
    while not (ROOT/'smoke_summary.json').is_file():
        current=json.loads((ROOT/'smoke_queue_status.json').read_text());assert current['stage']!='FAILED',current
        status('WAITING_ACTIVE_SINGLE_SMOKE',smoke_stage=current['stage']);time.sleep(5)
    smoke=json.loads((ROOT/'smoke_summary.json').read_text());model=Path(smoke['smoke_model_path']);step=smoke['smoke_stop_iteration']
    assert smoke['smoke_train_wall_seconds']>=540 and smoke['resume_extra_updates']==2
    assert not json.loads((model/'restore_audit.json').read_text())['differences']
    for line in (model/'runtime.jsonl').read_text().splitlines():assert all(math.isfinite(v) for v in json.loads(line)['losses'].values())
    audit_gate(ROOT);audit_gate(DUAL);source_audit('before_short_campaign')
    if (DUAL/'smoke_only_launch.json').is_file():
        pid=json.loads((DUAL/'smoke_only_launch.json').read_text())['pid']
        while not (DUAL/'smoke/returned.json').is_file():
            os.kill(pid,0);status('WAITING_ACTIVE_DUAL_SMOKE_BEFORE_REFERENCE',pid=pid);time.sleep(2)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            status('WAITING_DUAL_SMOKE_GPU_CONTEXT_RELEASE',pid=pid);time.sleep(2)
    source=Path(json.loads((ROOT/'continuation_contract.json').read_text())['source_checkpoint'])
    a5('reference_a5_single10k_retry1',source.parent,10000,[37],'delete_a37')
    a5('smoke_a5_evaluation',model,step,[37],'delete_a37')
    # The existing Full5 30K checkpoint supplies KEEP-only historical context.
    a5('reference_a5_full5_B30k',FULL5,30000,[5,26,18,6,37],'delete_all5')
    required=[FULL5/f'{prefix}10000.pth' for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']]
    write(ROOT/'reference_a5_full5_B10k_unavailable.json',dict(status='NOT_AVAILABLE_NO_ADDED_TRAINING',missing=[str(p) for p in required if not p.is_file()],optimization_updates=0))
    write(ROOT/'short_campaign_release_receipt.json',dict(status='SINGLE_FORMAL_DISABLED_DUAL_ONLY_FRESH5K_TO_MAX10K',
        single_new_formal_updates=0,dual_formal_total_budget=10000,learning_rate_horizon=30000,smoke_extra_updates_excluded=True,
        active_budget=str(ROOT/'active_budget.json'),released_unix=time.time(),shutdown=False))
    run('dual_short_queue',DUAL/'source/run_dual_short_queue.py')
    assert json.loads((DUAL/'queue_status.json').read_text())['stage']=='COMPLETE_DUAL_FRESH5K_EXACT_RESUME10K_AND_49FRAME_EVALUATIONS'
    source_audit('after_short_campaign')
    status('COMPLETE_SINGLE10K_REFERENCE_DUAL_FRESH5K_TO10K',single_new_formal_updates=0,dual_fresh_formal_iterations=10000,quality_accepted=False,human_verdict=None)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('FAILED',error=repr(exc));raise

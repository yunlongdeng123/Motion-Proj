"""True GPU smoke, then independent eval/full restore; stop for parent review."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(stage,**extra):write(ROOT/'smoke_queue_status.json',dict(stage=stage,controller_pid=os.getpid(),updated_unix=time.time(),**extra))
def run(label,script,*args):
    with (ROOT/(label+'.log')).open('x') as log:
        process=subprocess.Popen([PY,str(ROOT/'source'/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,pid=process.pid,log=str(ROOT/(label+'.log')))
        assert process.wait()==0,(label,process.returncode)
def verify_source():
    contract=json.loads((ROOT/'continuation_contract.json').read_text());rows={}
    for name,expected in contract['source_artifacts'].items():
        h=hashlib.sha256()
        with Path(expected['path']).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        rows[name]=dict(sha256=h.hexdigest(),unchanged=h.hexdigest()==expected['sha256'])
    assert all(row['unchanged'] for row in rows.values()),rows
    write(ROOT/'source_unchanged_after_smoke.json',dict(status='PASS_SOURCE_READ_ONLY',artifacts=rows))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='smoke');opts=parser.parse_args()
    label=opts.label
    contract=json.loads((ROOT/'continuation_contract.json').read_text())
    source=Path(contract['source_checkpoint'])
    run(label,'train_single.py','--mode','smoke','--checkpoint',source,'--output',ROOT/label,'--smoke-seconds','540')
    smoke=json.loads((ROOT/label/'returned.json').read_text());step=smoke['iteration']
    run(label+'_eval','evaluate_layers.py','--model',ROOT/label,'--iteration',step,'--output',ROOT/(label+'_eval'),'--quick')
    run(label+'_resume2step','train_single.py','--mode','resume2step','--checkpoint',ROOT/label/f'chkpnt{step}.pth','--output',ROOT/(label+'_resume2step'))
    restore=json.loads((ROOT/label/'restore_audit.json').read_text())
    replay=json.loads((ROOT/(label+'_eval')/'summary.json').read_text())
    restored=json.loads((ROOT/(label+'_resume2step')/'restore_audit.json').read_text())
    entry=json.loads((ROOT/(label+'_resume2step')/'resume_entry.json').read_text())
    assert restore['status']==restored['status']=='PASS_EXACT_FULL_RESTORE'
    assert replay['status']=='PASS_NATIVE_LAYER_RELOAD' and replay['native_png_replay_max_uint8_difference']<=1
    assert entry['completed_extra_updates']==2
    verify_source()
    summary=dict(status='PASS_REAL_GPU_SMOKE_SAVE_EVAL_FULL_RESTORE_NEXT_ENTRY',
        source_iteration=10000,smoke_model_path=str(ROOT/label),smoke_stop_iteration=step,smoke_additional_iterations=step-10000,resume_extra_updates=2,
        smoke_train_wall_seconds=smoke['elapsed_seconds'],native_replay_difference=replay['native_png_replay_max_uint8_difference'],
        max_allocated_bytes=smoke['max_allocated_bytes'],max_reserved_bytes=smoke['max_reserved_bytes'],
        formal_additional_updates=0,formal_total_budget=30000,learning_rate_horizon=30000,
        formal_source_checkpoint=str(source),formal_source_is_smoke=False,shutdown=False)
    write(ROOT/'smoke_summary.json',summary)
    status('AWAITING_PARENT_FORMAL_RELEASE',summary=str(ROOT/'smoke_summary.json'),shutdown=False)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('FAILED',error=repr(exc));raise

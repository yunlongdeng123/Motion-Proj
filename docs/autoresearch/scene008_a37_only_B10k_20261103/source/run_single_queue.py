"""单目标身份重训：真实长 smoke -> 保存/评价/完整恢复 -> fresh10k ->49帧验收。"""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable

def status(stage,**extra):
    (ROOT/'queue_status.json').write_text(json.dumps(dict(stage=stage,updated_unix=time.time(),**extra),indent=2)+'\n')

def run(label,script,*args):
    status(label)
    with (ROOT/(label+'.log')).open('x') as f:
        subprocess.run([PY,str(ROOT/'source'/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--after-smoke',action='store_true');options=ap.parse_args()
    if not options.after_smoke:run('smoke','train_single.py','--mode','smoke','--output',ROOT/'smoke')
    smoke=json.loads((ROOT/'smoke/returned.json').read_text());step=smoke['iteration']
    if not options.after_smoke:run('smoke_resume','train_single.py','--mode','resume','--checkpoint',ROOT/'smoke'/f'chkpnt{step}.pth','--output',ROOT/'smoke_resume')
    label='smoke_next_entry_retry1' if options.after_smoke else 'smoke_next_entry'
    run(label,'evaluate_layers.py','--model',ROOT/'smoke','--iteration',step,'--output',ROOT/label,'--quick')
    restore=json.loads((ROOT/'smoke_resume/restore_audit.json').read_text())
    entry=json.loads((ROOT/'smoke_resume/resume_entry.json').read_text())
    assert restore['status']=='PASS_EXACT_FULL_RESTORE' and entry['completed_extra_updates']==2
    summary=dict(status='PASS_REAL_GPU_SMOKE_SAVE_EVAL_RESTORE_NEXT_ENTRY',smoke_updates=step,resume_extra_updates=2,
        smoke_train_wall_seconds=smoke['elapsed_seconds'],formal_updates=0,identity_dynamic_aliases=[37],restored_full_state=True,
        next_entry='fresh process checkpoint load; native dynamic/composite/static-only render',formal_lr_horizon=30000,formal_stop=10000)
    (ROOT/'smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    run('formal10k','train_single.py','--mode','formal','--output',ROOT/'formal10k')
    assert json.loads((ROOT/'formal10k/returned.json').read_text())['actual_updates']==10000
    run('evaluation','evaluate_layers.py','--model',ROOT/'formal10k','--iteration','10000','--output',ROOT/'evaluation')
    status('COMPLETE_TRAIN_AND_LAYER_EVALUATION',formal_updates=10000,smoke_updates=step,resume_extra_updates=2,
        human_verdict=None,quality_accepted=False,shutdown=False)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('FAILED',error=repr(exc));raise

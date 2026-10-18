"""Finite 4 x 300 update queue. Does not start any 30K or R15 training."""
from pathlib import Path
import json,subprocess,sys,datetime,traceback
OUT=Path(__file__).resolve().parent
def write(state,**kw):
    (OUT/'queue_status.json').write_text(json.dumps(dict(state=state,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
def main():
    if (OUT/'pilot_queue_started.json').exists():raise FileExistsError('no duplicate queue or blind restart')
    c=json.loads((OUT/'experiment_contract.json').read_text());assert c['maximum_total_pilot_updates']==1200
    assert json.loads((OUT/'channel_gate.json').read_text())['status']=='PASSED'
    assert json.loads((OUT/'prior_trace_receipt.json').read_text())['status']=='PASSED_NATIVE_ADAPTER_RGB'
    (OUT/'pilot_queue_started.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),maximum_updates=1200)))
    done=[]
    try:
        for v in c['variants']:
            name=v['name'];target=OUT/'pilots'/name
            if target.exists():raise FileExistsError('partial/existing arm requires explicit step and checkpoint audit: '+str(target))
            cmd=[sys.executable,str(OUT/'implementation/train.py'),'--spec',v['spec'],'--output',str(target),'--iterations','300','--seed','0','--joint-background',
                '--refine-poses-after','0','--pose-translation-limit-m','.5','--box-boundary-weight','1.0','--pose-prior-weight','.01','--max-features','10',
                '--checkpoint-steps','100','300','--probe-steps','100','300',
                '--raw-body-opacity-weight',str(v['raw_body_weight']),'--identity-opacity-weight',str(v['ID_weight']),'--run']
            write('PILOT_RUNNING',arm=name,completed=done,maximum_updates=1200,full30k_started=False)
            target.parent.mkdir(exist_ok=True)
            with (OUT/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            status=json.loads((target/'status.json').read_text());assert status['state']=='complete' and status['step']==300
            done.append(name)
        write('PILOTS_COMPLETE_AWAITING_LOSS_RESTORE_AND_QUALITY_REVIEW',completed=done,pilot_updates=1200,full30k_started=False)
    except BaseException:
        write('FAILED_CHECKPOINTS_PRESERVED',completed=done,error=traceback.format_exc(),full30k_started=False);raise
if __name__=='__main__':main()

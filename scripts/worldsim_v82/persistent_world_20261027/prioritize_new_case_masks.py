"""Only replace one zero-work waiting parent to prevent preprocessing starvation."""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
from native_scene_controller import ROOT,PY,same_process,ticks,write
from resource_control import require_start

def main():
    require_start();dest=ROOT/'new_case_mask_priority_v1';dest.mkdir(exist_ok=False)
    scene='4295449061847708198_3769_000_3789_000';run=ROOT/'native_scene_training_v1'/scene
    old=json.loads((run/'status.json').read_text());assert old['pid']==44539 and old['start_ticks']=='903397702'
    assert same_process(old['pid'],old['start_ticks']) and old['phase']=='waiting_previous_scene_diagnostics' and not old.get('child')
    children=Path('/proc/%s/task/%s/children'%(old['pid'],old['pid']));assert not children.read_text().strip()
    assert not list(run.glob('*.log')) and not any((run/n).exists() for n in ['smoke_fresh','smoke_restore','formal30k'])
    receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='first scene diagnostics, then two additional real mask/target packs, then next long native training; CPU expansion remains asynchronous',old=old,training_updates_lost=0)
    write(dest/'receipt.json',receipt);os.kill(old['pid'],signal.SIGTERM)
    for _ in range(100):
        if not same_process(old['pid'],old['start_ticks']):break
        time.sleep(.05)
    assert not same_process(old['pid'],old['start_ticks']);require_start()
    cmd=[PY,str(ROOT/'code/native_scene_controller.py'),'--context',scene,'--resume-waiting','--after-motion','14705303724557273004_3105_000_3125_000','--mask-first-count','2']
    with (dest/'controller.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['new']=dict(pid=p.pid,start_ticks=ticks(p.pid),command=cmd);write(dest/'receipt.json',receipt)
    time.sleep(1);state=json.loads((run/'status.json').read_text());assert state['pid']==p.pid and state['mask_first_count']==2 and state['phase']=='waiting_previous_scene_diagnostics'
    receipt['verified']=True;write(dest/'receipt.json',receipt);write(ROOT/'native_scene_mask_priority_dispatch.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

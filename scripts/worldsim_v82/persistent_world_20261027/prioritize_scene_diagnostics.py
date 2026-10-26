"""One-off queue correction; never signal an active training or CUDA child."""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
from native_scene_controller import ROOT,PY,same_process,ticks,write

SCENES=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']
EXPECTED=[(34105,'903032402'),(34781,'903048683')]

def children(pid):
    value=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
    return [int(x) for x in value.split()]

def main():
    receipt_dir=ROOT/'scene_diagnostic_priority_v1';receipt_dir.mkdir(exist_ok=False)
    receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='human: first completed scene must enter multiple-edit diagnosis before later training',training_updates_lost=0,stopped=[],restarted=[],healthy_first_training_untouched=True)
    first=json.loads((ROOT/'native_scene_training_v1'/SCENES[0]/'status.json').read_text())
    assert first['phase']=='formal30k' and same_process(first['pid'],first['start_ticks']) and same_process(**first['child'])
    waiting=[]
    for scene,(pid,start) in zip(SCENES[1:],EXPECTED):
        run=ROOT/'native_scene_training_v1'/scene;state=json.loads((run/'status.json').read_text())
        assert (state['pid'],state['start_ticks'])==(pid,start) and same_process(pid,start)
        assert state['phase']=='waiting_gpu' and not state.get('child') and not children(pid)
        assert not any((run/x).exists() for x in ['smoke_fresh','smoke_restore','formal30k']) and not list(run.glob('*.log'))
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
        assert 'native_scene_controller.py' in cmd and scene in cmd
        waiting.append((scene,pid,start,state,cmd))
    gpu_query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    gpu_pids=[int(x.strip()) for x in gpu_query.splitlines() if x.strip().isdigit()]
    assert first['child']['pid'] in gpu_pids and all(pid not in gpu_pids for _,pid,_,_,_ in waiting)
    write(receipt_dir/'receipt.json',receipt)
    for scene,pid,start,state,cmd in waiting:
        assert same_process(pid,start) and not children(pid)
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not same_process(pid,start):break
            time.sleep(.05)
        assert not same_process(pid,start)
        receipt['stopped'].append(dict(context=scene,pid=pid,start_ticks=start,state=state,command=cmd,completed_updates=0));write(receipt_dir/'receipt.json',receipt)
    for i,scene in enumerate(SCENES[1:],1):
        cmd=[PY,str(ROOT/'code/native_scene_controller.py'),'--context',scene,'--resume-waiting','--after-motion',SCENES[i-1]]
        with (receipt_dir/f'controller_{i}.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=dict(pid=child.pid,start_ticks=ticks(child.pid),context=scene,after_motion=SCENES[i-1],command=cmd)
        receipt['restarted'].append(identity);write(receipt_dir/'receipt.json',receipt)
    time.sleep(2)
    for item in receipt['restarted']:
        state=json.loads((ROOT/'native_scene_training_v1'/item['context']/'status.json').read_text())
        assert state['pid']==item['pid'] and state['phase']=='waiting_previous_scene_diagnostics' and same_process(item['pid'],item['start_ticks']) and not state.get('child')
    assert same_process(first['pid'],first['start_ticks']) and same_process(**first['child'])
    receipt['verified']=True;write(receipt_dir/'receipt.json',receipt)
    write(ROOT/'native_scene_priority_dispatch.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

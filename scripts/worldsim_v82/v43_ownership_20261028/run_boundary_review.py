"""One additional matched edge-leakage control; no method/scene grid search."""
import json,os,time,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');D=R/'first_gpu_review_v1';PY='/root/autodl-tmp/envs/drivingeditor/bin/python';peak=0
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def main():
    global peak
    state=D/'boundary_controller.json';assert not state.exists();records=[]
    with (D/'boundary_NVML_1s.jsonl').open('a') as samples:
        for stage,steps in [('smoke',100),('restore',110),('formal',10000)]:
            out=D/'opacity/boundary'/stage;cmd=[PY,'-u',str(R/'code/train_opacity_review.py'),'--arm','boundary','--output',str(out),'--steps',str(steps),'--save-every',str(5000 if stage=='formal' else 100)]
            if stage=='restore':cmd+=['--resume',str(D/'opacity/boundary/smoke/checkpoint000100.pth')]
            with (D/('boundary_'+stage+'.log')).open('w') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);write(state,dict(phase='running',stage=stage,pid=os.getpid(),child_pid=child.pid,completed=records))
                while child.poll() is None:
                    used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','));peak=max(peak,used);samples.write(json.dumps(dict(time=time.time(),used_mib=used,utilization=util))+'\n');samples.flush();time.sleep(1)
                assert child.returncode==0,(stage,child.returncode)
            s=json.loads((out/'status.json').read_text());assert s['phase']=='complete'
            if stage=='restore':assert json.loads((out/'restore_verification.json').read_text())['all_passed']
            records.append(dict(stage=stage,status=s))
        write(state,dict(phase='complete',pid=os.getpid(),records=records,peak_NVML_mib=peak,extra_smoke_updates=110))
if __name__=='__main__':main()

"""One global GPU lease, at most two verified children, measured admission.

Complete real save/evaluate/restore smoke before each short formal10K arm.
Extra smoke100+restore10 is separate, not part of the formal update budget.
"""
import os,time,json,secrets,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/code')
from gpu_slot import exclusive
from v43_resource_control import require_start
from segmentation import ROOT,write
CTX='14705303724557273004_3105_000_3125_000';R=ROOT/'first_gpu_review_v1';PY='/root/autodl-tmp/envs/drivingeditor/bin/python';CODE=ROOT/'code'

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def main():
    require_start(CTX);assert not (R/'opacity_controller.json').exists();running={};done={};failure=None;peak=0
    plans=[(arm,stage) for stage in ['smoke','restore','formal'] for arm in ['legacy','balanced','field']]
    lease=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],token=secrets.token_hex(16),admitted_pids=[])
    def snapshot():write(R/'opacity_controller.json',dict(pid=os.getpid(),start_ticks=lease['start_ticks'],running={str(k):v['key'] for k,v in running.items()},completed=done,phase='failed' if failure else 'running',failure=failure,peak_NVML_mib=peak,concurrency_limit=2,extra_updates_per_arm=110))
    with exclusive():
        write(R/'lease.json',lease);log=(R/'opacity_NVML.jsonl').open('a')
        while plans or running:
            require_start(CTX)
            q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();total,used,util=[int(s) for s in q.split(',')];peak=max(peak,used);log.write(json.dumps(dict(time=time.time(),used_mib=used,utilization=util,owned_children=list(running)))+'\n');log.flush()
            for pid,v in list(running.items()):
                code=v['p'].poll()
                if code is None:continue
                v['log'].close();s=read(v['dest']/'status.json');key=v['key']
                if code or s.get('phase')!='complete':failure=dict(key=key,returncode=code,status=s);break
                if key.endswith('/restore'):assert read(v['dest']/'restore_verification.json').get('all_passed')
                done[key]=s;del running[pid]
            if failure:break
            # Smoke must cover every new path; formal never starts before its restore.
            ready=[]
            for arm,stage in plans:
                if stage=='restore' and arm+'/smoke' not in done:continue
                if stage=='formal' and arm+'/restore' not in done:continue
                ready.append((arm,stage))
            # Reserve at least6GiB per new worker, 4GiB safety; increase if observed.
            profiles=[s.get('cuda_peak_reserved',0)/(1024**2) for s in done.values()]+[read(v['dest']/'status.json').get('cuda_current_reserved',0)/(1024**2) for v in running.values()]
            reserve=max([6144]+[int(x*1.4+1024) for x in profiles])
            if ready and len(running)<2 and used+reserve+4096<total:
                arm,stage=ready[0];dest=R/'opacity'/arm/stage;assert not dest.exists();steps=100 if stage=='smoke' else 110 if stage=='restore' else 10000
                cmd=[PY,'-u',str(CODE/'train_opacity_review.py'),'--arm',arm,'--output',str(dest),'--steps',str(steps),'--save-every',str(100 if stage!='formal' else 5000),'--managed-round',str(R)]
                if stage=='restore':cmd+=['--resume',str(R/'opacity'/arm/'smoke/checkpoint000100.pth')]
                f=(R/(arm+'_'+stage+'.log')).open('w');env=dict(os.environ,V43_REVIEW_LEASE=lease['token']);child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                running[child.pid]=dict(p=child,key=arm+'/'+stage,dest=dest,log=f);lease['admitted_pids'].append(child.pid);write(R/'lease.json',lease);plans.remove((arm,stage));print('START',arm,stage,child.pid,flush=True)
            snapshot();time.sleep(2)
        if failure:
            # Stop only children started by this controller; preserve all files.
            for v in running.values():
                if v['p'].poll() is None:v['p'].terminate()
            snapshot();raise RuntimeError(json.dumps(failure))
        log.close();snapshot();s=read(R/'opacity_controller.json');s['phase']='complete';write(R/'opacity_controller.json',s)

if __name__=='__main__':main()

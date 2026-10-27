"""Two admitted arms, one global GPU lease. Never resume an unrelated queue."""
import os,time,json,secrets,subprocess,sys,argparse
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/code')
from gpu_slot import exclusive
from v43_resource_control import require_start
from segmentation import ROOT,write
CTX='14705303724557273004_3105_000_3125_000';R=ROOT/'target_unknown_v2';PY='/root/autodl-tmp/envs/drivingeditor/bin/python'

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--tag',default='pilot49');p.add_argument('--memory',default='clean_memory');p.add_argument('--protection',default='global',choices=['global','regions']);p.add_argument('--arms',nargs='+',choices=['joint','frozen_split'],default=['joint','frozen_split']);p.add_argument('--time-conditioned',action='store_true');p.add_argument('--video-teacher',action='store_true');a=p.parse_args();D=R/a.tag;D.mkdir(exist_ok=False);peak=0;running={};done={};plans=[(arm,stage) for stage in ['smoke','restore','formal'] for arm in a.arms]
    lease=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],token=secrets.token_hex(16),admitted_pids=[])
    with exclusive(),(D/'NVML.jsonl').open('a') as samples:
        write(D/'lease.json',lease)
        while plans or running:
            require_start(CTX);total,used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','));peak=max(peak,used);samples.write(json.dumps(dict(time=time.time(),used_mib=used,utilization=util))+'\n');samples.flush()
            for pid,v in list(running.items()):
                code=v['process'].poll()
                if code is None:continue
                v['log'].close();s=read(v['dest']/'status.json')
                if code or s.get('phase')!='complete':
                    for x in running.values():
                        if x['process'].poll() is None:x['process'].terminate()
                    write(D/'controller.json',dict(phase='failed',key=v['key'],exit_code=code,status=s));raise RuntimeError(v['key'])
                if v['key'].endswith('/restore'):assert read(v['dest']/'restore_verification.json').get('model_optimizer_exact')
                done[v['key']]=s;del running[pid]
            ready=[(arm,stage) for arm,stage in plans if stage=='smoke' or arm+'/'+('smoke' if stage=='restore' else 'restore') in done]
            if ready and len(running)<2 and total-used>8192:
                arm,stage=ready[0];dest=D/arm/stage;cmd=[PY,'-u',str(ROOT/'code/train_target_unknown.py'),'--arm',arm,'--output',str(dest),'--steps',str(100 if stage=='smoke' else 110 if stage=='restore' else a.steps),'--managed-round',str(D),'--memory',a.memory,'--protection',a.protection]
                if stage=='restore':cmd+=['--resume',str(D/arm/'smoke/checkpoint000100.pth')]
                if a.time_conditioned:cmd+=['--time-conditioned']
                if a.video_teacher:cmd+=['--video-teacher']
                f=(D/(arm+'_'+stage+'.log')).open('w');child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,V43_REVIEW_LEASE=lease['token']));running[child.pid]=dict(process=child,log=f,key=arm+'/'+stage,dest=dest);lease['admitted_pids'].append(child.pid);write(D/'lease.json',lease);plans.remove((arm,stage))
            write(D/'controller.json',dict(phase='running',pid=os.getpid(),running={str(k):v['key'] for k,v in running.items()},completed=done,NVML_peak_mib=peak,formal_updates_per_arm=a.steps,extra_smoke_updates_per_arm=110));time.sleep(1)
        s=read(D/'controller.json');s.update(phase='complete',completed=done,running={},NVML_peak_mib=peak);write(D/'controller.json',s)

if __name__=='__main__':main()

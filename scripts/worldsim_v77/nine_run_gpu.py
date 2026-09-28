"""本次有界GPU作业顺序执行，无定时任务、自动重试或后台新实验。"""
from nine_common import *
import os,subprocess,time,fcntl,traceback
lock=open(ROOT/'gpu_sequence.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert read(ROOT/'mask_state.json')['state']=='complete'
assert not (ROOT/'sequence_state.json').exists()
reg=read(ROOT/'registration.json');state=dict(state='running',pid=os.getpid(),completed=[],human_verdict=None);dump(ROOT/'sequence_state.json',state)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1')
jobs=[]
for s in reg['scenes']:
    if (ROOT/s['name']/'asset/source_rgba.png').exists():
        for stage in ['shape','paint']:jobs.append((s['name']+'_'+stage,'/root/autodl-tmp/envs/worldsim-v77-hunyuan/bin/python',['nine_asset_worker.py','--scene',s['name'],'--stage',stage]))
jobs += [('drive','/root/autodl-tmp/envs/driveeditor/bin/python',['nine_drive.py']),('omega','/root/autodl-tmp/envs/worldsim-v77/bin/python',['nine_omega.py'])]
try:
    for name,python,args in jobs:
        state['current']=name;dump(ROOT/'sequence_state.json',state);start=time.time()
        with (ROOT/f'{name}.log').open('w') as out:
            p=subprocess.run([python,*args],cwd='/root/autodl-tmp/work',env=env,stdout=out,stderr=subprocess.STDOUT,timeout=10800)
        row=dict(job=name,seconds=time.time()-start,returncode=p.returncode);state['completed'].append(row);dump(ROOT/'sequence_state.json',state)
        if p.returncode:raise RuntimeError(row)
        print('JOB_DONE',row,flush=True)
    state.update(state='complete',current=None);dump(ROOT/'sequence_state.json',state)
except Exception as ex:
    state.update(state='failed_engineering',error=repr(ex),trace=traceback.format_exc());dump(ROOT/'sequence_state.json',state);raise

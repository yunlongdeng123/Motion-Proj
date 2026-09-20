"""有限10项单卡队列；锁防重复，任一失败/OOM立即停止，不改设置重试。"""
import fcntl
import json
import subprocess
import sys
import time
from select_target import P1

def main():
    lock=(P1/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=P1/'queue.json'
    if state_path.exists():
        raise FileExistsError('已有队列记录；先核实终态，不自动重新启动')
    protocol=json.loads((P1/'protocol.json').read_text())
    jobs=[(s,c) for s in protocol['seed_values'] for c in protocol['cases']]
    state={'status':'running','planned':len(jobs),'completed':[],'current':None}
    for seed,case in jobs:
        stem=f'{case}-seed{seed}'
        state['current']=stem
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        with (P1/f'{stem}.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(__import__('pathlib').Path(__file__).with_name('run_localization.py')),
                                      '--case',case,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            with (P1/f'{stem}.gpu.csv').open('w') as gpu:
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.total,utilization.gpu',
                                          '--format=csv','-l','1'],stdout=gpu,stderr=subprocess.STDOUT)
                try:
                    code=process.wait()
                finally:
                    monitor.terminate()
                    monitor.wait()
        (P1/f'{stem}.exit_code').write_text(str(code)+'\n')
        if code!=0:
            state.update(status='stopped_on_failure',exit_code=code)
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
            return code
        state['completed'].append(stem)
        print(json.dumps({'completed':len(state['completed']),'planned':len(jobs),'last':stem}),flush=True)
    state.update(status='complete',current=None)
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    return 0

if __name__=='__main__':
    sys.exit(main())

"""跟随已确认运行的有限队列，在CPU逐项核验，不启动或重启生成。"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from select_target import P1

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--producer-pid',type=int,required=True)
    a=p.parse_args()
    protocol=json.loads((P1/'protocol.json').read_text())
    calibration=json.loads((P1/'evaluator_calibration.json').read_text())
    scripts=Path(__file__).parent
    completed=[]
    pairs=[]
    for seed in protocol['seed_values']:
        for case in protocol['cases']:
            stem=f'{case}-seed{seed}'
            result_path=P1/'rollouts'/f'{stem}.json'
            while not result_path.exists():
                os.kill(a.producer_pid,0)  # 生产进程消失即退出，不凭状态文件猜测或重跑。
                queue=json.loads((P1/'queue.json').read_text())
                assert queue['status']=='running',queue
                time.sleep(10)
            result=json.loads(result_path.read_text())
            assert result['status']=='complete',result['status']
            if case!='clean' and not (P1/'rollouts'/f'{stem}.response.json').exists():
                subprocess.run([sys.executable,str(scripts/'analyze_localization.py'),'response','--case',case,'--seed',str(seed)],check=True)
            if calibration['status']=='passed':
                if not (P1/'rollouts'/f'{stem}.detections.json').exists():
                    subprocess.run([sys.executable,str(scripts/'evaluate_localization.py'),'evaluate','--case',case,'--seed',str(seed)],check=True)
            completed.append(stem)
            (P1/'postprocess.json').write_text(json.dumps({'status':'running','completed':completed},indent=2)+'\n')
        for sign in ['negative','positive']:
            x=np.load(P1/'rollouts'/f'{sign}_persistent-seed{seed}.npy',mmap_mode='r')
            y=np.load(P1/'rollouts'/f'{sign}_restore-seed{seed}.npy',mmap_mode='r')
            same=np.array_equal(x[:37],y[:37])
            assert same,'同号干预的恢复前前缀不一致'
            pairs.append({'seed':seed,'sign':sign,'frames':[0,36],'raw_exactly_equal':bool(same)})
        subprocess.run([sys.executable,str(scripts/'preview_localization.py'),'--seed',str(seed)],check=True)
    (P1/'shared_prefix_audit.json').write_text(json.dumps({'status':'passed','comparisons':pairs},indent=2)+'\n')
    (P1/'postprocess.json').write_text(json.dumps({'status':'complete','completed':completed},indent=2)+'\n')

if __name__=='__main__':
    main()

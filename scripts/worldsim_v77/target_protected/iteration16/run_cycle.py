"""手动GPU入口；零训练和一次64步间保留图像review节点。"""
from common import *
import argparse, fcntl, subprocess, traceback


def main(action):
    import torch
    if not torch.cuda.is_available():raise SystemExit('等待用户开启GPU；没有后台等卡或CPU模型推理')
    assert read(O/'preflight.json')['CPU_ready']
    handle=open(O/'cycle.lock','a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(O/'controller_state.json');done=list(old.get('phases_complete',[]))
    phases=['zero'] if action=='zero' else ['train64','evaluate64']
    if action=='adapt64':
        assert 'zero' in done
        assert read(O/'zero_shot_gate.json')['decision']=='allow_one_64_step_adaptation'
    for phase in phases:
        if phase in done:continue
        dump(O/'controller_state.json',{'stage':'running','phase':phase,'phases_complete':done,'GPU_jobs':1,'max_training_steps':64})
        with (O/f'{phase}.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(Path(__file__).parent/'gpu_experiment.py'),phase],stdout=log,stderr=subprocess.STDOUT,check=True)
        done.append(phase)
    dump(O/'controller_state.json',{'stage':'zero_complete_pending_image_review' if action=='zero' else 'complete_pending_image_review',
        'phases_complete':done,'GPU_jobs':0,'training_steps':0 if action=='zero' else 64,
        'new_inference_windows':9 if action=='zero' else 20,'human_verdict':None,'no_auto_extra_training':True})
    subprocess.run([sys.executable,str(Path(__file__).parent/'review.py')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['zero','adapt64']);args=p.parse_args()
    try:main(args.action)
    except Exception as e:
        previous=read(O/'controller_state.json') if (O/'controller_state.json').exists() else {}
        dump(O/'controller_state.json',previous|{'stage':'stopped_on_error','GPU_jobs':0,'error':str(e),
            'traceback':traceback.format_exc(),'no_auto_extra_training':True})
        raise

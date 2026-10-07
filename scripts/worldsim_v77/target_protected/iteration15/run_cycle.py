"""用户开GPU后才手动运行；单控制器、两个固定检查点，不追加搜索。"""
from common import *
import fcntl, subprocess, time, traceback


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('等待用户开GPU：本入口不会在CPU加载模型或后台等卡')
    assert read(O/'preflight.json')['CPU_ready']
    handle=open(O/'cycle.lock','a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).parent/'gpu_experiment.py'
    phases=[('diagnose',[str(script),'diagnose']),
        ('train64',[str(script),'train','--step','64']),
        ('eval64',[str(script),'evaluate','--step','64']),
        ('train128',[str(script),'train','--step','128']),
        ('eval128',[str(script),'evaluate','--step','128']),
        ('report',[str(Path(__file__).parent/'review.py')])]
    old=read(O/'controller_state.json');done=list(old.get('phases_complete',[]))
    for name,args in phases:
        if name in done:continue
        dump(O/'controller_state.json',{'stage':'running','phase':name,'phases_complete':done,
            'GPU_jobs':1,'max_training_steps':128,'time':time.time()})
        # 子进程顺序结束后释放模型，再进入下一节点，不并发占GPU。
        with (O/f'{name}.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        done.append(name)
        dump(O/'controller_state.json',{'stage':'running','phases_complete':done,'GPU_jobs':0})
    dump(O/'controller_state.json',{'stage':'complete_pending_human_review','phases_complete':done,
        'GPU_jobs':0,'training_steps':128,'human_verdict':None,
        'stop':'本轮结束；不自动增加训练步数或开下一轮'})


if __name__=='__main__':
    try:main()
    except Exception as e:
        previous=read(O/'controller_state.json') if (O/'controller_state.json').exists() else {}
        dump(O/'controller_state.json',previous | {'stage':'stopped_on_error','GPU_jobs':0,
            'error':str(e),'traceback':traceback.format_exc(),'no_auto_extra_training':True})
        raise

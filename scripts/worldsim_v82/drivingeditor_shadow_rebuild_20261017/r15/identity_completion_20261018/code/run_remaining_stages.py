"""有限顺序队列：旧3渲染、旧5实例30K、旧5渲染。之后等待保存交付。"""
from identity_common import *
import datetime,os,subprocess,sys,time

def main():
    target=TASK/'queue_status.json'
    if target.exists():raise FileExistsError('不重复启动队列')
    stages=[('old3_render',[sys.executable,str(TASK/'code/render_identity_matrix.py'),'--model','old3']),
            ('owner5_train30k',[sys.executable,str(TASK/'code/train_owner5.py'),'--run']),
            ('old5_render',[sys.executable,str(TASK/'code/render_identity_matrix.py'),'--model','old5'])]
    policy=dict(task_id=TASK_ID,authorized_by='user requested complete old a6/a37 ownership and R15 a5/a26/a18 isolated/delete; GPU opened',
        old_five_instance_max_steps=30000,GN11_training_updates=0,R15_training_updates=0,
        no_new_R16_R17=True,no_subagents=True,shutdown_after_delivery_authorized=True)
    write(TASK/'execution_policy.json',policy)
    done=[];start=time.monotonic()
    try:
        for name,cmd in stages:
            write(target,dict(state='RUNNING',phase=name,controller_pid=os.getpid(),completed=done,
                command=cmd,started_singapore=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
            with (TASK/(name+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                write(TASK/'active_child.json',dict(stage=name,pid=child.pid,command=cmd))
                code=child.wait()
            if code:raise RuntimeError(name+' exited '+str(code))
            if name=='owner5_train30k':
                status=read(TASK/'train5/status.json')
                if status.get('state')!='complete' or status.get('step')!=30000:
                    raise RuntimeError('训练未完成30K，不冒充完整基线：'+str(status))
            done.append(name)
        write(target,dict(state='ALL_GPU_WORK_COMPLETE_AWAITING_DELIVERY',completed=done,
            controller_pid=os.getpid(),seconds=time.monotonic()-start,shutdown_executed=False))
    except BaseException as error:
        write(target,dict(state='NEEDS_ATTENTION_ASSETS_PRESERVED',completed=done,
            error=repr(error),controller_pid=os.getpid(),seconds=time.monotonic()-start));raise

if __name__=='__main__':main()

"""本轮有限控制器：等待现有GPU任务退出，串行SAM与CPU状态构建。"""
from pathlib import Path
import json,time,subprocess,os,fcntl
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected/iteration12')
O=T/'r22'
def main():
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'stage':'waiting_existing_input_control','pid':os.getpid()}
    def save():
        p=O/'controller_state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
    save();start=time.monotonic()
    while True:
        old=json.loads((T/'r21/baseline_state.json').read_text())
        if old['stage']=='complete_pending_review':break
        if time.monotonic()-start>1200:raise RuntimeError('既有对照超20分钟未完成，停止而不抢GPU')
        try:os.kill(old['pid'],0)
        except ProcessLookupError:raise RuntimeError('既有任务退出但未完成，先修复')
        time.sleep(5)
    # 状态落盘早于进程完全退出，等CUDA资源真正释放。
    for _ in range(30):
        try:os.kill(old['pid'],0)
        except ProcessLookupError:break
        time.sleep(1)
    else:raise RuntimeError('旧控制进程仍存活，不启动第二GPU任务')
    for name,python,script in [('observed_segmentation','worldsim-v77-sam2','segment_observed.py'),('CPU_projected_state','motionproj','build_state_probe.py')]:
        state['stage']=name;save()
        with (O/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([f'/root/autodl-tmp/envs/{python}/bin/python',str(S/script)],stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=proc.pid;save();rc=proc.wait(timeout=1800)
        if rc:state.update(stage='failed',failed_step=name,returncode=rc);save();raise RuntimeError(name)
    state.update(stage='complete_pending_quality',child_pid=None);save()
if __name__=='__main__':main()

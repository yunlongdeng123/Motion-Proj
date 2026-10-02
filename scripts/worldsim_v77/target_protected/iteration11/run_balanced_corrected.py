"""唯一控制器：训练检查完成才进入同输入推理，不执行电源操作。"""
from pathlib import Path
import os,sys,subprocess,fcntl,time,json
R=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r18');P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected/iteration11')
def dump(d):
 p=R/'controller_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 lock=open(R/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state={'pid':os.getpid(),'run':'r18','automatic_shutdown':False}
 for stage,script in [('training','balanced_corrected.py'),('evaluation','evaluate_balanced_corrected.py')]:
  with (R/(stage+'.log')).open('a') as log:
   child=subprocess.Popen(['/root/autodl-tmp/envs/driveeditor/bin/python',str(P/script)],stdout=log,stderr=subprocess.STDOUT)
   state.update(stage=stage,child_pid=child.pid,updated_unix=time.time());dump(state);rc=child.wait();state.update(returncode=rc,updated_unix=time.time());dump(state)
   if rc:state.update(stage='failed',failed_phase=stage);dump(state);raise RuntimeError(stage+' failed; preserve and diagnose, never shutdown')
 state.update(stage='complete_pending_review',child_pid=None);dump(state)
if __name__=='__main__':main()

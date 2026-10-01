"""有限控制器：同数量时序模块训练、同输入评测；不自动关机。"""
from pathlib import Path
import sys,os,fcntl,subprocess,time,traceback
sys.path.insert(0,str(Path(__file__).parent))
from process_sources import T,read,dump
O=T/'r14';P=Path(__file__).parent

def main():
 lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(T/'r13/evaluation/state.json')['all_cases_complete']
 state={'pid':os.getpid(),'stage':'starting','automatic_shutdown':False,'started_epoch':time.time()};dump(O/'controller_state.json',state)
 try:
  for stage,script in [('training','temporal_scope.py'),('evaluation','evaluate_scope.py')]:
   state.update(stage=stage);dump(O/'controller_state.json',state)
   with (O/f'{stage}.log').open('a') as log:
    job=subprocess.Popen([sys.executable,str(P/script)],stdout=log,stderr=subprocess.STDOUT,cwd=str(P))
    state.update(child_pid=job.pid);dump(O/'controller_state.json',state)
    code=job.wait()
   if code:raise RuntimeError(f'{stage} exit {code}; no automatic relaunch')
  assert read(O/'training/scope_control.json')['same_160_case_window_order_verified']
  assert read(O/'evaluation/state.json')['all_five_arms_complete']
  state.update(stage='complete_pending_review',finished_epoch=time.time(),child_pid=None);dump(O/'controller_state.json',state)
 except Exception:
  state.update(stage='failed',traceback=traceback.format_exc(),finished_epoch=time.time());dump(O/'controller_state.json',state);raise

if __name__=='__main__':main()

"""有限单轮控制器；锁防重复、失败停止，不定时重启/关机/追加训练。"""
from pathlib import Path
import subprocess,os,sys,fcntl,traceback
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,read,dump
O=T/'r10';P=Path(__file__).parent;PY='/root/autodl-tmp/envs/driveeditor/bin/python';CPU='/root/autodl-tmp/envs/motionproj/bin/python'
def main():
 lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=read(O/'controller_state.json') if (O/'controller_state.json').exists() else {}
 if old.get('stage')=='complete':return
 state={'stage':'starting','pid':os.getpid(),'finite_stages':['training160','frozen_four_arm_evaluation','summary','review_html'],'no_followup_training':True,'power_operation':None,'human_verdict':None};dump(O/'controller_state.json',state)
 stages=[('training',PY,'train_temporal.py'),('evaluation',PY,'evaluate_temporal.py'),('summary',CPU,'summarize_temporal.py'),('review_html',CPU,'build_temporal_review.py')]
 try:
  for stage,python,name in stages:
   state.update(stage=stage);dump(O/'controller_state.json',state)
   with (O/(stage+'.log')).open('a') as f:subprocess.run([python,str(P/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
  state.update(stage='complete');dump(O/'controller_state.json',state)
 except Exception as e:state.update(stage='failed',error=repr(e),traceback=traceback.format_exc());dump(O/'controller_state.json',state);raise
if __name__=='__main__':main()

from pathlib import Path
import os,json,subprocess,datetime,traceback,sys
E=Path(__file__).resolve().parent
def state(s,**kw):
 p=E/'A_recovery_queue_status.json';t=p.with_suffix('.partial');t.write_text(json.dumps(dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));t.replace(p)
def main():
 with (E/'A_recovery_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),resume_checkpoint=20000,logical_GN11_target=30000,authorized_extra_replay=5846,instance_budget=30000,checkpoint_interval=5000),f)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_HOME='/root/autodl-tmp/drivingeditor_repro/cache/torch')
 env['PATH']='/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH','');env['LD_LIBRARY_PATH']='/usr/local/cuda-11.8/lib64:/root/autodl-tmp/envs/drivingeditor/lib:'+env.get('LD_LIBRARY_PATH','')
 for name,script in [('GN11_RESUME20K_TO30K','resume_gn11_worker.py'),('DIRECT_INSTANCE30K_AND_FULL49','direct_after_recovery.py')]:
  state(name,extra_replay_budget=5846,checkpoint_interval=5000)
  with (E/('A_recovery_'+name+'.log')).open('x') as f:subprocess.run([sys.executable,'-B','-u',str(E/script)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 state('A_RECOVERED_30K_PLUS30K_FULL49_COMPLETE_AWAITING_REPORT',checkpoint_interval=5000)
 # B's input review may finish while A runs. Exclusive admission starts after
 # this whole chain, preventing a race at GN11 -> instance -> eval handoffs.
 ready=E/'intersection_input_passed.json'
 if ready.exists():
  assert not (E/'all_vehicle_ready.json').exists();(E/'all_vehicle_ready.json').write_bytes(ready.read_bytes())
if __name__=='__main__':
 try:main()
 except BaseException:state('FAILED_A_RECOVERY_PRESERVED_NO_RESTART',error=traceback.format_exc());raise

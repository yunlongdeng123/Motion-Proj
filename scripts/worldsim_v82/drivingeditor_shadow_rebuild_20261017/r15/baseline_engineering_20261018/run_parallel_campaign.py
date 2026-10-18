"""One finite coordinator: exact A continuation and the existing B30K budget."""
from pathlib import Path
import os,sys,json,subprocess,time,datetime,signal,fcntl,traceback
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
NEW=BASE/'campaign/r15_baseline_allvehicles_20261018';SID='scene0008_089_137'
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.partial');t.write_text(json.dumps(x,indent=2));t.replace(p)
def state(s,**kw):write(E/'parallel_queue_status.json',dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def main():
 with (E/'parallel_campaign_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),A_target=30000,B_target=30000,extra_replay_updates=0),f)
 capacity=read(E/'parallel_capacity_receipt.json');assert capacity['status']=='PASSED_PARALLEL_ACTUAL_CAPACITY'
 old=read(E/'direct_instance30k/status.json');assert old['state']=='yielded_to_priority_work'
 assert old['step']==capacity['A_durable_step'] and old['resume_checkpoint']==capacity['A_resume_checkpoint']
 # The old A supervisor is expected to end after its intentional yield.
 deadline=time.monotonic()+60
 while Path('/proc/52748').exists():
  if time.monotonic()>deadline:raise TimeoutError('old A supervisor did not finish')
  time.sleep(1)
 bstatus=read(E/'all_vehicle_train_queue_status.json');assert bstatus['pid']==52417 and bstatus['training_updates']==0
 proc=Path('/proc/52417');cmd=(proc/'cmdline').read_bytes();assert str(E/'run_all_vehicle30k_queue.py').encode() in cmd
 assert not (proc/'task/52417/children').read_text().strip();os.kill(52417,signal.SIGTERM)
 write(E/'parallel_cutover_receipt.json',dict(old_A_saved_step=old['step'],A_resume_checkpoint=old['resume_checkpoint'],A_replayed_updates=0,
  old_B_wait_controller=52417,B_prior_updates=0,new_B_same_budget=30000,reason='latest explicit human parallel request'))
 env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_HOME=str(BASE/'cache/torch'))
 env['PATH']='/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH','')
 env['LD_LIBRARY_PATH']='/usr/local/cuda-11.8/lib64:/root/autodl-tmp/envs/drivingeditor/lib:'+env.get('LD_LIBRARY_PATH','')
 def call(name,cmd):
  with (E/('parallel_'+name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 M=E/'all_vehicle_pipeline/metadata'/SID;T=BASE/'campaign/followup_scene_prep_20261013/tools/followup'
 bargs=['--verify-paired-steps','--scene-spec',str(M/'scene_spec.json'),'--readiness-file',str(M/'CPU_readiness.json'),'--reference-config',str(M/'reference_GN11.json'),'--source-repository',str(BASE/'campaign/followup_scene_prep_20261013/frozen_repository'),'--matrix-tools-dir',str(BASE/'campaign/hybrid30k_20261012/tools'),'--campaign-root',str(NEW),'--gate-file',str(M.parent/'GN11_reference_acceptance_authorization.json')]
 gate=read(E/'intersection_input_passed.json');assert gate['status']=='PASSED_LIDAR_CAMERA_INTERSECTION_INPUT_BINDING'
 assert not NEW.exists(),'B existing run needs consumed-step audit'
 state('PREPARING_B_EXISTING30K_BUDGET')
 call('B_prepare',[sys.executable,'-B','-u',str(T/'run_scene_gn11.py'),*bargs,'--prepare'])
 out=E/'direct_instance30k_parallel_resume';assert not out.exists()
 write(E/'direct_active_run.json',dict(output=str(out),coordinator_pid=os.getpid(),resumed_from=old['resume_checkpoint'],first_step=old['step']+1,target_step=30000,extra_replay_updates=0))
 lock=(BASE/'campaign/gpu_slot.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard=dict(state='ACTIVE',guardian_pid=os.getpid(),peer_pids=[],allocator_budgets_GiB=dict(A=16,B=13),reserve_GiB=2)
 write(E/'parallel_group_guard.json',guard)
 acmd=[sys.executable,'-B','-u',str(E/'run_A_native_shared.py'),'--spec',str(E/'direct_spec.json'),'--output',str(out),'--resume',old['resume_checkpoint'],'--iterations','30000','--seed','0','--joint-background','--checkpoint-steps','1','5000','10000','15000','20000','25000','30000','--run']
 bcmd=[sys.executable,'-B','-u',str(E/'run_B_native_shared.py'),*bargs,'--run']
 jobs={};files=[]
 for name,cmd in [('A',acmd),('B',bcmd)]:
  f=(E/('parallel_'+name+'_train.log')).open('x');files.append(f)
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);jobs[name]=p;guard['peer_pids'].append(p.pid);write(E/'parallel_group_guard.json',guard)
 write(E/'all_vehicle_train_queue_status.json',dict(state='PARALLEL_B_NATIVE30K_RUNNING',pid=os.getpid(),GPU_worker_pid=jobs['B'].pid,B_target=30000,second_stage_updates=0))
 state('A_AND_B_TRAINING_CONCURRENTLY',A_pid=jobs['A'].pid,B_pid=jobs['B'].pid,A_resumed_at=old['step']+1,A_replay_updates=0)
 while any(p.poll() is None for p in jobs.values()):time.sleep(5)
 codes={k:p.returncode for k,p in jobs.items()}
 for f in files:f.close()
 guard['state']='TRAINING_ENDED';write(E/'parallel_group_guard.json',guard);lock.close()
 if any(c!=0 for c in codes.values()):raise RuntimeError('preserve all checkpoints; inspect used-step states: '+str(codes))
 assert read(out/'status.json')['state']=='complete'
 br=NEW/'runs'/f'{SID}_hybrid_30000_seed0_GN11';assert read(br/'status.json')['status']=='complete'
 state('SERIAL_ZERO_UPDATE_FULL49_EVALUATIONS')
 call('A_full49',[sys.executable,'-B','-u',str(E/'evaluate_direct.py')])
 call('B_full49',[sys.executable,'-B','-u',str(E/'evaluate_all_vehicle_stage1.py')])
 write(E/'all_vehicle_train_queue_status.json',dict(state='ALL_VEHICLE30K_FULL49_COMPLETE_AWAITING_REPORT_GIT',pid=os.getpid(),training_updates=30000,second_stage_updates=0))
 state('PARALLEL_A_B_FULL49_COMPLETE_AWAITING_REPORT_GIT',A_logical_updates=30000,B_updates=30000,A_replay_updates=0,human_verdict=None)
if __name__=='__main__':
 try:main()
 except BaseException:state('FAILED_PARALLEL_CAMPAIGN_PRESERVED_NO_RESTART',error=traceback.format_exc());raise

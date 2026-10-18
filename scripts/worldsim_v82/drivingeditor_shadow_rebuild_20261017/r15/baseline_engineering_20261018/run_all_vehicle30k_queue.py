"""Registered30K budget. Starts only after full input coverage/identity gate.

The root heartbeat completes preparation and records admission. It may provide
an inspected parallel launcher after proving headroom; otherwise exclusive GPU
availability is required. No repeated training and no second stage for B.
"""
from pathlib import Path
import os,sys,json,subprocess,time,datetime,traceback
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
NEW=BASE/'campaign/r15_baseline_allvehicles_20261018';SID='scene0008_089_137'
TOOLS=BASE/'campaign/followup_scene_prep_20261013/tools/followup'
def read(p):return json.loads(p.read_text())
def state(s,**kw):
 p=E/'all_vehicle_train_queue_status.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(state=s,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));tmp.replace(p)
def stage(name,cmd):
 state('ALL_VEHICLE30K_PIPELINE_RUNNING',stage=name,command=cmd)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(BASE/'cache/torch'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 env['PATH']='/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH','')
 env['LD_LIBRARY_PATH']='/usr/local/cuda-11.8/lib64:/root/autodl-tmp/envs/drivingeditor/lib:'+env.get('LD_LIBRARY_PATH','')
 with (E/('all_vehicle30k_'+name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 with (E/'all_vehicle30k_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),maximum_updates=30000,second_stage_updates=0),f)
 start=time.monotonic();ready=E/'all_vehicle_ready.json'
 while not ready.exists():
  if time.monotonic()-start>24*3600:raise TimeoutError('coverage preparation wait24h exceeded; no GPU used')
  state('WAITING_FOR_FULL_VEHICLE_COVERAGE_AND_INITIALIZATION',training_updates=0,second_stage_updates=0,user_authorization_received=True)
  time.sleep(30)
 gate=read(ready)
 assert gate['status']=='PASSED_FULL_VEHICLE_INPUT_BINDING'
 for k in ['same_identity_set_for_initialization_masks_and_future_stage2','all49_full_RGB_retained','visible_vehicle_coverage_reviewed','all_visible_vehicles_have_real_initialization','unknown_not_used_as_background']:
  assert gate[k] is True,k
 assert gate['frames']==49 and gate['unresolved_visible_vehicles']==[]
 M=E/'all_vehicle_pipeline/metadata'/SID
 cmd=[sys.executable,'-B','-u',str(TOOLS/'run_scene_gn11.py'),'--verify-paired-steps','--scene-spec',str(M/'scene_spec.json'),'--readiness-file',str(M/'CPU_readiness.json'),'--reference-config',str(M/'reference_GN11.json'),'--source-repository',str(BASE/'campaign/followup_scene_prep_20261013/frozen_repository'),'--matrix-tools-dir',str(BASE/'campaign/hybrid30k_20261012/tools'),'--campaign-root',str(NEW),'--gate-file',str(M.parent/'GN11_reference_acceptance_authorization.json')]
 assert not NEW.exists(),'existing run requires used-step/checkpoint audit'
 stage('prepare',cmd+['--prepare'])
 sys.path.insert(0,str(E/'direct_implementation'));from priority import inspect_priority
 while True:
  p=inspect_priority()
  admission=read(E/'all_vehicle_memory_admission.json') if (E/'all_vehicle_memory_admission.json').exists() else None
  if admission and admission.get('status')=='PASSED_PARALLEL_ACTUAL_CAPACITY':
   assert admission['user_authorized'] and admission['combined_peak_plus_reserve_GiB']<=admission['total_GiB']
   launcher=Path(admission['audited_launcher']);assert launcher.is_file() and launcher.parent==E
   traincmd=cmd.copy();traincmd[3]=str(launcher);break
  if p['allowed']:
   traincmd=cmd;break
  state('WAITING_FOR_MEMORY_ADMISSION_OR_EXCLUSIVE_GPU',training_updates=0,other_compute_processes=p['other_compute_processes'])
  if time.monotonic()-start>24*3600:raise TimeoutError('GPU admission wait exceeded')
  time.sleep(30)
 stage('train',traincmd+['--run'])
 result=read(NEW/'runs'/f'{SID}_hybrid_30000_seed0_GN11/status.json');assert result['status']=='complete' and result['iteration']==30000
 # Evaluation uses exclusive admission after all concurrent training completes.
 while not inspect_priority()['allowed']:time.sleep(30)
 stage('full49',[sys.executable,'-B','-u',str(E/'evaluate_all_vehicle_stage1.py')])
 state('ALL_VEHICLE30K_FULL49_COMPLETE_AWAITING_REPORT_GIT',training_updates=30000,second_stage_updates=0,human_verdict=None,shutdown_performed=False)
if __name__=='__main__':
 try:main()
 except BaseException:
  state('FAILED_ALL_VEHICLE30K_PRESERVED_NO_RESTART',error=traceback.format_exc());raise

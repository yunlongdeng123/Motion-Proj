"""Finish the active GN11 then fresh instance30K, no intervening ablations."""
from pathlib import Path
import os,sys,json,subprocess,time,datetime,traceback
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
ROOT=E.parent;NEW=BASE/'campaign/r15_baseline_owner5_20261018'
RUN=NEW/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
def read(p):return json.loads(p.read_text())
def state(s,**kw):
 p=E/'direct_queue_status.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(state=s,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw),indent=2));tmp.replace(p)
def stage(name,cmd):
 state('DIRECT_PIPELINE_RUNNING',stage=name,command=cmd)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',TORCH_HOME=str(BASE/'cache/torch'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 env['PATH']='/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH','')
 env['LD_LIBRARY_PATH']='/usr/local/cuda-11.8/lib64:/root/autodl-tmp/envs/drivingeditor/lib:'+env.get('LD_LIBRARY_PATH','')
 with (E/('direct_'+name+'.log')).open('x') as f:subprocess.run([sys.executable,'-B','-u',*cmd],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 assert read(RUN/'status.json')['status']=='complete' and read(RUN/'status.json')['iteration']==30000
 import torch
 ck=RUN/'chkpnt30000.pth';envck=RUN/'env_light_chkpnt30000.pth'
 captured,step=torch.load(str(ck),map_location='cpu',weights_only=False,mmap=True)
 assert step==30000;count=len(captured[1]);del captured
 spec=read(ROOT/'identity_completion_20261018/spec5.json')
 spec['background_checkpoint']=str(ck);spec['environment_checkpoint']=str(envck)
 spec['stage']='matched_five_fresh_instance30k_direct'
 spec['background_repair']['expected_initial_points']=count
 spec['background_repair']['topology']='native_static_and_identity_preserving_rigid_spatial; inherited_SGCM_no_new_SGCM'
 spec['background_repair']['foreground_background_separation']='isolated_raw_body100_plus_visible_ID100_and_observed_complement_RGB'
 spec['baseline_extension']=dict(identities_match_both_stages=True,new_GN11_run=str(RUN),load_old_trained_actors=False,load_R15_parameters=False,full_frame_report=True,interstage_ablations=False)
 assert not (E/'direct_spec.json').exists()
 (E/'direct_spec.json').write_text(json.dumps(spec,indent=2))
 stage('CPU_spec',[str(E/'direct_implementation/train.py'),'--spec',str(E/'direct_spec.json')])
 assert read(E/'direct_CPU_gate.json')['status']=='PASSED'
 stage('instance30k',[str(E/'direct_implementation/train.py'),'--spec',str(E/'direct_spec.json'),'--output',str(E/'direct_instance30k'),'--iterations','30000','--seed','0','--joint-background','--checkpoint-steps','1','5000','10000','15000','20000','25000','30000','--run'])
 result=read(E/'direct_instance30k/status.json');assert result['state']=='complete' and result['step']==30000
 stage('full49',[str(E/'evaluate_direct.py')])
 state('DIRECT_30K_30K_FULL49_COMPLETE_AWAITING_LOCAL_REPORT_AND_GIT',GN11_updates=30000,instance_updates=30000,interstage_ablation_updates=0,full_frame=True,human_verdict=None,shutdown_performed=False)
if __name__=='__main__':
 try:main()
 except BaseException:
  state('FAILED_DIRECT_PIPELINE_PRESERVED_NO_AUTOMATIC_RESTART',error=traceback.format_exc());raise

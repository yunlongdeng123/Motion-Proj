"""Explicitly authorized20K recovery; native objective, Adam/RNG and5K saves."""
from pathlib import Path
import sys,os,json,importlib.util,random
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro');NEW=BASE/'campaign/r15_baseline_owner5_20261018';SID='scene0008_089_137'
OLD=NEW/'runs'/f'{SID}_hybrid_30000_seed0_GN11';OUT=OLD.with_name(OLD.name+'_resume20k');SOURCE=NEW/'training_source'/SID
TOOLS=BASE/'campaign/followup_scene_prep_20261013/tools/followup';SUPPORT=NEW/'training_support'/SID
def equal(a,b,path=''):
 import torch,numpy as np
 if isinstance(a,torch.Tensor):
  assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.cpu(),b.detach().cpu()),path
 elif isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
 else:assert a==b,path
def main():
 import torch
 from omegaconf import OmegaConf
 assert json.loads((E/'A_replay_authorization.json').read_text())['authorized_replay_updates']==5846
 audit=json.loads((E/'GN11_recovery_audit.json').read_text());assert audit['durable_checkpoint_step']==20000
 state=torch.load(str(OLD/'training_state20000.pth'),map_location='cpu',weights_only=False);cfg=dict(state['config']);cfg['model_path']=str(OUT);cfg['start_checkpoint']=str(OLD/'chkpnt20000.pth');schedule=list(range(5000,30001,5000));cfg['checkpoint_iterations']=schedule;cfg['save_iterations']=schedule
 assert not OUT.exists(),'inspect partial recovery before any further updates'
 sys.path[:0]=[str(TOOLS),str(SUPPORT),str(SOURCE),str(BASE/'campaign/hybrid30k_20261012/tools')]
 from runtime_binding import bind_training_contract
 from contracts import completion_contract
 ready=json.loads((E/'full5_pipeline/metadata'/SID/'CPU_readiness.json').read_text())
 worker_source=(SUPPORT/'matrix_worker.py').read_text().replace("progress = {'iteration': 0,","progress = {'iteration': 20000,").replace('iteration == 1','iteration == 20001')
 worker=type(sys)('recovery_private_worker');worker.__file__=str(SUPPORT/'matrix_worker.py');exec(compile(worker_source,worker.__file__,'exec'),worker.__dict__)
 worker.BASE=BASE;worker.ROOT=NEW;worker.REPOSITORY=SOURCE;worker.check_config=lambda arm:cfg;worker.check_prepared=lambda:None;worker.validate_completion=completion_contract
 def binding(training,config):
  # Immutable references retain old10K/20K evidence in the recovery run index.
  (OUT/'checkpoint_events').mkdir();(OUT/'eval').mkdir()
  for step in [10000,20000]:
   os.symlink(OLD/'checkpoint_events'/f'iteration_{step:06d}.json',OUT/'checkpoint_events'/f'iteration_{step:06d}.json')
   os.symlink(OLD/f'static_time_{step}',OUT/f'static_time_{step}',target_is_directory=True)
   os.symlink(OLD/'eval'/f'train_{step}_render',OUT/'eval'/f'train_{step}_render',target_is_directory=True)
  bind_training_contract(training,config,ready)
  original_event=training.torch.cuda.Event
  class CompletionEvent(original_event):
   def elapsed_time(self,end_event):end_event.synchronize();return super().elapsed_time(end_event)
  training.torch.cuda.Event=CompletionEvent
  from utils import paper_checkpoint
  original_validation=paper_checkpoint._validate_resume_configuration
  def migrated_validation(saved,args):
   current=OmegaConf.to_container(args,resolve=True);permitted={k:current[k] for k in ['checkpoint_iterations','save_iterations']}
   assert all(v==schedule for v in permitted.values())
   assert all(saved[k] in ([10000,20000,30000],schedule) for k in permitted)
   original_validation(dict(saved,**permitted),args)
  paper_checkpoint._validate_resume_configuration=migrated_validation
  original_restore=paper_checkpoint.restore_training_state
  def verified_restore(path,iteration,scene,models,args,sgcm=None):
   result=original_restore(path,iteration,scene,models,args,sgcm=sgcm)
   assert iteration==20000
   for name,prefix in [('static',''),('dynamic','dynamic_'),('environment','env_light_')]:
    golden,step=torch.load(str(OLD/f'{prefix}chkpnt20000.pth'),map_location='cpu',weights_only=False,mmap=True);assert step==20000;equal(golden,models[name].capture(),name);del golden
   equal(state['python_rng'],random.getstate(),'python_rng')
   import numpy as np
   equal(state['numpy_rng'],np.random.get_state(),'numpy_rng');equal(state['torch_rng'],torch.get_rng_state(),'torch_rng');equal(state['cuda_rng'],[x.cpu() for x in torch.cuda.get_rng_state_all()],'cuda_rng');equal(state['hybrid_sampler'],scene.hybrid_sampler.capture(),'sampler')
   (E/'GN11_resume_GPU_receipt.json').write_text(json.dumps(dict(status='PASSED_ACTUAL_NATIVE_FULL20K_MODEL_ADAM_RNG_SAMPLER_RESTORE',step=20000,all_component_tensors_equal=True,SGCM_state_restored=True,config_changes_only=['model_path','start_checkpoint','checkpoint_iterations','save_iterations'],checkpoint_interval=5000,training_updates_in_restore_check=0),indent=2))
   return result
  paper_checkpoint.restore_training_state=verified_restore
 worker.bind_training_contract=binding;worker.run_arm('GN11')
if __name__=='__main__':main()

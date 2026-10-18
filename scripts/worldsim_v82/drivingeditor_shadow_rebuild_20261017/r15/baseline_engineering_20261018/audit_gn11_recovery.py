from pathlib import Path
import os,json,sys,importlib.util,random,ast
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,torch
E=Path(__file__).resolve().parent;RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11');SOURCE=RUN.parents[1]/'training_source/scene0008_089_137'
def main():
 out=E/'GN11_recovery_audit.json';assert not out.exists()
 state=torch.load(str(RUN/'training_state20000.pth'),map_location='cpu',weights_only=False);assert state['iteration']==20000
 cfg=state['config'];assert cfg['iterations']==30000
 random.Random().setstate(state['python_rng']);np.random.RandomState().set_state(state['numpy_rng']);torch.Generator().set_state(state['torch_rng']);assert len(state['cuda_rng'])==1 and state['hybrid_sampler']['steps']==20000
 results=[]
 for name in ['chkpnt20000.pth','dynamic_chkpnt20000.pth','env_light_chkpnt20000.pth']:
  params,step=torch.load(str(RUN/name),map_location='cpu',weights_only=False,mmap=True);assert step==20000
  opts=[x for x in params if isinstance(x,dict) and 'param_groups' in x and 'state' in x] if isinstance(params,(tuple,list)) else []
  assert len(opts)==1,(name,type(params))
  opt=opts[0];assert opt['state']
  for s in opt['state'].values():
   for k,v in s.items():
    if isinstance(v,torch.Tensor):assert torch.isfinite(v).all(),(name,k)
  results.append(dict(file=name,step=step,optimizer_states=len(opt['state']),optimizer_groups=len(opt['param_groups']),Adam_tensors_finite=True));del params,opt
 spec=importlib.util.spec_from_file_location('native_checkpoint_recovery',SOURCE/'utils/paper_checkpoint.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 current=dict(cfg,model_path=str(RUN)+'_resume20k',start_checkpoint=str(RUN/'chkpnt20000.pth'));m._validate_resume_configuration(cfg,current)
 last=json.loads((RUN/'paired_steps.jsonl').read_text().splitlines()[-1]);assert last['iteration']==25847
 receipt=dict(status='CPU_FULL_20K_BUNDLE_RECOVERY_VALIDATED_GPU_RESTORE_PENDING',failed_forward_step=25847,completed_optimizer_updates_before_failure=25846,durable_checkpoint_step=20000,lost_uncheckpointed_updates=5846,logical_target_step=30000,logical_remaining_from_durable_checkpoint=10000,unused_original_execution_budget=4154,extra_replay_updates_required=5846,components=results,python_numpy_torch_saved_RNG_valid=True,cuda_rng_count=1,hybrid_sampler_steps=20000,SGCM_state_present=state['sgcm'] is not None,strict_configuration_unchanged_except_allowed_resume_paths=True,proposed_new_output=str(RUN)+'_resume20k',restoration_GPU_verified=False,training_restarted=False,reason='event elapsed_time read before recorded end event completion; synchronization fix is prepared; no optimizer update at failed step',human_verdict=None)
 out.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()

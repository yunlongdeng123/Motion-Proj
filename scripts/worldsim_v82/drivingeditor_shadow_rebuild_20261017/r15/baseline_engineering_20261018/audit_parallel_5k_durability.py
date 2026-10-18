"""Read-only CPU validation of actual A/B5K durability, never GPU restoration."""
from pathlib import Path
import os,json,sys,datetime,random,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,torch
torch.set_num_threads(2)
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro');SID='scene0008_089_137';step=5000
sys.path.insert(0,str(E/'direct_implementation'));from training_policy import validate_resume_checkpoint
def main():
 out=E/'parallel_5k_durability_receipt.json'
 if out.exists():print(out.read_text());return
 A=Path(json.loads((E/'direct_active_run.json').read_text())['output']);B=BASE/'campaign/r15_baseline_allvehicles_20261018/runs'/f'{SID}_hybrid_30000_seed0_GN11'
 p=A/'checkpoint_005000.pt';before=(p.stat().st_size,p.stat().st_mtime_ns);s=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
 assert s['step']==step and s['sampler']['steps']==step
 cfg=json.loads((A/'effective_config.json').read_text());schedule=[1,5000,10000,15000,20000,25000,30000]
 assert cfg['arguments']['checkpoint_steps']==schedule
 contract=validate_resume_checkpoint(s,cfg['training_policy'],30000,expected_step=step)
 count=[0]
 def finite(x):
  if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();count[0]+=1
  elif isinstance(x,dict):
   for v in x.values():finite(v)
  elif isinstance(x,(list,tuple)):
   for v in x:finite(v)
 for k in ['background','instances','environment']:finite(s[k])
 random.Random().setstate(s['python_rng']);np.random.RandomState().set_state(s['numpy_rng']);torch.Generator().set_state(s['torch_rng']);assert len(s['cuda_rng'])==1
 assert (p.stat().st_size,p.stat().st_mtime_ns)==before
 a=dict(step=step,bytes=before[0],model_Adam_tensor_fields_finite=count[0],sampler_steps=step,strict_resume_contract=contract,formal_runtime_present=bool(s['formal_runtime']),RNG_present_and_CPU_states_parse=True)
 del s
 event=json.loads((B/'checkpoint_events/iteration_005000.json').read_text());assert event['status']=='complete' and event['iteration']==step and len(event['frames'])==49
 assert set(event['checkpoints'])=={'static','dynamic','environment','training_state'}
 components=[]
 for kind,row in event['checkpoints'].items():
  p=Path(row['path']);before=(p.stat().st_size,p.stat().st_mtime_ns);assert before[0]==row['bytes']>0
  if kind=='training_state':continue
  state,n=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True);assert n==step;optim=[x for x in state if isinstance(x,dict) and 'param_groups' in x and 'state' in x];assert len(optim)==1 and optim[0]['state']
  count[0]=0;finite(state);assert (p.stat().st_size,p.stat().st_mtime_ns)==before
  components.append(dict(component=kind,bytes=before[0],optimizer_states=len(optim[0]['state']),finite_tensor_fields=count[0]));del state,optim
 p=Path(event['checkpoints']['training_state']['path']);s=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
 assert s['iteration']==step and s['sgcm'] is not None and s['hybrid_sampler']['steps']==step
 assert s['config']['checkpoint_iterations']==s['config']['save_iterations']==list(range(5000,30001,5000))
 random.Random().setstate(s['python_rng']);np.random.RandomState().set_state(s['numpy_rng']);torch.Generator().set_state(s['torch_rng']);assert len(s['cuda_rng'])==1
 result=dict(status='PASSED_ACTUAL_PARALLEL_A_B_5K_DURABILITY_CPU_READONLY',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),A=a,B=dict(step=step,components=components,training_state_bytes=p.stat().st_size,sampler_steps=step,SGCM_present=True,RNG_present_and_CPU_states_parse=True,complete_evaluation_frames=49,checkpoint_interval=5000),training_updates=0,GPU_used=False,actual_GPU_restore_verified=False,checkpoint_bytes_and_mtimes_preserved=True,human_verdict=None)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

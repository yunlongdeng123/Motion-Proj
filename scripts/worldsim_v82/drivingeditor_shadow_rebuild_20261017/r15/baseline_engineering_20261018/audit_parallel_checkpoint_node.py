"""Validate one explicitly named durable A/B node; CPU read-only, zero updates."""
from pathlib import Path
import argparse,datetime,json,os,random,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np
torch.set_num_threads(2)
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
def read(p):return json.loads(p.read_text())
def finite(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();return 1
 if isinstance(x,dict):return sum(finite(v) for v in x.values())
 if isinstance(x,(tuple,list)):return sum(finite(v) for v in x)
 return 0
def rng(s):
 random.Random().setstate(s['python_rng']);np.random.RandomState().set_state(s['numpy_rng']);torch.Generator().set_state(s['torch_rng'])
 assert len(s['cuda_rng'])==1 and all(isinstance(x,torch.Tensor) and x.numel()>0 for x in s['cuda_rng'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['A','B'],required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();assert a.step in range(5000,30001,5000)
 out=E/f'parallel_{a.model}_{a.step:06d}_durability_receipt.json'
 if out.exists():
  r=read(out);assert r['model']==a.model and r['step']==a.step;print(json.dumps(r));return
 before={};components=[]
 def snapshot(p):before[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns)
 if a.model=='A':
  d=Path(read(E/'direct_active_run.json')['output']);p=d/f'checkpoint_{a.step:06d}.pt';snapshot(p);s=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
  assert s['step']==s['sampler']['steps']==a.step;cfg=read(d/'effective_config.json');assert cfg['arguments']['checkpoint_steps']==[1,5000,10000,15000,20000,25000,30000]
  sys.path.insert(0,str(E/'direct_implementation'));from training_policy import validate_resume_checkpoint
  contract=validate_resume_checkpoint(s,cfg['training_policy'],30000,expected_step=a.step);assert s['formal_runtime'];rng(s)
  components.append(dict(component='full_bundle',bytes=before[str(p)][0],finite_model_Adam_tensor_fields=sum(finite(s[k]) for k in ['background','instances','environment'])))
  details=dict(sampler_steps=a.step,strict_resume_contract=contract,formal_runtime_present=True)
 else:
  d=BASE/'campaign/r15_baseline_allvehicles_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11';event=read(d/'checkpoint_events'/f'iteration_{a.step:06d}.json')
  assert event['status']=='complete' and event['iteration']==a.step and len(event['frames'])==49
  assert set(event['checkpoints'])=={'static','dynamic','environment','training_state'}
  for kind,row in event['checkpoints'].items():
   p=Path(row['path']);snapshot(p);assert before[str(p)][0]==row['bytes']>0
   if kind=='training_state':continue
   s,n=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True);assert n==a.step
   optim=[x for x in s if isinstance(x,dict) and 'param_groups' in x and 'state' in x];assert len(optim)==1 and optim[0]['state']
   components.append(dict(component=kind,bytes=before[str(p)][0],optimizer_states=len(optim[0]['state']),finite_model_Adam_tensor_fields=finite(s)));del s,optim
  p=Path(event['checkpoints']['training_state']['path']);s=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True);assert s['iteration']==s['hybrid_sampler']['steps']==a.step and s['sgcm'] is not None
  assert s['config']['checkpoint_iterations']==s['config']['save_iterations']==list(range(5000,30001,5000));rng(s)
  components.append(dict(component='training_state',bytes=before[str(p)][0]));details=dict(sampler_steps=a.step,SGCM_present=True,complete_evaluation_frames=49)
  for row in event['frames']:
   for v in row['channels'].values():assert Path(v).is_file()
 assert not torch.cuda.is_initialized()
 for p,v in before.items():assert (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v
 result=dict(status='PASSED_ACTUAL_NAMED_PARALLEL_CHECKPOINT_CPU_READONLY',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model=a.model,step=a.step,run=str(d),components=components,
  RNG_present_and_CPU_states_parse=True,details=details,GPU_used=False,actual_GPU_restore_verified=False,training_updates=0,checkpoint_bytes_and_mtimes_preserved=True,human_verdict=None)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

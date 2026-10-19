"""CPU-only audit of one actually published v2 native node; never resumes work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,datetime,json,gc,random
import numpy as np
import torch


def read(p):return json.loads(p.read_text())


def finite_tree(value,stats):
    if torch.is_tensor(value):
        stats['tensors']+=1;stats['elements']+=value.numel()
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all():raise FloatingPointError('nonfinite saved tensor')
    elif isinstance(value,dict):
        for v in value.values():finite_tree(v,stats)
    elif isinstance(value,(tuple,list)):
        for v in value:finite_tree(v,stats)
    elif isinstance(value,(float,np.floating)) and not np.isfinite(value):
        raise FloatingPointError('nonfinite saved scalar')


def optimizer_dicts(value):
    if isinstance(value,dict):
        if {'param_groups','state'}.issubset(value):yield value
        else:
            for v in value.values():yield from optimizer_dicts(v)
    elif isinstance(value,(list,tuple)):
        for v in value:yield from optimizer_dicts(v)


def audit(campaign,step):
    if step not in range(5000,30001,5000):raise ValueError('explicit5K native node required')
    run=campaign/'run30k';event=read(run/'checkpoint_events'/f'iteration_{step:06d}.json')
    if event['status']!='complete' or len(event['checkpoints'])!=4 or len(event['frames'])!=49:
        raise RuntimeError('published four-component49 node required')
    ids=[row['frame_id'].split('/')[-1] for row in event['frames']]
    if len(set(ids))!=49 or set(ids)!={f'0008{i:03d}' for i in range(89,138)}:raise RuntimeError('incorrect49 frame identities')
    out=campaign/'audits'/f'step_{step:06d}.json'
    names=[f'chkpnt{step}.pth',f'dynamic_chkpnt{step}.pth',f'env_light_chkpnt{step}.pth',f'training_state{step}.pth']
    file_stats={n:dict(bytes=(run/n).stat().st_size,mtime_ns=(run/n).stat().st_mtime_ns) for n in names}
    if out.exists():
        old=read(out)
        if old['status']=='PASSED_CPU_NODE_DURABILITY' and old['step']==step and old['files']==file_stats:return old
        raise RuntimeError('existing audit differs; preserve and diagnose')
    model_stats={}
    for n in names[:3]:
        value,actual=torch.load(run/n,map_location='cpu',weights_only=False)
        if actual!=step:raise ValueError('component step differs')
        stats=dict(tensors=0,elements=0);finite_tree(value,stats)
        optimizers=list(optimizer_dicts(value))
        if len(optimizers)!=1 or not optimizers[0]['state']:raise ValueError('saved Adam missing')
        steps=[float(v['step']) for v in optimizers[0]['state'].values() if isinstance(v,dict) and 'step' in v]
        if not steps or max(steps)>step:raise ValueError('invalid Adam step')
        stats.update(Adam_state_present=True,max_Adam_step=max(steps));model_stats[n]=stats
        del value,optimizers;gc.collect()
    state=torch.load(run/names[-1],map_location='cpu',weights_only=False)
    if state['iteration']!=step or state['hybrid_sampler']['steps']!=step:raise ValueError('continuation/sampler step differs')
    if state['config']['iterations']!=30000 or state['config']['seed']!=0 or state['config']['start_checkpoint'] is not None:
        raise ValueError('fresh30K configuration changed')
    if state['sgcm'] is None or not state['model_runtime']['static']['no_time_split']:raise ValueError('SGCM/static3D runtime missing')
    random.Random().setstate(state['python_rng']);np.random.RandomState().set_state(state['numpy_rng'])
    torch.Generator().set_state(state['torch_rng'])
    if not state['cuda_rng'] or any(not torch.is_tensor(v) or v.numel()==0 for v in state['cuda_rng']):raise ValueError('saved CUDA RNG missing')
    for row in event['frames']:
        p=run/'eval'/f'train_{step}_render/hybrid_raw'/f'{row["frame_id"].replace("/","_")}.npz'
        if not p.is_file() or p.stat().st_size==0:raise ValueError('native49 FP32 channel file missing')
    result=dict(status='PASSED_CPU_NODE_DURABILITY',step=step,files=file_stats,models=model_stats,
        sampler_steps=step,SGCM_present=True,RNG_present=True,native49_present=True,GPU_restore_performed=False,
        training_updates=0,quality_accepted=False,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    out.parent.mkdir(exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    a=p.parse_args();r=audit(a.campaign,a.step);print(json.dumps({k:r[k] for k in ['status','step','sampler_steps','native49_present','GPU_restore_performed']}))

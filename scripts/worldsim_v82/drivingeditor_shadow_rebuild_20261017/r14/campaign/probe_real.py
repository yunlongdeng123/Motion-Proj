"""Actual CUDA ABI and three-stage gradient/throughput, zero optimizer updates."""
from pathlib import Path
import sys,json,time,statistics
import torch
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'ownership_rebuild_r14';sys.path.insert(0,str(R/'implementation_r14_final/shadow_rebuild_v3'))
from shadow_rebuild.config import Config
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import objective,write
from shadow_rebuild.gpu_probe import probe
torch.set_num_threads(2);c=Config.load(D/'config.json');d=Dataset(D/'idsplat_frontend/data_IDSplat','cuda',True);results={'ABI':probe(c),'scene_training_updates':0,'stages':{}}
model=SceneModel(d.seed,c).cuda();generator=torch.Generator().manual_seed(c.seed)
for stage in ['geometry','material','joint']:
    model.set_stage(stage);timings=[]
    for i in [18,32]:
        model.zero_grad(set_to_none=True);frame=d.frame(i);torch.cuda.synchronize();start=time.monotonic()
        loss,terms,p=objective(model,frame,len(d.actor_ids),'cuda',stage,generator,global_step=1000);loss.backward();torch.cuda.synchronize()
        if not bool(torch.isfinite(loss)):raise FloatingPointError(stage+' nonfinite loss')
        grads={name:float(v.grad.detach().abs().sum()) for name,v in model.named_parameters() if v.grad is not None}
        if not all(__import__('math').isfinite(x) for x in grads.values()):raise FloatingPointError(stage+' nonfinite gradients')
        timings.append(time.monotonic()-start);print(json.dumps({'stage':stage,'probe':len(timings),'seconds':timings[-1],'loss':float(loss.detach()),'ownership_occ':float(terms['ownership_occupancy'].detach())}),flush=True)
    results['stages'][stage]={'median_forward_backward_seconds':statistics.median(timings),'gradient_nonzero_parameters':[name for name,v in grads.items() if v>0],'peak_allocated_bytes':torch.cuda.max_memory_allocated()};del p,loss,terms
results['estimated_training_hours_no_IO_or_densify']=sum(results['stages'][name]['median_forward_backward_seconds']*getattr(c,name+'_steps') for name in ['geometry','material','joint'])/3600
write(D/'GPU_probe.json',results);print(json.dumps(results),flush=True)

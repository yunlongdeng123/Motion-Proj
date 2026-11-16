"""Single-factor coverage correction: screen/world-road dependency, no OBB center veto.

Candidate membership is not identity or permission to hard-delete. Only fixed
per-row opacity is optimized, with the same road and preservation objectives.
"""
import gc,sys,random
from pathlib import Path
import numpy as np
import torch
import closedloop as t

PARENT=t.C
t.C=PARENT/'handoff_coverage_r2'
BASE_BINDING=t.binding_for
def binding(manifest,road):
 result=BASE_BINDING(manifest,road);result['candidate_algorithm_sha256']=t.sha(__file__);return result
t.binding_for=binding

def select(scene,manifest):
 n=len(scene.static.get_xyz);device=scene.static.get_xyz.device
 score=torch.zeros(n,device=device);guard=torch.zeros_like(score);count=torch.zeros(n,device=device,dtype=torch.int32)
 for key in range(199):
  with np.load(t.C/'domains'/f'{key:03d}.npz') as z:fit=z['fit'];target=z['target'];K=z['K']
  with torch.no_grad():v,owners,cam,_=scene._rows(key,(),(),False)
  f=v['features'].detach().clone();f[:,8:10]=0;f[:n,8:10]=1;f.requires_grad_(True)
  v=dict(v,mask=v['mask']&~scene._contains(owners,(10,)),features=f)
  pack=scene._bindings().raster(cam,scene.pipe,0,v,scene._background)
  a=torch.as_tensor(target&fit,device=device);k=torch.as_tensor(K,device=device)
  objective=pack['extra'][6][a].sum()+pack['extra'][7][k].sum()
  grad=torch.autograd.grad(objective,f)[0][:n,8:10]
  score=torch.maximum(score,grad[:,0]);guard=torch.maximum(guard,grad[:,1]);count+=(grad[:,0]>1e-4).int()
  del pack,v,f,grad,objective
  if key%25==0:print('SELECT_WITHOUT_OBB_VETO',key,flush=True)
 eligible=(scene._owners_s<=0)&(score>1e-4)&(count>=2)&~(guard>.02*score)
 ids=torch.where(eligible)[0]
 old=np.load(PARENT/'candidates.npz')['ids']
 np.savez_compressed(t.C/'candidates.npz',ids=ids.cpu().numpy(),score=score[ids].cpu().numpy(),guard=guard[ids].cpu().numpy(),target_views=count[ids].cpu().numpy())
 t.write(t.C/'candidate_receipt.json',dict(candidates=len(ids),old_candidates=len(old),newly_covered_rows=int((~np.isin(ids.cpu().numpy(),old)).sum()),
   single_changed_factor='remove physical OBB center rejection from opacity-fit candidate coverage',
   criterion='fixed owner<=0; >=2 same-sort target/world-road-supported contributions; retained maximum mass<=.02*target maximum mass',
   hard_delete=False,two_dimensional_expansion_used=False,road_geometry_and_field_unchanged=True,loss_sampling_budget_unchanged=True))
 return ids
t.select_candidates=select

def main():
 t.C.mkdir(exist_ok=False)
 for folder in ['field','domains']:(t.C/folder).symlink_to(PARENT/folder,target_is_directory=True)
 t.write(t.C/'active_scene.json',t.read(PARENT/'active_scene.json'))
 t.optimization('smoke_guarded',32)
 checkpoint=t.C/'smoke_guarded/checkpoint000032.pt'
 t.optimization('smoke_guarded_restore',33,str(checkpoint))
 t.optimization('smoke_guarded_replay',33,str(checkpoint))
 a=torch.load(t.C/'smoke_guarded_restore/checkpoint000033.pt',map_location='cpu',weights_only=False)
 b=torch.load(t.C/'smoke_guarded_replay/checkpoint000033.pt',map_location='cpu',weights_only=False)
 gd=float((a['gate']-b['gate']).abs().max());moment={k:float((a['optimizer']['state'][0][k]-b['optimizer']['state'][0][k]).abs().max()) for k in ['exp_avg','exp_avg_sq']}
 assert gd<1e-6 and max(moment.values())<1e-10
 t.write(t.C/'lifecycle.json',dict(full_state_restore_exact=True,next_gate_max_abs_difference=gd,next_moment_max_abs_difference=moment,smoke_updates=34,formal_updates=0,short_job=True))
 t.optimization('formal',1000)
 t.render('pilot')
 t.write(t.C/'status.json',dict(stage='COVERAGE_FORMAL_PILOT_READY',GPU='keep_on',formal_updates=1000))

if __name__=='__main__':main()

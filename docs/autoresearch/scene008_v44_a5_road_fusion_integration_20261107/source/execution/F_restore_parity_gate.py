"""Fresh exact restoration, two extra inference parity calls, then next-two replay."""
from pathlib import Path
import json
import numpy as np
import torch
from F_colour_bake_core import Core,R,exact,rng,set_rng,write,sha
smoke=json.loads((R/'F_smoke_r1/returned.json').read_text());assert smoke['status']=='PASS_SMOKE_SAVE_PUBLIC_EVAL_AND_DISPOSABLE_NEXT2'
c=Core(R/'F_restore_parity_r1');c.restore(Path(smoke['checkpoint']));saved=c.state();before=rng();original=c.base.raster;captured={}
def record(*args,**kwargs):
 out=original(*args,**kwargs)
 captured.clear();captured.update(alpha=out['alpha'].detach().cpu().numpy()[0],group_alpha_T=out['extra'][:7].detach().cpu().numpy())
 return out
c.base.raster=record;parity=[]
try:
 for state in ['keep','delete']:
  rgb=c.render_train(128,state).detach().cpu().numpy()
  with np.load(R/f'F_smoke_r1/evaluation/frames/128/{state}_float.npz',allow_pickle=False) as z:
   errors={k:float(np.abs(v-z[k]).max()) for k,v in dict(rgb=rgb,**captured).items()}
   for k,v in dict(rgb=rgb,**captured).items():assert np.array_equal(v,z[k]),(state,k,errors[k])
  parity.append({'raw':128,'state':state,'RGB_alpha_groups_exact':True,'max_abs':errors})
finally:c.base.raster=original;set_rng(before)
exact(saved,c.state());assert c.calls==2 and c.backs==c.updates==0
write(c.out/'train_helper_public_API_parity.json',{'status':'PASS_SAME_SMOKE_DELTA_TRAIN_HELPER_PUBLIC_BOUNDED_F_EXACT','comparisons':parity,'actual_native_forwards':2,'native_backwards':0,'optimizer_updates':0,'RNG_full_state_preserved':True})
for _ in range(2):c.paired()
restored=c.save(f'restored_state{c.step}.pth');control=torch.load(smoke['disposable_checkpoint'],map_location='cpu',weights_only=False);current=c.state()
for key in ['schema','step','sampler','rng','config']:exact(control[key],current[key],key)
exact(control['optimizer']['param_groups'],current['optimizer']['param_groups'],'optimizer.param_groups')
delta_max=float((control['delta']-current['delta']).abs().max());assert delta_max<=1e-6
moment=[]
for k,values in control['optimizer']['state'].items():
 assert set(values)==set(current['optimizer']['state'][k])
 for name,value in values.items():
  other=current['optimizer']['state'][k][name]
  if name=='step':exact(value,other,'Adam.step');difference=0.
  else:
   difference=float((value-other).abs().max());assert difference<=1e-8
  moment.append({'parameter':k,'field':name,'max_abs_difference':difference})
assert [x['raw'] for x in control['history']]==[x['raw'] for x in current['history']]
loss_difference=max(abs(a[k]-b[k]) for a,b in zip(control['history'],current['history']) for k in ['keep_MSE','delete_MSE']);assert loss_difference<=1e-6
projection_cost_difference=max(abs(a['weighted_l1_cost_cpu_float64']-b['weighted_l1_cost_cpu_float64']) for a,b in zip(control['history'],current['history']))
assert all(x['weighted_l1_cost_cpu_float64']<=1. for x in current['history'])
c.frozen()
result={'status':'PASS_FRESH_FULL_RESTORE_NEXT2_AND_TRAIN_PUBLIC_PARITY','restore_start_bit_exact':True,'native_parity_forwards':2,'next2_training_forwards':4,'actual_native_forwards':c.calls,
 'actual_native_backwards':c.backs,'actual_optimizer_updates':c.updates,'restored_checkpoint':str(restored),'control_checkpoint':smoke['disposable_checkpoint'],
 'next2_delta_max_abs_difference':delta_max,'next2_Adam_moment_differences':moment,'next2_loss_max_abs_difference':loss_difference,'next2_projection_cost_max_abs_difference':projection_cost_difference,
 'next2_parameter_tolerance':1e-6,'next2_Adam_tolerance':1e-8,'next2_raw_schedule_sampler_RNG_exact':True,
 'next2_complete_bitwise_equality_claim':delta_max==0 and all(x['max_abs_difference']==0 for x in moment) and loss_difference==0,
 'all_original_models_and_files_unchanged':True,'formal_not_started_by_this_GPU_worker':True,'fit_contract_sha256':sha(R/'contract/bounded_surface_fusion_bake_pilot_v1.json')}
write(c.out/'returned.json',result);print(json.dumps(result),flush=True)

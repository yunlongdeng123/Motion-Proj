from pathlib import Path
import json
import numpy as np
import torch
from C49_material_core import Core,R,cpu,exact,rng,set_rng,sha,write
smoke=json.loads((R/'smoke600s_r1/returned.json').read_text());assert smoke['status']=='PASS_REAL600S_C49_SMOKE_SAVE_EVAL_DISPOSABLE_NEXT2' and smoke['actual_training_seconds_before_disposable']>=600
c=Core(R/'smoke_freshrestore_r1');c.restore(Path(smoke['checkpoint']));saved=c.state();before=rng();parity=[]
for state in ['keep','delete']:
 with torch.no_grad():out=c.render(128,state)
 values={'rgb':out['rgb'].cpu().numpy(),'alpha':out['alpha'].cpu().numpy()[0],'depth_sum':out['depth_sum'].cpu().numpy()[0],'group_alpha_T':out['group_alpha_T'].cpu().numpy()}
 with np.load(R/f'smoke600s_r1/evaluation/frames/128/{state}_float.npz',allow_pickle=False) as z:
  errors={k:float(np.abs(v-z[k]).max()) for k,v in values.items()}
  for k,v in values.items():assert np.array_equal(v,z[k]),(state,k,errors[k])
 parity.append({'raw':128,'state':state,'max_abs':errors});del out
set_rng(before);exact(saved,c.state());write(c.out/'saved_parameter_public_render_parity.json',{'status':'PASS_SAVED_RGB_FRESH_RESTORE_NATIVE_PARITY','comparisons':parity,'actual_native_forwards':2,'optimizer_updates':0,'all_state_RNG_exact_before_next2':True})
for _ in range(2):c.paired()
restored=c.save(f'restored_state{c.step}.pth');current=c.state();control=torch.load(smoke['disposable_checkpoint'],map_location='cpu',weights_only=False)
for key in ['schema','step','config','sampler','rng']:exact(control[key],current[key],key)
exact(control['optimizer']['param_groups'],current['optimizer']['param_groups'])
rgbdiff=float((control['parent_RGB']-current['parent_RGB']).abs().max());assert rgbdiff<=1e-6
moment=[]
for parameter,state in control['optimizer']['state'].items():
 for name,value in state.items():
  other=current['optimizer']['state'][parameter][name]
  difference=float((value-other).abs().max()) if torch.is_tensor(value) else abs(value-other)
  if name=='step':assert difference==0
  else:assert difference<=1e-8
  moment.append({'parameter':parameter,'field':name,'max_abs_difference':difference})
assert [x['raw'] for x in control['history']]==[x['raw'] for x in current['history']]
lossdiff=max(abs(a[k]-b[k]) for a,b in zip(control['history'],current['history']) for k in ['keep_MSE','delete_MSE']);assert lossdiff<=1e-6
c.frozen();result={'status':'PASS_C49_600S_FRESH_FULL_RESTORE_AND_NEXT2','initial_restore_bitexact':True,'saved_public_RGB_parity_exact':True,'actual_native_forwards':c.calls,'actual_native_backwards':c.backs,'actual_optimizer_updates':c.updates,
 'restored_checkpoint':str(restored),'next2_RGB_max_abs_difference':rgbdiff,'next2_Adam_differences':moment,'next2_loss_max_abs_difference':lossdiff,'next2_sampler_and_all_RNG_exact':True,
 'next2_bitwise_equality_claim':rgbdiff==0 and all(x['max_abs_difference']==0 for x in moment) and lossdiff==0,'RGB_numerical_tolerance':1e-6,'Adam_numerical_tolerance':1e-8,'all_original_scene_and_source_guards_unchanged':True,'formal_started_by_this_worker':False}
write(c.out/'returned.json',result);print(json.dumps(result),flush=True)
